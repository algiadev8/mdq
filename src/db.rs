use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use rusqlite::functions::FunctionFlags;
use rusqlite::{Connection, ErrorCode, OpenFlags, OptionalExtension, params};
use sha2::{Digest, Sha256};
use walkdir::{DirEntry, WalkDir};

use crate::markdown::{normalize_target, parse_note_bytes};
use crate::model::{EmbeddingInput, LinkRef, NoteRef, NoteResolution, PageRecord, SearchHit};
use crate::query::MetadataFilter;
use crate::tokenize::{fts_query, index_text};

pub struct Database {
    connection: Connection,
}

struct NoteIdResolution {
    id: i64,
    used_fallback: bool,
}

fn configure_connection(connection: &Connection) -> Result<()> {
    connection.create_scalar_function(
        "mdq_lower",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |context| {
            let value: String = context.get(0)?;
            Ok(value.to_lowercase())
        },
    )?;
    connection.busy_timeout(Duration::from_secs(120))?;
    connection.execute_batch(
        "
        PRAGMA busy_timeout = 120000;
        PRAGMA foreign_keys = ON;
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        ",
    )?;
    Ok(())
}

/// Max attempts for [`retry_on_lock`]. `PRAGMA busy_timeout` already makes SQLite block
/// and retry internally for up to 120s per statement, but a whole multi-statement
/// operation (open-and-migrate, or a bulk write transaction) can still surface
/// `SQLITE_BUSY`/`SQLITE_LOCKED` if another process holds the lock for longer than that,
/// or if the failed statement was a `COMMIT` (which cannot simply be re-issued once the
/// owning `Transaction` has been consumed). Retrying the whole operation with backoff
/// covers that case.
const LOCK_RETRY_ATTEMPTS: u32 = 5;

fn is_lock_error(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(inner, _)
            if matches!(inner.code, ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)
    )
}

/// Retries `action` with exponential backoff while it fails with `SQLITE_BUSY` or
/// `SQLITE_LOCKED`, to ride out contention from other `mdq` processes accessing the
/// same index concurrently.
fn retry_on_lock<T>(mut action: impl FnMut() -> rusqlite::Result<T>) -> rusqlite::Result<T> {
    let mut delay = Duration::from_millis(50);
    for attempt in 1..=LOCK_RETRY_ATTEMPTS {
        match action() {
            Ok(value) => return Ok(value),
            Err(error) if attempt < LOCK_RETRY_ATTEMPTS && is_lock_error(&error) => {
                thread::sleep(delay);
                delay = (delay * 2).min(Duration::from_secs(2));
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("loop always returns on the final attempt")
}

impl Database {
    pub fn path(&self) -> Option<&Path> {
        self.connection.path().map(Path::new)
    }

    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let connection = retry_on_lock(|| Connection::open(path))
            .with_context(|| format!("failed to open index {}", path.display()))?;
        configure_connection(&connection)?;
        connection.execute_batch(
            "
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                body_start_line INTEGER NOT NULL DEFAULT 1,
                frontmatter_json TEXT,
                mtime INTEGER NOT NULL,
                ctime INTEGER NOT NULL DEFAULT 0,
                size INTEGER NOT NULL,
                content_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                heading TEXT,
                body TEXT NOT NULL,
                search_text TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS chunks_note_idx ON chunks(note_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                search_text,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE IF NOT EXISTS links (
                id INTEGER PRIMARY KEY,
                source_note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                raw_target TEXT NOT NULL,
                normalized_target TEXT NOT NULL,
                target_note_id INTEGER REFERENCES notes(id) ON DELETE SET NULL,
                heading TEXT,
                is_embed INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS links_source_idx ON links(source_note_id);
            CREATE INDEX IF NOT EXISTS links_target_idx ON links(target_note_id);
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY(content_hash, model)
            );
            ",
        )?;
        ensure_column(
            &connection,
            "chunks",
            "search_text",
            "TEXT NOT NULL DEFAULT ''",
        )?;
        ensure_column(
            &connection,
            "chunks",
            "content_hash",
            "TEXT NOT NULL DEFAULT ''",
        )?;
        ensure_column(&connection, "notes", "ctime", "INTEGER NOT NULL DEFAULT 0")?;
        ensure_column(
            &connection,
            "notes",
            "body_start_line",
            "INTEGER NOT NULL DEFAULT 1",
        )?;
        Ok(Self { connection })
    }

    pub fn open_existing(path: &Path) -> Result<Self> {
        let connection =
            retry_on_lock(|| Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE))
                .with_context(|| format!("failed to open existing index {}", path.display()))?;
        configure_connection(&connection)?;
        Ok(Self { connection })
    }

    pub fn rebuild(&mut self, vault: &Path) -> Result<IndexStats> {
        let transaction = self.connection.transaction()?;
        let stats = rebuild_in_transaction(&transaction, vault)?;
        transaction.commit()?;
        Ok(stats)
    }

    /// Like [`rebuild`](Self::rebuild), but never waits for another process's write lock:
    /// if a concurrent `mdq` process already holds it (e.g. it is refreshing the same
    /// index), this returns `Ok(None)` immediately instead of blocking, so the caller can
    /// decide to wait for that process to finish rather than contend with it.
    pub fn try_rebuild(&mut self, vault: &Path) -> Result<Option<IndexStats>> {
        self.connection.busy_timeout(Duration::ZERO)?;
        let result = (|| -> Result<Option<IndexStats>> {
            let transaction = match self
                .connection
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            {
                Ok(transaction) => transaction,
                Err(error) if is_lock_error(&error) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let stats = rebuild_in_transaction(&transaction, vault)?;
            match transaction.commit() {
                Ok(()) => Ok(Some(stats)),
                Err(error) if is_lock_error(&error) => Ok(None),
                Err(error) => Err(error.into()),
            }
        })();
        self.connection.busy_timeout(Duration::from_secs(120))?;
        result
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let match_query = fts_query(query);
        if match_query.is_empty() {
            return Ok(Vec::new());
        }
        let mut statement = self.connection.prepare(
            "
            SELECT c.id, n.path, n.title, c.heading, bm25(chunks_fts) AS rank, c.body
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN notes n ON n.id = c.note_id
            WHERE chunks_fts MATCH ?1
            ORDER BY rank
            LIMIT ?2
            ",
        )?;
        let rows = statement.query_map(params![match_query, limit as i64], |row| {
            let body: String = row.get(5)?;
            Ok(SearchHit {
                chunk_id: row.get(0)?,
                path: row.get(1)?,
                title: row.get(2)?,
                heading: row.get(3)?,
                score: -row.get::<_, f64>(4)?,
                snippet: source_snippet(&body, query, 180),
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn missing_embeddings(&self, model: &str) -> Result<Vec<EmbeddingInput>> {
        let mut statement = self.connection.prepare(
            "
            SELECT DISTINCT c.content_hash, n.title, c.heading, c.body
            FROM chunks c
            JOIN notes n ON n.id = c.note_id
            LEFT JOIN embeddings e
              ON e.content_hash = c.content_hash AND e.model = ?1
            WHERE e.content_hash IS NULL
            ORDER BY c.id
            ",
        )?;
        let rows = statement.query_map([model], |row| {
            let title: String = row.get(1)?;
            let heading: Option<String> = row.get(2)?;
            let body: String = row.get(3)?;
            Ok(EmbeddingInput {
                content_hash: row.get(0)?,
                text: format!(
                    "passage: {}\n{}\n{}",
                    title,
                    heading.as_deref().unwrap_or_default(),
                    body
                ),
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn store_embeddings(&mut self, model: &str, items: &[(String, Vec<f32>)]) -> Result<()> {
        let transaction = self.connection.transaction()?;
        {
            let mut insert = transaction.prepare(
                "INSERT INTO embeddings(content_hash, model, dimensions, vector)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(content_hash, model) DO UPDATE SET
                     dimensions = excluded.dimensions,
                     vector = excluded.vector",
            )?;
            for (content_hash, vector) in items {
                let blob = vector_to_blob(vector);
                insert.execute(params![content_hash, model, vector.len() as i64, blob])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn semantic_search(
        &self,
        model: &str,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        let mut statement = self.connection.prepare(
            "
            SELECT c.id, n.path, n.title, c.heading, c.body, e.vector
            FROM chunks c
            JOIN notes n ON n.id = c.note_id
            JOIN embeddings e ON e.content_hash = c.content_hash
            WHERE e.model = ?1 AND e.dimensions = ?2
            ",
        )?;
        let rows = statement.query_map(params![model, query_vector.len() as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Vec<u8>>(5)?,
            ))
        })?;
        let mut hits = Vec::new();
        for row in rows {
            let (chunk_id, path, title, heading, body, blob) = row?;
            let vector = blob_to_vector(&blob);
            hits.push(SearchHit {
                chunk_id,
                path,
                title,
                heading,
                score: dot_product(query_vector, &vector),
                snippet: source_snippet(&body, "", 180),
            });
        }
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        hits.truncate(limit);
        Ok(hits)
    }

    pub fn chunk_body(&self, chunk_id: i64) -> Result<Option<String>> {
        self.connection
            .query_row("SELECT body FROM chunks WHERE id = ?1", [chunk_id], |row| {
                row.get(0)
            })
            .optional()
            .map_err(Into::into)
    }

    pub fn query_frontmatter(&self, expression: &dyn MetadataFilter) -> Result<Vec<NoteRef>> {
        let mut statement = self.connection.prepare(
            "SELECT path, title, frontmatter_json FROM notes WHERE frontmatter_json IS NOT NULL ORDER BY path",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        let mut matches = Vec::new();
        for row in rows {
            let (path, file_stem_title, json) = row?;
            let value: Option<serde_json::Value> = serde_json::from_str(&json).ok();
            if value.as_ref().is_some_and(|v| expression.matches(v)) {
                // Prefer the frontmatter `title` field so that the JSON output is consistent
                // with what the filter expression matched against.
                let title = value
                    .as_ref()
                    .and_then(|v| v["title"].as_str())
                    .map(str::to_owned)
                    .unwrap_or(file_stem_title);
                matches.push(NoteRef { path, title });
            }
        }
        Ok(matches)
    }

    pub fn all_chunks(&self) -> Result<Vec<SearchHit>> {
        let mut statement = self.connection.prepare(
            "
            SELECT c.id, n.path, n.title, c.heading, c.body
            FROM chunks c
            JOIN notes n ON n.id = c.note_id
            ORDER BY c.id
            ",
        )?;
        let rows = statement.query_map([], |row| {
            let body: String = row.get(4)?;
            Ok(SearchHit {
                chunk_id: row.get(0)?,
                path: row.get(1)?,
                title: row.get(2)?,
                heading: row.get(3)?,
                score: 0.0,
                snippet: source_snippet(&body, "", 180),
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Returns format-neutral page records for compatibility query adapters.
    pub fn all_pages(&self) -> Result<Vec<PageRecord>> {
        let mut statement = self.connection.prepare(
            "
            SELECT path, title, body, body_start_line, frontmatter_json, mtime, ctime, size
            FROM notes
            ORDER BY path
            ",
        )?;
        let rows = statement.query_map([], |row| {
            let metadata = row
                .get::<_, Option<String>>(4)?
                .and_then(|json| serde_json::from_str(&json).ok())
                .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
            Ok(PageRecord {
                path: row.get(0)?,
                title: row.get(1)?,
                body: row.get(2)?,
                body_start_line: row.get::<_, i64>(3)? as usize,
                metadata,
                mtime: row.get(5)?,
                ctime: row.get(6)?,
                size: row.get::<_, i64>(7)? as u64,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn all_page_refs(&self) -> Result<Vec<NoteRef>> {
        let mut statement = self
            .connection
            .prepare("SELECT path, title FROM notes ORDER BY path")?;
        let rows = statement.query_map([], |row| {
            Ok(NoteRef {
                path: row.get(0)?,
                title: row.get(1)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn total_page_bytes(&self) -> Result<usize> {
        let bytes: i64 = self.connection.query_row(
            "
            SELECT COALESCE(SUM(
                length(path) + length(title) + length(body) +
                length(COALESCE(frontmatter_json, ''))
            ), 0)
            FROM notes
            ",
            [],
            |row| row.get(0),
        )?;
        Ok(usize::try_from(bytes).unwrap_or(usize::MAX))
    }

    pub fn page_record(&self, path: &str) -> Result<Option<PageRecord>> {
        self.connection
            .query_row(
                "
                SELECT path, title, body, body_start_line, frontmatter_json, mtime, ctime, size
                FROM notes
                WHERE path = ?1
                ",
                [path],
                |row| {
                    let metadata = row
                        .get::<_, Option<String>>(4)?
                        .and_then(|json| serde_json::from_str(&json).ok())
                        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
                    Ok(PageRecord {
                        path: row.get(0)?,
                        title: row.get(1)?,
                        body: row.get(2)?,
                        body_start_line: row.get::<_, i64>(3)? as usize,
                        metadata,
                        mtime: row.get(5)?,
                        ctime: row.get(6)?,
                        size: row.get::<_, i64>(7)? as u64,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn backlinks(&self, target: &str) -> Result<Vec<LinkRef>> {
        let target_id = self.resolve_note_id(target)?.map(|resolved| resolved.id);
        let Some(target_id) = target_id else {
            return Ok(Vec::new());
        };
        self.link_query(
            "
            SELECT s.path, s.title, t.path, t.title, l.raw_target, l.heading, l.is_embed, l.normalized_target
            FROM links l
            JOIN notes s ON s.id = l.source_note_id
            LEFT JOIN notes t ON t.id = l.target_note_id
            WHERE l.target_note_id = ?1
            ORDER BY s.path
            ",
            target_id,
        )
    }

    /// Every link in the vault, for bulk construction of compatibility
    /// `file.links` / `file.backlinks` / `file.embeds` fields without N+1 queries.
    pub fn all_links(&self) -> Result<Vec<LinkRef>> {
        let file_index = self.non_note_file_index()?;
        let mut statement = self.connection.prepare(
            "
            SELECT s.path, s.title, t.path, t.title, l.raw_target, l.heading, l.is_embed, l.normalized_target
            FROM links l
            JOIN notes s ON s.id = l.source_note_id
            LEFT JOIN notes t ON t.id = l.target_note_id
            ORDER BY s.path, l.id
            ",
        )?;
        let rows =
            statement.query_map([], |row| self.link_ref_from_row(row, file_index.as_ref()))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn outgoing_links(&self, source: &str) -> Result<Vec<LinkRef>> {
        let source_id = self.resolve_note_id(source)?.map(|resolved| resolved.id);
        let Some(source_id) = source_id else {
            return Ok(Vec::new());
        };
        self.link_query(
            "
            SELECT s.path, s.title, t.path, t.title, l.raw_target, l.heading, l.is_embed, l.normalized_target
            FROM links l
            JOIN notes s ON s.id = l.source_note_id
            LEFT JOIN notes t ON t.id = l.target_note_id
            WHERE l.source_note_id = ?1
            ORDER BY l.id
            ",
            source_id,
        )
    }

    pub fn unresolved_links(&self) -> Result<Vec<LinkRef>> {
        Ok(self
            .all_links()?
            .into_iter()
            .filter(|link| link.resolved_path.is_none())
            .collect())
    }

    fn link_query(&self, sql: &str, note_id: i64) -> Result<Vec<LinkRef>> {
        let file_index = self.non_note_file_index()?;
        let mut statement = self.connection.prepare(sql)?;
        let rows = statement.query_map([note_id], |row| {
            self.link_ref_from_row(row, file_index.as_ref())
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    fn link_ref_from_row(
        &self,
        row: &rusqlite::Row<'_>,
        file_index: Option<&NonNoteFileIndex>,
    ) -> rusqlite::Result<LinkRef> {
        let source_path: String = row.get(0)?;
        let target_path: Option<String> = row.get(2)?;
        let target_title: Option<String> = row.get(3)?;
        let normalized_target: String = row.get(7)?;
        let target = target_path
            .clone()
            .zip(target_title)
            .map(|(path, title)| NoteRef { path, title });
        let resolved_path = target_path.or_else(|| {
            file_index.and_then(|index| index.resolve(&source_path, &normalized_target))
        });
        Ok(LinkRef {
            source: NoteRef {
                path: source_path,
                title: row.get(1)?,
            },
            target,
            resolved_path,
            raw_target: row.get(4)?,
            heading: row.get(5)?,
            embed: row.get(6)?,
        })
    }

    pub fn note_body(&self, path: &str) -> Result<Option<(NoteRef, String)>> {
        let id = self.resolve_note_id(path)?.map(|resolved| resolved.id);
        let Some(id) = id else {
            return Ok(None);
        };
        self.connection
            .query_row(
                "SELECT path, title, body FROM notes WHERE id = ?1",
                [id],
                |row| {
                    Ok((
                        NoteRef {
                            path: row.get(0)?,
                            title: row.get(1)?,
                        },
                        row.get(2)?,
                    ))
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn status(&self, embedding_model: &str) -> Result<Status> {
        let count = |table: &str| -> Result<usize> {
            Ok(self
                .connection
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                    row.get::<_, i64>(0)
                })? as usize)
        };
        let vault = self.metadata("vault")?;
        let index_stale = match &vault {
            Some(path) => self.is_stale(Path::new(path))?,
            None => true,
        };
        Ok(Status {
            has_index: vault.is_some(),
            vault,
            indexed_at: self.metadata("indexed_at")?,
            notes: count("notes")?,
            chunks: count("chunks")?,
            links: count("links")?,
            unresolved_links: self.unresolved_links_count()?,
            embeddings: self.connection.query_row(
                "
                SELECT count(*)
                FROM chunks c
                WHERE EXISTS (
                    SELECT 1 FROM embeddings e WHERE e.content_hash = c.content_hash
                )
                ",
                [],
                |row| row.get::<_, i64>(0),
            )? as usize,
            cached_embeddings: count("embeddings")?,
            index_stale,
            embeddings_stale: self.embeddings_stale(embedding_model)?,
        })
    }

    /// True when the vault's Markdown files have changed since the last `rebuild`.
    pub fn is_stale(&self, vault: &Path) -> Result<bool> {
        Ok(self.staleness(vault)? > 0)
    }

    /// Number of Markdown files added, removed, or modified since the last `rebuild`.
    pub fn staleness(&self, vault: &Path) -> Result<usize> {
        let mut current = HashMap::<String, (i64, u64)>::new();
        for entry in WalkDir::new(vault)
            .follow_links(false)
            .into_iter()
            .filter_entry(visible_entry)
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
            })
        {
            let metadata = entry.metadata()?;
            let mtime = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs() as i64)
                .unwrap_or_default();
            let relative = entry
                .path()
                .strip_prefix(vault)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .replace('\\', "/");
            current.insert(relative, (mtime, metadata.len()));
        }

        let mut statement = self
            .connection
            .prepare("SELECT path, mtime, size FROM notes")?;
        let mut indexed = HashMap::<String, (i64, u64)>::new();
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)? as u64,
            ))
        })?;
        for row in rows {
            let (path, mtime, size) = row?;
            indexed.insert(path, (mtime, size));
        }

        let mut changed = 0;
        let all_paths: HashSet<&String> = current.keys().chain(indexed.keys()).collect();
        for path in all_paths {
            if current.get(path) != indexed.get(path) {
                changed += 1;
            }
        }
        Ok(changed)
    }

    /// True when chunks exist that have not yet been embedded with `model`.
    pub fn embeddings_stale(&self, model: &str) -> Result<bool> {
        Ok(self.missing_embeddings_count(model)? > 0)
    }

    /// Number of chunks that have not yet been embedded with `model`.
    pub fn missing_embeddings_count(&self, model: &str) -> Result<usize> {
        let missing: i64 = self.connection.query_row(
            "
            SELECT count(*)
            FROM chunks c
            LEFT JOIN embeddings e
              ON e.content_hash = c.content_hash AND e.model = ?1
            WHERE e.content_hash IS NULL
            ",
            [model],
            |row| row.get(0),
        )?;
        Ok(missing as usize)
    }

    /// True when this vault has a previously built index (as opposed to never indexed).
    pub fn has_index(&self) -> Result<bool> {
        Ok(self.metadata("vault")?.is_some())
    }

    /// True when at least one embedding has been cached for this vault.
    pub fn has_embeddings(&self) -> Result<bool> {
        Ok(self
            .connection
            .query_row("SELECT count(*) FROM embeddings", [], |row| {
                row.get::<_, i64>(0)
            })?
            > 0)
    }

    fn metadata(&self, key: &str) -> Result<Option<String>> {
        self.connection
            .query_row("SELECT value FROM metadata WHERE key = ?1", [key], |row| {
                row.get(0)
            })
            .optional()
            .map_err(Into::into)
    }

    /// Returns the path of the vault that was last indexed into this database.
    pub fn indexed_vault(&self) -> Result<Option<String>> {
        self.metadata("vault")
    }

    /// Resolves a note-like input and reports whether the unique title/filename
    /// fallback was needed.
    pub fn resolve_note(&self, input: &str) -> Result<Option<NoteResolution>> {
        let Some(resolved) = self.resolve_note_id(input)? else {
            return Ok(None);
        };
        self.connection
            .query_row(
                "SELECT path, title FROM notes WHERE id = ?1",
                [resolved.id],
                |row| {
                    Ok(NoteResolution {
                        note: NoteRef {
                            path: row.get(0)?,
                            title: row.get(1)?,
                        },
                        used_fallback: resolved.used_fallback,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    fn resolve_note_id(&self, input: &str) -> Result<Option<NoteIdResolution>> {
        let normalized = normalize_target(input);
        let filename = normalized.rsplit('/').next().unwrap_or(&normalized);
        // Try exact path match first — this is always unambiguous even when another note
        // shares the same filename or title.
        // SQLite's built-in lower() only handles ASCII. mdq_lower uses the same Unicode
        // lowercase mapping as normalize_target() and the index link resolver (e.g.
        // `Ｂ` and `ｂ`) while retaining the indexed query shape.
        let mut path_stmt = self.connection.prepare(
            "
            SELECT id FROM notes
            WHERE mdq_lower(path) = ?1 || '.md'
               OR mdq_lower(substr(path, 1, length(path) - 3)) = ?1
            LIMIT 2
            ",
        )?;
        let path_ids = path_stmt
            .query_map([&normalized], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<i64>>>()?;
        if path_ids.len() == 1 {
            return Ok(Some(NoteIdResolution {
                id: path_ids[0],
                used_fallback: false,
            }));
        }
        // Fall back to title / filename match (may be ambiguous).
        let mut title_stmt = self
            .connection
            .prepare("SELECT id FROM notes WHERE mdq_lower(title) = ?1 LIMIT 2")?;
        let title_ids = title_stmt
            .query_map([filename], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<i64>>>()?;
        Ok((title_ids.len() == 1).then(|| NoteIdResolution {
            id: title_ids[0],
            used_fallback: true,
        }))
    }

    /// Candidate paths for a note-like input. Exact resolution remains strict; this is
    /// only diagnostic data for callers after resolution failed or was ambiguous. The
    /// caller controls how many candidates are needed for its diagnostic output.
    pub fn note_candidates(&self, input: &str, limit: usize) -> Result<Vec<String>> {
        let normalized = normalize_target(input);
        let needle = normalized.rsplit('/').next().unwrap_or(&normalized);
        let limit = i64::try_from(limit).context("candidate limit exceeds SQLite range")?;
        let mut statement = self.connection.prepare(
            "SELECT path FROM notes
             WHERE mdq_lower(title) LIKE '%' || ?1 || '%'
                OR mdq_lower(path) LIKE '%' || ?1 || '%'
             ORDER BY path LIMIT ?2",
        )?;
        let rows = statement.query_map(params![needle, limit], |row| row.get(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    fn unresolved_links_count(&self) -> Result<usize> {
        let file_index = self.non_note_file_index()?;
        let mut statement = self.connection.prepare(
            "
            SELECT l.normalized_target, n.path
            FROM links l
            JOIN notes n ON n.id = l.source_note_id
            WHERE l.target_note_id IS NULL
            ",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut unresolved = 0;
        for row in rows {
            let (target, source_path) = row?;
            if !file_index
                .as_ref()
                .is_some_and(|index| index.resolve(&source_path, &target).is_some())
            {
                unresolved += 1;
            }
        }
        Ok(unresolved)
    }

    fn non_note_file_index(&self) -> Result<Option<NonNoteFileIndex>> {
        self.metadata("vault")?
            .map(|vault| NonNoteFileIndex::build(Path::new(&vault)))
            .transpose()
    }
}

struct NonNoteFileIndex {
    by_path: HashMap<String, String>,
    by_filename: HashMap<String, Vec<String>>,
}

impl NonNoteFileIndex {
    fn build(vault: &Path) -> Result<Self> {
        let mut by_path = HashMap::new();
        let mut by_filename: HashMap<String, Vec<String>> = HashMap::new();
        for entry in WalkDir::new(vault)
            .follow_links(false)
            .into_iter()
            .filter_entry(visible_entry)
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| {
                !entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
            })
        {
            let relative = entry
                .path()
                .strip_prefix(vault)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .replace('\\', "/");
            let key = collapse_path(&relative);
            by_path.insert(key, relative.clone());
            if let Some(filename) = relative.rsplit('/').next() {
                by_filename
                    .entry(filename.to_lowercase())
                    .or_default()
                    .push(relative);
            }
        }
        Ok(Self {
            by_path,
            by_filename,
        })
    }

    fn resolve(&self, source_path: &str, target: &str) -> Option<String> {
        let mut target_forms = vec![collapse_path(target)];
        if let Some((parent, _)) = source_path.rsplit_once('/') {
            target_forms.push(collapse_path(&format!("{parent}/{target}")));
        }
        target_forms.dedup();

        for form in &target_forms {
            if let Some(path) = self.by_path.get(form) {
                return Some(path.clone());
            }
        }

        if target_forms[0].contains('/') {
            return None;
        }
        self.by_filename
            .get(&target_forms[0])
            .filter(|paths| paths.len() == 1)
            .map(|paths| paths[0].clone())
    }
}

fn ensure_column(
    connection: &Connection,
    table: &str,
    column: &str,
    declaration: &str,
) -> Result<()> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if !columns.iter().any(|existing| existing == column) {
        connection.execute(
            &format!("ALTER TABLE {table} ADD COLUMN {column} {declaration}"),
            [],
        )?;
    }
    Ok(())
}

fn rebuild_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    vault: &Path,
) -> Result<IndexStats> {
    // Bump this whenever parsing/tokenization or the persisted index format changes.
    // Old indexes and indexes belonging to a different vault must be rebuilt fully.
    const INDEX_VERSION: &str = "1";
    let vault_path = canonical(vault)?.to_string_lossy().into_owned();
    let metadata = |key: &str| -> rusqlite::Result<Option<String>> {
        transaction
            .query_row("SELECT value FROM metadata WHERE key = ?1", [key], |row| {
                row.get(0)
            })
            .optional()
    };
    let incremental = metadata("index_version")?.as_deref() == Some(INDEX_VERSION)
        && metadata("vault")?.as_deref() == Some(vault_path.as_str());
    let mut previous = HashMap::<String, (i64, String)>::new();
    if incremental {
        let mut statement = transaction.prepare("SELECT path, id, content_hash FROM notes")?;
        let rows = statement.query_map([], |row| Ok((row.get(0)?, (row.get(1)?, row.get(2)?))))?;
        for row in rows {
            let (path, value) = row?;
            previous.insert(path, value);
        }
    } else {
        transaction.execute_batch(
            "INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all');
             DELETE FROM links;
             DELETE FROM chunks;
             DELETE FROM notes;",
        )?;
    }
    // Also create this for databases opened through open_existing() before migration.
    transaction.execute_batch("CREATE INDEX IF NOT EXISTS chunks_note_idx ON chunks(note_id)")?;

    let files: Vec<PathBuf> = WalkDir::new(vault)
        .follow_links(false)
        .into_iter()
        .filter_entry(visible_entry)
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
        })
        .map(DirEntry::into_path)
        .collect();

    let mut insert_note = transaction.prepare(
        "INSERT INTO notes(path, title, body, body_start_line, frontmatter_json, mtime, ctime, size, content_hash, id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
         ON CONFLICT(id) DO UPDATE SET
             body = excluded.body, body_start_line = excluded.body_start_line,
             frontmatter_json = excluded.frontmatter_json, mtime = excluded.mtime,
             ctime = excluded.ctime, size = excluded.size, content_hash = excluded.content_hash",
    )?;
    let mut insert_chunk = transaction.prepare(
        "INSERT INTO chunks(note_id, ordinal, heading, body, search_text, content_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?;
    let mut insert_fts =
        transaction.prepare("INSERT INTO chunks_fts(rowid, search_text) VALUES (?1, ?2)")?;
    let mut insert_link = transaction.prepare(
        "INSERT INTO links(source_note_id, raw_target, normalized_target, heading, is_embed)
         VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;
    let mut delete_fts = transaction.prepare(
        "INSERT INTO chunks_fts(chunks_fts, rowid, search_text)
         SELECT 'delete', id, search_text FROM chunks WHERE note_id = ?1",
    )?;
    let mut delete_note = transaction.prepare("DELETE FROM notes WHERE id = ?1")?;
    let mut delete_chunks = transaction.prepare("DELETE FROM chunks WHERE note_id = ?1")?;
    let mut delete_links = transaction.prepare("DELETE FROM links WHERE source_note_id = ?1")?;
    let mut refresh_times = transaction.prepare(
        "UPDATE notes SET mtime = ?2, ctime = ?3, size = ?4 WHERE id = ?1
         AND (mtime != ?2 OR ctime != ?3 OR size != ?4)",
    )?;
    let mut changed = !incremental;
    let mut identities_changed = !incremental;
    for file in &files {
        // Always hash the bytes: mtime/size alone miss same-size edits within one second.
        let bytes =
            std::fs::read(file).with_context(|| format!("failed to read {}", file.display()))?;
        let hash = hex::encode(Sha256::digest(&bytes));
        let path = file
            .strip_prefix(vault)
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/");
        let old = previous.remove(&path);
        let old_id = old.as_ref().map(|(id, _)| *id);
        if let Some((id, old_hash)) = old {
            if old_hash == hash {
                let metadata = std::fs::metadata(file)?;
                let seconds = |time: std::io::Result<std::time::SystemTime>| {
                    time.ok()
                        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|duration| duration.as_secs() as i64)
                };
                let mtime = seconds(metadata.modified()).unwrap_or_default();
                let ctime = seconds(metadata.created()).unwrap_or(mtime);
                refresh_times.execute(params![id, mtime, ctime, metadata.len() as i64])?;
                continue;
            }
            delete_fts.execute([id])?;
            delete_chunks.execute([id])?;
            delete_links.execute([id])?;
        } else {
            identities_changed = true;
        }
        changed = true;
        let note = parse_note_bytes(vault, file, &bytes, hash)?;
        insert_note.execute(params![
            note.path,
            note.title,
            note.body,
            note.body_start_line as i64,
            note.frontmatter.as_ref().map(serde_json::Value::to_string),
            note.mtime,
            note.ctime,
            note.size as i64,
            note.hash,
            old_id,
        ])?;
        let note_id = old_id.unwrap_or_else(|| transaction.last_insert_rowid());
        for chunk in note.chunks {
            let searchable = format!(
                "{} {} {}",
                note.title,
                chunk.heading.as_deref().unwrap_or_default(),
                chunk.body
            );
            let search_text = index_text(&searchable);
            let content_hash = hex::encode(Sha256::digest(searchable.as_bytes()));
            insert_chunk.execute(params![
                note_id,
                chunk.ordinal as i64,
                chunk.heading,
                chunk.body,
                search_text,
                content_hash,
            ])?;
            let chunk_id = transaction.last_insert_rowid();
            if incremental {
                insert_fts.execute(params![chunk_id, search_text])?;
            }
        }
        for link in note.links {
            insert_link.execute(params![
                note_id,
                link.raw_target,
                link.target,
                link.heading,
                link.is_embed,
            ])?;
        }
    }
    for (id, _) in previous.into_values() {
        delete_fts.execute([id])?;
        delete_note.execute([id])?;
        changed = true;
        identities_changed = true;
    }
    if !incremental {
        // External-content FTS can tokenize all persisted chunks in one bulk operation.
        transaction.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')", [])?;
    }
    if changed {
        if identities_changed {
            // Added/removed paths can make even unchanged sources ambiguous. Body-only
            // edits preserve note IDs, so their existing incoming edges remain valid.
            transaction.execute(
                "UPDATE links SET target_note_id = NULL WHERE target_note_id IS NOT NULL",
                [],
            )?;
        }
        resolve_links(transaction)?;
    }
    transaction.execute(
        "INSERT INTO metadata(key, value) VALUES ('vault', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [&vault_path],
    )?;
    transaction.execute(
        "INSERT INTO metadata(key, value) VALUES ('indexed_at', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [],
    )?;
    transaction.execute(
        "INSERT INTO metadata(key, value) VALUES ('index_version', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [INDEX_VERSION],
    )?;
    Ok(IndexStats {
        notes: files.len(),
        chunks: transaction.query_row("SELECT count(*) FROM chunks", [], |row| {
            row.get::<_, i64>(0)
        })? as usize,
        links: transaction
            .query_row("SELECT count(*) FROM links", [], |row| row.get::<_, i64>(0))?
            as usize,
    })
}

fn resolve_links(transaction: &rusqlite::Transaction<'_>) -> Result<()> {
    let mut by_path = HashMap::<String, Vec<i64>>::new();
    let mut by_title = HashMap::<String, Vec<i64>>::new();
    {
        let mut statement = transaction.prepare("SELECT id, path, title FROM notes")?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (id, path, title) = row?;
            let without_extension = path.strip_suffix(".md").unwrap_or(&path).to_lowercase();
            by_path.entry(without_extension).or_default().push(id);
            by_title.entry(title.to_lowercase()).or_default().push(id);
        }
    }
    let mut links = Vec::new();
    {
        let mut statement = transaction.prepare(
            "
            SELECT l.id, l.normalized_target, n.path
            FROM links l
            JOIN notes n ON n.id = l.source_note_id
            WHERE l.target_note_id IS NULL
            ",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            links.push(row?);
        }
    }
    let mut update = transaction.prepare("UPDATE links SET target_note_id = ?1 WHERE id = ?2")?;
    for (link_id, target, source_path) in links {
        let mut target_forms = vec![collapse_path(&target)];
        let source_parent = source_path.rsplit_once('/').map(|(parent, _)| parent);
        if let Some(parent) = source_parent {
            target_forms.push(collapse_path(&format!("{parent}/{target}")));
        }
        target_forms.dedup();

        let filename = target_forms[0]
            .rsplit('/')
            .next()
            .unwrap_or(&target_forms[0]);
        let exact = target_forms
            .iter()
            .find_map(|form| by_path.get(form).filter(|ids| ids.len() == 1));
        let candidates = exact.or_else(|| by_title.get(filename).filter(|ids| ids.len() == 1));
        if let Some(candidates) = candidates {
            update.execute(params![candidates[0], link_id])?;
        }
    }
    Ok(())
}

fn collapse_path(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let mut parts = Vec::new();
    for part in normalized.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            part => parts.push(part),
        }
    }
    parts.join("/").to_lowercase()
}

fn source_snippet(body: &str, query: &str, max_chars: usize) -> String {
    let body_chars: Vec<char> = body.chars().collect();
    if body_chars.len() <= max_chars {
        return body.replace('\n', " ");
    }

    let query_lower = query.to_lowercase();
    let body_lower = body.to_lowercase();
    let byte_position = body_lower.find(&query_lower).unwrap_or(0);
    let character_position = body[..byte_position].chars().count();
    let half = max_chars / 2;
    let start = character_position.saturating_sub(half);
    let end = (start + max_chars).min(body_chars.len());
    let mut snippet: String = body_chars[start..end].iter().collect();
    snippet = snippet.replace('\n', " ");
    if start > 0 {
        snippet.insert_str(0, "… ");
    }
    if end < body_chars.len() {
        snippet.push_str(" …");
    }
    snippet
}

fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn blob_to_vector(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn dot_product(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (*left as f64) * (*right as f64))
        .sum()
}

fn visible_entry(entry: &DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    let name = entry.file_name().to_string_lossy();
    !name.starts_with('.') && name != "node_modules" && name != "target"
}

pub fn default_db_path(vault: &Path) -> Result<PathBuf> {
    let canonical = canonical(vault)?;
    let digest = hex::encode(Sha256::digest(canonical.to_string_lossy().as_bytes()));
    let root = dirs::cache_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
        .context("cannot determine cache directory")?;
    Ok(root.join("mdq").join(&digest[..16]).join("index.sqlite3"))
}

fn canonical(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("path does not exist: {}", path.display()))
}

pub struct IndexStats {
    pub notes: usize,
    pub chunks: usize,
    pub links: usize,
}

#[derive(schemars::JsonSchema, serde::Serialize)]
pub struct Status {
    pub vault: Option<String>,
    pub indexed_at: Option<String>,
    pub has_index: bool,
    pub notes: usize,
    pub chunks: usize,
    pub links: usize,
    pub unresolved_links: usize,
    pub embeddings: usize,
    pub cached_embeddings: usize,
    pub index_stale: bool,
    pub embeddings_stale: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::Expression;
    use std::fs;

    #[test]
    fn indexes_generic_frontmatter_and_both_link_styles() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("notes");
        fs::create_dir_all(vault.join("nested")).unwrap();
        fs::write(
            vault.join("alpha.md"),
            "---\n任意項目:\n  状態: 有効\n---\n# Alpha\n[[nested/beta]]\n",
        )
        .unwrap();
        fs::write(
            vault.join("nested/beta.md"),
            "# Beta\n[Alpha](../alpha.md)\n検索対象の日本語本文\n",
        )
        .unwrap();

        let mut database = Database::open(&directory.path().join("index.sqlite3")).unwrap();
        let stats = database.rebuild(&vault).unwrap();
        assert_eq!(stats.notes, 2);
        assert_eq!(stats.links, 2);

        let expression = Expression::parse("任意項目.状態 = 有効").unwrap();
        let notes = database.query_frontmatter(&expression).unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].path, "alpha.md");

        let alpha_backlinks = database.backlinks("alpha").unwrap();
        assert_eq!(alpha_backlinks.len(), 1);
        assert_eq!(alpha_backlinks[0].source.path, "nested/beta.md");

        let beta_backlinks = database.backlinks("nested/beta").unwrap();
        assert_eq!(beta_backlinks.len(), 1);
        assert_eq!(beta_backlinks[0].source.path, "alpha.md");

        let hits = database.search("日本語", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "nested/beta.md");
    }

    #[test]
    fn query_frontmatter_finds_frontmatter_only_notes() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("notes");
        fs::create_dir_all(&vault).unwrap();
        fs::write(vault.join("meta_only.md"), "---\nkind: reference\n---\n").unwrap();
        fs::write(
            vault.join("with_body.md"),
            "---\nkind: reference\n---\n# Body\nsome content\n",
        )
        .unwrap();
        let mut database = Database::open(&directory.path().join("index.sqlite3")).unwrap();
        database.rebuild(&vault).unwrap();

        let expression = Expression::parse("kind = reference").unwrap();
        let notes = database.query_frontmatter(&expression).unwrap();
        assert_eq!(
            notes.len(),
            2,
            "frontmatter-only note must be returned by query_frontmatter"
        );
        let paths: Vec<&str> = notes.iter().map(|n| n.path.as_str()).collect();
        assert!(paths.contains(&"meta_only.md"));
        assert!(paths.contains(&"with_body.md"));
    }

    #[test]
    fn exact_path_resolves_despite_same_title_in_other_folder() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("notes");
        fs::create_dir_all(vault.join("a")).unwrap();
        fs::create_dir_all(vault.join("b")).unwrap();
        fs::write(vault.join("a/Alpha.md"), "# Alpha\n[[b/Alpha]]\n").unwrap();
        fs::write(vault.join("b/Alpha.md"), "# Alpha\n").unwrap();
        let mut database = Database::open(&directory.path().join("index.sqlite3")).unwrap();
        database.rebuild(&vault).unwrap();

        // Exact folder-qualified path must resolve even though both share the title "Alpha"
        let links = database.outgoing_links("a/Alpha").unwrap();
        assert_eq!(links.len(), 1);
        let backlinks = database.backlinks("b/Alpha").unwrap();
        assert_eq!(backlinks.len(), 1);

        let resolution = database.resolve_note("a/Alpha").unwrap().unwrap();
        assert_eq!(resolution.note.path, "a/Alpha.md");
        assert!(!resolution.used_fallback);
    }

    #[test]
    fn resolves_note_paths_with_unicode_case_mapping() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("notes");
        fs::create_dir_all(&vault).unwrap();
        fs::write(vault.join("Ｂ.md"), "# Fullwidth B\n[[target]]\n").unwrap();
        fs::write(vault.join("target.md"), "# Target\n").unwrap();

        let mut database = Database::open(&directory.path().join("index.sqlite3")).unwrap();
        database.rebuild(&vault).unwrap();

        let links = database.outgoing_links("Ｂ").unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].source.path, "Ｂ.md");

        let resolution = database.resolve_note("ｂ").unwrap().unwrap();
        assert_eq!(resolution.note.path, "Ｂ.md");
        assert!(!resolution.used_fallback);
    }

    #[test]
    fn reports_when_note_resolution_uses_unique_filename_fallback() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("notes");
        fs::create_dir_all(vault.join("actual")).unwrap();
        fs::write(vault.join("actual/Alpha.md"), "# Alpha\n").unwrap();

        let mut database = Database::open(&directory.path().join("index.sqlite3")).unwrap();
        database.rebuild(&vault).unwrap();

        let resolution = database
            .resolve_note("missing-directory/Alpha")
            .unwrap()
            .unwrap();
        assert_eq!(resolution.note.path, "actual/Alpha.md");
        assert!(resolution.used_fallback);
    }

    #[test]
    fn try_rebuild_returns_none_without_blocking_when_another_connection_holds_the_write_lock() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("notes");
        fs::create_dir_all(&vault).unwrap();
        fs::write(vault.join("a.md"), "# A\n").unwrap();
        let db_path = directory.path().join("index.sqlite3");

        let mut database = Database::open(&db_path).unwrap();
        database.rebuild(&vault).unwrap();

        // A second connection holds the write lock, simulating a concurrent `mdq` process
        // that is already refreshing the index.
        let mut other = Connection::open(&db_path).unwrap();
        configure_connection(&other).unwrap();
        let holder = other
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .unwrap();

        let started = std::time::Instant::now();
        let result = database.try_rebuild(&vault).unwrap();
        assert!(
            result.is_none(),
            "try_rebuild must not win a contended lock"
        );
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "try_rebuild must not block waiting for the lock"
        );

        holder.rollback().unwrap();
        let stats = database.try_rebuild(&vault).unwrap();
        assert!(
            stats.is_some(),
            "try_rebuild must succeed once the lock is free"
        );
    }

    #[test]
    fn existing_non_markdown_files_are_not_counted_as_unresolved_links() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("notes");
        fs::create_dir_all(vault.join("views")).unwrap();
        fs::create_dir_all(vault.join("assets")).unwrap();
        fs::write(vault.join("views/open.base"), "filters: \"true\"\n").unwrap();
        fs::write(vault.join("assets/logo.png"), "png").unwrap();
        fs::write(
            vault.join("index.md"),
            "# Index\n[Base](views/open.base)\n![[logo.png]]\n[[missing.pdf]]\n",
        )
        .unwrap();

        let mut database = Database::open(&directory.path().join("index.sqlite3")).unwrap();
        database.rebuild(&vault).unwrap();

        let status = database.status("test-model").unwrap();
        assert_eq!(status.links, 3);
        assert_eq!(status.unresolved_links, 1);

        let links = database.outgoing_links("index").unwrap();
        let resolved: Vec<Option<String>> =
            links.into_iter().map(|link| link.resolved_path).collect();
        assert!(resolved.contains(&Some("views/open.base".to_owned())));
        assert!(resolved.contains(&Some("assets/logo.png".to_owned())));
        assert!(resolved.contains(&None));

        let unresolved = database.unresolved_links().unwrap();
        assert_eq!(unresolved.len(), 1);
        assert_eq!(unresolved[0].source.path, "index.md");
        assert_eq!(unresolved[0].raw_target, "missing.pdf");
    }

    #[test]
    fn backlinks_returns_none_for_missing_note() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("notes");
        fs::create_dir_all(&vault).unwrap();
        fs::write(vault.join("a.md"), "# A\n").unwrap();
        let mut database = Database::open(&directory.path().join("index.sqlite3")).unwrap();
        database.rebuild(&vault).unwrap();

        let result = database.note_body("nonexistent").unwrap();
        assert!(
            result.is_none(),
            "note_body must return None for missing note"
        );
    }
}
