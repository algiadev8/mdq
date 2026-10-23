use std::fs;
use std::path::Path;

use mdq::db::Database;
use rusqlite::Connection;

fn snapshot(path: &Path) -> Vec<Vec<String>> {
    let connection = Connection::open(path).unwrap();
    [
        "SELECT path, title, body, body_start_line, frontmatter_json, mtime, ctime, size, content_hash FROM notes ORDER BY path",
        "SELECT n.path, c.ordinal, c.heading, c.body, c.search_text, c.content_hash FROM chunks c JOIN notes n ON n.id=c.note_id ORDER BY n.path, c.ordinal",
        "SELECT n.path, l.raw_target, l.normalized_target, l.heading, l.is_embed, t.path FROM links l JOIN notes n ON n.id=l.source_note_id LEFT JOIN notes t ON t.id=l.target_note_id ORDER BY n.path, l.raw_target",
    ].iter().map(|sql| {
        let mut statement = connection.prepare(sql).unwrap();
        let columns = statement.column_count();
        statement.query_map([], |row| {
            Ok((0..columns).map(|i| format!("{:?}", row.get_ref(i).unwrap())).collect::<Vec<_>>().join("|"))
        }).unwrap().collect::<rusqlite::Result<Vec<_>>>().unwrap()
    }).collect()
}

fn check_fts(path: &Path) {
    let connection = Connection::open(path).unwrap();
    connection
        .execute(
            "INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)",
            [],
        )
        .unwrap();
    let problems: i64 = connection
        .query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(problems, 0);
}

#[test]
fn repeated_refresh_matches_clean_build_after_every_change() {
    let directory = tempfile::tempdir().unwrap();
    let vault = directory.path().join("vault");
    fs::create_dir_all(vault.join("a")).unwrap();
    fs::create_dir_all(vault.join("b")).unwrap();
    fs::write(vault.join("source.md"), "---\nlink: '[[target]]'\n---\n# Source\noldtoken 日本語 [[a/target]] [wiki](a/target.md) `[[ignored]]`\n").unwrap();
    fs::write(vault.join("a/target.md"), "# Target\n日本語 Rust Rust\n").unwrap();
    fs::write(vault.join("meta.md"), "---\ntype: empty\n---\n").unwrap();
    let db_path = directory.path().join("refresh.sqlite3");
    let mut database = Database::open(&db_path).unwrap();
    for phase in 0..7 {
        match phase {
            2 => fs::write(vault.join("source.md"), "---\nlink: '[[target]]'\n---\n# Source\nnewtoken 日本語 [[a/target]] [wiki](a/target.md) `[[ignored]]`\n").unwrap(),
            3 => fs::write(vault.join("b/target.md"), "# Duplicate\nAmbiguous target\n").unwrap(),
            4 => fs::rename(vault.join("a/target.md"), vault.join("a/renamed.md")).unwrap(),
            5 => fs::remove_file(vault.join("b/target.md")).unwrap(),
            6 => fs::write(vault.join("meta.md"), "---\ntype: changed\n---\n# New heading\r\nContent 格子暗号\r\n").unwrap(),
            _ => {}
        }
        let stats = database.rebuild(&vault).unwrap();
        let clean_path = directory.path().join(format!("clean-{phase}.sqlite3"));
        let mut clean = Database::open(&clean_path).unwrap();
        let clean_stats = clean.rebuild(&vault).unwrap();
        assert_eq!(
            (stats.notes, stats.chunks, stats.links),
            (clean_stats.notes, clean_stats.chunks, clean_stats.links)
        );
        assert_eq!(snapshot(&db_path), snapshot(&clean_path), "phase {phase}");
        let links = database.outgoing_links("source").unwrap();
        let fallback = links
            .iter()
            .find(|link| link.raw_target == "target")
            .unwrap();
        let expected = match phase {
            0..=2 => Some("a/target.md"),
            3 => None, // Both filenames are target: fallback is now ambiguous.
            4 => Some("b/target.md"),
            _ => None,
        };
        assert_eq!(fallback.resolved_path.as_deref(), expected, "phase {phase}");
        for query in ["oldtoken", "newtoken", "日本語", "Rust", "格子暗号"] {
            let sorted_hits = |db: &Database| {
                let mut hits: Vec<_> = db
                    .search(query, 100)
                    .unwrap()
                    .into_iter()
                    .map(|h| (h.path, h.heading, h.snippet, h.score))
                    .collect();
                hits.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                hits
            };
            assert_eq!(
                sorted_hits(&database),
                sorted_hits(&clean),
                "{query}, phase {phase}"
            );
        }
        check_fts(&db_path);
    }
}

#[test]
fn failed_refresh_rolls_back_content_fts_and_metadata() {
    let directory = tempfile::tempdir().unwrap();
    let vault = directory.path().join("vault");
    fs::create_dir(&vault).unwrap();
    fs::write(vault.join("a.md"), "# A\noriginal [[b]] 日本語\n").unwrap();
    fs::write(vault.join("b.md"), "# B\noriginal\n").unwrap();
    let db_path = directory.path().join("index.sqlite3");
    let mut database = Database::open(&db_path).unwrap();
    database.rebuild(&vault).unwrap();
    let before = snapshot(&db_path);
    let indexed_at = database.status("test").unwrap().indexed_at;
    fs::write(vault.join("a.md"), "# A\nchanged\n").unwrap();
    fs::write(vault.join("b.md"), [0xff, 0xfe]).unwrap();
    assert!(database.rebuild(&vault).is_err());
    assert_eq!(before, snapshot(&db_path));
    assert_eq!(indexed_at, database.status("test").unwrap().indexed_at);
    assert_eq!(database.search("original", 10).unwrap().len(), 2);
    assert!(database.search("changed", 10).unwrap().is_empty());
    check_fts(&db_path);
}

#[test]
fn same_size_same_timestamp_edits_are_not_skipped_and_embeddings_are_reused() {
    let directory = tempfile::tempdir().unwrap();
    let vault = directory.path().join("vault");
    fs::create_dir(&vault).unwrap();
    let note = vault.join("a.md");
    fs::write(&note, "# A\noldword\n").unwrap();
    let db_path = directory.path().join("index.sqlite3");
    let mut database = Database::open(&db_path).unwrap();
    database.rebuild(&vault).unwrap();
    let missing = database.missing_embeddings("test").unwrap();
    assert_eq!(missing.len(), 1);
    database
        .store_embeddings("test", &[(missing[0].content_hash.clone(), vec![1.0, 0.0])])
        .unwrap();
    database.rebuild(&vault).unwrap();
    assert!(database.missing_embeddings("test").unwrap().is_empty());
    let mtime = fs::metadata(&note).unwrap().modified().unwrap();
    fs::write(&note, "# A\nnewword\n").unwrap();
    fs::File::options()
        .write(true)
        .open(&note)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(mtime))
        .unwrap();
    database.rebuild(&vault).unwrap();
    assert!(database.search("oldword", 10).unwrap().is_empty());
    assert_eq!(database.search("newword", 10).unwrap().len(), 1);
    assert_eq!(database.missing_embeddings("test").unwrap().len(), 1);
    // Reverting to cached content must not require inference again.
    fs::write(&note, "# A\noldword\n").unwrap();
    database.rebuild(&vault).unwrap();
    assert!(database.missing_embeddings("test").unwrap().is_empty());
    check_fts(&db_path);
}

#[test]
fn body_only_edit_preserves_note_and_incoming_link_ids() {
    let directory = tempfile::tempdir().unwrap();
    let vault = directory.path().join("vault");
    fs::create_dir(&vault).unwrap();
    fs::write(vault.join("source.md"), "# Source\n[[target]]\n").unwrap();
    fs::write(vault.join("target.md"), "# Target\noldword\n").unwrap();
    let db_path = directory.path().join("index.sqlite3");
    let mut database = Database::open(&db_path).unwrap();
    database.rebuild(&vault).unwrap();
    let ids = || -> (i64, i64, i64) {
        Connection::open(&db_path)
            .unwrap()
            .query_row(
                "SELECT l.id, l.source_note_id, l.target_note_id FROM links l",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap()
    };
    let before = ids();
    fs::write(
        vault.join("target.md"),
        "# Changed heading\nnewword [[source]]\n",
    )
    .unwrap();
    database.rebuild(&vault).unwrap();
    let original_edge: (i64, i64, i64) = Connection::open(&db_path).unwrap().query_row(
        "SELECT l.id, l.source_note_id, l.target_note_id FROM links l JOIN notes n ON n.id=l.source_note_id WHERE n.path='source.md'", [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    ).unwrap();
    assert_eq!(before, original_edge);
    assert_eq!(database.backlinks("source").unwrap().len(), 1);
    assert!(database.search("oldword", 10).unwrap().is_empty());
    assert_eq!(database.search("newword", 10).unwrap().len(), 1);
    check_fts(&db_path);
}

#[test]
fn legacy_or_outdated_index_is_fully_rebuilt_and_empty_vault_is_supported() {
    let directory = tempfile::tempdir().unwrap();
    let vault = directory.path().join("vault");
    fs::create_dir(&vault).unwrap();
    let note = vault.join("a.md");
    fs::write(&note, "# A\noriginal\n").unwrap();
    let db_path = directory.path().join("index.sqlite3");
    let mut database = Database::open(&db_path).unwrap();
    database.rebuild(&vault).unwrap();
    let before = snapshot(&db_path);
    for version in [None, Some("outdated")] {
        let connection = Connection::open(&db_path).unwrap();
        connection
            .execute("DELETE FROM metadata WHERE key = 'index_version'", [])
            .unwrap();
        if let Some(version) = version {
            connection
                .execute(
                    "INSERT INTO metadata(key,value) VALUES('index_version',?1)",
                    [version],
                )
                .unwrap();
        }
        connection
            .execute(
                "UPDATE chunks SET body = 'corrupt', search_text = 'corrupt'",
                [],
            )
            .unwrap();
        drop(connection);
        // Exercise the legacy open_existing() migration path too.
        database = Database::open_existing(&db_path).unwrap();
        database.rebuild(&vault).unwrap();
        assert_eq!(before, snapshot(&db_path));
        check_fts(&db_path);
    }
    fs::remove_file(note).unwrap();
    let stats = database.rebuild(&vault).unwrap();
    assert_eq!((stats.notes, stats.chunks, stats.links), (0, 0, 0));
    assert!(database.search("original", 10).unwrap().is_empty());
    check_fts(&db_path);
    assert_eq!(database.rebuild(&vault).unwrap().notes, 0);
}

#[test]
fn changing_vault_does_not_reuse_notes_with_matching_paths() {
    let directory = tempfile::tempdir().unwrap();
    let a = directory.path().join("a");
    let b = directory.path().join("b");
    fs::create_dir(&a).unwrap();
    fs::create_dir(&b).unwrap();
    fs::write(a.join("same.md"), "# Same\nidentical text\n").unwrap();
    fs::write(b.join("same.md"), "# Same\nidentical text\n").unwrap();
    fs::write(a.join("only-a.md"), "# A\noriginal\n").unwrap();
    let db_path = directory.path().join("index.sqlite3");
    let mut database = Database::open(&db_path).unwrap();
    database.rebuild(&a).unwrap();
    assert_eq!(database.rebuild(&b).unwrap().notes, 1);
    let clean_path = directory.path().join("clean.sqlite3");
    Database::open(&clean_path).unwrap().rebuild(&b).unwrap();
    assert_eq!(snapshot(&db_path), snapshot(&clean_path));
    check_fts(&db_path);
}
