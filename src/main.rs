use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use mdq::compat::CompatibilityEngine;
use mdq::core::{QueryContext, RecordSet};
use mdq::db::{Database, default_db_path};
use mdq::manual;
use mdq::model::{
    CompactContextItem, ContextItem, GraphOutput, IndexOutput, NoteRef, NoteResolution, SearchHit,
};
use mdq::pipeline::{PipelineEngine, StageSpec};
use mdq::semantic;
use regex::Regex;
use serde::Serialize;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Root directory of the Markdown collection.
    #[arg(long, global = true, default_value = ".")]
    vault: PathBuf,

    /// Override the generated SQLite index path.
    #[arg(long, global = true)]
    db: Option<PathBuf>,

    /// Emit machine-readable JSON.
    #[arg(long, global = true)]
    json: bool,

    /// Maximum number of changed files/chunks to refresh automatically before
    /// running a search command. Larger drift requires an explicit `index` run.
    #[arg(long, global = true, default_value_t = 500)]
    auto_threshold: usize,

    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BuildTarget {
    /// BM25 full-text index only.
    Bm25,
    /// Local semantic embeddings only.
    Embed,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SearchEngine {
    /// BM25 full-text retrieval only.
    Bm25,
    /// Semantic embedding retrieval only.
    Rag,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum GraphDirection {
    Outgoing,
    Incoming,
    Both,
}

#[derive(Subcommand)]
enum Command {
    /// Show the compact, agent-oriented command reference.
    Quickref,
    /// Build the BM25 index and local semantic embeddings.
    Index {
        /// Build only this target instead of both.
        #[arg(long, value_enum)]
        only: Option<BuildTarget>,
        /// Maximum documents per embedding batch; smaller CPU batches avoid large attention buffers.
        #[arg(long, default_value_t = semantic::DEFAULT_BATCH_SIZE)]
        batch_size: usize,
    },
    /// Retrieve source context for a query (default: hybrid BM25 + semantic).
    Search {
        /// Search query. Queries with spaces must be quoted.
        query: String,
        #[arg(short, long, default_value_t = 8)]
        limit: usize,
        /// Exclude note paths matching this regex. Can be repeated.
        #[arg(long)]
        exclude: Vec<String>,
        /// Include only note paths matching this regex. Can be repeated.
        #[arg(long)]
        path: Vec<String>,
        /// Total character budget for context output (default: unlimited).
        #[arg(long)]
        max_chars: Option<usize>,
        /// Use only one retrieval engine instead of the hybrid default.
        #[arg(long, value_enum)]
        only: Option<SearchEngine>,
        /// Include score and heading detail in the output.
        #[arg(short, long)]
        verbose: bool,
    },
    /// Run a native, Tasks, Base, Dataview, or DataviewJS query.
    Query {
        /// Inline query source. Use --file for .base files or longer scripts. Expressions with spaces or operators must be quoted: 'type = person'.
        expression: Option<String>,
        #[arg(long, default_value = "native")]
        language: String,
        #[arg(long)]
        file: Option<PathBuf>,
        /// Current note used by this.file and dv.current().
        #[arg(long)]
        current: Option<PathBuf>,
        /// Tasks-only custom status mapping: SYMBOL=TYPE or SYMBOL=NAME:TYPE[:NEXT].
        ///
        /// Examples: --tasks-status '-=CANCELLED', --tasks-status '?=Needs Triage:ON_HOLD'
        #[arg(long = "tasks-status")]
        tasks_status: Vec<String>,
        /// Tasks-only global filter string. Only checklist items containing this text are tasks.
        #[arg(long = "tasks-global-filter")]
        tasks_global_filter: Option<String>,
        /// Tasks-only global query prepended to every Tasks query unless it says `ignore global query`.
        #[arg(long = "tasks-global-query")]
        tasks_global_query: Option<String>,
        #[arg(short, long, default_value_t = 100)]
        limit: usize,
        /// Tasks only: include the full compatibility record in the output.
        #[arg(short, long)]
        verbose: bool,
    },
    /// List links from a note.
    Links { note: String },
    /// List links that do not resolve to a vault note or file.
    UnresolvedLinks,
    /// Traverse resolved links. `backlinks` is sugar for incoming depth 1.
    #[command(visible_alias = "backlinks")]
    Graph {
        /// One or more note paths, titles, or filenames to start from.
        #[arg(required = true)]
        notes: Vec<String>,
        #[arg(long, value_enum, default_value_t = GraphDirection::Both)]
        direction: GraphDirection,
        /// Maximum hops, or `unlimited`.
        #[arg(long, default_value = "2")]
        depth: String,
    },
    /// Run filters and rankers in the exact order supplied.
    Pipeline {
        /// Stage syntax: filter[@language]:EXPR, bm25:QUERY, rag:QUERY, bm25+rag:QUERY
        #[arg(long = "stage", required = true)]
        stages: Vec<String>,
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
        /// Return full chunk context instead of search snippets.
        #[arg(long)]
        context: bool,
        /// Total character budget for context output (default: unlimited).
        #[arg(long)]
        max_chars: Option<usize>,
        /// Include score and heading detail in the output.
        #[arg(short, long)]
        verbose: bool,
    },
    /// Show index metadata and counts.
    Status,
    /// Show the per-command manual, query language reference, and examples.
    #[command(alias = "man")]
    Manual {
        /// Topic: overview, index, search, query, backlinks, links,
        /// unresolved-links, graph, pipeline, status, native, tasks, base,
        /// dataview, dataviewjs, extensions, examples, all.
        topic: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if matches!(cli.command, Command::Quickref) {
        print!("{}", manual::quickref());
        return Ok(());
    }
    if let Command::Manual { topic } = &cli.command {
        let rendered = manual::render(topic.as_deref())?;
        print!("{rendered}");
        if !rendered.ends_with('\n') {
            println!();
        }
        return Ok(());
    }
    let vault = cli
        .vault
        .canonicalize()
        .with_context(|| format!("vault does not exist: {}", cli.vault.display()))?;
    if !vault.is_dir() {
        bail!("vault must be a directory: {}", vault.display());
    }
    let db_path = cli.db.unwrap_or(default_db_path(&vault)?);
    let mut database = if matches!(cli.command, Command::Index { .. }) {
        Database::open(&db_path)?
    } else {
        Database::open_existing(&db_path).with_context(|| {
            format!(
                "index is not initialized; run `mdq --vault {} index` first",
                vault.display()
            )
        })?
    };
    let pipeline = PipelineEngine::standard();

    let needs_fresh_index = !matches!(
        cli.command,
        Command::Index { .. } | Command::Status | Command::Manual { .. }
    );
    if needs_fresh_index {
        ensure_index_fresh(&mut database, &vault, cli.auto_threshold)?;
    }

    match cli.command {
        Command::Index { only, batch_size } => {
            let build_bm25 = only != Some(BuildTarget::Embed);
            let build_embed = only != Some(BuildTarget::Bm25);
            let stats = build_bm25.then(|| database.rebuild(&vault)).transpose()?;
            let embedded = build_embed
                .then(|| semantic::embed_missing(&mut database, batch_size))
                .transpose()?;
            if cli.json {
                print_json(&IndexOutput {
                    vault: vault.to_string_lossy().into_owned(),
                    database: db_path.to_string_lossy().into_owned(),
                    notes: stats.as_ref().map(|stats| stats.notes),
                    chunks: stats.as_ref().map(|stats| stats.chunks),
                    links: stats.as_ref().map(|stats| stats.links),
                    embedded,
                    model: build_embed.then_some(semantic::MODEL_ID),
                })?;
            } else {
                if let Some(stats) = &stats {
                    println!(
                        "indexed {} notes, {} chunks, {} links",
                        stats.notes, stats.chunks, stats.links
                    );
                }
                if let Some(embedded) = embedded {
                    println!("embedded {embedded} chunks with {}", semantic::MODEL_ID);
                }
                println!("{}", db_path.display());
            }
        }
        Command::Search {
            query,
            limit,
            exclude,
            path,
            max_chars,
            only,
            verbose,
        } => {
            if query.trim().is_empty() {
                bail!("search query cannot be empty");
            }
            if only != Some(SearchEngine::Bm25) {
                ensure_embeddings_fresh(&mut database, &vault, cli.auto_threshold)?;
            }
            let (stage, fetch_limit) = match only {
                Some(SearchEngine::Bm25) => ("bm25", limit.saturating_mul(3).max(limit)),
                Some(SearchEngine::Rag) => ("rag", limit.saturating_mul(3).max(limit)),
                None => ("bm25+rag", limit.saturating_mul(5).max(30)),
            };
            let fetch_limit = if exclude.is_empty() && path.is_empty() {
                fetch_limit
            } else {
                usize::MAX
            };
            let mut hits = run_alias(&pipeline, &database, stage, &query, fetch_limit)?;
            rerank_search_hits(&mut hits, &query);
            include_hits(&mut hits, &vault, &path)?;
            exclude_hits(&mut hits, &vault, &exclude)?;
            let context = build_context(
                &database,
                hits,
                limit,
                max_chars.unwrap_or(usize::MAX),
                Some(&query),
            )?;
            if context.len() == limit {
                eprintln!("note: showing top {limit} results; use --limit to see more");
            }
            output_context(context, cli.json, verbose)?;
        }
        Command::Query {
            expression,
            language,
            file,
            current,
            limit,
            verbose,
            tasks_status,
            tasks_global_filter,
            tasks_global_query,
        } => {
            let language = language.to_ascii_lowercase();
            let has_tasks_options = !tasks_status.is_empty()
                || tasks_global_filter.is_some()
                || tasks_global_query.is_some();
            if has_tasks_options && language != "tasks" {
                bail!("--tasks-* options can only be used with --language tasks");
            }
            let source = match (expression, file) {
                (Some(source), None) => source,
                (None, Some(path)) => std::fs::read_to_string(&path)
                    .with_context(|| format!("cannot read query file {}", path.display()))?,
                (Some(_), Some(_)) => {
                    bail!("provide either inline query source or --file, not both")
                }
                (None, None) => bail!("query source is required"),
            };
            if language == "native" {
                if source.trim().is_empty() {
                    bail!("query expression cannot be empty");
                }
                let expression = mdq::query::Expression::parse(&source)?;
                let mut notes = database.query_frontmatter(&expression)?;
                let total = notes.len();
                notes.truncate(limit);
                if total > limit {
                    eprintln!(
                        "note: {total} results found, showing first {limit} (use --limit to adjust)"
                    );
                }
                output_notes(&notes, cli.json)?;
            } else {
                let current_file = current
                    .map(|path| resolve_current_file(&vault, path))
                    .transpose()?;
                let context = QueryContext {
                    database: &database,
                    vault: &vault,
                    current_file,
                };
                let compatibility = if has_tasks_options {
                    CompatibilityEngine::standard_with_tasks_settings(
                        &tasks_status,
                        tasks_global_filter,
                        tasks_global_query,
                    )?
                } else {
                    CompatibilityEngine::standard()
                };
                let mut result = compatibility.execute(&language, &context, &source)?;
                let total = result.rows.len();
                result.rows.truncate(limit);
                if total > limit {
                    let msg = if language == "tasks" {
                        format!(
                            "Tasks query produced {total} rows after Tasks query limits; CLI --limit is additionally showing first {limit} rows"
                        )
                    } else {
                        format!(
                            "{total} results found, showing first {limit} (use --limit to adjust)"
                        )
                    };
                    result.diagnostics.push(msg.clone());
                    if !cli.json {
                        eprintln!("note: {msg}");
                    }
                }
                if language == "tasks" && !verbose {
                    result = compact_tasks_record_set(&result);
                }
                output_record_set(&result, cli.json)?;
            }
        }
        Command::Links { note } => {
            let Some(resolution) = database.resolve_note(&note)? else {
                bail!("note not found or ambiguous: {note}");
            };
            report_note_fallback(&note, &resolution);
            let links = database.outgoing_links(&resolution.note.path)?;
            if cli.json {
                print_json(&links)?;
            } else {
                for link in links {
                    let resolved = link
                        .resolved_path
                        .unwrap_or_else(|| "<unresolved>".to_owned());
                    println!("{}\t{}\t{}", link.raw_target, resolved, link.embed);
                }
            }
        }
        Command::UnresolvedLinks => {
            let links = database.unresolved_links()?;
            if cli.json {
                print_json(&links)?;
            } else {
                for link in links {
                    println!("{}\t{}\t{}", link.source.path, link.raw_target, link.embed);
                }
            }
        }
        Command::Graph {
            notes,
            mut direction,
            mut depth,
        } => {
            // Clap aliases do not expose which spelling selected the variant. Detect it
            // before parsing would add global state, so normalize `backlinks` in argv.
            let invoked_as_backlinks = std::env::args().any(|arg| arg == "backlinks");
            if invoked_as_backlinks {
                direction = GraphDirection::Incoming;
                depth = "1".to_owned();
            }
            let depth = parse_graph_depth(&depth)?;
            let graph = traverse_graph(&database, &notes, direction, depth)?;
            if cli.json {
                print_json(&graph)?;
            } else {
                output_notes(&graph.notes, false)?;
            }
        }
        Command::Pipeline {
            stages,
            limit,
            context,
            max_chars,
            verbose,
        } => {
            let stages = stages
                .iter()
                .map(|stage| StageSpec::parse(stage))
                .collect::<Result<Vec<_>>>()?;
            if stages
                .iter()
                .any(|stage| stage.name == "rag" || stage.name == "bm25+rag")
            {
                ensure_embeddings_fresh(&mut database, &vault, cli.auto_threshold)?;
            }
            let mut hits = pipeline.execute(&database, &stages)?;
            let total = hits.len();
            if context {
                let context = build_context(
                    &database,
                    hits,
                    limit,
                    max_chars.unwrap_or(usize::MAX),
                    None,
                )?;
                if total > limit {
                    eprintln!(
                        "note: {total} results found, showing first {limit} (use --limit to adjust)"
                    );
                }
                output_context(context, cli.json, verbose)?;
            } else {
                hits.truncate(limit);
                if total > limit {
                    eprintln!(
                        "note: {total} results found, showing first {limit} (use --limit to adjust)"
                    );
                }
                output_hits(&hits, cli.json, verbose)?;
            }
        }
        Command::Status => {
            let status = database.status(semantic::MODEL_ID)?;
            if cli.json {
                print_json(&status)?;
            } else {
                println!(
                    "vault: {}",
                    status.vault.as_deref().unwrap_or("<not indexed>")
                );
                println!("has_index: {}", status.has_index);
                println!(
                    "indexed_at: {}",
                    status.indexed_at.as_deref().unwrap_or("-")
                );
                println!("notes: {}", status.notes);
                println!("chunks: {}", status.chunks);
                println!("links: {}", status.links);
                println!("unresolved_links: {}", status.unresolved_links);
                println!("embeddings: {}", status.embeddings);
                println!("cached_embeddings: {}", status.cached_embeddings);
                println!("index_stale: {}", status.index_stale);
                println!("embeddings_stale: {}", status.embeddings_stale);
                println!("database: {}", db_path.display());
            }
        }
        Command::Manual { .. } | Command::Quickref => {
            unreachable!("documentation commands exit before database initialization")
        }
    }
    Ok(())
}

/// Bound on how many times [`ensure_index_fresh`] waits for a concurrent `mdq` process to
/// finish refreshing the index before giving up.
const MAX_LOCK_WAIT_ROUNDS: u32 = 8;

/// Refreshes the BM25 index when the vault has drifted from it. A small drift on an
/// already-indexed vault is refreshed automatically; a large or first-time drift requires
/// an explicit `index` run, since that may mean the wrong vault or an unbuilt index.
///
/// If another `mdq` process is already refreshing the same index, this does not contend
/// with it for the write lock: it waits (with randomized backoff, to avoid many processes
/// retrying in lockstep) for that process to finish and re-checks staleness, only
/// attempting the refresh itself if the index is still stale once the lock is free.
fn ensure_index_fresh(database: &mut Database, vault: &Path, threshold: usize) -> Result<()> {
    if let Some(indexed) = database.indexed_vault()? {
        if indexed != vault.to_string_lossy().as_ref() {
            bail!(
                "database was built for a different vault ({}); run `mdq --vault {} index` to re-index for this vault",
                indexed,
                vault.display()
            );
        }
    }

    for round in 0..MAX_LOCK_WAIT_ROUNDS {
        let changed = database.staleness(vault)?;
        if changed == 0 {
            return Ok(());
        }
        if !(database.has_index()? && changed <= threshold) {
            bail!(
                "index is stale ({changed} changed file(s)); run `mdq --vault {} index` to refresh",
                vault.display()
            );
        }

        eprintln!("vault changed ({changed} file(s)); refreshing index automatically");
        match database.try_rebuild(vault)? {
            Some(_) => return Ok(()),
            None => {
                let wait = jittered_wait(round);
                eprintln!(
                    "index is locked by another process refreshing it; waiting {:.1}s and re-checking",
                    wait.as_secs_f64()
                );
                std::thread::sleep(wait);
            }
        }
    }

    bail!(
        "index is still locked by another process after waiting; run `mdq --vault {} index` to refresh once it is free",
        vault.display()
    )
}

/// Randomized, capped exponential backoff so that many `mdq` processes racing to refresh
/// the same stale index don't all retry at the same moment.
fn jittered_wait(round: u32) -> Duration {
    let base_ms = 200u64.saturating_mul(1u64 << round.min(4));
    let jitter_ms = random_below(base_ms / 2 + 1);
    Duration::from_millis(base_ms + jitter_ms).min(Duration::from_secs(5))
}

/// A cheap, non-cryptographic random value in `[0, bound)`, without pulling in a `rand`
/// dependency just for retry jitter.
fn random_below(bound: u64) -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    if bound == 0 {
        return 0;
    }
    RandomState::new().build_hasher().finish() % bound
}

/// Refreshes embeddings when chunks are missing them, under the same threshold policy as
/// `ensure_index_fresh`.
fn ensure_embeddings_fresh(database: &mut Database, vault: &Path, threshold: usize) -> Result<()> {
    let missing = database.missing_embeddings_count(semantic::MODEL_ID)?;
    if missing == 0 {
        return Ok(());
    }
    if database.has_embeddings()? && missing <= threshold {
        eprintln!("{missing} chunk(s) missing embeddings; embedding automatically");
        semantic::embed_missing(database, semantic::DEFAULT_BATCH_SIZE)?;
        Ok(())
    } else {
        bail!(
            "embeddings are missing or stale ({missing} chunk(s)); run `mdq --vault {} index` to refresh, or use `--only bm25` to search without embeddings",
            vault.display()
        )
    }
}

fn output_hits(hits: &[SearchHit], json: bool, verbose: bool) -> Result<()> {
    if json {
        return print_json(hits);
    }
    for hit in hits {
        let heading = hit
            .heading
            .as_deref()
            .map(|heading| format!("#{heading}"))
            .unwrap_or_default();
        if verbose {
            println!("{}{} (score={:.6})", hit.path, heading, hit.score);
        } else {
            println!("{}{}", hit.path, heading);
        }
        println!("{}", hit.snippet);
    }
    Ok(())
}

fn output_notes(notes: &[NoteRef], json: bool) -> Result<()> {
    if json {
        return print_json(notes);
    }
    for note in notes {
        println!("{}", note.path);
    }
    Ok(())
}

fn output_record_set(result: &RecordSet, json: bool) -> Result<()> {
    if json {
        return print_json(result);
    }
    for row in &result.rows {
        println!("{}", serde_json::to_string(row)?);
    }
    for diagnostic in &result.diagnostics {
        eprintln!("warning: {diagnostic}");
    }
    Ok(())
}

fn compact_tasks_record_set(result: &RecordSet) -> RecordSet {
    let rows = result
        .rows
        .iter()
        .map(|row| {
            ["path", "line"]
                .into_iter()
                .filter_map(|key| row.get(key).cloned().map(|value| (key.to_owned(), value)))
                .chain(
                    row.get("originalMarkdown")
                        .cloned()
                        .map(|value| ("task".to_owned(), value)),
                )
                .collect()
        })
        .collect();
    let mut compact = RecordSet::new("tasks", rows);
    compact.columns = vec!["path".to_owned(), "line".to_owned(), "task".to_owned()];
    compact.diagnostics = result.diagnostics.clone();
    compact
}

fn parse_graph_depth(value: &str) -> Result<Option<usize>> {
    if value.eq_ignore_ascii_case("unlimited") {
        return Ok(None);
    }
    value.parse::<usize>().map(Some).with_context(|| {
        format!("invalid graph depth {value:?}; expected a non-negative integer or 'unlimited'")
    })
}

fn traverse_graph(
    database: &Database,
    inputs: &[String],
    direction: GraphDirection,
    depth: Option<usize>,
) -> Result<GraphOutput> {
    const NOTE_CANDIDATE_LIMIT: usize = 20;
    let mut starts = Vec::new();
    for input in inputs {
        let Some(resolution) = database.resolve_note(input)? else {
            let candidates = database.note_candidates(input, NOTE_CANDIDATE_LIMIT)?;
            let hint = if candidates.is_empty() {
                "no basename, suffix, or stem candidates found".to_owned()
            } else {
                format!("candidates: {}", candidates.join(", "))
            };
            bail!("note not found or ambiguous: {input}; {hint}; retry with an exact path");
        };
        report_note_fallback(input, &resolution);
        starts.push(resolution.note);
    }
    starts.sort_by(|a, b| a.path.cmp(&b.path));
    starts.dedup_by(|a, b| a.path == b.path);
    let mut queue = VecDeque::new();
    let mut seen = HashSet::new();
    for note in &starts {
        seen.insert(note.path.clone());
        queue.push_back((note.clone(), 0));
    }
    let mut result = starts.clone();

    while let Some((note, current_depth)) = queue.pop_front() {
        if depth.is_some_and(|depth| current_depth >= depth) {
            continue;
        }
        let mut neighbors = Vec::new();
        if direction != GraphDirection::Incoming {
            for link in database.outgoing_links(&note.path)? {
                if let Some(target) = link.target {
                    neighbors.push(target);
                }
            }
        }
        if direction != GraphDirection::Outgoing {
            for link in database.backlinks(&note.path)? {
                neighbors.push(link.source);
            }
        }
        neighbors.sort_by(|a, b| a.path.cmp(&b.path));
        for neighbor in neighbors {
            if seen.insert(neighbor.path.clone()) {
                queue.push_back((neighbor.clone(), current_depth + 1));
                result.push(neighbor);
            }
        }
    }
    Ok(GraphOutput {
        direction: format!("{direction:?}").to_ascii_lowercase(),
        depth,
        starts,
        notes: result,
    })
}

fn report_note_fallback(input: &str, resolution: &NoteResolution) {
    if resolution.used_fallback {
        eprintln!(
            "note: resolved {input:?} via basename/title fallback to {}",
            resolution.note.path
        );
    }
}

fn build_context(
    database: &Database,
    hits: Vec<SearchHit>,
    limit: usize,
    max_chars: usize,
    query: Option<&str>,
) -> Result<Vec<ContextItem>> {
    let mut best_by_path = HashMap::<String, SearchHit>::new();
    for hit in hits {
        best_by_path
            .entry(hit.path.clone())
            .and_modify(|existing| {
                if hit.score > existing.score {
                    *existing = hit.clone();
                }
            })
            .or_insert(hit);
    }
    let mut hits: Vec<SearchHit> = best_by_path.into_values().collect();
    hits.sort_by(|left, right| right.score.total_cmp(&left.score));

    let mut used = 0;
    let mut context = Vec::new();
    for hit in hits.into_iter().take(limit) {
        let Some(body) = database.chunk_body(hit.chunk_id)? else {
            continue;
        };
        let remaining = max_chars.saturating_sub(used);
        if remaining == 0 {
            break;
        }
        let header_len = hit.path.chars().count()
            + hit
                .heading
                .as_deref()
                .map(|h| 1 + h.chars().count())
                .unwrap_or(0);
        let text_budget = remaining.saturating_sub(header_len);
        if text_budget == 0 {
            continue;
        }
        let text: String = body.chars().take(text_budget).collect();
        used += header_len + text.chars().count();
        let match_reasons = match_reasons(&hit, query);
        context.push(ContextItem {
            path: hit.path,
            heading: hit.heading,
            score: hit.score,
            text,
            match_reasons,
        });
    }
    Ok(context)
}

fn rerank_search_hits(hits: &mut [SearchHit], query: &str) {
    let terms = query
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    for hit in hits.iter_mut() {
        let path = hit.path.to_ascii_lowercase();
        let title = hit.title.to_ascii_lowercase();
        let heading = hit
            .heading
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if terms.iter().any(|term| title == *term) {
            hit.score += 0.08;
        }
        if terms.iter().any(|term| title.contains(term)) {
            hit.score += 0.04;
        }
        if terms.iter().any(|term| heading.contains(term)) {
            hit.score += 0.025;
        }
        if terms.iter().any(|term| path.contains(term)) {
            hit.score += 0.015;
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
    });
}

fn match_reasons(hit: &SearchHit, query: Option<&str>) -> Vec<String> {
    let Some(query) = query else {
        return Vec::new();
    };
    let terms = query
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let mut reasons = vec!["retrieval".to_owned()];
    if terms
        .iter()
        .any(|term| hit.title.to_ascii_lowercase().contains(term))
    {
        reasons.push("title".to_owned());
    }
    if terms
        .iter()
        .any(|term| hit.path.to_ascii_lowercase().contains(term))
    {
        reasons.push("path".to_owned());
    }
    if terms.iter().any(|term| {
        hit.heading
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .contains(term)
    }) {
        reasons.push("heading".to_owned());
    }
    reasons
}

fn run_alias(
    pipeline: &PipelineEngine,
    database: &Database,
    stage: &str,
    argument: &str,
    limit: usize,
) -> Result<Vec<SearchHit>> {
    let spec = StageSpec::parse(&format!("{stage}:{argument}"))?;
    let mut hits = pipeline.execute(database, &[spec])?;
    hits.truncate(limit);
    Ok(hits)
}

fn exclude_hits(hits: &mut Vec<SearchHit>, vault: &Path, excludes: &[String]) -> Result<()> {
    if excludes.is_empty() {
        return Ok(());
    }
    let patterns = excludes
        .iter()
        .filter(|exclude| !normalize_exclude_pattern(vault, exclude).is_empty())
        .map(|exclude| compile_exclude_regex(vault, exclude))
        .collect::<Result<Vec<_>>>()?;
    hits.retain(|hit| {
        !patterns
            .iter()
            .any(|pattern| path_matches_exclude(&hit.path, pattern))
    });
    Ok(())
}

fn include_hits(hits: &mut Vec<SearchHit>, vault: &Path, includes: &[String]) -> Result<()> {
    if includes.is_empty() {
        return Ok(());
    }
    let patterns = includes
        .iter()
        .map(|include| compile_path_regex(vault, include, "--path"))
        .collect::<Result<Vec<_>>>()?;
    hits.retain(|hit| patterns.iter().any(|pattern| pattern.is_match(&hit.path)));
    Ok(())
}

fn compile_exclude_regex(vault: &Path, exclude: &str) -> Result<Regex> {
    compile_path_regex(vault, exclude, "--exclude")
}

fn compile_path_regex(vault: &Path, value: &str, flag: &str) -> Result<Regex> {
    let pattern = normalize_exclude_pattern(vault, value);
    Regex::new(&pattern).with_context(|| format!("invalid {flag} regex: {value}"))
}

fn normalize_exclude_pattern(vault: &Path, exclude: &str) -> String {
    let vault = vault.to_string_lossy().replace('\\', "/");
    let relative = exclude.strip_prefix(&vault).unwrap_or(exclude);
    relative
        .trim_start_matches('/')
        .trim_end_matches('/')
        .to_owned()
}

fn path_matches_exclude(path: &str, exclude: &Regex) -> bool {
    exclude.is_match(path)
}

fn resolve_current_file(vault: &Path, path: PathBuf) -> Result<PathBuf> {
    let resolved = if path.is_absolute() {
        path
    } else {
        vault.join(&path)
    };
    if !resolved.exists() {
        bail!("--current path does not exist: {}", resolved.display());
    }
    let canonical_vault = vault
        .canonicalize()
        .with_context(|| format!("cannot resolve vault path {}", vault.display()))?;
    let canonical_current = resolved
        .canonicalize()
        .with_context(|| format!("cannot resolve --current path {}", resolved.display()))?;
    if !canonical_current.starts_with(&canonical_vault) {
        bail!(
            "--current path must be inside the vault ({}): {}",
            canonical_vault.display(),
            canonical_current.display()
        );
    }
    Ok(canonical_current)
}

fn output_context(context: Vec<ContextItem>, json: bool, verbose: bool) -> Result<()> {
    if json {
        return if verbose {
            print_json(&context)
        } else {
            print_json(
                &context
                    .iter()
                    .map(CompactContextItem::from)
                    .collect::<Vec<_>>(),
            )
        };
    }
    for (index, item) in context.iter().enumerate() {
        if index > 0 {
            println!();
        }
        let heading = item
            .heading
            .as_deref()
            .map(|heading| format!("#{heading}"))
            .unwrap_or_default();
        if verbose {
            println!("{}{} (score={:.6})", item.path, heading, item.score);
        } else {
            println!("{}{}", item.path, heading);
        }
        println!("{}", item.text);
    }
    Ok(())
}

fn print_json<T: Serialize + ?Sized>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::fs;

    #[test]
    fn compact_tasks_output_keeps_only_location_and_original_markdown() {
        let row = BTreeMap::from([
            ("path".to_owned(), json!("Projects/alpha.md")),
            ("line".to_owned(), json!(10)),
            (
                "originalMarkdown".to_owned(),
                json!("  - [ ] Ship alpha 📅 2026-06-20"),
            ),
            ("description".to_owned(), json!("Ship alpha")),
            ("file".to_owned(), json!({"path": "Projects/alpha.md"})),
        ]);
        let mut full = RecordSet::new("tasks", vec![row]);
        full.diagnostics.push("example warning".to_owned());

        let compact = compact_tasks_record_set(&full);

        assert_eq!(compact.kind, "tasks");
        assert_eq!(compact.columns, ["path", "line", "task"]);
        assert_eq!(compact.rows[0].len(), 3);
        assert_eq!(compact.rows[0]["path"], "Projects/alpha.md");
        assert_eq!(compact.rows[0]["line"], 10);
        assert_eq!(compact.rows[0]["task"], "  - [ ] Ship alpha 📅 2026-06-20");
        assert_eq!(compact.diagnostics, ["example warning"]);
    }

    #[test]
    fn query_accepts_verbose_flag() {
        let cli = Cli::try_parse_from([
            "mdq",
            "query",
            "not done",
            "--language",
            "tasks",
            "--verbose",
        ])
        .unwrap();

        assert!(matches!(cli.command, Command::Query { verbose: true, .. }));
    }

    #[test]
    fn current_file_must_resolve_inside_vault() {
        let directory = tempfile::tempdir().unwrap();
        let vault = directory.path().join("vault");
        fs::create_dir_all(&vault).unwrap();
        fs::write(vault.join("inside.md"), "# Inside\n").unwrap();
        fs::write(directory.path().join("outside.md"), "# Outside\n").unwrap();

        assert!(resolve_current_file(&vault, PathBuf::from("inside.md")).is_ok());
        assert!(resolve_current_file(&vault, PathBuf::from("../outside.md")).is_err());
    }

    #[test]
    fn exclude_hits_matches_regexes_and_directory_spelling_variants() {
        let vault = PathBuf::from("/vault");
        let mut hits = vec![
            SearchHit {
                chunk_id: 1,
                path: "keep.md".to_owned(),
                title: "Keep".to_owned(),
                heading: None,
                score: 1.0,
                snippet: String::new(),
            },
            SearchHit {
                chunk_id: 2,
                path: "archive/old.md".to_owned(),
                title: "Old".to_owned(),
                heading: None,
                score: 0.9,
                snippet: String::new(),
            },
            SearchHit {
                chunk_id: 3,
                path: "drafts/todo.md".to_owned(),
                title: "Todo".to_owned(),
                heading: None,
                score: 0.8,
                snippet: String::new(),
            },
            SearchHit {
                chunk_id: 4,
                path: "test/item.md".to_owned(),
                title: "Test".to_owned(),
                heading: None,
                score: 0.7,
                snippet: String::new(),
            },
        ];

        exclude_hits(
            &mut hits,
            &vault,
            &[
                "^archive/".to_owned(),
                "/vault/drafts/.*\\.md$".to_owned(),
                "test/".to_owned(),
            ],
        )
        .unwrap();

        let paths: Vec<&str> = hits.iter().map(|hit| hit.path.as_str()).collect();
        assert_eq!(paths, vec!["keep.md"]);
    }

    #[test]
    fn exclude_regex_normalizes_directory_without_trailing_slash() {
        let vault = PathBuf::from("/vault");
        let mut hits = vec![SearchHit {
            chunk_id: 1,
            path: "test/item.md".to_owned(),
            title: "Test".to_owned(),
            heading: None,
            score: 1.0,
            snippet: String::new(),
        }];

        exclude_hits(&mut hits, &vault, &["test".to_owned()]).unwrap();

        assert!(hits.is_empty());
    }

    #[test]
    fn exclude_hits_reports_invalid_regex() {
        let vault = PathBuf::from("/vault");
        let mut hits = Vec::new();

        let error = exclude_hits(&mut hits, &vault, &["[".to_owned()]).unwrap_err();

        assert!(error.to_string().contains("invalid --exclude regex"));
    }
}
