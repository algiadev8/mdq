# mdq

`mdq` is a local-first search and retrieval CLI for arbitrary Markdown
collections. It does not depend on a note-taking application or on a fixed
frontmatter schema.

## Features

- SQLite FTS5 BM25 search
- language-independent Latin word and CJK bigram indexing
- arbitrary YAML frontmatter path queries
- Wiki links (`[[Note]]`, `![[Note]]`) and Markdown links
  (`[label](Note.md)`, `![embed](file)`)
- backlink and graph traversal across both link styles
- compact JSON or text context retrieval for RAG pipelines
- local multilingual semantic embeddings and hybrid RRF retrieval
- Tasks, Obsidian Base, Dataview DQL, and restricted DataviewJS queries
- parser-generator-based native and compatibility expression grammars
- built-in per-command manual (`mdq man COMMAND`), language reference, and a
  ready-made use-case collection (`mdq man examples`)
- indexes stored outside the collection in the user cache directory

## Build

Requires Rust 1.88 or newer (ONNX Runtime dependency).

```sh
cargo build --release
```

The first `index` run (or any `index` run that builds embeddings) downloads
the `multilingual-e5-small` model. Embeddings are cached in SQLite by content
hash, so rebuilding the Markdown index does not require recomputing unchanged
chunks. The downloaded model cache is approximately 500 MB and is stored
under the operating system cache directory.

## Usage

```sh
mdq --vault ~/notes index
mdq --vault ~/notes index --only bm25
mdq --vault ~/notes index --only embed
mdq --vault ~/notes pipeline \
  --stage 'filter:created >= 2026-01-01 and labels contains research' \
  --stage 'bm25+rag:public key encryption'
mdq --vault ~/notes search "lattice cryptography"
mdq --vault ~/notes search "lattice cryptography" --exclude '^archive/'
mdq --vault ~/notes search "lattice cryptography" --only bm25
mdq --vault ~/notes search "lattice cryptography" --only rag
mdq --vault ~/notes search "lattice cryptography" --verbose
mdq --vault ~/notes query 'project.state = active'
mdq --vault ~/notes query 'custom.items contains value'
mdq --vault ~/notes query --language tasks $'not done\nsort by due'
mdq --vault ~/notes query --language base --file views/projects.base
mdq --vault ~/notes query --language dataview \
  'TABLE file.name, status FROM "Projects" WHERE status = "active"'
mdq --vault ~/notes query --language dataviewjs \
  'dv.list(dv.pages().map(page => page.file.link))'
mdq --vault ~/notes backlinks "Folder/Note"
mdq --vault ~/notes links "Folder/Note"
mdq --vault ~/notes unresolved-links
mdq --vault ~/notes graph "Folder/Note" --depth 2
mdq --vault ~/notes graph "Folder/Note" --direction outgoing --depth unlimited
mdq --vault ~/notes status
mdq quickref
mdq manual
mdq man search
mdq man examples
mdq man tasks
```

`pipeline` is the canonical execution model. Stages run in the exact order
given and may be repeated:

```text
filter[@language]:EXPRESSION
bm25:QUERY
rag:QUERY
bm25+rag:QUERY
```

`search` and native `query` are convenience aliases for one-stage pipelines.
Compatibility query languages return structured records through a separate
adapter API because their result domain can be pages, tasks, groups, tables,
or rendered values rather than search chunks.

## Index freshness

`search`, `query`, `backlinks`, `links`, `unresolved-links`, `graph`, and
`pipeline` check the index against the vault before running. If the vault
already has an index and only a small number of files or chunks have drifted
(`--auto-threshold`, default 20), `mdq` refreshes the index or embeddings
automatically and proceeds. A vault that has never been indexed, or that has
drifted past the threshold, requires an explicit `mdq index` run instead of
refreshing silently.

Index updates are incremental: every Markdown file is read and SHA-256 checked,
while unchanged notes reuse their parsed chunks and full-text index entries.
Only changed/new notes are parsed and tokenized; removed notes are deleted.
Content-only edits preserve incoming links, while added/removed paths trigger
full link re-resolution (including ambiguity checks). Updates are atomic and
retain the embedding cache. Older index formats or a different vault trigger a
full rebuild automatically. The search-time freshness check still uses file
mtime and size; run `mdq index --only bm25` explicitly to detect same-size edits
that preserve the timestamp.

A network-free performance benchmark (initial build, unchanged vault, one edit)
is available with `cargo test --release --test index_performance -- --ignored --nocapture`.

## Embedding performance

Pending embeddings are grouped by effective tokenizer length before inference,
reducing padding work without changing the model, passage/query prefixes,
512-token context limit, pooling, or content-hash cache keys. The CPU default is
8 documents per batch; use `index --batch-size N` to tune other hardware.
Automatic embedding refreshes use the same default. Completed batches are saved
immediately, so interrupted indexing resumes from the embedding cache.

On Apple Silicon macOS, the default `MDQ_EMBED_BACKEND=auto` tries **CoreML**
(MLProgram, with GPU/Neural Engine selection managed by Apple). It specializes
fixed batch/token shapes to avoid the dynamic-dimension fallback of the original
ONNX model. Other platforms keep the CPU path. Initialization or inference
failure retries the same batch on CPU and disables CoreML for that operation.
Each new shape is checked against CPU embeddings before its results are cached.
The compiled CoreML cache is separated by model SHA-256, runtime/settings, and
shape. First use compiles device-specific models and can take tens of seconds
per shape; even cached session startup can make small one-off jobs slower than
CPU. This path is primarily useful for sustained indexing, not a promise that
every invocation is faster.

Use `MDQ_EMBED_BACKEND=cpu` to force the original CPU path, or
`MDQ_EMBED_BACKEND=coreml` for strict diagnostics (errors instead of silent CPU
fallback). `MDQ_COREML_PROFILE=target/coreml-profile` enables ORT execution
profiles recording which provider actually ran the graph.

To compare the original 64-document batching with the selected backend and
test the complete embedding pipeline against reference vectors, run:

```sh
cargo test --release --test embedding_performance -- --ignored --nocapture --test-threads=1
```

This manual benchmark requires the multilingual-e5-small model in the local
cache (otherwise fastembed downloads it). `MDQ_BENCH_THREADS` changes the inference
microbenchmark's thread count; the application retains 4 CPU threads. To isolate
the CPU path, prefix the command with `MDQ_EMBED_BACKEND=cpu`.

The Apple-only manual test below checks queries, partial batches, truncation,
CPU compatibility, and **steady-state inference** separately from compilation
and model/session startup:

```sh
cargo test --release --lib coreml_real_model_queries_tails_truncation_and_warm_throughput -- --ignored --nocapture
```

On an M3, 40 full-length (512-token) inputs took 4.231 s on CPU versus 0.873 s
through CoreML after warm-up (4.85x). These are synthetic measurements, not an
end-to-end promise: a 64-input mixed-length indexing run took about 139 s on
first compilation and 32 s with compiled artifacts cached, so short runs can
still favor CPU. CoreML compilation/session startup is excluded from the 4.85x
number. Compiled per-shape artifacts also require additional cache disk space.

## Output

Default text output is minimized for piping into an LLM: one path (with
`#heading` when present) per result, followed by its content, and no score or
label noise. This also applies to `--json`: search results contain only the
path, heading, and content by default. Pass `--verbose` to a `search` or
`pipeline` command to include the score and other ranking details.

Tasks queries emit compact rows containing `path`, `line`, and the original
Markdown task line as `task`. Pass `--verbose` to include the full Tasks
compatibility record; this also controls whether `--json` uses compact or full
rows.

## Native filter language

Frontmatter paths are generic dotted paths with optional array indices. No
property name has built-in meaning.

```text
arbitrary.nested.key = value
arbitrary.list[0] != value
score >= 3.5
date >= 2026-01-01
datetime < 2026-06-15T00:00:00+09:00
labels contains research
labels contains_all [research,cryptography]
labels overlaps [paper,project]
state in [active,paused]
name starts_with prefix
name ends_with suffix
name matches "^prefix-"
arbitrary.key exists
arbitrary.key missing
not archived = true
created >= 2026-01-01 and (score > 3 or labels contains urgent)
```

Supported scalar types come from YAML: strings, numbers, booleans, and null.
Lists and objects use YAML flow syntax. Ordering compares numbers numerically,
ISO dates and RFC 3339 timestamps chronologically, and other strings
lexicographically.

## Link resolution

`mdq` indexes both Wiki links and standard Markdown links. Relative Markdown
paths are resolved from the source file directory, percent-encoded paths are
decoded, and external URLs are ignored. A filename-only link is resolved only
when it identifies a single Markdown note or non-Markdown vault file;
ambiguous links remain unresolved. Non-Markdown files are not indexed as
notes, but existing vault files such as Base documents and attachments are not
counted as unresolved links in `status`.
Use `mdq --vault ~/notes unresolved-links` to print the actual unresolved link
records. For note arguments, an exact path is preferred; if it is not found,
the unique basename/title fallback is retained and a diagnostic on stderr shows
the indexed path it selected.

## Index location

By default, indexes are written under the operating system cache directory,
not inside the Markdown collection. Use `--db <path>` to override it.

## Extending mdq

The Rust library exposes:

- `QueryAdapter` and `CompatibilityEngine::register` for complete query
  languages returning `RecordSet`
- `ScriptEngine` for replaceable restricted script execution
- `QueryLanguage` and `MetadataFilter` for predicate-compatible metadata
  syntaxes
- `StageExecutor` and `PipelineEngine::register_stage` for ordered retrieval
  stages

Run `mdq manual extensions` or see
[`docs/architecture.md`](docs/architecture.md).
