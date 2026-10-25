use anyhow::{Context, Result, ensure};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::db::Database;
use crate::model::{EmbeddingInput, SearchHit};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple;

pub const MODEL_ID: &str = "multilingual-e5-small";
// Large CPU batches waste memory bandwidth on attention intermediates. Keep this
// small; --batch-size remains available for hardware-specific tuning.
pub const DEFAULT_BATCH_SIZE: usize = 8;

pub fn embed_missing(database: &mut Database, batch_size: usize) -> Result<usize> {
    let missing = database.missing_embeddings(MODEL_ID)?;
    if missing.is_empty() {
        return Ok(0);
    }

    let mut model = load_model()?;
    // The tokenizer pads each batch to its longest sequence. Sort by *effective*
    // token length (including the existing 512-token truncation), not byte length:
    // Japanese and Latin text have very different bytes/token ratios.
    let missing = order_by_token_length(missing, |text| {
        model
            .cpu
            .tokenizer
            .encode(text, true)
            .map(|encoding| encoding.len())
            .map_err(|error| anyhow::anyhow!("embedding tokenization failed: {error}"))
    })?;
    embed_batches(database, &missing, batch_size, |documents| {
        model
            .embed(documents, Some(batch_size.max(1)))
            .context("embedding inference failed")
    })
}

fn order_by_token_length(
    inputs: Vec<EmbeddingInput>,
    mut token_length: impl FnMut(&str) -> Result<usize>,
) -> Result<Vec<EmbeddingInput>> {
    let mut ranked = inputs
        .into_iter()
        .map(|input| token_length(&input.text).map(|length| (length, input)))
        .collect::<Result<Vec<_>>>()?;
    ranked.sort_by_key(|(length, _)| *length);
    Ok(ranked.into_iter().map(|(_, input)| input).collect())
}

fn embed_batches(
    database: &mut Database,
    missing: &[EmbeddingInput],
    batch_size: usize,
    mut infer: impl FnMut(&[&str]) -> Result<Vec<Vec<f32>>>,
) -> Result<usize> {
    let mut embedded = 0;
    for batch in missing.chunks(batch_size.max(1)) {
        let documents: Vec<&str> = batch.iter().map(|item| item.text.as_str()).collect();
        let vectors = infer(&documents)?;
        ensure!(
            vectors.len() == batch.len(),
            "embedding model returned {} vectors for {} inputs",
            vectors.len(),
            batch.len()
        );
        let items: Vec<(String, Vec<f32>)> = batch
            .iter()
            .zip(vectors)
            .map(|(input, vector)| (input.content_hash.clone(), vector))
            .collect();
        database.store_embeddings(MODEL_ID, &items)?;
        embedded += items.len();
        eprintln!("embedded {embedded}/{}", missing.len());
    }
    Ok(embedded)
}

pub fn search(database: &Database, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    let mut model = load_model()?;
    let query = format!("query: {query}");
    let mut vectors = model
        .embed(vec![query], Some(1))
        .context("query embedding failed")?;
    let vector = vectors
        .pop()
        .context("embedding model returned no vector")?;
    database.semantic_search(MODEL_ID, &vector, limit)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendPreference {
    Auto,
    Cpu,
    CoreMl,
}

impl BackendPreference {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("auto") {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "coreml" => Ok(Self::CoreMl),
            value => anyhow::bail!("invalid MDQ_EMBED_BACKEND={value:?}; use auto, cpu, or coreml"),
        }
    }
}

struct EmbeddingRuntime {
    cpu: TextEmbedding,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    apple: Option<apple::AppleAccelerator>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    preference: BackendPreference,
}

impl EmbeddingRuntime {
    fn embed<S: AsRef<str>>(
        &mut self,
        documents: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let documents: Vec<&str> = documents.as_ref().iter().map(AsRef::as_ref).collect();
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(vectors) = try_acceleration(
            &mut self.apple,
            self.preference == BackendPreference::CoreMl,
            |apple| apple.embed(&mut self.cpu, &documents),
        )? {
            return Ok(vectors);
        }
        self.cpu.embed(&documents, batch_size)
    }
}

// Retry the very same batch on CPU after an accelerator failure, and disable
// acceleration for the remainder of this operation (no repeated failure loop).
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
fn try_acceleration<A, T>(
    accelerator: &mut Option<A>,
    strict: bool,
    infer: impl FnOnce(&mut A) -> Result<T>,
) -> Result<Option<T>> {
    let Some(engine) = accelerator.as_mut() else {
        return Ok(None);
    };
    match infer(engine) {
        Ok(vectors) => Ok(Some(vectors)),
        Err(error) if strict => Err(error.context("CoreML inference failed (forced backend)")),
        Err(error) => {
            eprintln!("CoreML unavailable or inference failed; using CPU: {error:#}");
            *accelerator = None;
            Ok(None)
        }
    }
}

#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
fn select_accelerator<A>(
    preference: BackendPreference,
    initialize: impl FnOnce() -> Result<A>,
) -> Result<Option<A>> {
    // Availability is not a speed test: CoreML initialization and shape
    // changes cost seconds even with compiled artifacts cached. End-to-end
    // measurements currently favor CPU for incremental jobs and queries.
    // Keep auto conservative until an amortized acceleration policy is proven;
    // crucially, skip even accelerator setup/model hashing on this path.
    if preference != BackendPreference::CoreMl {
        return Ok(None);
    }
    match initialize() {
        Ok(engine) => {
            eprintln!("embedding: trying Apple CoreML (GPU/Neural Engine)");
            Ok(Some(engine))
        }
        Err(error) if preference == BackendPreference::CoreMl => Err(error),
        Err(error) => {
            eprintln!("CoreML unavailable; using CPU: {error:#}");
            Ok(None)
        }
    }
}

fn load_model() -> Result<EmbeddingRuntime> {
    let value = std::env::var("MDQ_EMBED_BACKEND").ok();
    let preference = BackendPreference::parse(value.as_deref())?;
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    ensure!(
        preference != BackendPreference::CoreMl,
        "CoreML requires Apple Silicon macOS"
    );
    let cache_dir = dirs::cache_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
        .context("cannot determine model cache directory")?
        .join("mdq")
        .join("models");
    std::fs::create_dir_all(&cache_dir).context("cannot create model cache directory")?;
    let cpu = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::MultilingualE5Small)
            .with_cache_dir(cache_dir.clone())
            .with_show_download_progress(true)
            .with_intra_threads(4),
    )
    .context("failed to load multilingual-e5-small")?;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let apple = select_accelerator(preference, || apple::AppleAccelerator::new(cache_dir))?;
    Ok(EmbeddingRuntime {
        cpu,
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        apple,
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        preference,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(hash: &str, text: &str) -> EmbeddingInput {
        EmbeddingInput {
            content_hash: hash.to_owned(),
            text: text.to_owned(),
        }
    }

    #[test]
    fn cpu_override_never_initializes_an_accelerator() {
        assert!(
            select_accelerator::<()>(BackendPreference::Cpu, || panic!(
                "CPU override must bypass CoreML"
            ))
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn auto_never_initializes_coreml_for_indexing_or_queries() {
        // A performance regression guard: availability must not trigger a
        // potentially minute-long compilation on the default execution path.
        assert!(
            select_accelerator::<()>(BackendPreference::Auto, || panic!(
                "auto must avoid CoreML setup, hashing, compilation and shape changes"
            ))
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn explicit_coreml_still_initializes_the_accelerator() {
        assert_eq!(
            select_accelerator(BackendPreference::CoreMl, || Ok(42)).unwrap(),
            Some(42)
        );
    }

    #[test]
    fn forced_coreml_propagates_initialization_errors() {
        assert!(
            select_accelerator::<()>(BackendPreference::CoreMl, || anyhow::bail!("not available"))
                .is_err()
        );
    }

    #[test]
    fn backend_selection_defaults_to_auto_and_allows_cpu_override() {
        assert_eq!(
            BackendPreference::parse(None).unwrap(),
            BackendPreference::Auto
        );
        assert_eq!(
            BackendPreference::parse(Some("cpu")).unwrap(),
            BackendPreference::Cpu
        );
        assert_eq!(
            BackendPreference::parse(Some("coreml")).unwrap(),
            BackendPreference::CoreMl
        );
        assert!(BackendPreference::parse(Some("unknown")).is_err());
    }

    #[test]
    fn automatic_acceleration_failure_disables_engine_and_requests_cpu_retry() {
        let mut engine = Some(());
        let vectors: Option<Vec<f32>> =
            try_acceleration(&mut engine, false, |_| anyhow::bail!("device lost")).unwrap();
        assert!(vectors.is_none());
        assert!(engine.is_none());
        let next: Option<Vec<f32>> = try_acceleration(&mut engine, false, |_| {
            panic!("failed engine must not run again")
        })
        .unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn forced_acceleration_reports_failure_instead_of_hiding_cpu_fallback() {
        let mut engine = Some(());
        assert!(
            try_acceleration::<_, ()>(&mut engine, true, |_| anyhow::bail!("unsupported")).is_err()
        );
    }

    #[test]
    fn successful_acceleration_is_returned_without_cpu_retry() {
        let mut engine = Some(());
        assert_eq!(
            try_acceleration(&mut engine, false, |_| Ok(vec![1.0])).unwrap(),
            Some(vec![1.0])
        );
        assert!(engine.is_some());
    }

    #[test]
    fn length_sort_is_stable_and_keeps_content_hashes_attached() {
        let inputs = vec![
            input("long", "abcdef"),
            input("first", "ab"),
            input("short", "x"),
            input("second", "cd"),
        ];
        let sorted = order_by_token_length(inputs, |text| Ok(text.len())).unwrap();
        assert_eq!(
            sorted
                .iter()
                .map(|item| item.content_hash.as_str())
                .collect::<Vec<_>>(),
            ["short", "first", "second", "long"]
        );
    }

    #[test]
    fn tokenization_errors_are_not_silently_ignored() {
        assert!(
            order_by_token_length(vec![input("hash", "text")], |_| anyhow::bail!(
                "tokenizer failed"
            ))
            .is_err()
        );
    }

    #[test]
    fn sorted_batches_store_vectors_under_the_correct_hash() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("index.sqlite3");
        let mut database = Database::open(&path).unwrap();
        let sorted = order_by_token_length(
            vec![
                input("long", "abcdef"),
                input("short", "x"),
                input("medium", "ab"),
            ],
            |text| Ok(text.len()),
        )
        .unwrap();
        let mut sizes = Vec::new();
        let count = embed_batches(&mut database, &sorted, 2, |documents| {
            sizes.push(documents.len());
            Ok(documents
                .iter()
                .map(|text| vec![text.len() as f32])
                .collect())
        })
        .unwrap();
        assert_eq!(count, 3);
        assert_eq!(sizes, [2, 1]);
        let connection = rusqlite::Connection::open(&path).unwrap();
        for (hash, value) in [("long", 6.0_f32), ("short", 1.0), ("medium", 2.0)] {
            let blob: Vec<u8> = connection
                .query_row(
                    "SELECT vector FROM embeddings WHERE content_hash=?1 AND model=?2",
                    rusqlite::params![hash, MODEL_ID],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(blob, value.to_le_bytes());
        }
    }

    #[test]
    fn missing_vectors_fail_before_a_partial_batch_is_cached() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("index.sqlite3");
        let mut database = Database::open(&path).unwrap();
        let inputs = vec![input("a", "a"), input("b", "b")];
        assert!(embed_batches(&mut database, &inputs, 2, |_| Ok(vec![vec![1.0]])).is_err());
        let count: i64 = rusqlite::Connection::open(path)
            .unwrap()
            .query_row("SELECT count(*) FROM embeddings", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn failed_inference_keeps_completed_batches_for_resumption() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("index.sqlite3");
        let mut database = Database::open(&path).unwrap();
        let inputs = vec![input("a", "a"), input("b", "b")];
        let mut calls = 0;
        let result = embed_batches(&mut database, &inputs, 0, |_| {
            calls += 1;
            if calls == 2 {
                anyhow::bail!("inference failed");
            }
            Ok(vec![vec![1.0]])
        });
        assert!(result.is_err());
        assert_eq!(calls, 2);
        let count: i64 = rusqlite::Connection::open(path)
            .unwrap()
            .query_row(
                "SELECT count(*) FROM embeddings WHERE content_hash='a'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn empty_index_does_not_initialize_or_download_a_model() {
        let directory = tempfile::tempdir().unwrap();
        let mut database = Database::open(&directory.path().join("index.sqlite3")).unwrap();
        assert_eq!(embed_missing(&mut database, DEFAULT_BATCH_SIZE).unwrap(), 0);
    }
}
