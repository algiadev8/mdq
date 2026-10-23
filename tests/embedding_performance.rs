//! Uses the local model cache; run manually with:
//! cargo test --release --test embedding_performance -- --ignored --nocapture
use std::time::Instant;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

fn documents() -> Vec<String> {
    let repeats = [1, 2, 3, 4, 6, 8, 12, 48];
    (0..64)
        .map(|i| {
            format!(
                "passage: Note {i}\nSection\n{}",
                "Rust indexing and retrieval. 日本語の埋め込み計算を高速化する。 "
                    .repeat(repeats[i % repeats.len()])
            )
        })
        .collect()
}

#[test]
#[ignore = "manual end-to-end benchmark requiring a cached model"]
fn benchmark_embedding_pipeline() {
    use mdq::{db::Database, semantic};
    let directory = tempfile::tempdir().unwrap();
    let vault = directory.path().join("vault");
    std::fs::create_dir(&vault).unwrap();
    for (i, text) in documents().iter().enumerate() {
        std::fs::write(
            vault.join(format!("note-{i}.md")),
            format!("# Section\n{text}"),
        )
        .unwrap();
    }
    let path = directory.path().join("index.sqlite3");
    let mut database = Database::open(&path).unwrap();
    database.rebuild(&vault).unwrap();
    let inputs = database.missing_embeddings(semantic::MODEL_ID).unwrap();
    let mut baseline_database = Database::open(&directory.path().join("baseline.sqlite3")).unwrap();
    let start = Instant::now();
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::MultilingualE5Small)
            .with_cache_dir(dirs::cache_dir().unwrap().join("mdq/models"))
            .with_show_download_progress(false)
            .with_intra_threads(4),
    )
    .unwrap();
    let texts: Vec<&str> = inputs.iter().map(|input| input.text.as_str()).collect();
    let baseline = model.embed(texts, Some(64)).unwrap();
    let items: Vec<_> = inputs
        .iter()
        .zip(&baseline)
        .map(|(input, vector)| (input.content_hash.clone(), vector.clone()))
        .collect();
    baseline_database
        .store_embeddings(semantic::MODEL_ID, &items)
        .unwrap();
    let original = start.elapsed();
    drop(model);
    let start = Instant::now();
    let count = semantic::embed_missing(&mut database, semantic::DEFAULT_BATCH_SIZE).unwrap();
    let optimized = start.elapsed();
    assert_eq!(count, inputs.len());
    let connection = rusqlite::Connection::open(&path).unwrap();
    let mut max_error = 0.0_f32;
    for (input, reference) in inputs.iter().zip(&baseline) {
        let blob: Vec<u8> = connection
            .query_row(
                "SELECT vector FROM embeddings WHERE content_hash=?1 AND model=?2",
                rusqlite::params![input.content_hash, semantic::MODEL_ID],
                |row| row.get(0),
            )
            .unwrap();
        let vector: Vec<f32> = blob
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(vector.len(), reference.len());
        for (&a, &b) in vector.iter().zip(reference) {
            max_error = max_error.max((a - b).abs());
        }
    }
    assert!(max_error < 0.00001, "max vector difference {max_error}");
    let start = Instant::now();
    assert_eq!(
        semantic::embed_missing(&mut database, semantic::DEFAULT_BATCH_SIZE).unwrap(),
        0
    );
    println!(
        "pipeline: original={:.3}s optimized={:.3}s speedup={:.2}x max_error={max_error:.9}; cached={:.3}ms",
        original.as_secs_f64(),
        optimized.as_secs_f64(),
        original.as_secs_f64() / optimized.as_secs_f64(),
        start.elapsed().as_secs_f64() * 1000.0
    );
}

#[test]
#[ignore = "manual benchmark requiring a cached multilingual-e5-small model"]
fn benchmark_embedding_inference() {
    let cache_dir = dirs::cache_dir().unwrap().join("mdq/models");
    let documents = documents();
    let threads = std::env::var("MDQ_BENCH_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let start = Instant::now();
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::MultilingualE5Small)
            .with_cache_dir(cache_dir)
            .with_show_download_progress(false)
            .with_intra_threads(threads),
    )
    .unwrap();
    println!(
        "model load: {:.3}s, threads={threads}",
        start.elapsed().as_secs_f64()
    );
    // Warm up graph/memory allocation before measuring inference.
    model.embed(["query: warmup"], Some(1)).unwrap();
    let mut order: Vec<usize> = (0..documents.len()).collect();
    let mut lengths = Vec::new();
    for text in &documents {
        lengths.push(model.tokenizer.encode(text.as_str(), true).unwrap().len());
    }
    println!("effective token lengths: {:?}", &lengths[..8]);
    let start = Instant::now();
    let baseline = model.embed(&documents, Some(64)).unwrap();
    println!("original batch=64: {:.3}s", start.elapsed().as_secs_f64());
    order.sort_by_key(|&i| lengths[i]);
    for batch_size in [1, 2, 4, 8, 16] {
        let sorted: Vec<&str> = order.iter().map(|&i| documents[i].as_str()).collect();
        let start = Instant::now();
        let optimized = model.embed(&sorted, Some(batch_size)).unwrap();
        let elapsed = start.elapsed();
        let mut min_cosine = 1.0_f64;
        let mut max_error = 0.0_f32;
        for (vector, &i) in optimized.iter().zip(&order) {
            assert_eq!(vector.len(), 384);
            let reference = &baseline[i];
            let dot: f64 = vector
                .iter()
                .zip(reference)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            min_cosine = min_cosine.min(dot / (norm(vector) * norm(reference)));
            for (&a, &b) in vector.iter().zip(reference) {
                max_error = max_error.max((a - b).abs());
            }
        }
        println!(
            "sorted batch={batch_size}: {:.3}s; min_cosine={min_cosine:.9}; max_error={max_error:.9}",
            elapsed.as_secs_f64()
        );
        assert!(min_cosine > 0.99999);
    }
}
