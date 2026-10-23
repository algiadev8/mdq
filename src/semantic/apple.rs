//! Fixed-shape CoreML execution for Apple Silicon. The CPU model/tokenizer stays
//! available to the caller for fallback; model weights and post-processing match it.
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use fastembed::{EmbeddingModel, OutputKey, Pooling, SingleBatchOutput, TextEmbedding};
use ndarray::Array2;
use ort::ep::{CoreML, ExecutionProvider};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use sha2::{Digest, Sha256};

const MAX_BATCH: usize = 8;
const MAX_TOKENS: usize = 512;

pub(super) struct AppleAccelerator {
    model_path: PathBuf,
    cache_root: PathBuf,
    // Sorted indexing changes buckets monotonically. Keep just one session alive
    // instead of retaining several copies of this ~470 MB model in RAM.
    session: Option<(usize, usize, Session)>,
    validated: bool,
}

impl AppleAccelerator {
    pub(super) fn new(default_cache: PathBuf) -> Result<Self> {
        ensure!(
            CoreML::default().is_available()?,
            "CoreML execution provider is unavailable"
        );
        let cache = std::env::var_os("HF_HOME")
            .map(PathBuf::from)
            .unwrap_or(default_cache.clone());
        let info = TextEmbedding::get_model_info(&EmbeddingModel::MultilingualE5Small)?;
        // CPU initialization has already retrieved these exact weights. Never
        // download a different model or depend on a hard-coded HF revision here.
        let model_path = hf_hub::Cache::new(cache)
            .model(info.model_code.clone())
            .get(&info.model_file)
            .context("cached E5 ONNX model not found")?;
        let mut file = File::open(&model_path)?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        // CoreML does not reliably invalidate compiled artifacts on weights-only
        // changes. Include the entire weight hash, ORT version, settings and shape.
        let cache_root = default_cache
            .join("coreml")
            .join("ort-1.24.2-mlprogram-all-v1")
            .join(hex::encode(hash.finalize()));
        Ok(Self {
            model_path,
            cache_root,
            session: None,
            validated: false,
        })
    }

    fn session(&mut self, batch: usize, length: usize) -> Result<&mut Session> {
        if !self
            .session
            .as_ref()
            .is_some_and(|(b, l, _)| *b == batch && *l == length)
        {
            self.finish_session();
            self.validated = false;
            let cache = self.cache_root.join(format!("b{batch}-l{length}"));
            std::fs::create_dir_all(&cache)?;
            let provider = CoreML::default()
                .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                .with_static_input_shapes(true)
                .with_model_cache_dir(cache.to_string_lossy())
                .build()
                .error_on_failure();
            let mut builder = Session::builder()?
                .with_intra_threads(4)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .with_dimension_override("batch_size", batch as i64)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .with_dimension_override("sequence_length", length as i64)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .with_execution_providers([provider])
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                // Do not report CoreML success while ORT silently executes the
                // graph on its CPU EP. Our caller provides an explicit fallback.
                .with_config_entry("session.disable_cpu_ep_fallback", "1")
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            // Diagnostic opt-in: ORT's profile records the provider executing each
            // graph node. Useful to verify acceleration, not just EP registration.
            if let Some(profile) = std::env::var_os("MDQ_COREML_PROFILE") {
                builder = builder
                    .with_profiling(PathBuf::from(profile))
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            }
            let session = builder.commit_from_file(&self.model_path)?;
            self.session = Some((batch, length, session));
        }
        Ok(&mut self.session.as_mut().unwrap().2)
    }

    fn finish_session(&mut self) {
        if let Some((_, _, mut session)) = self.session.take() {
            if std::env::var_os("MDQ_COREML_PROFILE").is_some() {
                match session.end_profiling() {
                    Ok(path) => eprintln!("CoreML execution profile: {path}"),
                    Err(error) => eprintln!("could not finish CoreML profile: {error}"),
                }
            }
        }
    }

    pub(super) fn embed(
        &mut self,
        cpu: &mut TextEmbedding,
        documents: &[&str],
    ) -> Result<Vec<Vec<f32>>> {
        let mut result = Vec::with_capacity(documents.len());
        for documents in documents.chunks(MAX_BATCH) {
            let mut encodings = Vec::with_capacity(documents.len());
            for text in documents {
                encodings.push(
                    cpu.tokenizer
                        .encode(*text, true)
                        .map_err(|e| anyhow::anyhow!("CoreML tokenization failed: {e}"))?,
                );
            }
            let longest = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
            let length = token_bucket(longest)?;
            let batch = if documents.len() == 1 { 1 } else { MAX_BATCH };
            let padding = cpu
                .tokenizer
                .get_padding()
                .context("missing E5 tokenizer padding configuration")?;
            let mut ids = vec![i64::from(padding.pad_id); batch * length];
            let mut masks = vec![0_i64; batch * length];
            let mut types = vec![i64::from(padding.pad_type_id); batch * length];
            for row in 0..batch {
                // Duplicate a valid row for the tail, avoiding all-masked dummy
                // sequences. Each E5 sequence is independent of the other rows.
                let encoding = &encodings[row.min(encodings.len() - 1)];
                for (column, ((&id, &mask), &kind)) in encoding
                    .get_ids()
                    .iter()
                    .zip(encoding.get_attention_mask())
                    .zip(encoding.get_type_ids())
                    .enumerate()
                {
                    ids[row * length + column] = i64::from(id);
                    masks[row * length + column] = i64::from(mask);
                    types[row * length + column] = i64::from(kind);
                }
            }
            let session = self.session(batch, length)?;
            let outputs = session.run(ort::inputs![
                "input_ids" => Tensor::from_array(([batch, length], ids))?,
                "attention_mask" => Tensor::from_array(([batch, length], masks.clone()))?,
                "token_type_ids" => Tensor::from_array(([batch, length], types))?,
            ])?;
            let output = SingleBatchOutput {
                outputs: outputs
                    .into_iter()
                    .map(|(name, value)| (name.to_owned(), value))
                    .collect(),
                attention_mask_array: Array2::from_shape_vec((batch, length), masks)?,
            };
            // Reuse fastembed's mean pooling, then its exact L2 normalization.
            let pooled = output.select_and_pool_output(
                &&[OutputKey::ByName("last_hidden_state")][..],
                Some(Pooling::Mean),
            )?;
            ensure!(
                pooled.ncols() == 384 && pooled.nrows() == batch,
                "unexpected CoreML output shape"
            );
            for row in pooled.rows().into_iter().take(documents.len()) {
                let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                ensure!(norm.is_finite() && norm > 0.0, "invalid CoreML embedding");
                let vector: Vec<_> = row.iter().map(|v| v / (norm + 1e-12)).collect();
                ensure!(
                    vector.iter().all(|v| v.is_finite()),
                    "non-finite CoreML embedding"
                );
                result.push(vector);
            }
            if !self.validated {
                // A backend change must not poison the existing CPU embedding
                // cache. Check actual model outputs before persisting any vectors.
                let reference = cpu.embed(documents, Some(MAX_BATCH))?;
                let actual = &result[result.len() - documents.len()..];
                ensure!(
                    compatible(&reference, actual),
                    "CoreML vectors are incompatible with the CPU embedding cache"
                );
                self.validated = true;
                eprintln!("embedding backend: Apple CoreML (CPU compatibility verified)");
            }
        }
        Ok(result)
    }
}

impl Drop for AppleAccelerator {
    fn drop(&mut self) {
        self.finish_session();
    }
}

fn compatible(reference: &[Vec<f32>], actual: &[Vec<f32>]) -> bool {
    reference.len() == actual.len()
        && reference.iter().zip(actual).all(|(a, b)| {
            if a.len() != b.len() || a.is_empty() {
                return false;
            }
            let dot: f64 = a
                .iter()
                .zip(b)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (norm(a) * norm(b));
            cosine.is_finite() && cosine >= 0.9999
        })
}

fn token_bucket(length: usize) -> Result<usize> {
    ensure!(
        (1..=MAX_TOKENS).contains(&length),
        "invalid E5 token count: {length}"
    );
    Ok(length.max(32).next_power_of_two())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "manual CPU/CoreML speed matrix; requires Apple device/cache access"]
    fn coreml_speed_matrix() {
        use std::time::Instant;
        let cache = dirs::cache_dir().unwrap().join("mdq/models");
        let start = Instant::now();
        let mut cpu = TextEmbedding::try_new(
            fastembed::InitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_cache_dir(cache.clone())
                .with_intra_threads(4),
        )
        .unwrap();
        let cpu_load = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut apple = AppleAccelerator::new(cache).unwrap();
        let apple_setup = start.elapsed().as_secs_f64();
        let mut rows = Vec::new();
        for words in [24, 120, 600] {
            let documents: Vec<_> = (0..MAX_BATCH)
                .map(|i| format!("passage: {i} {}", "word ".repeat(words)))
                .collect();
            let texts: Vec<_> = documents.iter().map(String::as_str).collect();
            let tokens = cpu.tokenizer.encode(texts[0], true).unwrap().len();
            let reference = cpu.embed(&texts, Some(MAX_BATCH)).unwrap();
            let start = Instant::now();
            let actual = apple.embed(&mut cpu, &texts).unwrap();
            let first_call = start.elapsed().as_secs_f64();
            assert!(compatible(&reference, &actual));
            let mut min_cosine = 1.0_f64;
            let mut max_error = 0.0_f32;
            for (a, b) in reference.iter().zip(&actual) {
                let dot: f64 = a
                    .iter()
                    .zip(b)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
                min_cosine = min_cosine.min(dot / (norm(a) * norm(b)));
                for (&a, &b) in a.iter().zip(b) {
                    max_error = max_error.max((a - b).abs());
                }
            }
            let mut cpu_samples = Vec::new();
            let mut apple_samples = Vec::new();
            for trial in 0..3 {
                // Alternate order to reduce systematic temperature/order bias.
                for coreml in if trial % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    for _ in 0..5 {
                        if coreml {
                            apple.embed(&mut cpu, &texts).unwrap();
                        } else {
                            cpu.embed(&texts, Some(MAX_BATCH)).unwrap();
                        }
                    }
                    let elapsed = start.elapsed().as_secs_f64();
                    if coreml {
                        apple_samples.push(elapsed);
                    } else {
                        cpu_samples.push(elapsed);
                    }
                }
            }
            let median = |samples: &[f64]| {
                let mut sorted = samples.to_vec();
                sorted.sort_by(f64::total_cmp);
                sorted[sorted.len() / 2]
            };
            let cpu_seconds = median(&cpu_samples);
            let coreml_seconds = median(&apple_samples);
            let row = serde_json::json!({"tokens":tokens,"shape_tokens":token_bucket(tokens).unwrap(),"documents":40,"batch_size":MAX_BATCH,"trials":3,"cpu_seconds":cpu_seconds,"coreml_seconds":coreml_seconds,"speedup":cpu_seconds/coreml_seconds,"coreml_first_call_seconds":first_call,"cpu_samples":cpu_samples,"coreml_samples":apple_samples,"min_cosine":min_cosine,"max_error":max_error});
            println!("COREML_SPEED {}", row);
            rows.push(row);
        }
        let report = serde_json::json!({"cpu_model_load_seconds":cpu_load,"apple_setup_seconds":apple_setup,"rows":rows,"notes":"Release build; CPU 4 threads; median of 3 interleaved trials; each trial embeds 40 documents; steady-state excludes setup/first call. Existing disk caches are retained. CoreML ORT CPU fallback is disabled."});
        std::fs::create_dir_all("target").unwrap();
        std::fs::write(
            "target/coreml-speed-results.json",
            serde_json::to_string_pretty(&report).unwrap(),
        )
        .unwrap();
    }

    #[test]
    #[ignore = "requires a cached E5 model and Apple CoreML device/cache access"]
    fn coreml_real_model_queries_tails_truncation_and_warm_throughput() {
        use std::time::Instant;
        let cache = dirs::cache_dir().unwrap().join("mdq/models");
        let mut cpu = TextEmbedding::try_new(
            fastembed::InitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_cache_dir(cache.clone())
                .with_intra_threads(4),
        )
        .unwrap();
        let mut apple = AppleAccelerator::new(cache).unwrap();
        let cases = [
            vec!["query: 日本語の検索".to_owned()],
            (0..3)
                .map(|i| format!("passage: {i} {}", "word ".repeat(40)))
                .collect(),
            (0..3)
                .map(|i| format!("passage: {i} {}", "日本語の性能と検索 ".repeat(200)))
                .collect(),
        ];
        for documents in cases {
            let texts: Vec<_> = documents.iter().map(String::as_str).collect();
            let reference = cpu.embed(&texts, Some(MAX_BATCH)).unwrap();
            let actual = apple.embed(&mut cpu, &texts).unwrap();
            assert_eq!(actual.len(), texts.len());
            assert!(compatible(&reference, &actual));
        }
        let documents: Vec<_> = (0..MAX_BATCH)
            .map(|i| format!("passage: {i} {}", "日本語の性能と検索 ".repeat(200)))
            .collect();
        let texts: Vec<_> = documents.iter().map(String::as_str).collect();
        let reference = cpu.embed(&texts, Some(MAX_BATCH)).unwrap();
        let actual = apple.embed(&mut cpu, &texts).unwrap();
        assert!(compatible(&reference, &actual));
        let start = Instant::now();
        for _ in 0..5 {
            cpu.embed(&texts, Some(MAX_BATCH)).unwrap();
        }
        let cpu_time = start.elapsed();
        let start = Instant::now();
        for _ in 0..5 {
            apple.embed(&mut cpu, &texts).unwrap();
        }
        let apple_time = start.elapsed();
        println!(
            "steady-state (40 x 512-token documents): CPU={:.3}s CoreML={:.3}s speedup={:.2}x",
            cpu_time.as_secs_f64(),
            apple_time.as_secs_f64(),
            cpu_time.as_secs_f64() / apple_time.as_secs_f64()
        );
    }

    #[test]
    fn compatibility_guard_rejects_invalid_or_different_vectors() {
        assert!(compatible(&[vec![1.0, 0.0]], &[vec![1.0, 0.0]]));
        assert!(!compatible(&[vec![1.0, 0.0]], &[vec![-1.0, 0.0]]));
        assert!(!compatible(&[vec![1.0, 0.0]], &[vec![f32::NAN, 0.0]]));
        assert!(!compatible(&[vec![1.0, 0.0]], &[vec![0.0, 0.0]]));
        assert!(!compatible(&[vec![1.0, 0.0]], &[vec![1.0]]));
        assert!(!compatible(&[vec![1.0, 0.0]], &[]));
    }

    #[test]
    fn buckets_never_truncate_valid_input() {
        for length in 1..=512 {
            let bucket = token_bucket(length).unwrap();
            assert!(bucket >= length && bucket <= 512 && bucket.is_power_of_two());
        }
        assert!(token_bucket(0).is_err());
        assert!(token_bucket(513).is_err());
    }
}
