//! StandardEngine — the production K/V cache, wrapped as a `KvEngine`.
//!
//! This is the live decode path: a per-layer `KvCache` of `(K_rope, V)`
//! tensors with optional sliding-window eviction. It is *the same code*
//! that `larql_inference::forward::generate_cached_bounded` runs today —
//! `prefill` and `decode_step` delegate to `kv_prefill_run` /
//! `kv_decode_step_run` in `larql-inference`.
//!
//! Output is bit-identical to today's `--kv-cache standard` (with
//! `window_size: None`) and `--kv-cache markov-bounded`
//! (with `window_size: Some(N)`).

use larql_compute::{cpu_backend, ComputeBackend};
use ndarray::Array2;

use crate::{EngineInfo, KvEngine};
use larql_inference::attention::KvCache;
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::hooks::NoopHook;
use larql_inference::forward::{kv_decode_step_run, kv_prefill_run};
use larql_inference::model::ModelWeights;

/// Production K/V cache engine. `window_size: None` = unbounded growth
/// (the `--kv-cache standard` flag); `Some(N)` = sliding window (the
/// `--kv-cache markov-bounded --context-window N` flag combo).
pub struct StandardEngine {
    window_size: Option<usize>,
    cache: Option<KvCache>,
    backend: Box<dyn ComputeBackend>,
}

impl StandardEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_backend())
    }

    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn ComputeBackend>) -> Self {
        Self {
            window_size,
            cache: None,
            backend,
        }
    }

    fn cache_memory_bytes(&self) -> usize {
        let Some(cache) = self.cache.as_ref() else {
            return 0;
        };
        cache
            .layers
            .iter()
            .filter_map(|opt| opt.as_ref())
            .map(|(k, v)| (k.len() + v.len()) * std::mem::size_of::<f32>())
            .sum()
    }
}

impl KvEngine for StandardEngine {
    fn name(&self) -> &str {
        "standard"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w}"),
            None => "window=full".into(),
        };
        let mem = self.cache_memory_bytes();
        EngineInfo {
            name: "standard".into(),
            description: format!(
                "production K/V tensor cache — full FP32 K/V per layer (mem={:.1}MB)",
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config,
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let (hidden, cache) = kv_prefill_run(
            weights,
            ffn,
            token_ids,
            self.window_size,
            Some(self.backend.as_ref()),
            &mut NoopHook,
        )?;
        self.cache = Some(cache);
        Some(hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let cache = self.cache.as_mut()?;
        kv_decode_step_run(
            weights,
            ffn,
            cache,
            token_id,
            Some(self.backend.as_ref()),
            &mut NoopHook,
        )
    }

    fn memory_bytes(&self) -> usize {
        self.cache_memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.cache
            .as_ref()
            .and_then(|c| c.layers.iter().filter_map(|l| l.as_ref()).next())
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0)
    }

    fn cold_bytes(&self) -> usize {
        // Standard cache does not have a separate cold tier — the K/V
        // tensors are the state. Sliding-window evictions drop data
        // entirely; nothing is moved to cold.
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn engine_name() {
        assert_eq!(StandardEngine::new(None).name(), "standard");
    }

    #[test]
    fn engine_info_unbounded() {
        let info = StandardEngine::new(None).info();
        assert!(info.config.contains("full"));
    }

    #[test]
    fn engine_info_windowed() {
        let info = StandardEngine::new(Some(128)).info();
        assert!(info.config.contains("128"));
    }

    #[test]
    fn memory_zero_before_prefill() {
        let eng = StandardEngine::new(None);
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn prefill_populates_cache_and_returns_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0, "cache should be populated");
        assert!(engine.window_tokens() >= 3);
    }

    #[test]
    fn decode_step_produces_finite_logits() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = engine.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(hidden_to_raw_logits(&weights, &h)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn cache_grows_with_decode_steps() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let after_prefill = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 1).expect("decode 1");
        let after_one = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 2).expect("decode 2");
        let after_two = engine.memory_bytes();
        assert!(after_one > after_prefill);
        assert!(after_two > after_one);
    }

    #[test]
    fn sliding_window_clips_cache() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let window = 2usize;
        let mut engine = StandardEngine::new(Some(window));
        // Prefill with 4 tokens — cache should clip to last `window` per layer.
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        assert!(
            engine.window_tokens() <= window,
            "expected window_tokens ≤ {window}, got {}",
            engine.window_tokens()
        );
    }

    #[test]
    fn decode_step_without_prefill_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        assert!(engine.decode_step(&weights, &ffn, 0).is_none());
    }

    // ── Step 4 parity gate ─────────────────────────────────────────────────
    //
    // `StandardEngine` is the engine-trait wrapper over the production K/V
    // cache. Driven through `generate_with_engine`, its token output must
    // be bit-identical to `generate_cached_backend` on the same inputs.
    // This is the unification's bit-parity gate (spec §8.4); failure here
    // blocks Step 5 (default flip).

    use larql_inference::forward::{generate_cached_backend, generate_with_engine};
    use larql_inference::test_utils::make_test_tokenizer;

    fn run_legacy(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
        window: Option<usize>,
    ) -> Vec<u32> {
        generate_cached_backend(
            weights,
            tokenizer,
            ffn,
            prompt,
            max,
            None,
            window,
            |_, _| {},
        )
    }

    fn run_engine(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
        window: Option<usize>,
    ) -> Vec<u32> {
        let mut engine = StandardEngine::new(window);
        generate_with_engine(
            &mut engine as &mut dyn crate::KvEngine,
            weights,
            tokenizer,
            ffn,
            prompt,
            max,
            |_, _| {},
        )
    }

    #[test]
    fn parity_standard_unbounded_matches_legacy() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[2u32, 3, 5, 7];
        let max = 6;
        let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, None);
        let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, None);
        assert_eq!(
            engine, legacy,
            "engine dispatch must produce identical tokens to generate_cached_backend (window=None)"
        );
    }

    #[test]
    fn parity_standard_windowed_matches_legacy() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[1u32, 2, 3, 4, 5];
        let max = 5;
        // Window smaller than prompt → exercises prefill-time clipping.
        let window = Some(3);
        let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, window);
        let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
        assert_eq!(
            engine, legacy,
            "engine dispatch must produce identical tokens to generate_cached_backend (sliding window)"
        );
    }

    #[test]
    fn parity_standard_short_prompt_long_window_matches_legacy() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[0u32, 1];
        let max = 4;
        let window = Some(64); // window > prompt — exercises decode-time growth past prompt
        let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, window);
        let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
        assert_eq!(
            engine, legacy,
            "engine dispatch must produce identical tokens at short-prompt long-window edge case"
        );
    }
}
