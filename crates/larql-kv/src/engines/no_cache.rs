//! NoCacheEngine — debug fallback that maintains no K/V cache.
//!
//! Every `decode_step` re-runs a full forward pass over the growing
//! prompt+generated sequence. O(N²) wall-clock; correctness baseline
//! against which other engines' bit-parity claims are measured.
//!
//! Wraps `kv_prefill_run` (discarding the cache each call) so the
//! forward-pass code is shared with `StandardEngine`.

use larql_compute::{cpu_backend, ComputeBackend};
use ndarray::Array2;

use crate::{EngineInfo, KvEngine};
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::hooks::NoopHook;
use larql_inference::forward::kv_prefill_run;
use larql_inference::model::ModelWeights;

/// No-cache decode. Stores only the running token sequence and re-runs
/// a full forward pass per step.
pub struct NoCacheEngine {
    tokens: Vec<u32>,
    backend: Box<dyn ComputeBackend>,
}

impl NoCacheEngine {
    pub fn new() -> Self {
        Self::with_backend(cpu_backend())
    }

    pub fn with_backend(backend: Box<dyn ComputeBackend>) -> Self {
        Self {
            tokens: Vec::new(),
            backend,
        }
    }
}

impl Default for NoCacheEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl KvEngine for NoCacheEngine {
    fn name(&self) -> &str {
        "no-cache"
    }

    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "no-cache".into(),
            description:
                "no K/V cache — full re-forward per step (O(N²)); correctness fallback only".into(),
            backend: self.backend.name().to_string(),
            config: String::new(),
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        self.tokens = token_ids.to_vec();
        let (hidden, _cache) = kv_prefill_run(
            weights,
            ffn,
            token_ids,
            None,
            Some(self.backend.as_ref()),
            &mut NoopHook,
        )?;
        Some(hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        self.tokens.push(token_id);
        let (hidden, _cache) = kv_prefill_run(
            weights,
            ffn,
            &self.tokens,
            None,
            Some(self.backend.as_ref()),
            &mut NoopHook,
        )?;
        Some(hidden)
    }

    fn memory_bytes(&self) -> usize {
        // Only persistent state is the token-id list.
        self.tokens.len() * std::mem::size_of::<u32>()
    }

    fn window_tokens(&self) -> usize {
        self.tokens.len()
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
        assert_eq!(NoCacheEngine::new().name(), "no-cache");
    }

    #[test]
    fn memory_zero_before_prefill() {
        let eng = NoCacheEngine::new();
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
    }

    #[test]
    fn prefill_returns_hidden_state() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = NoCacheEngine::new();
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.window_tokens(), 3);
    }

    #[test]
    fn decode_step_appends_and_returns_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = NoCacheEngine::new();
        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = engine.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.window_tokens(), 3);
        assert!(hidden_to_raw_logits(&weights, &h)
            .iter()
            .all(|v| v.is_finite()));
    }

    // ── Step 4 parity gate (NoCache) ─────────────────────────────────────
    //
    // Today's `--kv-cache none` does `predict_with_ffn(weights, tokenizer,
    // &ids_so_far, 1, ffn)` per step. `NoCacheEngine` driven through
    // `generate_with_engine` must produce the same token stream on
    // synthetic weights (the test substrate covers the production
    // non-PLE arch; PLE archs are checked at integration time).

    use larql_inference::forward::{generate_with_engine, predict_with_ffn};

    fn run_legacy_no_cache(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
    ) -> Vec<u32> {
        let mut ids = prompt.to_vec();
        let mut generated = Vec::with_capacity(max);
        for _ in 0..max {
            let result = predict_with_ffn(weights, tokenizer, &ids, 1, ffn);
            let Some(&next_id) = result.token_ids.first() else {
                break;
            };
            ids.push(next_id);
            generated.push(next_id);
        }
        generated
    }

    fn run_engine_no_cache(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
    ) -> Vec<u32> {
        let mut engine = NoCacheEngine::new();
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
    fn parity_no_cache_matches_legacy_predict_with_ffn() {
        use larql_inference::test_utils::make_test_tokenizer;
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[2u32, 3, 5];
        let max = 4;
        let legacy = run_legacy_no_cache(&weights, &tokenizer, &ffn, prompt, max);
        let engine = run_engine_no_cache(&weights, &tokenizer, &ffn, prompt, max);
        assert_eq!(
            engine, legacy,
            "NoCache engine dispatch must produce identical tokens to legacy predict_with_ffn loop"
        );
    }

    #[test]
    fn memory_grows_linearly_with_tokens() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = NoCacheEngine::new();
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let mem1 = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 1).expect("decode 1");
        let mem2 = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 2).expect("decode 2");
        let mem3 = engine.memory_bytes();
        assert!(mem2 > mem1);
        assert!(mem3 > mem2);
    }
}
