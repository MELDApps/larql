//! `MarkovResidualCodecEngine` — `KvEngine` implementation.

use larql_inference::ffn::FfnBackend;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::Array2;

use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
use crate::engines::markov_residual_codec::compute::{rs_decode_step_codec, rs_prefill_codec};
use crate::engines::markov_residual_codec::store::RsStoreCodec;
use crate::engines::markov_residual_codec::walk::{
    rs_decode_step_codec_walk, rs_prefill_codec_walk,
};
use crate::{EngineInfo, KvEngine};

/// `MarkovResidualCodecEngine` — `MarkovResidualEngine` with a codec-encoded
/// cold tier.
pub struct MarkovResidualCodecEngine {
    window_size: Option<usize>,
    codec: ColdResidualCodec,
    store: Option<RsStoreCodec>,
    backend: Box<dyn EngineBackend>,
    /// `true` once `prefill_quant` has taken the Metal fast path (which
    /// bypasses the residual store entirely). Subsequent `decode_step_quant`
    /// calls route through `fused_decode_step`. Matches `MarkovResidualEngine`.
    metal_prefill_done: bool,
}

impl MarkovResidualCodecEngine {
    /// Construct with the default CPU backend.
    pub fn new(window_size: Option<usize>, codec: ColdResidualCodec) -> Self {
        Self::with_backend(window_size, codec, cpu_engine_backend())
    }

    /// Construct with an explicit compute backend.
    pub fn with_backend(
        window_size: Option<usize>,
        codec: ColdResidualCodec,
        backend: Box<dyn EngineBackend>,
    ) -> Self {
        Self {
            window_size,
            codec,
            store: None,
            backend,
            metal_prefill_done: false,
        }
    }

    pub fn codec(&self) -> ColdResidualCodec {
        self.codec
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
    }
}

impl KvEngine for MarkovResidualCodecEngine {
    fn name(&self) -> &str {
        "markov-rs-codec"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w},codec={}", self.codec.label()),
            None => format!("window=full,codec={}", self.codec.label()),
        };
        let mem = self.store.as_ref().map_or(0, |s| s.memory_bytes());
        EngineInfo {
            name: "markov-rs-codec".into(),
            description: format!(
                "residual-stream KV replacement with {} cold codec (mem={:.1}MB)",
                self.codec.label(),
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config,
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let result = rs_prefill_codec(
            weights,
            token_ids,
            self.window_size,
            self.codec,
            self.backend.as_ref(),
        );
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let rs = self.store.take()?;
        let (hidden, new_rs) = rs_decode_step_codec(weights, token_id, rs, self.backend.as_ref())?;
        self.store = Some(new_rs);
        Some(hidden)
    }

    fn memory_bytes(&self) -> usize {
        self.total_memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }

    fn prefill_quant(
        &mut self,
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        // Same routing as `MarkovResidualEngine::prefill_quant`:
        //
        //   1. Try the Metal fused fast path. It bypasses the residual
        //      store entirely (no codec applied), so when this path runs
        //      the engine is effectively `Standard`-on-Metal — the codec
        //      only matters when overflow forces residual recompute.
        //   2. Otherwise dequant attention tensors and route through
        //      `rs_prefill_codec_walk` (Q4K-aware FFN via WalkFfn +
        //      Q4K-native K/V via `recompute_kv(Some(index))`).
        use crate::engines::unlimited_context::engine::fused_prefill;
        if let Some(h) = fused_prefill(weights, index, token_ids, backend) {
            self.metal_prefill_done = true;
            self.store = None;
            return Some(h);
        }
        self.metal_prefill_done = false;
        ensure_attn_tensors_dequantised(weights, index);
        let result = rs_prefill_codec_walk(
            weights,
            index,
            token_ids,
            self.window_size,
            self.codec,
            backend,
        );
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step_quant(
        &mut self,
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        use crate::engines::unlimited_context::engine::fused_decode_step;
        if self.metal_prefill_done {
            if let Some(h) = fused_decode_step(weights, index, token_id, backend) {
                return Some(h);
            }
        }
        ensure_attn_tensors_dequantised(weights, index);
        let rs = self.store.take()?;
        let (hidden, new_rs) = rs_decode_step_codec_walk(weights, index, token_id, rs, backend)?;
        self.store = Some(new_rs);
        Some(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::markov_residual::MarkovResidualEngine;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn engine_name_is_markov_rs_codec() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert_eq!(eng.name(), "markov-rs-codec");
    }

    #[test]
    fn engine_info_reports_codec_and_window() {
        let eng = MarkovResidualCodecEngine::new(Some(128), ColdResidualCodec::Bf16);
        let info = eng.info();
        assert!(info.config.contains("window=128"));
        assert!(info.config.contains("codec=bf16"));
        assert!(info.description.contains("bf16"));
    }

    #[test]
    fn engine_info_unbounded_window() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let info = eng.info();
        assert!(info.config.contains("window=full"));
    }

    #[test]
    fn engine_memory_zero_before_prefill() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn codec_accessor_returns_configured_codec() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert_eq!(eng.codec(), ColdResidualCodec::Bf16);
    }

    // ── Prefill / decode ──────────────────────────────────────────────────────

    #[test]
    fn prefill_populates_store_and_returns_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = eng.prefill(&weights, &ffn, &[0u32, 1, 2]).expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(eng.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_produces_finite_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = eng.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_step_without_prefill_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert!(eng.decode_step(&weights, &ffn, 0).is_none());
    }

    #[test]
    fn multiple_decode_steps_produce_consistent_shapes() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        for step in 0..3 {
            let h = eng
                .decode_step(&weights, &ffn, step as u32)
                .expect("decode");
            assert_eq!(h.shape(), &[1, weights.hidden_size], "step {step}");
        }
    }

    // ── Cold tier ─────────────────────────────────────────────────────────────

    #[test]
    fn windowed_prefill_creates_codec_encoded_cold_tier() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        assert!(eng.window_tokens() <= 2);
        assert!(
            eng.cold_bytes() > 0,
            "cold tier should be non-empty after overflow"
        );
    }

    #[test]
    fn encoded_cold_payload_is_half_of_f32_equivalent() {
        // Memory contract: bf16 cold payload is exactly 50% the size of an
        // f32 residual tier for the same positions. cold_bytes also bundles
        // cold_kv (which is K/V tensors, not residuals) — we measure the
        // payload directly via the store.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(Some(1), ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3, 4])
            .expect("prefill 5 tokens");
        let store = eng.store.as_ref().expect("store populated after prefill");
        let n_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let cold_positions = 4; // 5 tokens, window=1
        let f32_equivalent_payload = cold_positions * n_layers * hidden * 4;
        let payload: usize = store
            .cold_encoded
            .as_ref()
            .map(|layers| layers.iter().map(|l| l.payload.len()).sum())
            .unwrap_or(0);
        let expected_bf16_payload = cold_positions * n_layers * hidden * 2;
        assert_eq!(
            payload, expected_bf16_payload,
            "bf16 payload should be exactly 2 bytes per element × {cold_positions} × {n_layers} × {hidden}"
        );
        assert_eq!(
            payload * 2,
            f32_equivalent_payload,
            "bf16 cold payload should be exactly half of f32-equivalent"
        );
    }

    #[test]
    fn memory_grows_with_each_decode_step() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let m0 = eng.memory_bytes();
        eng.decode_step(&weights, &ffn, 1).expect("decode 1");
        let m1 = eng.memory_bytes();
        eng.decode_step(&weights, &ffn, 2).expect("decode 2");
        let m2 = eng.memory_bytes();
        assert!(m1 > m0);
        assert!(m2 > m1);
    }

    // ── Bf16 codec contract: bounded KL vs MarkovResidualEngine ───────────────

    #[test]
    fn bf16_output_is_close_to_markov_residual_baseline() {
        // The contract is "bounded KL", not bit-identity. Bf16 introduces
        // round-off on cold residuals; with the test fixture this stays
        // within a small per-element bound.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut baseline = MarkovResidualEngine::new(Some(2));
        let mut codec_eng = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        baseline
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("baseline prefill");
        codec_eng
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("codec prefill");
        let h_b = baseline.decode_step(&weights, &ffn, 4).expect("baseline");
        let h_c = codec_eng.decode_step(&weights, &ffn, 4).expect("codec");
        assert_eq!(h_b.shape(), h_c.shape());
        // Bf16 cold tier should leave the live forward pass within bf16
        // precision on average.
        let max_abs: f32 = h_b
            .iter()
            .zip(h_c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_baseline_abs: f32 = h_b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // 5% relative + small absolute tolerance is generous for a test
        // fixture; production calibration would tighten this.
        assert!(
            max_abs < max_baseline_abs * 0.05 + 1e-2,
            "max_abs={max_abs} exceeded tolerance (baseline max_abs={max_baseline_abs})"
        );
    }

    #[test]
    fn unbounded_codec_matches_markov_residual_when_no_overflow() {
        // With window=None and prompt small enough to never overflow, the
        // cold codec is never applied. Output should match
        // MarkovResidualEngine bit-for-bit.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut baseline = MarkovResidualEngine::new(None);
        let mut codec_eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        baseline
            .prefill(&weights, &ffn, &[0u32, 1])
            .expect("baseline");
        codec_eng
            .prefill(&weights, &ffn, &[0u32, 1])
            .expect("codec");
        let h_b = baseline.decode_step(&weights, &ffn, 2).expect("baseline");
        let h_c = codec_eng.decode_step(&weights, &ffn, 2).expect("codec");
        assert_eq!(h_b, h_c);
    }
}
