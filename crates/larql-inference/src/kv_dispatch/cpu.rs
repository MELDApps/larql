//! `KvDispatch` implementation for `larql_compute::CpuBackend`.
//!
//! Lives here (not in `larql-compute`) so the bodies can call into the
//! inference-side forward-pass functions (`run_attention_*`, `run_ffn`,
//! `forward_from_layer`). Orphan rules: the [`KvDispatch`] trait is
//! local to this crate, so implementing it for a foreign type
//! (`CpuBackend`) is allowed.
//!
//! See `docs/specs/compute-backend-redesign.md` §10.2 for the trait-
//! location rationale.
//!
//! ## Implementation strategy
//!
//! - `KvHandle` wraps **a single layer's** K and V tensors. Engines
//!   that need multi-layer caches hold a `Vec<KvHandle>` (one per
//!   layer). This matches the trait's per-layer API
//!   (`alloc_kv_buffer(layer, ...)`).
//! - `ResidualHandle` is a thin wrap around `Array2<f32>` — CPU has no
//!   device memory to manage.
//! - `attention_step` / `attention_prefill` delegate to the existing
//!   `run_attention_*` functions.
//! - `forward_from_layer` delegates to
//!   `crate::forward::forward_from_layer`.
//! - Engine-specific intents (`recompute_kv_from_residuals`,
//!   `compressed_kv_append`) stay at the trait defaults until Step 3
//!   migrates the engines that need them.

use larql_compute::CpuBackend;
use ndarray::Array2;

use super::{KvDispatch, KvHandle, KvHandleInner, ResidualHandle, ResidualHandleInner};
use crate::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend, SharedKV,
};
use crate::model::ModelWeights;

// ─── CpuKvHandle ────────────────────────────────────────────────────────────

/// Single-layer K/V cache held in host memory. Wraps the existing
/// `SharedKV = (K, V)` shape — `K` and `V` are owned `Array2<f32>`
/// growing by one row per `append_kv` call.
pub struct CpuKvHandle {
    /// Layer index this handle was minted for. Carried for debugging
    /// / future trait surface; not consulted by the current append /
    /// attend paths (the trait already takes `layer` per call).
    #[allow(dead_code)]
    layer: usize,
    kv_dim: usize,
    /// `None` before the first `append_kv` / `attention_prefill`.
    state: Option<SharedKV>,
}

impl CpuKvHandle {
    fn new(layer: usize, kv_dim: usize) -> Self {
        Self {
            layer,
            kv_dim,
            state: None,
        }
    }

    /// Replace the internal state — used by backend impls that
    /// populate the handle from the prefill path (which returns a
    /// fresh `SharedKV` rather than appending incrementally).
    fn replace_state(&mut self, kv: SharedKV) {
        self.state = Some(kv);
    }

    fn as_shared_kv(&self) -> Option<&SharedKV> {
        self.state.as_ref()
    }
}

impl KvHandleInner for CpuKvHandle {
    fn cached_len(&self) -> usize {
        self.state.as_ref().map_or(0, |(k, _)| k.shape()[0])
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Downcast helper — backend implementations use this to retrieve the
/// concrete handle type from an opaque `KvHandle`. Panics if the
/// handle was allocated by a different backend.
fn cpu_handle(h: &KvHandle) -> &CpuKvHandle {
    h.as_inner()
        .as_any()
        .downcast_ref::<CpuKvHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign handle (backend={}); \
                 handles must be allocated by the same backend that consumes them",
                h.backend_name()
            )
        })
}

fn cpu_handle_mut(h: &mut KvHandle) -> &mut CpuKvHandle {
    let name = h.backend_name();
    h.as_inner_mut()
        .as_any_mut()
        .downcast_mut::<CpuKvHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign handle (backend={name}); \
                 handles must be allocated by the same backend that consumes them"
            )
        })
}

// ─── CpuResidualHandle ──────────────────────────────────────────────────────

/// Host-resident residual upload. CPU has no device memory to manage,
/// so this is just a flat `Vec<f32>` wrapper. Storing flat matches
/// what `forward_from_layer` consumes (`&[f32]` interpreted as
/// `[seq_len, hidden]` row-major).
pub struct CpuResidualHandle {
    flat: Vec<f32>,
    shape: (usize, usize),
}

impl ResidualHandleInner for CpuResidualHandle {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn cpu_residual(r: &ResidualHandle) -> &CpuResidualHandle {
    r.as_inner()
        .as_any()
        .downcast_ref::<CpuResidualHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign residual handle (backend={}); \
                 handles must be allocated by the same backend that consumes them",
                r.backend_name()
            )
        })
}

// ─── CpuQ4kCacheHandle — Q4K cached-decode handle ──────────────────────────
//
// Wraps the production `CpuKvCache` (per-layer K/V) so it can flow through
// the dispatch trait's `KvHandle` shape. Cache populated by
// `cached_prefill_q4k`; consumed by `cached_decode_step_q4k`.
//
// One handle per engine (not per layer), unlike the legacy `CpuKvHandle`
// (one per layer for the f32 per-layer dispatch path). The two shapes
// coexist because they serve different dispatch granularities.

pub struct CpuQ4kCacheHandle {
    cache: crate::vindex::CpuKvCache,
}

impl KvHandleInner for CpuQ4kCacheHandle {
    fn cached_len(&self) -> usize {
        self.cache
            .iter()
            .filter_map(|o| o.as_ref())
            .map(|(k, _)| k.shape()[0])
            .next()
            .unwrap_or(0)
    }

    fn kv_dim(&self) -> usize {
        self.cache
            .iter()
            .filter_map(|o| o.as_ref())
            .map(|(k, _)| k.shape()[1])
            .next()
            .unwrap_or(0)
    }

    fn backend_name(&self) -> &'static str {
        "cpu-q4k"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn cpu_q4k_cache_mut(h: &mut KvHandle) -> &mut CpuQ4kCacheHandle {
    let backend_name = h.backend_name();
    h.as_inner_mut()
        .as_any_mut()
        .downcast_mut::<CpuQ4kCacheHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::cached_decode_step_q4k received a foreign handle \
                 (backend={backend_name}); handles must be allocated by the same \
                 backend that consumes them"
            )
        })
}

// ─── KvDispatch impl ────────────────────────────────────────────────────────

impl KvDispatch for CpuBackend {
    fn alloc_kv_buffer(&self, layer: usize, _max_tokens: usize, kv_dim: usize) -> KvHandle {
        // `max_tokens` is informational on CPU — we grow the buffer on
        // append rather than pre-allocate. GPU backends will pre-allocate.
        KvHandle::new(CpuKvHandle::new(layer, kv_dim))
    }

    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], _abs_position: usize) {
        // `abs_position` is informational on CPU — the K/V buffer is
        // ordered by insertion, and RoPE rotations are applied by the
        // caller (or by attention_step's underlying function).
        let h = cpu_handle_mut(handle);
        debug_assert_eq!(k_row.len(), h.kv_dim);
        debug_assert_eq!(v_row.len(), h.kv_dim);

        let new_k_row = Array2::from_shape_vec((1, k_row.len()), k_row.to_vec())
            .expect("k_row length doesn't match handle's kv_dim");
        let new_v_row = Array2::from_shape_vec((1, v_row.len()), v_row.to_vec())
            .expect("v_row length doesn't match handle's kv_dim");

        h.state = Some(match h.state.take() {
            Some((mut k, mut v)) => {
                k.append(ndarray::Axis(0), new_k_row.view()).unwrap();
                v.append(ndarray::Axis(0), new_v_row.view()).unwrap();
                (k, v)
            }
            None => (new_k_row, new_v_row),
        });
    }

    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        let h = cpu_handle_mut(handle);
        if let Some((k, v)) = h.state.as_mut() {
            let rows = k.shape()[0];
            if rows > window_size {
                let start = rows - window_size;
                let k_slice = k.slice(ndarray::s![start..rows, ..]).to_owned();
                let v_slice = v.slice(ndarray::s![start..rows, ..]).to_owned();
                *k = k_slice;
                *v = v_slice;
            }
        }
    }

    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        let h = cpu_handle(handle);
        h.state.as_ref().map(|(k, v)| (k.clone(), v.clone()))
    }

    fn attention_step(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        _index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<Array2<f32>> {
        // CpuBackend reads f32 attention tensors out of `weights.tensors`.
        // When the caller has a Q4K `VectorIndex`, it's expected to have
        // already populated `weights.tensors` via
        // `crate::vindex::ensure_attn_tensors_dequantised` before
        // dispatching here. Until phase-3 CPU Q4K matvec kernels land,
        // the `index` parameter is accepted for trait-shape compatibility
        // but not consumed.
        let h = cpu_handle_mut(kv);
        let prior_kv = h.as_shared_kv().cloned();
        let (h_post_attn, new_kv) = run_attention_block_decode_step_backend(
            weights,
            query,
            layer,
            prior_kv.as_ref(),
            abs_position,
            Some(self),
        )?;
        h.replace_state(new_kv);
        Some(h_post_attn)
    }

    fn attention_prefill(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        _window: Option<usize>,
        _index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        // See `attention_step` doc for the `_index` convention.
        let (h_post_attn, k_rope, v) =
            run_attention_with_kv_backend(weights, tokens_embedded, layer, Some(self))?;
        let kv_dim = k_rope.shape()[1];
        let mut handle = CpuKvHandle::new(layer, kv_dim);
        handle.replace_state((k_rope, v));
        Some((h_post_attn, KvHandle::new(handle)))
    }

    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        let s = residual.shape();
        let (rows, cols) = (s[0], s[1]);
        let flat = residual
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| residual.iter().copied().collect());
        Some(ResidualHandle::new(CpuResidualHandle {
            flat,
            shape: (rows, cols),
        }))
    }

    fn forward_from_layer(
        &self,
        weights: &ModelWeights,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let r = cpu_residual(residuals);
        let raw =
            crate::forward::forward_from_layer(weights, token_ids, &r.flat, start_layer, None);
        // The returned `RawForward` has `h_pre_norm` shape [seq_len, hidden];
        // engines want the last position's hidden as [1, hidden].
        let h = raw.h_pre_norm;
        let last = h.shape()[0] - 1;
        Some(h.slice(ndarray::s![last..=last, ..]).to_owned())
    }

    // `recompute_kv_from_residuals`, `compressed_kv_append`,
    // `attention_step_windowed`, and `residual_norm_store` use the
    // trait defaults (decomposition / unimplemented). Step 3 engine
    // migration adds overrides when the engines that consume them
    // actually need a CPU body.

    // ── Coarse fused intents ────────────────────────────────────────
    //
    // Route through the production cached-decode pipeline. Backend
    // inspects `index` (when present) and `weights` to pick the right
    // kernel — Q4K matvec today, future quant formats slot in without
    // changing the trait surface or the engine call sites.

    fn coarse_prefill(
        &self,
        weights: &mut ModelWeights,
        token_ids: &[u32],
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        if token_ids.is_empty() {
            return None;
        }
        // The cached-decode path needs the vindex (where the quant
        // weights live). For f32-only models, engines use the per-layer
        // `attention_prefill` dispatch path; no coarse f32 path here.
        let index = index?;
        if !crate::vindex::supports_cached_decode(weights) {
            // Hybrid MoE / cross-layer KV sharing models — the cached
            // path doesn't apply. Engine falls back to per-layer dispatch.
            return None;
        }
        let (h_full, cache, _timings) =
            crate::vindex::predict_kquant_prefill(weights, token_ids, index);
        let last = h_full.shape()[0] - 1;
        let h = h_full.slice(ndarray::s![last..=last, ..]).to_owned();
        let handle = KvHandle::new(CpuQ4kCacheHandle { cache });
        Some((h, handle))
    }

    fn coarse_decode_step(
        &self,
        weights: &mut ModelWeights,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
    ) -> Option<Array2<f32>> {
        let index = index?;
        let inner = cpu_q4k_cache_mut(handle);
        // Prefer direct-matvec (no per-layer dequant) when supported.
        if crate::vindex::supports_direct_matvec_decode(weights, index) {
            crate::vindex::predict_kquant_decode_step_direct(
                weights,
                token_id,
                index,
                self,
                &mut inner.cache,
                abs_position,
            )
        } else {
            crate::vindex::predict_kquant_decode_step(
                weights,
                token_id,
                index,
                &mut inner.cache,
                abs_position,
            )
            .map(|(h, _)| h)
        }
    }
}

#[cfg(test)]
mod tests {
    //! Step 2c parity tests live here. Each test exercises a `KvDispatch`
    //! method on `CpuBackend` and verifies the output matches the
    //! corresponding legacy function call bit-for-bit on synthetic
    //! weights.

    use super::*;
    use crate::test_utils::make_test_weights;
    use larql_compute::CpuBackend;

    fn backend() -> CpuBackend {
        CpuBackend
    }

    #[test]
    fn alloc_kv_buffer_creates_empty_handle() {
        let backend = backend();
        let handle = backend.alloc_kv_buffer(0, 32, 64);
        assert_eq!(handle.cached_len(), 0);
        assert_eq!(handle.kv_dim(), 64);
        assert_eq!(handle.backend_name(), "cpu");
    }

    #[test]
    fn append_kv_grows_handle() {
        let backend = backend();
        let mut handle = backend.alloc_kv_buffer(0, 32, 4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        backend.append_kv(&mut handle, &k, &v, 0);
        assert_eq!(handle.cached_len(), 1);
        backend.append_kv(&mut handle, &k, &v, 1);
        assert_eq!(handle.cached_len(), 2);
    }

    #[test]
    fn clip_kv_keeps_tail() {
        let backend = backend();
        let mut handle = backend.alloc_kv_buffer(0, 32, 2);
        for i in 0..5 {
            let row = vec![i as f32, i as f32];
            backend.append_kv(&mut handle, &row, &row, i);
        }
        assert_eq!(handle.cached_len(), 5);
        backend.clip_kv(&mut handle, 3);
        assert_eq!(handle.cached_len(), 3);
        // Tail should be rows for positions 2,3,4
        let (k, _v) = backend.read_kv_to_host(&handle).unwrap();
        assert_eq!(k[[0, 0]], 2.0);
        assert_eq!(k[[2, 0]], 4.0);
    }

    #[test]
    fn read_kv_to_host_returns_none_for_empty_handle() {
        let backend = backend();
        let handle = backend.alloc_kv_buffer(0, 32, 4);
        assert!(backend.read_kv_to_host(&handle).is_none());
    }

    #[test]
    fn upload_boundary_residual_roundtrips() {
        let backend = backend();
        let residual = Array2::from_shape_vec((3, 4), (0..12).map(|i| i as f32).collect()).unwrap();
        let handle = backend.upload_boundary_residual(&residual).unwrap();
        assert_eq!(handle.shape(), (3, 4));
        assert_eq!(handle.backend_name(), "cpu");
    }

    // ── Bit-parity tests vs legacy functions ─────────────────────────────

    #[test]
    fn attention_prefill_matches_legacy_run_attention_with_kv_backend() {
        let weights = make_test_weights();
        let backend = backend();
        let tokens = vec![0u32, 1, 2];
        let h_in = crate::forward::embed_tokens_pub(&weights, &tokens);

        // Trait dispatch.
        let (h_trait, handle) = backend
            .attention_prefill(&weights, &h_in, 0, None, None)
            .expect("attention_prefill");
        let (k_trait, v_trait) = backend.read_kv_to_host(&handle).unwrap();

        // Legacy direct call — same backend reference passed through.
        let (h_legacy, k_legacy, v_legacy) =
            run_attention_with_kv_backend(&weights, &h_in, 0, Some(&backend))
                .expect("legacy attention");

        assert_eq!(
            h_trait, h_legacy,
            "attention_prefill hidden must match legacy bit-for-bit"
        );
        assert_eq!(k_trait, k_legacy, "K must match legacy bit-for-bit");
        assert_eq!(v_trait, v_legacy, "V must match legacy bit-for-bit");
    }

    #[test]
    fn attention_step_matches_legacy_decode_step_backend() {
        let weights = make_test_weights();
        let backend = backend();
        let tokens = vec![0u32, 1, 2];
        let h_in = crate::forward::embed_tokens_pub(&weights, &tokens);

        // Populate handle via prefill.
        let (_, mut handle) = backend
            .attention_prefill(&weights, &h_in, 0, None, None)
            .unwrap();
        let prior_len = handle.cached_len();

        // Snapshot prior K/V before the trait call mutates the handle.
        let (k_prior, v_prior) = backend.read_kv_to_host(&handle).unwrap();
        let prior_kv = (k_prior, v_prior);

        // Build a 1-row query as if decoding the next token.
        let h_new = crate::forward::embed_tokens_pub(&weights, &[3u32]);
        let abs_position = tokens.len(); // next position

        // Trait dispatch — mutates handle.
        let h_trait = backend
            .attention_step(&weights, &h_new, &mut handle, 0, abs_position, None)
            .expect("attention_step");

        // Legacy: same prior K/V, same call.
        let (h_legacy, legacy_new_kv) = run_attention_block_decode_step_backend(
            &weights,
            &h_new,
            0,
            Some(&prior_kv),
            abs_position,
            Some(&backend),
        )
        .expect("legacy decode step");

        assert_eq!(
            h_trait, h_legacy,
            "attention_step hidden must match legacy bit-for-bit"
        );
        // Handle should now hold the legacy `new_kv` (prior + new row).
        let (k_after, v_after) = backend.read_kv_to_host(&handle).unwrap();
        assert_eq!(
            k_after, legacy_new_kv.0,
            "attention_step must mutate handle K to legacy new_kv.0"
        );
        assert_eq!(
            v_after, legacy_new_kv.1,
            "attention_step must mutate handle V to legacy new_kv.1"
        );
        assert_eq!(
            handle.cached_len(),
            prior_len + 1,
            "handle cached_len must grow by one row"
        );
    }

    #[test]
    fn forward_from_layer_matches_legacy() {
        let weights = make_test_weights();
        let backend = backend();
        let tokens = vec![0u32, 1, 2];

        // Build a synthetic boundary residual (single position, hidden-wide).
        let residual =
            Array2::from_shape_vec((1, weights.hidden_size), vec![0.0; weights.hidden_size])
                .unwrap();
        let residual_flat = residual.as_slice().unwrap().to_vec();

        let handle = backend.upload_boundary_residual(&residual).unwrap();
        let h_trait = backend
            .forward_from_layer(&weights, 1, &handle, &tokens)
            .expect("forward_from_layer");
        assert_eq!(h_trait.shape(), &[1, weights.hidden_size]);

        let legacy = crate::forward::forward_from_layer(&weights, &tokens, &residual_flat, 1, None);
        let last = legacy.h_pre_norm.shape()[0] - 1;
        let h_legacy = legacy
            .h_pre_norm
            .slice(ndarray::s![last..=last, ..])
            .to_owned();
        assert_eq!(
            h_trait, h_legacy,
            "forward_from_layer hidden must match legacy bit-for-bit"
        );
    }

    #[test]
    fn cross_backend_handle_panics() {
        // Construct a synthetic non-CPU handle (any other KvHandleInner)
        // and verify the downcast guard panics rather than silently
        // misinterpreting bytes.
        struct FakeHandle;
        impl KvHandleInner for FakeHandle {
            fn cached_len(&self) -> usize {
                0
            }
            fn kv_dim(&self) -> usize {
                0
            }
            fn backend_name(&self) -> &'static str {
                "fake"
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
        }

        let backend = backend();
        let fake = KvHandle::new(FakeHandle);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.read_kv_to_host(&fake);
        }));
        assert!(
            result.is_err(),
            "expected panic when foreign handle passed to CpuBackend"
        );
    }
}
