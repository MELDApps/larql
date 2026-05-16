//! Q4K attention-weight dequantisation helper.
//!
//! Bridges Q4K vindex data (`VectorIndex::attn_kquant_layer_data`) into
//! `ModelWeights::tensors` as f32 tensors, so `KvDispatch` backends
//! that don't (yet) have native Q4K kernels can fall back to the f32
//! attention path.
//!
//! Lives here (not in `larql-kv`) so the `KvDispatch` trait impls
//! (`CpuBackend`, `MetalBackend` in `crate::kv_dispatch::*`) and the
//! engines that consume them can both reach it without a `larql-kv →
//! larql-inference → larql-kv` cycle.
//!
//! ## Phasing
//!
//! Phase 1 (current): callers invoke this upfront before the
//! `KvDispatch::attention_prefill` loop on a Q4K-loaded `ModelWeights`.
//! Memory cost: all-layer Q/K/V/O f32 tensors stay resident.
//!
//! Phase 3 (future): CpuBackend gains native Q4K matvec via
//! `larql_compute::QuantMatVec::q4k_matvec` per-call; this bulk-dequant
//! helper becomes a debug fallback only.
//!
//! See `docs/specs/kv-dispatch-quantization.md`.

use crate::model::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

/// Dequantise attention Q4K weights (Q, K, V, O) for all layers into
/// `weights.tensors`. Idempotent — skips layers whose `attn_q_key` is
/// already present in `weights.tensors`.
///
/// No-op for layers where `index.attn_kquant_layer_data(layer)` returns
/// `None` (i.e., a layer with non-Q4K attention or no Q4K data at all).
pub fn ensure_attn_tensors_dequantised(weights: &mut ModelWeights, index: &VectorIndex) {
    let num_layers = weights.num_layers;
    for layer in 0..num_layers {
        let arch = &*weights.arch;
        let q_key = arch.attn_q_key(layer);
        if weights.tensors.contains_key(&q_key) {
            continue;
        }
        let Some(attn) = index.attn_kquant_layer_data(layer) else {
            continue;
        };
        let num_q = arch.num_q_heads_for_layer(layer);
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let hd = arch.head_dim_for_layer(layer);
        let hidden = weights.hidden_size;
        let q_dim = num_q * hd;
        let kv_dim = num_kv * hd;
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);
        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);
        weights.tensors.insert(q_key, w_q.into_shared());
        weights.tensors.insert(k_key, w_k.into_shared());
        weights.tensors.insert(v_key, w_v.into_shared());
        weights.tensors.insert(o_key, w_o.into_shared());
    }
}

fn dequantize_matrix(bytes: &[u8], format: &str, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    let padded = n.div_ceil(256) * 256;
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format: {format}"));
    let floats = (info.dequantize)(bytes, padded)
        .unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let truncated = if floats.len() > n {
        floats[..n].to_vec()
    } else {
        floats
    };
    Array2::from_shape_vec((rows, cols), truncated).expect("shape mismatch")
}
