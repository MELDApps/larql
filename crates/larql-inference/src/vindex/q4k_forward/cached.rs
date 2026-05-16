//! KV-cached CPU Q4_K decode.
//!
//! `predict_q4k_hidden` (sibling module) reprocesses the entire
//! `token_ids` sequence at every decode step — O(N²) work where N
//! grows with each generated token. This module splits that into
//! prefill (full-sequence pass that captures K/V per layer) plus
//! per-step decode (single-row attention against the cache + 1-row
//! FFN). Speedup scales linearly with decode length.
//!
//! Per-step Q4_K → f32 dequant via `insert_q4k_layer_tensors` is
//! still paid for now; eliminating it is a follow-up (route Q/K/V/O
//! and gate/up/down through `backend.q4k_matvec` directly).
//!
//! Scope: dense architectures only. Hybrid-MoE (Gemma 4 26B A4B)
//! and cross-layer KV sharing (Gemma 4 E2B) fall back to the slow
//! `predict_q4k_hidden` path — the caller decides via
//! [`supports_cached_decode`].

use larql_compute::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_gate_up_into, q4k_q8k_matvec_into, q6k_q8k_matvec_into, quantize_x_to_q8k_into,
    Q8KActivation,
};
use larql_compute::ComputeBackend;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::attention::{
    decode::{gqa_attention_decode_step, run_attention_block_decode_step_backend},
    rope::apply_rope_partial_at,
    run_attention_with_kv_backend,
};
use crate::ffn::WeightFfn;
use crate::forward::embed_tokens_pub;
use crate::forward::layer::apply_layer_scalar;
use crate::forward::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
use crate::forward::run_ffn;
use crate::forward::{add_bias, apply_norm};
use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

use super::tensors::{insert_q4k_layer_tensors, remove_layer_tensors};

/// Per-layer K/V captured during prefill. One entry per layer; matches
/// the [`crate::attention::decode::KvCache`] convention so future work
/// can swap in window clipping or surgery without churn here.
pub type CpuKvCache = Vec<Option<(Array2<f32>, Array2<f32>)>>;

/// Timing instrumentation for the cached CPU Q4K path. Times are
/// summed across all layers in a single call (prefill = one call;
/// decode = one call per generated token).
#[derive(Debug, Default, Clone, Copy)]
pub struct CachedTimings {
    pub dequant_ms: f64,
}

impl CachedTimings {
    fn merge(&mut self, other: CachedTimings) {
        self.dequant_ms += other.dequant_ms;
    }
}

/// True if the cached decode loop can handle this model. False for
/// hybrid-MoE (router/expert path runs through `run_moe_layer_cpu`)
/// and for architectures with cross-layer KV sharing (the decode-step
/// attention helper only knows the "this layer has its own K/V" case
/// today).
pub fn supports_cached_decode(weights: &ModelWeights) -> bool {
    if weights.arch.is_hybrid_moe() {
        return false;
    }
    for layer in 0..weights.num_layers {
        if weights.arch.kv_shared_source_layer(layer).is_some() {
            return false;
        }
    }
    true
}

/// Prefill: run the full prompt through every layer once, capturing
/// each layer's post-RoPE K and final V into the returned cache.
/// Returns the `[seq_len, hidden]` hidden state and the populated
/// cache. Caller takes the last row for lm_head.
pub fn predict_q4k_prefill(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
) -> (Array2<f32>, CpuKvCache, CachedTimings) {
    let num_layers = weights.num_layers;
    let mut cache: CpuKvCache = vec![None; num_layers];
    let mut timings = CachedTimings::default();

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..num_layers {
        let t0 = std::time::Instant::now();
        let inserted =
            insert_q4k_layer_tensors(weights, index, layer).unwrap_or_else(|err| panic!("{err}"));
        timings.dequant_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Attention with K/V capture. Backend stays None — we want the
        // CPU BLAS path for the dequantised f32 tensors that
        // `insert_q4k_layer_tensors` just placed in `weights.tensors`.
        let (h_post_attn, k_rope, v_final) =
            match run_attention_with_kv_backend(weights, &h, layer, None) {
                Some(t) => t,
                None => {
                    remove_layer_tensors(weights, inserted);
                    return (h, cache, timings);
                }
            };

        let ffn = WeightFfn { weights };
        let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);

        remove_layer_tensors(weights, inserted);

        cache[layer] = Some((k_rope, v_final));
        h = h_out;
    }

    (h, cache, timings)
}

/// Decode step: run a single new token through every layer using the
/// prefill cache. Each layer's cache entry is appended to in place.
/// Returns the new `[1, hidden]` hidden state for lm_head.
///
/// `abs_position` is the absolute RoPE position of the new token —
/// `prompt_len + steps_already_decoded`. The caller maintains this
/// counter (typical: `prompt_len + step_index` starting at 0).
pub fn predict_q4k_decode_step(
    weights: &mut ModelWeights,
    token_id: u32,
    index: &VectorIndex,
    cache: &mut CpuKvCache,
    abs_position: usize,
) -> Option<(Array2<f32>, CachedTimings)> {
    let num_layers = weights.num_layers;
    if cache.len() != num_layers {
        return None;
    }
    let mut timings = CachedTimings::default();

    // 1-row embed + 1-row PLE for the new token.
    let mut h = embed_tokens_pub(weights, &[token_id]);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);

    for layer in 0..num_layers {
        let t0 = std::time::Instant::now();
        let inserted =
            insert_q4k_layer_tensors(weights, index, layer).unwrap_or_else(|err| panic!("{err}"));
        timings.dequant_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let kv_entry = cache[layer].as_ref();
        let (h_post_attn, new_kv) = match run_attention_block_decode_step_backend(
            weights,
            &h,
            layer,
            kv_entry,
            abs_position,
            None,
        ) {
            Some(t) => t,
            None => {
                remove_layer_tensors(weights, inserted);
                return None;
            }
        };
        cache[layer] = Some(new_kv);

        let ffn = WeightFfn { weights };
        let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);

        remove_layer_tensors(weights, inserted);

        h = h_out;
    }

    Some((h, timings))
}

impl CachedTimings {
    /// Merge another timing block into self. Useful for accumulating
    /// per-step decode timings across a generation loop.
    pub fn add(&mut self, other: CachedTimings) {
        self.merge(other);
    }
}

// ── Phase 2: dequant-free decode step ───────────────────────────────────
//
// `predict_q4k_decode_step` (above) still pays the per-step Q4_K/Q6_K →
// f32 dequant cost via `insert_q4k_layer_tensors`. Profiling showed
// dequant is ~93% of CPU forward time even with the KV cache wired —
// gemm and attention are a small slice. This module routes Q/K/V/O and
// gate/up/down projections straight through `backend.quant_matvec`
// (CPU `q4k_matvec_into` / `q6k_matvec_into`), skipping the dequant
// staging entirely.

fn matvec_quant(
    backend: &dyn ComputeBackend,
    bytes: &[u8],
    format: &str,
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Option<Vec<f32>> {
    match format {
        "Q4_K" => backend.q4k_matvec(bytes, x, rows, cols),
        "Q6_K" => backend.q6k_matvec(bytes, x, rows, cols),
        _ => None,
    }
}

/// Format-aware Q*K × Q8_K matvec used by the production decode path.
/// Uses NEON `sdot` (Q4_K) or `vmlal_s8` (Q6_K) under the hood — ~2-3×
/// the f32-FMA throughput of `matvec_quant`. Returns `None` for any
/// unsupported format (caller falls through to dequant).
fn matvec_q4k_or_q6k_q8k(
    bytes: &[u8],
    format: &str,
    x_q8k: &Q8KActivation,
    rows: usize,
    cols: usize,
) -> Option<Vec<f32>> {
    if rows == 0 || cols == 0 {
        return Some(vec![0.0f32; rows]);
    }
    const ELEMS_PER_BLOCK: usize = 256;
    if !cols.is_multiple_of(ELEMS_PER_BLOCK) {
        return None;
    }
    let bytes_per_row = match format {
        "Q4_K" => (cols / ELEMS_PER_BLOCK) * 144,
        "Q6_K" => (cols / ELEMS_PER_BLOCK) * 210,
        _ => return None,
    };
    if bytes.len() < rows * bytes_per_row {
        return None;
    }

    // `q4k_q8k_matvec_into` (larql-compute) is a single-threaded
    // per-row loop. Wrap it with `par_chunks_mut(CHUNK_ROWS)` here so
    // every Q4_K/Q6_K × Q8_K matvec on the decode path scales across
    // the 11 perf cores on M3 Max — matching the rayon strategy of
    // `q4k_matvec_into` in `q4_common.rs`. Without this, decode runs
    // single-threaded and the sdot path actually regresses vs the
    // (rayon-parallel) f32 path despite each row being faster.
    use rayon::prelude::*;
    const CHUNK_ROWS: usize = 32;
    let mut out = vec![0.0f32; rows];
    let w_ref = bytes;
    out.par_chunks_mut(CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let row_start = chunk_idx * CHUNK_ROWS;
            let chunk_len = chunk.len().min(rows.saturating_sub(row_start));
            if chunk_len == 0 {
                return;
            }
            let w_chunk =
                &w_ref[row_start * bytes_per_row..(row_start + chunk_len) * bytes_per_row];
            match format {
                "Q4_K" => {
                    q4k_q8k_matvec_into(&mut chunk[..chunk_len], x_q8k, w_chunk, chunk_len, cols)
                }
                "Q6_K" => {
                    q6k_q8k_matvec_into(&mut chunk[..chunk_len], x_q8k, w_chunk, chunk_len, cols)
                }
                _ => {}
            }
        });
    Some(out)
}

/// True when every Q/K/V/O + gate/up/down slice for `layer` is in a
/// format the direct-matvec path knows how to handle. Used to gate
/// per-layer routing: the cached decode step prefers the direct
/// matvec when this returns true and falls back to the dequant path
/// otherwise (e.g. Q4_KF layers, padded down projections).
fn layer_supports_direct_matvec(index: &VectorIndex, layer: usize) -> bool {
    let attn = match index.attn_q4k_layer_data(layer) {
        Some(a) => a,
        None => return false,
    };
    for (_, fmt) in attn.iter() {
        if !matches!(*fmt, "Q4_K" | "Q6_K") {
            return false;
        }
    }
    let ffn = match index.interleaved_q4k_layer_data(layer) {
        Some(f) => f,
        None => return false,
    };
    for (_, fmt) in ffn.iter() {
        if !matches!(*fmt, "Q4_K" | "Q6_K") {
            return false;
        }
    }
    // The down projection in the FFN is sometimes stored with a padded
    // intermediate dim (rounded up to a 256-multiple). `q4k_matvec_into`
    // rejects non-multiple `cols`, which would silently zero the
    // output — refuse the direct path so the dequant fallback runs.
    let intermediate = index.num_features(layer);
    intermediate.is_multiple_of(larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
}

/// True when the whole model can run on the direct-matvec decode path.
/// Same gating as [`supports_cached_decode`] plus a per-layer format
/// check. Used by the bench labeler and as the cpu.rs routing key.
pub fn supports_direct_matvec_decode(weights: &ModelWeights, index: &VectorIndex) -> bool {
    if !supports_cached_decode(weights) {
        return false;
    }
    for layer in 0..weights.num_layers {
        if !layer_supports_direct_matvec(index, layer) {
            return false;
        }
    }
    true
}

fn vec_to_2d_row(v: Vec<f32>) -> Array2<f32> {
    let n = v.len();
    Array2::from_shape_vec((1, n), v).expect("matvec output shape")
}

/// One-row attention block using direct Q4_K/Q6_K matvec on the
/// quantised attention slices. Mirrors
/// [`crate::attention::decode::run_attention_block_decode_step_backend`]
/// but reads weights from `index.attn_q4k_layer_data(layer)` instead of
/// dequantised f32 in `weights.tensors`.
#[allow(clippy::too_many_arguments)]
fn run_attn_decode_step_q4k_direct(
    weights: &ModelWeights,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&(Array2<f32>, Array2<f32>)>,
    abs_position: usize,
) -> Option<(Array2<f32>, (Array2<f32>, Array2<f32>))> {
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let reps = num_q / num_kv;
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let norm_offset = arch.norm_weight_offset();

    let h_norm = apply_norm(
        weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );
    let h_norm_row: &[f32] = h_norm.row(0).to_slice().or_else(|| h_norm.as_slice())?;

    let attn = index.attn_q4k_layer_data(layer)?;
    let (q_bytes, q_fmt) = attn[0];
    let (k_bytes, k_fmt) = attn[1];
    let (v_bytes, v_fmt) = attn[2];
    let (o_bytes, o_fmt) = attn[3];

    // Q8_K-quantise `h_norm` once and reuse for Q / K / V projections.
    // sdot int8 dot is ~2-3× the f32 FMA throughput of the
    // `q4k_matvec_into` path; the quantisation step itself is O(hidden)
    // and amortises across the three projections (and O after attn).
    let mut h_norm_q8k = Q8KActivation::with_capacity(hidden);
    quantize_x_to_q8k_into(&mut h_norm_q8k, h_norm_row);

    let q_vec = matvec_q4k_or_q6k_q8k(q_bytes, q_fmt, &h_norm_q8k, q_dim, hidden)?;
    let mut q_full = vec_to_2d_row(q_vec);
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }

    let qk_offset = arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };
    let q_normed = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let layer_rope_base = arch.rope_base_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let q_rope = apply_rope_partial_at(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        abs_position,
    );

    let k_vec = matvec_q4k_or_q6k_q8k(k_bytes, k_fmt, &h_norm_q8k, kv_dim, hidden)?;
    let v_vec = matvec_q4k_or_q6k_q8k(v_bytes, v_fmt, &h_norm_q8k, kv_dim, hidden)?;
    let mut k_full_new = vec_to_2d_row(k_vec);
    let mut v_full_new = vec_to_2d_row(v_vec);
    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k_full_new, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v_full_new, bias);
    }
    if arch.has_v_norm() {
        v_full_new = rms_norm_heads_no_weight(&v_full_new, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full_new,
    };
    let k_new_rope = apply_rope_partial_at(
        &k_normed,
        num_kv,
        head_dim,
        layer_rope_base,
        rotary_frac,
        abs_position,
    );

    let (k_concat, v_concat) = match kv_entry {
        Some((k_cached, v_cached)) => {
            let total = k_cached.shape()[0] + 1;
            let mut k_out = Array2::<f32>::zeros((total, kv_dim));
            let mut v_out = Array2::<f32>::zeros((total, kv_dim));
            k_out
                .slice_mut(ndarray::s![..k_cached.shape()[0], ..])
                .assign(k_cached);
            v_out
                .slice_mut(ndarray::s![..v_cached.shape()[0], ..])
                .assign(v_cached);
            k_out
                .slice_mut(ndarray::s![k_cached.shape()[0].., ..])
                .assign(&k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&v_full_new);
            (k_out, v_out)
        }
        None => (k_new_rope, v_full_new),
    };

    let softcap = arch.attn_logit_softcapping();
    let attn_out = gqa_attention_decode_step(
        &q_rope, &k_concat, &v_concat, num_q, head_dim, reps, scale, softcap,
    );
    let attn_out_row: &[f32] = attn_out.row(0).to_slice().or_else(|| attn_out.as_slice())?;

    // Re-quantise the attention output for the O projection. Different
    // input from Q/K/V (attn_out vs h_norm), so we need a fresh Q8_K.
    let mut attn_out_q8k = Q8KActivation::with_capacity(q_dim);
    quantize_x_to_q8k_into(&mut attn_out_q8k, attn_out_row);
    let o_vec = matvec_q4k_or_q6k_q8k(o_bytes, o_fmt, &attn_out_q8k, hidden, q_dim)?;
    let mut attn_projected = vec_to_2d_row(o_vec);
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = apply_norm(
            weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h_new + &(&normed * res_mult)
        } else {
            h_new + &normed
        }
    } else if res_mult != 1.0 {
        h_new + &(&attn_projected * res_mult)
    } else {
        h_new + &attn_projected
    };

    Some((h_post_attn, (k_concat, v_concat)))
}

/// One-row gated FFN block using direct Q4_K/Q6_K matvec. Mirrors
/// [`crate::ffn::weight::dense_ffn_forward_backend`] but reads gate/up/
/// down from the vindex slices and avoids the f32 staging.
fn run_ffn_decode_step_q4k_direct(
    weights: &ModelWeights,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    h_post_attn: &Array2<f32>,
    layer: usize,
) -> Option<Array2<f32>> {
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let intermediate = index.num_features(layer);
    let norm_offset = arch.norm_weight_offset();

    // Pre-FFN norm: same selection logic as `run_ffn` — when the arch
    // uses post_norms, the pre-FFN key is `pre_feedforward_layernorm`;
    // otherwise it reuses `post_attention_layernorm` as the FFN input
    // norm. Falls back to weightless RMS when no key is set.
    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_in = match pre_ffn_key {
        Some(key) => apply_norm(weights, h_post_attn, &key, norm_offset),
        None => crate::residual::rms_norm(h_post_attn, None, norm_offset),
    };
    let h_in_row: &[f32] = h_in.row(0).to_slice().or_else(|| h_in.as_slice())?;

    let ffn = index.interleaved_q4k_layer_data(layer)?;
    let (gate_bytes, gate_fmt) = ffn[0];
    let (up_bytes, up_fmt) = ffn[1];
    let (down_bytes, down_fmt) = ffn[2];

    // Only Gated FFNs reach this path today (it's what predict_q4k_hidden
    // currently dequantises). Non-gated archs route through the dequant
    // fallback via the per-layer gate at the caller.
    if arch.ffn_type() != larql_models::FfnType::Gated {
        return None;
    }

    // Q8_K-quantise `h_in` once and feed it to both gate and up via the
    // sdot-based fused matvec. This is the int8-dot Q4_K × Q8_K path
    // that closes the bandwidth gap to llama.cpp on M3 Max.
    let mut h_in_q8k = Q8KActivation::with_capacity(hidden);
    quantize_x_to_q8k_into(&mut h_in_q8k, h_in_row);

    // Two separate matvecs, each rayon-parallel inside
    // `matvec_q4k_or_q6k_q8k`. The "fused gate+up" variant in
    // `larql-compute` (`q4k_q8k_gate_up_into`) is single-threaded;
    // the input vector (10 KB) stays in L1 across two sequential
    // calls anyway, so we don't need explicit fusion to keep `x`
    // hot. Splitting lets both matvecs run row-parallel.
    let gate_vec = matvec_q4k_or_q6k_q8k(gate_bytes, gate_fmt, &h_in_q8k, intermediate, hidden)?;
    let up_vec = matvec_q4k_or_q6k_q8k(up_bytes, up_fmt, &h_in_q8k, intermediate, hidden)?;

    // Element-wise activation: activation(gate) * up.
    let mut activated = vec![0.0f32; intermediate];
    match arch.activation() {
        larql_models::Activation::GeluTanh => {
            let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
            for i in 0..intermediate {
                let x = gate_vec[i];
                let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                let g = 0.5 * x * (1.0 + inner.tanh());
                activated[i] = g * up_vec[i];
            }
        }
        _ => {
            // SiLU = x * sigmoid(x). Same shape as dense_ffn_forward_backend.
            for i in 0..intermediate {
                let x = gate_vec[i];
                let sig = 1.0 / (1.0 + (-x).exp());
                let g = x * sig;
                activated[i] = g * up_vec[i];
            }
        }
    }

    // down projection: out = activated @ W_down.T → [hidden].
    // Re-quantise the post-activation vector (`intermediate`-wide) for
    // the down matvec — different input from gate/up.
    let mut activated_q8k = Q8KActivation::with_capacity(intermediate);
    quantize_x_to_q8k_into(&mut activated_q8k, &activated);
    let down_vec =
        matvec_q4k_or_q6k_q8k(down_bytes, down_fmt, &activated_q8k, hidden, intermediate)?;
    let mut out = vec_to_2d_row(down_vec);
    if let Some(bias) = arch
        .ffn_down_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut out, bias);
    }

    // Post-FFN residual + optional post-FFN layernorm. Same selection
    // logic as `run_ffn`: only fire when has_post_norms() AND the arch
    // exposes a post-FFN norm key.
    let res_mult = arch.residual_multiplier();
    let h_post_ffn = if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => apply_norm(weights, &out, &key, norm_offset),
            None => crate::residual::rms_norm(&out, None, norm_offset),
        };
        if res_mult != 1.0 {
            h_post_attn + &(&normed * res_mult)
        } else {
            h_post_attn + &normed
        }
    } else if res_mult != 1.0 {
        h_post_attn + &(&out * res_mult)
    } else {
        h_post_attn + &out
    };

    Some(h_post_ffn)
}

/// Dequant-free decode step. Same shape contract as
/// [`predict_q4k_decode_step`] but routes every projection through
/// `backend.quant_matvec` instead of the per-layer
/// `insert_q4k_layer_tensors` → dense f32 staging dance. Returns `None`
/// if any layer has a format the direct-matvec path doesn't handle
/// (caller falls back to [`predict_q4k_decode_step`]).
pub fn predict_q4k_decode_step_direct(
    weights: &mut ModelWeights,
    token_id: u32,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    cache: &mut CpuKvCache,
    abs_position: usize,
) -> Option<Array2<f32>> {
    let num_layers = weights.num_layers;
    if cache.len() != num_layers {
        return None;
    }

    let mut h = embed_tokens_pub(weights, &[token_id]);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);

    for layer in 0..num_layers {
        let kv_entry = cache[layer].as_ref();
        let (h_post_attn, new_kv) = run_attn_decode_step_q4k_direct(
            weights,
            index,
            backend,
            &h,
            layer,
            kv_entry,
            abs_position,
        )?;
        cache[layer] = Some(new_kv);

        let h_post_ffn =
            run_ffn_decode_step_q4k_direct(weights, index, backend, &h_post_attn, layer)?;
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);
        h = h_out;
    }

    Some(h)
}
