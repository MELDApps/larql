//! CPU Q4K generate path — used when the active backend does not support the
//! fused Q4 prefill + KV-cached decode pipeline (today: CpuBackend).

use super::{
    eos::EosConfig,
    types::{GenerateError, GenerateResult, StageTimings},
};
use crate::forward::PredictResult;
use crate::model::ModelWeights;
use larql_compute::prelude::*;

// ── Backend capability probe + CPU Q4K delegation ────────────────────────────
//
// `generate` / `generate_constrained` assume the backend implements the fused
// Q4 prefill + KV-cached decode pipeline (currently: Metal). Backends that
// lack it (CpuBackend) delegate to the per-layer CPU Q4K dequant path
// (`predict_q4k_hidden`), which mutates `weights.tensors` per layer — that's
// the single reason these functions take `&mut ModelWeights`.

/// True when the backend can handle the fused Q4 prefill + decode pipeline
/// directly. Metal: yes. Pure CPU: no — that path produces correct forward
/// results via the vindex Q4K dequant loop in `crate::vindex::q4k_forward`.
pub(super) fn backend_supports_fused_q4_pipeline(backend: &dyn ComputeBackend) -> bool {
    backend.supports(Capability::PrefillQ4) && backend.supports(Capability::DecodeToken)
}

/// CPU Q4K generate path. For dense single-stream architectures (no
/// hybrid MoE, no cross-layer KV sharing) this uses the KV-cached
/// driver in [`crate::vindex::predict_q4k_prefill`] +
/// [`crate::vindex::predict_q4k_decode_step`]: full prompt once at
/// prefill, then 1-row attention + 1-row FFN per generated token.
/// Falls back to the original O(N²) per-step `predict_q4k_hidden`
/// loop for hybrid MoE (Gemma 4 26B A4B) and Gemma 4 E2B
/// (cross-layer KV sharing).
pub(super) fn generate_via_cpu_q4k(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    eos: &EosConfig,
) -> GenerateResult {
    if max_tokens == 0 {
        return GenerateResult::empty_success();
    }

    if crate::vindex::supports_cached_decode(weights) {
        let use_direct = crate::vindex::supports_direct_matvec_decode(weights, index);
        generate_via_cpu_q4k_cached(
            weights, tokenizer, token_ids, max_tokens, index, eos, use_direct,
        )
    } else {
        generate_via_cpu_q4k_uncached(weights, tokenizer, token_ids, max_tokens, index, eos)
    }
}

/// KV-cached path. Decode work per step is O(1) in N (single-row
/// attention vs growing K/V) instead of O(N²). Dense architectures
/// without cross-layer KV sharing only.
///
/// `direct_matvec`: when true, decode steps skip the per-layer Q4_K →
/// f32 dequant staging and call `backend.quant_matvec` against the
/// vindex's raw Q4_K/Q6_K bytes directly. Massive win — dequant was
/// ~93% of CPU forward time on Gemma 3 4B Q4_K.
#[allow(clippy::too_many_arguments)]
fn generate_via_cpu_q4k_cached(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    eos: &EosConfig,
    direct_matvec: bool,
) -> GenerateResult {
    // ── Prefill ────────────────────────────────────────────────────
    let prefill_start = std::time::Instant::now();
    let (h_prompt, mut cache, _prefill_timings) =
        crate::vindex::predict_q4k_prefill(weights, token_ids, index);
    // Don't fold prefill dequant into per-step averages — bench numbers
    // already account for the prompt pass via `prefill_ms`. Mixing them
    // here would mis-attribute the one-shot prefill cost to decode.

    let backend: Box<dyn larql_compute::ComputeBackend> = Box::new(larql_compute::CpuBackend);
    // Q4_K view of the lm_head, synthesised from the f16 embeddings at
    // vindex load for tied-embedding models (Gemma, Llama). When
    // available we route the lm_head matmul through `q4k_matvec`
    // instead of the f32 row-parallel sgemv — drops lm_head bandwidth
    // from ~2.7 GB to ~0.7 GB per step.
    let lm_head_q4: Option<&[u8]> = index.storage.lm_head_q4_view().map(|b| b.as_ref());
    let lm_head_vocab = index.vocab_size;

    // lm_head + argmax on the last prompt position to seed decode.
    let h_last = last_row_as_2d(&h_prompt);
    let lm_head_start = std::time::Instant::now();
    let first = lm_head_predict(
        weights,
        &h_last,
        tokenizer,
        backend.as_ref(),
        lm_head_q4,
        lm_head_vocab,
    );
    let mut t_lm_head = lm_head_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);
    let mut t_cpu_fwd = 0.0f64;
    let mut t_dequant = 0.0f64;

    let mut next_id = match (first.token_ids.first(), first.predictions.first()) {
        (Some(&id), Some(first_pred)) => {
            tokens.push((first_pred.0.clone(), 1.0));
            if eos.is_eos_with_tokenizer(id, &first_pred.0, tokenizer) {
                return GenerateResult {
                    tokens,
                    prefill_ms,
                    decode_ms,
                    stage_timings: StageTimings {
                        cpu_fwd_ms_total: t_cpu_fwd,
                        lm_head_ms_total: t_lm_head,
                        dequant_ms_total: t_dequant,
                        ..Default::default()
                    },
                    error: None,
                };
            }
            id
        }
        _ => {
            return GenerateResult {
                tokens,
                prefill_ms,
                decode_ms,
                stage_timings: StageTimings::default(),
                error: Some(GenerateError::empty_output(
                    "CPU Q4K generation produced no first token",
                )),
            };
        }
    };

    // ── Decode loop ────────────────────────────────────────────────
    let prompt_len = token_ids.len();
    for step in 1..max_tokens {
        let abs_position = prompt_len + (step - 1);
        let t0 = std::time::Instant::now();
        let h_new = if direct_matvec {
            match crate::vindex::predict_q4k_decode_step_direct(
                weights,
                next_id,
                index,
                backend.as_ref(),
                &mut cache,
                abs_position,
            ) {
                Some(h) => h,
                None => break,
            }
        } else {
            match crate::vindex::predict_q4k_decode_step(
                weights,
                next_id,
                index,
                &mut cache,
                abs_position,
            ) {
                Some((h, step_timings)) => {
                    t_dequant += step_timings.dequant_ms;
                    h
                }
                None => break,
            }
        };
        let hidden_ms = t0.elapsed().as_secs_f64() * 1000.0;
        t_cpu_fwd += hidden_ms;

        let lm_head_start = std::time::Instant::now();
        let result = lm_head_predict(
            weights,
            &h_new,
            tokenizer,
            backend.as_ref(),
            lm_head_q4,
            lm_head_vocab,
        );
        let lm_head_ms = lm_head_start.elapsed().as_secs_f64() * 1000.0;
        t_lm_head += lm_head_ms;
        decode_ms.push(hidden_ms + lm_head_ms);

        let id = match result.token_ids.first() {
            Some(&id) => id,
            None => break,
        };
        let tok = result
            .predictions
            .first()
            .map(|p| p.0.clone())
            .unwrap_or_default();
        let stop = eos.is_eos_with_tokenizer(id, &tok, tokenizer);
        tokens.push((tok, 1.0));
        if stop {
            break;
        }
        next_id = id;
    }

    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings {
            embed_ms_total: 0.0,
            gpu_ms_total: 0.0,
            cpu_fwd_ms_total: t_cpu_fwd,
            gate_up_ms_total: 0.0,
            down_ms_total: 0.0,
            norm_ms_total: 0.0,
            lm_head_ms_total: t_lm_head,
            detok_ms_total: 0.0,
            dequant_ms_total: t_dequant,
        },
        error: None,
    }
}

/// Legacy O(N²) loop for architectures the cached path can't handle
/// (hybrid MoE, KV sharing). Re-runs `predict_q4k_hidden` over the
/// growing token sequence at every decode step.
fn generate_via_cpu_q4k_uncached(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    eos: &EosConfig,
) -> GenerateResult {
    let prefill_start = std::time::Instant::now();
    let (first, _, _) = predict_q4k_timed(weights, tokenizer, token_ids, 5, index);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);
    let mut t_cpu_fwd = 0.0f64;
    let mut t_lm_head = 0.0f64;

    let mut ids = token_ids.to_vec();
    if let (Some(&id), Some(first_pred)) = (first.token_ids.first(), first.predictions.first()) {
        tokens.push((first_pred.0.clone(), 1.0));
        let stop = eos.is_eos_with_tokenizer(id, &first_pred.0, tokenizer);
        ids.push(id);
        if stop {
            return GenerateResult {
                tokens,
                prefill_ms,
                decode_ms,
                stage_timings: StageTimings::default(),
                error: None,
            };
        }
    } else {
        return GenerateResult {
            tokens,
            prefill_ms,
            decode_ms,
            stage_timings: StageTimings::default(),
            error: Some(GenerateError::empty_output(
                "CPU Q4K generation produced no first token",
            )),
        };
    }

    for _step in 1..max_tokens {
        let t0 = std::time::Instant::now();
        let (result, hidden_ms, lm_head_ms) = predict_q4k_timed(weights, tokenizer, &ids, 5, index);
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        decode_ms.push(step_ms);
        t_cpu_fwd += hidden_ms;
        t_lm_head += lm_head_ms;

        match result.token_ids.first() {
            Some(&id) => {
                let tok = result
                    .predictions
                    .first()
                    .map(|p| p.0.clone())
                    .unwrap_or_default();
                let stop = eos.is_eos_with_tokenizer(id, &tok, tokenizer);
                tokens.push((tok, 1.0));
                ids.push(id);
                if stop {
                    break;
                }
            }
            None => break,
        }
    }

    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings {
            embed_ms_total: 0.0,
            gpu_ms_total: 0.0,
            cpu_fwd_ms_total: t_cpu_fwd,
            gate_up_ms_total: 0.0,
            down_ms_total: 0.0,
            norm_ms_total: 0.0,
            lm_head_ms_total: t_lm_head,
            detok_ms_total: 0.0,
            dequant_ms_total: 0.0,
        },
        error: None,
    }
}

fn last_row_as_2d(h: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    let mut out = ndarray::Array2::<f32>::zeros((1, hidden));
    out.row_mut(0).assign(&h.row(seq_len - 1));
    out
}

/// Decode-loop lm_head wrapper. When the vindex carries a Q4_K view of
/// the LM head (always true for tied-embedding models — Gemma 3/4,
/// Llama with `tie_word_embeddings=true`) the matmul runs through
/// `q4k_matvec` against the synthesised Q4_K bytes, saving ~2 GB of
/// f32 bandwidth per step on Gemma 3 4B. Falls back to the f32
/// row-parallel path when the Q4 view is missing.
fn lm_head_predict(
    weights: &ModelWeights,
    h: &ndarray::Array2<f32>,
    tokenizer: &tokenizers::Tokenizer,
    backend: &dyn larql_compute::ComputeBackend,
    q4_lm_head: Option<&[u8]>,
    vocab: usize,
) -> crate::forward::PredictResult {
    if let Some(q4_bytes) = q4_lm_head {
        if vocab > 0 {
            return crate::forward::predict::logits_to_predictions_q4_lm_head(
                weights, h, q4_bytes, vocab, backend, tokenizer, 5, 1.0,
            );
        }
    }
    crate::forward::predict::logits_to_predictions_pub(weights, h, tokenizer, 5, 1.0)
}

fn predict_q4k_timed(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
) -> (PredictResult, f64, f64) {
    let hidden_start = std::time::Instant::now();
    let h = crate::vindex::predict_q4k_hidden(weights, token_ids, index, None);
    let hidden_ms = hidden_start.elapsed().as_secs_f64() * 1000.0;

    let lm_head_start = std::time::Instant::now();
    let result =
        crate::forward::predict::logits_to_predictions_pub(weights, &h, tokenizer, top_k, 1.0);
    let lm_head_ms = lm_head_start.elapsed().as_secs_f64() * 1000.0;

    (result, hidden_ms, lm_head_ms)
}

/// Sampling-aware bridge to the CPU Q4_K constrained decoder. Threads
/// the caller's `SamplingConfig` (temperature/top_p/seed/penalties)
/// through to token selection over the masked logits.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate_constrained_via_cpu_q4k_streaming_sampled<M, F>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    mask_fn: M,
    on_token: F,
    sampling: super::sampling::SamplingConfig,
    eos: &EosConfig,
) -> GenerateResult
where
    M: FnMut(&[u32], &mut Vec<f32>),
    F: FnMut(u32, &str, f64),
{
    if max_tokens == 0 {
        return GenerateResult::empty_success();
    }

    let prefill_start = std::time::Instant::now();
    let out = crate::vindex::generate_q4k_cpu_constrained_streaming_sampled_with_eos(
        weights, tokenizer, token_ids, max_tokens, index, mask_fn, on_token, sampling, eos,
    );
    let total_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    // Heuristic split: attribute the first token to prefill, the rest to
    // decode. Matches the semantics of the GPU path closely enough for
    // bench-report purposes without tracking per-step timing inside the
    // constrained CPU loop.
    let n = out.len();
    let (prefill_ms, decode_ms_each) = if n == 0 {
        (total_ms, 0.0)
    } else {
        let avg = total_ms / n as f64;
        (avg, avg)
    };
    let tokens: Vec<(String, f64)> = out.into_iter().map(|(t, _)| (t, 1.0)).collect();
    let decode_ms = (1..tokens.len()).map(|_| decode_ms_each).collect();
    GenerateResult {
        tokens,
        prefill_ms,
        decode_ms,
        stage_timings: StageTimings::default(),
        error: None,
    }
}
