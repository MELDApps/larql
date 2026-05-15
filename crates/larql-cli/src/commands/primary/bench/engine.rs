//! KV-engine bench paths — `markov-rs` and `unlimited-context`.
//!
//! Two variants:
//!   * `run_engine`     — f32/f16 weight path (slow CPU pipeline).
//!   * `run_engine_q4k` — Q4K weight path (Metal pipeline; production).
//!
//! Both reuse the engine's `prefill` / `decode_step` API plus a greedy
//! argmax over `hidden_to_raw_logits` for next-token selection.

use std::time::Instant;

use larql_kv::EngineKind;

use super::args::BenchArgs;
use super::row::{compute_percentiles, BenchRow};

/// Run the CPU KV-engine bench path for a single engine kind.
///
/// Runs prefill on `token_ids` then decodes `args.tokens` steps with greedy
/// argmax. Reports prefill time, avg decode time, and engine memory.
pub(super) fn run_engine(
    weights: &larql_inference::ModelWeights,
    token_ids: &[u32],
    kv_ref_bytes: usize,
    kind: EngineKind,
    backend: Box<dyn larql_inference::ComputeBackend>,
    args: &BenchArgs,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::forward::hidden_to_raw_logits;

    let mut engine = kind.build_with_profiling(backend, args.profile);
    let info = engine.info();
    let label = if info.config.is_empty() {
        format!("{} [{}]", info.name, info.backend)
    } else {
        format!("{} [{}] ({})", info.name, info.backend, info.config)
    };

    if args.verbose {
        eprintln!("[bench] {}", info.summary());
    }

    // Prefill.
    let t_pre = Instant::now();
    let mut hidden = engine
        .prefill(weights, token_ids)
        .ok_or("engine prefill failed")?;
    let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

    // Decode loop: greedy argmax over vocab.
    let max_steps = args.warmup + args.tokens;
    let mut decode_ms_all: Vec<f64> = Vec::with_capacity(max_steps);
    let mut last_token = {
        let logits = hidden_to_raw_logits(weights, &hidden);
        argmax_token(&logits)
    };

    for _ in 0..max_steps {
        let t = Instant::now();
        hidden = engine
            .decode_step(weights, last_token)
            .ok_or("engine decode_step failed")?;
        decode_ms_all.push(t.elapsed().as_secs_f64() * 1000.0);
        last_token = argmax_token(&hidden_to_raw_logits(weights, &hidden));
    }

    let n_warm = args.warmup.min(decode_ms_all.len());
    let measured = &decode_ms_all[n_warm..];
    let measured_n = measured.len();
    let (avg_decode_ms, p50_ms, p99_ms, tok_per_s) = if measured_n == 0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let (avg, p50, p99) = compute_percentiles(measured);
        (avg, p50, p99, 1000.0 / avg)
    };

    let total_mem = engine.memory_bytes();
    let cold_mem = engine.cold_bytes();
    let hot_mem = total_mem.saturating_sub(cold_mem);
    let ratio = if total_mem > 0 {
        kv_ref_bytes as f64 / total_mem as f64
    } else {
        0.0
    };
    let note = format!(
        "hot={:.1}MB cold={:.1}MB  {:.0}× vs std-kv",
        hot_mem as f64 / 1_048_576.0,
        cold_mem as f64 / 1_048_576.0,
        ratio,
    );

    if args.verbose {
        eprintln!(
            "[bench] {} post-decode: {}",
            info.name,
            engine.info().description
        );
    }
    if args.profile {
        if let Some(summary) = engine.stage_summary() {
            summary.print();
        }
    }

    Ok(BenchRow {
        backend: label,
        prefill_ms,
        avg_decode_ms,
        p50_ms,
        p99_ms,
        tok_per_s,
        stages: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: measured_n,
        note,
    })
}

pub(super) fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Q4K engine bench: uses `prefill_q4k`/`decode_step_q4k` which route through
/// the Metal pipeline (`decode_token`) for UnlimitedContext and WalkFfn Q4K FFN
/// for MarkovRS — both significantly faster than the f32 path.
pub(super) fn run_engine_q4k(
    weights: &mut larql_inference::ModelWeights,
    index: &larql_vindex::VectorIndex,
    token_ids: &[u32],
    kv_ref_bytes: usize,
    kind: EngineKind,
    backend: Box<dyn larql_inference::ComputeBackend>,
    args: &BenchArgs,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    // We need two backend instances: one owned by the engine, one for Q4K calls.
    let want_metal_q4k = args.backends.contains("metal");
    let backend_for_q4k: Box<dyn larql_inference::ComputeBackend> = if want_metal_q4k {
        larql_inference::default_backend()
    } else {
        larql_inference::cpu_backend()
    };
    let mut engine = kind.build_with_profiling(backend, args.profile);
    let info = engine.info();
    let label = if info.config.is_empty() {
        format!("{} [{}] Q4K", info.name, info.backend)
    } else {
        format!("{} [{}] ({}) Q4K", info.name, info.backend, info.config)
    };

    if args.verbose {
        eprintln!("[bench] Q4K engine: {}", info.summary());
    }

    use larql_inference::layer_graph::generate::lm_head_topk;
    let be = backend_for_q4k.as_ref();

    // Pick next token via Metal lm_head (matches production path).
    // Defined as a macro-style helper to avoid closure borrow conflicts with &mut weights.
    macro_rules! pick_next {
        ($h:expr) => {{
            let h_1d = ndarray::Array1::from_iter($h.iter().copied());
            lm_head_topk(index, weights, &h_1d, 1, be)
                .first()
                .map(|(t, _)| *t)
                .unwrap_or_else(|| {
                    argmax_token(&larql_inference::forward::hidden_to_raw_logits(weights, $h))
                })
        }};
    }

    // Prefill via Q4K path.
    let t_pre = Instant::now();
    let mut hidden = engine
        .prefill_q4k(weights, index, token_ids, be)
        .ok_or("Q4K engine prefill failed")?;
    let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

    // Decode loop using Metal lm_head for token selection.
    let max_steps = args.warmup + args.tokens;
    let mut decode_ms_all: Vec<f64> = Vec::with_capacity(max_steps);
    let mut last_token = pick_next!(&hidden);

    for _ in 0..max_steps {
        let t = Instant::now();
        hidden = engine
            .decode_step_q4k(weights, index, last_token, be)
            .ok_or("Q4K engine decode_step failed")?;
        decode_ms_all.push(t.elapsed().as_secs_f64() * 1000.0);
        last_token = pick_next!(&hidden);
    }

    let n_warm = args.warmup.min(decode_ms_all.len());
    let measured = &decode_ms_all[n_warm..];
    let measured_n = measured.len();
    let (avg_decode_ms, p50_ms, p99_ms, tok_per_s) = if measured_n == 0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let (avg, p50, p99) = compute_percentiles(measured);
        (avg, p50, p99, 1000.0 / avg)
    };

    let total_mem = engine.memory_bytes();
    let cold_mem = engine.cold_bytes();
    let hot_mem = total_mem.saturating_sub(cold_mem);
    let ratio = if total_mem > 0 {
        kv_ref_bytes as f64 / total_mem as f64
    } else {
        0.0
    };
    let note = format!(
        "hot={:.1}MB cold={:.1}MB  {:.0}× vs std-kv",
        hot_mem as f64 / 1_048_576.0,
        cold_mem as f64 / 1_048_576.0,
        ratio,
    );

    if args.profile {
        if let Some(summary) = engine.stage_summary() {
            summary.print();
        }
    }

    Ok(BenchRow {
        backend: label,
        prefill_ms,
        avg_decode_ms,
        p50_ms,
        p99_ms,
        tok_per_s,
        stages: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: measured_n,
        note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_token_returns_index_of_max() {
        assert_eq!(argmax_token(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax_token(&[9.0, 0.0, 0.0]), 0);
        assert_eq!(argmax_token(&[0.0, 0.0, 5.5]), 2);
    }

    #[test]
    fn argmax_token_empty_returns_zero() {
        assert_eq!(argmax_token(&[]), 0);
    }

    #[test]
    fn argmax_token_handles_nan_gracefully() {
        // NaN comparisons return Equal under our partial_cmp fallback; the
        // first non-NaN slot wins.
        let v = [f32::NAN, 2.0, 1.0];
        // The fold uses max_by so NaN may or may not win depending on the
        // comparator; we only assert the function does not panic and
        // returns a valid index.
        let idx = argmax_token(&v);
        assert!((idx as usize) < v.len());
    }
}
