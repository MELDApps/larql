//! Local Metal/CPU bench (`run_larql`). Loads the vindex, picks a backend,
//! runs `generate()` with `warmup + tokens` steps, and reports per-stage
//! timings. Mirrors the production `walk_cmd` Q4K path.

use std::time::Instant;

use super::args::BenchArgs;
use super::row::{compute_percentiles, BenchRow};

/// Run the larql generate loop once with the selected backend.
///
/// Warmup runs are discarded; the measured window is `args.tokens` steps
/// AFTER warmup. Because the shared `generate()` doesn't expose a "run
/// N extra steps silently" hook, we run a single call with
/// `max_tokens = warmup + tokens` and subtract. Good enough — the
/// variance between the first-call warmup and later steady-state is
/// absorbed into the discarded prefix.
pub(super) fn run_larql(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    metal: bool,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::layer_graph::generate::generate;
    use larql_inference::layer_graph::CachedLayerGraph;

    if args.verbose {
        eprintln!(
            "[bench] loading vindex for {}…",
            if metal { "metal" } else { "cpu" }
        );
    }

    // Load the vindex once per backend. This mirrors `walk_cmd`'s Q4K
    // path — attention + interleaved Q4K mmaps, weights via the
    // Q4K-specific loader (the plain `load_model_weights` rejects
    // quantised vindexes).
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut q4_index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)?;
    q4_index.load_attn_q4k(vindex_path)?;
    q4_index.load_interleaved_q4k(vindex_path)?;

    let cfg = larql_vindex::load_vindex_config(vindex_path)?;
    if cfg.quant != larql_vindex::QuantFormat::Q4K {
        return Err(format!(
            "larql bench currently requires a Q4K vindex (got {:?})",
            cfg.quant,
        )
        .into());
    }
    let mut weights = larql_vindex::load_model_weights_q4k(vindex_path, &mut cb)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)?;
    let wrapped_prompt = larql_inference::chat::render_user_prompt(
        vindex_path,
        weights.arch.family(),
        args.prompt.as_str(),
    )
    .unwrap_or_else(|_| args.prompt.to_string());
    let token_ids: Vec<u32> =
        larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
            .map_err(|e| format!("tokenize: {e}"))?;

    let backend: Box<dyn larql_compute::ComputeBackend> = if metal {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let b = larql_compute::metal::MetalBackend::new().ok_or(
                "Metal backend unavailable — rebuild with `--features metal` on an M-series Mac",
            )?;
            Box::new(b)
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            return Err("Metal backend requires the `metal` feature on macOS".into());
        }
    } else {
        Box::new(larql_compute::CpuBackend)
    };

    let cached_layers = CachedLayerGraph::from_residuals(Vec::new());

    // Pre-warm: one generate call to allocate the KV cache (~1 GB on Gemma 3 4B)
    // and populate the Metal buffer caches. The prefill timer would otherwise
    // include this one-time allocation cost even though it is amortized to zero
    // in real multi-turn usage.
    if metal {
        let num_layers = weights.num_layers;
        let _ = generate(
            &mut weights,
            &tokenizer,
            &token_ids,
            1,
            &q4_index,
            &*backend,
            &cached_layers,
            0..num_layers,
        );
    }

    if args.profile {
        std::env::set_var("LARQL_PROFILE_SPLIT", "1");
    }
    let max_tokens = args.warmup + args.tokens;
    let num_layers = weights.num_layers;
    let t0 = Instant::now();
    let result = generate(
        &mut weights,
        &tokenizer,
        &token_ids,
        max_tokens,
        &q4_index,
        &*backend,
        &cached_layers,
        0..num_layers,
    );
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Q4_K dequant cache footprint after the run. The full-K Metal fast
    // path streams Q4_K bytes through `q4k_matmul_transb` and should NOT
    // populate this cache; the per-position fallback in walk_ffn/sparse
    // does. Print it on `-v` so the perf audit can verify which path
    // was taken without running vmmap.
    if args.verbose {
        let (slots, bytes) = q4_index.q4k_ffn_cache_stats();
        eprintln!(
            "[bench] q4k_ffn_cache after {}: {} populated slots, {:.1} MB",
            backend_name_for(metal),
            slots,
            bytes as f64 / 1_048_576.0,
        );
    }

    let n_warm = args.warmup.min(result.decode_ms.len());
    let measured = &result.decode_ms[n_warm..];
    let measured_n = measured.len();
    let (prefill_ms, avg_decode_ms, p50_ms, p99_ms, tok_per_s) = if measured_n == 0 {
        (result.prefill_ms, 0.0, 0.0, 0.0, 0.0)
    } else {
        let (avg, p50, p99) = compute_percentiles(measured);
        (result.prefill_ms, avg, p50, p99, 1000.0 / avg)
    };

    let backend_name = backend_name_for(metal);
    let note = if measured_n < args.tokens {
        format!(
            "early stop @{}/{} (EOS or GPU fallback)",
            measured_n, args.tokens
        )
    } else if measured_n == 0 {
        format!("no decode steps completed (wall {:.0}ms)", wall_ms)
    } else {
        String::new()
    };

    // StageTimings across ALL decode steps (including warmup); we'd need
    // to re-architect `generate` to bucket post-warmup only. Report the
    // raw totals and let the caller compute the post-warmup ratio
    // heuristically (~same within noise on 50-token runs).
    let stages = Some(result.stage_timings.avg_per_step(result.decode_ms.len()));

    Ok(BenchRow {
        backend: backend_name.to_string(),
        prefill_ms,
        avg_decode_ms,
        p50_ms,
        p99_ms,
        tok_per_s,
        stages,
        ffn_rtt_ms: None,
        attn_ms: None,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: measured_n,
        note,
    })
}

pub(super) fn backend_name_for(metal: bool) -> &'static str {
    if metal {
        "larql-metal"
    } else {
        "larql-cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_for_picks_label() {
        assert_eq!(backend_name_for(true), "larql-metal");
        assert_eq!(backend_name_for(false), "larql-cpu");
    }
}
