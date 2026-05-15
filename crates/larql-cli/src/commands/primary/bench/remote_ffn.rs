//! Remote FFN bench — attention runs locally, FFN is a HTTP round-trip
//! per layer per token. Wraps `run_remote_ffn_bench` with `run_concurrent_ffn`
//! for the `--concurrent N` parallel-client mode.

use super::args::BenchArgs;
use super::helpers::{aggregate_concurrent_rows, ConcurrentSample};
use super::row::{compute_percentiles, BenchRow};

/// Run `args.concurrent` parallel FFN clients against the same shard and
/// aggregate them into one row. With `concurrent == 1` this is a
/// pass-through to `run_remote_ffn_bench`.
///
/// Parallelism is achieved with plain `std::thread::spawn` — the inner
/// bench is synchronous (blocking reqwest under the hood). Threads share
/// the FFN server (the whole point of `--concurrent`).
pub(super) fn run_concurrent_ffn(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    ffn_url: &str,
    pref: larql_inference::WirePreference,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    let n = args.concurrent.max(1);
    if n == 1 {
        return run_remote_ffn_bench(vindex_path, args, ffn_url, pref);
    }

    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let vp = vindex_path.to_path_buf();
        let a = args.clone();
        let url = ffn_url.to_string();
        // The bench error type is `Box<dyn Error>` (not Send), so we stringify
        // any failure inside the worker thread and lift it back to a fresh
        // `Box<dyn Error>` in the parent.
        handles.push(std::thread::spawn(move || {
            run_remote_ffn_bench(&vp, &a, &url, pref).map_err(|e| e.to_string())
        }));
    }

    let mut rows: Vec<BenchRow> = Vec::with_capacity(n);
    for h in handles {
        match h.join() {
            Ok(Ok(row)) => rows.push(row),
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                return Err(
                    "concurrent FFN bench worker panicked — see stderr for details".into(),
                );
            }
        }
    }
    Ok(combine_concurrent_rows(rows, n))
}

/// Fold per-client `BenchRow`s into a single aggregate row. Latency fields
/// use the worst observed value across clients (tail latency is the right
/// thing to report under load); throughput is summed; wire bytes summed
/// when reported. The result row preserves the first client's `backend`
/// and `prefill_ms` (prefill is per-process initialization, not
/// per-client load).
pub(super) fn combine_concurrent_rows(rows: Vec<BenchRow>, n_clients: usize) -> BenchRow {
    debug_assert!(!rows.is_empty());
    let samples: Vec<ConcurrentSample> = rows
        .iter()
        .map(|r| ConcurrentSample {
            tok_per_s: r.tok_per_s,
            mean_ms: r.avg_decode_ms,
            p50_ms: r.p50_ms,
            p99_ms: r.p99_ms,
            wire_bytes_per_tok: r.wire_bytes_per_tok,
        })
        .collect();
    let agg = aggregate_concurrent_rows(&samples).expect("rows is non-empty by debug_assert");
    let first = &rows[0];
    let ffn_rtt_ms = rows
        .iter()
        .filter_map(|r| r.ffn_rtt_ms)
        .fold(0.0_f64, f64::max);
    let attn_ms = rows
        .iter()
        .filter_map(|r| r.attn_ms)
        .fold(0.0_f64, f64::max);
    let ffn_rtt_ms = if ffn_rtt_ms > 0.0 { Some(ffn_rtt_ms) } else { None };
    let attn_ms = if attn_ms > 0.0 { Some(attn_ms) } else { None };
    let total_steps: usize = rows.iter().map(|r| r.n_steps).sum();
    BenchRow {
        backend: format!("{} (×{n_clients} concurrent)", first.backend),
        prefill_ms: first.prefill_ms,
        avg_decode_ms: agg.mean_ms,
        p50_ms: agg.worst_p50_ms,
        p99_ms: agg.worst_p99_ms,
        tok_per_s: agg.aggregate_tok_per_s,
        stages: None,
        ffn_rtt_ms,
        attn_ms,
        wire_bytes_per_tok: agg.total_wire_bytes_per_tok,
        shard_efficiency: None,
        n_steps: total_steps,
        note: format!("concurrent={n_clients} | {}", first.note),
    }
}

/// Bench the remote-FFN path: attention runs locally on Metal, FFN is a
/// round-trip to `ffn_url` via `LayerShardedBackend`.
///
/// Reports overall tok/s plus a breakdown:
///   ffn-rtt  — time spent in the remote FFN closure (all layers summed)
///   attn+    — remainder = local attn + norm + lm_head + embed
pub(super) fn run_remote_ffn_bench(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    ffn_url: &str,
    wire_pref: larql_inference::WirePreference,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::{
        generate_with_remote_ffn, generate_with_remote_ffn_batch, LayerShardedBackend,
    };
    use std::time::Duration;

    if args.verbose {
        eprintln!("[bench] loading vindex for remote-ffn…");
    }

    let timeout = Duration::from_secs(args.ffn_timeout_secs);
    let backend = larql_compute::default_backend();

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_q4k(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load client weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index.load_attn_q4k(vindex_path)?;
    index.load_interleaved_q4k(vindex_path)?;
    let _ = index.load_lm_head_q4(vindex_path);

    eprintln!("Connecting to remote FFN at {ffn_url}…");
    let remote = LayerShardedBackend::connect_with_wire(ffn_url, timeout, wire_pref)
        .map_err(|e| format!("failed to connect to remote FFN: {e}"))?;
    eprintln!("  Attention:  {} (local)", backend.name());
    eprintln!("  FFN:        remote  ({})", ffn_url);

    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), &args.prompt)
            .unwrap_or_else(|_| args.prompt.clone());
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("tokenise: {e}"))?;

    let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(vindex_path);
    let max_tokens = args.warmup + args.tokens;

    let is_batch = args.ffn_dispatch.trim() == "batch";

    // Warmup run — discarded. Amortises TCP connection, Metal init.
    if args.verbose {
        eprintln!("[bench] remote-ffn warmup ({} tokens)…", args.warmup.max(1));
    }
    if is_batch {
        let _ = generate_with_remote_ffn_batch(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            args.warmup.max(1),
            &index,
            &*backend,
            &remote,
            &eos,
            1,
        );
    } else {
        let _ = generate_with_remote_ffn(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            args.warmup.max(1),
            &index,
            &*backend,
            &remote,
            &eos,
        );
    }

    // Reset wire counters so warmup bytes don't pollute the measurement.
    remote.reset_wire_counters();

    // Measured run.
    let t_wall = std::time::Instant::now();
    let result = if is_batch {
        generate_with_remote_ffn_batch(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
            1,
        )
        .map_err(|e| format!("remote-ffn generate failed (batch): {e}"))?
    } else {
        generate_with_remote_ffn(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
        )
        .map_err(|e| format!("remote-ffn generate failed: {e}"))?
    };
    let _wall_ms = t_wall.elapsed().as_secs_f64() * 1000.0;

    let n_warm = args.warmup.min(result.decode_ms.len());
    let measured_decode = &result.decode_ms[n_warm..];
    let measured_ffn = &result.ffn_rtt_ms[n_warm.min(result.ffn_rtt_ms.len())..];
    let n = measured_decode.len();

    let (prefill_ms, avg_decode_ms, p50_ms, p99_ms, tok_per_s, ffn_rtt_ms, attn_ms) = if n == 0 {
        (0.0, 0.0, 0.0, 0.0, 0.0, None, None)
    } else {
        let (avg_decode, p50, p99) = compute_percentiles(measured_decode);
        let avg_ffn = if measured_ffn.len() == n {
            Some(measured_ffn.iter().sum::<f64>() / n as f64)
        } else {
            None
        };
        let avg_attn = avg_ffn.map(|f| (avg_decode - f).max(0.0));
        (
            0.0,
            avg_decode,
            p50,
            p99,
            1000.0 / avg_decode,
            avg_ffn,
            avg_attn,
        )
    };

    let note = if n < args.tokens {
        format!("early stop @{}/{}", n, args.tokens)
    } else {
        String::new()
    };

    let wire_bytes_per_tok = if n > 0 {
        let total = remote.wire_bytes_sent() + remote.wire_bytes_recv();
        Some(total / n as u64)
    } else {
        None
    };

    let _ = weights; // keep alive

    let wire_label = match wire_pref {
        larql_inference::WirePreference::BestAvailable => String::new(),
        _ => format!(" [{}]", wire_pref.label()),
    };
    Ok(BenchRow {
        backend: format!(
            "remote-ffn-{}{} ({})",
            if is_batch { "batch" } else { "stream" },
            wire_label,
            ffn_url
        ),
        prefill_ms,
        avg_decode_ms,
        p50_ms,
        p99_ms,
        tok_per_s,
        stages: None,
        ffn_rtt_ms,
        attn_ms,
        wire_bytes_per_tok,
        shard_efficiency: None,
        n_steps: n,
        note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        backend: &str,
        tok: f64,
        mean: f64,
        p50: f64,
        p99: f64,
        ffn: Option<f64>,
        attn: Option<f64>,
        wb: Option<u64>,
        n_steps: usize,
    ) -> BenchRow {
        BenchRow {
            backend: backend.to_string(),
            prefill_ms: 1.5,
            avg_decode_ms: mean,
            p50_ms: p50,
            p99_ms: p99,
            tok_per_s: tok,
            stages: None,
            ffn_rtt_ms: ffn,
            attn_ms: attn,
            wire_bytes_per_tok: wb,
            shard_efficiency: None,
            n_steps,
            note: "ok".into(),
        }
    }

    #[test]
    fn combine_concurrent_aggregates_throughput_and_takes_worst_tail() {
        let rows = vec![
            row("remote-ffn (x)", 10.0, 100.0, 90.0, 200.0, Some(50.0), Some(50.0), Some(1000), 5),
            row("remote-ffn (x)", 12.0, 90.0, 88.0, 180.0, Some(45.0), Some(45.0), Some(1100), 5),
            row("remote-ffn (x)", 11.0, 95.0, 92.0, 220.0, Some(48.0), Some(47.0), Some(1050), 5),
        ];
        let agg = combine_concurrent_rows(rows, 3);
        assert!(agg.backend.contains("×3 concurrent"));
        assert!(agg.note.starts_with("concurrent=3"));
        assert!((agg.tok_per_s - 33.0).abs() < 1e-9);
        assert!((agg.p99_ms - 220.0).abs() < 1e-9);
        assert_eq!(agg.wire_bytes_per_tok, Some(3150));
        assert_eq!(agg.n_steps, 15);
        // The first row's prefill is preserved.
        assert!((agg.prefill_ms - 1.5).abs() < 1e-9);
        // ffn_rtt_ms and attn_ms become the max observed.
        assert_eq!(agg.ffn_rtt_ms, Some(50.0));
        assert_eq!(agg.attn_ms, Some(50.0));
    }

    #[test]
    fn combine_concurrent_handles_missing_breakdowns() {
        // When every client reports None for ffn/attn the aggregate should
        // also be None (not Some(0.0)).
        let rows = vec![
            row("x", 5.0, 100.0, 90.0, 200.0, None, None, None, 1),
            row("x", 6.0, 100.0, 90.0, 200.0, None, None, None, 1),
        ];
        let agg = combine_concurrent_rows(rows, 2);
        assert!(agg.ffn_rtt_ms.is_none());
        assert!(agg.attn_ms.is_none());
        assert!(agg.wire_bytes_per_tok.is_none());
    }
}
