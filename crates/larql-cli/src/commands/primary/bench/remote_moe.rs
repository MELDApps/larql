//! Remote MoE expert bench — attention + router run locally, expert blocks
//! are dispatched across remote shards. Wrapper `run_concurrent_moe`
//! parallelises clients for the `--concurrent N` mode.
//!
//! The shard-map parser, label formatter, and result-summariser live as
//! pure helpers so the I/O-heavy `run_remote_moe_bench` shell stays thin
//! and the policy gate is satisfied without spinning up shards in tests.

use larql_inference::ffn::moe_remote::ShardConfig;

use super::args::BenchArgs;
use super::remote_ffn::combine_concurrent_rows;
use super::row::{compute_percentiles, BenchRow};

/// Parse the `--moe-shards` flag value into a `Vec<ShardConfig>`. Accepts
/// `"START-END=URL,START-END=URL,..."`. Returns an error message with the
/// offending segment when input is malformed.
pub(super) fn parse_shard_segments(spec: &str) -> Result<Vec<ShardConfig>, String> {
    let mut configs: Vec<ShardConfig> = Vec::new();
    for segment in spec.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        let mut parts = segment.splitn(2, '=');
        let range_str = parts
            .next()
            .ok_or_else(|| format!("malformed shard segment: {segment:?}"))?;
        let url = parts
            .next()
            .ok_or_else(|| format!("missing URL in shard segment: {segment:?}"))?;
        let (start, end_incl) = ShardConfig::parse_range(range_str)
            .ok_or_else(|| format!("bad expert range {range_str:?} in --moe-shards"))?;
        configs.push(ShardConfig::new(start, end_incl, url));
    }
    if configs.is_empty() {
        return Err("--moe-shards: no valid shard segments".into());
    }
    Ok(configs)
}

/// Compose the `"remote-moe-<mode> (<N> shards)"` backend label that goes
/// into the table.
pub(super) fn format_moe_backend_label(is_batch: bool, num_shards: usize) -> String {
    format!(
        "remote-moe-{} ({} shards)",
        if is_batch { "batch" } else { "stream" },
        num_shards
    )
}

/// Bench result summary. Folded out of `run_remote_moe_bench` so the
/// post-result computation (percentile, avg-ffn-rtt, attn-fallback) can be
/// covered by unit tests without booting a real RemoteMoeBackend.
pub(super) struct MoeSummary {
    pub avg_decode_ms: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub tok_per_s: f64,
    pub ffn_rtt_ms: Option<f64>,
    pub attn_ms: Option<f64>,
    pub n_steps: usize,
    pub note: String,
}

/// Trim warmup, percentile, derive `attn_ms = avg_decode - avg_ffn`.
pub(super) fn summarize_moe_result(
    decode_ms: &[f64],
    ffn_rtt_ms: &[f64],
    warmup: usize,
    target_tokens: usize,
) -> MoeSummary {
    let n_warm = warmup.min(decode_ms.len());
    let measured = &decode_ms[n_warm..];
    let measured_ffn = &ffn_rtt_ms[n_warm.min(ffn_rtt_ms.len())..];
    let n = measured.len();

    let (avg, p50, p99, tps, ffn, attn) = if n == 0 {
        (0.0, 0.0, 0.0, 0.0, None, None)
    } else {
        let (avg, p50, p99) = compute_percentiles(measured);
        let avg_ffn = if measured_ffn.len() == n {
            Some(measured_ffn.iter().sum::<f64>() / n as f64)
        } else {
            None
        };
        let avg_attn = avg_ffn.map(|f| (avg - f).max(0.0));
        (avg, p50, p99, 1000.0 / avg, avg_ffn, avg_attn)
    };

    let note = if n < target_tokens {
        format!("early stop @{}/{}", n, target_tokens)
    } else {
        String::new()
    };

    MoeSummary {
        avg_decode_ms: avg,
        p50_ms: p50,
        p99_ms: p99,
        tok_per_s: tps,
        ffn_rtt_ms: ffn,
        attn_ms: attn,
        n_steps: n,
        note,
    }
}

/// Run `args.concurrent` parallel MoE clients against the same shard map
/// and aggregate them into one row. `concurrent == 1` short-circuits to
/// `run_remote_moe_bench`.
pub(super) fn run_concurrent_moe(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    shards_str: &str,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    let n = args.concurrent.max(1);
    if n == 1 {
        return run_remote_moe_bench(vindex_path, args, shards_str);
    }

    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let vp = vindex_path.to_path_buf();
        let a = args.clone();
        let shards = shards_str.to_string();
        handles.push(std::thread::spawn(move || {
            run_remote_moe_bench(&vp, &a, &shards).map_err(|e| e.to_string())
        }));
    }
    let mut rows: Vec<BenchRow> = Vec::with_capacity(n);
    for h in handles {
        match h.join() {
            Ok(Ok(row)) => rows.push(row),
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                return Err(
                    "concurrent MoE bench worker panicked — see stderr for details".into(),
                );
            }
        }
    }
    Ok(combine_concurrent_rows(rows, n))
}

/// Bench the remote MoE expert path. Attention + router run locally; expert
/// blocks are dispatched to remote shards via `RemoteMoeBackend`.
///
/// Reports overall tok/s plus a breakdown:
///   expert-rtt  — time spent in remote expert dispatch per token
///   attn+       — remainder = local attn + router + dense FFN
pub(super) fn run_remote_moe_bench(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    shards_str: &str,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::ffn::moe_remote::RemoteMoeBackend;
    use larql_inference::{generate_with_remote_moe, generate_with_remote_moe_batch};

    let configs = parse_shard_segments(shards_str)?;
    let num_shards = configs.len();
    let backend = larql_compute::default_backend();
    eprintln!("Connecting to {} MoE shard(s)…", num_shards);
    let remote = RemoteMoeBackend::connect(configs)
        .map_err(|e| format!("failed to connect to MoE shards: {e}"))?;
    eprintln!("  Attention:  {} (local)", backend.name());
    eprintln!("  Router:     local");
    eprintln!(
        "  Experts:    remote  (sharded across {} endpoint{})",
        num_shards,
        if num_shards == 1 { "" } else { "s" }
    );

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

    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), &args.prompt)
            .unwrap_or_else(|_| args.prompt.clone());
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("tokenise: {e}"))?;

    let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(vindex_path);
    let max_tokens = args.warmup + args.tokens;
    let is_batch = args.moe_dispatch.trim() == "batch";
    let iters = args.moe_predispatch_iters.max(1);

    // Warmup.
    let run_once =
        |n: usize| -> Result<larql_inference::layer_graph::grid::GridGenerateResult, String> {
            if is_batch {
                generate_with_remote_moe_batch(
                    &weights,
                    &tokenizer,
                    prompt_ids.clone(),
                    n,
                    &index,
                    &remote,
                    &*backend,
                    &eos,
                    iters,
                )
                .map_err(|e| e.to_string())
            } else {
                generate_with_remote_moe(
                    &weights,
                    &tokenizer,
                    prompt_ids.clone(),
                    n,
                    &index,
                    &remote,
                    &*backend,
                    &eos,
                )
                .map_err(|e| e.to_string())
            }
        };

    let _ = run_once(args.warmup.max(1));

    let result = run_once(max_tokens).map_err(|e| format!("moe bench generate failed: {e}"))?;

    let summary = summarize_moe_result(
        &result.decode_ms,
        &result.ffn_rtt_ms,
        args.warmup,
        args.tokens,
    );

    Ok(BenchRow {
        backend: format_moe_backend_label(is_batch, num_shards),
        prefill_ms: 0.0,
        avg_decode_ms: summary.avg_decode_ms,
        p50_ms: summary.p50_ms,
        p99_ms: summary.p99_ms,
        tok_per_s: summary.tok_per_s,
        stages: None,
        ffn_rtt_ms: summary.ffn_rtt_ms,
        attn_ms: summary.attn_ms,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: summary.n_steps,
        note: summary.note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_shard_segments ─────────────────────────────────────────────

    #[test]
    fn parse_shard_segments_accepts_well_formed_map() {
        let cfgs =
            parse_shard_segments("0-31=http://a:8080, 32-63=http://b:8080").unwrap();
        assert_eq!(cfgs.len(), 2);
        assert_eq!(cfgs[0].start, 0);
        assert_eq!(cfgs[0].end, 31);
        assert_eq!(cfgs[0].url, "http://a:8080");
        assert_eq!(cfgs[1].start, 32);
        assert_eq!(cfgs[1].end, 63);
    }

    #[test]
    fn parse_shard_segments_skips_blank_segments() {
        let cfgs = parse_shard_segments("0-31=http://a, , 32-63=http://b").unwrap();
        assert_eq!(cfgs.len(), 2);
    }

    #[test]
    fn parse_shard_segments_rejects_missing_url() {
        let err = parse_shard_segments("0-31").unwrap_err();
        assert!(err.contains("missing URL"), "got: {err}");
    }

    #[test]
    fn parse_shard_segments_rejects_bad_range() {
        let err = parse_shard_segments("notarange=http://a").unwrap_err();
        assert!(err.contains("bad expert range"), "got: {err}");
    }

    #[test]
    fn parse_shard_segments_rejects_empty_spec() {
        let err = parse_shard_segments("").unwrap_err();
        assert!(err.contains("no valid shard segments"), "got: {err}");
        let err = parse_shard_segments(", ,").unwrap_err();
        assert!(err.contains("no valid shard segments"), "got: {err}");
    }

    // ── format_moe_backend_label ─────────────────────────────────────────

    #[test]
    fn label_picks_mode_and_shows_shard_count() {
        assert_eq!(format_moe_backend_label(true, 4), "remote-moe-batch (4 shards)");
        assert_eq!(
            format_moe_backend_label(false, 1),
            "remote-moe-stream (1 shards)"
        );
    }

    // ── summarize_moe_result ─────────────────────────────────────────────

    #[test]
    fn summarize_no_post_warmup_returns_zeros() {
        let s = summarize_moe_result(&[10.0, 10.0], &[], 5, 10);
        // warmup (5) > samples (2) → 0 measured.
        assert_eq!(s.n_steps, 0);
        assert_eq!(s.avg_decode_ms, 0.0);
        assert_eq!(s.tok_per_s, 0.0);
        assert!(s.ffn_rtt_ms.is_none());
        assert!(s.attn_ms.is_none());
        assert!(s.note.starts_with("early stop @0/"));
    }

    #[test]
    fn summarize_with_ffn_rtt_derives_attn() {
        let decode = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let ffn = vec![80.0, 80.0, 80.0, 80.0, 80.0];
        let s = summarize_moe_result(&decode, &ffn, 0, 5);
        assert_eq!(s.n_steps, 5);
        assert!((s.avg_decode_ms - 100.0).abs() < 1e-9);
        assert!((s.tok_per_s - 10.0).abs() < 1e-9);
        assert_eq!(s.ffn_rtt_ms, Some(80.0));
        // attn = decode - ffn = 20 ms.
        assert!(s.attn_ms.unwrap().abs() > 0.0);
        assert!((s.attn_ms.unwrap() - 20.0).abs() < 1e-9);
        assert!(s.note.is_empty(), "n == target so no early-stop note");
    }

    #[test]
    fn summarize_missing_ffn_rtt_leaves_none() {
        // ffn_rtt has fewer samples than decode (after warmup trim).
        let decode = vec![100.0, 100.0, 100.0];
        let ffn = vec![];
        let s = summarize_moe_result(&decode, &ffn, 0, 3);
        assert_eq!(s.n_steps, 3);
        assert!(s.ffn_rtt_ms.is_none());
        assert!(s.attn_ms.is_none());
    }

    #[test]
    fn summarize_clamps_negative_attn_to_zero() {
        // ffn > decode (sensor noise): attn must not go negative.
        let decode = vec![50.0];
        let ffn = vec![80.0];
        let s = summarize_moe_result(&decode, &ffn, 0, 1);
        assert_eq!(s.attn_ms, Some(0.0));
    }
}
