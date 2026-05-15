//! Internal bench result types + percentile helpers.
//!
//! `BenchRow` is the in-memory per-run summary every backend produces.
//! `BenchJsonResult` / `BenchJsonRow` / `BenchJsonLatency` are the ADR-0012
//! JSON shape committed to `bench/baselines/*.json`. They live here together
//! so that any change to the row layout is one file's review surface.

pub(crate) struct BenchRow {
    pub backend: String,
    pub prefill_ms: f64,
    pub avg_decode_ms: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub tok_per_s: f64,
    pub stages: Option<larql_inference::layer_graph::generate::StageTimings>,
    /// Remote FFN path breakdown: average FFN round-trip ms per token.
    pub ffn_rtt_ms: Option<f64>,
    /// Estimated local attention+norm+lmhead ms per token (= decode - ffn_rtt).
    pub attn_ms: Option<f64>,
    /// Wire bytes sent + received per decode token (remote FFN paths only).
    pub wire_bytes_per_tok: Option<u64>,
    /// `--bench-grid` only: tok/s scaling efficiency vs. the single-shard
    /// run (1.0 = perfect linear scaling). `None` for non-grid rows.
    pub shard_efficiency: Option<f64>,
    pub n_steps: usize,
    pub note: String,
}

/// Machine-readable JSON output schema (ADR-0012).
#[derive(serde::Serialize)]
pub(crate) struct BenchJsonResult {
    pub timestamp: String,
    pub model: String,
    pub prompt: String,
    pub tokens: usize,
    pub wire: Option<String>,
    pub concurrent: usize,
    pub results: Vec<BenchJsonRow>,
}

#[derive(serde::Serialize)]
pub(crate) struct BenchJsonRow {
    pub backend: String,
    pub prefill_ms: f64,
    #[serde(rename = "ms_per_tok")]
    pub ms_per_tok: BenchJsonLatency,
    pub tok_per_s: f64,
    pub wire_bytes_per_tok: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard_efficiency: Option<f64>,
    pub n_steps: usize,
    pub note: String,
}

#[derive(serde::Serialize)]
pub(crate) struct BenchJsonLatency {
    pub mean: f64,
    pub p50: f64,
    pub p99: f64,
}

// ── Percentile helpers ───────────────────────────────────────────────────────

/// Nearest-rank percentile on an already-sorted slice. `p` is in [0, 100].
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// Sort + summarise — returns (mean, p50, p99). Pure; safe to call from
/// every backend module without holding any other locks.
pub(crate) fn compute_percentiles(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p50 = percentile(&sorted, 50.0);
    let p99 = percentile(&sorted, 99.0);
    (mean, p50, p99)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_empty_returns_zero() {
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn percentile_picks_nearest_rank() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // 50% of 10 → idx 5 → value 6.0
        assert_eq!(percentile(&sorted, 50.0), 6.0);
        // 99% of 10 → idx 9 → value 10.0
        assert_eq!(percentile(&sorted, 99.0), 10.0);
        // 100% maps to idx 9 (clamped to len-1).
        assert_eq!(percentile(&sorted, 100.0), 10.0);
    }

    #[test]
    fn compute_percentiles_empty_returns_zeros() {
        assert_eq!(compute_percentiles(&[]), (0.0, 0.0, 0.0));
    }

    #[test]
    fn compute_percentiles_summarises_unsorted_input() {
        let values = [10.0, 1.0, 5.0, 3.0, 7.0];
        let (mean, p50, p99) = compute_percentiles(&values);
        assert!((mean - 5.2).abs() < 1e-9);
        // 50% of 5 → idx 2 in sorted [1, 3, 5, 7, 10] → 5
        assert_eq!(p50, 5.0);
        // 99% of 5 → idx 4 → 10
        assert_eq!(p99, 10.0);
    }
}
