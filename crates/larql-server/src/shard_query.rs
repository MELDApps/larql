//! Sharded vindex KNN service (Exp 53).
//!
//! Hosts a pre-compiled `(input, output)` cache at one or more layers and
//! serves remote KNN lookups over gRPC. When a query vector hits an
//! indexed entry (cosine ≥ tau), the server returns the matching MLP
//! output; otherwise the client falls back to local FFN compute.
//!
//! Wire surface lives in `larql_router_protocol::shard_proto`; this
//! module implements the `ShardService` trait + the in-memory cache
//! it consults. Cache loading from disk is out of scope for the
//! transport-layer port — operators seed via `ShardCache::insert_layer`
//! or `ShardCache::seed_from_normed` (used by tests) until a portable
//! on-disk format is specified in a follow-up ADR.

use std::collections::HashMap;
use std::sync::Arc;

use larql_router_protocol::{ShardQuery, ShardResult, ShardService};
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

/// Pre-normalized L2 fudge factor matching the Python prototype
/// (`q / (||q|| + 1e-12)`). Keeps the lookup deterministic on
/// zero-norm queries — they round-trip to a uniform vector that
/// fails the tau gate.
const NORM_EPS: f32 = 1e-12;

/// One layer's slice of the shard cache: L2-normalized inputs stored
/// row-major as a flat `Vec<f32>` (`n_entries × d`) plus the raw
/// outputs (also row-major).
///
/// Stored normalized so the per-query hot path only needs to normalize
/// the query vector once and run an `n_entries × d` matvec.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerEntry {
    pub inputs_normed: Vec<f32>,
    pub outputs: Vec<f32>,
    pub n_entries: usize,
    pub d: usize,
}

/// Outcome of `ShardCache::knn_lookup`. `mlp_out` is `None` on a miss
/// (either layer not indexed or best-sim below tau); `best_sim` is
/// reported on hit *and* miss for telemetry / threshold tuning.
#[derive(Clone, Debug, PartialEq)]
pub struct ShardLookup {
    pub mlp_out: Option<Vec<f32>>,
    pub best_sim: f32,
}

/// Errors surfaced when wiring entries into the cache.
#[derive(Clone, Debug, PartialEq)]
pub enum CacheError {
    /// `inputs.len() != n_entries * d`.
    InputShape { got: usize, want: usize },
    /// `outputs.len() != n_entries * d`.
    OutputShape { got: usize, want: usize },
    /// `d == 0` — empty hidden dim is nonsensical.
    ZeroDim,
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::InputShape { got, want } => {
                write!(f, "shard cache: inputs len {got} != n_entries × d = {want}")
            }
            CacheError::OutputShape { got, want } => write!(
                f,
                "shard cache: outputs len {got} != n_entries × d = {want}"
            ),
            CacheError::ZeroDim => write!(f, "shard cache: d must be > 0"),
        }
    }
}

impl std::error::Error for CacheError {}

/// In-memory KNN cache addressed by `layer_id`. One `LayerEntry` per
/// indexed layer; layers not present produce a miss without computing
/// similarities.
#[derive(Default, Debug)]
pub struct ShardCache {
    layers: HashMap<u32, LayerEntry>,
    tau: f32,
}

impl ShardCache {
    /// New empty cache with the configured cosine threshold. Mirrors
    /// the Python default of `0.97` — entries below this similarity
    /// are treated as misses.
    pub fn new(tau: f32) -> Self {
        Self {
            layers: HashMap::new(),
            tau,
        }
    }

    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// Number of indexed layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Number of cached entries at `layer_id`, or `None` when the
    /// layer is not indexed.
    pub fn layer_size(&self, layer_id: u32) -> Option<usize> {
        self.layers.get(&layer_id).map(|e| e.n_entries)
    }

    /// Insert a precompiled layer. Inputs are L2-normalized row-wise
    /// before storage; callers may pass raw (unnormalized) vectors.
    pub fn insert_layer(
        &mut self,
        layer_id: u32,
        inputs: &[f32],
        outputs: Vec<f32>,
        n_entries: usize,
        d: usize,
    ) -> Result<(), CacheError> {
        if d == 0 {
            return Err(CacheError::ZeroDim);
        }
        let want = n_entries.saturating_mul(d);
        if inputs.len() != want {
            return Err(CacheError::InputShape {
                got: inputs.len(),
                want,
            });
        }
        if outputs.len() != want {
            return Err(CacheError::OutputShape {
                got: outputs.len(),
                want,
            });
        }
        let inputs_normed = l2_normalize_rows(inputs, n_entries, d);
        self.layers.insert(
            layer_id,
            LayerEntry {
                inputs_normed,
                outputs,
                n_entries,
                d,
            },
        );
        Ok(())
    }

    /// Test-only seeder: skips re-normalization. Pre-condition: every
    /// row of `inputs_normed` already has L2-norm ≈ 1.0. Use for
    /// fixtures where the normalization step is not under test.
    #[doc(hidden)]
    pub fn seed_from_normed(
        &mut self,
        layer_id: u32,
        inputs_normed: Vec<f32>,
        outputs: Vec<f32>,
        n_entries: usize,
        d: usize,
    ) -> Result<(), CacheError> {
        if d == 0 {
            return Err(CacheError::ZeroDim);
        }
        let want = n_entries.saturating_mul(d);
        if inputs_normed.len() != want {
            return Err(CacheError::InputShape {
                got: inputs_normed.len(),
                want,
            });
        }
        if outputs.len() != want {
            return Err(CacheError::OutputShape {
                got: outputs.len(),
                want,
            });
        }
        self.layers.insert(
            layer_id,
            LayerEntry {
                inputs_normed,
                outputs,
                n_entries,
                d,
            },
        );
        Ok(())
    }

    /// KNN lookup mirroring `server.py:knn_lookup`. Normalizes the
    /// query, runs an `n_entries × d` matvec for cosine similarity,
    /// gates on `tau`, then either returns the single best output
    /// (`k == 1`) or a positive-cosine-weighted average of the top-k
    /// outputs.
    pub fn knn_lookup(&self, layer_id: u32, query: &[f32], k: usize, tau: f32) -> ShardLookup {
        let Some(layer) = self.layers.get(&layer_id) else {
            return ShardLookup {
                mlp_out: None,
                best_sim: 0.0,
            };
        };
        if query.len() != layer.d {
            // Caller's dim mismatches the index — same outcome as a
            // miss; surfacing an error here would force the wire path
            // to translate it into Status which loses the fast-fallback
            // behaviour the prototype relies on.
            return ShardLookup {
                mlp_out: None,
                best_sim: 0.0,
            };
        }

        let q_normed = l2_normalize(query);
        let sims = cosine_similarities(&layer.inputs_normed, &q_normed, layer.n_entries, layer.d);
        let (best_idx, best_sim) = argmax(&sims);
        if best_sim < tau {
            return ShardLookup {
                mlp_out: None,
                best_sim,
            };
        }

        let k = k.max(1).min(layer.n_entries);
        if k == 1 {
            let start = best_idx * layer.d;
            let mlp = layer.outputs[start..start + layer.d].to_vec();
            return ShardLookup {
                mlp_out: Some(mlp),
                best_sim,
            };
        }

        let mlp = weighted_topk_average(&sims, &layer.outputs, k, layer.d);
        ShardLookup {
            mlp_out: Some(mlp),
            best_sim,
        }
    }
}

// ── Pure math helpers ────────────────────────────────────────────────────────

/// L2-normalize a single vector. Adds `NORM_EPS` to the denominator to
/// match the Python prototype's zero-norm handling.
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt() + NORM_EPS;
    v.iter().map(|x| x / norm).collect()
}

/// L2-normalize each of `n` consecutive `d`-vectors stored in a single
/// row-major buffer.
fn l2_normalize_rows(rows: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * d);
    for i in 0..n {
        out.extend(l2_normalize(&rows[i * d..(i + 1) * d]));
    }
    out
}

/// Compute cosine similarities between an `n×d` row-major matrix of
/// pre-normalized rows and a single pre-normalized query vector.
fn cosine_similarities(rows_normed: &[f32], q_normed: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut sims = Vec::with_capacity(n);
    for i in 0..n {
        let row = &rows_normed[i * d..(i + 1) * d];
        let s: f32 = row.iter().zip(q_normed.iter()).map(|(a, b)| a * b).sum();
        sims.push(s);
    }
    sims
}

/// Return `(argmax_index, max_value)` of `sims`. Empty input → `(0, 0.0)`.
fn argmax(sims: &[f32]) -> (usize, f32) {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &s) in sims.iter().enumerate() {
        if s > best {
            best = s;
            best_idx = i;
        }
    }
    if best == f32::NEG_INFINITY {
        (0, 0.0)
    } else {
        (best_idx, best)
    }
}

/// Weighted average of the top-`k` rows of `outputs` (row-major, `d`
/// wide), weighted by their positive cosine similarities. Negative
/// sims are clipped to zero before weighting; when every selected
/// weight is zero, falls back to a uniform average.
fn weighted_topk_average(sims: &[f32], outputs: &[f32], k: usize, d: usize) -> Vec<f32> {
    let mut order: Vec<(usize, f32)> = sims.iter().copied().enumerate().collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<(usize, f32)> = order.into_iter().take(k).collect();

    let mut weights: Vec<f32> = top.iter().map(|(_, s)| s.max(0.0)).collect();
    let w_sum: f32 = weights.iter().sum();
    if w_sum > NORM_EPS {
        for w in &mut weights {
            *w /= w_sum;
        }
    } else {
        let uniform = 1.0 / (k as f32);
        for w in &mut weights {
            *w = uniform;
        }
    }

    let mut acc = vec![0.0f32; d];
    for ((idx, _), w) in top.iter().zip(weights.iter()) {
        let start = idx * d;
        for j in 0..d {
            acc[j] += outputs[start + j] * *w;
        }
    }
    acc
}

// ── Wire helpers ─────────────────────────────────────────────────────────────

/// Decode an `f32 LE bytes` payload into a `Vec<f32>`. Returns an
/// `InvalidArgument` Status when the byte length is not a multiple of 4
/// — clients are expected to round-trip the proto schema, so a wrong
/// length is a protocol violation.
pub fn decode_f32_le(bytes: &[u8]) -> Result<Vec<f32>, Status> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Status::invalid_argument(format!(
            "f32 payload length must be a multiple of 4, got {}",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Encode an `&[f32]` slice as little-endian bytes. Mirrors the wire
/// convention used by `ExpertService`.
pub fn encode_f32_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

// ── gRPC service ─────────────────────────────────────────────────────────────

/// Tonic service backed by a shared `ShardCache`. The cache lives
/// behind an `RwLock` so future Insert RPCs can mutate it under a
/// write lock without touching the read-only query path's contention
/// profile.
pub struct ShardGrpcService {
    pub cache: Arc<RwLock<ShardCache>>,
}

impl ShardGrpcService {
    pub fn new(cache: Arc<RwLock<ShardCache>>) -> Self {
        Self { cache }
    }
}

#[tonic::async_trait]
impl ShardService for ShardGrpcService {
    async fn query(&self, request: Request<ShardQuery>) -> Result<Response<ShardResult>, Status> {
        let req = request.into_inner();
        let query = decode_f32_le(&req.query_vec)?;

        let guard = self.cache.read().await;
        let tau = if req.tau_override > 0.0 {
            req.tau_override
        } else {
            guard.tau()
        };
        let lookup = guard.knn_lookup(req.layer_id, &query, req.k as usize, tau);
        drop(guard);

        let (hit, mlp_bytes) = match lookup.mlp_out {
            Some(out) => (true, encode_f32_le(&out)),
            None => (false, Vec::new()),
        };
        Ok(Response::new(ShardResult {
            hit,
            mlp_out: mlp_bytes,
            best_sim: lookup.best_sim,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Wire helpers ─────────────────────────────────────────────────────────

    #[test]
    fn encode_decode_f32_le_round_trips() {
        let values = vec![1.0, -2.5, 0.0, 3.1415927];
        let bytes = encode_f32_le(&values);
        assert_eq!(bytes.len(), values.len() * 4);
        let back = decode_f32_le(&bytes).unwrap();
        assert_eq!(back, values);
    }

    #[test]
    fn decode_f32_le_rejects_odd_byte_lengths() {
        let err = decode_f32_le(&[0u8; 7]).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    // ── Pure math ────────────────────────────────────────────────────────────

    #[test]
    fn l2_normalize_unit_vector_is_idempotent() {
        let v = vec![1.0f32, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 1.0).abs() < 1e-6);
        assert_eq!(n[1], 0.0);
    }

    #[test]
    fn l2_normalize_zero_vector_returns_zero() {
        // NORM_EPS guards the divide; result has all-zero numerator.
        let v = vec![0.0f32; 4];
        let n = l2_normalize(&v);
        assert!(n.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn l2_normalize_rows_normalizes_each_row_independently() {
        let rows = vec![1.0, 0.0, 0.0, 3.0, 4.0, 0.0];
        let n = l2_normalize_rows(&rows, 2, 3);
        // row 0 already unit
        assert!((n[0] - 1.0).abs() < 1e-6);
        // row 1: |v| = 5, expect (0.6, 0.8, 0.0)
        assert!((n[3] - 0.6).abs() < 1e-6);
        assert!((n[4] - 0.8).abs() < 1e-6);
        assert_eq!(n[5], 0.0);
    }

    #[test]
    fn cosine_similarities_match_dot_product_for_normed_inputs() {
        let rows = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let q = vec![1.0, 0.0, 0.0];
        let sims = cosine_similarities(&rows, &q, 2, 3);
        assert!((sims[0] - 1.0).abs() < 1e-6);
        assert!(sims[1].abs() < 1e-6);
    }

    #[test]
    fn argmax_handles_empty_input() {
        assert_eq!(argmax(&[]), (0, 0.0));
        assert_eq!(argmax(&[0.5, -0.2, 0.9, 0.1]), (2, 0.9));
    }

    #[test]
    fn weighted_topk_average_falls_back_to_uniform_when_all_negative() {
        let sims = vec![-0.5, -0.3, -0.1];
        let outputs = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]; // 3 rows of d=2
        let avg = weighted_topk_average(&sims, &outputs, 3, 2);
        // Uniform avg: (1+2+3)/3 = 2.0; (0+0+0)/3 = 0.0
        assert!((avg[0] - 2.0).abs() < 1e-6);
        assert!((avg[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_topk_average_uses_positive_cosine_weights() {
        let sims = vec![0.9, 0.1, -0.5];
        // d=1 outputs: [10, 20, 100]
        let outputs = vec![10.0, 20.0, 100.0];
        let avg = weighted_topk_average(&sims, &outputs, 3, 1);
        // weights from positive sims: [0.9, 0.1, 0.0] / 1.0
        let expected = 10.0 * 0.9 + 20.0 * 0.1 + 100.0 * 0.0;
        assert!((avg[0] - expected).abs() < 1e-5);
    }

    // ── Cache + lookup ───────────────────────────────────────────────────────

    fn cache_with_two_entries(d: usize, tau: f32) -> ShardCache {
        // Layer 26 with two entries at d=4:
        //   row 0: [1, 0, 0, 0] → output [10, 20, 30, 40]
        //   row 1: [0, 1, 0, 0] → output [-1, -2, -3, -4]
        let inputs_normed = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let outputs = vec![10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0];
        let mut cache = ShardCache::new(tau);
        cache
            .seed_from_normed(26, inputs_normed, outputs, 2, d)
            .unwrap();
        cache
    }

    #[test]
    fn knn_lookup_hit_returns_argmax_output_when_k_is_one() {
        let cache = cache_with_two_entries(4, 0.97);
        // Query close to row 0 → expect outputs[0..4].
        let out = cache.knn_lookup(26, &[1.0, 0.0, 0.0, 0.0], 1, 0.97);
        let mlp = out.mlp_out.expect("hit");
        assert_eq!(mlp, vec![10.0, 20.0, 30.0, 40.0]);
        assert!((out.best_sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn knn_lookup_miss_when_below_tau() {
        let cache = cache_with_two_entries(4, 0.97);
        // Query orthogonal to both rows → best_sim ≈ 0 < tau.
        let out = cache.knn_lookup(26, &[0.0, 0.0, 1.0, 0.0], 1, 0.97);
        assert!(out.mlp_out.is_none());
        assert!(out.best_sim < 0.97);
    }

    #[test]
    fn knn_lookup_unknown_layer_is_a_miss() {
        let cache = cache_with_two_entries(4, 0.97);
        let out = cache.knn_lookup(99, &[1.0, 0.0, 0.0, 0.0], 1, 0.97);
        assert!(out.mlp_out.is_none());
        assert_eq!(out.best_sim, 0.0);
    }

    #[test]
    fn knn_lookup_dim_mismatch_is_a_miss() {
        let cache = cache_with_two_entries(4, 0.97);
        let out = cache.knn_lookup(26, &[1.0, 0.0, 0.0], 1, 0.97);
        assert!(out.mlp_out.is_none());
    }

    #[test]
    fn knn_lookup_k_greater_than_one_averages_top_k() {
        // Build a cache where the top-2 are tied at cos = 0.7071…
        //   row 0: [1, 0] output [10, 0]
        //   row 1: [0, 1] output [ 0, 10]
        // Query [1/√2, 1/√2] hits both with equal weight; average is
        // (5, 5).
        let mut cache = ShardCache::new(0.5);
        cache
            .seed_from_normed(
                0,
                vec![1.0, 0.0, 0.0, 1.0],
                vec![10.0, 0.0, 0.0, 10.0],
                2,
                2,
            )
            .unwrap();
        let q = vec![1.0 / 2f32.sqrt(), 1.0 / 2f32.sqrt()];
        let out = cache.knn_lookup(0, &q, 2, 0.5);
        let mlp = out.mlp_out.expect("hit");
        assert!((mlp[0] - 5.0).abs() < 1e-5);
        assert!((mlp[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn tau_override_can_force_hit_or_miss() {
        let cache = cache_with_two_entries(4, 0.5);
        // Query at cos = 0.7071 to both rows after normalization
        // (proportional to [1, 1, 0, 0]).
        let q = vec![1.0, 1.0, 0.0, 0.0];
        // tau = 0.5 → hit.
        assert!(cache.knn_lookup(26, &q, 1, 0.5).mlp_out.is_some());
        // tau = 0.99 → miss even though argmax is the same.
        assert!(cache.knn_lookup(26, &q, 1, 0.99).mlp_out.is_none());
    }

    #[test]
    fn insert_layer_validates_shape() {
        let mut cache = ShardCache::new(0.97);
        let err = cache
            .insert_layer(0, &[1.0, 0.0], vec![1.0, 0.0, 0.0], 1, 2)
            .unwrap_err();
        assert!(matches!(err, CacheError::OutputShape { .. }));

        let err = cache
            .insert_layer(0, &[1.0, 0.0, 0.0], vec![1.0, 0.0], 1, 2)
            .unwrap_err();
        assert!(matches!(err, CacheError::InputShape { .. }));

        let err = cache.insert_layer(0, &[], vec![], 0, 0).unwrap_err();
        assert!(matches!(err, CacheError::ZeroDim));
    }

    #[test]
    fn insert_layer_normalizes_unit_inputs() {
        let mut cache = ShardCache::new(0.97);
        cache
            .insert_layer(7, &[3.0, 4.0], vec![1.0, 1.0], 1, 2)
            .unwrap();
        // Direction-equal query at the same row → hit at cos = 1.
        let out = cache.knn_lookup(7, &[6.0, 8.0], 1, 0.97);
        assert!(out.mlp_out.is_some());
        assert!((out.best_sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn accessors_report_cache_shape() {
        let mut cache = ShardCache::new(0.5);
        assert!(cache.is_empty());
        cache
            .insert_layer(0, &[1.0, 0.0], vec![1.0, 1.0], 1, 2)
            .unwrap();
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.layer_size(0), Some(1));
        assert_eq!(cache.layer_size(99), None);
        assert!((cache.tau() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn cache_error_display_includes_lengths() {
        let e = CacheError::InputShape { got: 3, want: 4 };
        let s = format!("{e}");
        assert!(s.contains("3") && s.contains("4"));
        let e = CacheError::OutputShape { got: 2, want: 6 };
        let s = format!("{e}");
        assert!(s.contains("2") && s.contains("6"));
        let s = format!("{}", CacheError::ZeroDim);
        assert!(s.contains("d must"));
    }

    // ── gRPC handler (exercised in-process) ──────────────────────────────────

    #[tokio::test]
    async fn grpc_query_returns_hit_on_matching_vector() {
        let cache = Arc::new(RwLock::new(cache_with_two_entries(4, 0.97)));
        let svc = ShardGrpcService::new(cache);
        let req = Request::new(ShardQuery {
            layer_id: 26,
            k: 1,
            query_vec: encode_f32_le(&[1.0, 0.0, 0.0, 0.0]),
            tau_override: 0.0,
        });
        let resp = svc.query(req).await.unwrap().into_inner();
        assert!(resp.hit);
        let mlp = decode_f32_le(&resp.mlp_out).unwrap();
        assert_eq!(mlp, vec![10.0, 20.0, 30.0, 40.0]);
        assert!((resp.best_sim - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn grpc_query_returns_miss_when_below_tau() {
        let cache = Arc::new(RwLock::new(cache_with_two_entries(4, 0.97)));
        let svc = ShardGrpcService::new(cache);
        let req = Request::new(ShardQuery {
            layer_id: 26,
            k: 1,
            query_vec: encode_f32_le(&[0.0, 0.0, 1.0, 0.0]),
            tau_override: 0.0,
        });
        let resp = svc.query(req).await.unwrap().into_inner();
        assert!(!resp.hit);
        assert!(resp.mlp_out.is_empty());
        assert!(resp.best_sim < 0.97);
    }

    #[tokio::test]
    async fn grpc_tau_override_takes_precedence() {
        let cache = Arc::new(RwLock::new(cache_with_two_entries(4, 0.5)));
        let svc = ShardGrpcService::new(cache);
        // Query [1, 1, 0, 0] hits at cos = 0.7071. tau_override = 0.99 → miss.
        let req = Request::new(ShardQuery {
            layer_id: 26,
            k: 1,
            query_vec: encode_f32_le(&[1.0, 1.0, 0.0, 0.0]),
            tau_override: 0.99,
        });
        let resp = svc.query(req).await.unwrap().into_inner();
        assert!(!resp.hit);
    }

    #[tokio::test]
    async fn grpc_rejects_malformed_query_bytes() {
        let cache = Arc::new(RwLock::new(ShardCache::new(0.97)));
        let svc = ShardGrpcService::new(cache);
        let req = Request::new(ShardQuery {
            layer_id: 0,
            k: 1,
            query_vec: vec![0u8; 7], // not a multiple of 4
            tau_override: 0.0,
        });
        let err = svc.query(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }
}
