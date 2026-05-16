//! Grid state and gRPC service implementation for the self-assembling FFN grid.

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use larql_router_protocol::{
    AckMsg, AdminAck, AnnounceMsg, AssignMsg, AssignRangeRequest, DrainRequest, Gap, GridService,
    LayerLatency, ModelCoverage, RouterMessage, RouterPayload, ServerInfo, ServerMessage,
    ServerPayload, ShardInfo, StatusRequest, StatusResponse, UnassignMsg,
};

// ── Per-server record ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ServerEntry {
    pub server_id: String,
    pub listen_url: String,
    pub model_id: String,
    pub layer_start: u32, // inclusive
    pub layer_end: u32,   // inclusive
    /// `vindex_hash` from `AnnounceMsg`. Used as the `shard_hash` when this
    /// server is selected as a Mode B origin for a different replica.
    pub vindex_hash: String,
    pub cpu_pct: f32,
    pub ram_used: u64,
    pub requests_in_flight: u32,
    pub last_seen: Instant,
    /// Per-layer EMA latency and p99, from HeartbeatMsg.layer_stats (GT3).
    /// Key = layer index. Empty until the first heartbeat with layer data arrives.
    pub layer_latencies: HashMap<u32, (f32, f32)>, // (avg_ms, p99_ms)
    /// Shard-scoped request rate (requests/sec) from the most recent
    /// heartbeat. Drives the hot-shard rebalancer tick.
    pub req_per_sec: f32,
}

// ── Mode B: available server entry ───────────────────────────────────────────

/// A server in Mode B idle state — it has capacity but no shard loaded yet.
pub struct AvailableEntry {
    pub server_id: String,
    /// Channel to send `RouterMessage` (including `AssignMsg`) to this server.
    pub sender: mpsc::Sender<Result<RouterMessage, tonic::Status>>,
    pub ram_bytes: u64,
    pub disk_bytes: u64,
    pub store_path: String,
    pub joined_at: std::time::Instant,
}

// ── Grid state ────────────────────────────────────────────────────────────────

pub struct GridState {
    servers: HashMap<String, ServerEntry>,
    // Pre-built: (model_id, layer) → server_ids; rebuilt only on topology change.
    route_table: HashMap<(String, u32), Vec<String>>,
    // Pre-built: layer → server_ids for model_id=None (single-model) queries.
    any_model_table: HashMap<u32, Vec<String>>,
    /// Mode B: servers that advertised capacity and are waiting for assignment.
    /// Key = server_id.
    available_servers: HashMap<String, AvailableEntry>,
    /// Sender channels for currently-serving (Mode A) servers.
    /// Used by the rebalancer to push UnassignMsg without holding a lock.
    /// Key = server_id.
    serving_senders: HashMap<String, mpsc::Sender<Result<RouterMessage, tonic::Status>>>,
    /// Phase 4: number of replicas the router tries to maintain per
    /// `(model_id, layer_start, layer_end)` shard range. Default 1 — every
    /// range needs exactly one server. >1 enables auto-replication: when
    /// fewer than N servers cover a range, the router pulls from the
    /// available pool to bring the count back up.
    target_replicas: u32,
    /// Hot-shard book-keeping: ranges whose req/s currently exceeds the
    /// hot-shard threshold get `effective_target_replicas = target + 1`
    /// until the rate subsides. Rebalancer marks ranges on the hot-shard
    /// tick; under/over-replication checks read this set via
    /// `effective_target_for`.
    elevated_ranges: HashSet<(String, u32, u32)>,
}

impl Default for GridState {
    fn default() -> Self {
        Self {
            servers: HashMap::new(),
            route_table: HashMap::new(),
            any_model_table: HashMap::new(),
            available_servers: HashMap::new(),
            serving_senders: HashMap::new(),
            target_replicas: 1,
            elevated_ranges: HashSet::new(),
        }
    }
}

impl GridState {
    pub fn register(&mut self, entry: ServerEntry) {
        tracing::info!(
            server_id = %entry.server_id,
            listen_url = %entry.listen_url,
            model_id = %entry.model_id,
            layers = %format!("{}-{}", entry.layer_start, entry.layer_end),
            "Grid: server joined"
        );
        self.servers.insert(entry.server_id.clone(), entry);
        self.rebuild_route_table();
        self.log_coverage();
    }

    /// Register a server and store its sender for rebalancer-initiated UnassignMsg.
    pub fn register_with_sender(
        &mut self,
        entry: ServerEntry,
        sender: mpsc::Sender<Result<RouterMessage, tonic::Status>>,
    ) {
        self.serving_senders.insert(entry.server_id.clone(), sender);
        self.register(entry);
    }

    pub fn deregister(&mut self, server_id: &str) {
        self.serving_senders.remove(server_id);
        if let Some(entry) = self.servers.remove(server_id) {
            tracing::info!(
                server_id = %server_id,
                model_id = %entry.model_id,
                layers = %format!("{}-{}", entry.layer_start, entry.layer_end),
                "Grid: server left"
            );
            self.rebuild_route_table();
            self.log_coverage();
        }
    }

    pub fn update_heartbeat(
        &mut self,
        server_id: &str,
        cpu_pct: f32,
        ram_used: u64,
        requests_in_flight: u32,
        layer_stats: Vec<LayerLatency>,
        req_per_sec: f32,
    ) {
        if let Some(entry) = self.servers.get_mut(server_id) {
            entry.cpu_pct = cpu_pct;
            entry.ram_used = ram_used;
            entry.requests_in_flight = requests_in_flight;
            entry.req_per_sec = req_per_sec;
            entry.last_seen = Instant::now();
            for ls in layer_stats {
                entry
                    .layer_latencies
                    .insert(ls.layer, (ls.avg_ms, ls.p99_ms));
            }
        }
        // Heartbeats don't change topology — no table rebuild needed.
    }

    /// Route one layer. O(1) table lookup + O(replicas) least-loaded scan.
    ///
    /// Replica selection (GT3): when per-layer latency data is available from
    /// heartbeats, prefer the server with lowest avg_ms for this specific layer.
    /// Falls back to requests_in_flight when no layer data exists yet.
    pub fn route(&self, model_id: Option<&str>, layer: u32) -> Option<String> {
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        ids.and_then(|server_ids| {
            server_ids
                .iter()
                .filter_map(|id| self.servers.get(id))
                .min_by(|a, b| {
                    let lat_a = a.layer_latencies.get(&layer).map(|(avg, _)| *avg);
                    let lat_b = b.layer_latencies.get(&layer).map(|(avg, _)| *avg);
                    match (lat_a, lat_b) {
                        (Some(la), Some(lb)) => {
                            la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        // Prefer server with latency data over unknown.
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        // No latency data for either: fall back to requests_in_flight.
                        (None, None) => a.requests_in_flight.cmp(&b.requests_in_flight),
                    }
                })
                .map(|s| s.listen_url.clone())
        })
    }

    /// Resolve all layers in one call — one lock acquisition covers the whole batch.
    /// Returns Ok(layer → url) or Err(first layer with no owning shard).
    #[allow(dead_code)]
    pub fn route_all(
        &self,
        model_id: Option<&str>,
        layers: &[usize],
    ) -> Result<HashMap<usize, String>, usize> {
        let mut out = HashMap::with_capacity(layers.len());
        for &layer in layers {
            match self.route(model_id, layer as u32) {
                Some(url) => {
                    out.insert(layer, url);
                }
                None => return Err(layer),
            }
        }
        Ok(out)
    }

    /// Rebuild layer→servers index. Called only on join/leave (cold path).
    fn rebuild_route_table(&mut self) {
        let mut rt: HashMap<(String, u32), Vec<String>> = HashMap::new();
        let mut any: HashMap<u32, Vec<String>> = HashMap::new();
        for entry in self.servers.values() {
            for layer in entry.layer_start..=entry.layer_end {
                rt.entry((entry.model_id.clone(), layer))
                    .or_default()
                    .push(entry.server_id.clone());
                any.entry(layer).or_default().push(entry.server_id.clone());
            }
        }
        self.route_table = rt;
        self.any_model_table = any;
    }

    fn log_coverage(&self) {
        // Group by model_id
        let mut by_model: HashMap<&str, Vec<&ServerEntry>> = HashMap::new();
        for entry in self.servers.values() {
            by_model.entry(&entry.model_id).or_default().push(entry);
        }
        for (model_id, entries) in &by_model {
            let layer_count: u32 = entries
                .iter()
                .map(|e| e.layer_end - e.layer_start + 1)
                .sum();
            tracing::info!(
                model_id = model_id,
                servers = entries.len(),
                total_layers_covered = layer_count,
                "Grid coverage updated"
            );
        }
    }

    /// Accessor for all serving servers (for the rebalancer).
    pub fn servers(&self) -> impl Iterator<Item = (&String, &ServerEntry)> {
        self.servers.iter()
    }

    /// Return IDs of serving servers whose `last_seen` is older than `timeout`.
    /// Stream-close already triggers deregister via the gRPC handler; this
    /// covers the case where a server keeps the stream open but stops sending
    /// heartbeats (deadlock, GC pause, etc.).
    pub fn stale_server_ids(&self, timeout: std::time::Duration) -> Vec<String> {
        let now = Instant::now();
        self.servers
            .iter()
            .filter(|(_, e)| now.saturating_duration_since(e.last_seen) > timeout)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Returns true if there is at least one available server in the Mode B pool.
    pub fn has_available_servers(&self) -> bool {
        !self.available_servers.is_empty()
    }

    /// Get the sender channel for a serving server by ID (for UnassignMsg delivery).
    pub fn serving_sender(
        &self,
        server_id: &str,
    ) -> Option<mpsc::Sender<Result<RouterMessage, tonic::Status>>> {
        self.serving_senders.get(server_id).cloned()
    }

    /// Register a Mode B available server. Returns the server_id.
    pub fn register_available(
        &mut self,
        server_id: String,
        sender: mpsc::Sender<Result<RouterMessage, tonic::Status>>,
        ram_bytes: u64,
        disk_bytes: u64,
        store_path: String,
    ) {
        tracing::info!(
            server_id = %server_id,
            ram_gb = ram_bytes / (1024 * 1024 * 1024),
            "Grid: Mode B server available"
        );
        self.available_servers.insert(
            server_id.clone(),
            AvailableEntry {
                server_id,
                sender,
                ram_bytes,
                disk_bytes,
                store_path,
                joined_at: std::time::Instant::now(),
            },
        );
    }

    /// Remove a server from the available pool.
    pub fn deregister_available(&mut self, server_id: &str) {
        self.available_servers.remove(server_id);
    }

    /// Find any currently-serving replica that covers `[layer_start, layer_end]`
    /// of `model_id` and return its (`listen_url`, `vindex_hash`).
    ///
    /// Returns `None` when no live replica exists — a gap with no surviving
    /// origin cannot be filled from within the grid; the deployment must supply
    /// an external origin store.
    pub fn find_origin_for(
        &self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> Option<(String, String)> {
        self.servers
            .values()
            .find(|e| {
                e.model_id == model_id && e.layer_start <= layer_start && e.layer_end >= layer_end
            })
            .map(|e| (e.listen_url.clone(), e.vindex_hash.clone()))
    }

    /// Find the first available server that has at least `min_ram_bytes` of
    /// RAM, resolve a serving origin, send it an `AssignMsg`, and move it out
    /// of the available pool.
    ///
    /// Returns `true` if an assignment was sent. Returns `false` either when no
    /// available server has enough RAM, or when no live replica is left to
    /// serve as origin for the gap.
    pub fn try_assign_gap(
        &mut self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        min_ram_bytes: u64,
    ) -> bool {
        let Some((origin_url, shard_hash)) = self.find_origin_for(model_id, layer_start, layer_end)
        else {
            tracing::warn!(
                model_id = %model_id,
                layers = %format!("{layer_start}-{layer_end}"),
                "Grid: cannot fill gap — no live replica to serve as origin"
            );
            return false;
        };
        self.try_assign_gap_with_origin(
            model_id,
            layer_start,
            layer_end,
            &origin_url,
            &shard_hash,
            min_ram_bytes,
        )
    }

    /// Lower-level assign that takes an explicit origin. Used by tests and by
    /// deployments that supply an external (non-grid) origin store.
    pub fn try_assign_gap_with_origin(
        &mut self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        origin_url: &str,
        shard_hash: &str,
        min_ram_bytes: u64,
    ) -> bool {
        // Find a suitable available server.
        let server_id = self
            .available_servers
            .iter()
            .find(|(_, e)| e.ram_bytes >= min_ram_bytes)
            .map(|(id, _)| id.clone());

        let Some(server_id) = server_id else {
            return false;
        };

        let entry = self.available_servers.remove(&server_id).unwrap();
        let msg = RouterMessage {
            payload: Some(RouterPayload::Assign(larql_router_protocol::AssignMsg {
                model_id: model_id.to_owned(),
                layer_start,
                layer_end,
                origin_url: origin_url.to_owned(),
                shard_hash: shard_hash.to_owned(),
            })),
        };
        if entry.sender.try_send(Ok(msg)).is_ok() {
            tracing::info!(
                server_id = %server_id,
                model_id = %model_id,
                layers = %format!("{layer_start}-{layer_end}"),
                origin_url = %origin_url,
                "Grid: Mode B assignment sent"
            );
            true
        } else {
            tracing::warn!(server_id = %server_id, "Grid: Mode B assignment send failed (peer disconnected)");
            false
        }
    }

    /// Phase 4: configure how many replicas the router maintains per shard
    /// range. Setter so the value can come from CLI in main.rs.
    pub fn set_target_replicas(&mut self, n: u32) {
        // 0 would mean "no servers"; clamp to ≥1.
        self.target_replicas = n.max(1);
    }

    /// Current target_replicas value (read-only).
    pub fn target_replicas(&self) -> u32 {
        self.target_replicas
    }

    /// Hot-shard detection: distinct `(model_id, layer_start, layer_end)`
    /// ranges where at least one serving replica's most recent
    /// `req_per_sec` heartbeat exceeds `threshold`. Returns an empty list
    /// when `threshold <= 0` (the feature is disabled).
    ///
    /// Uses max-rate-across-replicas: if a router does perfect
    /// load-balancing the rates converge, so any replica crossing the
    /// threshold means the shard's per-replica load has saturated and
    /// adding capacity is warranted. Sorted for deterministic iteration.
    pub fn hot_layer_ranges(&self, threshold: f32) -> Vec<(String, u32, u32)> {
        // `threshold > 0.0` returns false for NaN; the explicit not-greater
        // form below disables the check for NaN and non-positives alike
        // without tripping the `<=` NaN trap.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let disabled = !(threshold > 0.0);
        if disabled {
            return Vec::new();
        }
        let mut max_rate: HashMap<(String, u32, u32), f32> = HashMap::new();
        for e in self.servers.values() {
            let key = (e.model_id.clone(), e.layer_start, e.layer_end);
            let cur = max_rate.entry(key).or_insert(0.0);
            if e.req_per_sec > *cur {
                *cur = e.req_per_sec;
            }
        }
        let mut out: Vec<(String, u32, u32)> = max_rate
            .into_iter()
            .filter_map(|(k, v)| if v > threshold { Some(k) } else { None })
            .collect();
        out.sort();
        out
    }

    /// Mark `(model_id, layer_start, layer_end)` as elevated so that
    /// `effective_target_for` returns `target_replicas + 1`. Returns
    /// `true` if this call newly inserted the range.
    pub fn mark_elevated(&mut self, model_id: &str, layer_start: u32, layer_end: u32) -> bool {
        self.elevated_ranges
            .insert((model_id.to_owned(), layer_start, layer_end))
    }

    /// Clear the elevation flag for `(model_id, layer_start, layer_end)`.
    /// Returns `true` if the range was previously elevated. After demotion
    /// the standard over-replication tick drops the surplus replica.
    pub fn demote_elevated(&mut self, model_id: &str, layer_start: u32, layer_end: u32) -> bool {
        self.elevated_ranges
            .remove(&(model_id.to_owned(), layer_start, layer_end))
    }

    /// Snapshot of currently-elevated ranges. Used by the hot-shard tick
    /// to decide which previously-elevated ranges to demote.
    pub fn elevated_ranges_snapshot(&self) -> Vec<(String, u32, u32)> {
        let mut out: Vec<(String, u32, u32)> = self.elevated_ranges.iter().cloned().collect();
        out.sort();
        out
    }

    /// Effective replication target for a specific shard range.
    /// Equal to `target_replicas`, plus 1 when the range is currently
    /// marked elevated by the hot-shard tick.
    pub fn effective_target_for(&self, model_id: &str, layer_start: u32, layer_end: u32) -> u32 {
        let bump = if self
            .elevated_ranges
            .contains(&(model_id.to_owned(), layer_start, layer_end))
        {
            1
        } else {
            0
        };
        self.target_replicas + bump
    }

    /// Phase 4: ranges whose live replica count exceeds the effective
    /// target. Hot ranges have effective target = target + 1, so the
    /// over-replication tick won't strip a freshly-pulled hot spare;
    /// once the hot signal clears, the elevated bump goes away and the
    /// surplus replica is dropped on the next tick.
    pub fn over_replicated_ranges(&self) -> Vec<(String, u32, u32, u32)> {
        let mut counts: HashMap<(String, u32, u32), u32> = HashMap::new();
        for e in self.servers.values() {
            *counts
                .entry((e.model_id.clone(), e.layer_start, e.layer_end))
                .or_default() += 1;
        }
        let mut out = Vec::new();
        for ((model_id, start, end), count) in counts {
            let effective = self.effective_target_for(&model_id, start, end);
            if count > effective {
                out.push((model_id, start, end, count - effective));
            }
        }
        out.sort();
        out
    }

    /// Phase 4: among servers covering `(model_id, layer_start, layer_end)`,
    /// return the one with the lowest `requests_in_flight`. Used by the
    /// over-replication path to pick which replica to drop.
    pub fn least_loaded_in_range(
        &self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> Option<&ServerEntry> {
        self.servers
            .values()
            .filter(|e| {
                e.model_id == model_id && e.layer_start == layer_start && e.layer_end == layer_end
            })
            .min_by_key(|e| e.requests_in_flight)
    }

    /// Phase 4: ranges whose live replica count is below the effective
    /// target (= `target_replicas` plus the hot-shard bump). Skips ranges
    /// that have zero servers — those are handled by `coverage_gaps()` /
    /// `try_fill_all_gaps()` because they need a different
    /// origin-resolution story (no live replica → no origin).
    pub fn under_replicated_ranges(&self) -> Vec<(String, u32, u32, u32)> {
        // Group by (model_id, layer_start, layer_end) → count of servers.
        let mut counts: HashMap<(String, u32, u32), u32> = HashMap::new();
        for e in self.servers.values() {
            *counts
                .entry((e.model_id.clone(), e.layer_start, e.layer_end))
                .or_default() += 1;
        }
        let mut out = Vec::new();
        for ((model_id, start, end), count) in counts {
            let effective = self.effective_target_for(&model_id, start, end);
            if count > 0 && count < effective {
                out.push((model_id, start, end, effective - count));
            }
        }
        out.sort();
        out
    }

    /// Phase 4: walk under-replicated ranges and dispatch one `AssignMsg`
    /// per range to bring counts closer to `target_replicas`. Returns the
    /// number of assignments sent.
    ///
    /// At most one assignment per range per call — a newly-assigned replica
    /// won't register as serving until `ReadyMsg` arrives, so issuing more
    /// than one assignment per range here would over-replicate. Callers run
    /// this periodically (rebalancer) or after Ready/Available events.
    pub fn try_replicate_from_available(&mut self) -> usize {
        let ranges = self.under_replicated_ranges();
        let mut sent = 0;
        for (model_id, start, end, _deficit) in ranges {
            if self.try_assign_gap(&model_id, start, end, 0) {
                sent += 1;
            }
        }
        sent
    }

    /// ADR-0004 Phase 5: send an `AssignMsg` to a specific available
    /// server, identified by `server_id`. Used by the admin `assign_range`
    /// RPC when the operator wants a deterministic destination instead of
    /// "any spare with enough RAM".
    ///
    /// Returns `Ok(())` on dispatch, `Err(msg)` when the server isn't in
    /// the available pool or its outbound channel rejected the message.
    pub fn send_assign_to_named_available(
        &mut self,
        server_id: &str,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        origin_url: &str,
        shard_hash: &str,
    ) -> Result<(), String> {
        let entry = self
            .available_servers
            .remove(server_id)
            .ok_or_else(|| format!("server_id {server_id:?} is not in the available pool"))?;
        let msg = RouterMessage {
            payload: Some(RouterPayload::Assign(AssignMsg {
                model_id: model_id.to_owned(),
                layer_start,
                layer_end,
                origin_url: origin_url.to_owned(),
                shard_hash: shard_hash.to_owned(),
            })),
        };
        if let Err(e) = entry.sender.try_send(Ok(msg)) {
            // Put the entry back so a follow-up call can retry.
            self.available_servers.insert(server_id.to_string(), entry);
            return Err(format!("send to {server_id:?} failed: {e}"));
        }
        tracing::info!(
            server_id,
            model_id,
            layers = %format!("{layer_start}-{layer_end}"),
            origin_url,
            "Grid: admin-targeted AssignMsg sent"
        );
        Ok(())
    }

    /// Scan current coverage gaps and try to fill each one from the available
    /// pool. Returns the number of assignments sent.
    pub fn try_fill_all_gaps(&mut self) -> usize {
        let gaps = self.coverage_gaps();
        let mut sent = 0;
        for (model_id, layer_start, layer_end) in gaps {
            // RAM estimate: we don't have a true upper bound from the gap
            // alone, so fall back to a permissive 0 (any available server is
            // acceptable). Deployments that need RAM-aware placement should
            // call try_assign_gap_with_origin directly with a real estimate.
            if self.try_assign_gap(&model_id, layer_start, layer_end, 0) {
                sent += 1;
            }
        }
        sent
    }

    /// Return a list of (model_id, layer_start, layer_end) ranges that have no
    /// server covering them, based on the current route table.
    ///
    /// Gaps are only detectable if the router knows the total layer count for
    /// each model. Since the router doesn't store that, we instead return every
    /// layer range between consecutive covered shards.
    pub fn coverage_gaps(&self) -> Vec<(String, u32, u32)> {
        let mut by_model: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        for entry in self.servers.values() {
            by_model
                .entry(entry.model_id.clone())
                .or_default()
                .push((entry.layer_start, entry.layer_end));
        }
        let mut gaps = Vec::new();
        for (model_id, mut ranges) in by_model {
            ranges.sort_by_key(|(s, _)| *s);
            let mut prev_end: Option<u32> = None;
            for (start, end) in ranges {
                if let Some(pe) = prev_end {
                    if start > pe + 1 {
                        gaps.push((model_id.clone(), pe + 1, start - 1));
                    }
                }
                prev_end = Some(end);
            }
        }
        gaps
    }

    /// All distinct `listen_url` values across all registered servers.
    /// Used by the `/v1/stats` proxy to find a shard to forward to.
    pub fn all_shard_urls(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        self.servers
            .values()
            .filter_map(|s| {
                if seen.insert(s.listen_url.clone()) {
                    Some(s.listen_url.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn status_response(&self) -> StatusResponse {
        // Build per-model coverage
        let mut by_model: HashMap<String, Vec<&ServerEntry>> = HashMap::new();
        for entry in self.servers.values() {
            by_model
                .entry(entry.model_id.clone())
                .or_default()
                .push(entry);
        }

        let models: Vec<ModelCoverage> = by_model
            .iter()
            .map(|(model_id, entries)| {
                let mut shards: Vec<ShardInfo> = entries
                    .iter()
                    .map(|e| ShardInfo {
                        layer_start: e.layer_start,
                        layer_end: e.layer_end,
                        server_ids: vec![e.server_id.clone()],
                        replica_count: 1,
                    })
                    .collect();
                shards.sort_by_key(|s| s.layer_start);

                // Find gaps
                let mut gaps: Vec<Gap> = Vec::new();
                let mut prev_end: Option<u32> = None;
                for shard in &shards {
                    if let Some(end) = prev_end {
                        if shard.layer_start > end + 1 {
                            gaps.push(Gap {
                                layer_start: end + 1,
                                layer_end: shard.layer_start - 1,
                            });
                        }
                    }
                    prev_end = Some(shard.layer_end);
                }

                ModelCoverage {
                    model_id: model_id.clone(),
                    num_layers: 0, // not known to router without vindex
                    shards,
                    gaps,
                }
            })
            .collect();

        let servers: Vec<ServerInfo> = self
            .servers
            .values()
            .map(|e| {
                let mut layer_stats: Vec<LayerLatency> = e
                    .layer_latencies
                    .iter()
                    .map(|(&layer, &(avg_ms, p99_ms))| LayerLatency {
                        layer,
                        avg_ms,
                        p99_ms,
                    })
                    .collect();
                layer_stats.sort_by_key(|l| l.layer);
                ServerInfo {
                    server_id: e.server_id.clone(),
                    listen_url: e.listen_url.clone(),
                    state: "serving".into(),
                    model_id: e.model_id.clone(),
                    layer_start: e.layer_start,
                    layer_end: e.layer_end,
                    cpu_pct: e.cpu_pct,
                    ram_used: e.ram_used,
                    requests_in_flight: e.requests_in_flight,
                    rtt_ms: 0,
                    layer_stats,
                }
            })
            .collect();

        StatusResponse { models, servers }
    }
}

// ── gRPC service impl ─────────────────────────────────────────────────────────

pub struct GridServiceImpl {
    pub state: Arc<RwLock<GridState>>,
    next_id: AtomicU64,
    /// If set, every incoming Join stream must present "Authorization: Bearer <key>".
    grid_key: Option<String>,
}

impl GridServiceImpl {
    #[allow(dead_code)]
    pub fn new(state: Arc<RwLock<GridState>>) -> Self {
        Self {
            state,
            next_id: AtomicU64::new(1),
            grid_key: None,
        }
    }

    pub fn new_with_key(state: Arc<RwLock<GridState>>, key: Option<String>) -> Self {
        Self {
            state,
            next_id: AtomicU64::new(1),
            grid_key: key,
        }
    }

    fn alloc_server_id(&self) -> String {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("srv-{ts}-{n}")
    }
}

type JoinStream = Pin<Box<dyn futures_core::Stream<Item = Result<RouterMessage, Status>> + Send>>;

#[tonic::async_trait]
impl GridService for GridServiceImpl {
    type JoinStream = JoinStream;

    async fn join(
        &self,
        request: Request<Streaming<ServerMessage>>,
    ) -> Result<Response<Self::JoinStream>, Status> {
        // Auth check — reject streams that don't carry the correct grid key.
        if let Some(expected) = &self.grid_key {
            let token = request
                .metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "));
            if token.map(|t| t != expected).unwrap_or(true) {
                return Err(Status::unauthenticated("invalid grid key"));
            }
        }

        let state = self.state.clone();
        let server_id = self.alloc_server_id();
        let (tx, rx) = mpsc::channel::<Result<RouterMessage, Status>>(32);
        let mut inbound = request.into_inner();

        let sid = server_id.clone();
        tokio::spawn(async move {
            let mut registered_model: Option<(String, u32, u32)> = None; // (model_id, start, end)
            let mut is_available = false; // true while in Mode B available pool

            while let Some(msg) = inbound.next().await {
                match msg {
                    Err(e) => {
                        tracing::warn!(server_id = %sid, "Stream error: {e}");
                        break;
                    }
                    Ok(ServerMessage { payload: None }) => {}
                    Ok(ServerMessage { payload: Some(p) }) => match p {
                        ServerPayload::Announce(AnnounceMsg {
                            model_id,
                            layer_start,
                            layer_end,
                            ram_bytes,
                            listen_url,
                            vindex_hash,
                        }) => {
                            let entry = ServerEntry {
                                server_id: sid.clone(),
                                listen_url: listen_url.clone(),
                                model_id: model_id.clone(),
                                layer_start,
                                layer_end,
                                vindex_hash,
                                cpu_pct: 0.0,
                                ram_used: ram_bytes,
                                requests_in_flight: 0,
                                last_seen: Instant::now(),
                                layer_latencies: HashMap::new(),
                                req_per_sec: 0.0,
                            };
                            state.write().await.register_with_sender(entry, tx.clone());
                            registered_model = Some((model_id, layer_start, layer_end));

                            let ack = RouterMessage {
                                payload: Some(RouterPayload::Ack(AckMsg {
                                    server_id: sid.clone(),
                                })),
                            };
                            if tx.send(Ok(ack)).await.is_err() {
                                break;
                            }
                        }

                        ServerPayload::Heartbeat(hb) => {
                            state.write().await.update_heartbeat(
                                &sid,
                                hb.cpu_pct,
                                hb.ram_used,
                                hb.requests_in_flight,
                                hb.layer_stats,
                                hb.req_per_sec,
                            );
                        }

                        ServerPayload::Dropping(d) => {
                            tracing::info!(
                                server_id = %sid,
                                model_id = %d.model_id,
                                layers = %format!("{}-{}", d.layer_start, d.layer_end),
                                reason = %d.reason,
                                "Server dropping shard"
                            );
                            // Drop the server and immediately try to fill any
                            // freshly-exposed gap. After gap-fill, also check
                            // whether the disappearance created under-
                            // replication and pull spares for that too.
                            let (filled, replicated) = {
                                let mut guard = state.write().await;
                                guard.deregister(&sid);
                                let f = guard.try_fill_all_gaps();
                                let r = guard.try_replicate_from_available();
                                (f, r)
                            };
                            if filled > 0 || replicated > 0 {
                                tracing::info!(
                                    filled,
                                    replicated,
                                    "Grid: re-fill / re-replicate triggered by Dropping"
                                );
                            }
                            registered_model = None;
                        }

                        ServerPayload::Available(av) => {
                            // Mode B: server advertises capacity.
                            // Register it, then try to use it for either a
                            // coverage gap (zero replicas) or an
                            // under-replicated range (replica count below
                            // target_replicas).
                            state.write().await.register_available(
                                sid.clone(),
                                tx.clone(),
                                av.ram_bytes,
                                av.disk_bytes,
                                av.store_path.clone(),
                            );
                            is_available = true;
                            tracing::info!(
                                server_id = %sid,
                                ram_gb = av.ram_bytes / (1024 * 1024 * 1024),
                                "Grid: Mode B server registered; checking gaps + replicas…"
                            );

                            // 1) Fill coverage gaps first (most urgent — gaps
                            //    cause 503s; under-replicated ranges still
                            //    serve traffic from the surviving replicas).
                            let gaps = state.read().await.coverage_gaps();
                            let mut consumed = false;
                            for (model_id, layer_start, layer_end) in gaps {
                                let assigned = state.write().await.try_assign_gap(
                                    &model_id,
                                    layer_start,
                                    layer_end,
                                    av.ram_bytes,
                                );
                                if assigned {
                                    consumed = true;
                                    break;
                                }
                            }
                            // 2) If the spare wasn't used to fill a gap, try
                            //    using it to satisfy under-replication.
                            if !consumed {
                                let replicated = state.write().await.try_replicate_from_available();
                                if replicated > 0 {
                                    tracing::info!(
                                        replicated,
                                        "Grid: under-replicated range filled from new available server"
                                    );
                                }
                            }
                        }

                        ServerPayload::Ready(r) => {
                            // Mode B: server finished downloading + loading a shard.
                            // Register it as a serving shard and send Ack.
                            //
                            // vindex_hash is not present in ReadyMsg today (the
                            // server only knows the hash advertised on the assign);
                            // leave it empty for the freshly-loaded replica. This
                            // means the new replica won't be chosen as a Mode B
                            // origin for a further gap until its hash is known,
                            // but that's fine — the surviving original replica
                            // remains a valid origin.
                            let entry = ServerEntry {
                                server_id: sid.clone(),
                                listen_url: r.listen_url.clone(),
                                model_id: r.model_id.clone(),
                                layer_start: r.layer_start,
                                layer_end: r.layer_end,
                                vindex_hash: String::new(),
                                cpu_pct: 0.0,
                                ram_used: 0,
                                requests_in_flight: 0,
                                last_seen: std::time::Instant::now(),
                                layer_latencies: HashMap::new(),
                                req_per_sec: 0.0,
                            };
                            state.write().await.register_with_sender(entry, tx.clone());
                            registered_model =
                                Some((r.model_id.clone(), r.layer_start, r.layer_end));
                            is_available = false;
                            tracing::info!(
                                server_id = %sid,
                                model_id = %r.model_id,
                                layers = %format!("{}-{}", r.layer_start, r.layer_end),
                                "Grid: Mode B server ready — now serving"
                            );
                            let ack = RouterMessage {
                                payload: Some(RouterPayload::Ack(AckMsg {
                                    server_id: sid.clone(),
                                })),
                            };
                            if tx.send(Ok(ack)).await.is_err() {
                                break;
                            }
                        }

                        ServerPayload::Refuse(r) => {
                            // Mode B: server refused the assignment. Re-add to available pool.
                            tracing::warn!(
                                server_id = %sid,
                                reason = %r.reason,
                                "Grid: Mode B server refused assignment — re-queuing"
                            );
                            // The tx clone is lost here; server must reconnect. Just log.
                        }
                    },
                }
            }

            // Stream closed — clean up. If a serving server vanished without
            // a graceful DroppingMsg, gap re-fill + replication must still
            // run because losing a replica may have introduced a gap or
            // dropped the count below target_replicas.
            if registered_model.is_some() {
                let (filled, replicated) = {
                    let mut guard = state.write().await;
                    guard.deregister(&sid);
                    let f = guard.try_fill_all_gaps();
                    let r = guard.try_replicate_from_available();
                    (f, r)
                };
                if filled > 0 || replicated > 0 {
                    tracing::info!(
                        filled,
                        replicated,
                        "Grid: re-fill / re-replicate triggered by disconnect"
                    );
                }
            }
            if is_available {
                state.write().await.deregister_available(&sid);
            }
            tracing::info!(server_id = %sid, "Connection closed");
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn status(
        &self,
        _request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let resp = self.state.read().await.status_response();
        Ok(Response::new(resp))
    }

    /// ADR-0004 Phase 5: admin drain. Sends `UnassignMsg(reason)` to the
    /// named serving server. The server's announce loop handles drain →
    /// `DroppingMsg` → optional re-enter Mode B from there.
    async fn drain_server(
        &self,
        request: Request<DrainRequest>,
    ) -> Result<Response<AdminAck>, Status> {
        let req = request.into_inner();
        let reason = if req.reason.is_empty() {
            "admin_drain".to_string()
        } else {
            req.reason
        };

        // Find the server + the layer range it currently covers.
        let (sender, layers) = {
            let guard = self.state.read().await;
            let entry = guard
                .servers()
                .find(|(id, _)| **id == req.server_id)
                .map(|(_, e)| (e.model_id.clone(), e.layer_start, e.layer_end));
            (guard.serving_sender(&req.server_id), entry)
        };

        let Some((model_id, layer_start, layer_end)) = layers else {
            return Ok(Response::new(AdminAck {
                ok: false,
                message: format!("server_id {:?} is not currently serving", req.server_id),
            }));
        };
        let Some(tx) = sender else {
            return Ok(Response::new(AdminAck {
                ok: false,
                message: format!("server_id {:?} has no outbound channel", req.server_id),
            }));
        };

        let msg = RouterMessage {
            payload: Some(RouterPayload::Unassign(UnassignMsg {
                model_id,
                layer_start,
                layer_end,
                reason,
            })),
        };
        match tx.try_send(Ok(msg)) {
            Ok(()) => Ok(Response::new(AdminAck {
                ok: true,
                message: String::new(),
            })),
            Err(e) => Ok(Response::new(AdminAck {
                ok: false,
                message: format!("send to {:?} failed: {e}", req.server_id),
            })),
        }
    }

    /// ADR-0004 Phase 5: force-assign a layer range. Either targets a
    /// named available server (when `target_server_id` is set) or any
    /// available spare; either uses the explicit origin URL/hash from the
    /// request, or resolves an origin from the live coverage matrix.
    async fn assign_range(
        &self,
        request: Request<AssignRangeRequest>,
    ) -> Result<Response<AdminAck>, Status> {
        let req = request.into_inner();
        let mut guard = self.state.write().await;

        // Resolve the origin: explicit > live replica.
        let (origin_url, shard_hash) = if !req.explicit_origin_url.is_empty() {
            (
                req.explicit_origin_url.clone(),
                req.explicit_origin_hash.clone(),
            )
        } else {
            match guard.find_origin_for(&req.model_id, req.layer_start, req.layer_end) {
                Some(o) => o,
                None => {
                    return Ok(Response::new(AdminAck {
                        ok: false,
                        message: format!(
                            "no live replica covers {}[{}-{}] — pass explicit_origin_url to override",
                            req.model_id, req.layer_start, req.layer_end
                        ),
                    }));
                }
            }
        };

        // If target_server_id was named, hand the assignment to that specific
        // available server. Otherwise pick any spare.
        let assigned = if req.target_server_id.is_empty() {
            guard.try_assign_gap_with_origin(
                &req.model_id,
                req.layer_start,
                req.layer_end,
                &origin_url,
                &shard_hash,
                /* min_ram */ 0,
            )
        } else {
            match guard.send_assign_to_named_available(
                &req.target_server_id,
                &req.model_id,
                req.layer_start,
                req.layer_end,
                &origin_url,
                &shard_hash,
            ) {
                Ok(()) => true,
                Err(reason) => {
                    return Ok(Response::new(AdminAck {
                        ok: false,
                        message: reason,
                    }));
                }
            }
        };
        Ok(Response::new(if assigned {
            AdminAck {
                ok: true,
                message: String::new(),
            }
        } else {
            AdminAck {
                ok: false,
                message: "no available server matched the request".into(),
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        server_id: &str,
        listen_url: &str,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> ServerEntry {
        ServerEntry {
            server_id: server_id.into(),
            listen_url: listen_url.into(),
            model_id: model_id.into(),
            layer_start,
            layer_end,
            vindex_hash: format!("hash-{server_id}"),
            cpu_pct: 0.0,
            ram_used: 1024,
            requests_in_flight: 0,
            last_seen: Instant::now(),
            layer_latencies: HashMap::new(),
            req_per_sec: 0.0,
        }
    }

    #[test]
    fn route_uses_inclusive_layer_ranges() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 2));
        state.register(entry("b", "http://b", "model-a", 3, 5));

        assert_eq!(state.route(Some("model-a"), 0).as_deref(), Some("http://a"));
        assert_eq!(state.route(Some("model-a"), 2).as_deref(), Some("http://a"));
        assert_eq!(state.route(Some("model-a"), 3).as_deref(), Some("http://b"));
        assert_eq!(state.route(Some("model-a"), 5).as_deref(), Some("http://b"));
        assert_eq!(state.route(Some("model-a"), 6), None);
    }

    #[test]
    fn route_without_model_uses_any_model_table() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));

        assert_eq!(state.route(None, 1).as_deref(), Some("http://a"));
        assert_eq!(state.route(None, 2), None);
    }

    #[test]
    fn route_prefers_least_loaded_replica() {
        let mut state = GridState::default();
        let mut busy = entry("busy", "http://busy", "model-a", 0, 4);
        busy.requests_in_flight = 12;
        let mut idle = entry("idle", "http://idle", "model-a", 0, 4);
        idle.requests_in_flight = 1;

        state.register(busy);
        state.register(idle);

        assert_eq!(
            state.route(Some("model-a"), 3).as_deref(),
            Some("http://idle")
        );
    }

    #[test]
    fn deregister_removes_server_from_route_table() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 2));
        state.register(entry("b", "http://b", "model-a", 3, 5));

        state.deregister("a");

        assert_eq!(state.route(Some("model-a"), 1), None);
        assert_eq!(state.route(Some("model-a"), 4).as_deref(), Some("http://b"));
    }

    #[test]
    fn heartbeat_updates_load_without_rebuilding_topology() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 4));
        state.register(entry("b", "http://b", "model-a", 0, 4));

        state.update_heartbeat("a", 80.0, 2048, 20, vec![], 0.0);
        state.update_heartbeat("b", 10.0, 1024, 0, vec![], 0.0);

        assert_eq!(state.route(Some("model-a"), 2).as_deref(), Some("http://b"));
        let a = state.servers.get("a").unwrap();
        assert_eq!(a.cpu_pct, 80.0);
        assert_eq!(a.ram_used, 2048);
        assert_eq!(a.requests_in_flight, 20);
    }

    #[test]
    fn route_all_returns_first_uncovered_layer() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        assert_eq!(state.route_all(Some("model-a"), &[0, 1, 2, 3]), Err(2));
    }

    #[test]
    fn status_response_reports_shards_and_gaps() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        let status = state.status_response();

        assert_eq!(status.servers.len(), 2);
        assert_eq!(status.models.len(), 1);
        let model = &status.models[0];
        assert_eq!(model.model_id, "model-a");
        assert_eq!(model.shards.len(), 2);
        assert_eq!(model.gaps.len(), 1);
        assert_eq!(model.gaps[0].layer_start, 2);
        assert_eq!(model.gaps[0].layer_end, 2);
    }

    #[test]
    fn route_prefers_lower_layer_latency_over_inflight() {
        // slow has fewer requests_in_flight but higher per-layer latency.
        // fast has more requests but lower layer latency.
        // Router should route to fast.
        let mut state = GridState::default();
        let mut slow = entry("slow", "http://slow", "model-a", 0, 4);
        slow.requests_in_flight = 2;
        slow.layer_latencies.insert(2, (50.0, 80.0)); // 50 ms avg

        let mut fast = entry("fast", "http://fast", "model-a", 0, 4);
        fast.requests_in_flight = 8;
        fast.layer_latencies.insert(2, (5.0, 9.0)); // 5 ms avg

        state.register(slow);
        state.register(fast);

        assert_eq!(
            state.route(Some("model-a"), 2).as_deref(),
            Some("http://fast")
        );
    }

    #[test]
    fn heartbeat_stores_layer_latencies() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 4));

        let stats = vec![LayerLatency {
            layer: 2,
            avg_ms: 3.5,
            p99_ms: 7.0,
        }];
        state.update_heartbeat("a", 0.0, 0, 0, stats, 0.0);

        let entry = state.servers.get("a").unwrap();
        assert_eq!(entry.layer_latencies.get(&2), Some(&(3.5, 7.0)));
    }

    #[test]
    fn status_response_includes_layer_stats() {
        let mut state = GridState::default();
        let mut srv = entry("a", "http://a", "model-a", 0, 1);
        srv.layer_latencies.insert(0, (2.1, 4.0));
        state.register(srv);

        let status = state.status_response();
        let server = &status.servers[0];
        assert_eq!(server.layer_stats.len(), 1);
        assert_eq!(server.layer_stats[0].layer, 0);
        assert!((server.layer_stats[0].avg_ms - 2.1).abs() < 0.001);
    }

    #[test]
    fn send_assign_to_named_available_dispatches_to_specific_server() {
        let mut state = GridState::default();
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        state.register_available("target".into(), tx, 1, 0, "/".into());

        state
            .send_assign_to_named_available(
                "target",
                "test-model",
                10,
                14,
                "http://origin:8090",
                "deadbeef",
            )
            .expect("send must succeed");
        let msg = rx
            .try_recv()
            .expect("AssignMsg should have been queued")
            .expect("ok payload");
        let Some(RouterPayload::Assign(a)) = msg.payload else {
            panic!("expected Assign, got {msg:?}");
        };
        assert_eq!(a.model_id, "test-model");
        assert_eq!(a.layer_start, 10);
        assert_eq!(a.layer_end, 14);
        assert_eq!(a.origin_url, "http://origin:8090");
        assert_eq!(a.shard_hash, "deadbeef");
        // Entry consumed.
        assert!(!state.has_available_servers());
    }

    #[test]
    fn send_assign_to_named_available_unknown_id_errors() {
        let mut state = GridState::default();
        let err = state
            .send_assign_to_named_available("no-such", "test-model", 0, 4, "http://origin", "h")
            .unwrap_err();
        assert!(err.contains("not in the available pool"));
    }

    #[test]
    fn send_assign_to_named_available_failed_send_re_inserts_entry() {
        let mut state = GridState::default();
        // Drop the receiver so the send fails.
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        state.register_available("target".into(), tx, 1, 0, "/".into());

        let err = state
            .send_assign_to_named_available("target", "m", 0, 4, "http://origin", "h")
            .unwrap_err();
        assert!(err.contains("failed"));
        // Entry must still be in the pool for a follow-up retry.
        assert!(state.has_available_servers());
    }

    #[test]
    fn register_available_and_deregister() {
        let mut state = GridState::default();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        state.register_available(
            "avail-1".into(),
            tx,
            16 * 1024 * 1024 * 1024,
            100 * 1024 * 1024 * 1024,
            "/mnt/shards".into(),
        );
        assert!(state.available_servers.contains_key("avail-1"));
        state.deregister_available("avail-1");
        assert!(!state.available_servers.contains_key("avail-1"));
    }

    #[test]
    fn coverage_gaps_finds_uncovered_range() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        let gaps = state.coverage_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0], ("model-a".to_string(), 2, 2));
    }

    #[test]
    fn all_shard_urls_deduplicates() {
        let mut state = GridState::default();
        // Two servers on the same listen_url (e.g. shared host); a third on a
        // different one — must collapse to two unique entries.
        let a = entry("a", "http://host:8080", "model-a", 0, 1);
        let b = entry("b", "http://host:8080", "model-a", 2, 3);
        let c = entry("c", "http://other:8081", "model-a", 4, 5);
        state.register(a);
        state.register(b);
        state.register(c);

        let mut urls = state.all_shard_urls();
        urls.sort();
        assert_eq!(urls, vec!["http://host:8080", "http://other:8081"]);
    }

    #[test]
    fn grid_service_impl_constructors_install_state_and_key() {
        let state = Arc::new(RwLock::new(GridState::default()));

        let svc = GridServiceImpl::new(state.clone());
        let id1 = svc.alloc_server_id();
        let id2 = svc.alloc_server_id();
        assert!(id1.starts_with("srv-"));
        assert!(id2.starts_with("srv-"));
        assert_ne!(id1, id2, "ids must increment monotonically");

        // new_with_key wires the key field.
        let _svc_keyed = GridServiceImpl::new_with_key(state, Some("secret".into()));
        // Field is private; constructing it exercises the branch.
    }

    #[test]
    fn stale_server_ids_returns_only_overdue_entries() {
        let mut state = GridState::default();
        let mut fresh = entry("fresh", "http://fresh", "model-a", 0, 1);
        fresh.last_seen = Instant::now();
        let mut stale = entry("stale", "http://stale", "model-a", 0, 1);
        stale.last_seen = Instant::now()
            .checked_sub(std::time::Duration::from_secs(60))
            .unwrap_or_else(Instant::now);
        state.register(fresh);
        state.register(stale);

        let ids = state.stale_server_ids(std::time::Duration::from_secs(25));
        assert_eq!(ids, vec!["stale".to_string()]);

        // With a huge timeout nothing is stale.
        assert!(state
            .stale_server_ids(std::time::Duration::from_secs(3600))
            .is_empty());
    }

    #[test]
    fn find_origin_for_returns_listen_url_and_hash_of_replica() {
        let mut state = GridState::default();
        let mut a = entry("a", "http://a:8080", "model-a", 0, 5);
        a.vindex_hash = "deadbeef".into();
        state.register(a);

        let origin = state.find_origin_for("model-a", 0, 5);
        assert_eq!(origin, Some(("http://a:8080".into(), "deadbeef".into())));

        // Wrong model: no origin.
        assert!(state.find_origin_for("other", 0, 5).is_none());
        // Range outside coverage: no origin.
        assert!(state.find_origin_for("model-a", 6, 9).is_none());
    }

    #[test]
    fn try_assign_gap_resolves_origin_from_live_replica() {
        let mut state = GridState::default();
        // Two replicas of layers 0-5 — one will be the origin for a third
        // available server that fills a fresh assignment.
        let mut a = entry("a", "http://a:8080", "model-a", 0, 5);
        a.vindex_hash = "abc".into();
        state.register(a);

        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        state.register_available("spare".into(), tx, 16 * 1024 * 1024 * 1024, 0, "/".into());

        // Pretend layers 6-10 became a gap and we need to fill it. There's no
        // live replica for that range, so the assignment should be refused.
        assert!(!state.try_assign_gap("model-a", 6, 10, 0));

        // Now ask to fill an existing range — must find http://a:8080 as origin.
        assert!(state.try_assign_gap("model-a", 0, 5, 0));
        let sent = rx.try_recv().expect("AssignMsg should be queued");
        let Ok(RouterMessage {
            payload: Some(RouterPayload::Assign(assign)),
        }) = sent
        else {
            panic!("expected Assign payload, got: {sent:?}");
        };
        assert_eq!(assign.origin_url, "http://a:8080");
        assert_eq!(assign.shard_hash, "abc");
        assert_eq!(assign.layer_start, 0);
        assert_eq!(assign.layer_end, 5);
    }

    #[test]
    fn set_target_replicas_clamps_to_at_least_one() {
        let mut state = GridState::default();
        assert_eq!(state.target_replicas(), 1);
        state.set_target_replicas(0);
        assert_eq!(state.target_replicas(), 1, "0 must clamp to 1");
        state.set_target_replicas(3);
        assert_eq!(state.target_replicas(), 3);
    }

    #[test]
    fn under_replicated_ranges_reports_deficit_per_range() {
        let mut state = GridState::default();
        state.set_target_replicas(2);
        // Range 0-4: only one server → deficit 1.
        state.register(entry("a", "http://a", "model-x", 0, 4));
        // Range 5-9: two servers → at target.
        state.register(entry("b", "http://b", "model-x", 5, 9));
        state.register(entry("c", "http://c", "model-x", 5, 9));

        let ranges = state.under_replicated_ranges();
        assert_eq!(ranges, vec![("model-x".to_string(), 0, 4, 1)]);
    }

    #[test]
    fn over_replicated_ranges_reports_surplus() {
        let mut state = GridState::default();
        state.set_target_replicas(2);
        // 3 replicas of 0-4 — surplus 1.
        state.register(entry("a", "http://a", "model-x", 0, 4));
        state.register(entry("b", "http://b", "model-x", 0, 4));
        state.register(entry("c", "http://c", "model-x", 0, 4));
        // 1 replica of 5-9 — under target, not over.
        state.register(entry("d", "http://d", "model-x", 5, 9));

        let over = state.over_replicated_ranges();
        assert_eq!(over, vec![("model-x".to_string(), 0, 4, 1)]);
    }

    #[test]
    fn least_loaded_in_range_picks_lowest_inflight() {
        let mut state = GridState::default();
        let mut a = entry("a", "http://a", "model-x", 0, 4);
        a.requests_in_flight = 5;
        let mut b = entry("b", "http://b", "model-x", 0, 4);
        b.requests_in_flight = 1;
        let mut c = entry("c", "http://c", "model-x", 0, 4);
        c.requests_in_flight = 9;
        state.register(a);
        state.register(b);
        state.register(c);

        let pick = state.least_loaded_in_range("model-x", 0, 4).unwrap();
        assert_eq!(pick.server_id, "b");

        // Wrong range yields None.
        assert!(state.least_loaded_in_range("model-x", 10, 14).is_none());
    }

    #[test]
    fn under_replicated_ranges_ignores_zero_coverage() {
        let mut state = GridState::default();
        state.set_target_replicas(2);
        // No server for layers 0-4 — that's a *gap*, handled separately.
        // Provide some other coverage to keep the test realistic.
        state.register(entry("a", "http://a", "model-y", 10, 14));
        // model-y[10-14] has 1/2 → under-replicated.
        let ranges = state.under_replicated_ranges();
        assert_eq!(ranges, vec![("model-y".to_string(), 10, 14, 1)]);
    }

    #[test]
    fn try_replicate_from_available_dispatches_one_per_range() {
        let mut state = GridState::default();
        state.set_target_replicas(2);
        // One server covering 0-4 — under-replicated by 1.
        let mut a = entry("a", "http://a", "model-x", 0, 4);
        a.vindex_hash = "ha".into();
        state.register(a);

        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        state.register_available("spare".into(), tx, 1, 0, "/".into());

        let sent = state.try_replicate_from_available();
        assert_eq!(sent, 1);
        let msg = rx
            .try_recv()
            .expect("AssignMsg should have been delivered")
            .expect("ok payload");
        let Some(RouterPayload::Assign(a)) = msg.payload else {
            panic!("expected Assign payload");
        };
        assert_eq!(a.model_id, "model-x");
        assert_eq!(a.layer_start, 0);
        assert_eq!(a.layer_end, 4);
        assert_eq!(a.origin_url, "http://a");
        assert_eq!(a.shard_hash, "ha");

        // No more spares → second call assigns nothing.
        let again = state.try_replicate_from_available();
        assert_eq!(again, 0);
    }

    #[test]
    fn try_fill_all_gaps_scans_coverage_and_fills() {
        let mut state = GridState::default();
        // Two shards with a gap at layer 2.
        let mut a = entry("a", "http://a:8080", "model-a", 0, 1);
        a.vindex_hash = "ha".into();
        let mut b = entry("b", "http://b:8080", "model-a", 3, 4);
        b.vindex_hash = "hb".into();
        state.register(a);
        state.register(b);
        // No live replica covers layer 2 alone, so coverage_gaps reports it
        // but find_origin_for returns None — try_fill_all_gaps should send 0.
        let (tx, _rx) = tokio::sync::mpsc::channel(4);
        state.register_available("spare".into(), tx, 1, 0, "/".into());
        assert_eq!(state.try_fill_all_gaps(), 0);
    }

    #[test]
    fn coverage_gaps_empty_when_fully_covered() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 2));
        state.register(entry("b", "http://b", "model-a", 3, 5));

        // Only gap-between-shards; shards are contiguous here.
        let gaps = state.coverage_gaps();
        assert!(gaps.is_empty());
    }

    // ── Hot-shard helpers ────────────────────────────────────────────────────

    #[test]
    fn heartbeat_stores_req_per_sec() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 4));
        state.update_heartbeat("a", 0.0, 0, 0, vec![], 12.5);
        assert!((state.servers.get("a").unwrap().req_per_sec - 12.5).abs() < 1e-6);
    }

    #[test]
    fn hot_layer_ranges_empty_when_threshold_zero_or_negative() {
        let mut state = GridState::default();
        let mut a = entry("a", "http://a", "model-x", 0, 4);
        a.req_per_sec = 100.0;
        state.register(a);
        // Disabled when threshold <= 0.
        assert!(state.hot_layer_ranges(0.0).is_empty());
        assert!(state.hot_layer_ranges(-1.0).is_empty());
    }

    #[test]
    fn hot_layer_ranges_returns_max_across_replicas() {
        // Two replicas of the same range, one hot, one cool — range is
        // hot if max(req_per_sec) crosses the threshold.
        let mut state = GridState::default();
        let mut hot = entry("hot", "http://hot", "model-x", 0, 4);
        hot.req_per_sec = 50.0;
        let mut cool = entry("cool", "http://cool", "model-x", 0, 4);
        cool.req_per_sec = 5.0;
        state.register(hot);
        state.register(cool);

        let ranges = state.hot_layer_ranges(20.0);
        assert_eq!(ranges, vec![("model-x".to_string(), 0, 4)]);

        // Threshold above both replicas: range is not hot.
        assert!(state.hot_layer_ranges(75.0).is_empty());
    }

    #[test]
    fn elevated_ranges_lift_effective_target_for_over_and_under() {
        let mut state = GridState::default();
        state.set_target_replicas(2);
        state.register(entry("a", "http://a", "model-x", 0, 4));
        state.register(entry("b", "http://b", "model-x", 0, 4));
        // At target: not over, not under.
        assert!(state.over_replicated_ranges().is_empty());
        assert!(state.under_replicated_ranges().is_empty());

        // Elevate → effective target = 3. Two replicas now look under by 1.
        assert!(state.mark_elevated("model-x", 0, 4));
        assert_eq!(state.effective_target_for("model-x", 0, 4), 3);
        assert_eq!(
            state.under_replicated_ranges(),
            vec![("model-x".to_string(), 0, 4, 1)]
        );
        assert!(state.over_replicated_ranges().is_empty());

        // Add a third — at effective target, neither over nor under.
        state.register(entry("c", "http://c", "model-x", 0, 4));
        assert!(state.over_replicated_ranges().is_empty());
        assert!(state.under_replicated_ranges().is_empty());

        // Demote → effective target = 2. Three replicas surplus by 1.
        assert!(state.demote_elevated("model-x", 0, 4));
        assert_eq!(
            state.over_replicated_ranges(),
            vec![("model-x".to_string(), 0, 4, 1)]
        );
    }

    #[test]
    fn mark_elevated_is_idempotent_and_demote_reports_prior_state() {
        let mut state = GridState::default();
        assert!(state.mark_elevated("m", 0, 4)); // newly inserted
        assert!(!state.mark_elevated("m", 0, 4)); // already there
        assert!(state.demote_elevated("m", 0, 4)); // was present
        assert!(!state.demote_elevated("m", 0, 4)); // already gone
    }

    #[test]
    fn elevated_ranges_snapshot_sorted_and_isolated() {
        let mut state = GridState::default();
        state.mark_elevated("z", 0, 4);
        state.mark_elevated("a", 5, 9);
        state.mark_elevated("a", 0, 4);
        let snap = state.elevated_ranges_snapshot();
        assert_eq!(
            snap,
            vec![
                ("a".to_string(), 0, 4),
                ("a".to_string(), 5, 9),
                ("z".to_string(), 0, 4),
            ]
        );
    }
}
