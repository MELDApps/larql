# Roadmap — larql-router / larql-router-protocol

---

## Current state (2026-05-15)

Self-assembling grid is feature-complete across ADR-0004 Phase 1–5, ADR-0010
(QUIC), ADR-0011 (Mode B + Phase B2 drain-then-reassign + replication), and
ADR-0012 Phase 2 (criterion micro-benchmarks). Static `--shards` (ADR-0003)
remains as a fallback and coexists with the grid.

The codebase is architecture-agnostic: routing logic reads layer ranges,
`model_id`, and server state from the grid protocol — no model-family
constants are hardcoded.

### What works today

- **Mode A** — `AnnounceMsg` → `AckMsg` registration + heartbeat loop + reconnect.
- **Mode B (Phase B1 + B2)** — `AvailableMsg` → `AssignMsg` → `ReadyMsg`; servers
  re-enter the available pool after an `UnassignMsg`-driven drain on the same
  stream.
- **Replication** — `--target-replicas N`; under-replicated ranges pull spares
  from the available pool, over-replicated ranges drop the least-loaded
  replica via `UnassignMsg`. Origin URLs resolved from any live replica via
  `find_origin_for`.
- **Hot-shard load-rate replication** — `--hot-shard-rps THRESHOLD`; when a
  shard's max `HeartbeatMsg.req_per_sec` across replicas exceeds the
  threshold, the rebalancer treats it as effectively under-replicated
  (`target + 1`) and pulls a spare. The elevated flag clears when the rate
  drops; over-replication then prunes the surplus on the next tick.
- **Stale heartbeat eviction** — rebalancer evicts serving servers whose
  `last_seen` exceeds `stale_heartbeat_timeout` (default 25 s = 2.5 ×
  heartbeat interval).
- **Per-layer latency-aware routing (GT3)** — `route()` prefers the server
  with lowest `layer_latencies[layer].avg_ms`; falls back to
  `requests_in_flight` when GT3 data is absent.
- **`GridService.Join`** bidirectional gRPC stream over TCP (default) or
  QUIC (`--features quic`).
- **QUIC transport (GT7)** — `--quic-port`, `--quic-cert`/`--quic-key` (or
  auto-generated self-signed cert), SHA-256 fingerprint pinning on the
  client side via `--quic-cert-fingerprint`. HTTP/2 carried over a single
  QUIC bi-stream; 0-RTT reconnect + TLS 1.3.
- **Admin CLI (Phase 5)** — `larql-router status` / `gaps` / `drain --server`
  / `assign --model M --layers A-B [--server S] [--origin-url URL]`. Backed
  by new `DrainServer` + `AssignRange` gRPC RPCs.
- `DroppingMsg` → deregistration + auto gap re-fill + auto re-replication.
- Static `--shards` mode with layer-range routing and per-shard parallel
  fan-out.
- Grid + static fallback via `AppState::resolve_all()`.
- `GET /grid-status` (served by `StatusResponse` with `layer_stats` per
  server).
- Auth: optional shared `--grid-key` Bearer token in gRPC metadata.
- Library crate (`larql_router::{grid, rebalancer, dispatch, shards, http,
  admin, cli_helpers}`) for tests and external consumers.
- Criterion benchmarks: `routing.rs` (route, route_all, heartbeat, rebuild)
  (GT9 ✅).

### What is not yet implemented

- **Cross-router federation** — multi-region routing (P2).
- **Expert-level routing** — MoE within-layer expert sharding (P2).
- **RTT-based routing** — `ServerInfo.rtt_ms` from active probes (P2).

---

## Live perf snapshot (2026-05-16, M3 Max)

| Path | tok/s |
|---|---|
| Gemma 3 4B local Metal (today's code) | **86.1** |
| ollama gemma3:4b (same machine) | 98.7 |
| Gemma 4 26B-A4B, 2-shard grid, gRPC streaming + UDS + TCP_NODELAY | 19.7 |

Per-call transport RTT (loopback):

- TCP HTTP: ~660 µs
- UDS HTTP: ~510 µs
- gRPC streaming (multiplexed): ~460 µs

gRPC routing hot path (in-process, criterion; rerun 2026-05-16, M3 Max):

| Op | 1 server | 10 servers | 100 servers |
|---|---|---|---|
| `route()` single layer | 94 ns | 233 ns † | 1.23 µs |
| `route_all()` 30 layers | 3.29 µs | 6.07 µs | 40.0 µs |
| `update_heartbeat()` | 273 ns | 271 ns | 272 ns |
| `rebuild_route_table()` 30 layers | 14.5 µs ‡ | 328 µs | 22.1 ms |
| `rebuild_route_table()` 62 layers | 19.4 µs | 652 µs | 44.3 ms |

`update_heartbeat()` is ~3–5% faster than the 2026-05-15 baseline
despite the added `req_per_sec` field assignment;
`rebuild_route_table()` improved 7–10% at higher server counts.

† `route()` at 10 servers was flagged as a +18% regression but with a
[201..272 ns] sample range and several high-severe outliers — looks
like thermal noise rather than a real change. Rerun on a cool
machine before bisecting.
‡ `rebuild_route_table()` 1srv_30layers had 12 outliers (10 high
severe); the 14.5 µs center has wide error bars.

QUIC has not been benched against TCP yet on real workloads — `quic` is
opt-in and not in the default-build path.

---

## Coverage

`make larql-router-coverage-summary`:

```
Coverage policy passed: total 91.48% lines,
                        7 files checked, 7 files at 90.0% default, 0 debt baselines.
```

Per-file:

| File | Lines |
|---|---|
| `shards.rs` | 100.00% |
| `dispatch.rs` | 100.00% |
| `admin.rs` | 99.64% |
| `cli_helpers.rs` | 98.59% |
| `http.rs` | 96.01% |
| `grid.rs` | 94.91% |
| `rebalancer.rs` | 93.54% |
| `main.rs` | (excluded — binary entry point) |

---

## Shipped (P1)

### GT3 — Per-layer latency in HeartbeatMsg ✅ shipped 2026-05-07

**Spec**: ADR-0011 §HeartbeatMsg Extension.

**What shipped:**
- `grid.proto`: `LayerLatency { layer, avg_ms, p99_ms }` message;
  `HeartbeatMsg.layer_stats = 4`; `ServerInfo.layer_stats = 11`.
- `ServerEntry.layer_latencies: HashMap<u32, (f32, f32)>`.
- `update_heartbeat()` accepts `Vec<LayerLatency>` and stores them.
- `route()` prefers server with lowest `layer_latencies[layer].avg_ms` when
  data exists; falls back to `requests_in_flight`.
- `status_response()` populates `ServerInfo.layer_stats` sorted by layer.

---

### GT5 — Mode B: gap-fill assignment ✅ shipped 2026-05-13

**Spec**: ADR-0011 §Phase B1 Protocol.

**What shipped:**
- `GridState` carries `available_servers`, `serving_senders`.
- `GridState::find_origin_for(model_id, start..=end) -> Option<(url, hash)>` —
  picks any currently-serving replica covering the range as origin.
- `GridState::try_assign_gap(...)` resolves origin automatically;
  `try_assign_gap_with_origin(...)` retained for external origins.
- `GridState::try_fill_all_gaps()` scans `coverage_gaps()` and fills each
  from the available pool.
- Gap re-fill auto-fires on `DroppingMsg` and stream-close paths.
- Server side: `larql-server` exposes `GET /v1/shard/{model_id}/{start}-{end}`
  as a tar stream so the spare can mirror the donor's vindex; matching tar
  unpack in `shard_loader.rs`.
- Server announce client transitions from Mode A to Mode B on the same
  gRPC stream after drain (`available_after_drain` config).
- Integration tests: `crates/larql-server/tests/test_grid_mode_b.rs` (full
  vertical handoff + negative path) and `test_grid_drain_reassign.rs`
  (Phase B2 cycle).

---

### GT6 — Dynamic rebalancing ✅ shipped 2026-05-13

**Spec**: ADR-0011 §Phase B2 Protocol.

**What shipped:**
- `rebalancer::check_imbalance` — sustained imbalance trigger
  (`max/min > threshold` over `sustained_window`).
- `rebalancer::check_under_replication` + `check_over_replication` — Phase 4
  replica-count enforcement (sends `UnassignMsg` to least-loaded victim when
  over-replicated; pulls from available pool when under-replicated).
- `rebalancer::evict_stale_heartbeats` — defensive eviction of servers that
  stop heartbeating without closing the stream.
- New `GridState::send_assign_to_named_available()` for the admin
  `assign --server <id>` path.

---

### GT7 — QUIC transport ✅ shipped 2026-05-15

**Spec**: ADR-0010 (full spec).

**What shipped (feature-gated under `quic`):**
- `crates/larql-router-protocol/src/transport/quic.rs`:
  - `QuicStream` — wraps `(SendStream, RecvStream)` as `AsyncRead+Write` + `tonic::transport::server::Connected`.
  - `self_signed_tls(server_name)` — rcgen-based dev cert with SHA-256 fingerprint.
  - `server_endpoint(addr, tls)` / `client_endpoint(bind, expected_fingerprint)`.
  - `FingerprintVerifier` — pins server cert by SHA-256 (no CA chain).
  - `spawn_accept_loop(endpoint)` — accepts QUIC conns + bi-streams, feeds tonic `serve_with_incoming`.
  - `connect_grpc_channel(endpoint, addr, server_name)` — full client wiring.
- Router: `--quic-port`, `--quic-cert`, `--quic-key`, `--quic-server-name`.
  Parallel QUIC listener alongside the TCP gRPC server.
- Server: `--quic-cert-fingerprint`. `announce::try_once` branches on
  `quic://` scheme via `connect_grid_channel`.
- Round-trip integration tests: announce → ack streaming + unary `Status`
  over QUIC (`crates/larql-router-protocol/tests/test_quic_roundtrip.rs`).

**Limitation:** This is QUIC-as-TCP-replacement (HTTP/2 over a single QUIC
bi-stream), not HTTP/3. Buys 0-RTT reconnect + TLS 1.3 + BBRv2 congestion
control; per-stream-independence is moot for `Join` (single bidi stream
per server). HTTP/3 for expert-fan-out would be a future ADR.

---

### GT9 — Criterion routing benchmarks ✅ shipped 2026-05-07

**Spec**: ADR-0012 §Layer 2.

**What shipped:**
- `crates/larql-router/benches/routing.rs`: `bench_route_single_layer`,
  `bench_route_all`, `bench_heartbeat_update`, `bench_rebuild_route_table`
  at 1/10/100 servers × 30/62 layers.
- `src/lib.rs` exposes `pub mod grid` for bench linking.
- Makefile: `make bench-routing` / `make bench-all`.

---

### Phase 5 — Admin CLI ✅ shipped 2026-05-15

**Spec**: ADR-0004 §"Admin API".

**What shipped:**
- New proto RPCs: `DrainServer(DrainRequest) -> AdminAck`,
  `AssignRange(AssignRangeRequest) -> AdminAck`.
- Server-side: `GridServiceImpl::drain_server`,
  `GridServiceImpl::assign_range` (resolves origin from live replica or
  accepts `explicit_origin_url`).
- CLI subcommands: `larql-router status` / `gaps [--model M]` /
  `drain --server ID [--reason R]` / `assign --model M --layers A-B [--server S] [--origin-url URL] [--origin-hash H]`.
- Pure helpers in `larql_router::admin`: `format_status`, `format_gaps`,
  `parse_layers`, plus RPC wrappers `admin_status`, `admin_gaps`,
  `admin_drain`, `admin_assign`.
- Integration tests in `crates/larql-router/tests/test_admin_rpcs.rs`.

---

### Hot-shard load-rate replication ✅ shipped 2026-05-15

**Spec**: ROADMAP P1 sketch (this file).

`target_replicas` enforces a *count*; this adds *rate-aware* replication.
A shard whose per-replica `req_per_sec` exceeds the configured threshold
is treated as under-replicated even at `replicas == target_replicas`,
prompting the rebalancer to pull one extra spare. When the rate subsides
the elevation is cleared and the existing over-replication tick drops
the surplus on the next pass.

**What shipped:**
- `grid.proto`: `HeartbeatMsg.req_per_sec = 5` (shard-scoped rate).
- Server: `LoadedModel.requests_total: Arc<AtomicU64>` bumped by
  `walk_ffn`. Heartbeat sender diffs against the last sample and divides
  by `HEARTBEAT_INTERVAL` to populate `req_per_sec`.
- `GridState`:
  - `ServerEntry.req_per_sec` updated by `update_heartbeat`.
  - `elevated_ranges: HashSet<(model_id, start, end)>`.
  - `hot_layer_ranges(threshold) -> Vec<...>` (max-rate-across-replicas).
  - `mark_elevated` / `demote_elevated` / `elevated_ranges_snapshot`.
  - `effective_target_for(model, start, end)` =
    `target_replicas + (1 if elevated else 0)`.
  - `under_replicated_ranges` / `over_replicated_ranges` consult the
    effective target instead of the raw `target_replicas`.
- `rebalancer::check_hot_shards`: marks newly hot ranges as elevated,
  demotes ranges whose rate has dropped below the threshold. Runs before
  under/over-replication so flips land in the same tick.
- `RebalancerConfig::hot_shard_rps_threshold: Option<f32>` with
  `with_hot_shard_threshold` builder.
- CLI: `--hot-shard-rps <f32>` flag on `larql-router`. Unset = disabled.

Validation path remains the same as before: with `--target-replicas 1
--hot-shard-rps 50` and the `--concurrent N` bench harness, a hot shard
pulls a spare to effectively become `target+1`, then drops back once
the bench finishes.

---

### Stale heartbeat eviction ✅ shipped 2026-05-15

**Spec**: ADR-0004 Phase 3 §"Stale heartbeat eviction".

**What shipped:**
- `GridState::stale_server_ids(timeout)` — pure helper, walks `last_seen`.
- `rebalancer::evict_stale_heartbeats` — async wrapper, deregisters + triggers gap-fill.
- `RebalancerConfig::stale_heartbeat_timeout` (default 25 s).

---

### Exp 41 — LAN preregistration matrix ✅ shipped 2026-05-15

**Spec**: `experiments/41_residual_transport_grid/{SPEC.md,REPORT.md:508-547}`.

Ported `run.py` orchestration into the Rust CLI as `larql bench
--bench-grid-lan PATH`. The Rust runner reads the same JSON config
schema (`runs[*]` with `id`, `command` template, `env`, optional
`estimate`) and emits a JSONL manifest with the same field shape, so
existing Python tooling reading `runs.jsonl` keeps working.

**What shipped:**
- `crates/larql-cli/src/commands/primary/bench/grid_lan.rs` — pure
  helpers (config types, `command_for` template substitution,
  `parse_bench_output`, `estimate_bytes` / `q8k_bytes`, CoV +
  retry-decision, `safe_name`, `selected_runs`). Unit-tested at 99.3%
  line coverage.
- `crates/larql-cli/src/commands/primary/bench/grid_lan_runtime.rs` —
  subprocess driver: per run, spawns `larql bench …`, archives
  stdout/stderr, captures returncode, writes JSONL. Excluded from
  coverage (matches `*_runtime.rs` convention).
- CLI flags on `larql bench`: `--bench-grid-lan PATH`,
  `--grid-lan-out DIR`, `--grid-lan-only ID` (repeatable),
  `--grid-lan-include-disabled`, `--grid-lan-dry-run`,
  `--grid-lan-cov-threshold` (default 0.15, mirrors Exp 41 spec),
  `--grid-lan-extra-repeats` (default 2).
- Exp 41 §LAN Preregistration retry rule: after the base repeats,
  the orchestrator computes per-row CoV across the
  `mean_ms_per_tok` samples and runs up to `extra_repeats` more times
  when the threshold trips.

Smoke-tested with the experiment's `config.example.json` —
`--grid-lan-dry-run --grid-lan-include-disabled` walks the full
5-run matrix and produces a structurally equivalent JSONL to
`run.py --dry-run`.

---

## P1 — Remaining

### Exp 53 — Rust port of the sharded-vindex shard endpoint

**Status:** Ready to lift. Python prototype validated: **0.085 ms loopback
RTT, 10/10 facts preserved, 0.2% step-budget overhead at 25 tok/s.**

Port `experiments/53_sharded_vindex/server.py:67-103` to a Rust shard
service inside `larql-server` so the existing GT7 QUIC transport in
`crates/larql-router-protocol/src/transport/quic.rs` can carry vindex
shard traffic alongside the existing `Join` stream. The Python protocol
is the wire-format proof-of-life; this is a transport-layer port, not a
new ADR. ~1 day.

**Caveat:** `ROADMAP_STATUS` lists "Multi-machine MoE — P2 unless
re-promoted". Lifting Exp 53 effectively promotes that item to P1. Don't
pull unless that promotion is wanted.

---

## P2 — Forward-looking

### Cross-router federation

Multiple routers cover different geographic regions. A client request is
forwarded to the regional router that owns the model shard. Requires a
router-to-router protocol (reuse `GridService.Join` with a `RouterMsg`
variant, or a separate `FederationService`). No implementation planned
until Act 2 multi-host demo is complete.

### Expert-level routing

Current routing is at the layer granularity (server owns a layer range).
For MoE models, a server could own a subset of experts within a layer,
not a range of layers. This requires the router to know expert IDs, not
just layer IDs. Proto messages already have `model_id` and
`layer_start/end`; extending to expert ranges is additive. ADR-0003
§Phase 2 covers this.

### RTT-based routing

`ServerInfo.rtt_ms` is defined in `StatusResponse` but never populated.
Adding active RTT probes (ICMP or HTTP HEAD) from the router to each
server would let the router prefer geographically closer servers as a
tie-breaker after latency-based routing from GT3.

### Cross-references — V1 / V2 work (other crates, tracked here for grid context)

These don't live in the router crate but bear on what the grid will be
asked to serve. Listed here so they're visible alongside transport / shard
work.

- **Exp 27 — hash routing across all layers (V1).** Top-2048 mask, 100%
  argmax recovered at KL=0.030 at L0 on Gemma 3 4B. Maps to
  `ROADMAP_STATUS` item #2 — "V1 hash routing across all layers". The
  L0 result is interp-validated; scaling across layers and architectures
  is the next step. Touches `larql-inference` / `larql-vindex`, not the
  router; the router's interest is in the resulting vindex shape (FFN
  rows become sparse-addressable, which changes shard-size economics).
- **Exp 26 — FP4 generality (V2).** `gemma3-4b-f16.vindex` is **99.83%
  per-feature block R<16-compliant natively, no QAT**; `down` is the
  long tail at 99.65%. Maps to `ROADMAP_STATUS` item #3 — "V2 FP4
  generality". Extending the audit script across architectures is the
  next step. Touches `larql-vindex` extraction; router impact is that
  FP4 shards quarter the wire-bytes-per-tok metric tracked by the bench
  harness.

---

### Real HTTP/3 (post-GT7)

GT7 wraps HTTP/2 over a single QUIC bi-stream — fine for the one-stream
`Join` call. For expert fan-out (8 parallel streams per token), real
HTTP/3 (via the `h3` crate + hyper-h3) would unlock per-stream
independence and avoid one stream's HoL stalling the others.
