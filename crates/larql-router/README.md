# larql-router

Layer-sharding router for distributed `larql-server` deployments.

## What it does

Fans out `POST /v1/walk-ffn` calls across multiple `larql-server`
shards, each owning a contiguous range of transformer layers, and
aggregates their results. The router is intentionally narrow — it
exposes only the endpoints needed for layer-fanout operation, not a
full transparent reverse proxy:

- `POST /v1/walk-ffn` — single-layer or multi-layer fan-out across
  the shard map. Multi-layer requests are dispatched in parallel
  to each owning shard and the results merged.
- `GET /v1/health` — liveness + grid coverage summary.

Other endpoints (`/v1/stats`, `/v1/walk`, `/v1/models`, etc.) live on
the individual shards — clients can call them directly on a shard's
HTTP port. The router exists to coordinate the fan-out, not to be
a full server.

## Two topologies

### Static `--shards` map

Router knows all shards' URLs at boot. Simplest ops; routes are
fixed for the router's lifetime.

```bash
larql-router \
    --shards 0-14=http://shard-a:9181,15-29=http://shard-b:9182 \
    --port 9090
```

### Self-assembling `--grid-port` + `--join`

Router exposes a gRPC port; shards register themselves with `--join
http://router:50052 --public-url http://shard:port`. The router
tracks coverage live and can accept / drop shards without a
restart.

```bash
# Router with HTTP on 9090 + grid gRPC on 50052
larql-router --grid-port 50052 --grid-key <secret> --port 9090

# Each shard joins (see larql-server docs for the full flag list)
larql-server <vindex> --port 9181 --layers 0-14 \
    --join http://router:50052 --grid-key <secret> \
    --public-url http://shard-a:9181
```

When a shard exits cleanly its announce stream closes; the router
logs `Grid: server left layers=N-M` and updates coverage. Requests
for now-uncovered layers return `HTTP 400 "layer N has no owning
shard in this router"` — clean error, not a hang. When the shard
restarts and re-joins, coverage automatically returns.

Both topologies serve the same HTTP API; clients don't need to know
which the operator picked.

## Self-assembling grid features

### Mode A vs Mode B

A `larql-server` joining the grid presents itself as either:

- **Mode A** (announced shard): the server has already loaded a
  specific layer range and sends `AnnounceMsg` to advertise it. This
  is the path used when the operator pins layer ownership via
  `--layers`.
- **Mode B** (available): the server has free disk + RAM but no shard
  loaded, sends `AvailableMsg`, and waits for the router to assign a
  layer range with `AssignMsg`. The server downloads the matching
  vindex shard via `GET /v1/shard/{model}/{start}-{end}` from a live
  origin, then sends `ReadyMsg` and transitions to Mode A.

Mode B lets a fresh server join a running grid and pick up coverage
without the operator pre-deciding which layers each box owns.

### Replication

`--target-replicas N` tells the router how many copies of each shard
range it should maintain. The rebalancer pulls spares from the
available pool when the count drops below `N`, and drops the
least-loaded replica when it climbs above. Combine with Mode B
servers as the spare pool.

### Dynamic rebalancing

Runs every `--rebalance-interval` seconds (default 30). Each tick:

1. Evicts servers whose heartbeat is older than 25 s (defensive
   against deadlocked TCP-but-no-progress connections).
2. Flips the elevated flag on shards exceeding the
   `--hot-shard-rps` threshold so their effective replica target is
   `target + 1`.
3. Pulls spares from the available pool for any under-replicated
   range.
4. Sends `UnassignMsg` to the least-loaded replica of any
   over-replicated range; the server drains in-flight requests for up
   to 30 s and re-enters Mode B if `--available-ram` was set.
5. Detects sustained per-layer latency imbalance
   (`--rebalance-threshold`, default 2× over a 60 s window) and
   evicts the slow replica.

Set `--rebalance-interval 0` to disable the background tick (you can
still drive moves manually via the admin RPCs).

### Admin CLI

The same binary doubles as an admin client:

```bash
larql-router status                                   # full grid + servers JSON
larql-router gaps [--model M]                         # uncovered layer ranges
larql-router drain --server <ID> [--reason "..."]     # send UnassignMsg
larql-router assign --model M --layers A-B \
    [--server <ID>] [--origin-url URL] [--origin-hash H]
```

These call `GridService.DrainServer` / `AssignRange` over the
router's gRPC port. `assign` resolves an origin from any live replica
unless `--origin-url` is set, which is the escape hatch for filling
a range that no surviving server still covers (S3, mirror, etc.).

### QUIC transport (opt-in)

Build with `--features quic` and start the router with `--quic-port`
to listen for `quic://router:PORT` joins alongside the TCP listener.
Servers pin the router cert with `--quic-cert-fingerprint <SHA-256>`
(printed at router startup when the self-signed cert is generated).

QUIC carries HTTP/2 over a single bidirectional stream — the same
tonic-generated client/server code as the TCP path. Buys 0-RTT
reconnect, TLS 1.3, and BBRv2 congestion control. Real HTTP/3
(per-stream independence) is a future ADR.

## Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--shards <SPEC>` | Comma-separated `START-END=URL` (inclusive bounds). Optional when `--grid-port` is set. | — |
| `--grid-port <PORT>` | gRPC server port for self-assembling grid. Servers connect with `--join`. | — |
| `--grid-key <KEY>` | Shared secret enforced on `--join` registrations. Reads `LARQL_GRID_KEY` env. Without it, the grid port is open (development only). | — |
| `--port <PORT>` | HTTP listen port. | 9090 |
| `--host <HOST>` | Bind address. | 0.0.0.0 |
| `--timeout-secs <N>` | Per-request timeout to backend shards. | 120 |
| `--target-replicas <N>` | Phase 4 replication target per shard range. `>1` pulls spares from the available pool to maintain count. | 1 |
| `--rebalance-interval <SECS>` | Rebalancer tick cadence; `0` disables dynamic rebalancing. | 30 |
| `--rebalance-threshold <RATIO>` | Latency-imbalance threshold (slowest replica / fastest) before the rebalancer evicts. | 2.0 |
| `--hot-shard-rps <FRAC>` | Hot-shard load-rate replication: shards whose max `req_per_sec` across replicas exceeds this value are treated as effectively under-replicated until the rate subsides. | — (disabled) |
| `--log-level <LEVEL>` | Logging level. | info |

Run `larql-router --help` for the full set, including the QUIC
transport (`--quic-port` / `--quic-cert` / `--quic-key`) and admin
subcommands (`larql-router status / gaps / drain / assign`). See
[`ROADMAP.md`](./ROADMAP.md) for the per-feature shipping notes.

## Live perf snapshot (2026-05-16, M3 Max)

End-to-end:

| Path | tok/s |
|---|---|
| Gemma 3 4B local Metal | **86.1** |
| ollama gemma3:4b (same machine) | 98.7 |
| Gemma 4 26B-A4B, 2-shard grid (gRPC streaming + UDS + TCP_NODELAY) | 19.7 |

Per-call transport RTT (loopback): TCP HTTP ~660 µs, UDS HTTP ~510 µs,
gRPC streaming (multiplexed) ~460 µs.

gRPC routing hot path (in-process criterion benches; 2026-05-16):

| Op | 1 server | 10 servers | 100 servers |
|---|---|---|---|
| `route()` single layer | 94 ns | 233 ns | 1.23 µs |
| `route_all()` 30 layers | 3.29 µs | 6.07 µs | 40.0 µs |
| `update_heartbeat()` | 273 ns | 271 ns | 272 ns |
| `rebuild_route_table()` 30 layers | 14.5 µs | 328 µs | 22.1 ms |

```bash
make bench-routing     # criterion sweeps; see crates/larql-router/benches/routing.rs
```

See [`ROADMAP.md` § Live perf snapshot](./ROADMAP.md#live-perf-snapshot-2026-05-16-m3-max)
for caveats on the noisier rows (notably `rebuild_route_table()
1srv_30layers`, which had high-severe outliers on this run).

QUIC has not been benched against TCP yet on real workloads — `quic`
is opt-in and not in the default build.

## Validation

Grid routing + rebalancing are covered by focused unit + integration tests:

- inclusive layer-range routing, model-specific + default single-model tables
- least-loaded replica selection from heartbeat load
- per-layer latency-aware routing (GT3 `HeartbeatMsg.layer_stats`)
- Mode B `Available → Assign → Ready` + Phase B2 drain-then-reassign
- under/over-replication ticks with effective-target bookkeeping
- hot-shard `req_per_sec` detection + elevation/demotion
- stale-heartbeat eviction
- gap-fill on `DroppingMsg` / disconnect
- admin RPCs (`status` / `gaps` / `drain` / `assign`)

```bash
cargo test -p larql-router                    # 150 tests (lib + integration)
cargo test -p larql-router-protocol --features quic
                                               # 18 tests (15 unit + 3 QUIC integration)
make larql-router-coverage-summary             # 91.60% total, 7/7 files ≥90%
make larql-router-protocol-coverage-summary    # 91.36% total, 1/1 files ≥90%
```

Test counts as of 2026-05-16.

## See also

- `crates/larql-router-protocol/README.md` — gRPC schema + the
  QUIC transport wrapper that backs `--quic-port`.
- `crates/larql-server/README.md` — shard configuration, recommended
  setups, the `--join` / `--public-url` / `--grid-key` flags.
- `crates/larql-server/docs/router-spec.md` — protocol-level spec
  for the gRPC schema, endpoint contracts, and binary wire format.
- [`ROADMAP.md`](./ROADMAP.md) — per-feature shipping notes and
  what's still on P1 / P2.
