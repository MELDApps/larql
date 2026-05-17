# Changelog — larql-server

All notable changes to `larql-server` are documented here.

The format follows the conventions of [Keep a Changelog](https://keepachangelog.com/),
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

## [2026-05-17] — infer.rs un-excluded; fixture wall hit on chat/completions/stream

Continued coverage push. `routes/infer.rs` joins `routes/explain.rs`
in the included set (50% → 97% with 9 new integration tests +
2 new in-file unit tests for `format_knn_override`). Started
coverage tests for `routes/openai/completions.rs` and
`routes/expert/warmup.rs` — both plateau without further fixture
work:

- **`openai/completions.rs` (40% → 56%)**: synthetic vindex's
  diagonal-ish f32 weights NaN partway through
  `generate_with_sampling`, so the per-prompt completion-loop body
  + the streaming spawn_blocking body stay uncovered. Tests
  exercise the handler validation (n>1, empty prompt, echo+stream
  rejection, batched+stream rejection, sampling params, stop
  strings, logprobs, SSE content-type), which lifts the entry
  branches. Stays excluded until stable synthetic generation weights
  land. 12 integration tests pass.
- **`expert/warmup.rs` (0% → 9%)**: only the env-gated and
  `is_hybrid_moe() == false` early-return branches are reachable
  with the current dense-llama synthetic. The deeper unit/expert
  filter resolution + HNSW build path needs a synthetic hybrid-MoE
  arch (router weights + N experts). 2 integration tests pass.

### Realistic-ceiling note

Pushing the remaining excluded files (`openai/chat.rs`,
`openai/completions.rs`, `stream.rs`, all `expert/*.rs`,
`walk_ffn/core.rs`, `walk_ffn/q8k.rs`) above 90% requires three
substantial fixture extensions:

1. **Stable generation weights** — synthetic vindex weights tuned
   so `predict_with_ffn` / `generate_with_sampling` produce
   finite (non-NaN) output for multiple tokens. Unlocks chat,
   completions, stream, infer's compare-mode deeper paths.
2. **Synthetic hybrid-MoE vindex** — router_proj + N expert weights
   + arch override returning `is_hybrid_moe() == true`. Unlocks
   all `expert/*.rs` files + `walk_ffn/core.rs`'s MoE branch.
3. **Synthetic Q4K-quantised vindex** — different storage format
   (interleaved_q4k.bin etc). Unlocks `walk_ffn/q8k.rs`.

Each is ~1-2 days of fixture engineering. The honest path to total
≥ 90% is to land those fixtures, not to keep adding ineffective
tests against the current dense-llama synthetic.

### Headline numbers (this session)

| Metric | Pre-session | End of session |
|---|---|---|
| Total | 69.82% | **74.52%** (+4.70 pp) |
| Included files | 91.87% | **92.61%** |
| Files at 90% default | 26 | **32** (+6) |
| Files excluded | 13 | 12 (replaced walk_ffn.rs monolith with core.rs + q8k.rs; net −1 after explain + infer un-exclusion) |
| Test count | 739 | **804** (+65) |

## [2026-05-17] — walk_ffn split into a module folder + coverage progress

Continuing the coverage push: split the 1434-line `routes/walk_ffn.rs`
monolith into a 7-file `routes/walk_ffn/` module folder, then drove
coverage on the new sub-files. Net: total 71.18% → **73.76%**;
included 91.98% → **92.42%**; 27 → **31** files at default 90%.

### Changed

- **`routes/walk_ffn.rs` → `routes/walk_ffn/`**: split into seven
  files matching the natural section headers in the original:
  - `types.rs` (100%) — `WalkFfnRequest`, `FfnEntry`, `FfnOutput`,
    `RifGuard`, `BINARY_CT`, `BATCH_MARKER`.
  - `binary.rs` (98.56%) — codec (`decode_binary_request`,
    `encode_binary_output` / `_f16` / `_i8`, `encode_json_full_output`)
    + the existing in-file unit tests + new tests for the f16 / i8
    encoder branches that the previous monolith left uncovered.
  - `validate.rs` (93.55%) — pure-function validators
    (`collect_scan_layers`, `validate_residual`, `validate_owned`)
    with 7 new in-file unit tests.
  - `dispatch.rs` (98.39%) — JSON entry (`run_walk_ffn` +
    `run_full_output` / `run_features_only`).
  - `handler.rs` (86.90% — debt baseline) — axum entrypoint
    `handle_walk_ffn`. The 13% uncovered are error paths inside
    `tokio::task::spawn_blocking` closures (no-model-loaded,
    read-body-error, JSON-serialize-error) that are hard to drive
    from integration tests without a special harness.
  - `core.rs` (24.89% — still excluded) — `run_full_output_core`.
    The MoE branches (~80 lines) need a remote-MoE backend the
    synthetic fixture doesn't provide.
  - `q8k.rs` (34.18% — still excluded) — `handle_walk_ffn_q8k`. Needs
    a Q4K-quantised vindex; same fixture gap.
  - `mod.rs` — re-exports preserving `crate::routes::walk_ffn::X`
    paths so callers (`routes/mod.rs`, `openapi.rs`) didn't have
    to change. utoipa's generated `__path_*` types are re-exported
    so the `#[derive(OpenApi)]` macro keeps finding them at the
    pre-split path.

### Coverage policy

- `routes/walk_ffn.rs` removed from `exclude_globs`; replaced with
  `routes/walk_ffn/core.rs` + `routes/walk_ffn/q8k.rs` (the two
  submodules that need MoE / Q4K fixtures to cover).
- `routes/walk_ffn/handler.rs` added at 86% debt baseline.
- Gate: total 73.76% lines, included 92.42%, 42 files checked,
  31 at default 90%, 11 debt baselines. **Up from 36 files / 26 at
  default / 10 baselines pre-split.**

### Tests

- 787 server tests pass (was 748 at start of session, +39 from
  walk_ffn split's in-file unit tests + coverage tests + new
  validate tests). No regressions across `test_grid_*`,
  `test_http_*`, `test_synthetic_vindex`, `test_walk_ffn_coverage`.

## [2026-05-17] — Synthetic-vindex test fixture + coverage push begins

Phase 1 of the larql-server coverage push (target: total ≥ 90%). Built
a reusable test fixture that constructs a complete f32 vindex on disk
from synthetic deterministic weights, then drove the pilot file
`routes/explain.rs` from 44.86% → 93.46% (clears the 90% per-file
floor). Total server coverage 69.82% → 71.18%; included-files
coverage 91.87% → 91.98%.

### Added

- **`tests/common/synthetic_vindex.rs`** — `build()` produces a tempdir
  with `index.json` (`has_model_weights: true`), `weight_manifest.json`,
  `gate_vectors.bin`, `attn_weights.bin`, `up_weights.bin`,
  `down_weights.bin`, `norms.bin`, `lm_head.bin`, `embeddings.bin`,
  `down_meta.bin`, and `tokenizer.json` — exactly what
  `larql_vindex::load_model_weights_with_opts` consumes. Synthetic
  weights match `larql-vindex/tests/test_vindex.rs::make_synthetic_model`:
  2 layers × hidden=8 × intermediate=4, vocab=16. Build time ~10ms.
- **`tests/common/mod.rs::model_with_real_weights(id)`** — returns
  `(Arc<LoadedModel>, SyntheticVindex)`. `LoadedModel.path` points at
  the fixture so `get_or_load_weights()` (called by every
  `full_output=true` route handler) succeeds. Sibling
  `_and_labels(id, labels)` variant seeds `probe_labels` for tests of
  the `relations_only` branches.
- **`tests/test_synthetic_vindex.rs`** — 9 tests:
  fixture smoke (2) + explain-handler coverage (7 — basic, attention,
  relations_only, relations + labels, band filter, multi-model,
  multi-model 404, invalid JSON). Runs in ~30ms.

### Changed

- **`coverage-policy.json`** — `routes/explain.rs` removed from
  `exclude_globs`; per-file 90% default now applies. 27 files (was 26)
  clear the default; 10 debt baselines unchanged. Total floor
  unchanged at 65%, included floor unchanged at 90%.

### Playbook for next sessions (Tasks #94 — remaining 7 excluded files)

Pattern that worked on explain.rs:

1. Pick the next excluded file from `coverage-policy.json::exclude_globs`.
   Recommended order by ROI: `routes/walk_ffn.rs` (874 missed lines, biggest
   single gain), `routes/openai/chat.rs` (547), `routes/openai/completions.rs`
   (303), `routes/stream.rs` (421), `routes/infer.rs` (195),
   `routes/expert/*` (5 files at 0-57%), `grpc.rs` (302).
2. Write one smoke test using `common::model_with_real_weights` that POSTs
   to the file's main route handler. Measure coverage delta.
3. Add 4-6 more tests targeting uncovered branches surfaced by
   `cargo llvm-cov report --package larql-server --json`. The vocab in
   `synthetic_vindex.rs` is small — most uncovered ranges are
   reachable by adjusting query params (`band`, `relations_only`,
   `with_attention`, `top_k`, full_output flags) or by giving the
   fixture a payload it can run through.
4. When the file clears 90%, remove it from `exclude_globs` and
   confirm `make larql-server-coverage` passes.

Caveats observed during the pilot:

- The fixture's tokenizer needs at least a small WordLevel vocab.
  An empty BPE encodes every prompt to 0 tokens; every per-token
  branch in the route handler then stays uncovered. The shipped
  fixture uses 12 WordLevel entries; adjust as needed.
- The fixture's intermediate / hidden sizes are tiny on purpose
  (build time matters). If a route needs larger shapes to exercise a
  specific branch (e.g. multi-head attention paths), bump
  `ModelDims` in `make_weights()`.
- `LoadedModel` is `!Clone`; pass `probe_labels` at construction
  via `model_with_real_weights_and_labels(id, labels)` rather than
  mutating after `Arc::new`.

## [2026-05-16] — Mode B / QUIC ROADMAP backfill + GT5 end-to-end test

ROADMAP-drift sweep: three G-MODEB / G-TRANSPORT items previously
listed as "Not started" were actually shipped between 2026-05-13 and
2026-05-15 (on the router side) and earlier on the server side. The
server ROADMAP was updated to reflect reality and the missing
end-to-end test was added.

### Fixed

- **GT5 (Mode B gap-fill) — server-side ROADMAP corrected + new
  end-to-end test.** `announce.rs::run_available_loop` had been wired
  end-to-end (`AvailableMsg` → handle `AssignMsg` →
  `shard_loader::download_and_load_shard` → `ReadyMsg` /
  `RefuseMsg` → loop until `AckMsg`) since 2026-05-13, but no
  integration test drove the *production* loop —
  `mode_b_full_vertical_handoff` inlined the protocol in the test
  body. New test
  `mode_b_try_once_available_drives_full_handshake` spawns the real
  loop via the newly-public `announce::try_once_available` entry
  point against an in-process router fixture and asserts Available →
  Assign → download → Ready → Ack lands in <3s.
- **Misleading Mode A AssignMsg log.** `announce.rs:413` used to log
  `"Received AssignMsg but Mode B not implemented — ignoring"` when
  a Mode A (already-serving) stream received an unexpected AssignMsg.
  Mode B *is* implemented, in `run_available_loop`; the stub message
  was misleading. Now logs a descriptive warning calling out that
  the router shouldn't target Mode A streams with AssignMsg.
- **Three stale ROADMAP entries marked shipped.** GT5, GT6 (dynamic
  rebalancing / drain-then-reassign — ADR-0011 §Phase B2), and GT7
  (QUIC transport — ADR-0010) all moved from `Not started` →
  `✅ Shipped` with code pointers and test references.
- **Three integration tests un-bit-rotted.**
  `tests/test_grid_mode_b.rs`, `tests/test_grid_replication.rs`,
  `tests/test_grid_drain_reassign.rs` had been broken since
  ADR-0018 (MoE expert routing) widened `try_assign_gap` to take
  `expert_start` / `expert_end` and moved `GridServiceImpl` to
  `larql_router::grid::service`. Patched all three (new
  signatures + import paths + `parking_lot::RwLock` for `GridState`
  to mirror the router's 2026-05-16 lock primitive swap). 9 tests in
  3 files all pass.
- **`parking_lot` added as a server dev-dependency.** Mirrors the
  router's `GridState` lock primitive so test fixtures can construct
  an `Arc<parking_lot::RwLock<GridState>>` directly.

### Known follow-up

- **GT5 hash-verification mismatch (P1).** `vindex_identity_hash`
  emits a 16-hex model-identity tag, but `shard_loader` expects a
  SHA-256 content hash on `AssignMsg.shard_hash`. Today deployments
  must pass an empty/placeholder hash so the verification is
  skipped. Real content hashing wants a new optional
  `shard_content_sha256` field on `AnnounceMsg` distinct from
  `vindex_hash`. See `ROADMAP.md` G-MODEB §GT5 "Known follow-up".

## [2026-05-10] — Code-review P0 sweep + coverage scaffolding

Five P0 fixes from the in-tree code review (REV1–REV5 in `ROADMAP.md`)
plus the missing larql-server Makefile coverage targets and a per-file
90% coverage policy.

### Fixed

- **REV1 — gRPC sort panics on NaN scores.** `grpc_describe` and
  `grpc_select` used `partial_cmp(...).unwrap()`, which panics on NaN.
  Replaced both call sites with a shared `cmp_score_desc(a, b)` helper
  that maps NaN → `Ordering::Equal`. A corrupted vindex or a future
  patched-scoring path that produces NaN no longer takes a gRPC worker
  down. Five new unit tests in `grpc.rs` lock the property.
- **REV2 — Non-constant-time API key comparison.** `auth.rs` used
  `==` on `&str`, which short-circuits and leaks bytewise progress
  through request timing. Tokens are now SHA-256-hashed and the digests
  compared via `subtle::ConstantTimeEq`. Module-level doc block names
  the threat model. `subtle` (already in the lockfile via rustls)
  added as a direct dep. Six new unit tests in `auth.rs`; six existing
  `http_auth_*` integration tests still pass with no behavioural
  change.
- **REV3 — `blocking_read` on tokio RwLock inside async path.**
  `SessionManager::apply_patch` previously called
  `model.patched.blocking_read()` while holding `sessions.write().await`
  on a worker thread, which on a multi-thread runtime stalls the
  worker (and risks deadlock against any task acquiring those locks
  in the opposite order). Restructured into fast-path / slow-path:
  the slow path drops the sessions write guard, awaits
  `model.patched.read()`, then re-acquires and uses
  `entry().or_insert_with(...)` to absorb the race where another task
  inserted the same `session_id`. No `blocking_read`/`blocking_write`
  on tokio locks is reachable from an `async fn` in `session.rs`
  anymore. Two new regression tests assert (a) forward progress when
  another task holds a `patched.read()` and (b) 16-way concurrent
  `apply_patch` on the same `session_id` finishes within a bounded
  deadline.
- **REV4 — OpenAI error envelope diverged from spec.** Non-streaming
  responses on `/v1/embeddings`, `/v1/completions`, and
  `/v1/chat/completions` returned `{"error": "msg"}` (flat); the OpenAI
  Python and JS SDKs expect
  `{"error": {"message", "type", "param", "code"}}` (nested) and broke
  on field access against the flat shape. Streaming SSE error chunks
  already used the nested form, so non-stream and stream errors were
  inconsistent. Introduced a new `OpenAIError` type with constructor
  helpers (`invalid_request`, `not_found`, `service_unavailable`,
  `server_error`) and an `IntoResponse` that renders the canonical
  nested envelope with `param`/`code` always present (possibly null).
  `From<ServerError>` lets internal helpers keep `ServerError` and
  propagate via `?`. The three OpenAI handler entry-point return
  types flipped to `Result<_, OpenAIError>` and 16 direct
  `return Err(ServerError::X(...))` sites converted to the matching
  `OpenAIError::Y(...)` constructor. LARQL paradigm endpoints keep the
  flat envelope. Six integration tests assert the nested shape on
  400/503 paths across the three handlers; seven unit tests cover the
  type itself.
- **REV5 — tool-call JSON parser surfaced 500 instead of 400 on
  malformed nested-brace output.** `build_tool_call_message` used
  `find('{')` + `rfind('}')` to extract JSON from constrained-decoder
  output, which silently picked the wrong slice on trailing junk /
  multiple objects / markdown wrappers and surfaced the parse failure
  as `ServerError::Internal` (500). Rewrote as a straight-line
  `serde_json::from_str(text.trim())` with structured diagnostics
  (`invalid JSON: …`, `tool output must be a JSON object`, missing-
  field reports), and flipped the call-site error class from
  `Internal` to `OpenAIError::invalid_request` so the client now sees
  **400 invalid_request_error** with a concrete message. Nine new
  unit tests cover happy path, surrounding whitespace, nested-brace
  arguments, trailing junk, empty/whitespace, non-object top-level,
  missing `name`/`arguments`, and invalid JSON.

### Added

- **Two-envelope error documentation.** `docs/server-spec.md §8.3.1`
  rewritten with the LARQL-flat / OpenAI-nested split and a canonical
  `type` table. README `Error Codes` section updated to match.
- **Makefile coverage targets** for larql-server, mirroring the
  larql-compute / larql-vindex pattern:
  `larql-server-test`, `larql-server-fmt-check`, `larql-server-lint`,
  `larql-server-coverage`, `larql-server-coverage-summary`,
  `larql-server-coverage-html`, `larql-server-coverage-policy`,
  `larql-server-ci`. Threshold variables: `LARQL_SERVER_COVERAGE_MIN`
  (default 65 — current baseline), `LARQL_SERVER_COVERAGE_POLICY`,
  `LARQL_SERVER_COVERAGE_REPORT`.
- **`coverage-policy.json`** with default 90% line floor, 28 per-file
  debt baselines snapshotted from the 2026-05-10 measurement, and the
  total floor at the measured 65.6% baseline. Policy semantics
  ratchet upward only — new / split files automatically inherit the
  90% default.

### Internal

- Cleared 5 pre-existing clippy errors in lib (`bootstrap.rs:230`
  boolean simplification, `metrics.rs:64` missing `Default` for
  `LayerLatencyTracker`, `walk_ffn.rs` doc indentation + needless
  lifetimes + redundant closure). `cargo clippy -p larql-server --lib
  --no-deps -- -D warnings` now clean.
- Updated `tests/test_expert_endpoint.rs` import: `cpu_moe_forward`
  and `MoeLayerWeights` moved from `larql_inference` to
  `larql_compute` in the upstream refactor; the test had a stale
  import that blocked `--tests` builds. Pure plumbing — matches the
  cargo error hint.
- Added `#[derive(Debug)]` to `ChatChoiceMessage`, `ToolCall`,
  `ToolCallFunction` to support `Result::unwrap_err()` in the new
  `build_tool_call_message` tests.

### Coverage snapshot (2026-05-10)

- **TOTAL**: 65.68% line / 72.18% function / 64.90% region.
- **At-or-above 90% default**: `routes/openai/error.rs` (100%),
  `routes/openai/util.rs` (99.6%), `routes/openai/embeddings.rs`
  (93.2%), `session.rs` (96.1%), `state.rs` (85.8% — debt baseline),
  `auth.rs` (98.0%), `wire.rs` (96.9%), `etag.rs` (100%), and 16
  others.
- **Largest debt items** (all carry baselines, must ratchet up):
  `routes/expert/{batch_legacy,multi_layer_batch,single,warmup}.rs`
  at 0% (need a live grid harness),
  `routes/openai/schema/mask.rs` at 0%, `bootstrap.rs` at 29.7%,
  `routes/openai/completions.rs` at 40.3%, `routes/walk_ffn.rs` at
  49.0%, `routes/openai/chat.rs` at 53.4%.
