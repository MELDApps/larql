# larql-kv

Pluggable KV-cache engines for `larql-inference`. Each engine implements the
full prefill + autoregressive decode loop but manages persistent inference
state differently — trading memory, accuracy, and speed.

The `KvEngine` trait + `EngineInfo` + `DecodeStageSummary` live in
`larql-inference::kv_engine`; this crate re-exports them so
`larql_kv::KvEngine` continues to work as the public surface. The trait
lives upstream so `larql-inference`'s decode dispatch
(`generate_with_engine`) can reference it without a circular dep. See
[`crates/larql-inference/docs/specs/kv-engine-unification.md`](../larql-inference/docs/specs/kv-engine-unification.md)
for the dep-graph rationale.

## Engine ladder

Six engines: `Standard` and `NoCache` wrap today's production behaviour;
the other four are the research engines that trade accuracy for memory
or speed.

| Engine | Mechanism | KV memory | Accuracy |
|---|---|---|---|
| [`standard`](src/engines/standard.rs) | Production K/V tensor cache, `window=None` = unbounded, `Some(N)` = sliding window | O(layers × seq × kv_dim × 4B) | exact — the reference |
| [`no_cache`](src/engines/no_cache.rs) | No K/V; full re-forward per step (O(N²)) | O(seq) token IDs only | exact — correctness fallback |
| [`markov_residual`](src/engines/markov_residual) | Residual-stream replacement, K/V recomputed | ~171 MB on Gemma 3 4B | exact (KL = 0.0) under contract |
| [`unlimited_context`](src/engines/unlimited_context) | Per-window K/V checkpoints | ~193 MB | exact within window |
| [`turbo_quant`](src/engines/turbo_quant) | WHT + Lloyd-Max 3/4-bit codec | ~12.7 GB on 370K-token corpus | cos ≈ 0.991 |
| [`apollo`](src/engines/apollo) | Boundary store + residual injection | ~11 MB on 370K-token corpus | task accuracy (first-token factual) |

Speed numbers (Gemma 3 4B, M3 Max, Metal Q4K) and compression ratios for
the research engines live in [`PERFORMANCE.md`](PERFORMANCE.md). Reference
for "compression" is full f16 KV at 49 GB on the same corpus.

### Standard vs MarkovResidual

`Standard` (this crate's wrapper over the production K/V cache) and
`MarkovResidual` (residual-stream replacement) are **different
mechanisms** that happen to produce bit-identical output on supported
architectures. Don't conflate them — the CLI's historical `--kv-cache
markov-bounded` flag maps to `Standard { window_size: Some(N) }`, **not**
`MarkovResidual`. Use the spec's table in §5 when in doubt.

## Usage

```rust
use larql_kv::{EngineKind, KvEngine};
use larql_compute::default_backend;
use larql_inference::ffn::WeightFfn;

// Parse a CLI engine spec.
let kind = EngineKind::from_name("standard:window=512").unwrap();

// Build an engine bound to a compute backend.
let mut engine: Box<dyn KvEngine> = kind.build(default_backend());

// FFN router — `WeightFfn` reads weights locally; pass any FfnBackend
// impl (e.g. `RemoteWalkBackend` for remote-FFN dispatch).
let ffn = WeightFfn { weights: &weights };

// Prefill, then decode autoregressively. Prefer the
// `larql_inference::forward::generate_with_engine` helper, which drives
// the same prefill + sample + decode loop legacy callers use.
let generated = larql_inference::forward::generate_with_engine(
    engine.as_mut(),
    &weights,
    &tokenizer,
    &ffn,
    &prompt_tokens,
    max_tokens,
    |id, tok| { /* on-token callback */ },
);
```

The engines also expose Q4K-quantised entry points
(`prefill_q4k` / `decode_step_q4k`) that route through the Metal
`decode_token` pipeline when a Q4K `VectorIndex` and a Metal backend are
available, falling back to the f32 path otherwise.

## CLI selectors

The CLI parses engine specs as `name` or `name:key=value[,key=value]`:

```text
standard                                  # production K/V cache, unbounded (default)
standard:window=1024                      # sliding-window K/V
no-cache                                  # full re-forward per step (O(N²)), debug only
markov-rs                                 # residual-stream replacement
markov-rs:window=1024
unlimited-context:window=256
turbo-quant:bits=3        # alias: tq3
turbo-quant               # bits=4 default; alias: tq4
apollo:layer=25,coef=8.0,top_k=12
```

Legacy aliases for `standard`: `full`, `fp32`. Legacy aliases for
windowed standard: `markov-bounded`, `bounded`, `sliding`. Legacy aliases
for `no-cache`: `none`, `off`.

All engines are reachable via `larql bench <model> --engine <spec>`. As
of the unification migration, `larql run` / `larql walk` / `larql-server`
will also accept `--engine` (currently behind `LARQL_KV_ENGINE_DISPATCH=1`
for parity-testing; see the unification spec for the rollout plan).

## Crate layout

```
larql-kv/
├── src/
│   ├── lib.rs          — EngineKind dispatch + re-exports of the trait surface
│   ├── accuracy.rs     — cosine, MSE, KL, JS, compare_hidden helpers
│   ├── profiler.rs     — per-stage decode timing accumulators
│   └── engines/
│       ├── standard.rs           — production K/V tensor cache (default)
│       ├── no_cache.rs           — full re-forward per step (debug fallback)
│       ├── apollo/               — boundary-residual injection, ~4,000× compression
│       ├── markov_residual/      — residual-stream KV replacement, KL = 0
│       ├── turbo_quant/          — WHT + Lloyd-Max K/V codec (3- or 4-bit)
│       └── unlimited_context/    — windowed re-prefill from checkpoints
├── benches/            — criterion microbenchmarks
├── examples/           — end-to-end demos on synthetic test_utils
└── coverage-policy.json — per-file ≥90% line-coverage policy
```

The `KvEngine` trait itself lives in
[`larql-inference/src/kv_engine.rs`](../larql-inference/src/kv_engine.rs).

## Architecture notes

- **Metal Q4K path.** All four engines route through the Metal
  `decode_token` full pipeline when a Q4K `VectorIndex` and Metal backend
  are available — 93–95 tok/s on Gemma 3 4B, matching the standard
  larql-metal path.
- **CPU fallback.** When Metal is unavailable, engines fall back to a CPU
  path using dequantised attention tensors (lazily inserted into
  `weights.tensors`) and `WalkFfn` for Q4K FFN.
- **Apollo compressed path.** When the store has boundary residuals
  captured at `crystal_layer` (default 30), `forward_from_layer` runs only
  `crystal_layer..num_layers` layers (~4 instead of 34), ~8.5× faster per
  step.

## Relationship to other crates

- **`larql-inference`** — provides the transformer primitives that engines
  compose (`attention::*`, `forward::*`, `ffn::BackendFfn`,
  `vindex::WalkFfn`, `model::ModelWeights`, `residual::*`,
  `layer_graph::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ`).
- **`larql-compute`** — the `ComputeBackend` trait engines dispatch through.
- **`larql-vindex`** — the `VectorIndex` engines query for Q4K weights.
- **`kv-cache-benchmark`** — criterion-driven comparison of all engines plus
  baselines (Standard KV, Graph Walk).

## History

Extracted from `larql-inference::engines` on 2026-05-09. See
[`CHANGELOG.md`](CHANGELOG.md). Forward-looking work in
[`ROADMAP.md`](ROADMAP.md).
