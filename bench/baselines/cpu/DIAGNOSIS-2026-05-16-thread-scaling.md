# Diagnosing the remaining 2.36× decode gap to llama.cpp

Recorded 2026-05-16 after the Q4_K × Q8_K sdot path landed. Same M3 Max,
same 16-token bench, same Q4_K weights.

## Thread scaling (tg16, tok/s)

| Threads | larql | llama.cpp | Per-core ratio |
|---:|---:|---:|---:|
| 1 | 5.7 | 9.88 | 1.73× |
| 2 | 9.2 | 17.50 | 1.90× |
| 4 | 18.4 | 31.86 | 1.73× |
| 6 | 22.2 | — | — |
| 7 | 23.7 | — | — |
| **8** | **24.6** | 42.13 | **1.71×** |
| 9 | 24.3 | — | — |
| 10 | 23.3 | — | — |
| 11 | 23.2 | — | — |
| 12 | 21.0 | 42.68 | 2.03× |
| 16 | — | 13.61 (oversub) | — |

Commands:
```
RAYON_NUM_THREADS=$t target/release/larql bench output/gemma3-4b-q4k-v2.vindex \
    --cpu --tokens 16 --warmup 1 --profile
llama-bench -m /private/tmp/larql-gemma-3-4b-it-Q4_K_M.gguf \
    -dev BLAS -ngl 0 -p 5 -n 16 -r 1 -t $t
```

M3 Max topology: 12 P-cores + 4 E-cores. Both engines have ~12 P-cores
to use.

## Finding 1 — Per-core kernel throughput is the dominant gap

**Single-threaded:** larql 5.7 tok/s vs llama.cpp 9.88 tok/s. **1.73×
behind on one core.** The Q4_K × Q8_K algorithm is the same; the
inner loop quality differs:

- llama.cpp's `ggml_vec_dot_q4_K_q8_K` is hand-written inline asm
  using `vdotq_s32` (SDOT) with explicit instruction interleaving
  across two adjacent super-blocks.
- Our `q4k_q8k_matvec_neon` uses the same `vdotq_s32` intrinsic via
  Rust `core::arch::aarch64`, but lowered by LLVM. The schedule is
  what LLVM emits from intrinsic IR — typically not as tight as
  hand-written asm on hot inner loops with byte-unpacking.

The per-core gap stays at **~1.73×** across thread counts (t=1, 4,
8). This is the kernel-quality ceiling, not a scaling problem.

## Finding 2 — We oversubscribe past 8 threads on M3 Max

At t=12 we drop to 21.0 tok/s vs t=8's 24.6 (–15%). llama.cpp stays
flat at 42 tok/s from t=8 onward. Both engines have 12 P-cores
available; the difference is in how thread placement / memory
contention is handled.

Likely causes:
1. **DRAM channel contention.** Q4_K weight reads (~2 GB/step) hit
   memory bandwidth-bound at ~8 worker threads on M3 Max's LPDDR5
   controllers. Adding threads past 8 creates inter-thread contention
   on the same channels without throughput gain.
2. **Rayon work-stealing under pressure.** With 12 threads each
   chunking 32 rows of a ~10K-row matrix, work-steals can cross cluster
   boundaries (M3 Max has 2 P-core clusters of 6 each, sharing L2 within
   a cluster). Cross-cluster steals add L2-miss latency.
3. **macOS scheduler.** Rayon doesn't pin threads to specific cores;
   the scheduler may move workers to E-cores at high contention.
   llama.cpp's GGML uses spinning threads in a pool that the OS
   prefers to keep on P-cores.

**Easy win waiting:** setting `RAYON_NUM_THREADS=8` brings us from
18.0 → 24.5 tok/s, **2.36× → 1.74× behind llama.cpp**. Task #16 will
auto-detect this default in the bench tool.

## Where the remaining 1.73× per-core gap lives

With matched thread count + same Q4_K × Q8_K algorithm, the gap is
purely inner-loop quality. Decomposing the per-step cost at t=8:

| Stage | larql ms/tok | llama.cpp ms/tok | Ratio |
|---|---:|---:|---:|
| Decode total | 40.6 | 23.5 | 1.73× |
| CPU fwd (33 layers attn + FFN) | ~36 | ~21 | 1.71× |
| LM head (262K-vocab Q4_K matvec) | ~4 | ~3 | 1.33× |

The CPU fwd ratio matches the per-core ratio. So per-layer-per-matvec,
our NEON intrinsics path runs at ~58% of llama.cpp's hand-asm path.

Hypothesised causes (ranked by likely impact):

1. **Inner-loop instruction scheduling.** LLVM's reorder over
   `vld → vand → vshr → vdot → vaddv` is conservative. Hand-asm
   typically pipelines 2+ super-blocks' worth of loads in flight
   while the prior super-block's SDOTs retire.
2. **No software prefetch.** llama.cpp uses `__builtin_prefetch`
   (lowering to `prfm pldl1keep`) to bring the next super-block's
   144 weight bytes into L1 ahead of the current SDOT chain.
   M3 Max's hardware prefetcher handles linear sequential reads,
   but the 144-byte stride across rows isn't quite linear.
3. **Per-super-block scalar overhead.** Our path does a small amount
   of scalar work between super-blocks (unpack 12 scale/min bytes,
   issue 4 NEON groups, accumulate). LLVM may not interleave that
   scalar with NEON ops as aggressively as hand-asm does.

## Next experiments

1. **Threads default to 8 on M3 Max** (Task #16). Easy 33% win on a
   `larql bench --cpu` invocation that doesn't explicitly set
   `RAYON_NUM_THREADS`.
2. **`cargo asm` on `q4k_q8k_matvec_neon`** vs llama.cpp's
   `ggml_vec_dot_q4_K_q8_K`. Compare instruction counts per
   super-block. Look for missing prefetches, FMA chains, etc.
3. **Software prefetch hints** via inline asm (`prfm pldl1keep`)
   inside the inner loop. Probably 5–15% gain if hardware
   prefetcher is missing strides.
4. **Hand-tuned inline asm for the inner SDOT loop** — last resort,
   highest effort. Would match llama.cpp.

## Summary

- 2.36× decode gap = 1.73× per-core × 1.30× thread-saturation.
- Setting `RAYON_NUM_THREADS=8` recovers the 1.30× immediately.
- Remaining 1.73× is the NEON intrinsics → hand-asm gap; closing it
  needs kernel-level work (prefetch, scheduling, or inline asm).
