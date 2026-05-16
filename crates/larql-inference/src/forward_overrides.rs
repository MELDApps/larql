//! Forward-path override surface.
//!
//! This module lives between `larql-models` (which parses model config into
//! the architecture trait) and the inference forward path (CPU + GPU).
//! Each helper here resolves an effective per-layer parameter by checking
//! a diagnostic env-var first, then falling back to whatever the
//! architecture exposes from the parsed `config.json`.
//!
//! ## Why the env vars exist
//!
//! The five env vars below were the diagnostic instruments used to
//! bisect the cross-engine forward-pass divergence documented in
//! [`docs/diagnoses/shannon-cross-engine-divergence.md`](../../../docs/diagnoses/shannon-cross-engine-divergence.md).
//! They are kept in tree even after the underlying loader bugs were fixed
//! so future regressions on new architectures can be localised the same
//! way without touching code. Production runs never need to set any of
//! them — when unset, every helper delegates to the architecture.
//!
//! ## Precedence
//!
//! For each parameter:
//!
//! 1. If the corresponding env var is set and parses, use it.
//! 2. Otherwise call the architecture's accessor on the parsed config.
//! 3. Architecture accessors carry their own defaults
//!    (see [`larql_models::defaults`]) for fields the model's config
//!    omits entirely.
//!
//! ## Env-var reference
//!
//! | Var | Type | Effect |
//! |---|---|---|
//! | `LARQL_FORCE_GLOBAL_LAYERS` | `all` or `<csv>` | Force listed layers onto global rope_base (sliding_window=0). |
//! | `LARQL_ROPE_POS_DIVISOR` | `f64` | Divide RoPE position by this factor on every layer. |
//! | `LARQL_ROPE_POS_DIVISOR_GLOBAL` | `f64` | Same, but only on `!is_sliding_window_layer(layer)`. |
//! | `LARQL_LLAMA3_ROPE_SCALING` | `factor,low,high,old_ctx` | Force HF llama3 scaling params. |
//! | `LARQL_NORM_EPS_OVERRIDE` | `f64` | Override `arch.norm_eps()`. |

use std::sync::OnceLock;

/// Diagnostic override for the sliding-window attention bisection.
///
/// `LARQL_FORCE_GLOBAL_LAYERS=all` forces every layer onto the global-attention
/// code path (sliding_window=0, rope_base = arch's full rope_theta). A comma-
/// separated index list (`LARQL_FORCE_GLOBAL_LAYERS=12,13,14`) targets specific
/// layers. Empty/unset leaves the architecture's per-layer routing untouched.
#[derive(Debug)]
enum ForceGlobalSpec {
    None,
    All,
    Layers(Vec<usize>),
}

fn force_global_spec() -> &'static ForceGlobalSpec {
    static CELL: OnceLock<ForceGlobalSpec> = OnceLock::new();
    CELL.get_or_init(|| match std::env::var("LARQL_FORCE_GLOBAL_LAYERS") {
        Err(_) => ForceGlobalSpec::None,
        Ok(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                ForceGlobalSpec::None
            } else if trimmed.eq_ignore_ascii_case("all") {
                ForceGlobalSpec::All
            } else {
                let layers: Vec<usize> = trimmed
                    .split(',')
                    .filter_map(|tok| tok.trim().parse::<usize>().ok())
                    .collect();
                if layers.is_empty() {
                    ForceGlobalSpec::None
                } else {
                    ForceGlobalSpec::Layers(layers)
                }
            }
        }
    })
}

/// Returns true when `LARQL_FORCE_GLOBAL_LAYERS` requests this layer be
/// forced onto the global-attention code path.
pub fn layer_forced_global(layer: usize) -> bool {
    match force_global_spec() {
        ForceGlobalSpec::None => false,
        ForceGlobalSpec::All => true,
        ForceGlobalSpec::Layers(v) => v.contains(&layer),
    }
}

/// Per-layer rope base honouring the `LARQL_FORCE_GLOBAL_LAYERS` diagnostic
/// override. Use this anywhere the CPU/GPU forward path would otherwise call
/// `arch.rope_base_for_layer(layer)` directly.
pub fn effective_rope_base_for_layer(
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> f64 {
    if layer_forced_global(layer) {
        arch.config().rope_base
    } else {
        arch.rope_base_for_layer(layer)
    }
}

/// Diagnostic position scale read from `LARQL_ROPE_POS_DIVISOR=<f64>`. Matches
/// HF `rope_scaling = {rope_type: linear, factor: <v>}`. Returns `1.0` when
/// the env var is unset. Applied uniformly to every layer.
fn rope_position_divisor() -> f64 {
    static CELL: OnceLock<f64> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("LARQL_ROPE_POS_DIVISOR")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0)
    })
}

/// Diagnostic position scale read from `LARQL_ROPE_POS_DIVISOR_GLOBAL=<f64>`,
/// applied only on global (non-sliding) layers. Gemma 3's HF config sets a
/// linear factor on full-attention layers only via the structured per-layer-
/// type `rope_scaling` form.
fn rope_position_divisor_global_only() -> f64 {
    static CELL: OnceLock<f64> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("LARQL_ROPE_POS_DIVISOR_GLOBAL")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0)
    })
}

/// Diagnostic override for HF `llama3` rope scaling, reading
/// `LARQL_LLAMA3_ROPE_SCALING=factor,low,high,old_ctx` (comma-separated).
/// E.g. `LARQL_LLAMA3_ROPE_SCALING=32,1,4,8192` matches Llama-3.2's config.
/// Returns `None` when the env var is unset or malformed (in which case
/// the arch-driven value from [`effective_llama3_rope_scaling`] is used).
fn llama3_rope_scaling_override() -> Option<larql_models::Llama3RopeScaling> {
    static CELL: OnceLock<Option<larql_models::Llama3RopeScaling>> = OnceLock::new();
    *CELL.get_or_init(|| {
        let raw = std::env::var("LARQL_LLAMA3_ROPE_SCALING").ok()?;
        let parts: Vec<f64> = raw
            .split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        if parts.len() != 4 {
            return None;
        }
        let s = larql_models::Llama3RopeScaling {
            factor: parts[0],
            low_freq_factor: parts[1],
            high_freq_factor: parts[2],
            original_max_position_embeddings: parts[3],
        };
        if s.factor > 0.0
            && s.low_freq_factor > 0.0
            && s.high_freq_factor > 0.0
            && s.original_max_position_embeddings > 0.0
            && s.high_freq_factor > s.low_freq_factor
        {
            Some(s)
        } else {
            None
        }
    })
}

/// Llama3 rope-scaling parameters for the forward pass — env-var override
/// first, then the architecture's parsed `rope_scaling`. Returns `None`
/// when neither is set (no scaling applied).
pub fn effective_llama3_rope_scaling(
    arch: &dyn larql_models::ModelArchitecture,
) -> Option<larql_models::Llama3RopeScaling> {
    llama3_rope_scaling_override().or_else(|| arch.llama3_rope_scaling())
}

/// Diagnostic norm-epsilon override read from `LARQL_NORM_EPS_OVERRIDE=<f64>`.
/// When set, replaces the architecture's `norm_eps()` value at every
/// `rms_norm_for_arch` / `layer_norm_for_arch` call site. Use to test
/// whether a hardcoded default is masking a config that expects a
/// different eps.
pub fn norm_eps_override() -> Option<f32> {
    static CELL: OnceLock<Option<f32>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("LARQL_NORM_EPS_OVERRIDE")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
    })
}

/// Effective per-layer RoPE position divisor.
///
/// Precedence: env-var overrides first (uniform `LARQL_ROPE_POS_DIVISOR` and
/// global-only `LARQL_ROPE_POS_DIVISOR_GLOBAL`), then the architecture's
/// own `rope_position_divisor_for_layer` (which reads the parsed
/// `config.rope_scaling`). Returns 1.0 (no scaling) when nothing applies.
pub fn effective_rope_position_divisor_for_layer(
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> f64 {
    let uniform_env = rope_position_divisor();
    let global_env = rope_position_divisor_global_only();
    if !arch.is_sliding_window_layer(layer) && global_env != 1.0 {
        return global_env;
    }
    if uniform_env != 1.0 {
        return uniform_env;
    }
    // Default: ask the architecture (parsed from rope_scaling in config.json).
    arch.rope_position_divisor_for_layer(layer)
}
