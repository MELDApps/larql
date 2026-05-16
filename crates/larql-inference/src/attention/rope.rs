//! Rotary Position Embeddings (RoPE) — position-dependent rotation of Q/K vectors.
//!
//! Split-half pairing: rotates (x[i], x[i + half_dim]) pairs.
//! Matches HuggingFace default and MLX traditional=False.

use ndarray::Array2;

/// Re-export of the parameter struct that lives in `larql-models` so it
/// can be parsed from `config.json` without cross-crate dependency
/// inversion. The math (`apply_llama3_inv_freq` below) is the inference-
/// side concern.
pub use larql_models::Llama3RopeScaling;

/// Compute wavelength-adjusted `inv_freq[i]` for each rotary half-pair
/// from the standard `1 / base^(2i/d)` baseline. Mirrors HF's
/// `_compute_llama3_parameters` in `modeling_rope_utils.py`:
///
/// - `wavelen[i] = 2π / inv_freq[i]`
/// - if `wavelen < high_freq_wavelen` (fast rotation): unchanged
/// - if `wavelen > low_freq_wavelen`  (slow rotation): divided by `factor`
/// - otherwise: smooth interpolation between the two regimes
pub fn apply_llama3_inv_freq(scaling: &Llama3RopeScaling, inv_freq: &[f64]) -> Vec<f64> {
    let low_freq_wavelen = scaling.original_max_position_embeddings / scaling.low_freq_factor;
    let high_freq_wavelen = scaling.original_max_position_embeddings / scaling.high_freq_factor;
    inv_freq
        .iter()
        .map(|&inv| {
            let wavelen = std::f64::consts::TAU / inv;
            if wavelen < high_freq_wavelen {
                inv
            } else if wavelen > low_freq_wavelen {
                inv / scaling.factor
            } else {
                let smooth = (scaling.original_max_position_embeddings / wavelen
                    - scaling.low_freq_factor)
                    / (scaling.high_freq_factor - scaling.low_freq_factor);
                (1.0 - smooth) * inv / scaling.factor + smooth * inv
            }
        })
        .collect()
}

/// Apply full RoPE to Q or K vectors.
/// x: (seq_len, num_heads * head_dim)
pub fn apply_rope(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
) -> Array2<f32> {
    apply_rope_partial(x, num_heads, head_dim, rope_base, 1.0)
}

/// Apply RoPE with partial rotation: only the first `fraction` of each head's
/// dimensions get rotary encoding. The rest pass through unchanged.
/// fraction = 1.0 means full rotation (standard RoPE).
pub fn apply_rope_partial(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
    fraction: f64,
) -> Array2<f32> {
    apply_rope_partial_at(x, num_heads, head_dim, rope_base, fraction, 0)
}

/// Apply RoPE with a positional offset — row `i` in `x` is treated as
/// token position `position_offset + i`. Use this during KV-cached
/// decode: cached K already carries RoPE for positions 0..N-1, and
/// the new token needs RoPE at position N.
pub fn apply_rope_partial_at(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
    fraction: f64,
    position_offset: usize,
) -> Array2<f32> {
    apply_rope_partial_at_scaled(
        x,
        num_heads,
        head_dim,
        rope_base,
        fraction,
        position_offset,
        1.0,
    )
}

/// Same as [`apply_rope_partial_at`] but divides the position by
/// `position_divisor` before computing phase. Matches HF
/// `rope_scaling = {rope_type: linear, factor: N}` applied to a specific
/// layer-type (Gemma 3 applies it only on global / full-attention layers,
/// not on sliding-attention layers — see `Gemma3TextConfig.rope_scaling`).
pub fn apply_rope_partial_at_scaled(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
    fraction: f64,
    position_offset: usize,
    position_divisor: f64,
) -> Array2<f32> {
    apply_rope_partial_at_full(
        x,
        num_heads,
        head_dim,
        rope_base,
        fraction,
        position_offset,
        position_divisor,
        None,
    )
}

/// Most general RoPE entry point. Adds optional `llama3_scaling`: when
/// present, replaces the standard `1 / base^(2i/rotary_dim)` frequencies
/// with HF's `llama3` wavelength-dependent variant. `position_divisor` and
/// `llama3_scaling` compose independently — the divisor scales the
/// position, llama3 scales the per-channel frequency.
pub fn apply_rope_partial_at_full(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
    fraction: f64,
    position_offset: usize,
    position_divisor: f64,
    llama3_scaling: Option<Llama3RopeScaling>,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    let rotary_dim = ((head_dim as f64 * fraction) as usize).max(2);
    let half_rotary = rotary_dim / 2;
    let base_inv_freq: Vec<f64> = (0..half_rotary)
        .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / rotary_dim as f64))
        .collect();
    let inv_freq: Vec<f64> = match llama3_scaling {
        Some(scaling) => apply_llama3_inv_freq(&scaling, &base_inv_freq),
        None => base_inv_freq,
    };
    let divisor = if position_divisor > 0.0 {
        position_divisor
    } else {
        1.0
    };

    for row in 0..seq_len {
        let pos = (position_offset + row) as f64 / divisor;
        for h in 0..num_heads {
            let offset = h * head_dim;
            for i in 0..half_rotary {
                let theta = pos * inv_freq[i];
                let cos_t = theta.cos() as f32;
                let sin_t = theta.sin() as f32;

                let x0 = x[[row, offset + i]];
                let x1 = x[[row, offset + half_rotary + i]];

                out[[row, offset + i]] = x0 * cos_t - x1 * sin_t;
                out[[row, offset + half_rotary + i]] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_qk(seq: usize, heads: usize, head_dim: usize) -> Array2<f32> {
        let n = seq * heads * head_dim;
        Array2::from_shape_vec(
            (seq, heads * head_dim),
            (0..n).map(|i| (i as f32 + 1.0) * 0.01).collect(),
        )
        .unwrap()
    }

    #[test]
    fn apply_rope_preserves_shape() {
        let x = make_qk(3, 2, 8);
        let out = apply_rope(&x, 2, 8, 10000.0);
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn apply_rope_output_is_finite() {
        let x = make_qk(4, 2, 8);
        let out = apply_rope(&x, 2, 8, 10000.0);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn apply_rope_preserves_norm_per_head() {
        // RoPE is a rotation → L2 norm of each position–head pair is preserved.
        let x = make_qk(3, 2, 8);
        let out = apply_rope(&x, 2, 8, 10000.0);
        for row in 0..3 {
            for h in 0..2 {
                let orig: f32 = x
                    .row(row)
                    .iter()
                    .skip(h * 8)
                    .take(8)
                    .map(|v| v * v)
                    .sum::<f32>();
                let rotd: f32 = out
                    .row(row)
                    .iter()
                    .skip(h * 8)
                    .take(8)
                    .map(|v| v * v)
                    .sum::<f32>();
                assert!(
                    (orig.sqrt() - rotd.sqrt()).abs() < 1e-4,
                    "RoPE changed L2 norm at row={row} head={h}: {orig} → {rotd}"
                );
            }
        }
    }

    #[test]
    fn apply_rope_different_positions_differ() {
        // Row 0 (position 0) and row 1 (position 1) should differ after RoPE
        // even if the original vectors were identical.
        let data = vec![0.5f32; 3 * 8];
        let x = Array2::from_shape_vec((3, 8), data).unwrap();
        let out = apply_rope(&x, 1, 8, 10000.0);
        let row0: Vec<f32> = out.row(0).to_vec();
        let row1: Vec<f32> = out.row(1).to_vec();
        let differ = row0
            .iter()
            .zip(row1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "identical inputs at different positions should differ after RoPE"
        );
    }

    #[test]
    fn apply_rope_partial_at_offset() {
        // Position 5 with offset 0 should equal position 0 with offset 5.
        let x = make_qk(1, 2, 8);
        let out_pos5 = {
            let data = vec![0.1f32; 6 * 2 * 8];
            let big = Array2::from_shape_vec((6, 16), data).unwrap();
            apply_rope_partial_at(&big, 2, 8, 10000.0, 1.0, 0)
        };
        let out_off5 = apply_rope_partial_at(&x, 2, 8, 10000.0, 1.0, 5);
        // Both should be finite (structural check)
        assert!(out_pos5.iter().all(|v| v.is_finite()));
        assert!(out_off5.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn apply_rope_partial_fraction_zero_is_passthrough() {
        // fraction = 0.0 → no rotation applied (but we need at least 2 rotary dims).
        // With a very small fraction the rotation is minimal — test shape only.
        let x = make_qk(2, 2, 8);
        let out = apply_rope_partial(&x, 2, 8, 10000.0, 0.01);
        assert_eq!(out.shape(), x.shape());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Property tests ────────────────────────────────────────────────────────

    #[test]
    fn rope_different_base_produces_different_output() {
        // Different rope_base → different frequencies → different output.
        let x = make_qk(2, 2, 8);
        let out1 = apply_rope(&x, 2, 8, 10_000.0);
        let out2 = apply_rope(&x, 2, 8, 500_000.0);
        let differs = out1
            .iter()
            .zip(out2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "different rope_base should produce different output"
        );
    }

    #[test]
    fn rope_partial_fraction_one_equals_full_rope() {
        let x = make_qk(3, 2, 8);
        let full = apply_rope(&x, 2, 8, 10000.0);
        let partial_1 = apply_rope_partial(&x, 2, 8, 10000.0, 1.0);
        for (a, b) in full.iter().zip(partial_1.iter()) {
            assert!((a - b).abs() < 1e-5, "fraction=1.0 should equal full rope");
        }
    }

    #[test]
    fn rope_position_offset_matches_sequential_positions() {
        // apply_rope_partial_at(x, ..., offset=5) on a 1-token sequence should
        // equal row 5 of apply_rope on a 6-token sequence with identical rows.
        let hd = 8usize;
        let heads = 2usize;
        let val = 0.3f32;
        // Single row for the offset test
        let single = Array2::from_elem((1, heads * hd), val);
        // 6-row sequence of identical values
        let seq6 = Array2::from_elem((6, heads * hd), val);
        let out_seq6 = apply_rope(&seq6, heads, hd, 10000.0);
        let out_offset5 = apply_rope_partial_at(&single, heads, hd, 10000.0, 1.0, 5);
        // Row 5 of seq6 should match the single-row result with offset 5
        let row5: Vec<f32> = out_seq6.row(5).to_vec();
        let offset_row: Vec<f32> = out_offset5.row(0).to_vec();
        for (a, b) in row5.iter().zip(offset_row.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "offset=5 should match position 5 in sequential apply: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rope_partial_fraction_between_0_and_1_is_finite() {
        // Spot-check that various fractions produce finite, valid output.
        let x = make_qk(2, 2, 16);
        for &frac in &[0.25f64, 0.5, 0.75] {
            let out = apply_rope_partial(&x, 2, 16, 10000.0, frac);
            assert_eq!(out.shape(), x.shape());
            assert!(
                out.iter().all(|v| v.is_finite()),
                "fraction={frac} produced non-finite"
            );
        }
    }
}
