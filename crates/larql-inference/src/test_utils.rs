//! Synthetic test fixtures for engine and layer-graph unit tests.
//!
//! Three helpers:
//! - `make_test_weights()` — fully functional 2-layer ModelWeights (no disk I/O)
//! - `make_test_vindex(weights)` — in-memory VectorIndex with random gate vectors
//! - `make_test_tokenizer(vocab_size)` — WordLevel tokenizer mapping token N to "[N]"
//!
//! Dimensions: vocab=32, hidden=16, intermediate=32, 2 q-heads, 1 kv-head,
//! head_dim=8, 2 layers. Forward pass ≈ 10 ms on CPU.

use larql_models::{detect_from_json, ModelWeights, WeightArray};
use ndarray::Array2;
use std::collections::HashMap;

/// Build a synthetic `ModelWeights` with all tensors populated.
/// Uses `TinyModelArch` key conventions (e.g. `"0.attn.q_proj.weight"`).
pub fn make_test_weights() -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const NUM_LAYERS: usize = 2;

    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut rng_state = 0xdeadbeef_u64;

    // LCG giving values in [-scale, +scale]
    let mut rand_mat = |rows: usize, cols: usize, scale: f32| -> WeightArray {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (rng_state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data)
            .unwrap()
            .into_shared()
    };

    // Embed + lm_head
    let embed = rand_mat(VOCAB, HIDDEN, 0.1);
    let lm_head = rand_mat(VOCAB, HIDDEN, 0.1);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Final norm (ones → valid unweighted RMSNorm fallback)
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    for layer in 0..NUM_LAYERS {
        // Attention projections
        tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, HIDDEN, 0.1));
        tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, HIDDEN, 0.1));
        tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, HIDDEN, 0.1));
        tensors.insert(arch.attn_o_key(layer), rand_mat(HIDDEN, q_dim, 0.1));
        // FFN — missing tensors cause panic, so always provide them
        tensors.insert(arch.ffn_gate_key(layer), rand_mat(INTER, HIDDEN, 0.1));
        tensors.insert(arch.ffn_up_key(layer), rand_mat(INTER, HIDDEN, 0.1));
        tensors.insert(arch.ffn_down_key(layer), rand_mat(HIDDEN, INTER, 0.1));
        // Layer norms
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; HIDDEN]);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

/// Build an in-memory `VectorIndex` with random gate vectors per layer.
/// The VectorIndex has no Q4K or interleaved data — `predict_honest` falls
/// through to the CPU path, and `WalkFfn` routes through the sparse fallback
/// that uses `weights.tensors`.
pub fn make_test_vindex(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    let n_features = weights.intermediate_size;
    let hidden = weights.hidden_size;

    // Each layer gets an independent LCG seed so gate matrices are distinct.
    let gate_vectors: Vec<Option<Array2<f32>>> = (0..weights.num_layers)
        .map(|l| {
            let mut state = 0xabcdef_u64.wrapping_add(l as u64 * 0x9e3779b97f4a7c15);
            let data: Vec<f32> = (0..n_features * hidden)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state as u32) as f32 / u32::MAX as f32 * 0.1 - 0.05
                })
                .collect();
            Some(Array2::from_shape_vec((n_features, hidden), data).unwrap())
        })
        .collect();

    let down_meta = vec![None; weights.num_layers];
    larql_vindex::VectorIndex::new(gate_vectors, down_meta, weights.num_layers, hidden)
}

/// Build a `tokenizers::Tokenizer` with a vocabulary of `vocab_size` tokens.
/// Token N decodes to `"[N]"`, so token IDs from `make_test_weights()` all
/// decode to valid (if meaningless) strings.
pub fn make_test_tokenizer(vocab_size: usize) -> tokenizers::Tokenizer {
    // WordLevel::builder().vocab() requires an AHashMap.
    // Build a simple BPE-less tokenizer via JSON serialization instead.
    let mut vocab_json = serde_json::Map::new();
    for i in 0..vocab_size as u64 {
        vocab_json.insert(format!("[{i}]"), serde_json::Value::Number(i.into()));
    }
    // Add UNK token at the end
    vocab_json.insert("[UNK]".into(), serde_json::Value::Number(vocab_size.into()));

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab_json,
            "unk_token": "[UNK]"
        }
    });

    let bytes = serde_json::to_vec(&tokenizer_json).expect("JSON serialization failed");
    tokenizers::Tokenizer::from_bytes(&bytes).expect("synthetic tokenizer construction failed")
}

/// All three synthetic fixtures bundled together. Build once per test module
/// via `OnceLock`; each field is cheaply borrowed.
pub struct TestFixtures {
    pub weights: ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub index: larql_vindex::VectorIndex,
}

impl TestFixtures {
    pub fn build() -> Self {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_vindex(&weights);
        Self {
            weights,
            tokenizer,
            index,
        }
    }
}

/// Serialise the synthetic `make_test_weights()` model + matching
/// vindex + tokenizer to an on-disk directory that any code path
/// reaching for `larql_vindex::load_vindex_config` /
/// `load_model_weights` will accept.
///
/// Replaces the previous "set `LARQL_MODEL` to a real Gemma snapshot"
/// pattern: tests can call this with a `tempfile::TempDir` and exercise
/// the full disk-loading pipeline without depending on multi-gigabyte
/// model artifacts in `~/.cache`.
///
/// The fixture is **synthetic**: the weights produce garbage logits.
/// Tests asserting plumbing (correct files written, correct error on
/// missing config, correct dispatch on backend type, etc.) work fine;
/// tests asserting semantic content ("model predicts Paris") still
/// need a real model and don't belong in `tests/`.
///
/// Layout written:
/// ```text
/// dir/
///   index.json              -- VindexConfig with has_model_weights=true
///   tokenizer.json          -- WordLevel "[0]".."[VOCAB-1]" tokenizer
///   embeddings.bin          -- VOCAB × HIDDEN f32 (from weights.embed)
///   weight_manifest.json    -- per-tensor offset/length manifest
///   attn_weights.bin        -- per-layer Q/K/V/O + norms
///   up_weights.bin          -- per-layer gate + up
///   down_weights.bin        -- per-layer down
///   norms.bin               -- final norm
///   lm_head.bin             -- output projection
///   gate_vectors.bin        -- vindex gate matrices (from make_test_vindex)
///   down_meta.bin           -- vindex down metadata (empty per layer)
/// ```
pub fn write_synthetic_model_dir(dir: &std::path::Path) -> Result<(), String> {
    use larql_vindex::{
        write_model_weights, ExtractLevel, MoeConfig, StorageDtype, VindexConfig, VindexModelConfig,
    };

    std::fs::create_dir_all(dir).map_err(|e| format!("create_dir_all: {e}"))?;

    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_vindex(&weights);

    // ── tokenizer.json ────────────────────────────────────────────────
    // Write a tokenizer that encodes `[N]` to id N *as a single token*
    // — `make_test_tokenizer`'s Whitespace pre-tokenizer would split
    // `[1]` into `[`, `1`, `]`, all of which UNK, blowing up the
    // embedding lookup with id=vocab_size. The on-disk fixture uses a
    // pre-tokenizer-free variant so test prompts like `EXPLAIN INFER
    // "[1]"` lookup directly. `tokenizer` is kept above for any caller
    // that needs the in-memory shape.
    let _ = &tokenizer; // returned by make_test_tokenizer; not the on-disk shape
    let tok_path = dir.join("tokenizer.json");
    std::fs::write(&tok_path, synthetic_tokenizer_json(weights.vocab_size))
        .map_err(|e| format!("write tokenizer.json: {e}"))?;

    // ── model_config + index.json ─────────────────────────────────────
    // `has_model_weights=true` is the gate the loader checks; without
    // it `load_model_weights` errors with "rebuild with extract --level
    // all". model_config carries the arch fields detect_from_json needs
    // to reconstruct the tinymodel arch on the loader side.
    let model_config = VindexModelConfig {
        model_type: "tinymodel".into(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: weights.rope_base,
        sliding_window: None,
        moe: None::<MoeConfig>,
        global_head_dim: None,
        num_global_kv_heads: None,
        partial_rotary_factor: None,
        sliding_window_pattern: None,
        layer_types: None,
        attention_k_eq_v: false,
        num_kv_shared_layers: None,
        per_layer_embed_dim: None,
        rope_local_base: None,
        query_pre_attn_scalar: None,
        final_logit_softcapping: None,
    };

    let mut config = VindexConfig {
        version: 2,
        model: "synthetic/tinymodel".into(),
        family: "tinymodel".into(),
        source: None,
        checksums: None,
        num_layers: weights.num_layers,
        hidden_size: weights.hidden_size,
        intermediate_size: weights.intermediate_size,
        vocab_size: weights.vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: Some(model_config),
        fp4: None,
        ffn_layout: None,
    };

    // Writes index.json + gate_vectors.bin + down_meta.bin.
    // `save_vindex` mutates `config` to record layer manifests.
    index
        .save_vindex(dir, &mut config)
        .map_err(|e| format!("save_vindex: {e}"))?;

    // ── Model weights (attn / up / down / norms / lm_head) ────────────
    let mut cb = larql_vindex::SilentBuildCallbacks;
    write_model_weights(&weights, dir, &mut cb).map_err(|e| format!("write_model_weights: {e}"))?;

    // ── Embeddings (vocab × hidden f32, little-endian) ────────────────
    let embed_slice = weights.embed.as_slice().ok_or("embed not contiguous")?;
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * 4);
    for &v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), &embed_bytes)
        .map_err(|e| format!("write embeddings.bin: {e}"))?;

    Ok(())
}

/// Build a tokenizer JSON whose vocab is `[0]`..`[vocab_size-1]` and
/// whose `pre_tokenizer` is **null** — so bracketed forms encode as a
/// single token instead of being split into `[`, `N`, `]` (all UNK)
/// by [`make_test_tokenizer`]'s Whitespace pre-tokenizer.
///
/// Used only by [`write_synthetic_model_dir`] so on-disk-fixture
/// callers can write test prompts like `"[1]"` and have them
/// encode to a single in-vocab id. `make_test_tokenizer` is kept
/// in its prior shape for backward-compatibility with in-memory
/// fixture consumers.
///
/// `[UNK]` is mapped to **id 0** (a real, in-range vocab slot) so any
/// stray UNK from text the loader processes through the model still
/// hits a valid embedding row — saves the embed lookup from panicking
/// with "Index N must be less than axis length N" when something
/// outside the bracket form sneaks into encoding.
fn synthetic_tokenizer_json(vocab_size: usize) -> String {
    let mut vocab_json = serde_json::Map::new();
    for i in 0..vocab_size as u64 {
        vocab_json.insert(format!("[{i}]"), serde_json::Value::Number(i.into()));
    }
    vocab_json.insert("[UNK]".into(), serde_json::Value::Number(0.into()));

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab_json,
            "unk_token": "[UNK]"
        }
    });
    serde_json::to_string(&tokenizer_json).expect("synthetic tokenizer json")
}

// ── Alternate-arch fixtures ─────────────────────────────────────────────
//
// `make_test_weights` uses the `tinymodel` arch which leaves many optional
// branches dormant (no bias keys, no QK norm, no post norms, gated FFN
// only). The fixtures below pin those branches by routing through a
// real arch impl that enables them. Each fixture provides exactly the
// tensors + vectors the matching forward path needs to reach finite
// output without panicking.

fn rand_mat_seeded(rows: usize, cols: usize, scale: f32, seed: u64) -> WeightArray {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data)
        .unwrap()
        .into_shared()
}

/// Build a synthetic `ModelWeights` configured as a Gemma 3-style arch.
///
/// Enables the dormant branches in `attention/{block, gpu}.rs` and
/// `forward/layer.rs` that tinymodel never reaches:
/// - **QK norm** — `attn_q_norm_key` / `attn_k_norm_key` return Some
/// - **post norms** — `has_post_norms()` is true; pre/post FFN norm keys
///   are populated, the FFN dispatch routes through the post-norm arm
/// - **GeluTanh activation** — `activation()` is `GeluTanh`, exercising
///   the gelu-tanh gate-up branches in `ffn/weight.rs` and `attention`
/// - **`embed_scale = sqrt(hidden)`** — non-1.0 embed scaling
/// - **`norm_weight_offset = 1.0`** — non-zero offset added to every
///   norm weight at runtime
pub fn make_gemma3_test_weights() -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const NUM_LAYERS: usize = 2;

    let arch_json = serde_json::json!({
        "model_type": "gemma3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        // Non-default scaling: exercises the `res_mult != 1.0` branch in
        // `forward/layer.rs::run_ffn` and `attention/gpu.rs::run_attention_block_gpu`.
        "residual_multiplier": 0.5,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    // Embed + lm_head — small, non-zero so post-norm RMS doesn't divide by 0.
    let embed = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0x9e3779b9);
    let lm_head = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0xa1b2c3d4);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Final norm — Gemma3 uses norm_weight_offset=1.0, so the saved
    // weight is the *delta* off identity. Zeros → unit-scale norm at
    // runtime (offset=1 + weight=0 → 1.0).
    vectors.insert(arch.final_norm_key().to_string(), vec![0.0; HIDDEN]);

    let mut seed_counter: u64 = 0xdeadbeef;
    let mut next_seed = || {
        seed_counter = seed_counter.wrapping_add(0x9e3779b97f4a7c15);
        seed_counter
    };

    for layer in 0..NUM_LAYERS {
        // Attention projections
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(HIDDEN, q_dim, 0.1, next_seed()),
        );

        // FFN
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(HIDDEN, INTER, 0.1, next_seed()),
        );

        // Layer norms — input + post-attention. norm_weight_offset=1.0
        // means saved weights are deltas; zeros = identity.
        vectors.insert(arch.input_layernorm_key(layer), vec![0.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![0.0; HIDDEN]);
        // Gemma3-specific: pre/post FFN norms (post-norms branch).
        if let Some(k) = arch.pre_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.0; HIDDEN]);
        }
        if let Some(k) = arch.post_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.0; HIDDEN]);
        }

        // QK norm — per-head dim weights.
        if let Some(k) = arch.attn_q_norm_key(layer) {
            vectors.insert(k, vec![0.0; HEAD_DIM]);
        }
        if let Some(k) = arch.attn_k_norm_key(layer) {
            vectors.insert(k, vec![0.0; HEAD_DIM]);
        }
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

/// Build a synthetic `ModelWeights` configured as a Starcoder2-style arch.
///
/// Enables the dormant branches:
/// - **Non-gated FFN** — `ffn_type()` is `NonGated`, exercising the
///   `else` arm in `ffn/weight.rs::dense_ffn_forward_backend`
/// - **FFN bias** — `ffn_up_bias_key` / `ffn_down_bias_key` return Some,
///   so the `add_bias` calls fire
/// - **Attention bias** — `attn_q_bias_key` / `attn_k_bias_key` /
///   `attn_v_bias_key` / `attn_o_bias_key` return Some
/// - **Gelu activation** — `activation()` is `Gelu`
pub fn make_starcoder2_test_weights() -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const NUM_LAYERS: usize = 2;

    let arch_json = serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
        // Non-default scaling: exercises the `res_mult != 1.0` branch in
        // the no-post-norms arm of `forward/layer.rs::run_ffn` and the
        // `attention_multiplier()` branch in `attention/gpu.rs`.
        "residual_multiplier": 0.5,
        "attention_multiplier": 2.0,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    let embed = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0x12345678);
    let lm_head = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0x87654321);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

    let mut seed_counter: u64 = 0xfeedbabe;
    let mut next_seed = || {
        seed_counter = seed_counter.wrapping_add(0x9e3779b97f4a7c15);
        seed_counter
    };

    for layer in 0..NUM_LAYERS {
        // Attention projections
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(HIDDEN, q_dim, 0.1, next_seed()),
        );

        // Attention biases — Starcoder2 has them.
        if let Some(k) = arch.attn_q_bias_key(layer) {
            vectors.insert(k, vec![0.01; q_dim]);
        }
        if let Some(k) = arch.attn_k_bias_key(layer) {
            vectors.insert(k, vec![0.01; kv_dim]);
        }
        if let Some(k) = arch.attn_v_bias_key(layer) {
            vectors.insert(k, vec![0.01; kv_dim]);
        }
        if let Some(k) = arch.attn_o_bias_key(layer) {
            vectors.insert(k, vec![0.01; HIDDEN]);
        }

        // FFN — non-gated, so up + down only. No gate matrix.
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(HIDDEN, INTER, 0.1, next_seed()),
        );
        // Add gate too — code may probe regardless of ffn_type for some paths.
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );

        // FFN biases — Starcoder2 has them.
        if let Some(k) = arch.ffn_up_bias_key(layer) {
            vectors.insert(k, vec![0.01; INTER]);
        }
        if let Some(k) = arch.ffn_down_bias_key(layer) {
            vectors.insert(k, vec![0.01; HIDDEN]);
        }

        // Layer norms — Starcoder2 uses standard LayerNorm/RMSNorm,
        // norm_weight_offset=0, so weights are the actual scale.
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; HIDDEN]);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

// ── Q4_K-aware synthetic fixture ─────────────────────────────────────────
//
// `make_test_weights` uses hidden=16, below Q4_K's 256-element
// super-block minimum. The cached / direct-matvec decode paths in
// `vindex/kquant_forward/cached.rs` require a vindex with real
// `attn_kquant_layer_data` + `interleaved_kquant_layer_data` manifests,
// so unit tests for those paths can't fit the tiny fixture. The
// helpers below build a hidden=256, intermediate=256 Gemma 3-style
// fixture with synthetic Q4_K bytes that round-trip through
// `larql_compute::cpu::ops::q4_common::quantize_q4_k`.

/// Hidden dimension for the Q4_K test fixture — minimum Q4_K-safe
/// multiple of 256.
pub const Q4K_TEST_HIDDEN: usize = 256;
/// Intermediate dimension for the Q4_K test fixture.
pub const Q4K_TEST_INTER: usize = 256;
/// Vocabulary size for the Q4_K test fixture.
pub const Q4K_TEST_VOCAB: usize = 256;
/// Layer count for the Q4_K test fixture.
pub const Q4K_TEST_NUM_LAYERS: usize = 2;

/// Build a synthetic `ModelWeights` sized to satisfy Q4_K's 256-element
/// super-block constraint. Uses Gemma 3 architecture so the
/// `has_post_norms` + `GeluTanh` branches in the cached decode path
/// are exercised.
pub fn make_test_q4k_weights() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = Q4K_TEST_HIDDEN / num_q;

    let arch_json = serde_json::json!({
        "model_type": "gemma3_text",
        "hidden_size": Q4K_TEST_HIDDEN,
        "num_hidden_layers": Q4K_TEST_NUM_LAYERS,
        "intermediate_size": Q4K_TEST_INTER,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_theta": 10000.0,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let mut seed = 0xc0ffee_u64;
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let embed = rand_mat_seeded(Q4K_TEST_VOCAB, Q4K_TEST_HIDDEN, 0.05, next_seed());
    let lm_head = embed.clone();
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    vectors.insert(
        arch.final_norm_key().to_string(),
        vec![1.0; Q4K_TEST_HIDDEN],
    );

    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    for layer in 0..Q4K_TEST_NUM_LAYERS {
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(Q4K_TEST_HIDDEN, q_dim, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(Q4K_TEST_INTER, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(Q4K_TEST_INTER, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(Q4K_TEST_HIDDEN, Q4K_TEST_INTER, 0.05, next_seed()),
        );

        vectors.insert(arch.input_layernorm_key(layer), vec![0.5; Q4K_TEST_HIDDEN]);
        vectors.insert(
            arch.post_attention_layernorm_key(layer),
            vec![0.5; Q4K_TEST_HIDDEN],
        );
        if let Some(k) = arch.pre_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; Q4K_TEST_HIDDEN]);
        }
        if let Some(k) = arch.post_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; Q4K_TEST_HIDDEN]);
        }
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers: Q4K_TEST_NUM_LAYERS,
        hidden_size: Q4K_TEST_HIDDEN,
        intermediate_size: Q4K_TEST_INTER,
        vocab_size: Q4K_TEST_VOCAB,
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        rope_base: 10_000.0,
    }
}

/// Wrap a byte payload in an anonymous read-only mmap. Used to build
/// in-memory test vindexes without touching the filesystem.
fn arc_mmap_from_bytes(payload: &[u8]) -> std::sync::Arc<memmap2::Mmap> {
    let mut anon = memmap2::MmapMut::map_anon(payload.len().max(1)).expect("anon mmap");
    if !payload.is_empty() {
        anon.copy_from_slice(payload);
    }
    let mmap = anon.make_read_only().expect("freeze");
    std::sync::Arc::new(mmap)
}

/// Build a fully-populated synthetic `VectorIndex` that satisfies the
/// cached + direct-matvec decode contract on the Q4_K weights from
/// [`make_test_q4k_weights`]. Quantises Q/K/V/O and gate/up/down to
/// Q4_K bytes via `quantize_q4_k`, installs them as the attn +
/// interleaved Q4_K storage, and synthesises a Q4_K lm_head view from
/// the (tied) embeddings.
pub fn make_test_q4k_vindex(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;

    let num_layers = weights.num_layers;
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;

    let q4k_for = |key: &str| -> Vec<u8> {
        let tensor = weights
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
        let slice = tensor.as_slice().expect("contiguous row-major");
        quantize_q4_k(slice)
    };

    let mut attn_payload: Vec<u8> = Vec::new();
    let mut attn_manifest: Vec<(usize, usize, String)> = Vec::new();
    for layer in 0..num_layers {
        for key in [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ] {
            let bytes = q4k_for(&key);
            let offset = attn_payload.len();
            let length = bytes.len();
            attn_payload.extend_from_slice(&bytes);
            attn_manifest.push((offset, length, "Q4_K".to_string()));
        }
    }

    let mut ffn_payload: Vec<u8> = Vec::new();
    let mut ffn_manifest: Vec<(usize, usize, String)> = Vec::new();
    for layer in 0..num_layers {
        for key in [
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ] {
            let bytes = q4k_for(&key);
            let offset = ffn_payload.len();
            let length = bytes.len();
            ffn_payload.extend_from_slice(&bytes);
            ffn_manifest.push((offset, length, "Q4_K".to_string()));
        }
    }

    let gate_vectors = vec![None; num_layers];
    let down_meta = vec![None; num_layers];
    let mut index = larql_vindex::VectorIndex::new(gate_vectors, down_meta, num_layers, hidden);
    index.vocab_size = weights.vocab_size;

    let attn_mmap = arc_mmap_from_bytes(&attn_payload);
    let ffn_mmap = arc_mmap_from_bytes(&ffn_payload);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_attn_q4k(attn_mmap, Some(attn_manifest));
        storage.set_interleaved_q4k(ffn_mmap, Some(ffn_manifest));
    }

    // Synth Q4_K lm_head from tied embedding (same lifecycle as
    // `synthesize_lm_head_q4` on a real tied-embedding vindex).
    let lm_head_slice = weights
        .lm_head
        .as_slice()
        .expect("lm_head contiguous row-major");
    let lm_head_q4 = quantize_q4_k(lm_head_slice);
    let lm_head_mmap = arc_mmap_from_bytes(&lm_head_q4);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_lm_head_q4_mmap(lm_head_mmap);
    }
    index
}

/// Bundled fixture for Q4_K decode-path tests. Mirrors `TestFixtures`.
pub struct Q4KTestFixtures {
    pub weights: ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub index: larql_vindex::VectorIndex,
}

impl Q4KTestFixtures {
    pub fn build() -> Self {
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        Self {
            weights,
            tokenizer,
            index,
        }
    }
}

#[cfg(test)]
mod synthetic_model_dir_tests {
    use super::*;
    use larql_vindex::{load_vindex_config, SilentLoadCallbacks};

    #[test]
    fn write_then_load_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_synthetic_model_dir(dir.path()).expect("write fixture");

        // 1. Config round-trips with the flags the EXPLAIN INFER pipeline gates on.
        let config = load_vindex_config(dir.path()).expect("load_vindex_config");
        assert!(
            config.has_model_weights,
            "fixture must set has_model_weights=true"
        );
        assert_eq!(config.quant, larql_vindex::QuantFormat::None);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.hidden_size, 16);
        let mc = config.model_config.as_ref().expect("model_config");
        assert_eq!(mc.model_type, "tinymodel");
        assert_eq!(mc.head_dim, 8);

        // 2. Weights load via the same path InferenceWeights::load uses.
        let mut cb = SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(dir.path(), &mut cb)
            .expect("load_model_weights against synthetic fixture");
        assert_eq!(weights.num_layers, 2);
        assert_eq!(weights.hidden_size, 16);
        assert_eq!(weights.vocab_size, 32);
        // Round-tripped tensors must be retrievable by the arch-keyed
        // names the forward pass walks — pick a representative entry.
        assert!(
            weights.tensors.contains_key(&weights.arch.attn_q_key(0)),
            "expected attn_q tensor for layer 0 after round-trip"
        );
        assert!(weights.tensors.contains_key(&weights.arch.ffn_gate_key(0)));
    }

    #[test]
    fn tokenizer_file_is_present_and_loadable() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_synthetic_model_dir(dir.path()).expect("write fixture");
        let tok_path = dir.path().join("tokenizer.json");
        assert!(tok_path.exists(), "tokenizer.json must be written");
        let _ = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer round-trips");
    }

    #[test]
    fn embeddings_bin_has_expected_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_synthetic_model_dir(dir.path()).expect("write fixture");
        let bytes = std::fs::read(dir.path().join("embeddings.bin")).expect("embeddings.bin");
        // 32 vocab × 16 hidden × 4 bytes = 2048
        assert_eq!(bytes.len(), 32 * 16 * 4);
    }
}
