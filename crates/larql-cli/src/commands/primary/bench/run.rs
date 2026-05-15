//! Bench orchestration: pick backends, drive each, render the table, emit
//! JSON. Heavy lifting lives in the per-backend modules; this file is just
//! the dispatch + JSON envelope.

use larql_kv::EngineKind;

use crate::commands::primary::cache;

use super::args::BenchArgs;
use super::engine::{run_engine, run_engine_q4k};
use super::helpers;
use super::local::run_larql;
use super::ollama::run_ollama;
use super::output::print_table;
use super::remote_ffn::run_concurrent_ffn;
use super::remote_moe::run_concurrent_moe;
use super::row::{BenchJsonLatency, BenchJsonResult, BenchJsonRow, BenchRow};

pub fn run(args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex_path = cache::resolve_model(&args.model)?;
    if !vindex_path.is_dir() {
        return Err(format!(
            "resolved model path is not a directory: {}",
            vindex_path.display(),
        )
        .into());
    }

    let requested_backends: Vec<&str> = args
        .backends
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let want_metal = requested_backends.contains(&"metal");
    let want_cpu = requested_backends.contains(&"cpu");
    let want_engine = args.engine.is_some();
    let want_ffn = args.ffn.is_some();
    let want_moe = args.moe_shards.is_some();
    if !want_metal && !want_cpu && args.ollama.is_none() && !want_engine && !want_ffn && !want_moe {
        return Err(
            "no backends selected: pass --backends metal,cpu, --ollama, --engine, --ffn, or --moe-shards".into(),
        );
    }

    println!("larql bench: {}", vindex_path.display());
    println!("Prompt: {:?}", args.prompt);
    println!(
        "Decode: {} tokens after {} warmup; backends={}{}",
        args.tokens,
        args.warmup,
        args.backends,
        args.ollama
            .as_deref()
            .map(|m| format!(", ollama={m}"))
            .unwrap_or_default(),
    );
    println!();

    let mut rows: Vec<BenchRow> = Vec::new();

    // GPU/CPU bench requires Q4K vindex. Skip silently when running engine-only
    // (engines need f32 weights from a non-Q4K vindex).
    let cfg = larql_vindex::load_vindex_config(&vindex_path)?;
    let is_q4k = cfg.quant == larql_vindex::QuantFormat::Q4K;

    if want_metal {
        if is_q4k {
            rows.push(run_larql(&vindex_path, &args, /* metal */ true)?);
        } else if !want_engine {
            return Err(format!(
                "GPU bench requires a Q4K vindex (got quant={:?}). \
                 Use a q4k vindex for GPU bench, or omit --backends and use --engine only.",
                cfg.quant,
            )
            .into());
        }
    }
    if want_cpu {
        if is_q4k {
            rows.push(run_larql(&vindex_path, &args, /* metal */ false)?);
        } else if !want_engine {
            return Err(format!(
                "CPU bench requires a Q4K vindex (got quant={:?}).",
                cfg.quant,
            )
            .into());
        }
    }
    if let Some(ref ollama_model) = args.ollama {
        rows.push(run_ollama(ollama_model, &args.prompt, args.tokens));
    }

    // KV engine rows.
    //
    // Q4K vindex → prefill_q4k / decode_step_q4k (Metal pipeline, fast path).
    // f16/f32 vindex → prefill / decode_step (f32 CPU path, slow but correct).
    if let Some(ref engine_list) = args.engine {
        let mut cb = larql_vindex::SilentLoadCallbacks;

        if is_q4k {
            let mut weights = larql_vindex::load_model_weights_q4k(&vindex_path, &mut cb)?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
            let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)?;
            index.load_attn_q4k(&vindex_path)?;
            index.load_interleaved_q4k(&vindex_path)?;
            let token_ids =
                larql_inference::encode_prompt(&tokenizer, &*weights.arch, args.prompt.as_str())
                    .map_err(|e| format!("tokenize: {e}"))?;
            let kv_ref_bytes =
                larql_kv::markov_residual::kv_memory_bytes_for_seq(&weights, token_ids.len());

            for engine_name in engine_list
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                match EngineKind::from_name(engine_name) {
                    Some(kind) => {
                        let backend = if want_metal {
                            larql_inference::default_backend()
                        } else {
                            larql_inference::cpu_backend()
                        };
                        rows.push(run_engine_q4k(
                            &mut weights,
                            &index,
                            &token_ids,
                            kv_ref_bytes,
                            kind,
                            backend,
                            &args,
                        )?);
                    }
                    None => eprintln!(
                        "unknown engine {:?} — supported: markov-rs, unlimited-context",
                        engine_name
                    ),
                }
            }
        } else {
            let weights = larql_vindex::load_model_weights(&vindex_path, &mut cb)?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
            let token_ids =
                larql_inference::encode_prompt(&tokenizer, &*weights.arch, args.prompt.as_str())
                    .map_err(|e| format!("tokenize: {e}"))?;
            let kv_ref_bytes =
                larql_kv::markov_residual::kv_memory_bytes_for_seq(&weights, token_ids.len());

            for engine_name in engine_list
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                match EngineKind::from_name(engine_name) {
                    Some(kind) => {
                        let backend = if want_metal {
                            larql_inference::default_backend()
                        } else {
                            larql_inference::cpu_backend()
                        };
                        rows.push(run_engine(
                            &weights,
                            &token_ids,
                            kv_ref_bytes,
                            kind,
                            backend,
                            &args,
                        )?);
                    }
                    None => eprintln!(
                        "unknown engine {:?} — supported: markov-rs, unlimited-context",
                        engine_name
                    ),
                }
            }
        }
    }

    if let Some(ref ffn_url) = args.ffn {
        let wire_prefs = match args.wire.as_deref() {
            Some(spec) => {
                let parsed = helpers::parse_wire_list(spec);
                if parsed.is_empty() {
                    vec![larql_inference::WirePreference::BestAvailable]
                } else {
                    parsed
                }
            }
            None => vec![larql_inference::WirePreference::BestAvailable],
        };
        for pref in wire_prefs {
            rows.push(run_concurrent_ffn(&vindex_path, &args, ffn_url, pref)?);
        }
    }

    if let Some(ref shards_str) = args.moe_shards {
        if args.bench_grid {
            // Grid scaling sweep: run with 1..N shards from the shard map.
            let shard_entries: Vec<&str> = shards_str
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            // Track single-shard tok/s so we can compute shard_efficiency
            // for the larger configurations (ADR-0012 §--bench-grid Mode).
            let mut single_shard_tok_per_s: Option<f64> = None;
            for n_shards in 1..=shard_entries.len() {
                let partial = shard_entries[..n_shards].join(",");
                let mut row = run_concurrent_moe(&vindex_path, &args, &partial)?;
                if n_shards == 1 {
                    single_shard_tok_per_s = Some(row.tok_per_s);
                }
                row.shard_efficiency = single_shard_tok_per_s.and_then(|base| {
                    helpers::shard_efficiency(row.tok_per_s, n_shards, base)
                });
                row.note = format!(
                    "{} shard{} | {}",
                    n_shards,
                    if n_shards == 1 { "" } else { "s" },
                    row.note
                );
                rows.push(row);
            }
        } else {
            rows.push(run_concurrent_moe(&vindex_path, &args, shards_str)?);
        }
    }

    print_table(&rows);

    // JSON output (ADR-0012).
    let want_json = args
        .output
        .as_deref()
        .map(|o| o.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
        || args.output_file.is_some();
    if want_json {
        let json_rows: Vec<BenchJsonRow> = rows
            .iter()
            .map(|r| BenchJsonRow {
                backend: r.backend.clone(),
                prefill_ms: r.prefill_ms,
                ms_per_tok: BenchJsonLatency {
                    mean: r.avg_decode_ms,
                    p50: r.p50_ms,
                    p99: r.p99_ms,
                },
                tok_per_s: r.tok_per_s,
                wire_bytes_per_tok: r.wire_bytes_per_tok,
                shard_efficiency: r.shard_efficiency,
                n_steps: r.n_steps,
                note: r.note.clone(),
            })
            .collect();
        let result = BenchJsonResult {
            timestamp: {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                format!("{secs}")
            },
            model: vindex_path.display().to_string(),
            prompt: args.prompt.clone(),
            tokens: args.tokens,
            wire: args.wire.clone(),
            concurrent: args.concurrent,
            results: json_rows,
        };
        let json_str = serde_json::to_string_pretty(&result)?;
        if let Some(ref path) = args.output_file {
            std::fs::write(path, &json_str)?;
            eprintln!("[bench] JSON written to {path}");
        } else {
            println!("{json_str}");
        }
    }
    Ok(())
}
