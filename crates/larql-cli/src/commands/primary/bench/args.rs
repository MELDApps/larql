//! Clap-derived CLI args for `larql bench`. Kept in its own file so
//! flag-surface changes don't churn the dispatch logic in `run.rs`.

use clap::Args;

#[derive(Args, Clone)]
pub struct BenchArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand.
    pub model: String,

    /// Prompt to time. Kept short by default to keep prefill consistent
    /// across runs.
    #[arg(long, default_value = "The capital of France is")]
    pub prompt: String,

    /// Number of decode steps to measure.
    #[arg(short = 'n', long = "tokens", default_value = "50")]
    pub tokens: usize,

    /// Discarded warmup steps before measurement (smooths first-call
    /// allocation / JIT effects in the Metal library).
    #[arg(long, default_value = "3")]
    pub warmup: usize,

    /// Comma-separated backend list. Supported: `metal`, `cpu`.
    #[arg(long, default_value = "metal")]
    pub backends: String,

    /// Also query a local Ollama server on the default port with this
    /// model name (e.g. `gemma3:4b`). Requires `ollama serve` running.
    #[arg(long, value_name = "MODEL")]
    pub ollama: Option<String>,

    /// Comma-separated KV engines to bench alongside the GPU path.
    /// Supported: `markov-rs`, `unlimited-context`.
    /// Example: `--engine markov-rs,unlimited-context`.
    #[arg(long, value_name = "ENGINE,...")]
    pub engine: Option<String>,

    /// Route FFN to a remote larql-server for the bench run.
    /// Attention runs locally on Metal; each layer's FFN is a round trip to
    /// the URL. Use this to bench the grid path for large models like 31B.
    /// Example: `--ffn http://127.0.0.1:8080`
    #[arg(long, value_name = "URL")]
    pub ffn: Option<String>,

    /// HTTP timeout in seconds for --ffn.
    #[arg(long, default_value = "60")]
    pub ffn_timeout_secs: u64,

    /// Dispatch strategy for --ffn.
    ///   streaming  (default) — one HTTP round-trip per layer per token.
    ///   batch      — all layers in parallel (Q8K NEON) per token.
    #[arg(long, default_value = "streaming", value_name = "streaming|batch")]
    pub ffn_dispatch: String,

    /// Bench the remote MoE expert path (Gemma 4 26B A4B etc.).
    /// Shard map: `"START-END=URL,START-END=URL,..."`.
    /// Example: `--moe-shards "0-63=http://a:8081,64-127=http://b:8082"`
    #[arg(long, value_name = "SHARDS")]
    pub moe_shards: Option<String>,

    /// Dispatch strategy for --moe-shards.
    ///   streaming  (default) — one round-trip per layer per token.
    ///   batch      — all layers in one round-trip per token (approximate).
    #[arg(long, default_value = "streaming", value_name = "streaming|batch")]
    pub moe_dispatch: String,

    /// Refinement iterations for `--moe-dispatch batch`.
    /// 1 = one dispatch + two Metal passes (fast, approximate).
    /// 2 = two dispatches + three passes (correct answer, ~half the speed).
    #[arg(long, default_value = "2")]
    pub moe_predispatch_iters: usize,

    /// Print per-stage timing breakdown for each engine (markov-rs only for now).
    #[arg(long)]
    pub profile: bool,

    /// Comma-separated wire formats to compare end-to-end. Requires --ffn.
    /// Supported: f32, f16, i8.
    /// Example: --wire f32,f16,i8
    #[arg(long, value_name = "f32,f16,i8")]
    pub wire: Option<String>,

    /// Run a shard-count scaling sweep.
    /// With --moe-shards: reruns with 1..N shards from the provided map.
    /// With --ffn: runs the same URL 1..3 times (simulated replicas).
    #[arg(long)]
    pub bench_grid: bool,

    /// Simulate N concurrent clients. Each runs the full bench independently;
    /// reports aggregate tok/s and per-client p99.
    #[arg(long, default_value = "1", value_name = "N")]
    pub concurrent: usize,

    /// Emit machine-readable JSON alongside the table output.
    /// Supported: json.
    #[arg(long, value_name = "json")]
    pub output: Option<String>,

    /// Write JSON output to this file instead of stdout.
    #[arg(long, value_name = "PATH")]
    pub output_file: Option<String>,

    /// Verbose load / warmup logging.
    #[arg(short, long)]
    pub verbose: bool,
}
