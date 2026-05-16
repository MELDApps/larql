//! HTTP server for the router: `AppState`, the `/v1/walk-ffn` handler, the
//! `/v1/stats` proxy, the `/v1/health` heartbeat, and the axum `Router`
//! factory. Moved out of `main.rs` so integration tests can build a Router
//! against a mock shard backend without spawning the binary.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::dispatch::{
    build_subrequest_body, group_layers_by_url, merge_shard_responses, resolve_static_only,
    unique_candidate_urls,
};
use crate::grid::GridState;
use crate::shards::{find_shard_for_layer, peek_binary, Shard};

/// Content-Type used by the FFN binary protocol. JSON requests use the
/// standard `application/json`.
pub const BINARY_CT: &str = "application/x-larql-ffn";

/// Shared HTTP service state. Holds the static shard map, an optional
/// grid handle, and a single reqwest client (whose connection pool is
/// reused across all outbound shard calls).
pub struct AppState {
    pub static_shards: Vec<Shard>,
    pub grid: Option<Arc<RwLock<GridState>>>,
    pub client: reqwest::Client,
}

impl AppState {
    /// Resolve every layer to its owning shard URL. Grid lookups take
    /// priority; any layer not covered by the grid falls back to the
    /// static shard map. Returns `Err(first uncovered layer)`.
    pub async fn resolve_all(
        &self,
        model_id: Option<&str>,
        layers: &[usize],
    ) -> Result<HashMap<usize, String>, usize> {
        if let Some(grid) = &self.grid {
            let guard = grid.read().await;
            let mut out = HashMap::with_capacity(layers.len());
            let mut static_needed: Vec<usize> = Vec::new();
            for &layer in layers {
                match guard.route(model_id, layer as u32) {
                    Some(url) => {
                        out.insert(layer, url);
                    }
                    None => static_needed.push(layer),
                }
            }
            drop(guard);
            for layer in static_needed {
                match find_shard_for_layer(&self.static_shards, layer) {
                    Some(s) => {
                        out.insert(layer, s.url.clone());
                    }
                    None => return Err(layer),
                }
            }
            return Ok(out);
        }
        resolve_static_only(&self.static_shards, layers)
    }
}

/// Build the axum `Router` for the public HTTP surface. Held separate
/// from the binary's `main()` so integration tests can mount it onto an
/// in-process listener.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/walk-ffn", post(handle_walk_ffn))
        .route("/v1/health", get(handle_health))
        .route("/v1/stats", get(handle_stats))
        .with_state(state)
}

// ── Handlers ────────────────────────────────────────────────────────────────

/// Returns `true` when `ct` is the FFN binary protocol marker. Pure;
/// extracted so the binary-vs-JSON branch can be unit-tested without
/// building a full HTTP request.
pub fn is_binary_content_type(ct: &str) -> bool {
    ct.starts_with(BINARY_CT)
}

/// Pull layer IDs and optional `model_id` out of a request body. For
/// binary bodies the header is peeked; for JSON bodies we look for a
/// `layers` array or a `layer` scalar plus an optional `model_id` field.
///
/// `Err(msg)` is returned to the caller as a 400 reply — the message is
/// already user-facing.
pub fn extract_layers_and_model_id(
    body: &[u8],
    is_binary: bool,
) -> Result<(Vec<usize>, Option<String>), String> {
    if is_binary {
        let layers =
            peek_binary(body).ok_or_else(|| "binary: truncated or malformed header".to_string())?;
        Ok((layers, None))
    } else {
        let peek: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;
        let layers: Vec<usize> = if let Some(arr) = peek.get("layers").and_then(|v| v.as_array()) {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        } else if let Some(n) = peek.get("layer").and_then(|v| v.as_u64()) {
            vec![n as usize]
        } else {
            return Err("must provide 'layer' or 'layers'".into());
        };
        let model_id = peek
            .get("model_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        Ok((layers, model_id))
    }
}

/// `POST /v1/walk-ffn` entry point. Errors are normalised to JSON
/// regardless of the request content-type so clients always see the same
/// envelope.
pub async fn handle_walk_ffn(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    match handle_walk_ffn_inner(state, request).await {
        Ok(r) => r,
        Err((status, msg)) => {
            let body = format!(r#"{{"error":{}}}"#, serde_json::Value::String(msg));
            Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body))
                .unwrap()
        }
    }
}

async fn handle_walk_ffn_inner(
    state: Arc<AppState>,
    request: axum::extract::Request,
) -> Result<Response, (StatusCode, String)> {
    let is_binary = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(is_binary_content_type)
        .unwrap_or(false);

    let body_bytes = axum::body::to_bytes(request.into_body(), 64 * 1024 * 1024)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("read body: {e}")))?;

    let (layers, model_id_owned) = extract_layers_and_model_id(&body_bytes, is_binary)
        .map_err(|m| (StatusCode::BAD_REQUEST, m))?;

    if layers.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty layer list".to_string()));
    }

    let mid = model_id_owned.as_deref();
    let layer_urls = state.resolve_all(mid, &layers).await.map_err(|missing| {
        (
            StatusCode::BAD_REQUEST,
            format!("layer {missing} has no owning shard in this router"),
        )
    })?;

    let unique_urls: std::collections::HashSet<&String> = layer_urls.values().collect();

    if unique_urls.len() == 1 || layers.len() == 1 {
        // All layers on the same shard — proxy raw bytes unchanged.
        let url = layer_urls.values().next().unwrap();
        let ct = if is_binary {
            BINARY_CT
        } else {
            "application/json"
        };
        return proxy_raw(&state.client, url, body_bytes, ct).await;
    }

    // Multi-shard dispatch.
    if is_binary {
        return Err((
            StatusCode::BAD_REQUEST,
            "binary fan-out across multiple shards is not supported; use JSON or split by shard"
                .to_string(),
        ));
    }

    let body_value: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;

    let by_url = group_layers_by_url(&layer_urls);

    let mut handles = Vec::new();
    for (url, shard_layers) in &by_url {
        let sub_body = build_subrequest_body(&body_value, shard_layers);
        let client = state.client.clone();
        let target = format!("{url}/v1/walk-ffn");
        handles.push(tokio::spawn(async move {
            client
                .post(&target)
                .json(&sub_body)
                .send()
                .await
                .map_err(|e| e.to_string())?
                .json::<Value>()
                .await
                .map_err(|e| e.to_string())
        }));
    }

    let responses: Vec<Value> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|jh| jh.map_err(|e| e.to_string()).and_then(|r| r))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard error: {e}")))?;

    let merged = merge_shard_responses(&responses);
    let json_bytes = serde_json::to_vec(&merged)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(json_bytes))
        .unwrap())
}

/// Forward raw bytes to a shard, passing the Content-Type header through.
async fn proxy_raw(
    client: &reqwest::Client,
    base_url: &str,
    body: Bytes,
    ct: &str,
) -> Result<Response, (StatusCode, String)> {
    let url = format!("{base_url}/v1/walk-ffn");
    let resp = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, ct)
        .body(body.to_vec())
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard {base_url}: {e}")))?;

    let status = resp.status();
    let resp_ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let resp_bytes = resp
        .bytes()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("read shard response: {e}")))?;

    Ok(Response::builder()
        .status(status.as_u16())
        .header(header::CONTENT_TYPE, resp_ct)
        .body(axum::body::Body::from(resp_bytes))
        .unwrap())
}

pub async fn handle_health() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// Proxy `/v1/stats` to the first reachable shard so that clients
/// connecting via `RemoteWalkBackend` (which reads `hidden_size` from
/// `/v1/stats`) work transparently through the router.
pub async fn handle_stats(State(state): State<Arc<AppState>>) -> Response {
    let grid_urls = if let Some(grid) = &state.grid {
        grid.read().await.all_shard_urls()
    } else {
        Vec::new()
    };
    let candidates = unique_candidate_urls(grid_urls, &state.static_shards);
    for url in candidates {
        let stats_url = format!("{url}/v1/stats");
        if let Ok(resp) = state.client.get(&stats_url).send().await {
            if resp.status().is_success() {
                if let Ok(bytes) = resp.bytes().await {
                    return Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from(bytes))
                        .unwrap();
                }
            }
        }
    }
    Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(r#"{"error":"no shard reachable"}"#))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_binary_content_type_recognises_marker_prefix() {
        assert!(is_binary_content_type(BINARY_CT));
        assert!(is_binary_content_type(
            "application/x-larql-ffn; charset=utf-8"
        ));
        assert!(!is_binary_content_type("application/json"));
        assert!(!is_binary_content_type(""));
    }

    #[test]
    fn extract_layers_from_binary_body() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&42u32.to_le_bytes());
        let (layers, model) = extract_layers_and_model_id(&buf, true).unwrap();
        assert_eq!(layers, vec![42]);
        assert!(model.is_none());
    }

    #[test]
    fn extract_layers_binary_truncated_returns_err() {
        let err = extract_layers_and_model_id(&[], true).unwrap_err();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn extract_layers_from_json_array() {
        let body = br#"{"layers":[0,1,2],"model_id":"gemma"}"#;
        let (layers, model) = extract_layers_and_model_id(body, false).unwrap();
        assert_eq!(layers, vec![0, 1, 2]);
        assert_eq!(model.as_deref(), Some("gemma"));
    }

    #[test]
    fn extract_layers_from_json_scalar() {
        let body = br#"{"layer":7}"#;
        let (layers, model) = extract_layers_and_model_id(body, false).unwrap();
        assert_eq!(layers, vec![7]);
        assert!(model.is_none());
    }

    #[test]
    fn extract_layers_json_missing_fields_errors() {
        let body = br#"{"foo":"bar"}"#;
        let err = extract_layers_and_model_id(body, false).unwrap_err();
        assert!(err.contains("must provide"));
    }

    #[test]
    fn extract_layers_invalid_json_errors() {
        let err = extract_layers_and_model_id(b"not json", false).unwrap_err();
        assert!(err.contains("invalid JSON"));
    }

    #[test]
    fn extract_layers_json_filters_non_numeric_entries() {
        let body = br#"{"layers":[0,"oops",2]}"#;
        let (layers, _) = extract_layers_and_model_id(body, false).unwrap();
        assert_eq!(layers, vec![0, 2]);
    }
}
