//! End-to-end round-trip test for the Exp 53 `ShardService`.
//!
//! Unit tests inside `shard_query.rs` exercise the handler trait method
//! directly. This file spins up the tonic server on an ephemeral TCP
//! port, drives one `Query` through the generated client, and asserts
//! the response decodes correctly — proving the proto + service
//! registration compile and serve correctly end-to-end.

use std::sync::Arc;

use larql_router_protocol::{ShardQuery, ShardServiceClient, ShardServiceServer};
use larql_server::shard_query::{encode_f32_le, ShardCache, ShardGrpcService};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

async fn spawn_server() -> (std::net::SocketAddr, Arc<RwLock<ShardCache>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cache = Arc::new(RwLock::new(ShardCache::new(0.97)));
    {
        let mut guard = cache.write().await;
        guard
            .seed_from_normed(
                26,
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                vec![10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0],
                2,
                4,
            )
            .unwrap();
    }
    let svc = ShardGrpcService::new(Arc::clone(&cache));
    let server_cache = Arc::clone(&cache);
    tokio::spawn(async move {
        Server::builder()
            .add_service(ShardServiceServer::new(svc))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
        // Keep cache reference alive until the server task ends (it
        // never does in these tests — the runtime is dropped first).
        drop(server_cache);
    });
    // Give tonic a tick to start accepting before we dial.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (addr, cache)
}

#[tokio::test]
async fn shard_query_round_trip_hit() {
    let (addr, _cache) = spawn_server().await;
    let mut client = ShardServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("connect to shard server");

    let resp = client
        .query(ShardQuery {
            layer_id: 26,
            k: 1,
            query_vec: encode_f32_le(&[1.0, 0.0, 0.0, 0.0]),
            tau_override: 0.0,
        })
        .await
        .expect("query rpc")
        .into_inner();

    assert!(resp.hit, "exact match should hit");
    assert!((resp.best_sim - 1.0).abs() < 1e-6);
    // Decode the mlp_out payload — should match the seeded row 0 output.
    let mlp = larql_server::shard_query::decode_f32_le(&resp.mlp_out).unwrap();
    assert_eq!(mlp, vec![10.0, 20.0, 30.0, 40.0]);
}

#[tokio::test]
async fn shard_query_round_trip_miss_below_tau() {
    let (addr, _cache) = spawn_server().await;
    let mut client = ShardServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("connect");

    // Orthogonal-to-everything query → best_sim ≈ 0 → miss.
    let resp = client
        .query(ShardQuery {
            layer_id: 26,
            k: 1,
            query_vec: encode_f32_le(&[0.0, 0.0, 1.0, 0.0]),
            tau_override: 0.0,
        })
        .await
        .expect("query rpc")
        .into_inner();

    assert!(!resp.hit);
    assert!(resp.mlp_out.is_empty());
    assert!(resp.best_sim < 0.97);
}

#[tokio::test]
async fn shard_query_round_trip_unknown_layer() {
    let (addr, _cache) = spawn_server().await;
    let mut client = ShardServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("connect");
    let resp = client
        .query(ShardQuery {
            layer_id: 99,
            k: 1,
            query_vec: encode_f32_le(&[1.0, 0.0, 0.0, 0.0]),
            tau_override: 0.0,
        })
        .await
        .expect("query rpc")
        .into_inner();
    assert!(!resp.hit);
}
