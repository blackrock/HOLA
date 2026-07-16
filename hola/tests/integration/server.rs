// Copyright 2026 BlackRock, Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Integration tests for the REST API (in-process HTTP).
//!
//! Exercises ask/tell/top_k/trials endpoints, space/objectives info,
//! checkpoints, error handling, cancel, and objective rescalarization.

use hola::hola_engine::{HolaEngine, ObjectiveConfig, ParamConfig, StrategyConfig, StudyConfig};
use hola::server::{ServerOptions, create_router, create_router_with_options, serve_with_options};
use http_body_util::BodyExt;
use serde_json::json;
use std::collections::BTreeMap;
use tower::ServiceExt;

// ==========================================================================
// Helpers
// ==========================================================================

fn minimal_config() -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: None,
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    }
}

fn sobol_config(seed: u64) -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: Some(StrategyConfig {
            strategy_type: "sobol".to_string(),
            refit_interval: 20,
            total_budget: None,
            exploration_budget: None,
            seed: Some(seed),
            elite_fraction: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    }
}

fn multi_param_config() -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([
            (
                "lr".to_string(),
                ParamConfig::Real {
                    min: 0.001,
                    max: 1.0,
                    scale: "log10".to_string(),
                },
            ),
            (
                "layers".to_string(),
                ParamConfig::Integer { min: 1, max: 10 },
            ),
            (
                "opt".to_string(),
                ParamConfig::Categorical {
                    choices: vec!["adam".into(), "sgd".into()],
                },
            ),
        ]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: None,
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    }
}

async fn json_request(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Option<serde_json::Value>,
) -> (u16, serde_json::Value) {
    json_request_with_headers(app, method, uri, body, &[]).await
}

async fn json_request_with_headers(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Option<serde_json::Value>,
    headers: &[(&str, &str)],
) -> (u16, serde_json::Value) {
    let mut builder = hyper::Request::builder().method(method).uri(uri);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let body = if let Some(b) = body {
        builder = builder.header("content-type", "application/json");
        axum::body::Body::from(serde_json::to_vec(&b).unwrap())
    } else {
        axum::body::Body::empty()
    };
    let req = builder.body(body).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

async fn options_request(
    app: axum::Router,
    uri: &str,
    origin: &str,
    requested_method: &str,
) -> hyper::Response<axum::body::Body> {
    let req = hyper::Request::builder()
        .method("OPTIONS")
        .uri(uri)
        .header("origin", origin)
        .header("access-control-request-method", requested_method)
        .body(axum::body::Body::empty())
        .unwrap();
    app.oneshot(req).await.unwrap()
}

async fn first_sse_chunk(response: hyper::Response<axum::body::Body>) -> String {
    let mut body = response.into_body();
    loop {
        let frame = tokio::time::timeout(std::time::Duration::from_secs(1), body.frame())
            .await
            .expect("SSE stream should yield a frame")
            .expect("SSE stream should remain open")
            .expect("SSE frame should be valid");
        if let Ok(data) = frame.into_data() {
            if !data.is_empty() {
                return String::from_utf8(data.to_vec()).unwrap();
            }
        }
    }
}

// ==========================================================================
// Core flow: ask -> tell -> top_k -> trials
// ==========================================================================

#[tokio::test]
async fn test_server_ask_endpoint() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let req = hyper::Request::builder()
        .method("POST")
        .uri("/api/ask")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let trial: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(trial["trial_id"], 0);
    assert!(trial["params"]["x"].is_number());
}

#[tokio::test]
async fn test_server_ask_idempotency_key_replays_the_allocated_trial() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let headers = [("idempotency-key", "worker-7-attempt-19")];
    let (first_status, first) =
        json_request_with_headers(app.clone(), "POST", "/api/ask", None, &headers).await;
    let (retry_status, retry) =
        json_request_with_headers(app.clone(), "POST", "/api/ask", None, &headers).await;
    let (next_status, next) = json_request_with_headers(
        app,
        "POST",
        "/api/ask",
        None,
        &[("idempotency-key", "worker-7-attempt-20")],
    )
    .await;

    assert_eq!(first_status, 200);
    assert_eq!(retry_status, 200);
    assert_eq!(next_status, 200);
    assert_eq!(retry, first);
    assert_eq!(first["trial_id"], 0);
    assert_eq!(next["trial_id"], 1);
}

#[tokio::test]
async fn test_server_ask_rejects_invalid_idempotency_keys() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();
    let oversized = "x".repeat(129);

    let (empty_status, empty) = json_request_with_headers(
        app.clone(),
        "POST",
        "/api/ask",
        None,
        &[("idempotency-key", "")],
    )
    .await;
    let (oversized_status, oversized_body) = json_request_with_headers(
        app,
        "POST",
        "/api/ask",
        None,
        &[("idempotency-key", &oversized)],
    )
    .await;

    assert_eq!(empty_status, 400);
    assert_eq!(oversized_status, 400);
    assert_eq!(empty["code"], "invalid_idempotency_key");
    assert_eq!(oversized_body["code"], "invalid_idempotency_key");
}

#[tokio::test]
async fn test_server_trial_leases_expire_and_heartbeat_renews_pending_work() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    options.lease_duration = std::time::Duration::from_millis(40);
    let app = create_router_with_options(engine, options).unwrap();

    let (_, expired) = json_request(app.clone(), "POST", "/api/ask", None).await;
    tokio::time::sleep(std::time::Duration::from_millis(75)).await;
    let (_, replacement) = json_request(app.clone(), "POST", "/api/ask", None).await;
    assert_eq!(expired["trial_id"], 0);
    assert_eq!(replacement["trial_id"], 1);

    let expired_tell = json!({"trial_id": 0, "metrics": {"loss": 0.5}});
    let (status, body) = json_request(app.clone(), "POST", "/api/tell", Some(expired_tell)).await;
    assert_eq!(status, 400);
    assert_eq!(body["code"], "tell_failed");

    let heartbeat = json!({"trial_id": 1});
    let (status, body) = json_request(app.clone(), "POST", "/api/heartbeat", Some(heartbeat)).await;
    assert_eq!(status, 200);
    assert_eq!(body["trial_id"], 1);
    assert!(body["lease_expires_at_ms"].as_u64().is_some());

    let tell = json!({"trial_id": 1, "metrics": {"loss": 0.25}});
    let (status, _) = json_request(app, "POST", "/api/tell", Some(tell)).await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn test_server_ask_tell_top_k_flow() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    // Ask
    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let trial_id = trial["trial_id"].as_u64().unwrap();

    // Tell
    let tell = json!({"trial_id": trial_id, "metrics": {"loss": 0.42}});
    let (status, result) = json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;
    assert_eq!(status, 200);
    assert_eq!(result["status"], "ok");
    assert_eq!(result["trial_count"], 1);
    assert_eq!(result["trial"]["trial_id"], trial_id);
    assert!(result["trial"]["score_vector"].is_object());

    // Top-k
    let (status, top) = json_request(
        app.clone(),
        "GET",
        "/api/top_k?k=1&include_infeasible=false",
        None,
    )
    .await;
    assert_eq!(status, 200);
    let top_arr = top.as_array().unwrap();
    assert_eq!(top_arr.len(), 1);
    assert_eq!(top_arr[0]["trial_id"], 0);
    assert!(top_arr[0]["params"].is_object());
    assert!(top_arr[0]["metrics"].is_object());
    assert!(top_arr[0]["scores"].is_object());
    assert!(top_arr[0]["rank"].is_u64());

    // Single-trial lookup
    let (status, single) = json_request(
        app.clone(),
        "GET",
        &format!("/api/trial/{trial_id}?include_infeasible=true"),
        None,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(single["trial_id"], trial_id);
    assert_eq!(single["metrics"]["loss"], 0.42);

    // Trials
    let (status, trials) = json_request(
        app,
        "GET",
        "/api/trials?sorted_by=index&include_infeasible=true",
        None,
    )
    .await;
    assert_eq!(status, 200);
    let trials_arr = trials.as_array().unwrap();
    assert_eq!(trials_arr.len(), 1);
}

#[tokio::test]
async fn test_server_top_k_empty() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (status, top) =
        json_request(app, "GET", "/api/top_k?k=1&include_infeasible=false", None).await;
    assert_eq!(status, 200);
    assert!(top.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_server_trial_count() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (status, body) = json_request(app, "GET", "/api/trial_count", None).await;
    assert_eq!(status, 200);
    assert_eq!(body["trial_count"], 0);
}

#[tokio::test]
async fn test_server_completed_count_remains_monotonic_when_history_is_bounded() {
    let mut config = minimal_config();
    config.max_leaderboard_size = Some(1);
    let engine = HolaEngine::from_config(config).unwrap();
    let app = create_router(engine).unwrap();

    for expected in 1..=2 {
        let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
        let (status, response) = json_request(
            app.clone(),
            "POST",
            "/api/tell",
            Some(json!({
                "trial_id": trial["trial_id"],
                "metrics": {"loss": expected as f64},
            })),
        )
        .await;
        assert_eq!(status, 200);
        assert_eq!(response["trial_count"], expected);
    }

    let (_, count) = json_request(app.clone(), "GET", "/api/trial_count", None).await;
    assert_eq!(count["trial_count"], 2);
    let (_, trials) = json_request(
        app,
        "GET",
        "/api/trials?sorted_by=index&include_infeasible=true",
        None,
    )
    .await;
    assert_eq!(trials.as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn test_trial_status_survives_bounded_leaderboard_eviction() {
    let mut config = minimal_config();
    config.max_leaderboard_size = Some(1);
    let engine = HolaEngine::from_config(config).unwrap();
    let app = create_router(engine).unwrap();

    let (_, evicted) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let evicted_id = evicted["trial_id"].as_u64().unwrap();
    let (status, _) = json_request(
        app.clone(),
        "POST",
        "/api/tell",
        Some(json!({"trial_id": evicted_id, "metrics": {"loss": 2.0}})),
    )
    .await;
    assert_eq!(status, 200);

    let (_, retained) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let retained_id = retained["trial_id"].as_u64().unwrap();
    let (status, _) = json_request(
        app.clone(),
        "POST",
        "/api/tell",
        Some(json!({"trial_id": retained_id, "metrics": {"loss": 1.0}})),
    )
    .await;
    assert_eq!(status, 200);

    let (_, trials) = json_request(
        app.clone(),
        "GET",
        "/api/trials?sorted_by=index&include_infeasible=true",
        None,
    )
    .await;
    assert_eq!(trials.as_array().unwrap().len(), 1);
    assert!(
        trials
            .as_array()
            .unwrap()
            .iter()
            .all(|trial| trial["trial_id"] != evicted_id),
        "the completed status must come from the receipt after leaderboard eviction"
    );

    let (status, lifecycle) = json_request(
        app.clone(),
        "GET",
        &format!("/api/trial/{evicted_id}/status"),
        None,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(
        lifecycle,
        json!({"status": "ok", "trial_id": evicted_id, "state": "completed"})
    );

    let (_, pending) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let pending_id = pending["trial_id"].as_u64().unwrap();
    let (status, lifecycle) = json_request(
        app.clone(),
        "GET",
        &format!("/api/trial/{pending_id}/status"),
        None,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(
        lifecycle,
        json!({"status": "ok", "trial_id": pending_id, "state": "pending"})
    );

    let (status, _) = json_request(
        app.clone(),
        "POST",
        "/api/cancel",
        Some(json!({"trial_id": pending_id})),
    )
    .await;
    assert_eq!(status, 200);
    let (status, lifecycle) =
        json_request(app, "GET", &format!("/api/trial/{pending_id}/status"), None).await;
    assert_eq!(status, 200);
    assert_eq!(
        lifecycle,
        json!({"status": "ok", "trial_id": pending_id, "state": "not_pending"})
    );
}

// ==========================================================================
// Security options: auth and CORS
// ==========================================================================

#[tokio::test]
async fn test_server_auth_rejects_missing_and_invalid_bearer() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    options.auth_token = Some("secret".to_string());
    let app = create_router_with_options(engine, options).unwrap();

    let (status, body) = json_request(app.clone(), "POST", "/api/ask", None).await;
    assert_eq!(status, 401);
    assert!(body["error"].as_str().unwrap().contains("bearer token"));

    for (method, uri, body) in [
        (
            "POST",
            "/api/tell",
            Some(json!({"trial_id": 0, "metrics": {"loss": 0.5}})),
        ),
        ("POST", "/api/cancel", Some(json!({"trial_id": 0}))),
        (
            "PATCH",
            "/api/objectives",
            Some(json!({"objectives": [{"field": "loss", "type": "minimize"}]})),
        ),
        ("POST", "/api/checkpoint/save", Some(json!({}))),
    ] {
        let (status, body) = json_request(app.clone(), method, uri, body).await;
        assert_eq!(status, 401, "{method} {uri}");
        assert!(body["error"].as_str().unwrap().contains("bearer token"));
    }

    let (status, body) = json_request_with_headers(
        app,
        "POST",
        "/api/ask",
        None,
        &[("authorization", "Bearer wrong")],
    )
    .await;
    assert_eq!(status, 401);
    assert!(body["error"].as_str().unwrap().contains("bearer token"));
}

#[tokio::test]
async fn test_server_auth_accepts_valid_bearer_for_mutations() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    options.auth_token = Some("secret".to_string());
    let app = create_router_with_options(engine, options).unwrap();

    let (status, trial) = json_request_with_headers(
        app.clone(),
        "POST",
        "/api/ask",
        None,
        &[("authorization", "Bearer secret")],
    )
    .await;
    assert_eq!(status, 200);

    let tell = json!({"trial_id": trial["trial_id"], "metrics": {"loss": 0.5}});
    let (status, body) = json_request_with_headers(
        app,
        "POST",
        "/api/tell",
        Some(tell),
        &[("authorization", "Bearer secret")],
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn test_server_read_only_token_cannot_mutate() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    options.auth_token = Some("admin-secret".to_string());
    options.read_auth_token = Some("monitor-secret".to_string());
    let app = create_router_with_options(engine, options).unwrap();

    let read_headers = [("authorization", "Bearer monitor-secret")];
    let (status, _) =
        json_request_with_headers(app.clone(), "GET", "/api/space", None, &read_headers).await;
    assert_eq!(status, 200);
    let (status, body) =
        json_request_with_headers(app.clone(), "POST", "/api/ask", None, &read_headers).await;
    assert_eq!(status, 401);
    assert_eq!(body["code"], "unauthorized");

    let (status, _) = json_request_with_headers(
        app,
        "POST",
        "/api/ask",
        None,
        &[("authorization", "Bearer admin-secret")],
    )
    .await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn test_server_token_protects_reads_and_sse_by_default() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    options.auth_token = Some("secret".to_string());
    let app = create_router_with_options(engine, options).unwrap();

    let (status, _) = json_request(app.clone(), "GET", "/api/trial_count", None).await;
    assert_eq!(status, 401, "reads must be protected by default");
    let (status, _) = json_request(app.clone(), "GET", "/api/trial/0/status", None).await;
    assert_eq!(status, 401, "trial lifecycle reads must be protected");

    let (status, _) = json_request_with_headers(
        app.clone(),
        "GET",
        "/api/trial_count",
        None,
        &[("authorization", "Bearer secret")],
    )
    .await;
    assert_eq!(status, 200);
    let (status, lifecycle) = json_request_with_headers(
        app.clone(),
        "GET",
        "/api/trial/0/status",
        None,
        &[("authorization", "Bearer secret")],
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(
        lifecycle,
        json!({"status": "ok", "trial_id": 0, "state": "not_pending"})
    );

    // SSE stream is open too. Check the initial response status only; the body
    // is a long-lived stream, so don't consume it.
    let req = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status().as_u16(),
        401,
        "SSE must be protected by default"
    );
}

#[tokio::test]
async fn test_server_read_auth_can_be_explicitly_disabled() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    options.auth_token = Some("secret".to_string());
    options.require_read_auth = false;
    let app = create_router_with_options(engine, options).unwrap();

    let (status, _) = json_request(app.clone(), "GET", "/api/trial_count", None).await;
    assert_eq!(status, 200, "explicit opt-out must leave reads open");

    // SSE is open under the same explicit opt-out.
    let req = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "explicit opt-out must leave SSE open"
    );
}

#[tokio::test]
async fn test_server_sse_ids_and_replays_missed_events() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (status, cursor) = json_request(app.clone(), "GET", "/api/event_cursor", None).await;
    assert_eq!(status, 200);
    assert_eq!(cursor["last_event_id"], "0");

    let (_, first) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let first_tell = json!({"trial_id": first["trial_id"], "metrics": {"loss": 0.5}});
    json_request(app.clone(), "POST", "/api/tell", Some(first_tell)).await;

    let replay_request = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .header("last-event-id", cursor["last_event_id"].as_str().unwrap())
        .body(axum::body::Body::empty())
        .unwrap();
    let replay_response = app.clone().oneshot(replay_request).await.unwrap();
    assert_eq!(replay_response.status(), 200);
    let first_event = first_sse_chunk(replay_response).await;
    assert!(
        first_event.contains("id: 1"),
        "unexpected SSE: {first_event}"
    );
    assert!(first_event.contains("TrialCompleted"));

    let (status, cursor) = json_request(app.clone(), "GET", "/api/event_cursor", None).await;
    assert_eq!(status, 200);
    assert_eq!(cursor["last_event_id"], "1");

    let (_, second) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let second_tell = json!({"trial_id": second["trial_id"], "metrics": {"loss": 0.25}});
    json_request(app.clone(), "POST", "/api/tell", Some(second_tell)).await;

    let reconnect_request = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .header("last-event-id", "1")
        .body(axum::body::Body::empty())
        .unwrap();
    let reconnect_response = app.oneshot(reconnect_request).await.unwrap();
    let second_event = first_sse_chunk(reconnect_response).await;
    assert!(
        second_event.contains("id: 2"),
        "unexpected SSE: {second_event}"
    );
    assert!(!second_event.contains("id: 1\n"));
}

#[tokio::test]
async fn test_server_sse_resets_when_requested_history_expired() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    for index in 0..257u64 {
        let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
        let tell = json!({
            "trial_id": trial["trial_id"],
            "metrics": {"loss": index as f64},
        });
        let (status, _) = json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;
        assert_eq!(status, 200);
    }

    let request = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .header("last-event-id", "0")
        .body(axum::body::Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let reset = first_sse_chunk(response).await;
    assert!(
        reset.contains("event: stream_reset"),
        "unexpected SSE: {reset}"
    );
    assert!(reset.contains("event history expired"));
}

#[tokio::test]
async fn test_server_sse_resets_when_cursor_is_from_an_older_server_process() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    // A fresh process starts its journal at zero. A browser may still carry a
    // higher Last-Event-ID from the process it just lost; silently accepting
    // it would leave the old dashboard snapshot in place forever.
    let request = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .header("last-event-id", "41")
        .body(axum::body::Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let reset = first_sse_chunk(response).await;
    assert!(
        reset.contains("event: stream_reset"),
        "unexpected SSE: {reset}"
    );
    assert!(reset.contains("event cursor is ahead of this server"));
}

#[tokio::test]
async fn test_server_sse_reports_broadcast_lag() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();
    let stream_request = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .body(axum::body::Body::empty())
        .unwrap();
    let stream_response = app.clone().oneshot(stream_request).await.unwrap();

    // Do not poll the response body until the bounded broadcast channel has
    // overflowed. The first body item must explicitly tell the client to
    // reconcile rather than silently dropping the gap.
    for index in 0..257u64 {
        let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
        let tell = json!({
            "trial_id": trial["trial_id"],
            "metrics": {"loss": index as f64},
        });
        json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;
    }

    let lagged = first_sse_chunk(stream_response).await;
    assert!(
        lagged.contains("event: stream_lagged"),
        "unexpected SSE: {lagged}"
    );
    assert!(lagged.contains("missed"));
}

#[tokio::test]
async fn test_health_and_readiness_endpoints() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();
    let (status, health) = json_request(app.clone(), "GET", "/healthz", None).await;
    assert_eq!(status, 200);
    assert_eq!(health["status"], "ok");
    let (status, ready) = json_request(app.clone(), "GET", "/readyz", None).await;
    assert_eq!(status, 200);
    assert_eq!(ready["status"], "ready");
    assert!(
        ready.get("trial_count").is_none(),
        "unauthenticated readiness must not disclose study data"
    );

    let request = hyper::Request::builder()
        .uri("/healthz")
        .body(axum::body::Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let request_id = response
        .headers()
        .get("x-request-id")
        .expect("every response should include a request ID")
        .to_str()
        .unwrap();
    assert_eq!(request_id.len(), 36);

    let request = hyper::Request::builder()
        .uri("/api/metrics")
        .body(axum::body::Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), 200);
    let metrics = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(metrics.contains("hola_trials_completed 0"));
    assert!(metrics.contains("hola_trials_retained 0"));
    assert!(metrics.contains("hola_http_requests_total"));
    assert!(metrics.contains("hola_http_failures_total"));
    assert!(metrics.contains("hola_http_request_duration_seconds_sum"));
    assert!(metrics.contains("hola_checkpoint_failures_total"));
    assert!(metrics.contains("hola_refit_failures_total"));
    assert!(metrics.contains("hola_events_published_total"));
}

#[tokio::test]
async fn test_non_loopback_library_bind_requires_token() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(0);
    options.host = "0.0.0.0".to_string();
    let error = serve_with_options(engine, options).await.unwrap_err();
    assert!(error.to_string().contains("auth token"));
}

#[test]
fn test_server_rejects_empty_programmatic_auth_tokens() {
    for token in ["", "   "] {
        let engine = HolaEngine::from_config(minimal_config()).unwrap();
        let mut options = ServerOptions::new(8000);
        options.auth_token = Some(token.to_string());
        let error = create_router_with_options(engine, options).unwrap_err();
        assert!(error.to_string().contains("auth_token must not be empty"));

        let engine = HolaEngine::from_config(minimal_config()).unwrap();
        let mut options = ServerOptions::new(8000);
        options.auth_token = Some("admin".to_string());
        options.read_auth_token = Some(token.to_string());
        let error = create_router_with_options(engine, options).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("read_auth_token must not be empty")
        );
    }
}

#[tokio::test]
async fn test_server_cors_allows_configured_origin_only() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    options.cors_allowed_origins = vec!["http://allowed.example".to_string()];
    let app = create_router_with_options(engine, options).unwrap();

    let allowed = options_request(app.clone(), "/api/ask", "http://allowed.example", "POST").await;
    assert_eq!(
        allowed
            .headers()
            .get("access-control-allow-origin")
            .unwrap(),
        "http://allowed.example"
    );

    let disallowed =
        options_request(app.clone(), "/api/ask", "http://disallowed.example", "POST").await;
    assert!(
        disallowed
            .headers()
            .get("access-control-allow-origin")
            .is_none()
    );

    let sse_preflight = hyper::Request::builder()
        .method("OPTIONS")
        .uri("/api/events")
        .header("origin", "http://allowed.example")
        .header("access-control-request-method", "GET")
        .header("access-control-request-headers", "last-event-id")
        .body(axum::body::Body::empty())
        .unwrap();
    let sse_preflight = app.clone().oneshot(sse_preflight).await.unwrap();
    assert_eq!(sse_preflight.status(), 200);
    assert_eq!(
        sse_preflight
            .headers()
            .get("access-control-allow-origin")
            .unwrap(),
        "http://allowed.example"
    );
    assert!(
        sse_preflight
            .headers()
            .get("access-control-allow-headers")
            .unwrap()
            .to_str()
            .unwrap()
            .split(',')
            .any(|header| header.trim().eq_ignore_ascii_case("last-event-id"))
    );

    let ask_preflight = hyper::Request::builder()
        .method("OPTIONS")
        .uri("/api/ask")
        .header("origin", "http://allowed.example")
        .header("access-control-request-method", "POST")
        .header("access-control-request-headers", "idempotency-key")
        .body(axum::body::Body::empty())
        .unwrap();
    let ask_preflight = app.oneshot(ask_preflight).await.unwrap();
    assert_eq!(ask_preflight.status(), 200);
    assert!(
        ask_preflight
            .headers()
            .get("access-control-allow-headers")
            .unwrap()
            .to_str()
            .unwrap()
            .split(',')
            .any(|header| header.trim().eq_ignore_ascii_case("idempotency-key"))
    );
}

#[tokio::test]
async fn test_server_rejects_disallowed_origin_before_simple_post_mutates_state() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine.clone()).unwrap();

    // An empty POST is a browser "simple request": the browser dispatches it
    // without a preflight, so merely omitting Access-Control-Allow-Origin would
    // still let a hostile page allocate work.
    let (status, body) = json_request_with_headers(
        app,
        "POST",
        "/api/ask",
        None,
        &[
            ("origin", "https://attacker.example"),
            ("host", "127.0.0.1:8000"),
        ],
    )
    .await;

    assert_eq!(status, 403);
    assert_eq!(body["code"], "origin_forbidden");
    assert_eq!(engine.pending_count().await, 0);
}

#[tokio::test]
async fn test_server_accepts_same_origin_and_explicitly_allowed_origin_posts() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    options.cors_allowed_origins = vec!["https://worker.example".to_string()];
    let app = create_router_with_options(engine, options).unwrap();

    let (same_origin_status, _) = json_request_with_headers(
        app.clone(),
        "POST",
        "/api/ask",
        None,
        &[
            ("origin", "http://127.0.0.1:8000"),
            ("host", "127.0.0.1:8000"),
        ],
    )
    .await;
    let (allowed_status, _) = json_request_with_headers(
        app,
        "POST",
        "/api/ask",
        None,
        &[
            ("origin", "https://worker.example"),
            ("host", "127.0.0.1:8000"),
        ],
    )
    .await;

    assert_eq!(same_origin_status, 200);
    assert_eq!(allowed_status, 200);
}

#[tokio::test]
async fn test_server_cors_malformed_origin_errors_without_panic() {
    // A bad operator-configured origin must surface a clean error naming the
    // offending value, not panic while building the router.
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    // Control characters are not valid HTTP header values.
    options.cors_allowed_origins = vec!["http://bad\norigin".to_string()];

    let result = create_router_with_options(engine, options);
    assert!(result.is_err());
    let message = result.err().unwrap().to_string();
    assert!(
        message.contains("CORS origin"),
        "error should name the CORS origin, got: {message}"
    );
}

// ==========================================================================
// Error handling
// ==========================================================================

#[tokio::test]
async fn test_server_tell_unknown_trial() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let tell = json!({"trial_id": 999, "metrics": {"loss": 0.5}});
    let (status, err) = json_request(app, "POST", "/api/tell", Some(tell)).await;
    assert_eq!(status, 400);
    assert!(err["error"].as_str().unwrap().contains("999"));
}

#[tokio::test]
async fn test_server_tell_is_idempotent_but_rejects_conflicting_metrics() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let tell = json!({"trial_id": trial["trial_id"], "metrics": {"loss": 0.5}});

    let (status, _) = json_request(app.clone(), "POST", "/api/tell", Some(tell.clone())).await;
    assert_eq!(status, 200);

    let (status, body) = json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;
    assert_eq!(status, 200, "an exact retry must be idempotent");
    assert_eq!(body["trial_count"], 1);

    let conflicting = json!({"trial_id": trial["trial_id"], "metrics": {"loss": 0.7}});
    let (status, body) = json_request(app, "POST", "/api/tell", Some(conflicting)).await;
    assert_eq!(status, 400);
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("different metrics")
    );
}

#[tokio::test]
async fn test_server_duplicate_tell_replays_response_without_duplicate_completion_event() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let tell = json!({"trial_id": trial["trial_id"], "metrics": {"loss": 0.5}});
    let (first_status, first_body) =
        json_request(app.clone(), "POST", "/api/tell", Some(tell.clone())).await;
    let (retry_status, retry_body) =
        json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;

    assert_eq!(first_status, 200);
    assert_eq!(retry_status, 200);
    assert_eq!(
        retry_body, first_body,
        "retry should replay the committed response"
    );

    // Event 1 belongs to the original commit. Reconnecting after it must not
    // immediately receive event 2 for the idempotent retry.
    let replay_request = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .header("last-event-id", "1")
        .body(axum::body::Body::empty())
        .unwrap();
    // Keep one router/state clone alive so dropping the one-shot service does
    // not close the broadcast sender and end the SSE body under test.
    let replay_response = app.clone().oneshot(replay_request).await.unwrap();
    assert_eq!(replay_response.status(), 200);
    let mut body = replay_response.into_body();
    let next_event = tokio::time::timeout(std::time::Duration::from_millis(100), async {
        loop {
            let frame = body.frame().await?;
            let frame = frame.expect("SSE frame should be valid");
            if let Ok(data) = frame.into_data() {
                let text = String::from_utf8_lossy(&data);
                if text.contains("id:") {
                    return Some(text.into_owned());
                }
            }
        }
    })
    .await;
    assert!(
        !matches!(next_event, Ok(Some(_))),
        "an idempotent tell retry emitted a duplicate SSE event: {next_event:?}"
    );
}

// ==========================================================================
// Cancel
// ==========================================================================

#[tokio::test]
async fn test_server_cancel_endpoint() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let cancel = json!({"trial_id": trial["trial_id"]});
    let (status, body) = json_request(app, "POST", "/api/cancel", Some(cancel)).await;
    assert_eq!(status, 200);
    assert_eq!(body["status"], "ok");
}

// ==========================================================================
// Info endpoints: space, objectives
// ==========================================================================

#[tokio::test]
async fn test_server_space_endpoint() {
    let engine = HolaEngine::from_config(multi_param_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (status, body) = json_request(app, "GET", "/api/space", None).await;
    assert_eq!(status, 200);

    let params = body["params"].as_array().unwrap();
    assert_eq!(params.len(), 3);
    let names: Vec<&str> = params.iter().map(|p| p["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"lr"));
    assert!(names.contains(&"layers"));
    assert!(names.contains(&"opt"));
}

#[tokio::test]
async fn test_server_space_with_all_param_types() {
    let engine = HolaEngine::from_config(multi_param_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (_, body) = json_request(app, "GET", "/api/space", None).await;
    let params = body["params"].as_array().unwrap();

    let find =
        |name: &str| -> &serde_json::Value { params.iter().find(|p| p["name"] == name).unwrap() };
    assert_eq!(find("lr")["type"], "real");
    assert_eq!(find("lr")["scale"], "log10");
    assert_eq!(find("layers")["type"], "integer");
    assert_eq!(find("opt")["type"], "categorical");
    assert_eq!(find("opt")["choices"], json!(["adam", "sgd"]));
}

#[tokio::test]
async fn test_server_get_objectives_endpoint() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (status, body) = json_request(app, "GET", "/api/objectives", None).await;
    assert_eq!(status, 200);
    assert_eq!(body["objectives"].as_array().unwrap().len(), 1);
    assert_eq!(body["objectives"][0]["field"], "loss");
}

// ==========================================================================
// Objectives: update and rescalarize
// ==========================================================================

#[tokio::test]
async fn test_server_update_objectives() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    // Connect a second client before the mutation. Objective changes can move
    // every historical rank, so an already-live dashboard must be told to
    // refresh even though no trial completed.
    let stream_request = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .body(axum::body::Body::empty())
        .unwrap();
    let stream_response = app.clone().oneshot(stream_request).await.unwrap();
    assert_eq!(stream_response.status(), 200);

    let patch = json!({"objectives": [{"field": "accuracy", "type": "maximize", "priority": 1.0}]});
    let (status, result) = json_request(app, "PATCH", "/api/objectives", Some(patch)).await;
    assert_eq!(status, 200);
    assert_eq!(result["status"], "ok");

    let event = first_sse_chunk(stream_response).await;
    assert!(event.contains("id: 1"), "unexpected SSE: {event}");
    assert!(
        event.contains("\"type\":\"ObjectivesChanged\""),
        "unexpected SSE: {event}"
    );
}

#[tokio::test]
async fn test_server_update_objectives_rejects_invalid_type() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let patch = json!({"objectives": [{"field": "accuracy", "type": "larger", "priority": 1.0}]});
    let (status, result) = json_request(app, "PATCH", "/api/objectives", Some(patch)).await;
    assert_eq!(status, 400);
    assert!(
        result["error"]
            .as_str()
            .unwrap()
            .contains("Objective 'accuracy'")
    );
}

#[tokio::test]
async fn test_server_update_objectives_rescalarizes() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let tell = json!({"trial_id": trial["trial_id"], "metrics": {"loss": 0.5, "accuracy": 0.9}});
    json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;

    // Verify trial exists before rescalarization
    let (_, top_before) = json_request(
        app.clone(),
        "GET",
        "/api/top_k?k=1&include_infeasible=true",
        None,
    )
    .await;
    assert!(!top_before.as_array().unwrap().is_empty());

    let patch = json!({"objectives": [{"field": "accuracy", "type": "maximize", "priority": 1.0}]});
    json_request(app.clone(), "PATCH", "/api/objectives", Some(patch)).await;

    // After rescalarization, trial still exists
    let (_, top_after) =
        json_request(app, "GET", "/api/top_k?k=1&include_infeasible=true", None).await;
    assert!(!top_after.as_array().unwrap().is_empty());
}

// ==========================================================================
// Sequential asks + monotonic IDs
// ==========================================================================

#[tokio::test]
async fn test_server_ask_sequential_ids() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let mut prev_id: Option<u64> = None;
    for _ in 0..5 {
        let (status, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
        assert_eq!(status, 200);
        let id = trial["trial_id"].as_u64().unwrap();
        if let Some(prev) = prev_id {
            assert!(id > prev);
        }
        prev_id = Some(id);
    }
}

// ==========================================================================
// Checkpoint save
// ==========================================================================

#[tokio::test]
async fn test_server_creates_configured_checkpoint_directory() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let checkpoint_dir = root.path().join("missing/nested");
    let mut options = ServerOptions::new(8000);
    options.checkpoint_dir = checkpoint_dir.clone();

    let _app = create_router_with_options(engine, options).unwrap();
    assert!(checkpoint_dir.is_dir());
}

#[tokio::test]
async fn test_server_checkpoint_save_endpoint() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut options = ServerOptions::new(8000);
    options.checkpoint_dir = dir.path().to_path_buf();
    let app = create_router_with_options(engine, options).unwrap();

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let tell = json!({"trial_id": trial["trial_id"], "metrics": {"loss": 0.5}});
    json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;

    let save_req = json!({"description": "server test"});

    let (status, body) =
        json_request(app.clone(), "POST", "/api/checkpoint/save", Some(save_req)).await;
    assert_eq!(status, 200);
    assert_eq!(body["checkpoint_type"], "full");
    let path = std::path::PathBuf::from(body["path"].as_str().unwrap());
    assert_eq!(path.parent(), Some(dir.path()));
    assert!(
        path.file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("checkpoint_")
    );
    assert!(path.exists());

    let saved: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
    assert!(saved.get("config").is_some());
    assert!(saved["checkpoint"].get("strategy_state").is_some());
    assert_eq!(
        body["trials_saved"], saved["checkpoint"]["metadata"]["n_trials"],
        "the response count must come from the exact written snapshot"
    );

    let (status, second) = json_request(
        app,
        "POST",
        "/api/checkpoint/save",
        Some(json!({"description": "second server checkpoint"})),
    )
    .await;
    assert_eq!(status, 200);
    assert_ne!(second["path"], body["path"]);
    assert!(std::path::Path::new(second["path"].as_str().unwrap()).exists());
}

#[tokio::test]
async fn test_server_checkpoint_save_preserves_sobol_sequence() {
    let config = sobol_config(123);
    let baseline = HolaEngine::from_config(config.clone()).unwrap();
    let engine = HolaEngine::from_config(config).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut options = ServerOptions::new(8000);
    options.checkpoint_dir = dir.path().to_path_buf();
    let app = create_router_with_options(engine, options).unwrap();

    for _ in 0..3 {
        let baseline_trial = baseline.ask().await.unwrap();
        baseline
            .tell(
                baseline_trial.trial_id,
                json!({"loss": baseline_trial.params["x"]}),
            )
            .await
            .unwrap();

        let (status, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
        assert_eq!(status, 200);
        assert_eq!(trial["params"], baseline_trial.params);
        let tell =
            json!({"trial_id": trial["trial_id"], "metrics": {"loss": trial["params"]["x"]}});
        let (status, _) = json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;
        assert_eq!(status, 200);
    }
    let expected_next = baseline.ask().await.unwrap();

    let save_req = json!({"description": "server full"});
    let (status, body) = json_request(app, "POST", "/api/checkpoint/save", Some(save_req)).await;
    assert_eq!(status, 200);
    assert_eq!(body["checkpoint_type"], "full");

    let path = std::path::PathBuf::from(body["path"].as_str().unwrap());
    let restored = HolaEngine::load_from_checkpoint(&path).await.unwrap();
    let restored_next = restored.ask().await.unwrap();
    assert_eq!(restored_next.trial_id, expected_next.trial_id);
    assert_eq!(restored_next.params, expected_next.params);
}

#[tokio::test]
async fn test_server_checkpoint_save_rejects_client_controlled_paths() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut options = ServerOptions::new(8000);
    options.checkpoint_dir = dir.path().to_path_buf();
    let app = create_router_with_options(engine, options).unwrap();

    let protected = dir.path().join("study.yaml");
    std::fs::write(&protected, "important configuration\n").unwrap();

    // Even an authenticated client cannot choose an in-root filename. This is
    // stronger than traversal filtering: it prevents replacement of the study
    // YAML or any other service-writable file beside the checkpoint directory.
    for requested in [
        serde_json::json!("study.yaml"),
        serde_json::json!("../escape.json"),
        serde_json::json!(dir.path().join("escape.json").to_string_lossy()),
    ] {
        let (status, body) = json_request(
            app.clone(),
            "POST",
            "/api/checkpoint/save",
            Some(json!({"path": requested})),
        )
        .await;
        assert_eq!(status, 400);
        assert_eq!(body["code"], "invalid_request");
    }
    assert_eq!(
        std::fs::read_to_string(protected).unwrap(),
        "important configuration\n"
    );

    let (status, body) = json_request(
        app,
        "POST",
        "/api/checkpoint/save",
        Some(json!({"description": "server-generated name"})),
    )
    .await;
    assert_eq!(status, 200);
    let generated = std::path::PathBuf::from(body["path"].as_str().unwrap());
    assert_eq!(generated.parent(), Some(dir.path()));
    assert_ne!(generated, dir.path().join("study.yaml"));
    assert!(generated.exists());
}

// ==========================================================================
// Request limits: body size and max_trials cap
// ==========================================================================

#[tokio::test]
async fn test_server_rejects_oversized_body() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    // Build a JSON body well above the 64 KiB cap; the body-size limit must
    // reject it before the handler ever parses it.
    let big = "a".repeat(128 * 1024);
    let req = hyper::Request::builder()
        .method("POST")
        .uri("/api/tell")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(big))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    // Must be 413 (Payload Too Large) from the body-size layer, which rejects on
    // size before the JSON extractor runs. Without the layer this body would
    // instead reach the extractor and fail JSON parsing with 400, so asserting
    // exactly 413 discriminates a regression of the size cap.
    assert_eq!(
        status, 413,
        "oversized body must be rejected by the size cap"
    );
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "application/json"
    );
    let body: serde_json::Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["code"], "payload_too_large");
}

#[tokio::test]
async fn test_server_normalizes_json_extractor_errors() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();
    let request = hyper::Request::builder()
        .method("POST")
        .uri("/api/tell")
        .header("content-type", "application/json")
        .body(axum::body::Body::from("{"))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), 400);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "application/json"
    );
    let body: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["code"], "invalid_request");
}

fn max_trials_config(max_trials: usize) -> StudyConfig {
    let mut config = minimal_config();
    config.max_trials = Some(max_trials);
    config
}

#[tokio::test]
async fn test_server_ask_rejects_past_max_trials() {
    let engine = HolaEngine::from_config(max_trials_config(1)).unwrap();
    let app = create_router(engine).unwrap();

    // First ask consumes the only budgeted trial (now pending).
    let (status, _) = json_request(app.clone(), "POST", "/api/ask", None).await;
    assert_eq!(status, 200);

    // Second ask is past the configured limit and must surface the engine error.
    let (status, body) = json_request(app, "POST", "/api/ask", None).await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("max_trials"));
}

// ==========================================================================
// Pareto front
// ==========================================================================

fn multi_objective_config() -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![
            ObjectiveConfig {
                field: "loss".to_string(),
                obj_type: "minimize".to_string(),
                target: Some(0.0),
                limit: Some(5.0),
                priority: 1.0,
                group: None,
            },
            ObjectiveConfig {
                field: "latency".to_string(),
                obj_type: "minimize".to_string(),
                target: Some(0.0),
                limit: Some(100.0),
                priority: 2.0,
                group: None,
            },
        ],
        strategy: None,
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    }
}

#[tokio::test]
async fn test_server_pareto_front_multi_objective() {
    let engine = HolaEngine::from_config(multi_objective_config()).unwrap();
    let app = create_router(engine).unwrap();

    // Complete a few trials
    for i in 0..3 {
        let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
        let tell = json!({
            "trial_id": trial["trial_id"],
            "metrics": {"loss": (i as f64) * 0.5, "latency": 50.0 - (i as f64) * 10.0}
        });
        json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;
    }

    let (status, body) = json_request(
        app,
        "GET",
        "/api/pareto_front?front=0&include_infeasible=false",
        None,
    )
    .await;
    assert_eq!(status, 200);
    assert!(body.is_array());
    let front = body.as_array().unwrap();
    assert!(!front.is_empty());
    // Each trial in the front should have CompletedTrial fields
    for trial in front {
        assert!(trial["trial_id"].is_u64());
        assert!(trial["params"].is_object());
        assert!(trial["metrics"].is_object());
        assert!(trial["scores"].is_object());
        assert!(trial["rank"].is_u64());
    }
}

#[tokio::test]
async fn test_server_pareto_front_scalar_returns_empty() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    // Scalar study with no trials: pareto_front returns empty array
    let (status, body) = json_request(
        app,
        "GET",
        "/api/pareto_front?front=0&include_infeasible=false",
        None,
    )
    .await;
    assert_eq!(status, 200);
    assert!(body.as_array().unwrap().is_empty());
}

// ==========================================================================
// Multiple metric fields
// ==========================================================================

#[tokio::test]
async fn test_server_tell_with_multiple_fields() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let tell = json!({
        "trial_id": trial["trial_id"],
        "metrics": {"loss": 0.3, "accuracy": 0.9, "latency": 50.0}
    });
    let (status, body) = json_request(app, "POST", "/api/tell", Some(tell)).await;
    assert_eq!(status, 200);
    assert_eq!(body["status"], "ok");
}
