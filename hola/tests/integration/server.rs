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
use hola::server::{ServerOptions, create_router, create_router_with_options};
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
        (
            "POST",
            "/api/checkpoint/save",
            Some(json!({"path": "checkpoint.json"})),
        ),
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
async fn test_server_reads_open_by_default_with_token() {
    // Default (read auth off): read endpoints stay open even when a write token
    // is configured.
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    options.auth_token = Some("secret".to_string());
    let app = create_router_with_options(engine, options).unwrap();

    let (status, _) = json_request(app.clone(), "GET", "/api/trial_count", None).await;
    assert_eq!(
        status, 200,
        "reads must remain open without --require-read-auth"
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
        200,
        "SSE must remain open without --require-read-auth"
    );
}

#[tokio::test]
async fn test_server_read_auth_opt_in_gates_reads_and_sse() {
    // With read auth enabled, read endpoints and the SSE stream require the
    // bearer token, while remaining accessible with it.
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let mut options = ServerOptions::new(8000);
    options.auth_token = Some("secret".to_string());
    options.require_read_auth = true;
    let app = create_router_with_options(engine, options).unwrap();

    let (status, _) = json_request(app.clone(), "GET", "/api/trial_count", None).await;
    assert_eq!(
        status, 401,
        "read endpoint must reject a missing token when read-auth is on"
    );

    let (status, _) = json_request_with_headers(
        app.clone(),
        "GET",
        "/api/trial_count",
        None,
        &[("authorization", "Bearer secret")],
    )
    .await;
    assert_eq!(status, 200, "read endpoint must accept the valid token");

    // SSE stream is gated too.
    let req = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(
        resp.status().as_u16(),
        401,
        "SSE must require the token when read-auth is on"
    );

    // SSE accepts the valid token. Check the initial response status only; the
    // body is a long-lived stream, so don't consume it.
    let req = hyper::Request::builder()
        .method("GET")
        .uri("/api/events")
        .header("authorization", "Bearer secret")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "SSE must accept the valid token when read-auth is on"
    );
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

    let disallowed = options_request(app, "/api/ask", "http://disallowed.example", "POST").await;
    assert!(
        disallowed
            .headers()
            .get("access-control-allow-origin")
            .is_none()
    );
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
async fn test_server_double_tell_returns_400() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine).unwrap();

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let tell = json!({"trial_id": trial["trial_id"], "metrics": {"loss": 0.5}});

    let (status, _) = json_request(app.clone(), "POST", "/api/tell", Some(tell.clone())).await;
    assert_eq!(status, 200);

    let (status, body) = json_request(app, "POST", "/api/tell", Some(tell)).await;
    assert_eq!(status, 400);
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains(&trial["trial_id"].to_string())
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

    let patch = json!({"objectives": [{"field": "accuracy", "type": "maximize", "priority": 1.0}]});
    let (status, result) = json_request(app, "PATCH", "/api/objectives", Some(patch)).await;
    assert_eq!(status, 200);
    assert_eq!(result["status"], "ok");
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
async fn test_server_checkpoint_save_endpoint() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut options = ServerOptions::new(8000);
    options.checkpoint_dir = dir.path().to_path_buf();
    let app = create_router_with_options(engine, options).unwrap();

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let tell = json!({"trial_id": trial["trial_id"], "metrics": {"loss": 0.5}});
    json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;

    let path = dir.path().join("ckpt.json");
    let save_req = json!({"path": "ckpt.json", "description": "server test"});

    let (status, body) = json_request(app, "POST", "/api/checkpoint/save", Some(save_req)).await;
    assert_eq!(status, 200);
    assert_eq!(body["checkpoint_type"], "full");
    // The response returns the resolved checkpoint path so clients can load it back.
    assert_eq!(body["path"].as_str().unwrap(), path.to_str().unwrap());
    assert!(path.exists());

    let saved: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
    assert!(saved.get("config").is_some());
    assert!(saved["checkpoint"].get("strategy_state").is_some());
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

    let path = dir.path().join("full.json");
    let save_req = json!({"path": "full.json", "description": "server full"});
    let (status, body) = json_request(app, "POST", "/api/checkpoint/save", Some(save_req)).await;
    assert_eq!(status, 200);
    assert_eq!(body["checkpoint_type"], "full");

    let restored = HolaEngine::load_from_checkpoint(&path).await.unwrap();
    let restored_next = restored.ask().await.unwrap();
    assert_eq!(restored_next.trial_id, expected_next.trial_id);
    assert_eq!(restored_next.params, expected_next.params);
}

#[tokio::test]
async fn test_server_checkpoint_save_rejects_unconfined_paths() {
    let engine = HolaEngine::from_config(minimal_config()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut options = ServerOptions::new(8000);
    options.checkpoint_dir = dir.path().to_path_buf();
    let app = create_router_with_options(engine, options).unwrap();

    let absolute_req = json!({"path": dir.path().join("escape.json").to_string_lossy()});
    let (status, body) = json_request(
        app.clone(),
        "POST",
        "/api/checkpoint/save",
        Some(absolute_req),
    )
    .await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("relative"));

    let traversal_req = json!({"path": "../escape.json"});
    let (status, body) = json_request(
        app.clone(),
        "POST",
        "/api/checkpoint/save",
        Some(traversal_req),
    )
    .await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("relative"));

    // An empty path is rejected outright (cannot resolve to a file).
    let empty_req = json!({"path": ""});
    let (status, body) =
        json_request(app.clone(), "POST", "/api/checkpoint/save", Some(empty_req)).await;
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("empty"));

    // A symlink that lives inside the checkpoint dir but points outside it must
    // not let a lexically-confined path escape after canonicalization.
    #[cfg(unix)]
    {
        let outside = tempfile::tempdir().unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(outside.path(), &link).unwrap();

        let escape_req = json!({"path": "link/escape.json"});
        let (status, body) =
            json_request(app, "POST", "/api/checkpoint/save", Some(escape_req)).await;
        assert_eq!(status, 400);
        assert!(body["error"].as_str().unwrap().contains("relative"));
        // The file must not have been written through the symlink.
        assert!(!outside.path().join("escape.json").exists());
    }
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
