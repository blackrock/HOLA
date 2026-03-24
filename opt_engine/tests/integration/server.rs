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
//! Exercises ask/tell/best/leaderboard endpoints, space/mode/objectives info,
//! checkpoints, error handling, and objective rescalarization.

use http_body_util::BodyExt;
use opt_engine::dyn_engine::{DynEngine, ObjectiveConfig, ParamConfig, StudyConfig};
use opt_engine::server::create_router;
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
            ParamConfig::Continuous {
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
    }
}

fn multi_param_config() -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([
            (
                "lr".to_string(),
                ParamConfig::Continuous {
                    min: 0.001,
                    max: 1.0,
                    scale: "log10".to_string(),
                },
            ),
            (
                "layers".to_string(),
                ParamConfig::Discrete { min: 1, max: 10 },
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
    }
}

async fn json_request(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Option<serde_json::Value>,
) -> (u16, serde_json::Value) {
    let mut builder = hyper::Request::builder().method(method).uri(uri);
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

// ==========================================================================
// Core flow: ask → tell → best → leaderboard
// ==========================================================================

#[tokio::test]
async fn test_server_ask_endpoint() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

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
async fn test_server_ask_tell_best_flow() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

    // Ask
    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let trial_id = trial["trial_id"].as_u64().unwrap();

    // Tell
    let tell = json!({"trial_id": trial_id, "metrics": {"loss": 0.42}});
    let (status, result) = json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;
    assert_eq!(status, 200);
    assert_eq!(result["status"], "ok");
    assert_eq!(result["trial_count"], 1);

    // Best
    let (status, best) = json_request(app.clone(), "GET", "/api/best", None).await;
    assert_eq!(status, 200);
    assert_eq!(best["trial_id"], 0);
    assert!(best["observation"].is_number());

    // Leaderboard
    let (status, lb) = json_request(app, "GET", "/api/leaderboard", None).await;
    assert_eq!(status, 200);
    assert_eq!(lb["total"], 1);
    assert_eq!(lb["trials"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn test_server_best_empty() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

    let (status, best) = json_request(app, "GET", "/api/best", None).await;
    assert_eq!(status, 200);
    assert!(best["trial_id"].is_null());
}

#[tokio::test]
async fn test_server_leaderboard_empty() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

    let (status, body) = json_request(app, "GET", "/api/leaderboard", None).await;
    assert_eq!(status, 200);
    assert_eq!(body["total"], 0);
}

// ==========================================================================
// Error handling
// ==========================================================================

#[tokio::test]
async fn test_server_tell_unknown_trial() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

    let tell = json!({"trial_id": 999, "metrics": {"loss": 0.5}});
    let (status, err) = json_request(app, "POST", "/api/tell", Some(tell)).await;
    assert_eq!(status, 400);
    assert!(err["error"].as_str().unwrap().contains("999"));
}

#[tokio::test]
async fn test_server_double_tell_returns_400() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

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
// Info endpoints: space, mode, objectives
// ==========================================================================

#[tokio::test]
async fn test_server_space_endpoint() {
    let engine = DynEngine::from_config(multi_param_config()).unwrap();
    let app = create_router(engine);

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
    let engine = DynEngine::from_config(multi_param_config()).unwrap();
    let app = create_router(engine);

    let (_, body) = json_request(app, "GET", "/api/space", None).await;
    let params = body["params"].as_array().unwrap();

    let find =
        |name: &str| -> &serde_json::Value { params.iter().find(|p| p["name"] == name).unwrap() };
    assert_eq!(find("lr")["type"], "continuous");
    assert_eq!(find("lr")["scale"], "log10");
    assert_eq!(find("layers")["type"], "discrete");
    assert_eq!(find("opt")["type"], "categorical");
}

#[tokio::test]
async fn test_server_mode_endpoint() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

    let (status, body) = json_request(app, "GET", "/api/mode", None).await;
    assert_eq!(status, 200);
    assert_eq!(body["mode"], "live");
}

#[tokio::test]
async fn test_server_get_objectives_endpoint() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

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
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

    let patch = json!({"objectives": [{"field": "accuracy", "type": "maximize", "priority": 1.0}]});
    let (status, result) = json_request(app, "PATCH", "/api/objectives", Some(patch)).await;
    assert_eq!(status, 200);
    assert_eq!(result["status"], "ok");
}

#[tokio::test]
async fn test_server_update_objectives_rescalarizes() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let tell = json!({"trial_id": trial["trial_id"], "metrics": {"loss": 0.5, "accuracy": 0.9}});
    json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;

    let (_, best_before) = json_request(app.clone(), "GET", "/api/best", None).await;
    assert!(best_before["observation"].as_f64().unwrap() > 0.0);

    let patch = json!({"objectives": [{"field": "accuracy", "type": "maximize", "priority": 1.0}]});
    json_request(app.clone(), "PATCH", "/api/objectives", Some(patch)).await;

    let (_, best_after) = json_request(app, "GET", "/api/best", None).await;
    assert!(best_after["observation"].as_f64().unwrap() < 0.0);
}

// ==========================================================================
// Sequential asks + monotonic IDs
// ==========================================================================

#[tokio::test]
async fn test_server_ask_sequential_ids() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

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
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let tell = json!({"trial_id": trial["trial_id"], "metrics": {"loss": 0.5}});
    json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ckpt.json");
    let save_req = json!({"path": path.to_string_lossy(), "description": "server test"});

    let (status, _) = json_request(app, "POST", "/api/checkpoint/save", Some(save_req)).await;
    assert_eq!(status, 200);
    assert!(path.exists());
}

// ==========================================================================
// Pareto front
// ==========================================================================

fn multi_objective_config() -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    }
}

#[tokio::test]
async fn test_server_pareto_front_multi_objective() {
    let engine = DynEngine::from_config(multi_objective_config()).unwrap();
    let app = create_router(engine);

    // Complete a few trials
    for i in 0..3 {
        let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
        let tell = json!({
            "trial_id": trial["trial_id"],
            "metrics": {"loss": (i as f64) * 0.5, "latency": 50.0 - (i as f64) * 10.0}
        });
        json_request(app.clone(), "POST", "/api/tell", Some(tell)).await;
    }

    let (status, body) = json_request(app, "GET", "/api/pareto_front", None).await;
    assert_eq!(status, 200);
    assert!(body.is_array());
    let front = body.as_array().unwrap();
    assert!(!front.is_empty());
    // Each trial in the front should have trial_id, candidate, objectives
    for trial in front {
        assert!(trial["trial_id"].is_u64());
        assert!(trial["candidate"].is_object());
        assert!(trial["objectives"].is_object());
        assert!(trial["objectives"]["loss"].is_f64());
        assert!(trial["objectives"]["latency"].is_f64());
    }
}

#[tokio::test]
async fn test_server_pareto_front_scalar_returns_400() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

    let (status, _) = json_request(app, "GET", "/api/pareto_front", None).await;
    assert_eq!(status, 400);
}

// ==========================================================================
// Multiple metric fields
// ==========================================================================

#[tokio::test]
async fn test_server_tell_with_multiple_fields() {
    let engine = DynEngine::from_config(minimal_config()).unwrap();
    let app = create_router(engine);

    let (_, trial) = json_request(app.clone(), "POST", "/api/ask", None).await;
    let tell = json!({
        "trial_id": trial["trial_id"],
        "metrics": {"loss": 0.3, "accuracy": 0.9, "latency": 50.0}
    });
    let (status, body) = json_request(app, "POST", "/api/tell", Some(tell)).await;
    assert_eq!(status, 200);
    assert_eq!(body["status"], "ok");
}
