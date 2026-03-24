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

//! Axum HTTP server for the DynEngine.
//!
//! Provides REST endpoints for distributed Ask/Tell optimization,
//! Server-Sent Events for real-time dashboard integration, and
//! dashboard API endpoints for space/objectives/checkpoint management.
//!
//! # Endpoints
//!
//! - `POST /api/ask` - Request the next trial
//! - `POST /api/tell` - Report trial results
//! - `GET /api/leaderboard` - Get all completed trials
//! - `GET /api/best` - Get the best trial
//! - `GET /api/pareto_front` - Get Pareto front (multi-objective only)
//! - `PATCH /api/objectives` - Update objectives mid-run
//! - `GET /api/objectives` - Get current objectives
//! - `GET /api/space` - Get parameter space metadata
//! - `POST /api/checkpoint/save` - Save a checkpoint
//! - `GET /api/mode` - Get the server mode ("live")
//! - `GET /api/events` - SSE stream of engine events

use crate::dyn_engine::{DynEngine, DynTrial, ObjectiveConfig};
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{
        Json,
        sse::{Event, Sse},
    },
    routing::{get, patch, post},
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

// =============================================================================
// Shared state
// =============================================================================

/// Events emitted by the engine for SSE consumers.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum EngineEvent {
    TrialCompleted { trial_id: u64, score: f64 },
    RefitOccurred { n_trials: usize },
}

pub struct ServerState {
    pub engine: DynEngine,
    pub events_tx: broadcast::Sender<EngineEvent>,
}

// =============================================================================
// Request/Response types
// =============================================================================

#[derive(Deserialize)]
struct TellRequest {
    trial_id: u64,
    metrics: serde_json::Value,
}

#[derive(Deserialize)]
struct SaveCheckpointRequest {
    #[serde(default = "default_checkpoint_path")]
    path: String,
    #[serde(default)]
    description: Option<String>,
}

fn default_checkpoint_path() -> String {
    "checkpoint.json".to_string()
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// =============================================================================
// Handlers
// =============================================================================

async fn handle_ask(State(state): State<Arc<ServerState>>) -> Json<DynTrial> {
    let trial = state.engine.ask().await;
    Json(trial)
}

async fn handle_tell(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<TellRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    match state.engine.tell(req.trial_id, req.metrics).await {
        Ok(()) => {
            let n = state.engine.trial_count().await;
            let best = state.engine.best().await;
            let score = best
                .map(|b| b.observation.as_f64().unwrap_or(f64::INFINITY))
                .unwrap_or(f64::INFINITY);

            let _ = state.events_tx.send(EngineEvent::TrialCompleted {
                trial_id: req.trial_id,
                score,
            });

            Ok(Json(serde_json::json!({
                "status": "ok",
                "trial_count": n,
            })))
        }
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e }))),
    }
}

async fn handle_leaderboard(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    let trials = state.engine.trials_as_json().await;
    let total = trials.len();
    Json(serde_json::json!({
        "trials": trials,
        "total": total,
    }))
}

async fn handle_best(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    match state.engine.best().await {
        Some(best) => Json(serde_json::json!({
            "trial_id": best.trial_id,
            "candidate": best.candidate,
            "observation": best.observation,
            "raw_metrics": best.raw_metrics,
        })),
        None => Json(serde_json::json!({ "trial_id": null })),
    }
}

#[derive(Deserialize)]
struct UpdateObjectivesRequest {
    objectives: Vec<ObjectiveConfig>,
}

async fn handle_pareto_front(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    match state.engine.pareto_front().await {
        Ok(front) => {
            let trials: Vec<serde_json::Value> = front
                .into_iter()
                .map(|t| {
                    let mut objectives = serde_json::Map::new();
                    if let Some(ref raw) = t.raw_metrics {
                        for key in t.observation.keys() {
                            if let Some(val) = raw.get(key).and_then(|v| v.as_f64()) {
                                objectives.insert(
                                    key.clone(),
                                    serde_json::Value::from(val),
                                );
                            }
                        }
                    }
                    serde_json::json!({
                        "trial_id": t.trial_id,
                        "candidate": t.candidate,
                        "objectives": objectives,
                    })
                })
                .collect();
            Ok(Json(serde_json::Value::Array(trials)))
        }
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e})))),
    }
}

async fn handle_update_objectives(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<UpdateObjectivesRequest>,
) -> Json<serde_json::Value> {
    state.engine.update_objectives(req.objectives).await;
    let n = state.engine.trial_count().await;
    Json(serde_json::json!({
        "status": "ok",
        "rescalarized_trials": n,
    }))
}

async fn handle_get_objectives(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    let objectives = state.engine.objectives().await;
    Json(serde_json::json!({ "objectives": objectives }))
}

async fn handle_space(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    let params: Vec<serde_json::Value> = state
        .engine
        .space_config()
        .into_iter()
        .map(|(name, info)| {
            serde_json::json!({
                "name": name,
                "type": info.param_type,
                "min": info.min,
                "max": info.max,
                "scale": info.scale,
            })
        })
        .collect();
    Json(serde_json::json!({ "params": params }))
}

async fn handle_checkpoint_save(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<SaveCheckpointRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    match state
        .engine
        .save_leaderboard_checkpoint_to(&req.path, req.description.as_deref())
        .await
    {
        Ok(()) => {
            let n = state.engine.trial_count().await;
            Ok(Json(serde_json::json!({
                "status": "ok",
                "path": req.path,
                "trials_saved": n,
            })))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

async fn handle_mode() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "mode": "live" }))
}

async fn handle_events(
    State(state): State<Arc<ServerState>>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.events_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| match result {
        Ok(event) => {
            let data = serde_json::to_string(&event).unwrap_or_default();
            Some(Ok(Event::default().data(data)))
        }
        Err(_) => None,
    });
    Sse::new(stream)
}

// =============================================================================
// Router & Server
// =============================================================================

/// Create the Axum router for the engine server.
pub fn create_router(engine: DynEngine) -> Router {
    let (events_tx, _) = broadcast::channel(256);
    let state = Arc::new(ServerState { engine, events_tx });

    let cors = CorsLayer::permissive();

    Router::new()
        .route("/api/ask", post(handle_ask))
        .route("/api/tell", post(handle_tell))
        .route("/api/leaderboard", get(handle_leaderboard))
        .route("/api/best", get(handle_best))
        .route("/api/pareto_front", get(handle_pareto_front))
        .route(
            "/api/objectives",
            patch(handle_update_objectives).get(handle_get_objectives),
        )
        .route("/api/space", get(handle_space))
        .route("/api/checkpoint/save", post(handle_checkpoint_save))
        .route("/api/mode", get(handle_mode))
        .route("/api/events", get(handle_events))
        .layer(cors)
        .with_state(state)
}

/// Create the Axum router with the dashboard served from a local directory.
///
/// API routes under `/api/*` take priority; all other paths fall through to
/// serve static files from `dashboard_dir`.
pub fn create_router_with_dashboard(engine: DynEngine, dashboard_dir: &Path) -> Router {
    create_router(engine).fallback_service(ServeDir::new(dashboard_dir))
}

/// Start the server on the given port. Blocks until the server is shut down.
///
/// If `dashboard_dir` is provided, the dashboard UI is served at `/`.
pub async fn serve(
    engine: DynEngine,
    port: u16,
    dashboard_dir: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let router = match dashboard_dir {
        Some(dir) => create_router_with_dashboard(engine, dir),
        None => create_router(engine),
    };
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    if let Some(dir) = dashboard_dir {
        eprintln!(
            "HOLA server listening on port {port} (dashboard: {})",
            dir.display()
        );
    } else {
        eprintln!("HOLA server listening on port {port}");
    }
    axum::serve(listener, router).await?;
    Ok(())
}
