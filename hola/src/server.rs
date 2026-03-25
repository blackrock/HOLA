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

//! Axum HTTP server for the HolaEngine.
//!
//! Provides REST endpoints for distributed Ask/Tell optimization,
//! Server-Sent Events for real-time dashboard integration, and
//! dashboard API endpoints for space/objectives/checkpoint management.
//!
//! # Endpoints
//!
//! - `POST /api/ask` - Request the next trial
//! - `POST /api/tell` - Report trial results
//! - `POST /api/cancel` - Cancel a pending trial
//! - `GET /api/top_k` - Get top-k trials by rank
//! - `GET /api/pareto_front` - Get Pareto front trials
//! - `GET /api/trials` - Get all trials with scoring/ranking
//! - `GET /api/trial_count` - Get number of completed trials
//! - `PATCH /api/objectives` - Update objectives mid-run
//! - `GET /api/objectives` - Get current objectives
//! - `GET /api/space` - Get parameter space metadata
//! - `POST /api/checkpoint/save` - Save a checkpoint (internal)
//! - `GET /api/events` - SSE stream of engine events

use crate::hola_engine::{HolaEngine, ObjectiveConfig};
use axum::{
    Router,
    extract::{Query, State},
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
    pub engine: HolaEngine,
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
struct CancelRequest {
    trial_id: u64,
}

#[derive(Deserialize)]
struct TopKQuery {
    k: usize,
    #[serde(default)]
    include_infeasible: Option<bool>,
}

#[derive(Deserialize)]
struct ParetoQuery {
    #[serde(default)]
    front: Option<usize>,
    #[serde(default)]
    include_infeasible: Option<bool>,
}

#[derive(Deserialize)]
struct TrialsQuery {
    #[serde(default)]
    sorted_by: Option<String>,
    #[serde(default)]
    include_infeasible: Option<bool>,
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

#[derive(Deserialize)]
struct UpdateObjectivesRequest {
    objectives: Vec<ObjectiveConfig>,
}

// =============================================================================
// Handlers
// =============================================================================

async fn handle_ask(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    match state.engine.ask().await {
        Ok(trial) => Ok(Json(serde_json::to_value(&trial).unwrap())),
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e }))),
    }
}

async fn handle_tell(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<TellRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    match state.engine.tell(req.trial_id, req.metrics).await {
        Ok(completed) => {
            let n = state.engine.trial_count().await;

            // Extract score for SSE event from score_vector
            let score = completed
                .score_vector
                .as_object()
                .and_then(|m| m.values().next())
                .and_then(|v| v.as_f64())
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

async fn handle_cancel(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CancelRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    match state.engine.cancel(req.trial_id).await {
        Ok(()) => Ok(Json(serde_json::json!({ "status": "ok" }))),
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e }))),
    }
}

async fn handle_top_k(
    State(state): State<Arc<ServerState>>,
    Query(q): Query<TopKQuery>,
) -> Json<serde_json::Value> {
    let include_infeasible = q.include_infeasible.unwrap_or(false);
    let trials = state.engine.top_k(q.k, include_infeasible).await;
    Json(serde_json::to_value(&trials).unwrap_or_default())
}

async fn handle_pareto_front(
    State(state): State<Arc<ServerState>>,
    Query(q): Query<ParetoQuery>,
) -> Json<serde_json::Value> {
    let front = q.front.unwrap_or(0);
    let include_infeasible = q.include_infeasible.unwrap_or(false);
    let trials = state.engine.pareto_front(front, include_infeasible).await;
    Json(serde_json::to_value(&trials).unwrap_or_default())
}

async fn handle_trials(
    State(state): State<Arc<ServerState>>,
    Query(q): Query<TrialsQuery>,
) -> Json<serde_json::Value> {
    let sorted_by = q.sorted_by.as_deref().unwrap_or("index");
    let include_infeasible = q.include_infeasible.unwrap_or(true);
    let trials = state.engine.trials(sorted_by, include_infeasible).await;
    Json(serde_json::to_value(&trials).unwrap_or_default())
}

async fn handle_trial_count(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    let count = state.engine.trial_count().await;
    Json(serde_json::json!({ "trial_count": count }))
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
            let mut obj = serde_json::json!({
                "name": name,
                "type": info.param_type,
                "min": info.min,
                "max": info.max,
                "scale": info.scale,
            });
            if let Some(choices) = &info.choices {
                obj["choices"] = serde_json::json!(choices);
            }
            obj
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
pub fn create_router(engine: HolaEngine) -> Router {
    let (events_tx, _) = broadcast::channel(256);
    let state = Arc::new(ServerState { engine, events_tx });

    let cors = CorsLayer::permissive();

    Router::new()
        .route("/api/ask", post(handle_ask))
        .route("/api/tell", post(handle_tell))
        .route("/api/cancel", post(handle_cancel))
        .route("/api/top_k", get(handle_top_k))
        .route("/api/pareto_front", get(handle_pareto_front))
        .route("/api/trials", get(handle_trials))
        .route("/api/trial_count", get(handle_trial_count))
        .route(
            "/api/objectives",
            patch(handle_update_objectives).get(handle_get_objectives),
        )
        .route("/api/space", get(handle_space))
        .route("/api/checkpoint/save", post(handle_checkpoint_save))
        .route("/api/events", get(handle_events))
        .layer(cors)
        .with_state(state)
}

/// Create the Axum router with the dashboard served from a local directory.
///
/// API routes under `/api/*` take priority; all other paths fall through to
/// serve static files from `dashboard_dir`.
pub fn create_router_with_dashboard(engine: HolaEngine, dashboard_dir: &Path) -> Router {
    create_router(engine).fallback_service(ServeDir::new(dashboard_dir))
}

/// Start the server on the given port. Blocks until the server is shut down.
///
/// If `dashboard_dir` is provided, the dashboard UI is served at `/`.
pub async fn serve(
    engine: HolaEngine,
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
