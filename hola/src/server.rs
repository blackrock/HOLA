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
//! - `GET /api/trial/{trial_id}` - Get one completed trial with scoring/ranking
//! - `GET /api/trials` - Get all trials with scoring/ranking
//! - `GET /api/trial_count` - Get number of completed trials
//! - `PATCH /api/objectives` - Update objectives mid-run
//! - `GET /api/objectives` - Get current objectives
//! - `GET /api/space` - Get parameter space metadata
//! - `POST /api/checkpoint/save` - Save a full checkpoint
//! - `GET /api/events` - SSE stream of engine events

use crate::hola_engine::{CompletedTrial, HolaEngine, ObjectiveConfig};
use axum::{
    Router,
    extract::{Path as AxumPath, Query, State},
    http::{
        HeaderMap, HeaderValue, Method, StatusCode,
        header::{AUTHORIZATION, CONTENT_TYPE},
    },
    response::{
        Json,
        sse::{Event, Sse},
    },
    routing::{get, patch, post},
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::services::ServeDir;

// =============================================================================
// Shared state
// =============================================================================

/// Events emitted by the engine for SSE consumers.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum EngineEvent {
    TrialCompleted {
        trial_id: u64,
        score: f64,
        trial: CompletedTrial,
    },
    RefitOccurred {
        n_trials: usize,
    },
}

pub struct ServerState {
    pub engine: HolaEngine,
    pub events_tx: broadcast::Sender<EngineEvent>,
    auth_token: Option<String>,
    checkpoint_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ServerOptions {
    pub host: String,
    pub port: u16,
    pub dashboard_dir: Option<PathBuf>,
    pub auth_token: Option<String>,
    pub checkpoint_dir: PathBuf,
    pub cors_allowed_origins: Vec<String>,
}

impl ServerOptions {
    pub fn new(port: u16) -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port,
            dashboard_dir: None,
            auth_token: None,
            checkpoint_dir: PathBuf::from("."),
            cors_allowed_origins: Vec::new(),
        }
    }
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
struct TrialQuery {
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

fn unauthorized() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error: "Missing or invalid bearer token".to_string(),
        }),
    )
}

fn authorize_mutation(
    state: &ServerState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let Some(token) = &state.auth_token else {
        return Ok(());
    };

    let expected = format!("Bearer {token}");
    match headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    {
        Some(actual) if actual == expected => Ok(()),
        _ => Err(unauthorized()),
    }
}

fn invalid_checkpoint_path(message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
}

fn resolve_checkpoint_path(
    state: &ServerState,
    requested: &str,
) -> Result<PathBuf, (StatusCode, Json<ErrorResponse>)> {
    let path = Path::new(requested);
    if path.as_os_str().is_empty() {
        return Err(invalid_checkpoint_path("Checkpoint path must not be empty"));
    }
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(invalid_checkpoint_path(
            "Checkpoint path must be relative to the configured checkpoint directory",
        ));
    }

    Ok(state.checkpoint_dir.join(path))
}

async fn handle_ask(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
    match state.engine.ask().await {
        Ok(trial) => Ok(Json(serde_json::to_value(&trial).unwrap())),
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e }))),
    }
}

async fn handle_tell(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<TellRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
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
                trial: completed.clone(),
            });

            Ok(Json(serde_json::json!({
                "status": "ok",
                "trial_count": n,
                "trial": completed,
            })))
        }
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e }))),
    }
}

async fn handle_cancel(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<CancelRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
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

async fn handle_trial(
    State(state): State<Arc<ServerState>>,
    AxumPath(trial_id): AxumPath<u64>,
    Query(q): Query<TrialQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let include_infeasible = q.include_infeasible.unwrap_or(true);
    match state
        .engine
        .completed_trial(trial_id, include_infeasible)
        .await
    {
        Some(trial) => Ok(Json(serde_json::to_value(&trial).unwrap_or_default())),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Trial {trial_id} not found"),
            }),
        )),
    }
}

async fn handle_trial_count(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    let count = state.engine.trial_count().await;
    Json(serde_json::json!({ "trial_count": count }))
}

async fn handle_update_objectives(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<UpdateObjectivesRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
    match state.engine.update_objectives(req.objectives).await {
        Ok(()) => {
            let n = state.engine.trial_count().await;
            Ok(Json(serde_json::json!({
                "status": "ok",
                "rescalarized_trials": n,
            })))
        }
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e }))),
    }
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
    headers: HeaderMap,
    Json(req): Json<SaveCheckpointRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
    let path = resolve_checkpoint_path(&state, &req.path)?;
    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        ));
    }

    match state
        .engine
        .save_full_checkpoint(&path, req.description.as_deref())
        .await
    {
        Ok(()) => {
            let n = state.engine.trial_count().await;
            Ok(Json(serde_json::json!({
                "status": "ok",
                "checkpoint_type": "full",
                "path": path.to_string_lossy(),
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

fn build_cors(origins: &[String]) -> CorsLayer {
    let mut cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::PATCH])
        .allow_headers([CONTENT_TYPE, AUTHORIZATION]);

    if !origins.is_empty() {
        let parsed: Vec<HeaderValue> = origins
            .iter()
            .map(|origin| {
                origin
                    .parse()
                    .expect("CORS origins must be valid HTTP header values")
            })
            .collect();
        cors = cors.allow_origin(AllowOrigin::list(parsed));
    }

    cors
}

/// Create the Axum router for the engine server.
pub fn create_router(engine: HolaEngine) -> Router {
    create_router_with_options(engine, ServerOptions::new(8000))
}

/// Create the Axum router for the engine server with explicit server options.
pub fn create_router_with_options(engine: HolaEngine, options: ServerOptions) -> Router {
    let (events_tx, _) = broadcast::channel(256);
    let state = Arc::new(ServerState {
        engine,
        events_tx,
        auth_token: options.auth_token,
        checkpoint_dir: options.checkpoint_dir,
    });

    let cors = build_cors(&options.cors_allowed_origins);

    Router::new()
        .route("/api/ask", post(handle_ask))
        .route("/api/tell", post(handle_tell))
        .route("/api/cancel", post(handle_cancel))
        .route("/api/top_k", get(handle_top_k))
        .route("/api/pareto_front", get(handle_pareto_front))
        .route("/api/trial/{trial_id}", get(handle_trial))
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
    let mut options = ServerOptions::new(8000);
    options.dashboard_dir = Some(dashboard_dir.to_path_buf());
    create_router_with_dashboard_and_options(engine, options)
}

/// Create the Axum router with the dashboard and explicit server options.
pub fn create_router_with_dashboard_and_options(
    engine: HolaEngine,
    options: ServerOptions,
) -> Router {
    let dashboard_dir = options.dashboard_dir.clone();
    let router = create_router_with_options(engine, options);
    match dashboard_dir {
        Some(dir) => router.fallback_service(ServeDir::new(dir)),
        None => router,
    }
}

/// Start the server on the given port. Blocks until the server is shut down.
///
/// If `dashboard_dir` is provided, the dashboard UI is served at `/`.
pub async fn serve(
    engine: HolaEngine,
    port: u16,
    dashboard_dir: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut options = ServerOptions::new(port);
    options.dashboard_dir = dashboard_dir.map(Path::to_path_buf);
    serve_with_options(engine, options).await
}

/// Start the server with explicit host, auth, CORS, and checkpoint options.
pub async fn serve_with_options(
    engine: HolaEngine,
    options: ServerOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let router = match options.dashboard_dir.as_deref() {
        Some(_) => create_router_with_dashboard_and_options(engine, options.clone()),
        None => create_router_with_options(engine, options.clone()),
    };
    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", options.host, options.port)).await?;
    if let Some(dir) = &options.dashboard_dir {
        eprintln!(
            "HOLA server listening on {}:{} (dashboard: {})",
            options.host,
            options.port,
            dir.display()
        );
    } else {
        eprintln!("HOLA server listening on {}:{}", options.host, options.port);
    }
    axum::serve(listener, router).await?;
    Ok(())
}
