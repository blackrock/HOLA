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
    extract::{DefaultBodyLimit, Path as AxumPath, Query, State},
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
use std::error::Error;
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
        /// Scalar score for dashboards. For a multi-objective study this is the
        /// scalarized value stored under the alphabetically-first key of the
        /// trial's `score_vector` (a `BTreeMap`, hence sorted) iteration order.
        /// It is `None` (serialized as JSON `null`) when no finite score is
        /// available, rather than a sentinel like infinity.
        score: Option<f64>,
        trial: CompletedTrial,
    },
}

pub struct ServerState {
    pub engine: HolaEngine,
    pub events_tx: broadcast::Sender<EngineEvent>,
    auth_token: Option<String>,
    require_read_auth: bool,
    checkpoint_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ServerOptions {
    pub host: String,
    pub port: u16,
    pub dashboard_dir: Option<PathBuf>,
    pub auth_token: Option<String>,
    /// When `true`, read-only endpoints and the SSE stream also require the
    /// bearer token (only has an effect when `auth_token` is set). Defaults to
    /// `false`, which keeps read access open while mutations stay protected.
    pub require_read_auth: bool,
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
            require_read_auth: false,
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

/// Enforce the bearer token when one is configured. Shared by the mutation and
/// read authorization checks.
fn check_bearer(
    auth_token: &Option<String>,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let Some(token) = auth_token else {
        return Ok(());
    };

    let expected = format!("Bearer {token}");
    let provided = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    // Compare in constant time to avoid leaking the token via a timing side
    // channel; `==` would short-circuit on the first differing byte.
    match provided {
        Some(actual)
            if constant_time_eq::constant_time_eq(actual.as_bytes(), expected.as_bytes()) =>
        {
            Ok(())
        }
        _ => Err(unauthorized()),
    }
}

/// Mutating endpoints always require the token when one is configured.
fn authorize_mutation(
    state: &ServerState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    check_bearer(&state.auth_token, headers)
}

/// Read endpoints and the SSE stream require the token only when the server was
/// started with read authentication enabled (`require_read_auth`).
fn authorize_read(
    state: &ServerState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if state.require_read_auth {
        check_bearer(&state.auth_token, headers)
    } else {
        Ok(())
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

    let joined = state.checkpoint_dir.join(path);

    // Defeat symlink escapes that survive the lexical guard: canonicalize both
    // the configured root and the parent directory the file would land in, then
    // require the resolved parent to stay within the root. The parent is
    // created on demand at save time, so canonicalize the nearest existing
    // ancestor and re-append the not-yet-created tail before comparing.
    let canonical_root = state.checkpoint_dir.canonicalize().map_err(|_| {
        invalid_checkpoint_path("Configured checkpoint directory is not accessible")
    })?;
    let parent = joined.parent().unwrap_or(&joined);
    let canonical_parent = canonicalize_existing_ancestor(parent);
    if !canonical_parent.starts_with(&canonical_root) {
        return Err(invalid_checkpoint_path(
            "Checkpoint path must be relative to the configured checkpoint directory",
        ));
    }

    Ok(joined)
}

/// Canonicalize the nearest existing ancestor of `path` and re-append the
/// remaining (not-yet-created) components, resolving any symlinks along the
/// existing portion of the path.
fn canonicalize_existing_ancestor(path: &Path) -> PathBuf {
    let mut existing = path;
    let mut tail: Vec<&std::ffi::OsStr> = Vec::new();
    loop {
        if let Ok(canonical) = existing.canonicalize() {
            let mut resolved = canonical;
            for component in tail.iter().rev() {
                resolved.push(component);
            }
            return resolved;
        }
        match (existing.file_name(), existing.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name);
                existing = parent;
            }
            // No existing ancestor could be canonicalized; fall back to the
            // original path so the start_with check fails closed.
            _ => return path.to_path_buf(),
        }
    }
}

async fn handle_ask(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
    match state.engine.ask().await {
        Ok(trial) => Ok(Json(serde_json::to_value(&trial).unwrap_or_default())),
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

            // Extract the scalar score for the SSE event from the score_vector.
            // `score_vector` deserializes from a `BTreeMap`, so its JSON object
            // keys iterate in sorted order; `values().next()` therefore yields
            // the value of the alphabetically-first key (the multi-objective
            // scalarized score). A missing or non-finite score becomes an
            // explicit `None`, serialized as JSON `null`, instead of a sentinel.
            let score = completed
                .score_vector
                .as_object()
                .and_then(|m| m.values().next())
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite());

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
    headers: HeaderMap,
    Query(q): Query<TopKQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let include_infeasible = q.include_infeasible.unwrap_or(false);
    let trials = state.engine.top_k(q.k, include_infeasible).await;
    Ok(Json(serde_json::to_value(&trials).unwrap_or_default()))
}

async fn handle_pareto_front(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Query(q): Query<ParetoQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let front = q.front.unwrap_or(0);
    let include_infeasible = q.include_infeasible.unwrap_or(false);
    let trials = state.engine.pareto_front(front, include_infeasible).await;
    Ok(Json(serde_json::to_value(&trials).unwrap_or_default()))
}

async fn handle_trials(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Query(q): Query<TrialsQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let sorted_by = q.sorted_by.as_deref().unwrap_or("index");
    let include_infeasible = q.include_infeasible.unwrap_or(true);
    let trials = state.engine.trials(sorted_by, include_infeasible).await;
    Ok(Json(serde_json::to_value(&trials).unwrap_or_default()))
}

async fn handle_trial(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    AxumPath(trial_id): AxumPath<u64>,
    Query(q): Query<TrialQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
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

async fn handle_trial_count(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let count = state.engine.trial_count().await;
    Ok(Json(serde_json::json!({ "trial_count": count })))
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

async fn handle_get_objectives(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let objectives = state.engine.objectives().await;
    Ok(Json(serde_json::json!({ "objectives": objectives })))
}

async fn handle_space(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
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
    Ok(Json(serde_json::json!({ "params": params })))
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
        // Log the underlying error (which can contain filesystem paths) only
        // server-side; return a generic message so paths do not leak to clients.
        // (`eprintln!` matches this module's existing server-side logging; the
        // crate does not depend on `tracing` directly.)
        eprintln!("failed to create checkpoint directory: {e}");
        return Err(checkpoint_save_failed());
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
                // Return the resolved checkpoint path so clients can load it
                // back (e.g. `Study.load(path)`); the directory is operator-
                // configured and traversal is blocked by resolve_checkpoint_path.
                "path": path.to_string_lossy(),
                "trials_saved": n,
            })))
        }
        Err(e) => {
            eprintln!("failed to save checkpoint: {e}");
            Err(checkpoint_save_failed())
        }
    }
}

fn checkpoint_save_failed() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: "failed to save checkpoint".to_string(),
        }),
    )
}

async fn handle_events(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Result<
    Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>,
    (StatusCode, Json<ErrorResponse>),
> {
    authorize_read(&state, &headers)?;
    let rx = state.events_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| match result {
        Ok(event) => {
            let data = serde_json::to_string(&event).unwrap_or_default();
            Some(Ok(Event::default().data(data)))
        }
        Err(_) => None,
    });
    Ok(Sse::new(stream))
}

// =============================================================================
// Router & Server
// =============================================================================

fn build_cors(origins: &[String]) -> Result<CorsLayer, Box<dyn Error>> {
    let mut cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::PATCH])
        .allow_headers([CONTENT_TYPE, AUTHORIZATION]);

    if !origins.is_empty() {
        let parsed: Vec<HeaderValue> = origins
            .iter()
            .map(|origin| {
                origin
                    .parse::<HeaderValue>()
                    .map_err(|e| format!("invalid CORS origin '{origin}': {e}").into())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        cors = cors.allow_origin(AllowOrigin::list(parsed));
    }

    Ok(cors)
}

/// Maximum accepted request body size, in bytes. Caps memory a single client
/// can force the server to buffer for a request.
const MAX_BODY_BYTES: usize = 64 * 1024;

/// Create the Axum router for the engine server.
///
/// Returns an error if any configured CORS origin is not a valid HTTP header
/// value.
pub fn create_router(engine: HolaEngine) -> Result<Router, Box<dyn Error>> {
    create_router_with_options(engine, ServerOptions::new(8000))
}

/// Create the Axum router for the engine server with explicit server options.
///
/// Returns an error if any configured CORS origin is not a valid HTTP header
/// value.
pub fn create_router_with_options(
    engine: HolaEngine,
    options: ServerOptions,
) -> Result<Router, Box<dyn Error>> {
    let (events_tx, _) = broadcast::channel(256);
    let state = Arc::new(ServerState {
        engine,
        events_tx,
        auth_token: options.auth_token,
        require_read_auth: options.require_read_auth,
        checkpoint_dir: options.checkpoint_dir,
    });

    let cors = build_cors(&options.cors_allowed_origins)?;

    Ok(Router::new()
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
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(cors)
        .with_state(state))
}

/// Create the Axum router with the dashboard served from a local directory.
///
/// API routes under `/api/*` take priority; all other paths fall through to
/// serve static files from `dashboard_dir`.
pub fn create_router_with_dashboard(
    engine: HolaEngine,
    dashboard_dir: &Path,
) -> Result<Router, Box<dyn Error>> {
    let mut options = ServerOptions::new(8000);
    options.dashboard_dir = Some(dashboard_dir.to_path_buf());
    create_router_with_dashboard_and_options(engine, options)
}

/// Create the Axum router with the dashboard and explicit server options.
///
/// Returns an error if any configured CORS origin is not a valid HTTP header
/// value.
pub fn create_router_with_dashboard_and_options(
    engine: HolaEngine,
    options: ServerOptions,
) -> Result<Router, Box<dyn Error>> {
    let dashboard_dir = options.dashboard_dir.clone();
    let router = create_router_with_options(engine, options)?;
    Ok(match dashboard_dir {
        Some(dir) => router.fallback_service(ServeDir::new(dir)),
        None => router,
    })
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
        Some(_) => create_router_with_dashboard_and_options(engine, options.clone())?,
        None => create_router_with_options(engine, options.clone())?,
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
