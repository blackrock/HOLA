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
//! - `POST /api/heartbeat` - Renew a pending trial lease
//! - `GET /api/top_k` - Get top-k trials by rank
//! - `GET /api/pareto_front` - Get Pareto front trials
//! - `GET /api/trial/{trial_id}` - Get one completed trial with scoring/ranking
//! - `GET /api/trial/{trial_id}/status` - Reconcile a distributed trial lifecycle
//! - `GET /api/trials` - Get all trials with scoring/ranking
//! - `GET /api/trial_count` - Get number of completed trials
//! - `PATCH /api/objectives` - Update objectives mid-run
//! - `GET /api/objectives` - Get current objectives
//! - `GET /api/space` - Get parameter space metadata
//! - `POST /api/checkpoint/save` - Save a full checkpoint
//! - `GET /api/events` - SSE stream of engine events

use crate::hola_engine::{CompletedTrial, HolaEngine, ObjectiveConfig, TrialLifecycle};
use axum::{
    Router,
    extract::{DefaultBodyLimit, Path as AxumPath, Query, Request, State},
    http::{
        HeaderMap, HeaderName, HeaderValue, Method, StatusCode,
        header::{AUTHORIZATION, CONTENT_TYPE, HOST, ORIGIN},
    },
    middleware,
    response::{
        IntoResponse, Json, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, patch, post},
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::error::Error;
use std::future::{Future, IntoFuture};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Semaphore, broadcast, oneshot, watch};
use tokio_stream::wrappers::{BroadcastStream, WatchStream};
use tokio_stream::{Stream, StreamExt};
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::services::ServeDir;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::{DefaultOnResponse, TraceLayer};
use tracing::{Instrument, Level};

// =============================================================================
// Shared state
// =============================================================================

/// Events emitted by the engine for SSE consumers.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum EngineEvent {
    TrialCompleted {
        trial_id: u64,
        /// Scalar score for single-objective dashboards. Multi-objective trials
        /// have no canonical scalar and therefore use `None` (JSON `null`).
        score: Option<f64>,
        trial: CompletedTrial,
    },
    /// Objective topology/weights changed; clients must refresh the complete
    /// ranked snapshot because historical ranks may all have moved.
    ObjectivesChanged,
}

pub struct ServerState {
    pub engine: HolaEngine,
    events_tx: broadcast::Sender<SequencedEngineEvent>,
    /// Retained shutdown state lets long-lived SSE responses close as soon as
    /// graceful shutdown begins, including when they subscribe concurrently
    /// with the signal.
    shutdown: watch::Sender<bool>,
    event_journal: StdMutex<EventJournal>,
    auth_token: Option<String>,
    read_auth_token: Option<String>,
    require_read_auth: bool,
    cors_allowed_origins: Vec<String>,
    checkpoint_dir: PathBuf,
    lease_duration: Duration,
    /// Serialize HTTP tell commit + event publication so SSE order matches
    /// engine commit order even when post-commit ranking/refit work yields.
    tell_lock: Arc<tokio::sync::Mutex<()>>,
    /// Bounds mutation tasks that outlive a timed-out/disconnected request.
    /// Once a tell/objective mutation has started, it is detached so committed
    /// post-processing cannot be cancelled with the HTTP response future.
    mutation_task_slots: Arc<Semaphore>,
    http_requests_total: AtomicU64,
    http_failures_total: AtomicU64,
    http_latency_micros_total: AtomicU64,
    checkpoint_failures_total: AtomicU64,
    checkpoint_file_sequence: AtomicU64,
    events_published_total: AtomicU64,
}

const EVENT_HISTORY_CAPACITY: usize = 256;
const MAX_DETACHED_MUTATIONS: usize = 256;

#[derive(Clone, Debug)]
struct SequencedEngineEvent {
    id: u64,
    event: EngineEvent,
}

struct CloseOnShutdown<S> {
    inner: S,
    shutdown: WatchStream<bool>,
    _shutdown_guard: watch::Sender<bool>,
}

impl<S: Stream + Unpin> Stream for CloseOnShutdown<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match Pin::new(&mut self.shutdown).poll_next(cx) {
                Poll::Ready(Some(true) | None) => return Poll::Ready(None),
                Poll::Ready(Some(false)) => {}
                Poll::Pending => break,
            }
        }
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

#[derive(Debug)]
struct EventJournal {
    next_id: u64,
    history: VecDeque<SequencedEngineEvent>,
}

impl EventJournal {
    fn new() -> Self {
        Self {
            next_id: 1,
            history: VecDeque::with_capacity(EVENT_HISTORY_CAPACITY),
        }
    }

    fn record(&mut self, event: EngineEvent) -> SequencedEngineEvent {
        let sequenced = SequencedEngineEvent {
            id: self.next_id,
            event,
        };
        self.next_id = self.next_id.saturating_add(1);
        if self.history.len() == EVENT_HISTORY_CAPACITY {
            self.history.pop_front();
        }
        self.history.push_back(sequenced.clone());
        sequenced
    }

    fn last_id(&self) -> u64 {
        self.next_id.saturating_sub(1)
    }
}

#[derive(Clone, Debug)]
pub struct ServerOptions {
    pub host: String,
    pub port: u16,
    pub dashboard_dir: Option<PathBuf>,
    pub auth_token: Option<String>,
    /// Optional read-only bearer token. It can access GET/SSE/metrics endpoints
    /// but is never accepted by mutation endpoints.
    pub read_auth_token: Option<String>,
    /// When `true`, read-only endpoints and the SSE stream also require the
    /// bearer token (only has an effect when `auth_token` is set). Defaults to
    /// `true`, so configuring a token protects the entire API by default.
    pub require_read_auth: bool,
    pub checkpoint_dir: PathBuf,
    pub cors_allowed_origins: Vec<String>,
    /// Maximum duration for ordinary API requests. The long-lived SSE stream
    /// is excluded from this timeout.
    pub request_timeout: Duration,
    /// Lifetime of a distributed trial allocation before it must be completed,
    /// cancelled, or renewed through `/api/heartbeat`.
    pub lease_duration: Duration,
    /// Maximum time to drain in-flight requests after a shutdown signal.
    pub shutdown_timeout: Duration,
}

impl ServerOptions {
    pub fn new(port: u16) -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port,
            dashboard_dir: None,
            auth_token: None,
            read_auth_token: None,
            require_read_auth: true,
            checkpoint_dir: PathBuf::from("."),
            cors_allowed_origins: Vec::new(),
            request_timeout: Duration::from_secs(30),
            lease_duration: Duration::from_secs(2 * 60 * 60),
            shutdown_timeout: Duration::from_secs(10),
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
struct HeartbeatRequest {
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
#[serde(deny_unknown_fields)]
struct SaveCheckpointRequest {
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    code: &'static str,
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
            code: "unauthorized",
            error: "Missing or invalid bearer token".to_string(),
        }),
    )
}

fn forbidden_origin() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::FORBIDDEN,
        Json(ErrorResponse {
            code: "origin_forbidden",
            error: "Request origin is not allowed".to_string(),
        }),
    )
}

fn request_id(headers: &HeaderMap) -> &str {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("missing")
}

/// Reject browser requests from origins outside the configured allow-list
/// before a handler can mutate state. CORS response headers alone are not a
/// CSRF defense: browsers still dispatch "simple" cross-origin requests such
/// as an empty `POST /api/ask`, even when they hide the response from script.
///
/// Same-origin requests are recognized by comparing the Origin authority with
/// the request Host header. Non-browser clients normally omit Origin and remain
/// unaffected.
fn authorize_origin(
    state: &ServerState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let Some(origin) = headers.get(ORIGIN).and_then(|value| value.to_str().ok()) else {
        return Ok(());
    };

    if state
        .cors_allowed_origins
        .iter()
        .any(|allowed| allowed == origin)
    {
        return Ok(());
    }

    let same_origin = origin
        .parse::<axum::http::Uri>()
        .ok()
        .and_then(|uri| {
            uri.authority()
                .map(|authority| authority.as_str().to_owned())
        })
        .zip(headers.get(HOST).and_then(|value| value.to_str().ok()))
        .is_some_and(|(origin_authority, host)| origin_authority.eq_ignore_ascii_case(host));

    if same_origin {
        Ok(())
    } else {
        Err(forbidden_origin())
    }
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

fn bearer_matches(auth_token: &Option<String>, headers: &HeaderMap) -> bool {
    let Some(token) = auth_token else {
        return false;
    };
    let expected = format!("Bearer {token}");
    headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|actual| {
            constant_time_eq::constant_time_eq(actual.as_bytes(), expected.as_bytes())
        })
}

/// Mutating endpoints always require the token when one is configured.
fn authorize_mutation(
    state: &ServerState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    authorize_origin(state, headers)?;
    check_bearer(&state.auth_token, headers)
}

/// Read endpoints and the SSE stream require the token only when the server was
/// started with read authentication enabled (`require_read_auth`).
fn authorize_read(
    state: &ServerState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    authorize_origin(state, headers)?;
    if state.require_read_auth {
        if (state.auth_token.is_none() && state.read_auth_token.is_none())
            || bearer_matches(&state.auth_token, headers)
            || bearer_matches(&state.read_auth_token, headers)
        {
            Ok(())
        } else {
            Err(unauthorized())
        }
    } else {
        Ok(())
    }
}

async fn record_http_metrics(
    State(state): State<Arc<ServerState>>,
    request: Request,
    next: middleware::Next,
) -> Response {
    state.http_requests_total.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let mut response = next.run(request).await;
    let status = response.status();
    let has_json_body = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/json"));
    let framework_error = match status {
        StatusCode::REQUEST_TIMEOUT => Some((
            "request_timeout",
            "request exceeded the configured server timeout",
        )),
        StatusCode::PAYLOAD_TOO_LARGE => {
            Some(("payload_too_large", "request body exceeds the server limit"))
        }
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY if !has_json_body => {
            Some(("invalid_request", "request body or parameters are invalid"))
        }
        _ => None,
    };
    if let Some((code, error)) = framework_error {
        let status = if status == StatusCode::UNPROCESSABLE_ENTITY {
            StatusCode::BAD_REQUEST
        } else {
            status
        };
        response = (
            status,
            Json(ErrorResponse {
                code,
                error: error.to_string(),
            }),
        )
            .into_response();
    }
    if response.status().is_client_error() || response.status().is_server_error() {
        state.http_failures_total.fetch_add(1, Ordering::Relaxed);
    }
    let elapsed_micros = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
    state
        .http_latency_micros_total
        .fetch_add(elapsed_micros, Ordering::Relaxed);
    response
}

fn generated_checkpoint_path(state: &ServerState) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let sequence = state
        .checkpoint_file_sequence
        .fetch_add(1, Ordering::Relaxed);
    state
        .checkpoint_dir
        .join(format!("checkpoint_{timestamp:013}_{sequence:06}.json"))
}

async fn handle_ask(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
    let idempotency_key = headers
        .get("idempotency-key")
        .map(|value| value.to_str().map(str::to_owned))
        .transpose()
        .map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    code: "invalid_idempotency_key",
                    error: "Idempotency-Key must be valid ASCII".to_string(),
                }),
            )
        })?;
    if idempotency_key
        .as_ref()
        .is_some_and(|key| key.is_empty() || key.len() > 128)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                code: "invalid_idempotency_key",
                error: "Idempotency-Key must contain 1 to 128 characters".to_string(),
            }),
        ));
    }

    let result = if let Some(key) = idempotency_key {
        state
            .engine
            .ask_idempotent_with_lease(&key, state.lease_duration)
            .await
    } else {
        state.engine.ask_with_lease(state.lease_duration).await
    };
    match result {
        Ok(trial) => Ok(Json(serde_json::to_value(&trial).unwrap_or_default())),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                code: "ask_failed",
                error: e,
            }),
        )),
    }
}

async fn handle_tell(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<TellRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
    let permit = Arc::clone(&state.mutation_task_slots)
        .acquire_owned()
        .await
        .map_err(|error| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    code: "mutation_queue_unavailable",
                    error: error.to_string(),
                }),
            )
        })?;
    let task = tokio::spawn(
        async move {
            let _permit = permit;
            process_tell(state, req).await
        }
        .instrument(tracing::Span::current()),
    );
    task.await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                code: "tell_task_failed",
                error: format!("tell task failed: {error}"),
            }),
        )
    })?
}

async fn process_tell(
    state: Arc<ServerState>,
    req: TellRequest,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let tell_guard = Arc::clone(&state.tell_lock).lock_owned().await;
    let event_state = Arc::clone(&state);
    match state
        .engine
        .tell_with_outcome_on_commit(req.trial_id, req.metrics, move |completed, _| {
            // Only a one-component score vector is a meaningful scalar. A
            // multi-objective vector is deliberately not collapsed by choosing
            // an arbitrary key.
            let score = completed
                .score_vector
                .as_object()
                .filter(|scores| scores.len() == 1)
                .and_then(|m| m.values().next())
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite());
            let event = EngineEvent::TrialCompleted {
                trial_id: completed.trial_id,
                score,
                trial: completed.clone(),
            };
            let mut journal = event_state
                .event_journal
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let sequenced = journal.record(event);
            // Publish while holding the journal lock so subscribers observe
            // the same order as the monotonically assigned event IDs.
            let _ = event_state.events_tx.send(sequenced);
            event_state
                .events_published_total
                .fetch_add(1, Ordering::Relaxed);
            // Commit/event ordering is now fixed. Release serialization before
            // refit/checkpoint awaits so slow maintenance never stalls later
            // tells or objective updates.
            drop(tell_guard);
        })
        .await
    {
        Ok(outcome) => {
            let post_commit_warnings = outcome.post_commit_warnings;
            let completed = outcome.completed;

            Ok(Json(serde_json::json!({
                "status": "ok",
                "trial_count": outcome.trial_count,
                "trial": completed,
                "post_commit_warnings": post_commit_warnings,
            })))
        }
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                code: "tell_failed",
                error: e,
            }),
        )),
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
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                code: "cancel_failed",
                error: e,
            }),
        )),
    }
}

async fn handle_heartbeat(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<HeartbeatRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
    match state
        .engine
        .heartbeat(req.trial_id, state.lease_duration)
        .await
    {
        Ok(expires_at) => Ok(Json(serde_json::json!({
            "status": "ok",
            "trial_id": req.trial_id,
            "lease_expires_at_ms": expires_at,
        }))),
        Err(error) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                code: "heartbeat_failed",
                error,
            }),
        )),
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
                code: "trial_not_found",
                error: format!("Trial {trial_id} not found"),
            }),
        )),
    }
}

async fn handle_trial_status(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    AxumPath(trial_id): AxumPath<u64>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let lifecycle = match state.engine.trial_lifecycle(trial_id).await {
        TrialLifecycle::Completed => "completed",
        TrialLifecycle::Pending => "pending",
        TrialLifecycle::NotPending => "not_pending",
    };
    Ok(Json(serde_json::json!({
        "status": "ok",
        "trial_id": trial_id,
        "state": lifecycle,
    })))
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
    let permit = Arc::clone(&state.mutation_task_slots)
        .acquire_owned()
        .await
        .map_err(|error| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    code: "mutation_queue_unavailable",
                    error: error.to_string(),
                }),
            )
        })?;
    let task = tokio::spawn(
        async move {
            let _permit = permit;
            process_update_objectives(state, req).await
        }
        .instrument(tracing::Span::current()),
    );
    task.await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                code: "objective_update_task_failed",
                error: format!("objective update task failed: {error}"),
            }),
        )
    })?
}

async fn process_update_objectives(
    state: Arc<ServerState>,
    req: UpdateObjectivesRequest,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let tell_guard = Arc::clone(&state.tell_lock).lock_owned().await;
    let event_state = Arc::clone(&state);
    let rescalarized_trials = Arc::new(AtomicUsize::new(0));
    let committed_count = Arc::clone(&rescalarized_trials);
    match state
        .engine
        .update_objectives_on_commit(req.objectives, move |_, retained| {
            committed_count.store(retained, Ordering::Relaxed);
            let mut journal = event_state
                .event_journal
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let sequenced = journal.record(EngineEvent::ObjectivesChanged);
            let _ = event_state.events_tx.send(sequenced);
            event_state
                .events_published_total
                .fetch_add(1, Ordering::Relaxed);
            drop(tell_guard);
        })
        .await
    {
        Ok(()) => {
            let n = rescalarized_trials.load(Ordering::Relaxed);
            Ok(Json(serde_json::json!({
                "status": "ok",
                "rescalarized_trials": n,
            })))
        }
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                code: "invalid_objectives",
                error: e,
            }),
        )),
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

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn handle_ready(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    // Readiness is intentionally unauthenticated for orchestrator probes, so it
    // must not disclose study state such as the completed-trial count.
    let _ = state.engine.retained_trial_count().await;
    Json(serde_json::json!({ "status": "ready" }))
}

async fn handle_metrics(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Result<([(axum::http::HeaderName, &'static str); 1], String), (StatusCode, Json<ErrorResponse>)>
{
    authorize_read(&state, &headers)?;
    let completed = state.engine.trial_count().await;
    let retained = state.engine.retained_trial_count().await;
    let pending = state.engine.pending_count().await;
    let requests = state.http_requests_total.load(Ordering::Relaxed);
    let failures = state.http_failures_total.load(Ordering::Relaxed);
    let latency_seconds =
        state.http_latency_micros_total.load(Ordering::Relaxed) as f64 / 1_000_000.0;
    let checkpoint_failures = state
        .checkpoint_failures_total
        .load(Ordering::Relaxed)
        .saturating_add(state.engine.checkpoint_failure_count());
    let refit_failures = state.engine.refit_failure_count();
    let events = state.events_published_total.load(Ordering::Relaxed);
    let body = format!(
        "# TYPE hola_trials_completed gauge\n\
         hola_trials_completed {completed}\n\
         # TYPE hola_trials_retained gauge\n\
         hola_trials_retained {retained}\n\
         # TYPE hola_trials_pending gauge\n\
         hola_trials_pending {pending}\n\
         # TYPE hola_http_requests_total counter\n\
         hola_http_requests_total {requests}\n\
         # TYPE hola_http_failures_total counter\n\
         hola_http_failures_total {failures}\n\
         # TYPE hola_http_request_duration_seconds_sum counter\n\
         hola_http_request_duration_seconds_sum {latency_seconds}\n\
         # TYPE hola_checkpoint_failures_total counter\n\
         hola_checkpoint_failures_total {checkpoint_failures}\n\
         # TYPE hola_refit_failures_total counter\n\
         hola_refit_failures_total {refit_failures}\n\
         # TYPE hola_events_published_total counter\n\
         hola_events_published_total {events}\n"
    );
    Ok(([(CONTENT_TYPE, "text/plain; version=0.0.4")], body))
}

/// Return a replay watermark for snapshot consumers. A client reads this
/// cursor before fetching `/api/trials`, then opens `/api/events` with it as
/// `Last-Event-ID`; events racing the REST snapshot are replayed harmlessly.
async fn handle_event_cursor(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let last_event_id = state
        .event_journal
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .last_id();
    // Encode as a decimal string so JavaScript retains the full u64 cursor.
    Ok(Json(serde_json::json!({
        "last_event_id": last_event_id.to_string(),
    })))
}

async fn handle_checkpoint_save(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<SaveCheckpointRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
    let path = generated_checkpoint_path(&state);
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            // Log the underlying error (which can contain filesystem paths)
            // only server-side; return a generic message so paths do not leak
            // to clients.
            state
                .checkpoint_failures_total
                .fetch_add(1, Ordering::Relaxed);
            tracing::error!(
                request_id = request_id(&headers),
                error_code = "checkpoint_save_failed",
                error = %e,
                "failed to create checkpoint directory"
            );
            return Err(checkpoint_save_failed());
        }
    }

    match state
        .engine
        .save_full_checkpoint_with_metadata(&path, req.description.as_deref())
        .await
    {
        Ok(saved) => Ok(Json(serde_json::json!({
            "status": "ok",
            "checkpoint_type": "full",
            // Return the resolved checkpoint path so clients can load it
            // back (e.g. `Study.load(path)`); the directory is operator-
            // configured and traversal is blocked by resolve_checkpoint_path.
            "path": path.to_string_lossy(),
            "trials_saved": saved.n_trials,
            "created_at": saved.created_at,
        }))),
        Err(e) => {
            state
                .checkpoint_failures_total
                .fetch_add(1, Ordering::Relaxed);
            tracing::error!(
                request_id = request_id(&headers),
                error_code = "checkpoint_save_failed",
                error = %e,
                "failed to save checkpoint"
            );
            Err(checkpoint_save_failed())
        }
    }
}

fn checkpoint_save_failed() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            code: "checkpoint_save_failed",
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
    let shutdown = state.shutdown.subscribe();
    let requested_id = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let (replay, snapshot_last_id, replay_gap) = {
        let journal = state
            .event_journal
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let snapshot_last_id = journal.last_id();
        let oldest_id = journal.history.front().map(|event| event.id);
        let expired_cursor = requested_id
            .zip(oldest_id)
            .is_some_and(|(requested, oldest)| requested.saturating_add(1) < oldest);
        // Event IDs restart with a new server process. A client reconnecting
        // with a cursor from the previous process must be told to replace its
        // snapshot instead of waiting forever for an ID greater than that
        // stale cursor.
        let future_cursor = requested_id.is_some_and(|requested| requested > snapshot_last_id);
        let replay_gap = expired_cursor || future_cursor;
        let replay = requested_id.map_or_else(Vec::new, |requested| {
            journal
                .history
                .iter()
                .filter(|event| event.id > requested)
                .cloned()
                .collect()
        });
        (replay, snapshot_last_id, replay_gap)
    };

    let mut initial = Vec::with_capacity(replay.len() + usize::from(replay_gap));
    if replay_gap {
        let reason = if requested_id.is_some_and(|requested| requested > snapshot_last_id) {
            "event cursor is ahead of this server"
        } else {
            "event history expired"
        };
        initial.push(Ok(Event::default()
            .event("stream_reset")
            .data(serde_json::json!({ "reason": reason }).to_string())));
    }
    initial.extend(replay.into_iter().map(|event| Ok(sse_event(&event))));

    let live = BroadcastStream::new(rx).filter_map(move |result| match result {
        Ok(event) if event.id > snapshot_last_id => Some(Ok(sse_event(&event))),
        Ok(_) => None,
        Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(missed)) => {
            Some(Ok(Event::default()
                .event("stream_lagged")
                .data(serde_json::json!({ "missed": missed }).to_string())))
        }
    });
    let stream = CloseOnShutdown {
        inner: tokio_stream::iter(initial).chain(live),
        shutdown: WatchStream::new(shutdown),
        _shutdown_guard: state.shutdown.clone(),
    };
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    ))
}

fn sse_event(event: &SequencedEngineEvent) -> Event {
    let data = serde_json::to_string(&event.event).unwrap_or_default();
    Event::default().id(event.id.to_string()).data(data)
}

// =============================================================================
// Router & Server
// =============================================================================

fn build_cors(origins: &[String]) -> Result<CorsLayer, Box<dyn Error>> {
    let mut cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::PATCH])
        .allow_headers([
            CONTENT_TYPE,
            AUTHORIZATION,
            HeaderName::from_static("last-event-id"),
            HeaderName::from_static("idempotency-key"),
        ]);

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
    create_router_with_options_and_shutdown(engine, options).map(|(router, _)| router)
}

fn create_router_with_options_and_shutdown(
    engine: HolaEngine,
    options: ServerOptions,
) -> Result<(Router, watch::Sender<bool>), Box<dyn Error>> {
    if options.lease_duration.is_zero() {
        return Err("lease_duration must be greater than zero".into());
    }
    if options.request_timeout.is_zero() {
        return Err("request_timeout must be greater than zero".into());
    }
    if options.shutdown_timeout.is_zero() {
        return Err("shutdown_timeout must be greater than zero".into());
    }
    if options
        .auth_token
        .as_deref()
        .is_some_and(|token| token.trim().is_empty())
    {
        return Err("auth_token must not be empty or whitespace-only".into());
    }
    if options
        .read_auth_token
        .as_deref()
        .is_some_and(|token| token.trim().is_empty())
    {
        return Err("read_auth_token must not be empty or whitespace-only".into());
    }
    if options.read_auth_token.is_some() && options.auth_token.is_none() {
        return Err("read_auth_token requires an auth_token for mutation endpoints".into());
    }
    std::fs::create_dir_all(&options.checkpoint_dir).map_err(|error| {
        format!(
            "failed to create checkpoint directory '{}': {error}",
            options.checkpoint_dir.display()
        )
    })?;
    let (events_tx, _) = broadcast::channel(256);
    let (shutdown, _) = watch::channel(false);
    let state = Arc::new(ServerState {
        engine,
        events_tx,
        shutdown: shutdown.clone(),
        event_journal: StdMutex::new(EventJournal::new()),
        auth_token: options.auth_token,
        read_auth_token: options.read_auth_token,
        require_read_auth: options.require_read_auth,
        cors_allowed_origins: options.cors_allowed_origins.clone(),
        checkpoint_dir: options.checkpoint_dir,
        lease_duration: options.lease_duration,
        tell_lock: Arc::new(tokio::sync::Mutex::new(())),
        mutation_task_slots: Arc::new(Semaphore::new(MAX_DETACHED_MUTATIONS)),
        http_requests_total: AtomicU64::new(0),
        http_failures_total: AtomicU64::new(0),
        http_latency_micros_total: AtomicU64::new(0),
        checkpoint_failures_total: AtomicU64::new(0),
        checkpoint_file_sequence: AtomicU64::new(0),
        events_published_total: AtomicU64::new(0),
    });
    let metrics_state = Arc::clone(&state);

    let cors = build_cors(&options.cors_allowed_origins)?;

    let api = Router::new()
        .route("/api/ask", post(handle_ask))
        .route("/api/tell", post(handle_tell))
        .route("/api/cancel", post(handle_cancel))
        .route("/api/heartbeat", post(handle_heartbeat))
        .route("/api/top_k", get(handle_top_k))
        .route("/api/pareto_front", get(handle_pareto_front))
        .route("/api/trial/{trial_id}/status", get(handle_trial_status))
        .route("/api/trial/{trial_id}", get(handle_trial))
        .route("/api/trials", get(handle_trials))
        .route("/api/trial_count", get(handle_trial_count))
        .route(
            "/api/objectives",
            patch(handle_update_objectives).get(handle_get_objectives),
        )
        .route("/api/space", get(handle_space))
        .route("/api/metrics", get(handle_metrics))
        .route("/api/event_cursor", get(handle_event_cursor))
        .route("/api/checkpoint/save", post(handle_checkpoint_save))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            options.request_timeout,
        ));

    let request_id_header = HeaderName::from_static("x-request-id");
    let router = Router::new()
        .route("/healthz", get(handle_health))
        .route("/readyz", get(handle_ready))
        .merge(api)
        // SSE is deliberately outside the ordinary request timeout.
        .route("/api/events", get(handle_events))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &axum::http::Request<_>| {
                    let request_id = request
                        .headers()
                        .get("x-request-id")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or("missing");
                    tracing::info_span!(
                        "http_request",
                        method = %request.method(),
                        uri = %request.uri(),
                        request_id = %request_id,
                    )
                })
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .layer(middleware::from_fn_with_state(
            metrics_state,
            record_http_metrics,
        ))
        .layer(PropagateRequestIdLayer::new(request_id_header.clone()))
        .layer(SetRequestIdLayer::new(request_id_header, MakeRequestUuid))
        .layer(cors)
        .with_state(state);
    Ok((router, shutdown))
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
    if !is_loopback_host(&options.host) && options.auth_token.is_none() {
        return Err(format!(
            "an auth token is required when binding to non-loopback host '{}'",
            options.host
        )
        .into());
    }
    let dashboard_dir = options.dashboard_dir.clone();
    let (router, sse_shutdown) = create_router_with_options_and_shutdown(engine, options.clone())?;
    let router = match dashboard_dir {
        Some(dir) => router.fallback_service(ServeDir::new(dir)),
        None => router,
    };
    let listener = tokio::net::TcpListener::bind((options.host.as_str(), options.port)).await?;
    if let Some(dir) = &options.dashboard_dir {
        tracing::info!(
            host = options.host,
            port = options.port,
            dashboard = %dir.display(),
            "HOLA server listening"
        );
    } else {
        tracing::info!(
            host = options.host,
            port = options.port,
            "HOLA server listening"
        );
    }
    let shutdown = async move {
        shutdown_signal().await;
        sse_shutdown.send_replace(true);
    };
    serve_listener_with_shutdown(listener, router, shutdown, options.shutdown_timeout).await?;
    Ok(())
}

async fn serve_listener_with_shutdown<F>(
    listener: tokio::net::TcpListener,
    router: Router,
    shutdown: F,
    drain_timeout: Duration,
) -> std::io::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let (started_tx, started_rx) = oneshot::channel();
    let shutdown = async move {
        shutdown.await;
        let _ = started_tx.send(());
    };
    let server = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .into_future();
    tokio::pin!(server);

    tokio::select! {
        result = &mut server => result,
        _ = started_rx => {
            tracing::info!(
                timeout_seconds = drain_timeout.as_secs_f64(),
                "shutdown signal received; draining in-flight requests"
            );
            match tokio::time::timeout(drain_timeout, &mut server).await {
                Ok(result) => result,
                Err(_) => {
                    tracing::warn!(
                        timeout_seconds = drain_timeout.as_secs_f64(),
                        "graceful shutdown deadline reached; closing remaining connections"
                    );
                    Ok(())
                }
            }
        }
    }
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %error, "failed to install Ctrl-C handler");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => tracing::error!(error = %error, "failed to install SIGTERM handler"),
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hola_engine::{ObjectiveConfig, ParamConfig, StudyConfig};
    use std::collections::BTreeMap;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn shutdown_signal_closes_live_sse_without_waiting_for_drain_deadline() {
        let engine = HolaEngine::from_config(StudyConfig {
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
        })
        .unwrap();
        let checkpoint_dir = tempfile::tempdir().unwrap();
        let mut options = ServerOptions::new(0);
        options.checkpoint_dir = checkpoint_dir.path().to_path_buf();
        let (app, sse_shutdown) = create_router_with_options_and_shutdown(engine, options).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server = tokio::spawn(serve_listener_with_shutdown(
            listener,
            app,
            async move {
                let _ = shutdown_rx.await;
                sse_shutdown.send_replace(true);
            },
            Duration::from_secs(2),
        ));

        let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
        client
            .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), async {
            let mut chunk = [0u8; 1024];
            while !response.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = client.read(&mut chunk).await.unwrap();
                assert!(read > 0, "SSE response closed before sending headers");
                response.extend_from_slice(&chunk[..read]);
            }
        })
        .await
        .expect("SSE response headers should arrive");
        assert!(response.starts_with(b"HTTP/1.1 200 OK"));

        let started = Instant::now();
        shutdown_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .expect("SSE should not consume the two-second fallback drain deadline")
            .expect("server task should join")
            .expect("server should close cleanly after ending SSE streams");
        assert!(started.elapsed() < Duration::from_secs(1));

        let mut trailing = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), client.read_to_end(&mut trailing))
            .await
            .expect("the SSE socket should close promptly")
            .unwrap();
    }

    #[tokio::test]
    async fn bounded_shutdown_closes_a_stuck_request_after_the_drain_deadline() {
        let handler_started = Arc::new(tokio::sync::Notify::new());
        let app = Router::new().route(
            "/hang",
            get({
                let handler_started = Arc::clone(&handler_started);
                move || {
                    let handler_started = Arc::clone(&handler_started);
                    async move {
                        handler_started.notify_one();
                        std::future::pending::<()>().await;
                        StatusCode::OK
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server = tokio::spawn(serve_listener_with_shutdown(
            listener,
            app,
            async move {
                let _ = shutdown_rx.await;
            },
            Duration::from_millis(25),
        ));

        let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
        client
            .write_all(b"GET /hang HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), handler_started.notified())
            .await
            .expect("the hanging request should reach its handler");

        shutdown_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .expect("bounded graceful shutdown should finish")
            .expect("server task should join")
            .expect("server should close cleanly at its drain deadline");
    }

    #[tokio::test]
    async fn cancelled_http_tell_future_does_not_cancel_commit_or_duplicate_event() {
        let engine = HolaEngine::from_config(StudyConfig {
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
        })
        .unwrap();
        let trial = engine.ask().await.unwrap();
        let (events_tx, _) = broadcast::channel(256);
        let checkpoint_dir = tempfile::tempdir().unwrap();
        let state = Arc::new(ServerState {
            engine: engine.clone(),
            events_tx,
            shutdown: watch::channel(false).0,
            event_journal: StdMutex::new(EventJournal::new()),
            auth_token: None,
            read_auth_token: None,
            require_read_auth: true,
            cors_allowed_origins: Vec::new(),
            checkpoint_dir: checkpoint_dir.path().to_path_buf(),
            lease_duration: Duration::from_secs(60),
            tell_lock: Arc::new(tokio::sync::Mutex::new(())),
            mutation_task_slots: Arc::new(Semaphore::new(MAX_DETACHED_MUTATIONS)),
            http_requests_total: AtomicU64::new(0),
            http_failures_total: AtomicU64::new(0),
            http_latency_micros_total: AtomicU64::new(0),
            checkpoint_failures_total: AtomicU64::new(0),
            checkpoint_file_sequence: AtomicU64::new(0),
            events_published_total: AtomicU64::new(0),
        });

        // Hold the serialization lock so the request wrapper can spawn and
        // detach its owned mutation task before that task commits anything.
        let tell_guard = state.tell_lock.lock().await;
        let request_state = Arc::clone(&state);
        let outer = tokio::spawn(handle_tell(
            State(request_state),
            HeaderMap::new(),
            Json(TellRequest {
                trial_id: trial.trial_id,
                metrics: serde_json::json!({"loss": 0.25}),
            }),
        ));
        tokio::time::timeout(Duration::from_secs(1), async {
            while state.mutation_task_slots.available_permits() == MAX_DETACHED_MUTATIONS {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("handler should transfer a bounded permit to the detached task");
        outer.abort();
        assert!(outer.await.unwrap_err().is_cancelled());
        drop(tell_guard);

        tokio::time::timeout(Duration::from_secs(1), async {
            while engine.trial_count().await != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("detached tell should finish after its HTTP future is cancelled");
        assert_eq!(
            state
                .event_journal
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .last_id(),
            1
        );

        let _response = handle_tell(
            State(Arc::clone(&state)),
            HeaderMap::new(),
            Json(TellRequest {
                trial_id: trial.trial_id,
                metrics: serde_json::json!({"loss": 0.25}),
            }),
        )
        .await
        .expect("retry should replay the committed receipt");
        assert_eq!(
            state
                .event_journal
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .last_id(),
            1,
            "receipt replay must not publish a second completion event"
        );
    }
}
