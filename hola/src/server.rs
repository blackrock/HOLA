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
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use opt_engine::persistence::create_private_dir_all;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::error::Error;
use std::future::Future;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, broadcast, oneshot, watch};
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
    /// Bounds concurrent `/api/events` subscribers. Each one holds a
    /// connection, a broadcast receiver, and a keep-alive timer indefinitely.
    event_stream_slots: Arc<Semaphore>,
    /// Bounds ranked reads running on the blocking pool. The permit travels
    /// into the blocking task, so a request that times out or disconnects
    /// keeps its slot until the work it started has actually finished.
    ranked_read_slots: Arc<Semaphore>,
    /// At most one checkpoint save runs at a time. Like the read slots, the
    /// permit is owned by the detached save task.
    checkpoint_save_slot: Arc<Semaphore>,
    max_saved_checkpoints: usize,
    http_requests_total: AtomicU64,
    http_failures_total: AtomicU64,
    http_latency_micros_total: AtomicU64,
    checkpoint_failures_total: AtomicU64,
    checkpoint_file_sequence: AtomicU64,
    events_published_total: AtomicU64,
}

const EVENT_HISTORY_CAPACITY: usize = 256;
const MAX_DETACHED_MUTATIONS: usize = 256;
/// Concurrent `/api/events` streams. Further subscribers receive 503 until one
/// disconnects.
const MAX_EVENT_STREAMS: usize = 64;
/// Ranked reads (`top_k`, `trials`, `pareto_front`, one trial) allowed on the
/// blocking pool at once. Further reads wait for a slot within the request
/// timeout.
const MAX_CONCURRENT_RANKED_READS: usize = 4;
/// Open connections served at once. Further connections are closed as soon as
/// they are accepted, so a flood cannot exhaust file descriptors or tasks.
const MAX_CONNECTIONS: usize = 1024;
/// Time a client may take to deliver a complete request head. hyper restarts
/// this timer while a keep-alive connection waits for its next request, so it
/// also bounds idle connections. It exceeds common client pool idle timeouts
/// (90 s) so well-behaved clients close first.
const REQUEST_HEAD_TIMEOUT: Duration = Duration::from_secs(120);
/// Bytes buffered per connection while reading a request head.
const MAX_REQUEST_HEAD_BYTES: usize = 64 * 1024;
/// Default for [`ServerOptions::max_saved_checkpoints`].
pub const DEFAULT_MAX_SAVED_CHECKPOINTS: usize = 32;

#[derive(Clone, Debug)]
struct SequencedEngineEvent {
    id: u64,
    event: EngineEvent,
}

struct CloseOnShutdown<S> {
    inner: S,
    shutdown: WatchStream<bool>,
    _shutdown_guard: watch::Sender<bool>,
    /// Returned to `event_stream_slots` when the response body is dropped.
    _stream_permit: OwnedSemaphorePermit,
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
    /// Number of files `POST /api/checkpoint/save` may leave in
    /// `checkpoint_dir`. Once reached, further saves are refused until an
    /// operator removes or archives older files; the server never deletes a
    /// saved checkpoint itself.
    pub max_saved_checkpoints: usize,
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
            max_saved_checkpoints: DEFAULT_MAX_SAVED_CHECKPOINTS,
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

fn forbidden_host() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::FORBIDDEN,
        Json(ErrorResponse {
            code: "host_forbidden",
            error: "This server has no auth token, so it only answers requests addressed to \
                    localhost or a loopback IP address. Configure an auth token to serve it \
                    under another host name."
                .to_string(),
        }),
    )
}

/// Host part of an HTTP authority: `host`, `host:port`, `[v6]`, or `[v6]:port`.
fn authority_host(authority: &str) -> Option<&str> {
    let is_port = |port: &str| !port.is_empty() && port.bytes().all(|byte| byte.is_ascii_digit());
    if let Some(rest) = authority.strip_prefix('[') {
        let (host, tail) = rest.split_once(']')?;
        return (tail.is_empty() || tail.strip_prefix(':').is_some_and(is_port)).then_some(host);
    }
    match authority.rsplit_once(':') {
        Some((host, port)) => (is_port(port) && !host.contains(':')).then_some(host),
        None => Some(authority),
    }
}

/// Whether an authority names this machine's loopback interface.
fn is_loopback_authority(authority: &str) -> bool {
    authority_host(authority).is_some_and(|host| {
        let host = host.strip_suffix('.').unwrap_or(host);
        host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    })
}

/// Refuse requests addressed to an unexpected host name when no token protects
/// the API.
///
/// A server without a token can only be bound to loopback, and relies on the
/// browser's same-origin policy to keep web pages away from it. DNS rebinding
/// defeats that: a page served from `attacker.example` re-resolves its own
/// name to 127.0.0.1 and then talks to this server as a same-origin peer. The
/// browser still sends the attacker's name as `Host` (and as `Origin`, which
/// therefore matches), so the name the request was addressed to is the one
/// thing that tells the two cases apart. Clients that send no `Host` at all
/// are not browsers and are unaffected.
///
/// With a token configured the check is unnecessary (a rebinding page has no
/// token) and would break reverse proxies that forward a public host name.
async fn reject_unexpected_host(
    State(state): State<Arc<ServerState>>,
    request: Request,
    next: middleware::Next,
) -> Response {
    if state.auth_token.is_none() {
        let headers_allowed = request
            .headers()
            .get_all(HOST)
            .iter()
            .all(|value| value.to_str().is_ok_and(is_loopback_authority));
        let target_allowed = request
            .uri()
            .authority()
            .is_none_or(|authority| is_loopback_authority(authority.as_str()));
        if !(headers_allowed && target_allowed) {
            return forbidden_host().into_response();
        }
    }
    next.run(request).await
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
            "lease_duration_ms": state.lease_duration.as_millis().min(u128::from(u64::MAX)).max(1),
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

/// Run a ranked read on the blocking pool under the read concurrency limit.
///
/// Ranking, cloning, and serializing the retained history is CPU-bound work
/// proportional to its size, and a request timeout cannot interrupt it. Doing
/// it on the async workers would let a few such requests stall every other
/// request, including tells and health probes. The slot is released only when
/// the blocking work ends, not when the request future is dropped.
async fn ranked_read<T, F>(
    state: &Arc<ServerState>,
    read: F,
) -> Result<T, (StatusCode, Json<ErrorResponse>)>
where
    F: FnOnce(&HolaEngine) -> T + Send + 'static,
    T: Send + 'static,
{
    let permit = Arc::clone(&state.ranked_read_slots)
        .acquire_owned()
        .await
        .map_err(|error| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    code: "read_queue_unavailable",
                    error: error.to_string(),
                }),
            )
        })?;
    let engine = state.engine.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        read(&engine)
    })
    .await
    .map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                code: "read_task_failed",
                error: format!("read task failed: {error}"),
            }),
        )
    })
}

async fn handle_top_k(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Query(q): Query<TopKQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let include_infeasible = q.include_infeasible.unwrap_or(false);
    let trials = ranked_read(&state, move |engine| {
        serde_json::to_value(engine.top_k_blocking(q.k, include_infeasible)).unwrap_or_default()
    })
    .await?;
    Ok(Json(trials))
}

async fn handle_pareto_front(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Query(q): Query<ParetoQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let front = q.front.unwrap_or(0);
    let include_infeasible = q.include_infeasible.unwrap_or(false);
    let trials = ranked_read(&state, move |engine| {
        serde_json::to_value(engine.pareto_front_blocking(front, include_infeasible))
            .unwrap_or_default()
    })
    .await?;
    Ok(Json(trials))
}

async fn handle_trials(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Query(q): Query<TrialsQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let sorted_by = q.sorted_by.unwrap_or_else(|| "index".to_string());
    let include_infeasible = q.include_infeasible.unwrap_or(true);
    let trials = ranked_read(&state, move |engine| {
        serde_json::to_value(engine.trials_blocking(&sorted_by, include_infeasible))
            .unwrap_or_default()
    })
    .await?;
    Ok(Json(trials))
}

async fn handle_trial(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    AxumPath(trial_id): AxumPath<u64>,
    Query(q): Query<TrialQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_read(&state, &headers)?;
    let include_infeasible = q.include_infeasible.unwrap_or(true);
    let trial = ranked_read(&state, move |engine| {
        engine
            .completed_trial_blocking(trial_id, include_infeasible)
            .map(|trial| serde_json::to_value(&trial).unwrap_or_default())
    })
    .await?;
    match trial {
        Some(trial) => Ok(Json(trial)),
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

/// Whether `name` is a file this server generated for `/api/checkpoint/save`
/// (`checkpoint_<milliseconds>_<sequence>.json`). Automatic checkpoints use a
/// single numeric part and are rotated by the engine instead.
fn is_saved_checkpoint_name(name: &str) -> bool {
    let is_number = |part: &str| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit());
    name.strip_prefix("checkpoint_")
        .and_then(|rest| rest.strip_suffix(".json"))
        .and_then(|stem| stem.split_once('_'))
        .is_some_and(|(timestamp, sequence)| is_number(timestamp) && is_number(sequence))
}

/// Create the checkpoint directory if needed and count the saved checkpoints
/// it already holds.
fn count_saved_checkpoints(directory: &Path) -> std::io::Result<usize> {
    create_private_dir_all(directory)?;
    let mut count = 0;
    for entry in std::fs::read_dir(nonempty_path(directory))? {
        if entry?
            .file_name()
            .to_str()
            .is_some_and(is_saved_checkpoint_name)
        {
            count += 1;
        }
    }
    Ok(count)
}

async fn handle_checkpoint_save(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<SaveCheckpointRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    authorize_mutation(&state, &headers)?;
    // A save clones the full study state and writes it on the blocking pool,
    // where neither the request timeout nor a disconnect can stop it. Allow
    // one at a time and let the detached task own the slot, so repeated or
    // abandoned requests cannot pile up clones, blocking threads, and files.
    let permit = Arc::clone(&state.checkpoint_save_slot)
        .try_acquire_owned()
        .map_err(|_| {
            (
                StatusCode::CONFLICT,
                Json(ErrorResponse {
                    code: "checkpoint_in_progress",
                    error: "another checkpoint save is still running; retry when it completes"
                        .to_string(),
                }),
            )
        })?;
    let request_id = request_id(&headers).to_owned();
    let task = tokio::spawn(
        async move {
            let _permit = permit;
            save_checkpoint(state, req.description, request_id).await
        }
        .instrument(tracing::Span::current()),
    );
    task.await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                code: "checkpoint_task_failed",
                error: format!("checkpoint task failed: {error}"),
            }),
        )
    })?
}

async fn save_checkpoint(
    state: Arc<ServerState>,
    description: Option<String>,
    request_id: String,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let fail = |error: &dyn std::fmt::Display, message: &'static str| {
        // Log the underlying error (which can contain filesystem paths) only
        // server-side; return a generic message so paths do not leak to
        // clients.
        state
            .checkpoint_failures_total
            .fetch_add(1, Ordering::Relaxed);
        tracing::error!(
            request_id = request_id,
            error_code = "checkpoint_save_failed",
            error = %error,
            "{message}"
        );
        checkpoint_save_failed()
    };

    let directory = state.checkpoint_dir.clone();
    let existing = match tokio::task::spawn_blocking(move || count_saved_checkpoints(&directory))
        .await
        .map_err(std::io::Error::other)
        .and_then(|result| result)
    {
        Ok(existing) => existing,
        Err(error) => return Err(fail(&error, "failed to prepare checkpoint directory")),
    };
    let limit = state.max_saved_checkpoints;
    if existing >= limit {
        return Err((
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                code: "checkpoint_quota_exceeded",
                error: format!(
                    "the checkpoint directory already holds {existing} saved checkpoints \
                     (limit {limit}); remove or archive older files, then retry"
                ),
            }),
        ));
    }

    let path = generated_checkpoint_path(&state);
    match state
        .engine
        .save_full_checkpoint_with_metadata(&path, description.as_deref())
        .await
    {
        Ok(saved) => {
            // The request that asked for this file may already have timed
            // out; the log is then the only record of where it went.
            tracing::info!(
                request_id = request_id,
                path = %path.display(),
                trials_saved = saved.n_trials,
                "checkpoint saved"
            );
            Ok(Json(serde_json::json!({
                "status": "ok",
                "checkpoint_type": "full",
                // Return the resolved checkpoint path so clients can load it
                // back (e.g. `Study.load(path)`); the directory is operator-
                // configured and the file name is generated by the server.
                "path": path.to_string_lossy(),
                "trials_saved": saved.n_trials,
                "created_at": saved.created_at,
            })))
        }
        Err(error) => Err(fail(&error, "failed to save checkpoint")),
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
    let stream_permit = Arc::clone(&state.event_stream_slots)
        .try_acquire_owned()
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    code: "too_many_event_streams",
                    error: format!(
                        "the server already has {MAX_EVENT_STREAMS} open event streams; \
                         close one and retry"
                    ),
                }),
            )
        })?;
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
        _stream_permit: stream_permit,
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
    if options.max_saved_checkpoints == 0 {
        return Err("max_saved_checkpoints must be at least 1".into());
    }
    create_private_dir_all(&options.checkpoint_dir).map_err(|error| {
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
        event_stream_slots: Arc::new(Semaphore::new(MAX_EVENT_STREAMS)),
        ranked_read_slots: Arc::new(Semaphore::new(MAX_CONCURRENT_RANKED_READS)),
        checkpoint_save_slot: Arc::new(Semaphore::new(1)),
        max_saved_checkpoints: options.max_saved_checkpoints,
        http_requests_total: AtomicU64::new(0),
        http_failures_total: AtomicU64::new(0),
        http_latency_micros_total: AtomicU64::new(0),
        checkpoint_failures_total: AtomicU64::new(0),
        checkpoint_file_sequence: AtomicU64::new(0),
        events_published_total: AtomicU64::new(0),
    });
    let metrics_state = Arc::clone(&state);
    let host_state = Arc::clone(&state);

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
        .layer(middleware::from_fn_with_state(
            host_state,
            reject_unexpected_host,
        ))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &axum::http::Request<_>| {
                    let request_id = request
                        .headers()
                        .get("x-request-id")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or("missing");
                    // Log the path only. A query string can carry a
                    // credential that a client put there by mistake.
                    tracing::info_span!(
                        "http_request",
                        method = %request.method(),
                        uri = %request.uri().path(),
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
    let (router, _) = create_router_with_dashboard_and_shutdown(engine, &options)?;
    Ok(router)
}

fn create_router_with_dashboard_and_shutdown(
    engine: HolaEngine,
    options: &ServerOptions,
) -> Result<(Router, watch::Sender<bool>), Box<dyn Error>> {
    if let Some(dashboard_dir) = &options.dashboard_dir {
        // Checked before the router creates the checkpoint directory, so a
        // rejected configuration leaves nothing behind in the dashboard.
        let checkpoint_dirs = std::iter::once(options.checkpoint_dir.as_path())
            .chain(engine.auto_checkpoint_directory());
        ensure_dashboard_excludes_checkpoints(dashboard_dir, checkpoint_dirs)?;
    }
    let (router, shutdown) = create_router_with_options_and_shutdown(engine, options.clone())?;
    let router = match &options.dashboard_dir {
        Some(dashboard_dir) => router.fallback_service(ServeDir::new(dashboard_dir)),
        None => router,
    };
    Ok((router, shutdown))
}

/// The current directory for an empty path, as the filesystem treats it.
fn nonempty_path(path: &Path) -> &Path {
    if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    }
}

/// Resolve symlinks in the part of `path` that exists and keep the rest.
///
/// A checkpoint directory that has not been created yet still has a known
/// final location once its existing ancestors are resolved.
fn resolve_existing_prefix(path: &Path) -> PathBuf {
    let mut missing = Vec::new();
    let mut existing = path;
    loop {
        if let Ok(resolved) = existing.canonicalize() {
            return missing
                .iter()
                .rev()
                .fold(resolved, |path, part| path.join(part));
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                missing.push(name);
                existing = parent;
            }
            _ => return path.to_path_buf(),
        }
    }
}

/// Refuse a dashboard directory that contains a checkpoint directory.
///
/// Every file below the dashboard directory is served to anyone who can reach
/// the server, without the read token, because the UI must load before a token
/// can be entered. A checkpoint directory at or below it would publish full
/// study state past read authentication.
fn ensure_dashboard_excludes_checkpoints<'a>(
    dashboard_dir: &Path,
    checkpoint_dirs: impl Iterator<Item = &'a Path>,
) -> Result<(), Box<dyn Error>> {
    let dashboard = nonempty_path(dashboard_dir)
        .canonicalize()
        .map_err(|error| {
            format!(
                "dashboard directory '{}' is not accessible: {error}",
                dashboard_dir.display()
            )
        })?;
    let dashboard_as_written = std::path::absolute(nonempty_path(dashboard_dir))?;
    for checkpoint_dir in checkpoint_dirs {
        // Compare both the written and the resolved locations: a checkpoint
        // path written beneath the dashboard directory is reachable through
        // it even when a symlink makes it resolve elsewhere.
        let as_written = std::path::absolute(nonempty_path(checkpoint_dir))?;
        let resolved = resolve_existing_prefix(&as_written);
        if resolved.starts_with(&dashboard) || as_written.starts_with(&dashboard_as_written) {
            return Err(format!(
                "the dashboard directory '{}' contains the checkpoint directory '{}'. Files \
                 under the dashboard directory are served without authentication, so \
                 checkpoints must be kept outside it.",
                dashboard_dir.display(),
                checkpoint_dir.display()
            )
            .into());
        }
    }
    Ok(())
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

/// A background server whose listener has already been bound.
///
/// Dropping the handle requests graceful shutdown. Call [`Self::stop`] to wait
/// until requests have drained and the listener can be reused.
pub struct ServerHandle {
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
}

impl ServerHandle {
    /// Stop this server and wait for bounded graceful shutdown.
    pub async fn stop(mut self) -> std::io::Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        match self.task.take() {
            Some(task) => task.await.map_err(std::io::Error::other)?,
            None => Ok(()),
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

/// Bind a server and return its background lifecycle handle.
///
/// Configuration and bind failures are returned before success is reported.
/// The caller must keep the handle alive for as long as it wants to serve.
pub async fn start_server_with_options(
    engine: HolaEngine,
    options: ServerOptions,
) -> Result<ServerHandle, Box<dyn std::error::Error>> {
    let (listener, router, sse_shutdown) = prepare_server(engine, &options).await?;
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let shutdown = async move {
        tokio::select! {
            _ = shutdown_signal() => {},
            _ = shutdown_rx => {},
        }
        sse_shutdown.send_replace(true);
    };
    let task = tokio::spawn(serve_listener_with_shutdown(
        listener,
        router,
        shutdown,
        options.shutdown_timeout,
    ));
    Ok(ServerHandle {
        shutdown: Some(shutdown_tx),
        task: Some(task),
    })
}

/// Start the server with explicit host, auth, CORS, and checkpoint options.
pub async fn serve_with_options(
    engine: HolaEngine,
    options: ServerOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let (listener, router, sse_shutdown) = prepare_server(engine, &options).await?;
    let shutdown = async move {
        shutdown_signal().await;
        sse_shutdown.send_replace(true);
    };
    serve_listener_with_shutdown(listener, router, shutdown, options.shutdown_timeout).await?;
    Ok(())
}

async fn prepare_server(
    engine: HolaEngine,
    options: &ServerOptions,
) -> Result<(tokio::net::TcpListener, Router, watch::Sender<bool>), Box<dyn std::error::Error>> {
    if !is_loopback_host(&options.host) && options.auth_token.is_none() {
        return Err(format!(
            "an auth token is required when binding to non-loopback host '{}'",
            options.host
        )
        .into());
    }
    let (router, sse_shutdown) = create_router_with_dashboard_and_shutdown(engine, options)?;
    let listener = tokio::net::TcpListener::bind((options.host.as_str(), options.port)).await?;
    tracing::info!(
        host = options.host,
        port = options.port,
        dashboard = ?options.dashboard_dir,
        "HOLA server listening"
    );
    Ok((listener, router, sse_shutdown))
}

/// Accept and serve connections until `shutdown` resolves, then drain.
///
/// This replaces `axum::serve`, which configures no timer on its connections:
/// a client could then open a socket, never finish a request head, and hold
/// the connection forever, and nothing limited how many it could open.
async fn serve_listener_with_shutdown<F>(
    listener: tokio::net::TcpListener,
    router: Router,
    shutdown: F,
    drain_timeout: Duration,
) -> std::io::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    serve_listener_with_limits(
        listener,
        router,
        shutdown,
        drain_timeout,
        ConnectionLimits {
            head_timeout: REQUEST_HEAD_TIMEOUT,
            max_connections: MAX_CONNECTIONS,
        },
    )
    .await
}

#[derive(Clone, Copy)]
struct ConnectionLimits {
    head_timeout: Duration,
    max_connections: usize,
}

async fn serve_listener_with_limits<F>(
    listener: tokio::net::TcpListener,
    router: Router,
    shutdown: F,
    drain_timeout: Duration,
    limits: ConnectionLimits,
) -> std::io::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let mut http = hyper::server::conn::http1::Builder::new();
    http.timer(TokioTimer::new())
        .header_read_timeout(limits.head_timeout)
        .max_buf_size(MAX_REQUEST_HEAD_BYTES);
    let graceful = GracefulShutdown::new();
    let connection_slots = Arc::new(Semaphore::new(limits.max_connections));
    let mut connections = tokio::task::JoinSet::new();
    let mut refused: u64 = 0;
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            () = &mut shutdown => break,
            // Reap finished connection tasks so the set tracks only live ones.
            Some(_) = connections.join_next(), if !connections.is_empty() => {}
            accepted = listener.accept() => {
                let stream = match accepted {
                    Ok((stream, _peer)) => stream,
                    Err(error) => {
                        if !is_connection_error(&error) {
                            // Typically file-descriptor exhaustion. Back off
                            // instead of spinning on the same failure.
                            tracing::error!(error = %error, "failed to accept a connection");
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                        continue;
                    }
                };
                let Ok(permit) = Arc::clone(&connection_slots).try_acquire_owned() else {
                    // Closing immediately keeps the listener responsive and
                    // the process within its limits; idle connections free
                    // their slots when the request-head timer expires.
                    refused += 1;
                    if refused.is_power_of_two() {
                        tracing::warn!(
                            limit = limits.max_connections,
                            refused,
                            "connection limit reached; closing new connections"
                        );
                    }
                    continue;
                };
                let service = TowerToHyperService::new(router.clone());
                let connection = graceful.watch(http.serve_connection(TokioIo::new(stream), service));
                connections.spawn(async move {
                    let _permit = permit;
                    if let Err(error) = connection.await {
                        tracing::debug!(error = %error, "connection closed with an error");
                    }
                });
            }
        }
    }

    // Stop accepting, then let in-flight requests finish.
    drop(listener);
    tracing::info!(
        timeout_seconds = drain_timeout.as_secs_f64(),
        "shutdown signal received; draining in-flight requests"
    );
    if tokio::time::timeout(drain_timeout, graceful.shutdown())
        .await
        .is_err()
    {
        tracing::warn!(
            timeout_seconds = drain_timeout.as_secs_f64(),
            "graceful shutdown deadline reached; closing remaining connections"
        );
    }
    // Dropping the set aborts whatever is still running, which closes its
    // socket; nothing outlives this function.
    connections.shutdown().await;
    Ok(())
}

/// Accept errors that concern one peer rather than the listener.
fn is_connection_error(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::ConnectionReset
    )
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

    fn test_engine() -> HolaEngine {
        HolaEngine::from_config(StudyConfig {
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
        .unwrap()
    }

    fn test_state(engine: HolaEngine, checkpoint_dir: &Path) -> Arc<ServerState> {
        Arc::new(ServerState {
            engine,
            events_tx: broadcast::channel(256).0,
            shutdown: watch::channel(false).0,
            event_journal: StdMutex::new(EventJournal::new()),
            auth_token: None,
            read_auth_token: None,
            require_read_auth: true,
            cors_allowed_origins: Vec::new(),
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            lease_duration: Duration::from_secs(60),
            tell_lock: Arc::new(tokio::sync::Mutex::new(())),
            mutation_task_slots: Arc::new(Semaphore::new(MAX_DETACHED_MUTATIONS)),
            event_stream_slots: Arc::new(Semaphore::new(MAX_EVENT_STREAMS)),
            ranked_read_slots: Arc::new(Semaphore::new(MAX_CONCURRENT_RANKED_READS)),
            checkpoint_save_slot: Arc::new(Semaphore::new(1)),
            max_saved_checkpoints: DEFAULT_MAX_SAVED_CHECKPOINTS,
            http_requests_total: AtomicU64::new(0),
            http_failures_total: AtomicU64::new(0),
            http_latency_micros_total: AtomicU64::new(0),
            checkpoint_failures_total: AtomicU64::new(0),
            checkpoint_file_sequence: AtomicU64::new(0),
            events_published_total: AtomicU64::new(0),
        })
    }

    /// Send one in-process request and return its status and JSON body.
    async fn send(
        app: &Router,
        method: &str,
        uri: &str,
        headers: &[(&str, &str)],
    ) -> (StatusCode, serde_json::Value) {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let mut request = axum::http::Request::builder().method(method).uri(uri);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let body = if method == "GET" {
            axum::body::Body::empty()
        } else {
            request = request.header("content-type", "application/json");
            axum::body::Body::from("{}")
        };
        let response = app
            .clone()
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or_default())
    }

    #[test]
    fn loopback_authorities_are_recognized_exactly() {
        for authority in [
            "localhost",
            "localhost:8000",
            "LOCALHOST:8000",
            "localhost.:8000",
            "127.0.0.1",
            "127.0.0.1:8000",
            "127.8.9.10:1",
            "[::1]",
            "[::1]:8000",
        ] {
            assert!(is_loopback_authority(authority), "{authority}");
        }
        for authority in [
            "",
            "attacker.example",
            "attacker.example:8000",
            "localhost.attacker.example",
            "127.0.0.1.attacker.example:8000",
            "notlocalhost",
            "192.168.1.5:8000",
            "0.0.0.0:8000",
            "[::ffff:127.0.0.1]:8000",
            "::1",
            "localhost:",
            "localhost:80a",
            "[::1]x",
            "user@localhost",
        ] {
            assert!(!is_loopback_authority(authority), "{authority}");
        }
    }

    #[tokio::test]
    async fn tokenless_server_rejects_rebound_hosts_before_reads_or_mutations() {
        let engine = test_engine();
        let checkpoint_dir = tempfile::tempdir().unwrap();
        let mut options = ServerOptions::new(0);
        options.checkpoint_dir = checkpoint_dir.path().to_path_buf();
        let (app, _shutdown) =
            create_router_with_options_and_shutdown(engine.clone(), options).unwrap();

        // After DNS rebinding the browser addresses the loopback server by the
        // attacker's name, so Host and Origin agree with each other.
        let rebound = [
            ("host", "attacker.example:8000"),
            ("origin", "http://attacker.example:8000"),
        ];
        for (method, uri) in [
            ("POST", "/api/ask"),
            ("POST", "/api/checkpoint/save"),
            ("GET", "/api/trials"),
            ("GET", "/api/space"),
            ("GET", "/api/events"),
            ("GET", "/api/metrics"),
        ] {
            let (status, body) = send(&app, method, uri, &rebound).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{method} {uri}");
            assert_eq!(body["code"], "host_forbidden", "{method} {uri}");
        }
        // Same-origin GETs carry no Origin header at all.
        let (status, body) =
            send(&app, "GET", "/api/trials", &[("host", "attacker.example")]).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(body["code"], "host_forbidden");
        assert_eq!(engine.pending_count().await, 0, "no trial may be allocated");
        assert!(
            std::fs::read_dir(checkpoint_dir.path())
                .unwrap()
                .next()
                .is_none(),
            "no checkpoint may be written"
        );

        for host in ["localhost:8000", "127.0.0.1:8000", "[::1]:8000"] {
            let origin = format!("http://{host}");
            let headers = [("host", host), ("origin", origin.as_str())];
            let (status, _) = send(&app, "GET", "/api/trial_count", &headers).await;
            assert_eq!(status, StatusCode::OK, "{host}");
        }
        // Non-browser clients that send no Host are unaffected.
        let (status, _) = send(&app, "POST", "/api/ask", &[]).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn token_protected_server_accepts_a_proxied_host_name() {
        let checkpoint_dir = tempfile::tempdir().unwrap();
        let mut options = ServerOptions::new(0);
        options.checkpoint_dir = checkpoint_dir.path().to_path_buf();
        options.auth_token = Some("secret".to_string());
        let (app, _shutdown) =
            create_router_with_options_and_shutdown(test_engine(), options).unwrap();

        let proxied = [
            ("host", "optimizer.example.com"),
            ("authorization", "Bearer secret"),
        ];
        let (status, _) = send(&app, "POST", "/api/ask", &proxied).await;
        assert_eq!(status, StatusCode::OK);
        let (status, body) = send(
            &app,
            "POST",
            "/api/ask",
            &[("host", "optimizer.example.com")],
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["code"], "unauthorized");
    }

    #[tokio::test]
    async fn event_streams_are_limited_and_release_their_slot_on_disconnect() {
        use tower::ServiceExt;

        let checkpoint_dir = tempfile::tempdir().unwrap();
        let mut options = ServerOptions::new(0);
        options.checkpoint_dir = checkpoint_dir.path().to_path_buf();
        let (app, _shutdown) =
            create_router_with_options_and_shutdown(test_engine(), options).unwrap();
        let open_stream = || {
            let request = axum::http::Request::builder()
                .uri("/api/events")
                .body(axum::body::Body::empty())
                .unwrap();
            app.clone().oneshot(request)
        };

        let mut streams = Vec::new();
        for _ in 0..MAX_EVENT_STREAMS {
            let response = open_stream().await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            streams.push(response);
        }
        let (status, body) = send(&app, "GET", "/api/events", &[]).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["code"], "too_many_event_streams");
        // Ordinary requests are unaffected by the stream limit.
        let (status, _) = send(&app, "GET", "/api/trial_count", &[]).await;
        assert_eq!(status, StatusCode::OK);

        streams.pop();
        let response = open_stream().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn checkpoint_saves_stop_at_the_quota_without_deleting_files() {
        let checkpoint_dir = tempfile::tempdir().unwrap();
        let mut options = ServerOptions::new(0);
        options.checkpoint_dir = checkpoint_dir.path().to_path_buf();
        options.max_saved_checkpoints = 2;
        let (app, _shutdown) =
            create_router_with_options_and_shutdown(test_engine(), options).unwrap();
        // Files the server did not generate through this endpoint do not count.
        std::fs::write(checkpoint_dir.path().join("checkpoint_000050.json"), "{}").unwrap();
        std::fs::write(checkpoint_dir.path().join("notes.txt"), "").unwrap();

        let mut saved = Vec::new();
        for _ in 0..2 {
            let (status, body) = send(&app, "POST", "/api/checkpoint/save", &[]).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            saved.push(PathBuf::from(body["path"].as_str().unwrap()));
        }
        let (status, body) = send(&app, "POST", "/api/checkpoint/save", &[]).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["code"], "checkpoint_quota_exceeded");
        assert!(
            saved.iter().all(|path| path.is_file()),
            "nothing is rotated out"
        );
        assert_eq!(std::fs::read_dir(checkpoint_dir.path()).unwrap().count(), 4);

        std::fs::remove_file(&saved[0]).unwrap();
        let (status, _) = send(&app, "POST", "/api/checkpoint/save", &[]).await;
        assert_eq!(status, StatusCode::OK);

        let mut zero = ServerOptions::new(0);
        zero.checkpoint_dir = checkpoint_dir.path().to_path_buf();
        zero.max_saved_checkpoints = 0;
        assert!(create_router_with_options(test_engine(), zero).is_err());
    }

    #[test]
    fn only_server_generated_names_count_as_saved_checkpoints() {
        assert!(is_saved_checkpoint_name(
            "checkpoint_1794873278799_000003.json"
        ));
        for name in [
            "checkpoint_000050.json",
            "checkpoint_1794873278799_000003.json.tmp",
            ".checkpoint_1794873278799_000003.json.tmp.1.2",
            "checkpoint__000003.json",
            "checkpoint_17_x.json",
            "other_17_3.json",
        ] {
            assert!(!is_saved_checkpoint_name(name), "{name}");
        }
    }

    #[tokio::test]
    async fn a_second_checkpoint_save_is_refused_while_one_is_running() {
        let checkpoint_dir = tempfile::tempdir().unwrap();
        let state = test_state(test_engine(), checkpoint_dir.path());
        let save = |state: Arc<ServerState>| {
            handle_checkpoint_save(
                State(state),
                HeaderMap::new(),
                Json(SaveCheckpointRequest { description: None }),
            )
        };

        // Stand in for a save whose blocking write is still in progress after
        // its request timed out: the detached task owns the only slot.
        let running = Arc::clone(&state.checkpoint_save_slot)
            .try_acquire_owned()
            .unwrap();
        let (status, Json(error)) = save(Arc::clone(&state)).await.unwrap_err();
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(error.code, "checkpoint_in_progress");
        assert!(
            std::fs::read_dir(checkpoint_dir.path())
                .unwrap()
                .next()
                .is_none()
        );

        drop(running);
        let _saved = save(Arc::clone(&state))
            .await
            .expect("the slot is free again");
        assert_eq!(state.checkpoint_save_slot.available_permits(), 1);
    }

    #[tokio::test]
    async fn a_ranked_read_keeps_its_slot_until_its_blocking_work_ends() {
        let checkpoint_dir = tempfile::tempdir().unwrap();
        let state = test_state(test_engine(), checkpoint_dir.path());
        let (release, blocked) = std::sync::mpsc::channel::<()>();

        let request_state = Arc::clone(&state);
        let request = tokio::spawn(async move {
            ranked_read(&request_state, move |_engine| {
                let _ = blocked.recv();
            })
            .await
        });
        let in_use = |state: &ServerState| {
            MAX_CONCURRENT_RANKED_READS - state.ranked_read_slots.available_permits()
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            while in_use(&state) != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the read should start");

        // A request timeout or disconnect drops the handler future, but the
        // blocking work it started cannot be interrupted.
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        assert_eq!(
            in_use(&state),
            1,
            "an abandoned read still occupies its slot"
        );

        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while in_use(&state) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the slot should be released when the work ends");
    }

    #[tokio::test]
    async fn dashboard_directory_must_not_contain_a_checkpoint_directory() {
        let root = tempfile::tempdir().unwrap();
        let dashboard = root.path().join("dashboard");
        let outside = root.path().join("checkpoints");
        std::fs::create_dir_all(dashboard.join("nested")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let router = |checkpoint_dir: &Path, engine: HolaEngine| {
            let mut options = ServerOptions::new(0);
            options.dashboard_dir = Some(dashboard.clone());
            options.checkpoint_dir = checkpoint_dir.to_path_buf();
            create_router_with_dashboard_and_options(engine, options)
        };

        assert!(router(&outside, test_engine()).is_ok());
        // The dashboard may live inside the checkpoint directory's parent.
        assert!(router(root.path(), test_engine()).is_ok());
        for exposed in [
            dashboard.clone(),
            dashboard.join("nested"),
            dashboard.join("new"),
        ] {
            let error = router(&exposed, test_engine()).unwrap_err().to_string();
            assert!(
                error.contains("contains the checkpoint directory"),
                "{error}"
            );
        }
        assert!(
            !dashboard.join("new").exists(),
            "a rejected configuration must not create its checkpoint directory"
        );

        // Automatic checkpoints configured in the study are covered as well.
        let mut config = test_engine().study_config().await;
        config.checkpoint = Some(crate::hola_engine::CheckpointConfig {
            directory: dashboard.join("auto").to_string_lossy().into_owned(),
            interval: 10,
            max_checkpoints: None,
            load_from: None,
        });
        let engine = HolaEngine::from_config(config).unwrap();
        let error = router(&outside, engine).unwrap_err().to_string();
        assert!(
            error.contains("contains the checkpoint directory"),
            "{error}"
        );

        #[cfg(unix)]
        {
            // Written beneath the dashboard, resolving elsewhere: still served.
            std::os::unix::fs::symlink(&outside, dashboard.join("link")).unwrap();
            let error = router(&dashboard.join("link"), test_engine())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("contains the checkpoint directory"),
                "{error}"
            );

            // Written elsewhere, resolving beneath the dashboard through a
            // symlinked parent, and not created yet.
            std::os::unix::fs::symlink(&dashboard, root.path().join("alias")).unwrap();
            let error = router(&root.path().join("alias").join("later"), test_engine())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("contains the checkpoint directory"),
                "{error}"
            );
            assert!(!dashboard.join("later").exists());
        }
    }

    /// Serve a health-only router with small connection limits.
    async fn limited_server(
        limits: ConnectionLimits,
    ) -> (
        std::net::SocketAddr,
        oneshot::Sender<()>,
        tokio::task::JoinHandle<std::io::Result<()>>,
    ) {
        let app = Router::new().route("/healthz", get(handle_health));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server = tokio::spawn(serve_listener_with_limits(
            listener,
            app,
            async move {
                let _ = shutdown_rx.await;
            },
            Duration::from_secs(2),
            limits,
        ));
        (address, shutdown_tx, server)
    }

    /// Read until the peer closes the socket; a reset counts as closed.
    async fn read_until_closed(client: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut received = Vec::new();
        let mut chunk = [0u8; 1024];
        loop {
            match client.read(&mut chunk).await {
                Ok(0) | Err(_) => return received,
                Ok(read) => received.extend_from_slice(&chunk[..read]),
            }
        }
    }

    #[tokio::test]
    async fn stalled_request_heads_and_idle_connections_are_closed() {
        let (address, shutdown, server) = limited_server(ConnectionLimits {
            head_timeout: Duration::from_millis(200),
            max_connections: 8,
        })
        .await;

        // A request head that never completes.
        let mut stalled = tokio::net::TcpStream::connect(address).await.unwrap();
        stalled
            .write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n")
            .await
            .unwrap();
        // A connection that sends nothing at all.
        let mut silent = tokio::net::TcpStream::connect(address).await.unwrap();
        // A keep-alive connection that goes idle after one complete request.
        let mut idle = tokio::net::TcpStream::connect(address).await.unwrap();
        idle.write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();

        let started = Instant::now();
        let (stalled, silent, idle) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(
                read_until_closed(&mut stalled),
                read_until_closed(&mut silent),
                read_until_closed(&mut idle),
            )
        })
        .await
        .expect("the server should close every connection at its head timeout");
        assert!(started.elapsed() >= Duration::from_millis(150));
        assert!(
            stalled.is_empty() || stalled.starts_with(b"HTTP/1.1 408"),
            "{}",
            String::from_utf8_lossy(&stalled)
        );
        assert!(silent.is_empty() || silent.starts_with(b"HTTP/1.1 408"));
        assert!(
            idle.starts_with(b"HTTP/1.1 200 OK"),
            "the complete request is served"
        );

        shutdown.send(()).unwrap();
        server.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn connections_beyond_the_limit_are_closed_until_a_slot_frees() {
        let (address, shutdown, server) = limited_server(ConnectionLimits {
            head_timeout: Duration::from_secs(30),
            max_connections: 2,
        })
        .await;
        let request = b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";

        // Two connections that hold their slots by staying silent.
        let first = tokio::net::TcpStream::connect(address).await.unwrap();
        let second = tokio::net::TcpStream::connect(address).await.unwrap();
        // The listener accepts in order, so by the time a third connection is
        // refused the first two own both slots.
        let mut third = tokio::net::TcpStream::connect(address).await.unwrap();
        let _ = third.write_all(request).await;
        let refused = tokio::time::timeout(Duration::from_secs(5), read_until_closed(&mut third))
            .await
            .expect("a connection over the limit should be closed promptly");
        assert!(refused.is_empty(), "{}", String::from_utf8_lossy(&refused));

        drop(first);
        let served = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
                let _ = client.write_all(request).await;
                let response = read_until_closed(&mut client).await;
                if response.starts_with(b"HTTP/1.1 200 OK") {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        served.expect("a freed slot should admit a new connection");

        drop(second);
        shutdown.send(()).unwrap();
        server.await.unwrap().unwrap();
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
            event_stream_slots: Arc::new(Semaphore::new(MAX_EVENT_STREAMS)),
            ranked_read_slots: Arc::new(Semaphore::new(MAX_CONCURRENT_RANKED_READS)),
            checkpoint_save_slot: Arc::new(Semaphore::new(1)),
            max_saved_checkpoints: DEFAULT_MAX_SAVED_CHECKPOINTS,
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
