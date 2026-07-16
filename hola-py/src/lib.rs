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

//! Python bindings for the HOLA optimization engine via PyO3.

use hola_engine::hola_engine::{
    DEFAULT_MAX_REFIT_CANDIDATES, DEFAULT_MAX_REFIT_SAMPLES, HolaEngine, ObjectiveConfig,
    ParamConfig, StrategyConfig, StudyConfig,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::de::DeserializeOwned;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use uuid::Uuid;

// All public HOLA exceptions remain ValueError subclasses for compatibility
// with callers that caught the binding's historical catch-all ValueError.
create_exception!(
    hola_opt,
    HolaError,
    PyValueError,
    "Base exception for errors raised by HOLA."
);
create_exception!(
    hola_opt,
    ConfigurationError,
    HolaError,
    "Invalid study, strategy, space, objective, or client configuration."
);
create_exception!(
    hola_opt,
    CheckpointError,
    HolaError,
    "Checkpoint loading or saving failed."
);
create_exception!(
    hola_opt,
    RemoteError,
    HolaError,
    "A remote server request, response, or protocol operation failed."
);
create_exception!(
    hola_opt,
    ObjectiveError,
    HolaError,
    "An objective result violated its declared contract or could not be reported."
);

// =============================================================================
// Space building helpers
// =============================================================================

/// Real-valued parameter with configurable scale.
///
/// Args:
///     min: Lower bound (in actual values, not exponents).
///     max: Upper bound (in actual values, not exponents).
///     scale: Sampling scale: "linear" (default), "log" (natural log), "ln"
///         (alias for "log"), or "log10".
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Real {
    #[pyo3(get)]
    min: f64,
    #[pyo3(get)]
    max: f64,
    #[pyo3(get)]
    scale: String,
}

#[pymethods]
impl Real {
    #[new]
    #[pyo3(signature = (min, max, scale="linear"))]
    fn new(min: f64, max: f64, scale: &str) -> PyResult<Self> {
        match scale {
            // "ln" is a natural-log alias accepted by the engine's config layer
            // and is treated identically to "log".
            "linear" | "log" | "ln" | "log10" => {}
            _ => {
                return Err(ConfigurationError::new_err(
                    "scale must be \"linear\", \"log\", \"ln\", or \"log10\"",
                ));
            }
        }
        Ok(Self {
            min,
            max,
            scale: scale.to_string(),
        })
    }
}

/// Integer parameter within an inclusive range.
///
/// Args:
///     min: Lower bound (inclusive).
///     max: Upper bound (inclusive).
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Integer {
    #[pyo3(get)]
    min: i64,
    #[pyo3(get)]
    max: i64,
}

#[pymethods]
impl Integer {
    #[new]
    fn new(min: i64, max: i64) -> Self {
        Self { min, max }
    }
}

/// Categorical parameter (choose from a list of string labels).
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Categorical {
    #[pyo3(get)]
    choices: Vec<String>,
}

#[pymethods]
impl Categorical {
    #[new]
    fn new(choices: Vec<String>) -> Self {
        Self { choices }
    }
}

// =============================================================================
// Objective helpers
// =============================================================================

/// Minimize an objective field using TLP (Target-Limit-Priority) scoring.
///
/// Args:
///     field: Name of the metric to minimize (must appear in the dict returned
///         by the objective function).
///     target: "Good enough" value; at or below this, the TLP score is 0.
///     limit: Worst acceptable value; beyond this, the trial is infeasible (score = inf).
///     priority: The score at the limit and relative weight P_i. The linear
///         segment's slope is P_i / (limit − target). Default 1.0.
///     group: Priority-group label. Objectives sharing the same group are summed
///         into one component of the group-cost vector for Pareto ranking. When
///         omitted, defaults to the field name (one group per objective).
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Minimize {
    #[pyo3(get)]
    field: String,
    #[pyo3(get)]
    target: Option<f64>,
    #[pyo3(get)]
    limit: Option<f64>,
    #[pyo3(get)]
    priority: f64,
    #[pyo3(get)]
    group: Option<String>,
}

#[pymethods]
impl Minimize {
    #[new]
    #[pyo3(signature = (field, target=None, limit=None, priority=1.0, group=None))]
    fn new(
        field: String,
        target: Option<f64>,
        limit: Option<f64>,
        priority: f64,
        group: Option<String>,
    ) -> Self {
        Self {
            field,
            target,
            limit,
            priority,
            group,
        }
    }
}

/// Maximize an objective field using TLP (Target-Limit-Priority) scoring.
///
/// Args:
///     field: Name of the metric to maximize (must appear in the dict returned
///         by the objective function).
///     target: "Good enough" value; at or above this, the TLP score is 0.
///     limit: Worst acceptable value; below this, the trial is infeasible (score = inf).
///     priority: The score at the limit and relative weight P_i. The linear
///         segment's slope is P_i / (limit − target). Default 1.0.
///     group: Priority-group label. See Minimize for details.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Maximize {
    #[pyo3(get)]
    field: String,
    #[pyo3(get)]
    target: Option<f64>,
    #[pyo3(get)]
    limit: Option<f64>,
    #[pyo3(get)]
    priority: f64,
    #[pyo3(get)]
    group: Option<String>,
}

#[pymethods]
impl Maximize {
    #[new]
    #[pyo3(signature = (field, target=None, limit=None, priority=1.0, group=None))]
    fn new(
        field: String,
        target: Option<f64>,
        limit: Option<f64>,
        priority: f64,
        group: Option<String>,
    ) -> Self {
        Self {
            field,
            target,
            limit,
            priority,
            group,
        }
    }
}

// =============================================================================
// Strategy configuration classes
// =============================================================================

/// GMM strategy configuration.
///
/// Configures the Gaussian Mixture Model strategy. Use this class instead of
/// the string `"gmm"` when you need to adjust refit behavior.
///
/// Args:
///     refit_interval: How often the GMM is refit, in completed trials (default: 20).
///     elite_fraction: Fraction of top trials used for GMM refitting (default: 0.25).
///         Must be between 0.0 and 1.0.
///     exploration_budget: Number of Sobol exploration trials before GMM exploitation
///         begins. When omitted, computed automatically from the number of dimensions.
///     max_refit_samples: Maximum elite samples used by one GMM fit (default: 4096).
///     max_refit_candidates: Maximum retained trials ranked during elite selection
///         (default: 16384). Longer histories use deterministic stratified coverage.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Gmm {
    #[pyo3(get)]
    refit_interval: Option<usize>,
    #[pyo3(get)]
    elite_fraction: Option<f64>,
    #[pyo3(get)]
    exploration_budget: Option<usize>,
    #[pyo3(get)]
    max_refit_samples: Option<usize>,
    #[pyo3(get)]
    max_refit_candidates: Option<usize>,
}

#[pymethods]
impl Gmm {
    #[new]
    #[pyo3(signature = (refit_interval=None, elite_fraction=None, exploration_budget=None, max_refit_samples=None, max_refit_candidates=None))]
    fn new(
        refit_interval: Option<usize>,
        elite_fraction: Option<f64>,
        exploration_budget: Option<usize>,
        max_refit_samples: Option<usize>,
        max_refit_candidates: Option<usize>,
    ) -> PyResult<Self> {
        if let Some(ef) = elite_fraction {
            if !ef.is_finite() || ef <= 0.0 || ef > 1.0 {
                return Err(ConfigurationError::new_err(
                    "elite_fraction must be finite and between 0.0 (exclusive) and 1.0 (inclusive)",
                ));
            }
        }
        if let Some(ri) = refit_interval {
            if ri == 0 {
                return Err(ConfigurationError::new_err(
                    "refit_interval must be at least 1",
                ));
            }
        }
        let effective_max_refit_samples = max_refit_samples.unwrap_or(DEFAULT_MAX_REFIT_SAMPLES);
        let effective_max_refit_candidates =
            max_refit_candidates.unwrap_or(DEFAULT_MAX_REFIT_CANDIDATES);
        if effective_max_refit_samples == 0 {
            return Err(ConfigurationError::new_err(
                "max_refit_samples must be at least 1",
            ));
        }
        if effective_max_refit_candidates < effective_max_refit_samples {
            return Err(ConfigurationError::new_err(format!(
                "max_refit_candidates must be at least max_refit_samples ({effective_max_refit_samples}), got {effective_max_refit_candidates}",
            )));
        }
        Ok(Self {
            refit_interval,
            elite_fraction,
            exploration_budget,
            max_refit_samples,
            max_refit_candidates,
        })
    }
}

/// Sobol strategy configuration.
///
/// Owen-scrambled Sobol quasi-random sequences provide better space coverage
/// than pure random sampling. Use this class instead of the string `"sobol"`
/// for consistency with other strategy classes.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Sobol;

#[pymethods]
impl Sobol {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Random strategy configuration.
///
/// Uniform pseudo-random sampling. Use this class instead of the string
/// `"random"` for consistency with other strategy classes.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Random;

#[pymethods]
impl Random {
    #[new]
    fn new() -> Self {
        Self
    }
}

// =============================================================================
// Space builder
// =============================================================================

/// Named parameter space builder.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Space {
    params: BTreeMap<String, ParamConfig>,
}

#[pymethods]
impl Space {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut params = BTreeMap::new();
        if let Some(dict) = kwargs {
            for (key, val) in dict.iter() {
                let name: String = key.extract()?;
                let config = extract_param_config(&val)?;
                params.insert(name, config);
            }
        }
        Ok(Self { params })
    }
}

fn extract_param_config(obj: &Bound<'_, PyAny>) -> PyResult<ParamConfig> {
    if let Ok(r) = obj.extract::<Real>() {
        return Ok(ParamConfig::Real {
            min: r.min,
            max: r.max,
            scale: r.scale,
        });
    }
    if let Ok(d) = obj.extract::<Integer>() {
        return Ok(ParamConfig::Integer {
            min: d.min,
            max: d.max,
        });
    }
    if let Ok(c) = obj.extract::<Categorical>() {
        return Ok(ParamConfig::Categorical { choices: c.choices });
    }
    Err(ConfigurationError::new_err(
        "Parameter must be Real, Integer, or Categorical",
    ))
}

// =============================================================================
// Trial types
// =============================================================================

/// A trial returned by Study.ask().
#[pyclass]
struct Trial {
    #[pyo3(get)]
    trial_id: u64,
    #[pyo3(get)]
    params: Py<PyAny>,
}

#[pymethods]
impl Trial {
    fn __repr__(&self, py: Python<'_>) -> String {
        let params_str = self
            .params
            .bind(py)
            .repr()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "?".to_string());
        format!("Trial(trial_id={}, params={})", self.trial_id, params_str)
    }
}

/// A completed trial with scoring, ranking, and Pareto front information.
#[pyclass]
struct CompletedTrial {
    #[pyo3(get)]
    trial_id: u64,
    #[pyo3(get)]
    params: Py<PyAny>,
    #[pyo3(get)]
    metrics: Py<PyAny>,
    #[pyo3(get)]
    scores: Py<PyAny>,
    #[pyo3(get)]
    score_vector: Py<PyAny>,
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get)]
    pareto_front: usize,
    #[pyo3(get)]
    completed_at: u64,
}

#[pymethods]
impl CompletedTrial {
    fn __repr__(&self, py: Python<'_>) -> String {
        let params_str = self
            .params
            .bind(py)
            .repr()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "?".to_string());
        format!(
            "CompletedTrial(trial_id={}, rank={}, params={})",
            self.trial_id, self.rank, params_str
        )
    }
}

/// Convert a Rust CompletedTrial to a Python CompletedTrial.
fn rust_to_py_completed(
    py: Python<'_>,
    ct: &hola_engine::hola_engine::CompletedTrial,
) -> PyResult<CompletedTrial> {
    Ok(CompletedTrial {
        trial_id: ct.trial_id,
        // Parameters and raw metrics are arbitrary user JSON. In particular,
        // categorical values and metadata may legitimately be the literal
        // strings "inf", "-inf", or "nan", so preserve them verbatim.
        params: json_to_py(py, &ct.params, JsonStringMode::Literal)?,
        metrics: json_to_py(py, &ct.metrics, JsonStringMode::Literal)?,
        // Scores are numeric protocol fields. The engine represents their
        // non-finite values with the documented string sentinels because JSON
        // numbers cannot encode them, so decode only in these numeric maps.
        scores: json_to_py(py, &ct.scores, JsonStringMode::DecodeNonFinite)?,
        score_vector: json_to_py(py, &ct.score_vector, JsonStringMode::DecodeNonFinite)?,
        rank: ct.rank,
        pareto_front: ct.pareto_front,
        completed_at: ct.completed_at,
    })
}

// =============================================================================
// Study: the main user-facing class
// =============================================================================

const DEFAULT_CONNECT_TIMEOUT_SECONDS: f64 = 10.0;
const DEFAULT_REQUEST_TIMEOUT_SECONDS: f64 = 30.0;
const MAX_ERROR_BODY_CHARS: usize = 4096;

/// One runtime owns all async work issued by Python studies for the lifetime of
/// the process. Keeping it in a `OnceLock` prevents each `Study` from creating a
/// full worker pool while still allowing calls from multiple Python threads.
static SHARED_RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();

fn shared_runtime() -> PyResult<&'static tokio::runtime::Runtime> {
    match SHARED_RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .thread_name("hola-py-worker")
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to create runtime: {e}"))
    }) {
        Ok(runtime) => Ok(runtime),
        Err(message) => Err(HolaError::new_err(message.clone())),
    }
}

fn timeout_duration(name: &str, seconds: f64) -> PyResult<Duration> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(ConfigurationError::new_err(format!(
            "{name} must be a finite number greater than zero"
        )));
    }
    Ok(Duration::from_secs_f64(seconds))
}

struct RemoteHttpClient {
    client: reqwest::Client,
    token: Option<String>,
    /// One unresolved ask key is retained until a complete, valid response is
    /// received. The mutex also serializes concurrent calls so two callers
    /// cannot accidentally share the same in-flight key/trial.
    ask_idempotency_key: Mutex<Option<String>>,
}

impl RemoteHttpClient {
    fn new(token: Option<String>, connect_timeout: f64, request_timeout: f64) -> PyResult<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(timeout_duration("connect_timeout", connect_timeout)?)
            .timeout(timeout_duration("request_timeout", request_timeout)?)
            .build()
            .map_err(|e| {
                ConfigurationError::new_err(format!("Failed to create HTTP client: {e}"))
            })?;
        Ok(Self {
            client,
            token,
            ask_idempotency_key: Mutex::new(None),
        })
    }

    fn with_auth(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.token.as_deref() {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    fn get(&self, url: String) -> reqwest::RequestBuilder {
        self.with_auth(self.client.get(url))
    }

    fn post(&self, url: String) -> reqwest::RequestBuilder {
        self.with_auth(self.client.post(url))
    }

    fn patch(&self, url: String) -> reqwest::RequestBuilder {
        self.with_auth(self.client.patch(url))
    }
}

async fn send_remote(request: reqwest::RequestBuilder) -> Result<reqwest::Response, String> {
    request.send().await.map_err(|e| {
        if e.is_timeout() {
            format!("HTTP request timed out: {e}")
        } else if e.is_connect() {
            format!("HTTP connection failed: {e}")
        } else {
            format!("HTTP request failed: {e}")
        }
    })
}

async fn response_error(response: reqwest::Response) -> String {
    let status = response.status();
    let body = response
        .text()
        .await
        .unwrap_or_else(|e| format!("unable to read error response: {e}"));
    let detail = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .or_else(|| value.get("message"))
                .map(|detail| {
                    detail
                        .as_str()
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| detail.to_string())
                })
        })
        .unwrap_or_else(|| {
            let trimmed = body.trim();
            if trimmed.is_empty() {
                "empty response body".to_string()
            } else {
                trimmed.chars().take(MAX_ERROR_BODY_CHARS).collect()
            }
        });
    format!("Server returned HTTP {status}: {detail}")
}

async fn checked_response(request: reqwest::RequestBuilder) -> Result<reqwest::Response, String> {
    let response = send_remote(request).await?;
    if response.status().is_success() {
        Ok(response)
    } else {
        Err(response_error(response).await)
    }
}

async fn response_json<T: DeserializeOwned>(response: reqwest::Response) -> Result<T, String> {
    response
        .json()
        .await
        .map_err(|e| format!("Invalid JSON response from server: {e}"))
}

async fn checked_json<T: DeserializeOwned>(request: reqwest::RequestBuilder) -> Result<T, String> {
    response_json(checked_response(request).await?).await
}

fn validate_remote_trial_identity(
    trial: serde_json::Value,
    expected_trial_id: u64,
    source: &str,
) -> Result<serde_json::Value, String> {
    let returned_trial_id = trial
        .get("trial_id")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("{source} is missing trial_id"))?;
    if returned_trial_id != expected_trial_id {
        return Err(format!(
            "{source} returned trial_id {returned_trial_id} instead of {expected_trial_id}"
        ));
    }
    Ok(trial)
}

fn validate_status_ok(body: &serde_json::Value, operation: &str) -> Result<(), String> {
    if body.get("status").and_then(serde_json::Value::as_str) != Some("ok") {
        return Err(format!(
            "{operation} acknowledgement is missing canonical status 'ok'"
        ));
    }
    Ok(())
}

fn validate_post_commit_warnings(
    body: &serde_json::Value,
    operation: &str,
) -> Result<Vec<String>, String> {
    match body.get("post_commit_warnings") {
        None => Ok(Vec::new()),
        Some(serde_json::Value::Array(warnings)) => warnings
            .iter()
            .map(|warning| {
                warning.as_str().map(ToOwned::to_owned).ok_or_else(|| {
                    format!(
                        "{operation} acknowledgement has a non-string post_commit_warnings entry"
                    )
                })
            })
            .collect(),
        Some(_) => Err(format!(
            "{operation} acknowledgement has non-array post_commit_warnings"
        )),
    }
}

/// Internal representation: local engine or remote HTTP client. Async work is
/// executed on `SHARED_RUNTIME`, never on a runtime owned by an individual study.
// Exactly one StudyInner exists per Study (never stored in a collection), so the
// size gap between the engine-bearing Local variant and the Remote variant is
// immaterial; boxing the engine would only add an indirection for no benefit.
#[allow(clippy::large_enum_variant)]
enum StudyInner {
    Local { engine: HolaEngine },
    Remote { url: String, http: RemoteHttpClient },
}

/// The main optimization study.
///
/// Usage:
///     study = Study(space=Space(lr=Real(1e-4, 0.1, scale="log10")), objectives=[Minimize("loss")])
///     trial = study.ask()
///     ct = study.tell(trial.trial_id, {"loss": 0.42})
///     top = study.top_k(3)
///
/// Args:
///     space: Parameter space to search.
///     objectives: One or more Minimize/Maximize objectives.
///     strategy: Sampling strategy, as a string ("gmm", "sobol", "random") or a
///         strategy class (Gmm, Sobol, Random). Defaults to GMM.
///     seed: Optional RNG seed for reproducible sampling.
///     max_trials: Optional cap on the total number of trials (ask gating).
///         None (default) means unbounded.
///     max_leaderboard_size: Opt-in cap on how many completed trials are
///         retained. None (default) keeps the leaderboard unbounded and
///         preserves the retain-everything behavior. When set, must be >= 1.
#[pyclass(skip_from_py_object)]
struct Study {
    inner: StudyInner,
}

fn extract_objectives(objectives: &Bound<'_, PyList>) -> PyResult<Vec<ObjectiveConfig>> {
    if objectives.len() == 0 {
        return Err(ConfigurationError::new_err(
            "At least one objective is required (e.g., [Minimize('loss')])",
        ));
    }
    let mut obj_configs = Vec::new();
    for item in objectives.iter() {
        if let Ok(m) = item.extract::<Minimize>() {
            obj_configs.push(ObjectiveConfig {
                field: m.field,
                obj_type: "minimize".to_string(),
                target: m.target,
                limit: m.limit,
                priority: m.priority,
                group: m.group,
            });
        } else if let Ok(m) = item.extract::<Maximize>() {
            obj_configs.push(ObjectiveConfig {
                field: m.field,
                obj_type: "maximize".to_string(),
                target: m.target,
                limit: m.limit,
                priority: m.priority,
                group: m.group,
            });
        } else {
            return Err(ConfigurationError::new_err(
                "Objectives must be Minimize or Maximize instances",
            ));
        }
    }
    Ok(obj_configs)
}

#[pymethods]
impl Study {
    #[new]
    #[pyo3(signature = (space, objectives, strategy=None, seed=None, max_trials=None, max_leaderboard_size=None))]
    fn new(
        space: Space,
        objectives: &Bound<'_, PyList>,
        strategy: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
        max_trials: Option<usize>,
        max_leaderboard_size: Option<usize>,
    ) -> PyResult<Self> {
        let obj_configs = extract_objectives(objectives)?;

        if let Some(0) = max_leaderboard_size {
            return Err(ConfigurationError::new_err(
                "max_leaderboard_size must be at least 1",
            ));
        }

        // Accept either a string shortcut or a strategy configuration class.
        // Default to "gmm" when strategy is None.
        let strategy_config = match strategy {
            None => StrategyConfig {
                strategy_type: "gmm".to_string(),
                refit_interval: 20,
                total_budget: max_trials,
                exploration_budget: None,
                seed,
                elite_fraction: None,
                max_refit_samples: DEFAULT_MAX_REFIT_SAMPLES,
                max_refit_candidates: DEFAULT_MAX_REFIT_CANDIDATES,
            },
            Some(s) => {
                // Extract each strategy class exactly once: probe with a single
                // extract() and reuse its value, rather than extracting in a
                // match guard and again in the arm body.
                if let Ok(gmm) = s.extract::<Gmm>() {
                    StrategyConfig {
                        strategy_type: "gmm".to_string(),
                        refit_interval: gmm.refit_interval.unwrap_or(20),
                        total_budget: max_trials,
                        exploration_budget: gmm.exploration_budget,
                        seed,
                        elite_fraction: gmm.elite_fraction,
                        max_refit_samples: gmm
                            .max_refit_samples
                            .unwrap_or(DEFAULT_MAX_REFIT_SAMPLES),
                        max_refit_candidates: gmm
                            .max_refit_candidates
                            .unwrap_or(DEFAULT_MAX_REFIT_CANDIDATES),
                    }
                } else if s.extract::<Sobol>().is_ok() {
                    StrategyConfig {
                        strategy_type: "sobol".to_string(),
                        refit_interval: 20,
                        total_budget: max_trials,
                        exploration_budget: None,
                        seed,
                        elite_fraction: None,
                        max_refit_samples: DEFAULT_MAX_REFIT_SAMPLES,
                        max_refit_candidates: DEFAULT_MAX_REFIT_CANDIDATES,
                    }
                } else if s.extract::<Random>().is_ok() {
                    StrategyConfig {
                        strategy_type: "random".to_string(),
                        refit_interval: 20,
                        total_budget: max_trials,
                        exploration_budget: None,
                        seed,
                        elite_fraction: None,
                        max_refit_samples: DEFAULT_MAX_REFIT_SAMPLES,
                        max_refit_candidates: DEFAULT_MAX_REFIT_CANDIDATES,
                    }
                } else {
                    let name: String = s.extract().map_err(|_| {
                        ConfigurationError::new_err(
                            "strategy must be a string (\"gmm\", \"sobol\", \"random\") \
                             or a strategy class (Gmm, Sobol, Random)",
                        )
                    })?;
                    StrategyConfig {
                        strategy_type: name,
                        refit_interval: 20,
                        total_budget: max_trials,
                        exploration_budget: None,
                        seed,
                        elite_fraction: None,
                        max_refit_samples: DEFAULT_MAX_REFIT_SAMPLES,
                        max_refit_candidates: DEFAULT_MAX_REFIT_CANDIDATES,
                    }
                }
            }
        };

        let config = StudyConfig {
            space: space.params,
            objectives: obj_configs,
            strategy: Some(strategy_config),
            checkpoint: None,
            max_trials,
            max_leaderboard_size,
        };

        let engine = HolaEngine::from_config(config)
            .map_err(|e| ConfigurationError::new_err(format!("Failed to create engine: {e}")))?;
        // Initialize the process-wide runtime only after configuration validation
        // succeeds, so invalid studies do not create worker threads as a side effect.
        shared_runtime()?;

        Ok(Self {
            inner: StudyInner::Local { engine },
        })
    }

    /// Connect to an existing HOLA server.
    ///
    /// The connection is established lazily on the first ``ask``/``tell``; this
    /// call only validates and stores the URL (no network request is made here).
    ///
    /// Args:
    ///     url: Base URL of the HOLA server.
    ///     token: Optional bearer token sent with every remote request.
    ///     connect_timeout: Maximum seconds allowed to establish a connection.
    ///     request_timeout: Maximum seconds allowed for a complete HTTP request.
    #[staticmethod]
    #[pyo3(signature = (url, token=None, *, connect_timeout=DEFAULT_CONNECT_TIMEOUT_SECONDS, request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS))]
    fn connect(
        url: &str,
        token: Option<String>,
        connect_timeout: f64,
        request_timeout: f64,
    ) -> PyResult<Self> {
        // Validate the URL up front so a malformed or non-HTTP URL fails here
        // with a clear error rather than at the first ask/tell. No network
        // request is made; the connection is established lazily.
        let mut parsed = reqwest::Url::parse(url)
            .map_err(|e| ConfigurationError::new_err(format!("Invalid server URL '{url}': {e}")))?;
        match parsed.scheme() {
            "http" | "https" => {}
            other => {
                return Err(ConfigurationError::new_err(format!(
                    "Server URL must use the http or https scheme, got '{other}'"
                )));
            }
        }
        if parsed.host_str().is_none() {
            return Err(ConfigurationError::new_err(format!(
                "Invalid server URL '{url}': missing host"
            )));
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(ConfigurationError::new_err(
                "Server URL must not contain embedded userinfo; pass token= for authentication",
            ));
        }
        if parsed.query().is_some() {
            return Err(ConfigurationError::new_err(
                "Server URL must not contain query parameters",
            ));
        }
        if parsed.fragment().is_some() {
            return Err(ConfigurationError::new_err(
                "Server URL must not contain a fragment",
            ));
        }
        let normalized_path = parsed.path().trim_end_matches('/').to_string();
        parsed.set_path(&normalized_path);
        let normalized_url = parsed.to_string().trim_end_matches('/').to_string();
        let http = RemoteHttpClient::new(token, connect_timeout, request_timeout)?;
        shared_runtime()?;
        Ok(Self {
            inner: StudyInner::Remote {
                url: normalized_url,
                http,
            },
        })
    }

    /// Load a study from a saved checkpoint.
    ///
    /// The checkpoint file must have been saved with ``study.save()``, which
    /// embeds the full study configuration (space, objectives) alongside the
    /// trial history and strategy state.
    ///
    /// Args:
    ///     path: Path to the checkpoint JSON file.
    ///
    /// Returns:
    ///     A fully restored Study that can immediately resume ``ask``/``tell``.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let runtime = shared_runtime()?;
        // Loading performs filesystem I/O, JSON validation, and strategy
        // reconstruction. Own the path and release the GIL for the whole
        // operation so unrelated Python threads keep making progress on large
        // checkpoints.
        let path = path.to_string();
        let engine = py
            .detach(|| runtime.block_on(HolaEngine::load_from_checkpoint(&path)))
            .map_err(|e| CheckpointError::new_err(format!("Failed to load checkpoint: {e}")))?;

        Ok(Self {
            inner: StudyInner::Local { engine },
        })
    }

    /// Request the next trial to evaluate.
    fn ask(&self, py: Python<'_>) -> PyResult<Trial> {
        let runtime = shared_runtime()?;
        match &self.inner {
            StudyInner::Local { engine } => {
                // Release the GIL while the engine future runs (RwLock, sampling).
                let dyn_trial = py
                    .detach(|| runtime.block_on(engine.ask()))
                    .map_err(HolaError::new_err)?;
                let params = json_to_py(py, &dyn_trial.params, JsonStringMode::Literal)?;
                Ok(Trial {
                    trial_id: dyn_trial.trial_id,
                    params,
                })
            }
            StudyInner::Remote { url, http } => {
                // Release the GIL during the HTTP round-trip. Keep the same key
                // after any transport, status, JSON, or schema error: the server
                // may already have allocated the trial even though this client
                // did not receive an unambiguous response. Validation happens
                // while the key slot is locked; only a complete response rotates
                // it. This also gives concurrent ask() calls distinct keys.
                let (trial_id, params_json): (u64, serde_json::Value) = py
                    .detach(|| {
                        let mut key_slot = http.ask_idempotency_key.lock().map_err(|_| {
                            "Remote ask idempotency state is unavailable".to_string()
                        })?;
                        let idempotency_key = key_slot
                            .get_or_insert_with(|| Uuid::new_v4().to_string())
                            .clone();
                        let resp: serde_json::Value = runtime.block_on(checked_json(
                            http.post(format!("{url}/api/ask"))
                                .header("Idempotency-Key", idempotency_key),
                        ))?;
                        let trial_id = resp
                            .get("trial_id")
                            .and_then(|v| v.as_u64())
                            .ok_or_else(|| "Missing 'trial_id' in server response".to_string())?;
                        let params = resp
                            .get("params")
                            .filter(|params| params.is_object())
                            .cloned()
                            .ok_or_else(|| {
                                "Missing or non-object 'params' in server response".to_string()
                            })?;
                        *key_slot = None;
                        Ok::<_, String>((trial_id, params))
                    })
                    .map_err(RemoteError::new_err)?;
                let params = json_to_py(py, &params_json, JsonStringMode::Literal)?;
                Ok(Trial { trial_id, params })
            }
        }
    }

    /// Report the result of a trial. Returns the scored and ranked CompletedTrial.
    fn tell(
        &self,
        py: Python<'_>,
        trial_id: u64,
        metrics: &Bound<'_, PyDict>,
    ) -> PyResult<CompletedTrial> {
        // Convert the metrics dict to owned JSON before releasing the GIL.
        let raw = py_dict_to_json(metrics)?;
        let runtime = shared_runtime()?;
        match &self.inner {
            StudyInner::Local { engine } => {
                // Release the GIL during the engine future (RwLock, GMM refit,
                // Pareto ranking).
                let completed = py
                    .detach(|| runtime.block_on(engine.tell(trial_id, raw)))
                    .map_err(ObjectiveError::new_err)?;
                rust_to_py_completed(py, &completed)
            }
            StudyInner::Remote { url, http } => {
                // Release the GIL during the HTTP round-trip(s).
                let (trial_json, post_commit_warnings): (serde_json::Value, Vec<String>) = py
                    .detach(|| {
                        runtime.block_on(async {
                            let tell_body: serde_json::Value = checked_json(
                                http.post(format!("{url}/api/tell"))
                                    .json(&serde_json::json!({
                                        "trial_id": trial_id,
                                        "metrics": raw,
                                    })),
                            )
                            .await?;
                            validate_status_ok(&tell_body, "Tell")?;
                            let post_commit_warnings =
                                validate_post_commit_warnings(&tell_body, "Tell")?;
                            if let Some(trial) = tell_body.get("trial") {
                                return Ok((
                                    validate_remote_trial_identity(
                                        trial.clone(),
                                        trial_id,
                                        "Tell acknowledgement",
                                    )?,
                                    post_commit_warnings,
                                ));
                            }

                            // Compatibility fallback for older servers whose
                            // tell response did not embed the completed trial.
                            // A missing single-trial endpoint is expected on
                            // those versions; every other HTTP error is surfaced.
                            let trial_resp = send_remote(http.get(format!(
                                "{url}/api/trial/{trial_id}?include_infeasible=true"
                            )))
                            .await?;
                            if trial_resp.status().is_success() {
                                let trial = response_json(trial_resp).await?;
                                return Ok((
                                    validate_remote_trial_identity(
                                        trial,
                                        trial_id,
                                        "Single-trial response",
                                    )?,
                                    post_commit_warnings,
                                ));
                            }
                            if trial_resp.status() != reqwest::StatusCode::NOT_FOUND
                                && trial_resp.status() != reqwest::StatusCode::METHOD_NOT_ALLOWED
                            {
                                return Err(response_error(trial_resp).await);
                            }

                            let trials_resp: Vec<serde_json::Value> = checked_json(http.get(
                                format!("{url}/api/trials?sorted_by=index&include_infeasible=true"),
                            ))
                            .await?;
                            let trial = trials_resp
                                .into_iter()
                                .find(|t| {
                                    t.get("trial_id").and_then(|v| v.as_u64()) == Some(trial_id)
                                })
                                .ok_or_else(|| {
                                    format!("Trial {trial_id} not found in server response")
                                })?;
                            Ok((trial, post_commit_warnings))
                        })
                    })
                    .map_err(RemoteError::new_err)?;

                let ct: hola_engine::hola_engine::CompletedTrial =
                    serde_json::from_value(trial_json)
                        .map_err(|e| RemoteError::new_err(format!("Deserialization error: {e}")))?;
                let runtime_warning = py.get_type::<PyRuntimeWarning>();
                let warnings = PyModule::import(py, "warnings")?;
                for warning in post_commit_warnings {
                    // Warning filters can be configured as errors. A trial is
                    // already committed at this point, so never turn operator
                    // notification policy into an ambiguous tell exception.
                    if warnings
                        .call_method1("warn", (&warning, &runtime_warning, 2))
                        .is_err()
                    {
                        eprintln!("[hola] post-commit warning: {warning}");
                    }
                }
                rust_to_py_completed(py, &ct)
            }
        }
    }

    /// Cancel a pending trial.
    fn cancel(&self, py: Python<'_>, trial_id: u64) -> PyResult<()> {
        let runtime = shared_runtime()?;
        match &self.inner {
            StudyInner::Local { engine } => py
                .detach(|| runtime.block_on(engine.cancel(trial_id)))
                .map_err(HolaError::new_err),
            StudyInner::Remote { url, http } => py
                .detach(|| {
                    runtime.block_on(async {
                        let acknowledgement: serde_json::Value = checked_json(
                            http.post(format!("{url}/api/cancel"))
                                .json(&serde_json::json!({ "trial_id": trial_id })),
                        )
                        .await?;
                        validate_status_ok(&acknowledgement, "Cancel")?;
                        if let Some(value) = acknowledgement.get("trial_id") {
                            let acknowledged_id = value.as_u64().ok_or_else(|| {
                                "Cancel acknowledgement has a non-integer trial_id".to_string()
                            })?;
                            if acknowledged_id != trial_id {
                                return Err(format!(
                                    "Cancel acknowledgement returned trial_id {acknowledged_id} instead of {trial_id}"
                                ));
                            }
                        }
                        Ok::<(), String>(())
                    })
                })
                .map_err(RemoteError::new_err),
        }
    }

    /// Renew the server-managed lease for a pending remote trial.
    ///
    /// Returns the new absolute lease deadline as Unix milliseconds. Local
    /// studies do not use distributed leases and reject this operation.
    fn heartbeat(&self, py: Python<'_>, trial_id: u64) -> PyResult<u64> {
        let runtime = shared_runtime()?;
        match &self.inner {
            StudyInner::Local { .. } => Err(ConfigurationError::new_err(
                "heartbeat() is only available for remote connections",
            )),
            StudyInner::Remote { url, http } => {
                let response: serde_json::Value = py
                    .detach(|| {
                        runtime.block_on(checked_json(
                            http.post(format!("{url}/api/heartbeat"))
                                .json(&serde_json::json!({ "trial_id": trial_id })),
                        ))
                    })
                    .map_err(RemoteError::new_err)?;
                if response.get("status").and_then(|value| value.as_str()) != Some("ok") {
                    return Err(RemoteError::new_err(
                        "Heartbeat response is missing status 'ok'",
                    ));
                }
                let response_trial_id = response
                    .get("trial_id")
                    .and_then(|value| value.as_u64())
                    .ok_or_else(|| {
                    RemoteError::new_err("Heartbeat response is missing 'trial_id'")
                })?;
                if response_trial_id != trial_id {
                    return Err(RemoteError::new_err(format!(
                        "Heartbeat response returned trial_id {response_trial_id} instead of {trial_id}"
                    )));
                }
                response
                    .get("lease_expires_at_ms")
                    .and_then(|value| value.as_u64())
                    .ok_or_else(|| {
                        RemoteError::new_err("Heartbeat response is missing 'lease_expires_at_ms'")
                    })
            }
        }
    }

    /// Get the top-k trials by rank.
    #[pyo3(signature = (k, include_infeasible=false))]
    fn top_k(&self, py: Python<'_>, k: usize, include_infeasible: bool) -> PyResult<Py<PyList>> {
        let runtime = shared_runtime()?;
        match &self.inner {
            StudyInner::Local { engine } => {
                let trials = py.detach(|| runtime.block_on(engine.top_k(k, include_infeasible)));
                completed_vec_to_pylist(py, &trials)
            }
            StudyInner::Remote { url, http } => {
                let resp: Vec<serde_json::Value> = py
                    .detach(|| {
                        runtime.block_on(checked_json(http.get(format!(
                            "{url}/api/top_k?k={k}&include_infeasible={include_infeasible}"
                        ))))
                    })
                    .map_err(RemoteError::new_err)?;
                json_vec_to_completed_pylist(py, resp)
            }
        }
    }

    /// Get trials on a specific Pareto front.
    #[pyo3(signature = (front=0, include_infeasible=false))]
    fn pareto_front(
        &self,
        py: Python<'_>,
        front: usize,
        include_infeasible: bool,
    ) -> PyResult<Py<PyList>> {
        let runtime = shared_runtime()?;
        match &self.inner {
            StudyInner::Local { engine } => {
                let trials =
                    py.detach(|| runtime.block_on(engine.pareto_front(front, include_infeasible)));
                completed_vec_to_pylist(py, &trials)
            }
            StudyInner::Remote { url, http } => {
                let resp: Vec<serde_json::Value> = py
                    .detach(|| {
                        runtime.block_on(checked_json(http.get(format!(
                                    "{url}/api/pareto_front?front={front}&include_infeasible={include_infeasible}"
                                ))))
                    })
                    .map_err(RemoteError::new_err)?;
                json_vec_to_completed_pylist(py, resp)
            }
        }
    }

    /// Get all trials with scoring and ranking.
    #[pyo3(signature = (sorted_by="index", include_infeasible=true))]
    fn trials(
        &self,
        py: Python<'_>,
        sorted_by: &str,
        include_infeasible: bool,
    ) -> PyResult<Py<PyList>> {
        // Own the sort key so the detach closure does not borrow the Python str.
        let sorted_by = sorted_by.to_string();
        let runtime = shared_runtime()?;
        match &self.inner {
            StudyInner::Local { engine } => {
                let trials =
                    py.detach(|| runtime.block_on(engine.trials(&sorted_by, include_infeasible)));
                completed_vec_to_pylist(py, &trials)
            }
            StudyInner::Remote { url, http } => {
                let resp: Vec<serde_json::Value> = py
                    .detach(|| {
                        runtime.block_on(checked_json(http.get(format!(
                                    "{url}/api/trials?sorted_by={sorted_by}&include_infeasible={include_infeasible}"
                                ))))
                    })
                    .map_err(RemoteError::new_err)?;
                json_vec_to_completed_pylist(py, resp)
            }
        }
    }

    /// Number of completed trials.
    fn trial_count(&self, py: Python<'_>) -> PyResult<usize> {
        let runtime = shared_runtime()?;
        match &self.inner {
            StudyInner::Local { engine } => {
                Ok(py.detach(|| runtime.block_on(engine.trial_count())))
            }
            StudyInner::Remote { url, http } => {
                let resp: serde_json::Value = py
                    .detach(|| {
                        runtime.block_on(checked_json(http.get(format!("{url}/api/trial_count"))))
                    })
                    .map_err(RemoteError::new_err)?;
                resp.get("trial_count")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize)
                    .ok_or_else(|| RemoteError::new_err("Missing 'trial_count' in server response"))
            }
        }
    }

    /// Update objectives mid-run, re-scalarizing all trials.
    fn update_objectives(&self, py: Python<'_>, objectives: &Bound<'_, PyList>) -> PyResult<()> {
        // Convert objectives to owned Rust configs before releasing the GIL.
        let obj_configs = extract_objectives(objectives)?;
        let runtime = shared_runtime()?;
        match &self.inner {
            StudyInner::Local { engine } => py
                .detach(|| runtime.block_on(engine.update_objectives(obj_configs)))
                .map_err(ConfigurationError::new_err),
            StudyInner::Remote { url, http } => py
                .detach(|| {
                    runtime.block_on(async {
                        let acknowledgement: serde_json::Value = checked_json(
                            http.patch(format!("{url}/api/objectives"))
                                .json(&serde_json::json!({ "objectives": obj_configs })),
                        )
                        .await?;
                        validate_status_ok(&acknowledgement, "Objective update")?;
                        acknowledgement
                            .get("rescalarized_trials")
                            .and_then(serde_json::Value::as_u64)
                            .ok_or_else(|| {
                                "Objective update acknowledgement is missing rescalarized_trials"
                                    .to_string()
                            })?;
                        Ok::<(), String>(())
                    })
                })
                .map_err(RemoteError::new_err),
        }
    }

    /// Save a checkpoint to disk.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        match &self.inner {
            StudyInner::Local { engine } => {
                // Own the path so the detach closure does not borrow the Python str.
                let path = path.to_string();
                let runtime = shared_runtime()?;
                py.detach(|| runtime.block_on(engine.save(&path)))
                    .map_err(|e| CheckpointError::new_err(format!("Save failed: {e}")))
            }
            StudyInner::Remote { .. } => Err(ConfigurationError::new_err(
                "save() is only available for local studies, not remote connections",
            )),
        }
    }

    /// Run an objective function for n_trials, automating the ask/tell loop.
    ///
    /// Args:
    ///     func: objective function mapping params dict -> metrics dict
    ///     n_trials: total number of trials to run
    ///     n_workers: number of worker threads (default: 1 = sequential).
    ///         Workers run on a Python ThreadPoolExecutor and share one
    ///         interpreter, so the GIL still applies: n_workers > 1 only
    ///         overlaps objectives that are I/O-bound or that release the GIL
    ///         themselves (e.g. NumPy/PyTorch kernels, native extensions,
    ///         subprocesses). A CPU-bound pure-Python objective is still
    ///         serialized by the GIL and will not speed up with more workers;
    ///         for that case use a process pool around the ask/tell loop.
    ///         The engine's own ask/tell/scoring work releases the GIL, so the
    ///         engine itself is not a serialization bottleneck.
    ///
    /// Returns self so you can chain: study.run(func, 100).top_k(3)
    #[pyo3(signature = (func, n_trials, n_workers=1))]
    fn run(
        slf: Py<Self>,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        n_trials: usize,
        n_workers: usize,
    ) -> PyResult<Py<Self>> {
        if n_trials == 0 {
            return Ok(slf);
        }
        // Never create more executor threads than there is work to perform.
        let n_workers = n_workers.clamp(1, n_trials);

        if n_workers <= 1 {
            // Sequential path: no thread pool overhead
            for _ in 0..n_trials {
                let trial = {
                    let study = slf.borrow(py);
                    study.ask(py)?
                };
                let trial_id = trial.trial_id;

                // Evaluate the objective and report the result. If the objective
                // raises (or returns a non-dict), cancel the trial we just asked
                // for so it does not linger in the engine's pending set or consume
                // exploration budget, then propagate the error.
                let outcome = (|| -> PyResult<()> {
                    let result = func.call1((trial.params,))?;
                    let metrics_dict = result.cast::<PyDict>().map_err(|_| {
                        ObjectiveError::new_err("Objective function must return a dict")
                    })?;
                    slf.borrow(py).tell_for_run(py, trial_id, metrics_dict)?;
                    Ok(())
                })();
                if let Err(e) = outcome {
                    let _ = slf.borrow(py).cancel(py, trial_id);
                    // Preserve the objective error, but leave every earlier
                    // committed trial with a fully ranked retry receipt.
                    let _ = slf.borrow(py).finalize_run_rankings(py);
                    return Err(e);
                }
            }
            slf.borrow(py).finalize_run_rankings(py)?;
        } else {
            // Parallel path: use Python's concurrent.futures.ThreadPoolExecutor
            let cf = py.import("concurrent.futures")?;
            let executor = cf.getattr("ThreadPoolExecutor")?.call1((n_workers,))?;

            // Trials asked for but not yet told. If the objective raises partway
            // through, these are cancelled so the engine does not leak pending
            // trials or mis-count its exploration budget.
            let mut outstanding: Vec<u64> = Vec::new();

            let outcome = (|| -> PyResult<()> {
                let mut pending: Vec<(u64, Py<PyAny>)> = Vec::with_capacity(n_workers);
                let mut submitted = 0usize;
                let mut deferred_ask_error: Option<PyErr> = None;

                // Keep at most n_workers evaluations in flight. Recording the
                // id before submit ensures a rare submit failure still cancels
                // the trial that ask() already reserved in the engine.
                let submit_one = |pending: &mut Vec<(u64, Py<PyAny>)>,
                                  outstanding: &mut Vec<u64>|
                 -> PyResult<()> {
                    let trial = slf.borrow(py).ask(py)?;
                    let trial_id = trial.trial_id;
                    outstanding.push(trial_id);
                    let future = executor.call_method1("submit", (func, trial.params))?;
                    pending.push((trial_id, future.unbind()));
                    Ok(())
                };

                for _ in 0..n_workers {
                    match submit_one(&mut pending, &mut outstanding) {
                        Ok(()) => submitted += 1,
                        Err(error) => {
                            deferred_ask_error = Some(error);
                            break;
                        }
                    }
                }

                while !pending.is_empty() {
                    // `as_completed` is rebuilt from the current in-flight set
                    // and advanced once. This waits for whichever objective
                    // finishes first, rather than blocking on submission order.
                    let futures = PyList::empty(py);
                    for (_, future) in &pending {
                        futures.append(future.bind(py))?;
                    }
                    let finished = cf
                        .getattr("as_completed")?
                        .call1((futures,))?
                        .call_method0("__next__")?;
                    let finished_index = pending
                        .iter()
                        .position(|(_, future)| future.bind(py).is(&finished))
                        .ok_or_else(|| {
                            HolaError::new_err("Executor returned a future that was not in flight")
                        })?;
                    let trial_id = pending[finished_index].0;
                    let result = finished.call_method0("result")?;
                    let metrics_dict = result.cast::<PyDict>().map_err(|_| {
                        ObjectiveError::new_err("Objective function must return a dict")
                    })?;
                    slf.borrow(py).tell_for_run(py, trial_id, metrics_dict)?;
                    pending.swap_remove(finished_index);
                    outstanding.retain(|&id| id != trial_id);

                    // Refill immediately after tell(), so later suggestions can
                    // adapt to this just-completed observation and fast workers
                    // do not wait for the slowest member of a fixed batch.
                    if submitted < n_trials && deferred_ask_error.is_none() {
                        match submit_one(&mut pending, &mut outstanding) {
                            Ok(()) => submitted += 1,
                            Err(error) => deferred_ask_error = Some(error),
                        }
                    }
                }
                // An ask can be rejected while already-submitted objectives are
                // still completing (for example, at max_trials). Drain those
                // futures first so an objective exception is not masked by the
                // speculative refill error; surface the ask error only when all
                // in-flight evaluations completed successfully.
                if let Some(error) = deferred_ask_error {
                    return Err(error);
                }
                Ok(())
            })();

            // Cancel any still-outstanding trials on failure (the engine-side
            // cancel only frees pending bookkeeping), then always shut the
            // executor down. shutdown(cancel_futures=True) drops queued tasks
            // that have not started yet and waits for already-running
            // evaluations to finish before returning.
            if outcome.is_err() {
                for trial_id in &outstanding {
                    let _ = slf.borrow(py).cancel(py, *trial_id);
                }
            }
            let shutdown_kwargs = PyDict::new(py);
            let _ = shutdown_kwargs.set_item("cancel_futures", true);
            let shutdown = executor.call_method("shutdown", (), Some(&shutdown_kwargs));

            // On the failure path, surface the original objective error and
            // ignore any shutdown error. On the success path, propagate a
            // shutdown error so it is not silently swallowed.
            if let Err(error) = outcome {
                // Finalization is best-effort on failure so the original
                // objective/executor error remains the one Python observes.
                let _ = slf.borrow(py).finalize_run_rankings(py);
                return Err(error);
            }
            let finalization = slf.borrow(py).finalize_run_rankings(py);
            shutdown?;
            finalization?;
        }

        Ok(slf)
    }

    /// Start a REST server for this study.
    ///
    /// Clones the engine (cheap: shared state via Arc) and starts an HTTP
    /// server. Both local calls and remote HTTP requests share the same
    /// leaderboard and strategy state.
    ///
    /// Args:
    ///     port: listen port (default: 8000)
    ///     background: if True, runs in background thread and returns immediately.
    ///         The study remains usable for local ask/tell while serving.
    ///         If False (default), blocks until the server is stopped.
    #[pyo3(signature = (port=8000, background=false, dashboard_path=None))]
    fn serve(
        &self,
        py: Python<'_>,
        port: u16,
        background: bool,
        dashboard_path: Option<String>,
    ) -> PyResult<()> {
        match &self.inner {
            StudyInner::Local { engine } => {
                let engine_clone = engine.clone();
                let dash = dashboard_path.map(std::path::PathBuf::from);
                let runtime = shared_runtime()?;
                if background {
                    // Detach the server task onto the process-wide runtime. The
                    // static runtime owns its lifecycle, so neither a dedicated
                    // caller thread nor another Tokio worker pool is needed.
                    std::mem::drop(runtime.spawn(async move {
                        if let Err(e) =
                            hola_engine::server::serve(engine_clone, port, dash.as_deref()).await
                        {
                            eprintln!("HOLA server error: {e}");
                        }
                    }));
                    Ok(())
                } else {
                    // Block the current thread until the server is stopped.
                    // Release the GIL for the server's whole lifetime so other
                    // Python threads keep running; only owned values are moved in.
                    // `serve` returns `Box<dyn Error>`, which is not `Send`, so
                    // stringify the error inside the closure (its return value
                    // must cross the GIL-release boundary as a `Send` type).
                    py.detach(|| {
                        runtime
                            .block_on(hola_engine::server::serve(
                                engine_clone,
                                port,
                                dash.as_deref(),
                            ))
                            .map_err(|e| e.to_string())
                    })
                    .map_err(|e| HolaError::new_err(format!("Server error: {e}")))
                }
            }
            StudyInner::Remote { .. } => Err(ConfigurationError::new_err(
                "serve() is only available for local studies, not remote connections",
            )),
        }
    }
}

impl Study {
    /// Internal completion path for `run()`. A local run does not expose the
    /// per-trial `CompletedTrial`, so the engine can defer leaderboard-wide
    /// ranking and materialize all receipts once at batch exit. Remote studies
    /// retain the ordinary HTTP `tell` contract because no private batch
    /// endpoint exists.
    fn tell_for_run(
        &self,
        py: Python<'_>,
        trial_id: u64,
        metrics: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        match &self.inner {
            StudyInner::Local { engine } => {
                let raw = py_dict_to_json(metrics)?;
                let runtime = shared_runtime()?;
                py.detach(|| runtime.block_on(engine.tell_without_ranking(trial_id, raw)))
                    .map_err(ObjectiveError::new_err)
            }
            StudyInner::Remote { .. } => self.tell(py, trial_id, metrics).map(|_| ()),
        }
    }

    fn finalize_run_rankings(&self, py: Python<'_>) -> PyResult<()> {
        match &self.inner {
            StudyInner::Local { engine } => {
                let runtime = shared_runtime()?;
                py.detach(|| runtime.block_on(engine.finalize_deferred_rankings()))
                    .map_err(HolaError::new_err)
            }
            StudyInner::Remote { .. } => Ok(()),
        }
    }
}

// =============================================================================
// Helper: convert Vec<CompletedTrial> to Python list
// =============================================================================

fn completed_vec_to_pylist(
    py: Python<'_>,
    trials: &[hola_engine::hola_engine::CompletedTrial],
) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);
    for ct in trials {
        let py_ct = rust_to_py_completed(py, ct)?;
        list.append(Py::new(py, py_ct)?)?;
    }
    Ok(list.into())
}

fn json_vec_to_completed_pylist(
    py: Python<'_>,
    items: Vec<serde_json::Value>,
) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);
    // Consume each Value via into_iter() so deserialization does not clone.
    for item in items {
        let ct: hola_engine::hola_engine::CompletedTrial = serde_json::from_value(item)
            .map_err(|e| RemoteError::new_err(format!("Deserialization error: {e}")))?;
        let py_ct = rust_to_py_completed(py, &ct)?;
        list.append(Py::new(py, py_ct)?)?;
    }
    Ok(list.into())
}

// =============================================================================
// JSON <-> Python conversion helpers
// =============================================================================

#[derive(Clone, Copy)]
enum JsonStringMode {
    /// Preserve every JSON string exactly as supplied by the user/server.
    Literal,
    /// Decode the engine's non-finite sentinels in numeric score fields.
    DecodeNonFinite,
}

fn json_to_py(
    py: Python<'_>,
    val: &serde_json::Value,
    string_mode: JsonStringMode,
) -> PyResult<Py<PyAny>> {
    match val {
        serde_json::Value::Null => Ok(py.None().into_pyobject(py)?.unbind()),
        serde_json::Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(u) = n.as_u64() {
                // u64 values larger than i64::MAX do not fit in i64; route them
                // through u64 so the integer precision is preserved.
                Ok(u.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s)
            if matches!(string_mode, JsonStringMode::DecodeNonFinite) && s == "inf" =>
        {
            Ok(f64::INFINITY.into_pyobject(py)?.into_any().unbind())
        }
        serde_json::Value::String(s)
            if matches!(string_mode, JsonStringMode::DecodeNonFinite) && s == "-inf" =>
        {
            Ok(f64::NEG_INFINITY.into_pyobject(py)?.into_any().unbind())
        }
        serde_json::Value::String(s)
            if matches!(string_mode, JsonStringMode::DecodeNonFinite) && s == "nan" =>
        {
            Ok(f64::NAN.into_pyobject(py)?.into_any().unbind())
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item, string_mode)?)?;
            }
            Ok(list.into())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v, string_mode)?)?;
            }
            Ok(dict.into())
        }
    }
}

fn py_dict_to_json(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (key, val) in dict.iter() {
        let k: String = key
            .extract()
            .map_err(|_| ObjectiveError::new_err("Objective metrics keys must be strings"))?;
        let v = py_to_json(&val)?;
        map.insert(k, v);
    }
    Ok(serde_json::Value::Object(map))
}

fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::json!(i))
    } else if let Ok(u) = obj.extract::<u64>() {
        // Integers larger than i64::MAX do not fit in i64; encode them as u64 so
        // they survive as JSON integers instead of falling through to the lossy
        // f64 branch below. json_to_py decodes the u64 arm symmetrically.
        Ok(serde_json::json!(u))
    } else if let Ok(f) = obj.extract::<f64>() {
        // serde_json cannot represent non-finite f64 as a JSON number (it would
        // silently become null), so encode using the engine's string convention
        // ("inf", "-inf", "nan"). Raw metrics preserve those JSON strings on
        // readback because the protocol cannot distinguish them from literal
        // user strings; only computed numeric score fields decode sentinels.
        if f.is_finite() {
            Ok(serde_json::json!(f))
        } else if f.is_nan() {
            Ok(serde_json::Value::from("nan"))
        } else if f > 0.0 {
            Ok(serde_json::Value::from("inf"))
        } else {
            Ok(serde_json::Value::from("-inf"))
        }
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        py_dict_to_json(dict)
    } else if let Ok(list) = obj.cast::<PyList>() {
        let arr: Result<Vec<_>, _> = list.iter().map(|item| py_to_json(&item)).collect();
        Ok(serde_json::Value::Array(arr?))
    } else {
        Err(ObjectiveError::new_err(format!(
            "Cannot convert Python object to JSON: {:?}",
            obj.get_type().name()?
        )))
    }
}

// =============================================================================
// Module
// =============================================================================

#[pymodule]
fn hola_opt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("HolaError", py.get_type::<HolaError>())?;
    m.add("ConfigurationError", py.get_type::<ConfigurationError>())?;
    m.add("CheckpointError", py.get_type::<CheckpointError>())?;
    m.add("RemoteError", py.get_type::<RemoteError>())?;
    m.add("ObjectiveError", py.get_type::<ObjectiveError>())?;
    m.add_class::<Real>()?;
    m.add_class::<Integer>()?;
    m.add_class::<Categorical>()?;
    m.add_class::<Minimize>()?;
    m.add_class::<Maximize>()?;
    m.add_class::<Gmm>()?;
    m.add_class::<Sobol>()?;
    m.add_class::<Random>()?;
    m.add_class::<Space>()?;
    m.add_class::<Study>()?;
    m.add_class::<Trial>()?;
    m.add_class::<CompletedTrial>()?;
    Ok(())
}
