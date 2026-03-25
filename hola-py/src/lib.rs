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
    HolaEngine, ObjectiveConfig, ParamConfig, StrategyConfig, StudyConfig,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeMap;

// =============================================================================
// Space building helpers
// =============================================================================

/// Real-valued parameter with configurable scale.
///
/// Args:
///     min: Lower bound (in actual values, not exponents).
///     max: Upper bound (in actual values, not exponents).
///     scale: Sampling scale — "linear" (default), "log" (natural log), or "log10".
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
            "linear" | "log" | "log10" => {}
            _ => {
                return Err(PyValueError::new_err(
                    "scale must be \"linear\", \"log\", or \"log10\"",
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
///     target: "Good enough" value — at or below this, the TLP score is 0.
///     limit: "Unacceptable" value — beyond this, the trial is infeasible (score = inf).
///     priority: Per-objective weight/slope P_i in the TLP formula:
///         φ_i = P_i × (value − target) / (limit − target). Default 1.0.
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
///     target: "Good enough" value — at or above this, the TLP score is 0.
///     limit: "Unacceptable" value — below this, the trial is infeasible (score = inf).
///     priority: Per-objective weight/slope P_i in the TLP formula. Default 1.0.
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
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Gmm {
    #[pyo3(get)]
    refit_interval: Option<usize>,
    #[pyo3(get)]
    elite_fraction: Option<f64>,
    #[pyo3(get)]
    exploration_budget: Option<usize>,
}

#[pymethods]
impl Gmm {
    #[new]
    #[pyo3(signature = (refit_interval=None, elite_fraction=None, exploration_budget=None))]
    fn new(
        refit_interval: Option<usize>,
        elite_fraction: Option<f64>,
        exploration_budget: Option<usize>,
    ) -> PyResult<Self> {
        if let Some(ef) = elite_fraction
            && (ef <= 0.0 || ef > 1.0)
        {
            return Err(PyValueError::new_err(
                "elite_fraction must be between 0.0 (exclusive) and 1.0 (inclusive)",
            ));
        }
        if let Some(ri) = refit_interval
            && ri == 0
        {
            return Err(PyValueError::new_err("refit_interval must be at least 1"));
        }
        Ok(Self {
            refit_interval,
            elite_fraction,
            exploration_budget,
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
        return Ok(ParamConfig::Continuous {
            min: r.min,
            max: r.max,
            scale: r.scale,
        });
    }
    if let Ok(d) = obj.extract::<Integer>() {
        return Ok(ParamConfig::Discrete {
            min: d.min,
            max: d.max,
        });
    }
    if let Ok(c) = obj.extract::<Categorical>() {
        return Ok(ParamConfig::Categorical { choices: c.choices });
    }
    Err(PyValueError::new_err(
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
        params: json_to_py(py, &ct.params)?,
        metrics: json_to_py(py, &ct.metrics)?,
        scores: json_to_py(py, &ct.scores)?,
        score_vector: json_to_py(py, &ct.score_vector)?,
        rank: ct.rank,
        pareto_front: ct.pareto_front,
        completed_at: ct.completed_at,
    })
}

// =============================================================================
// Study: the main user-facing class
// =============================================================================

/// Internal representation: local engine or remote HTTP client.
enum StudyInner {
    Local {
        engine: HolaEngine,
        runtime: tokio::runtime::Runtime,
    },
    Remote {
        url: String,
        client: reqwest::Client,
        runtime: tokio::runtime::Runtime,
    },
}

/// The main optimization study.
///
/// Usage:
///     study = Study(space=Space(lr=Real(1e-4, 0.1, scale="log10")), objectives=[Minimize("loss")])
///     trial = study.ask()
///     ct = study.tell(trial.trial_id, {"loss": 0.42})
///     top = study.top_k(3)
#[pyclass(skip_from_py_object)]
struct Study {
    inner: StudyInner,
}

fn extract_objectives(objectives: &Bound<'_, PyList>) -> PyResult<Vec<ObjectiveConfig>> {
    if objectives.len() == 0 {
        return Err(PyValueError::new_err(
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
            return Err(PyValueError::new_err(
                "Objectives must be Minimize or Maximize instances",
            ));
        }
    }
    Ok(obj_configs)
}

#[pymethods]
impl Study {
    #[new]
    #[pyo3(signature = (space, objectives, strategy=None, seed=None, max_trials=None))]
    fn new(
        space: Space,
        objectives: &Bound<'_, PyList>,
        strategy: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
        max_trials: Option<usize>,
    ) -> PyResult<Self> {
        let obj_configs = extract_objectives(objectives)?;

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
            },
            Some(s) if s.extract::<Gmm>().is_ok() => {
                let gmm = s.extract::<Gmm>()?;
                StrategyConfig {
                    strategy_type: "gmm".to_string(),
                    refit_interval: gmm.refit_interval.unwrap_or(20),
                    total_budget: max_trials,
                    exploration_budget: gmm.exploration_budget,
                    seed,
                    elite_fraction: gmm.elite_fraction,
                }
            }
            Some(s) if s.extract::<Sobol>().is_ok() => StrategyConfig {
                strategy_type: "sobol".to_string(),
                refit_interval: 20,
                total_budget: max_trials,
                exploration_budget: None,
                seed,
                elite_fraction: None,
            },
            Some(s) if s.extract::<Random>().is_ok() => StrategyConfig {
                strategy_type: "random".to_string(),
                refit_interval: 20,
                total_budget: max_trials,
                exploration_budget: None,
                seed,
                elite_fraction: None,
            },
            Some(s) => {
                let name: String = s.extract().map_err(|_| {
                    PyValueError::new_err(
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
                }
            }
        };

        let config = StudyConfig {
            space: space.params,
            objectives: obj_configs,
            strategy: Some(strategy_config),
            checkpoint: None,
            max_trials,
        };

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {e}")))?;

        let engine = HolaEngine::from_config(config)
            .map_err(|e| PyValueError::new_err(format!("Failed to create engine: {e}")))?;

        Ok(Self {
            inner: StudyInner::Local { engine, runtime },
        })
    }

    /// Connect to an existing HOLA server.
    #[staticmethod]
    fn connect(url: &str) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {e}")))?;
        let client = reqwest::Client::new();
        Ok(Self {
            inner: StudyInner::Remote {
                url: url.trim_end_matches('/').to_string(),
                client,
                runtime,
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
    fn load(path: &str) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {e}")))?;

        let engine = runtime
            .block_on(HolaEngine::load_from_checkpoint(path))
            .map_err(|e| PyValueError::new_err(format!("Failed to load checkpoint: {e}")))?;

        Ok(Self {
            inner: StudyInner::Local { engine, runtime },
        })
    }

    /// Request the next trial to evaluate.
    fn ask(&self, py: Python<'_>) -> PyResult<Trial> {
        match &self.inner {
            StudyInner::Local { engine, runtime } => {
                let dyn_trial = runtime
                    .block_on(engine.ask())
                    .map_err(PyValueError::new_err)?;
                let params = json_to_py(py, &dyn_trial.params)?;
                Ok(Trial {
                    trial_id: dyn_trial.trial_id,
                    params,
                })
            }
            StudyInner::Remote {
                url,
                client,
                runtime,
            } => {
                let resp: serde_json::Value = runtime
                    .block_on(async {
                        client
                            .post(format!("{url}/api/ask"))
                            .send()
                            .await
                            .map_err(|e| format!("HTTP error: {e}"))?
                            .json()
                            .await
                            .map_err(|e| format!("JSON error: {e}"))
                    })
                    .map_err(PyValueError::new_err)?;

                let trial_id = resp
                    .get("trial_id")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        PyValueError::new_err("Missing 'trial_id' in server response")
                    })?;
                let params_json = resp
                    .get("params")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let params = json_to_py(py, &params_json)?;
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
        let raw = py_dict_to_json(metrics)?;
        match &self.inner {
            StudyInner::Local { engine, runtime } => {
                let completed = runtime
                    .block_on(engine.tell(trial_id, raw))
                    .map_err(PyValueError::new_err)?;
                rust_to_py_completed(py, &completed)
            }
            StudyInner::Remote {
                url,
                client,
                runtime,
            } => {
                // Remote tell returns lightweight response, so we tell then
                // fetch the trial's details via top_k.
                runtime
                    .block_on(async {
                        let resp = client
                            .post(format!("{url}/api/tell"))
                            .json(&serde_json::json!({
                                "trial_id": trial_id,
                                "metrics": raw,
                            }))
                            .send()
                            .await
                            .map_err(|e| format!("HTTP error: {e}"))?;

                        if !resp.status().is_success() {
                            let body = resp
                                .text()
                                .await
                                .unwrap_or_else(|_| "unknown error".to_string());
                            return Err(format!("Server error: {body}"));
                        }
                        Ok(())
                    })
                    .map_err(PyValueError::new_err)?;

                // Fetch the trial details from the server
                let trials_resp: Vec<serde_json::Value> = runtime
                    .block_on(async {
                        client
                            .get(format!(
                                "{url}/api/trials?sorted_by=index&include_infeasible=true"
                            ))
                            .send()
                            .await
                            .map_err(|e| format!("HTTP error: {e}"))?
                            .json()
                            .await
                            .map_err(|e| format!("JSON error: {e}"))
                    })
                    .map_err(PyValueError::new_err)?;

                // Find the trial we just told
                let trial_json = trials_resp
                    .iter()
                    .find(|t| t.get("trial_id").and_then(|v| v.as_u64()) == Some(trial_id))
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "Trial {trial_id} not found in server response"
                        ))
                    })?;

                let ct: hola_engine::hola_engine::CompletedTrial =
                    serde_json::from_value(trial_json.clone()).map_err(|e| {
                        PyValueError::new_err(format!("Deserialization error: {e}"))
                    })?;
                rust_to_py_completed(py, &ct)
            }
        }
    }

    /// Cancel a pending trial.
    fn cancel(&self, trial_id: u64) -> PyResult<()> {
        match &self.inner {
            StudyInner::Local { engine, runtime } => runtime
                .block_on(engine.cancel(trial_id))
                .map_err(PyValueError::new_err),
            StudyInner::Remote {
                url,
                client,
                runtime,
            } => runtime
                .block_on(async {
                    let resp = client
                        .post(format!("{url}/api/cancel"))
                        .json(&serde_json::json!({ "trial_id": trial_id }))
                        .send()
                        .await
                        .map_err(|e| format!("HTTP error: {e}"))?;

                    if !resp.status().is_success() {
                        let body = resp
                            .text()
                            .await
                            .unwrap_or_else(|_| "unknown error".to_string());
                        return Err(format!("Server error: {body}"));
                    }
                    Ok(())
                })
                .map_err(PyValueError::new_err),
        }
    }

    /// Get the top-k trials by rank.
    #[pyo3(signature = (k, include_infeasible=false))]
    fn top_k(&self, py: Python<'_>, k: usize, include_infeasible: bool) -> PyResult<Py<PyList>> {
        match &self.inner {
            StudyInner::Local { engine, runtime } => {
                let trials = runtime.block_on(engine.top_k(k, include_infeasible));
                completed_vec_to_pylist(py, &trials)
            }
            StudyInner::Remote {
                url,
                client,
                runtime,
            } => {
                let resp: Vec<serde_json::Value> = runtime
                    .block_on(async {
                        client
                            .get(format!(
                                "{url}/api/top_k?k={k}&include_infeasible={include_infeasible}"
                            ))
                            .send()
                            .await
                            .map_err(|e| format!("HTTP error: {e}"))?
                            .json()
                            .await
                            .map_err(|e| format!("JSON error: {e}"))
                    })
                    .map_err(PyValueError::new_err)?;
                json_vec_to_completed_pylist(py, &resp)
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
        match &self.inner {
            StudyInner::Local { engine, runtime } => {
                let trials = runtime.block_on(engine.pareto_front(front, include_infeasible));
                completed_vec_to_pylist(py, &trials)
            }
            StudyInner::Remote {
                url,
                client,
                runtime,
            } => {
                let resp: Vec<serde_json::Value> = runtime
                    .block_on(async {
                        client
                            .get(format!(
                                "{url}/api/pareto_front?front={front}&include_infeasible={include_infeasible}"
                            ))
                            .send()
                            .await
                            .map_err(|e| format!("HTTP error: {e}"))?
                            .json()
                            .await
                            .map_err(|e| format!("JSON error: {e}"))
                    })
                    .map_err(PyValueError::new_err)?;
                json_vec_to_completed_pylist(py, &resp)
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
        match &self.inner {
            StudyInner::Local { engine, runtime } => {
                let trials = runtime.block_on(engine.trials(sorted_by, include_infeasible));
                completed_vec_to_pylist(py, &trials)
            }
            StudyInner::Remote {
                url,
                client,
                runtime,
            } => {
                let resp: Vec<serde_json::Value> = runtime
                    .block_on(async {
                        client
                            .get(format!(
                                "{url}/api/trials?sorted_by={sorted_by}&include_infeasible={include_infeasible}"
                            ))
                            .send()
                            .await
                            .map_err(|e| format!("HTTP error: {e}"))?
                            .json()
                            .await
                            .map_err(|e| format!("JSON error: {e}"))
                    })
                    .map_err(PyValueError::new_err)?;
                json_vec_to_completed_pylist(py, &resp)
            }
        }
    }

    /// Number of completed trials.
    fn trial_count(&self) -> PyResult<usize> {
        match &self.inner {
            StudyInner::Local { engine, runtime } => Ok(runtime.block_on(engine.trial_count())),
            StudyInner::Remote {
                url,
                client,
                runtime,
            } => {
                let resp: serde_json::Value = runtime
                    .block_on(async {
                        client
                            .get(format!("{url}/api/trial_count"))
                            .send()
                            .await
                            .map_err(|e| format!("HTTP error: {e}"))?
                            .json()
                            .await
                            .map_err(|e| format!("JSON error: {e}"))
                    })
                    .map_err(PyValueError::new_err)?;
                resp.get("trial_count")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize)
                    .ok_or_else(|| {
                        PyValueError::new_err("Missing 'trial_count' in server response")
                    })
            }
        }
    }

    /// Update objectives mid-run, re-scalarizing all trials.
    fn update_objectives(&self, objectives: &Bound<'_, PyList>) -> PyResult<()> {
        let obj_configs = extract_objectives(objectives)?;
        match &self.inner {
            StudyInner::Local { engine, runtime } => {
                runtime.block_on(engine.update_objectives(obj_configs));
                Ok(())
            }
            StudyInner::Remote {
                url,
                client,
                runtime,
            } => runtime
                .block_on(async {
                    let resp = client
                        .patch(format!("{url}/api/objectives"))
                        .json(&serde_json::json!({ "objectives": obj_configs }))
                        .send()
                        .await
                        .map_err(|e| format!("HTTP error: {e}"))?;
                    if !resp.status().is_success() {
                        let body = resp
                            .text()
                            .await
                            .unwrap_or_else(|_| "unknown error".to_string());
                        return Err(format!("Server error: {body}"));
                    }
                    Ok(())
                })
                .map_err(PyValueError::new_err),
        }
    }

    /// Save a checkpoint to disk.
    fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            StudyInner::Local { engine, runtime } => runtime
                .block_on(engine.save(path))
                .map_err(|e| PyValueError::new_err(format!("Save failed: {e}"))),
            StudyInner::Remote { .. } => Err(PyValueError::new_err(
                "save() is only available for local studies, not remote connections",
            )),
        }
    }

    /// Run an objective function for n_trials, automating the ask/tell loop.
    ///
    /// Args:
    ///     func: objective function mapping params dict -> metrics dict
    ///     n_trials: total number of trials to run
    ///     n_workers: number of parallel workers (default: 1 = sequential)
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
        let n_workers = n_workers.max(1);

        if n_workers <= 1 {
            // Sequential path — no thread pool overhead
            for _ in 0..n_trials {
                let trial = {
                    let study = slf.borrow(py);
                    study.ask(py)?
                };

                let result = func.call1((trial.params,))?;

                let metrics_dict = result
                    .cast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("Objective function must return a dict"))?;
                let study = slf.borrow(py);
                study.tell(py, trial.trial_id, metrics_dict)?;
            }
        } else {
            // Parallel path — use Python's concurrent.futures.ThreadPoolExecutor
            let cf = py.import("concurrent.futures")?;
            let executor = cf.getattr("ThreadPoolExecutor")?.call1((n_workers,))?;

            let mut remaining = n_trials;
            while remaining > 0 {
                let batch_size = remaining.min(n_workers);

                // Ask for a batch of trials
                let mut pending_trials: Vec<Trial> = Vec::with_capacity(batch_size);
                for _ in 0..batch_size {
                    let study = slf.borrow(py);
                    let trial = study.ask(py)?;
                    pending_trials.push(trial);
                }

                // Submit all to executor
                let mut futures: Vec<(u64, Py<PyAny>)> = Vec::with_capacity(batch_size);
                for trial in &pending_trials {
                    let future =
                        executor.call_method1("submit", (func, trial.params.clone_ref(py)))?;
                    futures.push((trial.trial_id, future.unbind()));
                }

                // Collect results and tell
                for (trial_id, future) in futures {
                    let future_bound = future.bind(py);
                    let result = future_bound.call_method0("result")?;

                    let metrics_dict = result.cast::<PyDict>().map_err(|_| {
                        PyValueError::new_err("Objective function must return a dict")
                    })?;
                    let study = slf.borrow(py);
                    study.tell(py, trial_id, metrics_dict)?;
                }

                remaining -= batch_size;
            }

            // Shut down executor
            executor.call_method0("shutdown")?;
        }

        Ok(slf)
    }

    /// Start a REST server for this study.
    ///
    /// Clones the engine (cheap — shared state via Arc) and starts an HTTP
    /// server. Both local calls and remote HTTP requests share the same
    /// leaderboard and strategy state.
    ///
    /// Args:
    ///     port: listen port (default: 8000)
    ///     background: if True, runs in background thread and returns immediately.
    ///         The study remains usable for local ask/tell while serving.
    ///         If False (default), blocks until the server is stopped.
    #[pyo3(signature = (port=8000, background=false, dashboard_path=None))]
    fn serve(&self, port: u16, background: bool, dashboard_path: Option<String>) -> PyResult<()> {
        match &self.inner {
            StudyInner::Local { engine, runtime } => {
                let engine_clone = engine.clone();
                let dash = dashboard_path.map(std::path::PathBuf::from);
                if background {
                    // Spawn server on a new thread with its own tokio runtime.
                    std::thread::spawn(move || {
                        let rt = tokio::runtime::Runtime::new()
                            .expect("Failed to create server runtime");
                        if let Err(e) = rt.block_on(hola_engine::server::serve(
                            engine_clone,
                            port,
                            dash.as_deref(),
                        )) {
                            eprintln!("HOLA server error: {e}");
                        }
                    });
                    Ok(())
                } else {
                    // Block the current thread until the server is stopped.
                    runtime
                        .block_on(hola_engine::server::serve(
                            engine_clone,
                            port,
                            dash.as_deref(),
                        ))
                        .map_err(|e| PyValueError::new_err(format!("Server error: {e}")))
                }
            }
            StudyInner::Remote { .. } => Err(PyValueError::new_err(
                "serve() is only available for local studies, not remote connections",
            )),
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
    items: &[serde_json::Value],
) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);
    for item in items {
        let ct: hola_engine::hola_engine::CompletedTrial = serde_json::from_value(item.clone())
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {e}")))?;
        let py_ct = rust_to_py_completed(py, &ct)?;
        list.append(Py::new(py, py_ct)?)?;
    }
    Ok(list.into())
}

// =============================================================================
// JSON <-> Python conversion helpers
// =============================================================================

fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match val {
        serde_json::Value::Null => Ok(py.None().into_pyobject(py)?.unbind()),
        serde_json::Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) if s == "inf" => {
            Ok(f64::INFINITY.into_pyobject(py)?.into_any().unbind())
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into())
        }
    }
}

fn py_dict_to_json(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (key, val) in dict.iter() {
        let k: String = key.extract()?;
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
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        py_dict_to_json(dict)
    } else if let Ok(list) = obj.cast::<PyList>() {
        let arr: Result<Vec<_>, _> = list.iter().map(|item| py_to_json(&item)).collect();
        Ok(serde_json::Value::Array(arr?))
    } else {
        Err(PyValueError::new_err(format!(
            "Cannot convert Python object to JSON: {:?}",
            obj.get_type().name()?
        )))
    }
}

// =============================================================================
// Module
// =============================================================================

#[pymodule]
fn hola(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
