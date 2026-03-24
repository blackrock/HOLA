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

//! Type-erased engine boundary for dynamic frontends (Python, REST, CLI).
//!
//! [`HolaEngine`] hides the static generic types behind a flat JSON interface
//! so that non-Rust callers can interact with the optimizer without knowing
//! the concrete space, strategy, or transformer types at compile time.
//! Parameters are `BTreeMap<String, serde_json::Value>` and metrics are
//! `serde_json::Value`.
//!
//! Use [`Engine`](opt_engine::engine::Engine) when you have concrete Rust types
//! and want full compile-time verification; use `HolaEngine` when the
//! parameter space is defined at runtime (e.g., from a YAML config or
//! Python `dict`).

use opt_engine::leaderboard::{Leaderboard, Trial};
use opt_engine::persistence::{AutoCheckpointConfig, LeaderboardCheckpoint};
use opt_engine::scales::{LinearScale, Log10Scale, LogScale, Scale};
use opt_engine::spaces::{CategoricalSpace, ContinuousSpace, DiscreteSpace};
use opt_engine::strategies::{GmmStrategy, RandomStrategy, SobolStrategy};
use opt_engine::traits::{RefitConfig, SampleSpace, StandardizedSpace, Strategy};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

// =============================================================================
// Parameter metadata for dashboard API
// =============================================================================

/// Metadata describing a single parameter dimension, sent to the dashboard
/// so it can auto-configure axis labels, scales, and choice dropdowns.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParamInfo {
    pub param_type: String, // "continuous", "discrete", or "categorical"
    pub min: f64,
    pub max: f64,
    pub scale: String, // "linear", "log", "log10"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<String>>,
}

// =============================================================================
// DynDimension: closed set of parameter dimension types
// =============================================================================

/// A single parameter dimension in a [`DynSpace`].
///
/// This is a closed enum covering the built-in parameter types. For custom
/// parameter types with full compile-time verification, use the generic
/// [`Engine`](opt_engine::engine::Engine) with the [`SampleSpace`] /
/// [`StandardizedSpace`] traits directly.
enum DynDimension {
    ContinuousLinear(ContinuousSpace<LinearScale>),
    ContinuousLog(ContinuousSpace<LogScale>),
    ContinuousLog10(ContinuousSpace<Log10Scale>),
    Discrete(DiscreteSpace),
    Categorical(CategoricalSpace),
}

#[allow(clippy::wrong_self_convention)] // mirrors `StandardizedSpace::from_unit_cube` on inner spaces
impl DynDimension {
    fn dimensionality(&self) -> usize {
        match self {
            Self::ContinuousLinear(s) => s.dimensionality(),
            Self::ContinuousLog(s) => s.dimensionality(),
            Self::ContinuousLog10(s) => s.dimensionality(),
            Self::Discrete(s) => s.dimensionality(),
            Self::Categorical(s) => s.dimensionality(),
        }
    }

    fn to_unit_cube(&self, val: &serde_json::Value) -> Option<Vec<f64>> {
        match self {
            Self::ContinuousLinear(s) => val.as_f64().map(|v| s.to_unit_cube(&v)),
            Self::ContinuousLog(s) => val.as_f64().map(|v| s.to_unit_cube(&v)),
            Self::ContinuousLog10(s) => val.as_f64().map(|v| s.to_unit_cube(&v)),
            Self::Discrete(s) => val.as_i64().map(|v| s.to_unit_cube(&v)),
            Self::Categorical(s) => val.as_str().map(|v| s.to_unit_cube(&v.to_string())),
        }
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<serde_json::Value> {
        match self {
            Self::ContinuousLinear(s) => s.from_unit_cube(vec).map(serde_json::Value::from),
            Self::ContinuousLog(s) => s.from_unit_cube(vec).map(serde_json::Value::from),
            Self::ContinuousLog10(s) => s.from_unit_cube(vec).map(serde_json::Value::from),
            Self::Discrete(s) => s.from_unit_cube(vec).map(serde_json::Value::from),
            Self::Categorical(s) => s.from_unit_cube(vec).map(serde_json::Value::from),
        }
    }

    fn contains(&self, val: &serde_json::Value) -> bool {
        match self {
            Self::ContinuousLinear(s) => val.as_f64().is_some_and(|v| s.contains(&v)),
            Self::ContinuousLog(s) => val.as_f64().is_some_and(|v| s.contains(&v)),
            Self::ContinuousLog10(s) => val.as_f64().is_some_and(|v| s.contains(&v)),
            Self::Discrete(s) => val.as_i64().is_some_and(|v| s.contains(&v)),
            Self::Categorical(s) => val.as_str().is_some_and(|v| s.contains(&v.to_string())),
        }
    }

    fn clamp(&self, val: &serde_json::Value) -> serde_json::Value {
        match self {
            Self::ContinuousLinear(s) => val
                .as_f64()
                .map(|v| serde_json::Value::from(s.clamp(&v)))
                .unwrap_or_else(|| val.clone()),
            Self::ContinuousLog(s) => val
                .as_f64()
                .map(|v| serde_json::Value::from(s.clamp(&v)))
                .unwrap_or_else(|| val.clone()),
            Self::ContinuousLog10(s) => val
                .as_f64()
                .map(|v| serde_json::Value::from(s.clamp(&v)))
                .unwrap_or_else(|| val.clone()),
            Self::Discrete(s) => val
                .as_i64()
                .map(|v| serde_json::Value::from(s.clamp(&v)))
                .unwrap_or_else(|| val.clone()),
            Self::Categorical(s) => val
                .as_str()
                .map(|v| serde_json::Value::from(s.clamp(&v.to_string())))
                .unwrap_or_else(|| val.clone()),
        }
    }

    fn to_param_config(&self) -> ParamConfig {
        match self {
            Self::ContinuousLinear(s) => ParamConfig::Continuous {
                min: s.min,
                max: s.max,
                scale: "linear".to_string(),
            },
            Self::ContinuousLog(s) => ParamConfig::Continuous {
                min: s.min,
                max: s.max,
                scale: "log".to_string(),
            },
            Self::ContinuousLog10(s) => ParamConfig::Continuous {
                min: s.min,
                max: s.max,
                scale: "log10".to_string(),
            },
            Self::Discrete(s) => ParamConfig::Discrete {
                min: s.min,
                max: s.max,
            },
            Self::Categorical(s) => ParamConfig::Categorical {
                choices: s.choices.clone(),
            },
        }
    }

    fn param_info(&self) -> ParamInfo {
        match self {
            Self::ContinuousLinear(s) => ParamInfo {
                param_type: "continuous".into(),
                min: s.min,
                max: s.max,
                scale: LinearScale::name().to_string(),
                choices: None,
            },
            Self::ContinuousLog(s) => ParamInfo {
                param_type: "continuous".into(),
                min: s.min,
                max: s.max,
                scale: LogScale::name().to_string(),
                choices: None,
            },
            Self::ContinuousLog10(s) => ParamInfo {
                param_type: "continuous".into(),
                min: s.min,
                max: s.max,
                scale: Log10Scale::name().to_string(),
                choices: None,
            },
            Self::Discrete(s) => ParamInfo {
                param_type: "discrete".into(),
                min: s.min as f64,
                max: s.max as f64,
                scale: "linear".into(),
                choices: None,
            },
            Self::Categorical(s) => ParamInfo {
                param_type: "categorical".into(),
                min: 0.0,
                max: (s.cardinality() - 1) as f64,
                scale: "linear".into(),
                choices: Some(s.choices.clone()),
            },
        }
    }
}

// =============================================================================
// DynSpace: named parameter space built from DynDimension variants
// =============================================================================

/// A flat, named parameter space built from [`DynDimension`] variants.
///
/// Candidates are serialized as JSON objects (e.g., `{"lr": 0.01, "batch": 32}`).
/// Internally, each named dimension is stored behind an `Arc`, so cloning a
/// `DynSpace` is a cheap reference-count bump.
#[derive(Clone)]
pub struct DynSpace {
    dims: Arc<Vec<(String, DynDimension)>>,
}

impl std::fmt::Debug for DynSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynSpace")
            .field("n_dims", &self.dims.len())
            .field(
                "names",
                &self
                    .dims
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for DynSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl DynSpace {
    pub fn new() -> Self {
        Self {
            dims: Arc::new(Vec::new()),
        }
    }

    pub fn add_continuous(mut self, name: &str, min: f64, max: f64) -> Self {
        Arc::get_mut(&mut self.dims)
            .expect("DynSpace is being built; refcount must be 1")
            .push((
                name.to_string(),
                DynDimension::ContinuousLinear(ContinuousSpace::new(min, max)),
            ));
        self
    }

    pub fn add_continuous_log(mut self, name: &str, min: f64, max: f64) -> Self {
        Arc::get_mut(&mut self.dims)
            .expect("DynSpace is being built; refcount must be 1")
            .push((
                name.to_string(),
                DynDimension::ContinuousLog(ContinuousSpace::with_scale(min, max, LogScale)),
            ));
        self
    }

    pub fn add_continuous_log10(mut self, name: &str, min: f64, max: f64) -> Self {
        Arc::get_mut(&mut self.dims)
            .expect("DynSpace is being built; refcount must be 1")
            .push((
                name.to_string(),
                DynDimension::ContinuousLog10(ContinuousSpace::with_scale(min, max, Log10Scale)),
            ));
        self
    }

    pub fn add_discrete(mut self, name: &str, min: i64, max: i64) -> Self {
        Arc::get_mut(&mut self.dims)
            .expect("DynSpace is being built; refcount must be 1")
            .push((
                name.to_string(),
                DynDimension::Discrete(DiscreteSpace::new(min, max)),
            ));
        self
    }

    pub fn add_categorical(mut self, name: &str, choices: Vec<String>) -> Self {
        Arc::get_mut(&mut self.dims)
            .expect("DynSpace is being built; refcount must be 1")
            .push((
                name.to_string(),
                DynDimension::Categorical(CategoricalSpace::new(choices)),
            ));
        self
    }
}

impl SampleSpace for DynSpace {
    type Domain = serde_json::Value;

    fn contains(&self, point: &serde_json::Value) -> bool {
        let obj = match point.as_object() {
            Some(o) => o,
            None => return false,
        };
        self.dims
            .iter()
            .all(|(name, dim)| obj.get(name).is_some_and(|v| dim.contains(v)))
    }

    fn clamp(&self, point: &serde_json::Value) -> serde_json::Value {
        let obj = match point.as_object() {
            Some(o) => o,
            None => return point.clone(),
        };
        let mut clamped = obj.clone();
        for (name, dim) in self.dims.iter() {
            if let Some(val) = clamped.get(name.as_str()).cloned() {
                clamped.insert(name.clone(), dim.clamp(&val));
            }
        }
        serde_json::Value::Object(clamped)
    }
}

impl StandardizedSpace for DynSpace {
    fn dimensionality(&self) -> usize {
        self.dims.iter().map(|(_, d)| d.dimensionality()).sum()
    }

    fn to_unit_cube(&self, point: &serde_json::Value) -> Vec<f64> {
        let obj = point.as_object().expect("DynSpace expects a JSON object");
        let mut vec = Vec::with_capacity(self.dimensionality());
        for (name, dim) in self.dims.iter() {
            if let Some(val) = obj.get(name) {
                if let Some(sub) = dim.to_unit_cube(val) {
                    vec.extend(sub);
                    continue;
                }
                eprintln!(
                    "[hola] Warning: parameter '{name}' has invalid type, falling back to midpoint"
                );
            } else {
                eprintln!(
                    "[hola] Warning: parameter '{name}' missing from JSON object, falling back to midpoint"
                );
            }
            // Fallback: midpoint
            vec.extend(std::iter::repeat_n(0.5, dim.dimensionality()));
        }
        vec
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<serde_json::Value> {
        let mut map = serde_json::Map::new();
        let mut offset = 0;
        for (name, dim) in self.dims.iter() {
            let d = dim.dimensionality();
            if offset + d > vec.len() {
                return None;
            }
            let val = dim.from_unit_cube(&vec[offset..offset + d])?;
            map.insert(name.clone(), val);
            offset += d;
        }
        Some(serde_json::Value::Object(map))
    }
}

// =============================================================================
// DynStrategy wrapper
// =============================================================================

/// Type-erased strategy operating on DynSpace.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum DynStrategyInner {
    Random(RandomStrategy<DynSpace>),
    Sobol(SobolStrategy<DynSpace>),
    Gmm(GmmStrategy<DynSpace>),
    Auto(AutoStrategy),
}

/// Two-phase strategy: Sobol exploration followed by GMM exploitation.
///
/// During the first `exploration_budget` trials, candidates are drawn from a
/// Sobol sequence. After that, candidates are drawn from a Gaussian mixture
/// model that is periodically refit to elite trials.
///
/// The default exploration budget follows the formula from the paper:
/// `min(floor(S / 5), 50 + 2n)`, where `S` is the intended number of
/// simulations and `n` is the dimensionality.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoStrategy {
    sobol: SobolStrategy<DynSpace>,
    gmm: GmmStrategy<DynSpace>,
    exploration_budget: usize,
    trial_count: usize,
}

impl AutoStrategy {
    /// Compute the default exploration budget from the paper's formula,
    /// rounded down to the nearest power of two to preserve the balanced
    /// space-filling properties of the Sobol sequence.
    ///
    /// `total_budget` is `S`, the intended total number of simulations.
    /// `dim` is `n`, the dimensionality of the search space.
    pub fn default_exploration_budget(total_budget: usize, dim: usize) -> usize {
        let a = total_budget / 5;
        let b = 50 + 2 * dim;
        let raw = a.min(b);
        // Round down to the nearest power of two so the Sobol sequence
        // retains its low-discrepancy guarantee.
        if raw < 2 {
            raw
        } else {
            1 << (usize::BITS - 1 - raw.leading_zeros())
        }
    }

    pub fn new(dim: usize, exploration_budget: usize, seed: Option<u64>) -> Self {
        let (sobol_seed, gmm_seed) = match seed {
            Some(s) => (s as u32, s),
            None => (42, rand::random()),
        };
        Self {
            sobol: SobolStrategy::new(sobol_seed),
            gmm: GmmStrategy::uniform_prior(gmm_seed, dim, 0.1),
            exploration_budget,
            trial_count: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynStrategy {
    inner: DynStrategyInner,
}

impl Strategy for DynStrategy {
    type Space = DynSpace;
    type Observation = f64;

    fn suggest(&self, space: &DynSpace) -> serde_json::Value {
        match &self.inner {
            DynStrategyInner::Random(s) => s.suggest(space),
            DynStrategyInner::Sobol(s) => s.suggest(space),
            DynStrategyInner::Gmm(s) => s.suggest(space),
            DynStrategyInner::Auto(s) => {
                if s.trial_count < s.exploration_budget {
                    s.sobol.suggest(space)
                } else {
                    s.gmm.suggest(space)
                }
            }
        }
    }

    fn update(&mut self, candidate: &serde_json::Value, observation: f64) {
        match &mut self.inner {
            DynStrategyInner::Random(s) => s.update(candidate, observation),
            DynStrategyInner::Sobol(s) => s.update(candidate, observation),
            DynStrategyInner::Gmm(s) => s.update(candidate, observation),
            DynStrategyInner::Auto(s) => {
                s.trial_count += 1;
                s.sobol.update(candidate, observation);
                s.gmm.update(candidate, observation);
            }
        }
    }
}

impl opt_engine::traits::RefittableStrategy for DynStrategy {
    fn refit(&mut self, space: &DynSpace, trials: &[(serde_json::Value, f64)]) {
        match &mut self.inner {
            DynStrategyInner::Gmm(s) => s.refit(space, trials),
            DynStrategyInner::Auto(s) => s.gmm.refit(space, trials),
            _ => {}
        }
    }
}

// =============================================================================
// Configuration types for constructing DynEngine from YAML/JSON
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ParamConfig {
    Continuous {
        min: f64,
        max: f64,
        #[serde(default = "default_scale")]
        scale: String,
    },
    Discrete {
        min: i64,
        max: i64,
    },
    Categorical {
        choices: Vec<String>,
    },
}

fn default_scale() -> String {
    "linear".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObjectiveConfig {
    pub field: String,
    #[serde(alias = "type")]
    pub obj_type: String,
    #[serde(default)]
    pub target: Option<f64>,
    #[serde(default)]
    pub limit: Option<f64>,
    #[serde(default = "default_priority")]
    pub priority: f64,
    /// Explicit priority-group label. Objectives sharing the same group are
    /// summed into one component of the group-cost vector used for Pareto
    /// ranking.  When omitted, defaults to the field name (one group per
    /// objective).
    #[serde(default)]
    pub group: Option<String>,
}

fn default_priority() -> f64 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyConfig {
    #[serde(alias = "type")]
    pub strategy_type: String,
    #[serde(default = "default_refit_interval")]
    pub refit_interval: usize,
    /// Total simulation budget S (used by "auto" to compute exploration threshold).
    #[serde(default)]
    pub total_budget: Option<usize>,
    /// Override the exploration budget directly instead of using the formula.
    #[serde(default)]
    pub exploration_budget: Option<usize>,
    /// Optional seed for reproducible runs. When `None`, strategies use their
    /// default seeding (Sobol=42, others use random seeds).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Fraction of top trials used for GMM refitting (default: 0.25).
    /// Must be in (0.0, 1.0].
    #[serde(default)]
    pub elite_fraction: Option<f64>,
}

fn default_refit_interval() -> usize {
    20
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudyConfig {
    pub space: BTreeMap<String, ParamConfig>,
    pub objectives: Vec<ObjectiveConfig>,
    #[serde(default)]
    pub strategy: Option<StrategyConfig>,
    #[serde(default)]
    pub checkpoint: Option<CheckpointConfig>,
    #[serde(default)]
    pub max_trials: Option<usize>,
}

/// Configuration for automatic checkpointing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints.
    pub directory: String,
    /// Checkpoint every N trials.
    #[serde(default = "default_checkpoint_interval")]
    pub interval: usize,
    /// Maximum number of checkpoint files to keep (oldest are deleted).
    #[serde(default)]
    pub max_checkpoints: Option<usize>,
    /// Path to a checkpoint file to resume from on startup.
    #[serde(default)]
    pub load_from: Option<String>,
}

fn default_checkpoint_interval() -> usize {
    50
}

// =============================================================================
// DynEngine: the top-level Ask/Tell interface
// =============================================================================

/// A trial returned by `ask()`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynTrial {
    pub trial_id: u64,
    pub params: serde_json::Value,
}

/// A completed trial with full scoring, ranking, and Pareto front information.
///
/// This is the public-facing trial type returned by `tell()`, `top_k()`,
/// `pareto_front()`, and `trials()`. It is a computed view assembled from the
/// underlying leaderboard data — not a stored type.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletedTrial {
    /// Trial identifier.
    pub trial_id: u64,
    /// Candidate configuration that was evaluated.
    pub params: serde_json::Value,
    /// Raw worker output (what `tell()` received), untransformed.
    pub metrics: serde_json::Value,
    /// Per-objective scored values after TLP/direction handling.
    /// e.g., `{"loss": 0.3, "latency": 0.8}`.
    /// 0 = target met, (0,1) = between target and limit, inf = infeasible.
    pub scores: serde_json::Value,
    /// Per-priority-group aggregated scores (objectives summed within group).
    /// This is what ranking and Pareto use.
    pub score_vector: serde_json::Value,
    /// 0-indexed overall rank.
    /// Scalar: by score ascending (lower = better).
    /// Vector: NSGA-II (Pareto front, then crowding distance).
    pub rank: usize,
    /// 0-indexed Pareto front membership. Always present (== rank for scalar).
    pub pareto_front: usize,
    /// When `tell()` was called (unix seconds).
    pub completed_at: u64,
}

// =============================================================================
// DynLeaderboard: scalar or vector leaderboard dispatch
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum DynLeaderboard {
    Scalar(Leaderboard<serde_json::Value, f64>),
    Vector(Leaderboard<serde_json::Value, BTreeMap<String, f64>>),
}

impl DynLeaderboard {
    fn push_with_raw(
        &mut self,
        candidate: serde_json::Value,
        raw_metrics: serde_json::Value,
        objectives: &[ObjectiveConfig],
    ) -> (u64, f64) {
        match self {
            DynLeaderboard::Scalar(lb) => {
                let score = scalarize_raw(&raw_metrics, objectives);
                let id = lb.push_with_raw(candidate, score, raw_metrics);
                (id, score)
            }
            DynLeaderboard::Vector(lb) => {
                let obs = vectorize_raw(&raw_metrics, objectives);
                let score = scalarize_observation(&obs, objectives);
                let id = lb.push_with_raw(candidate, obs, raw_metrics);
                (id, score)
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            DynLeaderboard::Scalar(lb) => lb.len(),
            DynLeaderboard::Vector(lb) => lb.len(),
        }
    }

    /// Return top-k trials as (candidate, scalarized_score) for strategy refit.
    fn top_k_for_refit(
        &self,
        k: usize,
        objectives: &[ObjectiveConfig],
    ) -> Vec<(serde_json::Value, f64)> {
        match self {
            DynLeaderboard::Scalar(lb) => lb
                .top_k(k)
                .into_iter()
                .map(|t| (t.candidate, t.observation))
                .collect(),
            DynLeaderboard::Vector(lb) => lb
                .top_k_scalarized(k, |obs| scalarize_observation(obs, objectives))
                .into_iter()
                .map(|t| {
                    (
                        t.candidate,
                        scalarize_observation(&t.observation, objectives),
                    )
                })
                .collect(),
        }
    }

    fn rescalarize(&mut self, objectives: &[ObjectiveConfig]) {
        match self {
            DynLeaderboard::Scalar(lb) => {
                lb.rescalarize(|raw| Some(scalarize_raw(raw, objectives)));
            }
            DynLeaderboard::Vector(lb) => {
                lb.rescalarize(|raw| Some(vectorize_raw(raw, objectives)));
            }
        }
    }

    /// Get a single completed trial by ID, computing its rank and Pareto front.
    fn get_completed(
        &self,
        trial_id: u64,
        objectives: &[ObjectiveConfig],
    ) -> Option<CompletedTrial> {
        // Build the full ranked list, then find the requested trial.
        let all = self.completed_trials("rank", false, objectives);
        all.into_iter()
            .find(|ct| ct.trial_id == trial_id)
            .or_else(|| {
                // Trial might be infeasible — try again with infeasible included
                let all_with_inf = self.completed_trials("rank", true, objectives);
                all_with_inf.into_iter().find(|ct| ct.trial_id == trial_id)
            })
    }

    /// Return all trials as CompletedTrial with ranking and scoring.
    fn completed_trials(
        &self,
        sorted_by: &str,
        include_infeasible: bool,
        objectives: &[ObjectiveConfig],
    ) -> Vec<CompletedTrial> {
        match self {
            DynLeaderboard::Scalar(lb) => {
                // Sort all trials by observation to assign ranks
                let sorted = if include_infeasible {
                    lb.sorted_all()
                } else {
                    lb.sorted()
                };
                let mut results: Vec<CompletedTrial> = sorted
                    .into_iter()
                    .enumerate()
                    .map(|(rank, t)| build_completed_scalar(t, rank, objectives))
                    .collect();
                sort_completed(&mut results, sorted_by);
                results
            }
            DynLeaderboard::Vector(lb) => {
                // Use NSGA-II ranking for vector studies
                let ranked = if include_infeasible {
                    lb.ranked_trials_all()
                } else {
                    lb.ranked_trials()
                };
                let mut results: Vec<CompletedTrial> = ranked
                    .into_iter()
                    .map(|rt| {
                        let pareto_front = rt.rank.saturating_sub(1); // 1-indexed → 0-indexed
                        build_completed_vector(rt.trial, pareto_front, objectives)
                    })
                    .collect();
                // Assign overall rank based on NSGA-II ordering (already sorted by crowded_compare)
                for (i, ct) in results.iter_mut().enumerate() {
                    ct.rank = i;
                }
                sort_completed(&mut results, sorted_by);
                results
            }
        }
    }

    /// Return top-k trials as CompletedTrial.
    fn top_k_completed(
        &self,
        k: usize,
        include_infeasible: bool,
        objectives: &[ObjectiveConfig],
    ) -> Vec<CompletedTrial> {
        let all = self.completed_trials("rank", include_infeasible, objectives);
        all.into_iter().take(k).collect()
    }

    /// Return trials on a specific Pareto front as CompletedTrial.
    ///
    /// Returns an empty list for scalar (single-group) studies, since Pareto
    /// ranking only applies to multi-objective (vector) studies.
    fn pareto_front_completed(
        &self,
        front: usize,
        include_infeasible: bool,
        objectives: &[ObjectiveConfig],
    ) -> Vec<CompletedTrial> {
        match self {
            DynLeaderboard::Scalar(_) => Vec::new(),
            DynLeaderboard::Vector(_) => {
                let all = self.completed_trials("rank", include_infeasible, objectives);
                all.into_iter()
                    .filter(|ct| ct.pareto_front == front)
                    .collect()
            }
        }
    }
}

/// Build a CompletedTrial from a scalar leaderboard trial.
fn build_completed_scalar(
    t: Trial<serde_json::Value, f64>,
    rank: usize,
    objectives: &[ObjectiveConfig],
) -> CompletedTrial {
    let metrics = t.raw_metrics.clone().unwrap_or(serde_json::Value::Null);
    let scores = compute_scores(&metrics, objectives);
    let score_vector = compute_score_vector(&metrics, objectives);
    CompletedTrial {
        trial_id: t.trial_id,
        params: t.candidate,
        metrics,
        scores,
        score_vector,
        rank,
        pareto_front: rank, // scalar: pareto_front == rank
        completed_at: t.timestamp,
    }
}

/// Build a CompletedTrial from a vector leaderboard trial.
fn build_completed_vector(
    t: Trial<serde_json::Value, BTreeMap<String, f64>>,
    pareto_front: usize,
    objectives: &[ObjectiveConfig],
) -> CompletedTrial {
    let metrics = t.raw_metrics.clone().unwrap_or(serde_json::Value::Null);
    let scores = compute_scores(&metrics, objectives);
    let score_vector = f64_map_to_json(&t.observation);
    CompletedTrial {
        trial_id: t.trial_id,
        params: t.candidate,
        metrics,
        scores,
        score_vector,
        rank: 0, // assigned later by caller
        pareto_front,
        completed_at: t.timestamp,
    }
}

/// Convert a `BTreeMap<String, f64>` to a JSON object, representing infinity as `"inf"`.
fn f64_map_to_json(map: &BTreeMap<String, f64>) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    for (k, v) in map {
        obj.insert(
            k.clone(),
            if v.is_infinite() {
                serde_json::Value::from("inf")
            } else if v.is_nan() {
                serde_json::Value::Null
            } else {
                serde_json::Value::from(*v)
            },
        );
    }
    serde_json::Value::Object(obj)
}

/// Compute per-objective TLP scores φ_i from raw metrics.
///
/// Each score is P_i × normalized_distance, matching the paper's formula.
fn compute_scores(raw: &serde_json::Value, objectives: &[ObjectiveConfig]) -> serde_json::Value {
    let mut scores = serde_json::Map::new();
    for obj in objectives {
        let val = raw.get(&obj.field).and_then(|v| v.as_f64());
        let score = match val {
            Some(v) => objective_score(v, &obj.obj_type, obj.target, obj.limit) * obj.priority,
            None => f64::INFINITY,
        };
        scores.insert(
            obj.field.clone(),
            if score.is_infinite() {
                serde_json::Value::from("inf")
            } else {
                serde_json::Value::from(score)
            },
        );
    }
    serde_json::Value::Object(scores)
}

/// Compute per-priority-group aggregated scores from raw metrics.
fn compute_score_vector(
    raw: &serde_json::Value,
    objectives: &[ObjectiveConfig],
) -> serde_json::Value {
    if count_priority_groups(objectives) > 1 {
        let vec = vectorize_raw(raw, objectives);
        f64_map_to_json(&vec)
    } else {
        // Single group: wrap the scalar score
        let score = scalarize_raw(raw, objectives);
        let key = objectives
            .first()
            .map(|o| o.field.clone())
            .unwrap_or_else(|| "score".to_string());
        let mut map = serde_json::Map::new();
        if score.is_infinite() {
            map.insert(key, serde_json::Value::from("inf"));
        } else {
            map.insert(key, serde_json::Value::from(score));
        }
        serde_json::Value::Object(map)
    }
}

/// Sort a Vec<CompletedTrial> by the given criterion.
fn sort_completed(trials: &mut [CompletedTrial], sorted_by: &str) {
    match sorted_by {
        "rank" => trials.sort_by_key(|t| t.rank),
        "completed_at" => trials.sort_by_key(|t| t.completed_at),
        "index" => trials.sort_by_key(|t| t.trial_id),
        field => {
            // Sort by a specific score field (ascending)
            trials.sort_by(|a, b| {
                let a_val = a
                    .scores
                    .get(field)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::INFINITY);
                let b_val = b
                    .scores
                    .get(field)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::INFINITY);
                a_val
                    .partial_cmp(&b_val)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
}

/// Count the number of distinct priority groups in the objectives.
fn count_priority_groups(objectives: &[ObjectiveConfig]) -> usize {
    let mut groups: Vec<String> = objectives.iter().map(group_key).collect();
    groups.sort_unstable();
    groups.dedup();
    groups.len()
}

/// The HOLA engine. Single entry point for Python FFI and REST API.
///
/// Wraps [`opt_engine::Engine`] with type-erased JSON interfaces and HOLA-specific
/// orchestration: TLP scalarization, auto-refitting, checkpointing, and the
/// CompletedTrial view.
///
/// Cloning a `HolaEngine` is cheap (Arc reference-count bumps). Both the
/// original and the clone share the same underlying state, so changes made
/// through one are visible through the other.
#[derive(Clone)]
pub struct HolaEngine {
    space: DynSpace,
    state: Arc<RwLock<HolaEngineState>>,
    objectives: Arc<RwLock<Vec<ObjectiveConfig>>>,
    refit_config: Option<RefitConfig>,
    auto_checkpoint: Option<AutoCheckpointConfig>,
    max_trials: Option<usize>,
}

struct HolaEngineState {
    strategy: DynStrategy,
    leaderboard: DynLeaderboard,
    next_pending_id: u64,
    pending: BTreeMap<u64, serde_json::Value>,
    cancelled: HashSet<u64>,
}

impl HolaEngine {
    /// Build a HolaEngine from a StudyConfig (parsed from YAML/JSON).
    pub fn from_config(config: StudyConfig) -> Result<Self, String> {
        if config.objectives.is_empty() {
            return Err("At least one objective is required. \
                 Example: objectives: [{ field: \"loss\", obj_type: \"minimize\" }]"
                .to_string());
        }

        let mut space = DynSpace::new();
        for (name, param) in &config.space {
            space = match param {
                ParamConfig::Continuous { min, max, scale } => match scale.as_str() {
                    "log" | "ln" => {
                        if *min <= 0.0 || *max <= 0.0 {
                            return Err(format!(
                                "Parameter '{name}': log scale requires min > 0 and max > 0, got min={min}, max={max}",
                            ));
                        }
                        if *min >= *max {
                            return Err(format!(
                                "Parameter '{name}': min must be less than max, got min={min}, max={max}",
                            ));
                        }
                        space.add_continuous_log(name, *min, *max)
                    }
                    "log10" => {
                        if *min <= 0.0 || *max <= 0.0 {
                            return Err(format!(
                                "Parameter '{name}': log10 scale requires min > 0 and max > 0, got min={min}, max={max}",
                            ));
                        }
                        if *min >= *max {
                            return Err(format!(
                                "Parameter '{name}': min must be less than max, got min={min}, max={max}",
                            ));
                        }
                        space.add_continuous_log10(name, *min, *max)
                    }
                    _ => space.add_continuous(name, *min, *max),
                },
                ParamConfig::Discrete { min, max } => space.add_discrete(name, *min, *max),
                ParamConfig::Categorical { choices } => {
                    space.add_categorical(name, choices.clone())
                }
            };
        }

        let dim = space.dimensionality();
        let strategy_cfg = config.strategy.as_ref();
        let strategy_type = strategy_cfg
            .map(|s| s.strategy_type.as_str())
            .unwrap_or("gmm");

        let refit_interval = strategy_cfg.map(|s| s.refit_interval).unwrap_or(20);
        let seed = strategy_cfg.and_then(|s| s.seed);
        let max_trials = config
            .max_trials
            .or_else(|| strategy_cfg.and_then(|s| s.total_budget));

        let (strategy, refit_config) = match strategy_type {
            "random" => (
                DynStrategy {
                    inner: DynStrategyInner::Random(match seed {
                        Some(s) => RandomStrategy::new(s),
                        None => RandomStrategy::auto_seed(),
                    }),
                },
                None,
            ),
            "sobol" => (
                DynStrategy {
                    inner: DynStrategyInner::Sobol(SobolStrategy::new(
                        seed.map(|s| s as u32).unwrap_or(42),
                    )),
                },
                None,
            ),
            // "gmm" (default): Sobol exploration followed by GMM exploitation
            _ => {
                let exploration_budget = strategy_cfg
                    .and_then(|s| s.exploration_budget)
                    .unwrap_or_else(|| {
                        let total = max_trials.unwrap_or(200);
                        AutoStrategy::default_exploration_budget(total, dim)
                    });
                (
                    DynStrategy {
                        inner: DynStrategyInner::Auto(AutoStrategy::new(
                            dim,
                            exploration_budget,
                            seed,
                        )),
                    },
                    Some(RefitConfig::with_quantile(
                        exploration_budget,
                        refit_interval,
                        strategy_cfg.and_then(|s| s.elite_fraction).unwrap_or(0.25),
                    )),
                )
            }
        };

        let auto_checkpoint = config.checkpoint.as_ref().map(|c| {
            let mut ac = AutoCheckpointConfig::new(&c.directory, c.interval);
            ac.max_checkpoints = c.max_checkpoints;
            ac
        });

        let leaderboard = if count_priority_groups(&config.objectives) > 1 {
            DynLeaderboard::Vector(Leaderboard::new())
        } else {
            DynLeaderboard::Scalar(Leaderboard::new())
        };

        Ok(Self {
            space,
            state: Arc::new(RwLock::new(HolaEngineState {
                strategy,
                leaderboard,
                next_pending_id: 0,
                pending: BTreeMap::new(),
                cancelled: HashSet::new(),
            })),
            objectives: Arc::new(RwLock::new(config.objectives)),
            refit_config,
            auto_checkpoint,
            max_trials,
        })
    }

    /// Ask for the next trial to evaluate.
    ///
    /// Returns an error if `max_trials` has been reached.
    pub async fn ask(&self) -> Result<DynTrial, String> {
        let mut state = self.state.write().await;
        if let Some(max) = self.max_trials {
            let total = state.leaderboard.len() + state.pending.len();
            if total >= max {
                return Err(format!(
                    "max_trials ({max}) reached ({} completed, {} pending)",
                    state.leaderboard.len(),
                    state.pending.len()
                ));
            }
        }
        let params = state.strategy.suggest(&self.space);
        let id = state.next_pending_id;
        state.next_pending_id += 1;
        state.pending.insert(id, params.clone());
        Ok(DynTrial {
            trial_id: id,
            params,
        })
    }

    /// Tell the engine the result of a trial, returning the scored and ranked trial.
    pub async fn tell(
        &self,
        trial_id: u64,
        raw_metrics: serde_json::Value,
    ) -> Result<CompletedTrial, String> {
        let objectives = self.objectives.read().await.clone();

        let mut state = self.state.write().await;

        if state.cancelled.contains(&trial_id) {
            return Err(format!("Trial {trial_id} has been cancelled"));
        }

        let candidate = state
            .pending
            .remove(&trial_id)
            .ok_or_else(|| format!("Unknown trial_id: {trial_id}"))?;

        let (_trial_id, score) =
            state
                .leaderboard
                .push_with_raw(candidate.clone(), raw_metrics, &objectives);
        state.strategy.update(&candidate, score);

        let n_trials = state.leaderboard.len();
        let completed = state
            .leaderboard
            .get_completed(trial_id, &objectives)
            .ok_or_else(|| format!("Failed to build CompletedTrial for {trial_id}"))?;
        drop(state);

        // Auto-refit if configured
        if let Some(ref config) = self.refit_config
            && config.should_refit(n_trials)
        {
            let state_guard = self.state.read().await;
            let k = config.selection_count(n_trials);
            let trials = state_guard.leaderboard.top_k_for_refit(k, &objectives);
            let mut strategy_snapshot = state_guard.strategy.clone();
            let space_clone = self.space.clone();
            drop(state_guard);

            let fitted = tokio::task::spawn_blocking(move || {
                use opt_engine::traits::RefittableStrategy;
                strategy_snapshot.refit(&space_clone, &trials);
                strategy_snapshot
            })
            .await
            .map_err(|e| e.to_string())?;

            self.state.write().await.strategy = fitted;
        }

        // Auto-checkpoint if configured
        if let Some(ref config) = self.auto_checkpoint
            && config.should_checkpoint(n_trials)
        {
            if let Err(e) = self
                .save_leaderboard_checkpoint_to(
                    config.filename(n_trials),
                    Some(&format!("Auto-checkpoint at {n_trials} trials")),
                )
                .await
            {
                eprintln!("[hola] Warning: auto-checkpoint failed: {e}");
            } else {
                eprintln!("[hola] Auto-checkpoint saved at {n_trials} trials");
                // Rotate old checkpoints
                if let Some(max) = config.max_checkpoints {
                    Self::rotate_checkpoints(&config.directory, &config.prefix, max);
                }
            }
        }

        Ok(completed)
    }

    /// Cancel a pending trial.
    pub async fn cancel(&self, trial_id: u64) -> Result<(), String> {
        let mut state = self.state.write().await;
        if state.pending.remove(&trial_id).is_some() {
            state.cancelled.insert(trial_id);
            Ok(())
        } else {
            Err(format!(
                "Trial {trial_id} is not pending (may be completed or unknown)"
            ))
        }
    }

    /// Get the top-k trials by rank.
    pub async fn top_k(&self, k: usize, include_infeasible: bool) -> Vec<CompletedTrial> {
        let objectives = self.objectives.read().await.clone();
        self.state
            .read()
            .await
            .leaderboard
            .top_k_completed(k, include_infeasible, &objectives)
    }

    /// Get the number of completed trials.
    pub async fn trial_count(&self) -> usize {
        self.state.read().await.leaderboard.len()
    }

    /// Get trials on a specific Pareto front.
    pub async fn pareto_front(
        &self,
        front: usize,
        include_infeasible: bool,
    ) -> Vec<CompletedTrial> {
        let objectives = self.objectives.read().await.clone();
        self.state.read().await.leaderboard.pareto_front_completed(
            front,
            include_infeasible,
            &objectives,
        )
    }

    /// Get all trials with scoring and ranking.
    pub async fn trials(&self, sorted_by: &str, include_infeasible: bool) -> Vec<CompletedTrial> {
        let objectives = self.objectives.read().await.clone();
        self.state.read().await.leaderboard.completed_trials(
            sorted_by,
            include_infeasible,
            &objectives,
        )
    }

    /// Access the space configuration.
    pub fn space(&self) -> &DynSpace {
        &self.space
    }

    /// Get parameter metadata for dashboard auto-configuration.
    pub fn space_config(&self) -> Vec<(String, ParamInfo)> {
        self.space
            .dims
            .iter()
            .map(|(name, dim)| (name.clone(), dim.param_info()))
            .collect()
    }

    /// Reconstruct the `StudyConfig` from the engine's internal state.
    ///
    /// This is used to persist the config alongside checkpoint data so that
    /// `Study.load()` can fully restore a study without the user re-specifying
    /// the space and objectives.
    pub async fn study_config(&self) -> StudyConfig {
        let space: BTreeMap<String, ParamConfig> = self
            .space
            .dims
            .iter()
            .map(|(name, dim)| (name.clone(), dim.to_param_config()))
            .collect();
        let objectives = self.objectives.read().await.clone();
        StudyConfig {
            space,
            objectives,
            strategy: None, // Strategy state is in the checkpoint itself
            checkpoint: None,
            max_trials: self.max_trials,
        }
    }

    /// Get the current objectives configuration.
    pub async fn objectives(&self) -> Vec<ObjectiveConfig> {
        self.objectives.read().await.clone()
    }

    /// Re-scalarize all trials using the current objectives.
    pub async fn rescalarize(&self) {
        let objectives = self.objectives.read().await.clone();
        self.state
            .write()
            .await
            .leaderboard
            .rescalarize(&objectives);
    }

    /// Update objectives and re-scalarize (for mid-run dashboard adjustments).
    ///
    /// Persists the new objectives so that subsequent `tell()` calls use the
    /// updated scalarization. If a refittable strategy (e.g., GMM) is configured,
    /// a refit is triggered immediately so the sampling distribution reflects
    /// the new objective weights.
    pub async fn update_objectives(&self, objectives: Vec<ObjectiveConfig>) {
        // Persist the new objectives
        *self.objectives.write().await = objectives.clone();
        // Re-scalarize all historical trials with the new objectives
        let n_trials = {
            let mut state = self.state.write().await;
            state.leaderboard.rescalarize(&objectives);
            state.leaderboard.len()
        };

        // Trigger an immediate refit so the strategy reflects the new scalarization
        if let Some(ref config) = self.refit_config
            && n_trials >= config.min_trials
        {
            let state_guard = self.state.read().await;
            let k = config.selection_count(n_trials);
            let trials = state_guard.leaderboard.top_k_for_refit(k, &objectives);
            let mut strategy_snapshot = state_guard.strategy.clone();
            let space_clone = self.space.clone();
            drop(state_guard);

            if let Ok(fitted) = tokio::task::spawn_blocking(move || {
                use opt_engine::traits::RefittableStrategy;
                strategy_snapshot.refit(&space_clone, &trials);
                strategy_snapshot
            })
            .await
            {
                self.state.write().await.strategy = fitted;
            }
        }
    }

    // =========================================================================
    // Persistence (stable public API)
    // =========================================================================

    /// Save a full checkpoint (leaderboard + strategy state).
    ///
    /// This is the stable persistence API. Use `save` / `load` for checkpointing.
    pub async fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.save_full_checkpoint(path, None).await
    }

    /// Load a full checkpoint, restoring both leaderboard and strategy state.
    ///
    /// This is the stable persistence API. Use `save` / `load` for checkpointing.
    pub async fn load(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.load_full_checkpoint(path).await
    }

    // =========================================================================
    // Persistence (internal)
    // =========================================================================

    /// Save a leaderboard-only checkpoint (trial history, no strategy state).
    ///
    /// Uses atomic writes (write-to-temp + fsync + rename) to prevent data loss.
    pub async fn save_leaderboard_checkpoint_to(
        &self,
        path: impl AsRef<std::path::Path>,
        description: Option<&str>,
    ) -> std::io::Result<()> {
        let state = self.state.read().await;
        match &state.leaderboard {
            DynLeaderboard::Scalar(lb) => {
                LeaderboardCheckpoint::new(lb.clone(), description).save_json(path)
            }
            DynLeaderboard::Vector(lb) => {
                LeaderboardCheckpoint::new(lb.clone(), description).save_json(path)
            }
        }
    }

    /// Load a leaderboard-only checkpoint, replacing the current trial history.
    ///
    /// The strategy is NOT restored — call with `refit()` afterward if using GMM.
    pub async fn load_leaderboard_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let objectives = self.objectives.read().await;
        let leaderboard = if count_priority_groups(&objectives) > 1 {
            let cp: LeaderboardCheckpoint<serde_json::Value, BTreeMap<String, f64>> =
                LeaderboardCheckpoint::load_json(path)?;
            let n = cp.leaderboard.len();
            eprintln!("[hola] Loaded leaderboard checkpoint with {n} trials");
            DynLeaderboard::Vector(cp.leaderboard)
        } else {
            let cp: LeaderboardCheckpoint<serde_json::Value, f64> =
                LeaderboardCheckpoint::load_json(path)?;
            let n = cp.leaderboard.len();
            eprintln!("[hola] Loaded leaderboard checkpoint with {n} trials");
            DynLeaderboard::Scalar(cp.leaderboard)
        };
        self.state.write().await.leaderboard = leaderboard;
        Ok(())
    }

    /// Save a full checkpoint (leaderboard + strategy state + config).
    ///
    /// The saved JSON has the format:
    /// ```json
    /// {
    ///   "config": { ...StudyConfig... },
    ///   "checkpoint": { "leaderboard": ..., "strategy_state": ..., "metadata": ... }
    /// }
    /// ```
    ///
    /// The `config` key allows `load_from_checkpoint` to reconstruct the engine
    /// without the caller re-specifying the space and objectives.
    pub async fn save_full_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
        description: Option<&str>,
    ) -> std::io::Result<()> {
        let config = self.study_config().await;
        let state = self.state.read().await;
        let strategy = state.strategy.clone();
        let checkpoint_json = match &state.leaderboard {
            DynLeaderboard::Scalar(lb) => {
                let checkpoint =
                    opt_engine::persistence::Checkpoint::new(lb.clone(), strategy, description);
                drop(state);
                serde_json::to_value(&checkpoint)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            }
            DynLeaderboard::Vector(lb) => {
                let checkpoint =
                    opt_engine::persistence::Checkpoint::new(lb.clone(), strategy, description);
                drop(state);
                serde_json::to_value(&checkpoint)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            }
        };

        let wrapper = serde_json::json!({
            "config": config,
            "checkpoint": checkpoint_json,
        });

        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let file = std::fs::File::create(&tmp)?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &wrapper)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let file = writer
            .into_inner()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    }

    /// Load a full checkpoint, restoring both leaderboard and strategy state.
    ///
    /// Handles both the new format (with `"config"` + `"checkpoint"` wrapper)
    /// and the legacy format (direct checkpoint without config).
    pub async fn load_full_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let raw: serde_json::Value = {
            let file = std::fs::File::open(path.as_ref())?;
            let reader = std::io::BufReader::new(file);
            serde_json::from_reader(reader)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        };

        // New format has "checkpoint" key; legacy format has "leaderboard" at root.
        let checkpoint_json = if raw.get("checkpoint").is_some() {
            raw.get("checkpoint").unwrap().clone()
        } else {
            raw
        };

        let objectives = self.objectives.read().await;
        if count_priority_groups(&objectives) > 1 {
            let cp: opt_engine::persistence::Checkpoint<
                serde_json::Value,
                BTreeMap<String, f64>,
                DynStrategy,
            > = serde_json::from_value(checkpoint_json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let n_loaded = cp.leaderboard.len();
            let mut state = self.state.write().await;
            state.leaderboard = DynLeaderboard::Vector(cp.leaderboard);
            state.strategy = cp.strategy_state;
            eprintln!("[hola] Loaded full checkpoint with {n_loaded} trials");
        } else {
            let cp: opt_engine::persistence::Checkpoint<serde_json::Value, f64, DynStrategy> =
                serde_json::from_value(checkpoint_json)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let n_loaded = cp.leaderboard.len();
            let mut state = self.state.write().await;
            state.leaderboard = DynLeaderboard::Scalar(cp.leaderboard);
            state.strategy = cp.strategy_state;
            eprintln!("[hola] Loaded full checkpoint with {n_loaded} trials");
        }
        Ok(())
    }

    /// Load a study from a checkpoint file, reconstructing the engine from the
    /// embedded `StudyConfig`.
    ///
    /// The checkpoint must have been saved with the new format that includes
    /// the `"config"` key. Returns an error if the config is missing (i.e.,
    /// the file was saved with an older version of HOLA).
    pub async fn load_from_checkpoint(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let raw: serde_json::Value = {
            let file = std::fs::File::open(path.as_ref())
                .map_err(|e| format!("Failed to open checkpoint file: {e}"))?;
            let reader = std::io::BufReader::new(file);
            serde_json::from_reader(reader)
                .map_err(|e| format!("Failed to parse checkpoint JSON: {e}"))?
        };

        let config_value = raw.get("config").ok_or_else(|| {
            "Checkpoint file does not contain a 'config' key. \
             This file was likely saved with an older version of HOLA. \
             To load it, create a Study with the same space/objectives \
             and call study.load(path) instead."
                .to_string()
        })?;

        let config: StudyConfig = serde_json::from_value(config_value.clone())
            .map_err(|e| format!("Failed to parse StudyConfig from checkpoint: {e}"))?;

        let engine = Self::from_config(config)?;
        engine
            .load_full_checkpoint(path)
            .await
            .map_err(|e| format!("Failed to load checkpoint data: {e}"))?;
        Ok(engine)
    }

    /// Delete oldest checkpoint files to keep at most `max` files.
    fn rotate_checkpoints(directory: &std::path::Path, prefix: &str, max: usize) {
        let pattern = format!("{}_{}", prefix, "");
        let mut checkpoints: Vec<std::path::PathBuf> = std::fs::read_dir(directory)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().is_some_and(|ext| ext == "json")
                    && p.file_name()
                        .and_then(|f| f.to_str())
                        .is_some_and(|f| f.starts_with(&pattern))
            })
            .collect();

        if checkpoints.len() <= max {
            return;
        }

        checkpoints.sort();
        let to_delete = checkpoints.len() - max;
        for path in checkpoints.into_iter().take(to_delete) {
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!("[hola] Warning: failed to delete old checkpoint {path:?}: {e}");
            }
        }
    }
}

/// Collapse multi-field raw metrics into a single scalar cost F(x).
///
/// Implements the paper's formula: F(x) = Σ φ_i(f_i(x)) where
/// φ_i(u) = P_i × (u − T_i)/(L_i − T_i) for the in-range case.
/// Each objective's TLP score is multiplied by its priority weight P_i.
fn scalarize_raw(raw: &serde_json::Value, objectives: &[ObjectiveConfig]) -> f64 {
    let mut total = 0.0;
    for obj in objectives {
        let val = match raw.get(&obj.field).and_then(|v| v.as_f64()) {
            Some(v) => v,
            None => return f64::INFINITY,
        };
        total += objective_score(val, &obj.obj_type, obj.target, obj.limit) * obj.priority;
    }
    total
}

/// Compute per-group cost vector C(x) from raw metrics.
///
/// Groups objectives by their explicit group label. Within each group,
/// the group cost is: C_g(x) = Σ_{i ∈ G_g} P_i × φ_i(f_i(x))
/// where P_i is the per-objective priority weight.
fn vectorize_raw(raw: &serde_json::Value, objectives: &[ObjectiveConfig]) -> BTreeMap<String, f64> {
    let mut groups: BTreeMap<String, f64> = BTreeMap::new();
    for obj in objectives {
        let val = match raw.get(&obj.field).and_then(|v| v.as_f64()) {
            Some(v) => v,
            None => {
                groups.insert(group_key(obj), f64::INFINITY);
                continue;
            }
        };

        let score = objective_score(val, &obj.obj_type, obj.target, obj.limit) * obj.priority;
        *groups.entry(group_key(obj)).or_insert(0.0) += score;
    }
    groups
}

/// Sum of group costs → single scalar for strategy updates.
///
/// Each group's cost already includes per-objective priority weights,
/// so this is a plain sum: F(x) = Σ_g C_g(x).
fn scalarize_observation(obs: &BTreeMap<String, f64>, _objectives: &[ObjectiveConfig]) -> f64 {
    obs.values().sum()
}

/// Derive a stable group key from an objective's group label.
/// Falls back to the field name when no explicit group is set.
fn group_key(obj: &ObjectiveConfig) -> String {
    obj.group.clone().unwrap_or_else(|| obj.field.clone())
}

/// Compute TLP score for a single value (shared by scalarize_raw and vectorize_raw).
fn objective_score(val: f64, obj_type: &str, target: Option<f64>, limit: Option<f64>) -> f64 {
    match obj_type {
        "minimize" => match (target, limit) {
            (Some(t), Some(l)) => opt_engine::objectives::tlp_score(val, t, l),
            _ => val,
        },
        "maximize" => match (target, limit) {
            (Some(t), Some(l)) => opt_engine::objectives::tlp_score(val, t, l),
            _ => opt_engine::objectives::directed_value(val, true),
        },
        _ => val,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dyn_space_basic() {
        let space = DynSpace::new()
            .add_continuous("lr", 0.0, 1.0)
            .add_discrete("layers", 1, 10);

        assert_eq!(space.dimensionality(), 2);

        let point = serde_json::json!({"lr": 0.5, "layers": 5});
        assert!(space.contains(&point));

        let cube = space.to_unit_cube(&point);
        assert_eq!(cube.len(), 2);

        let restored = space.from_unit_cube(&cube).unwrap();
        let obj = restored.as_object().unwrap();
        assert!((obj["lr"].as_f64().unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_space_log10() {
        let space = DynSpace::new().add_continuous_log10("lr", 1e-4, 0.1);
        assert_eq!(space.dimensionality(), 1);

        let point = serde_json::json!({"lr": 0.01}); // 10^-2
        assert!(space.contains(&point));
    }

    #[test]
    fn test_scalarize_raw_with_objectives() {
        let objectives = vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: Some(0.0),
            limit: Some(1.0),
            priority: 1.0,
            group: None,
        }];

        let raw = serde_json::json!({"loss": 0.5});
        let score = scalarize_raw(&raw, &objectives);
        assert!((score - 0.5).abs() < 1e-10);

        let raw_perfect = serde_json::json!({"loss": 0.0});
        let score_perfect = scalarize_raw(&raw_perfect, &objectives);
        assert!((score_perfect).abs() < 1e-10);
    }

    #[test]
    fn test_dyn_space_builder_api() {
        let space = DynSpace::new()
            .add_continuous("x", 0.0, 1.0)
            .add_continuous_log10("lr", 1e-4, 0.1)
            .add_discrete("layers", 1, 10)
            .add_categorical("opt", vec!["adam".into(), "sgd".into()]);

        assert_eq!(space.dimensionality(), 4);
    }

    #[test]
    fn test_dyn_space_from_unit_cube_wrong_length() {
        let space = DynSpace::new()
            .add_continuous("x", 0.0, 1.0)
            .add_continuous("y", 0.0, 1.0);

        assert_eq!(space.dimensionality(), 2);
        assert!(space.from_unit_cube(&[0.5]).is_none());
        assert!(space.from_unit_cube(&[0.5, 0.5]).is_some());
        assert!(space.from_unit_cube(&[0.5, 0.5, 0.5]).is_some());
    }

    #[test]
    fn test_dyn_space_unit_cube_roundtrip() {
        let space = DynSpace::new()
            .add_continuous("x", 0.0, 10.0)
            .add_discrete("n", 1, 5)
            .add_categorical("opt", vec!["a".into(), "b".into()]);

        let point = serde_json::json!({"x": 5.0, "n": 3, "opt": "b"});
        let unit = space.to_unit_cube(&point);
        assert_eq!(unit.len(), 3);
        assert!(unit.iter().all(|v| *v >= 0.0 && *v <= 1.0));

        let restored = space.from_unit_cube(&unit).unwrap();
        assert!((restored.get("x").unwrap().as_f64().unwrap() - 5.0).abs() < 1e-9);
        assert_eq!(restored.get("n").unwrap().as_i64().unwrap(), 3);
        assert_eq!(restored.get("opt").unwrap().as_str().unwrap(), "b");
    }

    #[test]
    fn test_dyn_space_contains() {
        let space = DynSpace::new()
            .add_continuous("x", 0.0, 1.0)
            .add_discrete("n", 1, 5)
            .add_categorical("opt", vec!["a".into(), "b".into()]);

        assert!(space.contains(&serde_json::json!({"x": 0.5, "n": 3, "opt": "a"})));
        assert!(!space.contains(&serde_json::json!({"x": 2.0, "n": 3, "opt": "a"})));
        assert!(!space.contains(&serde_json::json!({"x": 0.5, "n": 10, "opt": "a"})));
        assert!(!space.contains(&serde_json::json!({"x": 0.5, "n": 3, "opt": "unknown"})));
    }

    #[test]
    fn test_dyn_space_clamp() {
        let space = DynSpace::new()
            .add_continuous("x", 0.0, 1.0)
            .add_discrete("n", 1, 5);

        let clamped = space.clamp(&serde_json::json!({"x": 2.0, "n": 10}));
        assert!((clamped.get("x").unwrap().as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(clamped.get("n").unwrap().as_i64().unwrap(), 5);
    }

    #[test]
    fn test_dyn_space_log_scales() {
        let space = DynSpace::new()
            .add_continuous_log("lr", 0.001, 1.0)
            .add_continuous_log10("alpha", 1e-4, 0.1);

        assert_eq!(space.dimensionality(), 2);

        let point = serde_json::json!({"lr": 0.01, "alpha": 0.01});
        assert!(space.contains(&point));

        let unit = space.to_unit_cube(&point);
        assert_eq!(unit.len(), 2);
        assert!(unit.iter().all(|v| *v >= 0.0 && *v <= 1.0));

        let restored = space.from_unit_cube(&unit).unwrap();
        let lr = restored.get("lr").unwrap().as_f64().unwrap();
        assert!((lr - 0.01).abs() / 0.01 < 1e-6);
    }
}
