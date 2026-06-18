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
//! the concrete space, strategy, or leaderboard types at compile time.
//! Parameters are `BTreeMap<String, serde_json::Value>` and metrics are
//! `serde_json::Value`.
//!
//! Reach for `HolaEngine` when the parameter space is defined at runtime (e.g.,
//! from a YAML config or Python `dict`); compose `opt_engine`'s building blocks
//! directly when you have concrete Rust types and want full compile-time
//! verification.

use opt_engine::leaderboard::{Leaderboard, Trial};
use opt_engine::persistence::{
    AutoCheckpointConfig, LeaderboardCheckpoint, ObservationKind, check_format_version_bytes,
    read_checkpoint_capped, sync_parent_dir, unique_temp_path,
};
use opt_engine::scales::{LinearScale, Log10Scale, LogScale, Scale};
use opt_engine::spaces::{CategoricalSpace, ContinuousSpace, DiscreteSpace};
use opt_engine::strategies::{GmmStrategy, RandomStrategy, SobolStrategy};
use opt_engine::traits::{RefitConfig, SampleSpace, StandardizedSpace, Strategy};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{Mutex, RwLock};

/// Maximum space dimensionality the Sobol backend supports. It ships 256
/// dimensions of direction numbers and panics in release beyond that, so
/// Sobol-based strategies are rejected above this at construction.
const MAX_SOBOL_DIMS: usize = 256;

/// Upper bound on the number of cancelled trial ids retained for the
/// tell-after-cancel rejection message.
///
/// A cancelled id is always below `next_pending_id`, and `ask()` only ever
/// allocates ids `>= next_pending_id` (which never decreases), so a cancelled id
/// can never be reissued — its only remaining purpose is to let `tell()` report
/// "has been cancelled" instead of "unknown" for a result that arrives after the
/// cancel. That race resolves promptly, so a bounded window of the most-recently
/// cancelled ids is sufficient; the set is pruned to this many entries (keeping
/// the largest/newest ids) so it cannot grow without bound over a long run.
const MAX_CANCELLED_RETAINED: usize = 4096;

// =============================================================================
// Parameter metadata for dashboard API
// =============================================================================

/// Metadata describing a single parameter dimension, sent to the dashboard
/// so it can auto-configure axis labels, scales, and choice dropdowns.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParamInfo {
    pub param_type: String, // "real", "integer", or "categorical"
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
/// parameter types with full compile-time verification, compose `opt_engine`'s
/// building blocks directly using the [`SampleSpace`] / [`StandardizedSpace`]
/// traits.
#[derive(Clone)]
enum DynDimension {
    RealLinear(ContinuousSpace<LinearScale>),
    RealLog(ContinuousSpace<LogScale>),
    RealLog10(ContinuousSpace<Log10Scale>),
    Integer(DiscreteSpace),
    Categorical(CategoricalSpace),
}

#[allow(clippy::wrong_self_convention)] // mirrors `StandardizedSpace::from_unit_cube` on inner spaces
impl DynDimension {
    fn dimensionality(&self) -> usize {
        match self {
            Self::RealLinear(s) => s.dimensionality(),
            Self::RealLog(s) => s.dimensionality(),
            Self::RealLog10(s) => s.dimensionality(),
            Self::Integer(s) => s.dimensionality(),
            Self::Categorical(s) => s.dimensionality(),
        }
    }

    fn to_unit_cube(&self, val: &serde_json::Value) -> Option<Vec<f64>> {
        match self {
            Self::RealLinear(s) => val.as_f64().map(|v| s.to_unit_cube(&v)),
            Self::RealLog(s) => val.as_f64().map(|v| s.to_unit_cube(&v)),
            Self::RealLog10(s) => val.as_f64().map(|v| s.to_unit_cube(&v)),
            Self::Integer(s) => val.as_i64().map(|v| s.to_unit_cube(&v)),
            Self::Categorical(s) => val.as_str().map(|v| s.to_unit_cube(&v.to_string())),
        }
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<serde_json::Value> {
        match self {
            Self::RealLinear(s) => s.from_unit_cube(vec).map(serde_json::Value::from),
            Self::RealLog(s) => s.from_unit_cube(vec).map(serde_json::Value::from),
            Self::RealLog10(s) => s.from_unit_cube(vec).map(serde_json::Value::from),
            Self::Integer(s) => s.from_unit_cube(vec).map(serde_json::Value::from),
            Self::Categorical(s) => s.from_unit_cube(vec).map(serde_json::Value::from),
        }
    }

    fn contains(&self, val: &serde_json::Value) -> bool {
        match self {
            Self::RealLinear(s) => val.as_f64().is_some_and(|v| s.contains(&v)),
            Self::RealLog(s) => val.as_f64().is_some_and(|v| s.contains(&v)),
            Self::RealLog10(s) => val.as_f64().is_some_and(|v| s.contains(&v)),
            Self::Integer(s) => val.as_i64().is_some_and(|v| s.contains(&v)),
            Self::Categorical(s) => val.as_str().is_some_and(|v| s.contains(&v.to_string())),
        }
    }

    fn clamp(&self, val: &serde_json::Value) -> serde_json::Value {
        match self {
            Self::RealLinear(s) => val
                .as_f64()
                .map(|v| serde_json::Value::from(s.clamp(&v)))
                .unwrap_or_else(|| val.clone()),
            Self::RealLog(s) => val
                .as_f64()
                .map(|v| serde_json::Value::from(s.clamp(&v)))
                .unwrap_or_else(|| val.clone()),
            Self::RealLog10(s) => val
                .as_f64()
                .map(|v| serde_json::Value::from(s.clamp(&v)))
                .unwrap_or_else(|| val.clone()),
            Self::Integer(s) => val
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
            Self::RealLinear(s) => ParamConfig::Real {
                min: s.min,
                max: s.max,
                scale: "linear".to_string(),
            },
            Self::RealLog(s) => ParamConfig::Real {
                min: s.min,
                max: s.max,
                scale: "log".to_string(),
            },
            Self::RealLog10(s) => ParamConfig::Real {
                min: s.min,
                max: s.max,
                scale: "log10".to_string(),
            },
            Self::Integer(s) => ParamConfig::Integer {
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
            Self::RealLinear(s) => ParamInfo {
                param_type: "real".into(),
                min: s.min,
                max: s.max,
                scale: LinearScale::name().to_string(),
                choices: None,
            },
            Self::RealLog(s) => ParamInfo {
                param_type: "real".into(),
                min: s.min,
                max: s.max,
                scale: LogScale::name().to_string(),
                choices: None,
            },
            Self::RealLog10(s) => ParamInfo {
                param_type: "real".into(),
                min: s.min,
                max: s.max,
                scale: Log10Scale::name().to_string(),
                choices: None,
            },
            Self::Integer(s) => ParamInfo {
                param_type: "integer".into(),
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

    pub fn add_real(mut self, name: &str, min: f64, max: f64) -> Self {
        // Copy-on-write: if a clone still shares the dims, make_mut clones them
        // first so a shared DynSpace is never mutated and this never panics.
        Arc::make_mut(&mut self.dims).push((
            name.to_string(),
            DynDimension::RealLinear(ContinuousSpace::new(min, max)),
        ));
        self
    }

    pub fn add_real_log(mut self, name: &str, min: f64, max: f64) -> Self {
        // Copy-on-write: if a clone still shares the dims, make_mut clones them
        // first so a shared DynSpace is never mutated and this never panics.
        Arc::make_mut(&mut self.dims).push((
            name.to_string(),
            DynDimension::RealLog(ContinuousSpace::with_scale(min, max, LogScale)),
        ));
        self
    }

    pub fn add_real_log10(mut self, name: &str, min: f64, max: f64) -> Self {
        // Copy-on-write: if a clone still shares the dims, make_mut clones them
        // first so a shared DynSpace is never mutated and this never panics.
        Arc::make_mut(&mut self.dims).push((
            name.to_string(),
            DynDimension::RealLog10(ContinuousSpace::with_scale(min, max, Log10Scale)),
        ));
        self
    }

    pub fn add_integer(mut self, name: &str, min: i64, max: i64) -> Self {
        // Copy-on-write: if a clone still shares the dims, make_mut clones them
        // first so a shared DynSpace is never mutated and this never panics.
        Arc::make_mut(&mut self.dims).push((
            name.to_string(),
            DynDimension::Integer(DiscreteSpace::new(min, max)),
        ));
        self
    }

    pub fn add_categorical(mut self, name: &str, choices: Vec<String>) -> Self {
        // Copy-on-write: if a clone still shares the dims, make_mut clones them
        // first so a shared DynSpace is never mutated and this never panics.
        Arc::make_mut(&mut self.dims).push((
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
        // Match the graceful degradation of the sibling contains()/clamp(): a
        // non-object point has no named parameters to read, so fall back to a
        // midpoint vector of the correct length instead of panicking.
        let obj = match point.as_object() {
            Some(o) => o,
            None => {
                eprintln!(
                    "[hola] Warning: to_unit_cube expected a JSON object, falling back to midpoint"
                );
                return vec![0.5; self.dimensionality()];
            }
        };
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
        // Reject over-length input: every coordinate must be consumed exactly,
        // so a vector longer than the space's dimensionality is malformed.
        if offset != vec.len() {
            return None;
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
#[derive(Debug)]
pub struct AutoStrategy {
    sobol: SobolStrategy<DynSpace>,
    gmm: GmmStrategy<DynSpace>,
    exploration_budget: usize,
    trial_count: usize,
    issued_count: AtomicUsize,
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
            // Fold the high 32 bits into the low 32 instead of truncating, so two
            // u64 seeds that differ only in their high bits yield distinct Sobol
            // seeds. Deterministic: the same u64 always folds to the same u32.
            Some(s) => ((s ^ (s >> 32)) as u32, s),
            None => (42, rand::random()),
        };
        Self {
            sobol: SobolStrategy::new(sobol_seed),
            gmm: GmmStrategy::uniform_prior(gmm_seed, dim, 0.1),
            exploration_budget,
            trial_count: 0,
            issued_count: AtomicUsize::new(0),
        }
    }
}

impl Clone for AutoStrategy {
    fn clone(&self) -> Self {
        Self {
            sobol: self.sobol.clone(),
            gmm: self.gmm.clone(),
            exploration_budget: self.exploration_budget,
            trial_count: self.trial_count,
            issued_count: AtomicUsize::new(self.issued_count.load(Ordering::Relaxed)),
        }
    }
}

impl Serialize for AutoStrategy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("AutoStrategy", 5)?;
        state.serialize_field("sobol", &self.sobol)?;
        state.serialize_field("gmm", &self.gmm)?;
        state.serialize_field("exploration_budget", &self.exploration_budget)?;
        state.serialize_field("trial_count", &self.trial_count)?;
        state.serialize_field("issued_count", &self.issued_count.load(Ordering::Relaxed))?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for AutoStrategy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct AutoStrategySerde {
            sobol: SobolStrategy<DynSpace>,
            gmm: GmmStrategy<DynSpace>,
            exploration_budget: usize,
            trial_count: usize,
            #[serde(default)]
            issued_count: Option<usize>,
        }

        let state = AutoStrategySerde::deserialize(deserializer)?;
        let issued_count = state.issued_count.unwrap_or(state.trial_count);
        Ok(Self {
            sobol: state.sobol,
            gmm: state.gmm,
            exploration_budget: state.exploration_budget,
            trial_count: state.trial_count,
            issued_count: AtomicUsize::new(issued_count.max(state.trial_count)),
        })
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
                let issued = s.issued_count.fetch_add(1, Ordering::Relaxed);
                if issued < s.exploration_budget {
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
                s.issued_count.fetch_max(s.trial_count, Ordering::Relaxed);
                s.sobol.update(candidate, observation);
                s.gmm.update(candidate, observation);
            }
        }
    }
}

impl DynStrategy {
    /// The exploration budget of the underlying `auto` strategy, if any.
    ///
    /// Used after a resume to re-anchor the refit schedule to the checkpoint's
    /// real budget rather than the default recomputed by `from_config`.
    fn exploration_budget(&self) -> Option<usize> {
        match &self.inner {
            DynStrategyInner::Auto(s) => Some(s.exploration_budget),
            _ => None,
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

    fn reconcile_after_refit(&mut self, live: &Self) {
        match (&mut self.inner, &live.inner) {
            (DynStrategyInner::Auto(s), DynStrategyInner::Auto(l)) => {
                // refit rebuilt only the GMM model; everything else is live
                // sampling state that advanced while the refit ran off-lock.
                s.sobol = l.sobol.clone();
                s.trial_count = l.trial_count;
                s.issued_count
                    .store(l.issued_count.load(Ordering::Relaxed), Ordering::Relaxed);
            }
            (DynStrategyInner::Gmm(s), DynStrategyInner::Gmm(l)) => s.reconcile_after_refit(l),
            // Sobol/Random refit is a no-op, so the off-lock snapshot is stale;
            // keep the live sampler (its advanced index/counter) intact.
            (DynStrategyInner::Sobol(s), DynStrategyInner::Sobol(l)) => *s = l.clone(),
            (DynStrategyInner::Random(s), DynStrategyInner::Random(l)) => *s = l.clone(),
            _ => {}
        }
    }
}

// =============================================================================
// Configuration types for constructing HolaEngine from YAML/JSON
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ParamConfig {
    Real {
        min: f64,
        max: f64,
        #[serde(default = "default_scale")]
        scale: String,
    },
    Integer {
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
    /// Optional cap on the number of trials retained in the leaderboard.
    ///
    /// When `None` (the default) the leaderboard is unbounded and retains every
    /// completed trial. When set to `Some(n)`,
    /// the leaderboard keeps at most `n` trials; once full, each new trial evicts
    /// one existing trial per the leaderboard's documented eviction policy. This
    /// is opt-in and intended for very long-running studies where the full trial
    /// history would otherwise grow without bound.
    #[serde(default)]
    pub max_leaderboard_size: Option<usize>,
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

fn validate_study_config(config: &StudyConfig) -> Result<(), String> {
    validate_space_config(&config.space)?;
    validate_objectives(&config.objectives)?;
    if let Some(strategy) = &config.strategy {
        validate_strategy_config(strategy)?;
    }
    if let Some(checkpoint) = &config.checkpoint
        && checkpoint.interval == 0
    {
        return Err("checkpoint.interval must be at least 1".to_string());
    }
    if config.max_leaderboard_size == Some(0) {
        return Err("max_leaderboard_size must be at least 1".to_string());
    }
    Ok(())
}

fn validate_space_config(space: &BTreeMap<String, ParamConfig>) -> Result<(), String> {
    if space.is_empty() {
        return Err("At least one parameter is required".to_string());
    }

    for (name, param) in space {
        if name.trim().is_empty() {
            return Err("Parameter names must not be empty".to_string());
        }
        match param {
            ParamConfig::Real { min, max, scale } => {
                if !min.is_finite() || !max.is_finite() {
                    return Err(format!(
                        "Parameter '{name}': real bounds must be finite, got min={min}, max={max}",
                    ));
                }
                if min >= max {
                    return Err(format!(
                        "Parameter '{name}': min must be less than max, got min={min}, max={max}",
                    ));
                }
                match scale.as_str() {
                    "linear" => {
                        // The linear internal span is (max - min). If that
                        // subtraction overflows to a non-finite value the space
                        // would silently collapse every point to a fixed value
                        // (the to_unit_cube/from_unit_cube degenerate guards), so
                        // reject it here. Log/ln/log10 spans are ln(max)-ln(min),
                        // which stay finite for any finite positive min < max.
                        if !(max - min).is_finite() {
                            return Err(format!(
                                "Parameter '{name}': range too large; max - min overflows to a non-finite value",
                            ));
                        }
                    }
                    "log" | "ln" | "log10" => {
                        if *min <= 0.0 || *max <= 0.0 {
                            return Err(format!(
                                "Parameter '{name}': {scale} scale requires min > 0 and max > 0, got min={min}, max={max}",
                            ));
                        }
                    }
                    other => {
                        return Err(format!(
                            "Parameter '{name}': unknown real scale '{other}'. Expected one of: linear, log, ln, log10",
                        ));
                    }
                }
            }
            ParamConfig::Integer { min, max } => {
                if min > max {
                    return Err(format!(
                        "Parameter '{name}': integer min must be <= max, got min={min}, max={max}",
                    ));
                }
            }
            ParamConfig::Categorical { choices } => {
                if choices.is_empty() {
                    return Err(format!(
                        "Parameter '{name}': categorical choices must not be empty",
                    ));
                }
            }
        }
    }

    Ok(())
}

fn validate_objectives(objectives: &[ObjectiveConfig]) -> Result<(), String> {
    if objectives.is_empty() {
        return Err("At least one objective is required. \
             Example: objectives: [{ field: \"loss\", type: \"minimize\" }]"
            .to_string());
    }

    for obj in objectives {
        if obj.field.trim().is_empty() {
            return Err("Objective field names must not be empty".to_string());
        }
        match obj.obj_type.as_str() {
            "minimize" | "maximize" => {}
            other => {
                return Err(format!(
                    "Objective '{}': unknown objective type '{}'. Expected 'minimize' or 'maximize'",
                    obj.field, other
                ));
            }
        }
        if !obj.priority.is_finite() || obj.priority < 0.0 {
            return Err(format!(
                "Objective '{}': priority must be finite and non-negative, got {}",
                obj.field, obj.priority
            ));
        }
        if let Some(target) = obj.target
            && !target.is_finite()
        {
            return Err(format!(
                "Objective '{}': target must be finite, got {}",
                obj.field, target
            ));
        }
        if let Some(limit) = obj.limit
            && !limit.is_finite()
        {
            return Err(format!(
                "Objective '{}': limit must be finite, got {}",
                obj.field, limit
            ));
        }

        // When both bounds are given, their ordering encodes the optimization
        // direction (target is the value to reach, limit the worst acceptable
        // value). Reject orderings that contradict the declared `type` so a
        // misconfigured objective fails loudly instead of silently optimizing
        // in the wrong direction.
        if let (Some(target), Some(limit)) = (obj.target, obj.limit) {
            match obj.obj_type.as_str() {
                "minimize" if target >= limit => {
                    return Err(format!(
                        "Objective '{}': a 'minimize' objective requires target < limit, \
                         got target={target}, limit={limit}. \
                         To maximize, use type: maximize with target > limit.",
                        obj.field
                    ));
                }
                "maximize" if target <= limit => {
                    return Err(format!(
                        "Objective '{}': a 'maximize' objective requires target > limit, \
                         got target={target}, limit={limit}. \
                         To minimize, use type: minimize with target < limit.",
                        obj.field
                    ));
                }
                _ => {}
            }
        }
    }

    Ok(())
}

fn validate_strategy_config(strategy: &StrategyConfig) -> Result<(), String> {
    match strategy.strategy_type.as_str() {
        "random" | "sobol" | "gmm" | "auto" => {}
        other => {
            return Err(format!(
                "Unknown strategy type '{other}'. Expected one of: random, sobol, gmm, auto",
            ));
        }
    }

    if strategy.refit_interval == 0 {
        return Err("strategy.refit_interval must be at least 1".to_string());
    }
    if let Some(elite_fraction) = strategy.elite_fraction
        && (!elite_fraction.is_finite() || elite_fraction <= 0.0 || elite_fraction > 1.0)
    {
        return Err(format!(
            "strategy.elite_fraction must be finite and in (0, 1], got {elite_fraction}",
        ));
    }

    Ok(())
}

// =============================================================================
// HolaEngine: the top-level Ask/Tell interface
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

/// Kind of checkpoint loaded by [`HolaEngine::load_checkpoint_with_fallback`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointLoadKind {
    /// Full checkpoint with leaderboard and strategy state.
    Full,
    /// Legacy leaderboard-only checkpoint with trial history but no strategy state.
    Leaderboard,
}

impl CheckpointLoadKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CheckpointLoadKind::Full => "full",
            CheckpointLoadKind::Leaderboard => "leaderboard",
        }
    }
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
    fn for_objectives(objectives: &[ObjectiveConfig]) -> Self {
        if count_priority_groups(objectives) > 1 {
            DynLeaderboard::Vector(Leaderboard::new())
        } else {
            DynLeaderboard::Scalar(Leaderboard::new())
        }
    }

    /// Apply an optional retention cap to the underlying leaderboard.
    ///
    /// `None` leaves the leaderboard unbounded (default). `Some(n)` caps it at
    /// `n` trials using the leaderboard's documented eviction policy. Delegates
    /// to the inner `Leaderboard` so a single eviction policy is shared by both
    /// the scalar and vector topologies.
    fn set_max_size(&mut self, max_size: Option<usize>) {
        match self {
            DynLeaderboard::Scalar(lb) => lb.set_max_size(max_size),
            DynLeaderboard::Vector(lb) => lb.set_max_size(max_size),
        }
    }

    /// The current retention cap, if any. `None` means unbounded.
    fn max_size(&self) -> Option<usize> {
        match self {
            DynLeaderboard::Scalar(lb) => lb.max_size(),
            DynLeaderboard::Vector(lb) => lb.max_size(),
        }
    }

    fn push_with_raw(
        &mut self,
        trial_id: u64,
        candidate: serde_json::Value,
        raw_metrics: serde_json::Value,
        objectives: &[ObjectiveConfig],
    ) -> (u64, f64) {
        match self {
            DynLeaderboard::Scalar(lb) => {
                let score = scalarize_raw(&raw_metrics, objectives);
                let id = lb.push_with_raw_trial_id(candidate, score, raw_metrics, trial_id);
                (id, score)
            }
            DynLeaderboard::Vector(lb) => {
                let obs = vectorize_raw(&raw_metrics, objectives);
                let score = scalarize_observation(&obs, objectives);
                let id = lb.push_with_raw_trial_id(candidate, obs, raw_metrics, trial_id);
                (id, score)
            }
        }
    }

    /// The observation topology of this leaderboard, used to tag checkpoints
    /// so loads do not have to guess Scalar-vs-Vector from current objectives.
    fn observation_kind(&self) -> ObservationKind {
        match self {
            DynLeaderboard::Scalar(_) => ObservationKind::Scalar,
            DynLeaderboard::Vector(_) => ObservationKind::Vector,
        }
    }

    fn contains_trial_id(&self, trial_id: u64) -> bool {
        match self {
            DynLeaderboard::Scalar(lb) => lb.get(trial_id).is_some(),
            DynLeaderboard::Vector(lb) => lb.get(trial_id).is_some(),
        }
    }

    fn next_trial_id(&self) -> u64 {
        match self {
            DynLeaderboard::Scalar(lb) => lb.next_trial_id(),
            DynLeaderboard::Vector(lb) => lb.next_trial_id(),
        }
    }

    /// Monotonic count of trials ever pushed, forwarded from the inner
    /// `Leaderboard`. Increments on every push and is never decremented by
    /// eviction, so for an unbounded board it equals `len()` and for a capped
    /// one it keeps growing past the cap. This is the correct basis for the
    /// `max_trials` stopping check.
    fn total_completed(&self) -> u64 {
        match self {
            DynLeaderboard::Scalar(lb) => lb.total_completed(),
            DynLeaderboard::Vector(lb) => lb.total_completed(),
        }
    }

    fn normalize_next_trial_id(&mut self) -> u64 {
        match self {
            DynLeaderboard::Scalar(lb) => lb.normalize_next_trial_id(),
            DynLeaderboard::Vector(lb) => lb.normalize_next_trial_id(),
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

    fn migrate_for_objectives(&mut self, objectives: &[ObjectiveConfig]) {
        let should_be_vector = count_priority_groups(objectives) > 1;
        match (&mut *self, should_be_vector) {
            (DynLeaderboard::Scalar(_), false) | (DynLeaderboard::Vector(_), true) => {
                self.rescalarize(objectives);
                return;
            }
            _ => {}
        }

        // Rebuilding the board with a fresh Leaderboard would drop the configured
        // retention cap, so capture it here and re-apply it to the migrated board
        // below, keeping a bounded study bounded across an objective change.
        let max_size = self.max_size();
        // Rebuilding also re-pushes only the retained (post-eviction) trials, so a
        // fresh board's total_completed would collapse to the retained count and
        // lose history. Carry the prior count over so the monotonic completed
        // counter that backs the max_trials stopping check keeps growing past the
        // cap across an objective change. set_total_completed clamps to
        // max(prior_total, len), and prior_total is always >= the retained
        // length, so the carried value wins.
        let prior_total = self.total_completed();
        let migrated = match self {
            DynLeaderboard::Scalar(lb) => {
                let mut migrated = Leaderboard::new();
                for trial in lb.iter() {
                    let raw_metrics = trial.raw_metrics.clone();
                    let raw = raw_metrics.as_ref().unwrap_or(&serde_json::Value::Null);
                    migrated.push_existing_trial(Trial {
                        candidate: trial.candidate.clone(),
                        observation: vectorize_raw(raw, objectives),
                        raw_metrics,
                        trial_id: trial.trial_id,
                        timestamp: trial.timestamp,
                    });
                }
                migrated.set_total_completed(prior_total);
                DynLeaderboard::Vector(migrated)
            }
            DynLeaderboard::Vector(lb) => {
                let mut migrated = Leaderboard::new();
                for trial in lb.iter() {
                    let raw_metrics = trial.raw_metrics.clone();
                    let raw = raw_metrics.as_ref().unwrap_or(&serde_json::Value::Null);
                    migrated.push_existing_trial(Trial {
                        candidate: trial.candidate.clone(),
                        observation: scalarize_raw(raw, objectives),
                        raw_metrics,
                        trial_id: trial.trial_id,
                        timestamp: trial.timestamp,
                    });
                }
                migrated.set_total_completed(prior_total);
                DynLeaderboard::Scalar(migrated)
            }
        };
        *self = migrated;
        self.set_max_size(max_size);
    }

    /// Get a single completed trial by ID, computing its rank and Pareto front.
    fn get_completed(
        &self,
        trial_id: u64,
        include_infeasible: bool,
        objectives: &[ObjectiveConfig],
    ) -> Option<CompletedTrial> {
        match self {
            DynLeaderboard::Scalar(lb) => {
                let trial = lb.get(trial_id)?.clone();
                if !include_infeasible
                    && !Leaderboard::<serde_json::Value, f64>::trial_is_feasible(&trial)
                {
                    return None;
                }

                let rank = lb
                    .iter()
                    .filter(|other| {
                        include_infeasible
                            || Leaderboard::<serde_json::Value, f64>::trial_is_feasible(other)
                    })
                    .filter(|other| {
                        other
                            .observation
                            .partial_cmp(&trial.observation)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| other.trial_id.cmp(&trial.trial_id))
                            == std::cmp::Ordering::Less
                    })
                    .count();

                Some(build_completed_scalar(trial, rank, objectives))
            }
            DynLeaderboard::Vector(_) => {
                let all = self.completed_trials("rank", include_infeasible, objectives);
                all.into_iter().find(|ct| ct.trial_id == trial_id)
            }
        }
    }

    /// Build the just-told trial's view cheaply, keeping any expensive global
    /// ranking out from under the caller's write lock.
    ///
    /// The scalar path computes its O(n) `rank_of` directly and returns a fully
    /// populated `CompletedTrial`. The vector path fills `pareto_front` via the
    /// front-peeling `pareto_rank_of` (no trial clones) and returns a cheap
    /// `(trial_id, observation)` snapshot of the participating trials plus the
    /// target id, so the caller can finish the NSGA-II global rank off-lock with
    /// [`vector_global_rank`] instead of running the full ranking (and cloning
    /// every trial) while the write lock is held.
    #[allow(clippy::type_complexity)]
    fn completed_for_tell(
        &self,
        trial_id: u64,
        include_infeasible: bool,
        objectives: &[ObjectiveConfig],
    ) -> Option<(
        CompletedTrial,
        Option<(Vec<(u64, BTreeMap<String, f64>)>, u64)>,
    )> {
        match self {
            DynLeaderboard::Scalar(_) => {
                let completed = self.get_completed(trial_id, include_infeasible, objectives)?;
                Some((completed, None))
            }
            DynLeaderboard::Vector(lb) => {
                let trial = lb.get(trial_id)?.clone();
                let pareto_front = lb.pareto_rank_of(trial_id, include_infeasible)?;
                let completed = build_completed_vector(trial, pareto_front, objectives);
                let snapshot: Vec<(u64, BTreeMap<String, f64>)> = lb
                    .iter()
                    .filter(|t| {
                        include_infeasible
                            || Leaderboard::<serde_json::Value, BTreeMap<String, f64>>::trial_is_feasible(
                                t,
                            )
                    })
                    .map(|t| (t.trial_id, t.observation.clone()))
                    .collect();
                Some((completed, Some((snapshot, trial_id))))
            }
        }
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

/// Whether observation `a` dominates `b` (no worse in any group, strictly
/// better in at least one), assuming minimization. Mirrors the leaderboard's
/// own domination relation so the off-lock rank below matches NSGA-II exactly.
fn observation_dominates(a: &BTreeMap<String, f64>, b: &BTreeMap<String, f64>) -> bool {
    let mut dominated_some = false;
    for (key, &va) in a {
        let vb = b.get(key).copied().unwrap_or(f64::INFINITY);
        if va > vb {
            return false;
        }
        if va < vb {
            dominated_some = true;
        }
    }
    for key in b.keys() {
        if !a.contains_key(key) {
            return false;
        }
    }
    dominated_some
}

/// Compute a single trial's 0-indexed NSGA-II global rank from a cheap snapshot
/// of `(trial_id, observation)` pairs, without cloning trials or building the
/// full ranked view.
///
/// `participants` must already be filtered to the same set the leaderboard's
/// ranked view would use (feasible-only or all). The returned rank reproduces
/// the ordering of `Leaderboard::ranked_trials`/`ranked_trials_all`: fronts are
/// concatenated in non-domination order and, because that view selects whole
/// fronts when ranking every trial, members keep their snapshot (iteration)
/// order within a front. The global rank is therefore the number of trials in
/// earlier fronts plus the target's position within its own front. Returns
/// `None` if the target is not present in `participants`.
fn vector_global_rank(participants: &[(u64, BTreeMap<String, f64>)], target: u64) -> Option<usize> {
    let n = participants.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            if observation_dominates(&participants[i].1, &participants[j].1) {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if observation_dominates(&participants[j].1, &participants[i].1) {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut current: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();
    let mut rank_base = 0usize;
    while !current.is_empty() {
        if let Some(pos) = current.iter().position(|&i| participants[i].0 == target) {
            return Some(rank_base + pos);
        }
        rank_base += current.len();
        let mut next = Vec::new();
        for &i in &current {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next.push(j);
                }
            }
        }
        current = next;
    }
    None
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
            Some(v) => {
                // An infinite objective score means infeasible and must stay
                // infinite regardless of priority. Multiplying through would turn
                // the legitimate priority == 0.0 case into 0.0 * INFINITY = NaN,
                // which would silently corrupt the reported score, so keep it
                // infinite.
                let s = objective_score(v, &obj.obj_type, obj.target, obj.limit);
                if s.is_infinite() { s } else { s * obj.priority }
            }
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
/// A self-contained optimization engine built on `opt_engine`'s building blocks
/// (spaces, strategies, leaderboard, scales, and objectives), composed behind
/// type-erased JSON interfaces and HOLA-specific orchestration: TLP
/// scalarization and ranking, auto-refitting, checkpointing, and the
/// CompletedTrial view.
///
/// Cloning a `HolaEngine` is cheap (Arc reference-count bumps). Both the
/// original and the clone share the same underlying state, so changes made
/// through one are visible through the other.
#[derive(Clone)]
pub struct HolaEngine {
    space: DynSpace,
    state: Arc<RwLock<HolaEngineState>>,
    /// Serializes refits so a stale off-lock fit cannot overwrite a newer model.
    /// Cheap to clone (Arc); shared across engine clones like `state`.
    refit_lock: Arc<Mutex<()>>,
    refit_config: Option<RefitConfig>,
    /// The effective strategy settings this engine was built with (strategy
    /// type, refit interval, elite fraction, seed, exploration budget). Emitted
    /// by `study_config()` so a checkpoint records the real values; on resume
    /// `from_config` then rebuilds an identical `refit_config` instead of one
    /// anchored to a default exploration budget.
    strategy_template: Option<StrategyConfig>,
    auto_checkpoint: Option<AutoCheckpointConfig>,
    max_trials: Option<usize>,
    /// Opt-in leaderboard retention cap (`None` = unbounded). Recorded here so
    /// `study_config()` can emit it into checkpoints and a resumed study rebuilds
    /// with the same bound.
    max_leaderboard_size: Option<usize>,
}

struct HolaEngineState {
    strategy: DynStrategy,
    leaderboard: DynLeaderboard,
    /// Objectives live here, behind the same lock as the leaderboard, so a
    /// `tell()` reads objectives + scalarizes + pushes atomically, and
    /// `update_objectives` swaps objectives + migrates the leaderboard
    /// atomically. Sharing one lock keeps a concurrent update from changing
    /// objectives between a `tell()`'s read and its push, which would scalarize
    /// against stale objectives or misclassify the trial.
    objectives: Vec<ObjectiveConfig>,
    next_pending_id: u64,
    pending: BTreeMap<u64, serde_json::Value>,
    cancelled: HashSet<u64>,
}

impl HolaEngineState {
    fn reset_transient_trial_state_after_load(&mut self) {
        self.next_pending_id = self.leaderboard.normalize_next_trial_id();
        self.pending.clear();
        self.cancelled.clear();
    }

    /// Record a cancelled trial id and bound the retained set.
    ///
    /// The set only feeds the tell-after-cancel rejection message (reuse is
    /// already impossible via the monotonic `next_pending_id`), so when it grows
    /// past [`MAX_CANCELLED_RETAINED`] the oldest (smallest) ids are dropped,
    /// keeping the newest window. This caps memory over a long run while
    /// preserving the rejection guard for trials cancelled recently enough that a
    /// late `tell` could still arrive.
    fn record_cancelled(&mut self, trial_id: u64) {
        self.cancelled.insert(trial_id);
        if self.cancelled.len() > MAX_CANCELLED_RETAINED {
            // Drop the smallest ids first: a smaller id was cancelled earlier and
            // is the least likely to still have an in-flight tell.
            let drop_count = self.cancelled.len() - MAX_CANCELLED_RETAINED;
            let mut ids: Vec<u64> = self.cancelled.iter().copied().collect();
            ids.sort_unstable();
            for id in ids.into_iter().take(drop_count) {
                self.cancelled.remove(&id);
            }
        }
    }
}

impl HolaEngine {
    /// Build a HolaEngine from a StudyConfig (parsed from YAML/JSON).
    pub fn from_config(config: StudyConfig) -> Result<Self, String> {
        validate_study_config(&config)?;

        let mut space = DynSpace::new();
        for (name, param) in &config.space {
            space = match param {
                ParamConfig::Real { min, max, scale } => match scale.as_str() {
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
                        space.add_real_log(name, *min, *max)
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
                        space.add_real_log10(name, *min, *max)
                    }
                    _ => space.add_real(name, *min, *max),
                },
                ParamConfig::Integer { min, max } => space.add_integer(name, *min, *max),
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

        // Primary guard for the Sobol dimension limit. The Sobol backend ships
        // only 256 dimensions of direction numbers and panics in release past
        // that. A pure "sobol" strategy cannot serve a higher-dimensional space,
        // so reject it with a clear error before any sampling reaches the
        // backend. "auto" and "gmm" use Sobol only for their exploration phase,
        // which falls back to uniform random sampling above the limit while GMM
        // exploitation (the valuable part in high dimensions) is unaffected, so
        // they are allowed with a single warning.
        if dim > MAX_SOBOL_DIMS {
            match strategy_type {
                "sobol" => {
                    return Err(format!(
                        "the 'sobol' strategy supports at most {MAX_SOBOL_DIMS} dimensions \
                         (this space has {dim}); use 'random', 'gmm', or 'auto' for \
                         higher-dimensional spaces."
                    ));
                }
                "auto" | "gmm" => {
                    eprintln!(
                        "[hola] Warning: space has {dim} dimensions (> {MAX_SOBOL_DIMS}); \
                         the Sobol exploration phase falls back to uniform random sampling. \
                         GMM exploitation is unaffected."
                    );
                }
                _ => {}
            }
        }

        let refit_interval = strategy_cfg.map(|s| s.refit_interval).unwrap_or(20);
        let seed = strategy_cfg.and_then(|s| s.seed);
        let max_trials = config
            .max_trials
            .or_else(|| strategy_cfg.and_then(|s| s.total_budget));

        // Track the effective exploration budget and elite fraction so the
        // template recorded for resume carries the real values rather than
        // letting a later `from_config` recompute them from defaults.
        let mut effective_exploration_budget: Option<usize> = None;
        let mut effective_elite_fraction: Option<f64> = None;
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
                        // Fold high bits into low instead of truncating so seeds
                        // differing only in bits >= 32 produce distinct sequences.
                        seed.map(|s| (s ^ (s >> 32)) as u32).unwrap_or(42),
                    )),
                },
                None,
            ),
            // "gmm" (default): Sobol exploration followed by GMM exploitation
            "gmm" | "auto" => {
                let exploration_budget = strategy_cfg
                    .and_then(|s| s.exploration_budget)
                    .unwrap_or_else(|| {
                        let total = max_trials.unwrap_or(200);
                        AutoStrategy::default_exploration_budget(total, dim)
                    });
                let elite_fraction = strategy_cfg.and_then(|s| s.elite_fraction).unwrap_or(0.25);
                effective_exploration_budget = Some(exploration_budget);
                effective_elite_fraction = Some(elite_fraction);
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
                        elite_fraction,
                    )),
                )
            }
            _ => unreachable!("strategy type was validated before construction"),
        };

        // Record the effective strategy settings so `study_config()` can emit a
        // concrete `StrategyConfig` into checkpoints. On resume this lets
        // `from_config` rebuild the identical `refit_config` (anchored to the
        // real exploration budget), keeping the refit schedule continuous.
        let strategy_template = Some(StrategyConfig {
            strategy_type: strategy_type.to_string(),
            refit_interval,
            total_budget: max_trials,
            exploration_budget: effective_exploration_budget,
            seed,
            elite_fraction: effective_elite_fraction,
        });

        let auto_checkpoint = config.checkpoint.as_ref().map(|c| {
            let mut ac = AutoCheckpointConfig::new(&c.directory, c.interval);
            ac.max_checkpoints = c.max_checkpoints;
            ac
        });

        let mut leaderboard = DynLeaderboard::for_objectives(&config.objectives);
        // Opt-in bounded mode. None (default) keeps the leaderboard unbounded
        // and retains every completed trial.
        leaderboard.set_max_size(config.max_leaderboard_size);

        Ok(Self {
            space,
            state: Arc::new(RwLock::new(HolaEngineState {
                strategy,
                leaderboard,
                objectives: config.objectives,
                next_pending_id: 0,
                pending: BTreeMap::new(),
                cancelled: HashSet::new(),
            })),
            refit_lock: Arc::new(Mutex::new(())),
            refit_config,
            strategy_template,
            auto_checkpoint,
            max_trials,
            max_leaderboard_size: config.max_leaderboard_size,
        })
    }

    /// Ask for the next trial to evaluate.
    ///
    /// Returns an error if `max_trials` has been reached.
    pub async fn ask(&self) -> Result<DynTrial, String> {
        let mut state = self.state.write().await;
        if let Some(max) = self.max_trials {
            // Count distinct trials against the budget via the monotonic
            // total_completed() counter plus the in-flight pending trials.
            // total_completed() counts only successful pushes; it never counts
            // pending ids or cancelled-id gaps and is never decremented by
            // eviction, so completed/pending/cancelled stay disjoint and each
            // trial is counted once. This avoids next_trial_id()'s id-span,
            // which double-counts out-of-order pending trials (a pending id
            // below a completed id) and charges budget for cancelled trials.
            // For an unbounded board total_completed() equals len(), so the
            // default behavior is unchanged; for a capped board it keeps
            // growing past the cap, so a bounded study still terminates.
            let completed = state.leaderboard.total_completed() as usize;
            let total = completed + state.pending.len();
            if total >= max {
                return Err(format!(
                    "max_trials ({max}) reached ({completed} completed, {} pending)",
                    state.pending.len()
                ));
            }
        }
        let params = state.strategy.suggest(&self.space);
        let mut id = state.next_pending_id.max(state.leaderboard.next_trial_id());
        while state.pending.contains_key(&id)
            || state.cancelled.contains(&id)
            || state.leaderboard.contains_trial_id(id)
        {
            id = id
                .checked_add(1)
                .ok_or_else(|| "Exhausted trial ID space".to_string())?;
        }
        state.next_pending_id = id
            .checked_add(1)
            .ok_or_else(|| "Exhausted trial ID space".to_string())?;
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
        let mut state = self.state.write().await;

        if state.cancelled.contains(&trial_id) {
            return Err(format!("Trial {trial_id} has been cancelled"));
        }

        if state.leaderboard.contains_trial_id(trial_id) {
            return Err(format!("Trial {trial_id} has already been completed"));
        }

        let candidate = state
            .pending
            .remove(&trial_id)
            .ok_or_else(|| format!("Unknown trial_id: {trial_id}"))?;

        // Read objectives, scalarize, and push under the single state lock so a
        // concurrent update_objectives cannot scalarize this trial against a
        // half-applied objective set.
        let objectives = state.objectives.clone();
        let (stored_trial_id, score) =
            state
                .leaderboard
                .push_with_raw(trial_id, candidate.clone(), raw_metrics, &objectives);
        if stored_trial_id != trial_id {
            return Err(format!(
                "Internal trial ID mismatch: pending trial {trial_id} was stored as {stored_trial_id}"
            ));
        }
        state.strategy.update(&candidate, score);

        let n_trials = state.leaderboard.len();

        // Build this trial's view cheaply under the lock. The scalar path returns
        // a fully ranked CompletedTrial (its rank_of is O(n)). The vector path
        // fills pareto_front via front-peeling (no trial clones) and returns a
        // light (trial_id, observation) snapshot so the O(M*N^2) NSGA-II global
        // rank is computed off-lock below, rather than re-ranking and cloning the
        // whole board while the write lock is held. The resulting CompletedTrial
        // is identical in content to the prior full-ranking path.
        let (mut completed, vector_rank_inputs) = state
            .leaderboard
            .completed_for_tell(stored_trial_id, true, &objectives)
            .ok_or_else(|| format!("Failed to build CompletedTrial for {stored_trial_id}"))?;
        drop(state);

        // Vector studies: finish the global NSGA-II rank off-lock from the cheap
        // observation snapshot. Scalar studies already have their rank set.
        if let Some((participants, target)) = vector_rank_inputs {
            completed.rank = vector_global_rank(&participants, target)
                .ok_or_else(|| format!("Failed to rank CompletedTrial for {stored_trial_id}"))?;
        }

        // Auto-refit if configured
        if let Some(ref config) = self.refit_config
            && config.should_refit(n_trials)
            // Skip this periodic refit if one is already in flight: it rebuilds
            // from the leaderboard and yields a current model, so dropping a
            // boundary is harmless. Holding the guard across the fit prevents a
            // stale model from overwriting a newer one.
            && let Ok(_refit_guard) = self.refit_lock.try_lock()
        {
            let state_guard = self.state.read().await;
            let k = config.selection_count(n_trials);
            // Refit against the current objectives, which may have changed since
            // this trial was scored if an update_objectives ran concurrently.
            let refit_objectives = state_guard.objectives.clone();
            let trials = state_guard
                .leaderboard
                .top_k_for_refit(k, &refit_objectives);
            let mut strategy_snapshot = state_guard.strategy.clone();
            let space_clone = self.space.clone();
            drop(state_guard);

            let mut fitted = tokio::task::spawn_blocking(move || {
                use opt_engine::traits::RefittableStrategy;
                strategy_snapshot.refit(&space_clone, &trials);
                strategy_snapshot
            })
            .await
            .map_err(|e| e.to_string())?;

            {
                use opt_engine::traits::RefittableStrategy;
                let mut guard = self.state.write().await;
                fitted.reconcile_after_refit(&guard.strategy);
                guard.strategy = fitted;
            }
        }

        // Auto-checkpoint if configured
        if let Some(ref config) = self.auto_checkpoint
            && config.should_checkpoint(n_trials)
        {
            // Save a full checkpoint (leaderboard + strategy_state + config) so
            // resuming from an auto-checkpoint restores strategy/exploration
            // progress instead of resetting it.
            if let Err(e) = self
                .save_full_checkpoint(
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
            state.record_cancelled(trial_id);
            Ok(())
        } else {
            Err(format!(
                "Trial {trial_id} is not pending (may be completed or unknown)"
            ))
        }
    }

    /// Get the top-k trials by rank.
    pub async fn top_k(&self, k: usize, include_infeasible: bool) -> Vec<CompletedTrial> {
        let state = self.state.read().await;
        state
            .leaderboard
            .top_k_completed(k, include_infeasible, &state.objectives)
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
        let state = self.state.read().await;
        state
            .leaderboard
            .pareto_front_completed(front, include_infeasible, &state.objectives)
    }

    /// Get a single completed trial by ID with scoring and ranking.
    pub async fn completed_trial(
        &self,
        trial_id: u64,
        include_infeasible: bool,
    ) -> Option<CompletedTrial> {
        let state = self.state.read().await;
        state
            .leaderboard
            .get_completed(trial_id, include_infeasible, &state.objectives)
    }

    /// Get all trials with scoring and ranking.
    pub async fn trials(&self, sorted_by: &str, include_infeasible: bool) -> Vec<CompletedTrial> {
        let state = self.state.read().await;
        state
            .leaderboard
            .completed_trials(sorted_by, include_infeasible, &state.objectives)
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
        // Emit the effective strategy settings, refreshing exploration_budget
        // from the live strategy. The deserialized strategy state still
        // overrides sampling on resume; carrying the real budget here only fixes
        // the refit schedule, which `from_config` would otherwise re-anchor to a
        // default exploration budget when strategy is None.
        let state = self.state.read().await;
        let strategy = self.strategy_template.clone().map(|mut tmpl| {
            if let Some(budget) = state.strategy.exploration_budget() {
                tmpl.exploration_budget = Some(budget);
            }
            tmpl
        });
        let objectives = state.objectives.clone();
        drop(state);
        StudyConfig {
            space,
            objectives,
            strategy,
            checkpoint: None,
            max_trials: self.max_trials,
            max_leaderboard_size: self.max_leaderboard_size,
        }
    }

    /// Get the current objectives configuration.
    pub async fn objectives(&self) -> Vec<ObjectiveConfig> {
        self.state.read().await.objectives.clone()
    }

    /// Re-scalarize all trials using the current objectives.
    pub async fn rescalarize(&self) {
        let mut state = self.state.write().await;
        let objectives = state.objectives.clone();
        state.leaderboard.rescalarize(&objectives);
    }

    /// Update objectives and re-scalarize (for mid-run dashboard adjustments).
    ///
    /// Persists the new objectives so that subsequent `tell()` calls use the
    /// updated scalarization. If a refittable strategy (e.g., GMM) is configured,
    /// a refit is triggered immediately so the sampling distribution reflects
    /// the new objective weights.
    pub async fn update_objectives(&self, objectives: Vec<ObjectiveConfig>) -> Result<(), String> {
        validate_objectives(&objectives)?;

        // Swap objectives and migrate the leaderboard atomically under one write
        // lock so no concurrent tell() observes a half-updated state (new
        // objectives but an un-migrated leaderboard, or vice versa).
        let n_trials = {
            let mut state = self.state.write().await;
            state.objectives = objectives.clone();
            state.leaderboard.migrate_for_objectives(&objectives);
            state.leaderboard.len()
        };

        // Trigger an immediate refit so the strategy reflects the new scalarization.
        // Wait for any in-flight refit (do not skip): this refit must run against
        // the new objectives, and waiting guarantees its model is not clobbered.
        if let Some(ref config) = self.refit_config
            && n_trials >= config.min_trials
        {
            let _refit_guard = self.refit_lock.lock().await;
            let state_guard = self.state.read().await;
            let k = config.selection_count(n_trials);
            let trials = state_guard.leaderboard.top_k_for_refit(k, &objectives);
            let mut strategy_snapshot = state_guard.strategy.clone();
            let space_clone = self.space.clone();
            drop(state_guard);

            if let Ok(mut fitted) = tokio::task::spawn_blocking(move || {
                use opt_engine::traits::RefittableStrategy;
                strategy_snapshot.refit(&space_clone, &trials);
                strategy_snapshot
            })
            .await
            {
                use opt_engine::traits::RefittableStrategy;
                let mut guard = self.state.write().await;
                fitted.reconcile_after_refit(&guard.strategy);
                guard.strategy = fitted;
            }
        }
        Ok(())
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

    /// Load a checkpoint, preferring full checkpoints and falling back to
    /// legacy leaderboard-only files.
    ///
    /// This is used by CLI config `checkpoint.load_from`, which historically
    /// accepted leaderboard-only checkpoints. Full checkpoints preserve search
    /// strategy state; leaderboard-only checkpoints preserve completed trials.
    pub async fn load_checkpoint_with_fallback(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<CheckpointLoadKind> {
        let path = path.as_ref();
        // Read capped and version-gate the envelope before inspecting it, then
        // delegate to the matching loader (which reads and gates again on its
        // own load path).
        let bytes = read_checkpoint_capped(path)?;
        check_format_version_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let raw: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let has_strategy_state = raw
            .get("checkpoint")
            .unwrap_or(&raw)
            .get("strategy_state")
            .is_some();
        if has_strategy_state {
            self.load_full_checkpoint(path).await?;
            Ok(CheckpointLoadKind::Full)
        } else {
            self.load_leaderboard_checkpoint(path).await?;
            Ok(CheckpointLoadKind::Leaderboard)
        }
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
        let kind = state.leaderboard.observation_kind();
        match &state.leaderboard {
            DynLeaderboard::Scalar(lb) => {
                let mut cp = LeaderboardCheckpoint::new(lb.clone(), description);
                cp.observation_kind = kind;
                cp.save_json(path)
            }
            DynLeaderboard::Vector(lb) => {
                let mut cp = LeaderboardCheckpoint::new(lb.clone(), description);
                cp.observation_kind = kind;
                cp.save_json(path)
            }
        }
    }

    /// Load a leaderboard-only checkpoint, replacing the current trial history.
    ///
    /// The strategy is not restored; call with `refit()` afterward if using GMM.
    pub async fn load_leaderboard_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        // Read capped, version-gate, then parse from the bounded byte buffer so
        // an oversized or wrong-version file is rejected before deserialization.
        let bytes = read_checkpoint_capped(path.as_ref())?;
        check_format_version_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let raw: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut state = self.state.write().await;
        let current_is_vector = count_priority_groups(&state.objectives) > 1;

        // Determine the stored observation topology. Honor the `observation_kind`
        // tag when present so we deserialize the correct leaderboard type
        // regardless of the engine's current objectives. Older checkpoints
        // predate the tag; fall back to the current objective topology for them.
        let stored_kind = match raw.get("observation_kind") {
            Some(v) => Some(
                serde_json::from_value::<ObservationKind>(v.clone())
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
            ),
            None => None,
        };

        if let Some(kind) = stored_kind {
            let stored_is_vector = matches!(kind, ObservationKind::Vector);
            if stored_is_vector != current_is_vector {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "leaderboard checkpoint observation_kind ({}) does not match the \
                         current objective topology ({}); the checkpoint was saved with a \
                         different number of priority groups. Load it into a study whose \
                         objectives produce {} observations.",
                        if stored_is_vector { "vector" } else { "scalar" },
                        if current_is_vector {
                            "vector"
                        } else {
                            "scalar"
                        },
                        if stored_is_vector { "vector" } else { "scalar" },
                    ),
                ));
            }
        }

        // After the consistency check above, the stored topology equals the
        // current one, so deserialize the matching leaderboard type.
        let leaderboard = if current_is_vector {
            let cp: LeaderboardCheckpoint<serde_json::Value, BTreeMap<String, f64>> =
                serde_json::from_value(raw)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let n = cp.leaderboard.len();
            eprintln!("[hola] Loaded leaderboard checkpoint with {n} trials");
            DynLeaderboard::Vector(cp.leaderboard)
        } else {
            let cp: LeaderboardCheckpoint<serde_json::Value, f64> = serde_json::from_value(raw)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let n = cp.leaderboard.len();
            eprintln!("[hola] Loaded leaderboard checkpoint with {n} trials");
            DynLeaderboard::Scalar(cp.leaderboard)
        };

        state.leaderboard = leaderboard;
        // The loaded board carries whatever cap (if any) it was saved with;
        // re-apply the engine's configured cap so the bound stays consistent.
        state.leaderboard.set_max_size(self.max_leaderboard_size);
        state.reset_transient_trial_state_after_load();
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
        // Use a per-write-unique temp name so concurrent saves to the same target
        // do not clobber each other, and remove the temp on every error path so a
        // failed write never leaks a leftover temp.
        let tmp = unique_temp_path(path);
        let file = std::fs::File::create(&tmp)?;
        let result = (|| {
            let mut writer = std::io::BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, &wrapper)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let file = writer
                .into_inner()
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            file.sync_all()?;
            std::fs::rename(&tmp, path)
        })();
        match result {
            Ok(()) => {
                // Make the rename durable by fsyncing the parent directory.
                sync_parent_dir(path);
                Ok(())
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                Err(e)
            }
        }
    }

    /// Load a full checkpoint, restoring both leaderboard and strategy state.
    ///
    /// Handles both the new format (with `"config"` + `"checkpoint"` wrapper)
    /// and the legacy format (direct checkpoint without config).
    pub async fn load_full_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        // Read capped, version-gate, then parse from the bounded byte buffer so
        // an oversized or wrong-version file is rejected before deserialization.
        let bytes = read_checkpoint_capped(path.as_ref())?;
        check_format_version_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let raw: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // New format has "checkpoint" key; legacy format has "leaderboard" at root.
        let checkpoint_json = if raw.get("checkpoint").is_some() {
            raw.get("checkpoint").unwrap().clone()
        } else {
            raw
        };

        let mut state = self.state.write().await;
        if count_priority_groups(&state.objectives) > 1 {
            let cp: opt_engine::persistence::Checkpoint<
                serde_json::Value,
                BTreeMap<String, f64>,
                DynStrategy,
            > = serde_json::from_value(checkpoint_json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let n_loaded = cp.leaderboard.len();
            state.leaderboard = DynLeaderboard::Vector(cp.leaderboard);
            state.strategy = cp.strategy_state;
            state.reset_transient_trial_state_after_load();
            eprintln!("[hola] Loaded full checkpoint with {n_loaded} trials");
        } else {
            let cp: opt_engine::persistence::Checkpoint<serde_json::Value, f64, DynStrategy> =
                serde_json::from_value(checkpoint_json)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let n_loaded = cp.leaderboard.len();
            state.leaderboard = DynLeaderboard::Scalar(cp.leaderboard);
            state.strategy = cp.strategy_state;
            state.reset_transient_trial_state_after_load();
            eprintln!("[hola] Loaded full checkpoint with {n_loaded} trials");
        }
        // The loaded board carries whatever cap (if any) it was saved with;
        // re-apply the engine's configured cap so the bound stays consistent.
        state.leaderboard.set_max_size(self.max_leaderboard_size);
        Ok(())
    }

    /// Load a study from a checkpoint file, reconstructing the engine from the
    /// embedded `StudyConfig`.
    ///
    /// The checkpoint must have been saved with the new format that includes
    /// the `"config"` key. Returns an error if the config is missing (i.e.,
    /// the file was saved with an older version of HOLA).
    pub async fn load_from_checkpoint(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        // Read capped and version-gate the file before extracting the embedded
        // config, mirroring the gating on the data load path below.
        let bytes = read_checkpoint_capped(path.as_ref())
            .map_err(|e| format!("Failed to read checkpoint file: {e}"))?;
        check_format_version_bytes(&bytes)?;
        let raw: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Failed to parse checkpoint JSON: {e}"))?;

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
        let pattern = format!("{prefix}_");
        // Pair each candidate with its modified-time so we can order by age. A
        // lexicographic sort on the filenames mis-orders at the digit-count
        // boundary (e.g. "checkpoint_999999" sorts after "checkpoint_1000000"),
        // which would delete the newest file instead of the oldest. mtime is
        // unaffected by the zero-padding width changing.
        let mut checkpoints: Vec<(std::time::SystemTime, std::path::PathBuf)> =
            std::fs::read_dir(directory)
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
                .map(|p| {
                    let mtime = p
                        .metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    (mtime, p)
                })
                .collect();

        if checkpoints.len() <= max {
            return;
        }

        // Oldest first, so the leading `to_delete` entries are the ones to evict.
        checkpoints.sort_by_key(|(mtime, _)| *mtime);
        let to_delete = checkpoints.len() - max;
        for (_, path) in checkpoints.into_iter().take(to_delete) {
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
        // An infinite objective score means infeasible and must dominate the
        // sum regardless of priority. Multiplying through would turn the
        // legitimate priority == 0.0 case into 0.0 * INFINITY = NaN, which
        // would silently corrupt the score and ranking, so keep it infinite.
        let s = objective_score(val, &obj.obj_type, obj.target, obj.limit);
        total += if s.is_infinite() { s } else { s * obj.priority };
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

        // An infinite objective score means infeasible and must dominate the
        // group cost regardless of priority. Multiplying through would turn the
        // legitimate priority == 0.0 case into 0.0 * INFINITY = NaN, which
        // would silently corrupt the observation and ranking, so keep it infinite.
        let s = objective_score(val, &obj.obj_type, obj.target, obj.limit);
        let score = if s.is_infinite() { s } else { s * obj.priority };
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
        _ => f64::INFINITY,
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
            .add_real("lr", 0.0, 1.0)
            .add_integer("layers", 1, 10);

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
        let space = DynSpace::new().add_real_log10("lr", 1e-4, 0.1);
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
    fn test_zero_priority_infeasible_stays_infinite_not_nan() {
        // priority == 0.0 is legitimate ("ignore unless infeasible"). An
        // infeasible value (above limit) makes objective_score infinite; the
        // weighted contribution must stay +INFINITY, not become 0.0 * INF = NaN.
        let objectives = vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: Some(0.0),
            limit: Some(1.0),
            priority: 0.0,
            group: None,
        }];

        // value above the limit is infeasible.
        let raw = serde_json::json!({"loss": 2.0});

        let scalar = scalarize_raw(&raw, &objectives);
        assert!(!scalar.is_nan(), "scalar score must not be NaN");
        assert!(
            scalar.is_infinite() && scalar.is_sign_positive(),
            "infeasible objective must keep the scalar score at +INFINITY, got {scalar}"
        );

        let vec = vectorize_raw(&raw, &objectives);
        let group = vec.get("loss").copied().expect("group present");
        assert!(!group.is_nan(), "vector group cost must not be NaN");
        assert!(
            group.is_infinite() && group.is_sign_positive(),
            "infeasible objective must keep the group cost at +INFINITY, got {group}"
        );
    }

    #[test]
    fn test_compute_scores_zero_priority_infeasible_stays_infinite_not_nan() {
        // compute_scores also weights objective_score by priority, so a
        // priority == 0.0 infeasible objective must serialize as "inf", not
        // become 0.0 * INFINITY = NaN (which would serialize as a numeric
        // score and silently look feasible).
        let objectives = vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: Some(0.0),
            limit: Some(1.0),
            priority: 0.0,
            group: None,
        }];

        // value above the limit is infeasible.
        let raw = serde_json::json!({"loss": 2.0});

        let scores = compute_scores(&raw, &objectives);
        let entry = scores.get("loss").expect("score present");
        // The infinite-score branch serializes as the string "inf"; a NaN would
        // instead fall through to a finite numeric value.
        assert_eq!(
            entry,
            &serde_json::Value::from("inf"),
            "infeasible priority-0 objective must serialize as \"inf\", got {entry:?}"
        );
    }

    #[test]
    fn test_validate_objectives_rejects_type_ordering_mismatch() {
        let obj = |field: &str, ty: &str, target, limit| ObjectiveConfig {
            field: field.to_string(),
            obj_type: ty.to_string(),
            target: Some(target),
            limit: Some(limit),
            priority: 1.0,
            group: None,
        };

        // maximize with target < limit (and minimize with target > limit)
        // contradict the declared direction and must be rejected, not silently
        // optimized the wrong way.
        assert!(validate_objectives(&[obj("acc", "maximize", 0.0, 1.0)]).is_err());
        assert!(validate_objectives(&[obj("loss", "minimize", 1.0, 0.0)]).is_err());
        // target == limit is degenerate (neither < nor >) and also rejected.
        assert!(validate_objectives(&[obj("x", "minimize", 0.5, 0.5)]).is_err());

        // Consistent orderings are accepted.
        assert!(
            validate_objectives(&[
                obj("loss", "minimize", 0.0, 1.0),
                obj("acc", "maximize", 0.95, 0.5),
            ])
            .is_ok()
        );
    }

    #[test]
    fn test_validate_space_rejects_linear_span_overflow() {
        let study = |min: f64, max: f64, scale: &str| StudyConfig {
            space: BTreeMap::from([(
                "x".to_string(),
                ParamConfig::Real {
                    min,
                    max,
                    scale: scale.to_string(),
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
        };

        // A linear span of f64::MAX - (-f64::MAX) overflows to +inf, which would
        // silently collapse the space to a fixed value, so it must be rejected.
        let overflow = study(-f64::MAX, f64::MAX, "linear");
        assert!(validate_study_config(&overflow).is_err());
        assert!(validate_space_config(&overflow.space).is_err());

        // A large but finite linear span is fine.
        assert!(validate_space_config(&study(-1e6, 1e6, "linear").space).is_ok());

        // Log spans are ln(max) - ln(min), finite for any finite positive
        // bounds, so a positive-bounded log param stays accepted.
        assert!(validate_space_config(&study(1e-6, 1e6, "log").space).is_ok());
    }

    #[test]
    fn test_dyn_space_builder_api() {
        let space = DynSpace::new()
            .add_real("x", 0.0, 1.0)
            .add_real_log10("lr", 1e-4, 0.1)
            .add_integer("layers", 1, 10)
            .add_categorical("opt", vec!["adam".into(), "sgd".into()]);

        assert_eq!(space.dimensionality(), 4);
    }

    #[test]
    fn test_dyn_space_from_unit_cube_wrong_length() {
        let space = DynSpace::new()
            .add_real("x", 0.0, 1.0)
            .add_real("y", 0.0, 1.0);

        assert_eq!(space.dimensionality(), 2);
        // Too short: cannot fill every dimension.
        assert!(space.from_unit_cube(&[0.5]).is_none());
        // Exact length: accepted.
        assert!(space.from_unit_cube(&[0.5, 0.5]).is_some());
        // Too long: trailing coordinates would be silently dropped, so reject.
        assert!(space.from_unit_cube(&[0.5, 0.5, 0.5]).is_none());
    }

    #[test]
    fn test_dyn_space_to_unit_cube_non_object_midpoint() {
        // A non-object Value has no named parameters to read; to_unit_cube must
        // degrade gracefully to a midpoint vector instead of panicking, matching
        // the sibling contains()/clamp() handling.
        let space = DynSpace::new()
            .add_real("x", 0.0, 1.0)
            .add_integer("n", 1, 5)
            .add_categorical("opt", vec!["a".into(), "b".into()]);
        let dim = space.dimensionality();

        for non_object in [
            serde_json::json!(42),
            serde_json::json!("not an object"),
            serde_json::json!([1, 2, 3]),
            serde_json::Value::Null,
        ] {
            let unit = space.to_unit_cube(&non_object);
            assert_eq!(unit.len(), dim, "midpoint vector must match dimensionality");
            assert!(
                unit.iter().all(|v| (*v - 0.5).abs() < 1e-12),
                "non-object input must map to all-midpoint coordinates"
            );
        }
    }

    #[test]
    fn test_dyn_space_make_mut_copy_on_write() {
        // Building on a clone must not panic (Arc::get_mut would require a
        // refcount of 1) and must leave the original untouched while the clone
        // gains the new dimension.
        let a = DynSpace::new().add_real("x", 0.0, 1.0);
        let b = a.clone().add_integer("y", 0, 10);

        assert_eq!(a.dimensionality(), 1);
        assert_eq!(b.dimensionality(), 2);

        // `a` still holds only "x".
        assert!(a.contains(&serde_json::json!({"x": 0.5})));
        // `b` holds both "x" and "y" with the correct bounds.
        assert!(b.contains(&serde_json::json!({"x": 0.5, "y": 5})));
        assert!(!b.contains(&serde_json::json!({"x": 0.5, "y": 20})));
    }

    #[test]
    fn test_sobol_seed_folds_high_bits() {
        // Two u64 seeds differing only in bits >= 32 must fold to different u32
        // Sobol seeds and therefore yield different first Sobol points, while the
        // same seed reproduces deterministically.
        let space = DynSpace::new()
            .add_real("x", 0.0, 1.0)
            .add_real("y", 0.0, 1.0);

        // Drive the production fold in AutoStrategy::new by passing the RAW u64
        // seed (not a pre-folded u32), so reverting that fold to a truncating
        // `s as u32` would make `low` and `high` collide and fail the assert.
        let first_point = |seed: u64| -> Vec<f64> {
            let auto = AutoStrategy::new(2, 100, Some(seed));
            auto.sobol
                .suggest(&space)
                .as_object()
                .unwrap()
                .values()
                .map(|v| v.as_f64().unwrap())
                .collect()
        };

        let low = 1u64;
        let high = 1u64 | (1u64 << 40); // differs only in bits >= 32
        assert_ne!(
            first_point(low),
            first_point(high),
            "seeds differing only in high bits must not collide after folding"
        );
        // Determinism: the same seed reproduces the same first point.
        assert_eq!(first_point(high), first_point(high));
    }

    #[test]
    fn test_dyn_space_unit_cube_roundtrip() {
        let space = DynSpace::new()
            .add_real("x", 0.0, 10.0)
            .add_integer("n", 1, 5)
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
            .add_real("x", 0.0, 1.0)
            .add_integer("n", 1, 5)
            .add_categorical("opt", vec!["a".into(), "b".into()]);

        assert!(space.contains(&serde_json::json!({"x": 0.5, "n": 3, "opt": "a"})));
        assert!(!space.contains(&serde_json::json!({"x": 2.0, "n": 3, "opt": "a"})));
        assert!(!space.contains(&serde_json::json!({"x": 0.5, "n": 10, "opt": "a"})));
        assert!(!space.contains(&serde_json::json!({"x": 0.5, "n": 3, "opt": "unknown"})));
    }

    #[test]
    fn test_dyn_space_clamp() {
        let space = DynSpace::new()
            .add_real("x", 0.0, 1.0)
            .add_integer("n", 1, 5);

        let clamped = space.clamp(&serde_json::json!({"x": 2.0, "n": 10}));
        assert!((clamped.get("x").unwrap().as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(clamped.get("n").unwrap().as_i64().unwrap(), 5);
    }

    #[test]
    fn test_dyn_space_log_scales() {
        let space = DynSpace::new()
            .add_real_log("lr", 0.001, 1.0)
            .add_real_log10("alpha", 1e-4, 0.1);

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

    #[test]
    fn test_param_config_serde_new_names() {
        let yaml = r#"
            x:
              type: real
              min: 0.0
              max: 1.0
            n:
              type: integer
              min: 1
              max: 10
            opt:
              type: categorical
              choices: ["a", "b"]
        "#;
        let space: BTreeMap<String, ParamConfig> = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(space["x"], ParamConfig::Real { .. }));
        assert!(matches!(space["n"], ParamConfig::Integer { .. }));
        assert!(matches!(space["opt"], ParamConfig::Categorical { .. }));
    }

    #[test]
    fn test_param_config_serializes_new_names() {
        let config = ParamConfig::Real {
            min: 0.0,
            max: 1.0,
            scale: "linear".to_string(),
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["type"], "real");

        let config = ParamConfig::Integer { min: 1, max: 10 };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["type"], "integer");
    }

    #[test]
    fn test_auto_reconcile_keeps_fitted_model_and_live_counters() {
        use opt_engine::strategies::GmmParams;
        use opt_engine::traits::RefittableStrategy;

        // `live` is the engine's current strategy: its sampling state advanced
        // (via concurrent suggests/tells) while the refit ran off-lock, and its
        // GMM still holds the pre-refit single-component prior.
        let mut live = DynStrategy {
            inner: DynStrategyInner::Auto(AutoStrategy::new(1, 4, Some(7))),
        };
        if let DynStrategyInner::Auto(a) = &mut live.inner {
            a.gmm.set_params(GmmParams::uniform_prior(1, 0.1));
            a.trial_count = 9;
            a.issued_count.store(12, Ordering::Relaxed);
        }

        // `fitted` is the off-lock snapshot: it carries the freshly fitted GMM
        // (two components) but stale sampling counters from before the refit.
        let two_cluster_samples: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![if i < 25 { 0.2 } else { 0.8 }])
            .collect();
        let fitted_model = GmmParams::fit(&two_cluster_samples, 2, 100, 1e-6, 1e-4, 1);
        assert_eq!(fitted_model.n_components(), 2);

        let mut fitted = live.clone();
        if let DynStrategyInner::Auto(a) = &mut fitted.inner {
            a.gmm.set_params(fitted_model);
            a.trial_count = 5;
            a.issued_count.store(5, Ordering::Relaxed);
        }

        // Advance `live`'s Sobol sampler *after* the off-lock snapshot was taken,
        // so the two strategies hold divergent Sobol indices: `live` is ahead,
        // `fitted` carries the stale pre-refit index. Reconciliation must adopt
        // the live (advanced) index, otherwise resumed sampling would reissue
        // already-drawn Sobol points.
        let space = DynSpace::new().add_real("x", 0.0, 1.0);
        let live_sobol_index = if let DynStrategyInner::Auto(a) = &live.inner {
            a.sobol.suggest(&space);
            a.sobol.suggest(&space);
            a.sobol.suggest(&space);
            // The Sobol index is private cross-crate; observe it via serialization.
            serde_json::to_value(&a.sobol).unwrap()["index"].clone()
        } else {
            panic!("expected Auto strategy");
        };
        // The advanced live index must be strictly ahead of the stale snapshot,
        // so the assertion below can only pass if reconciliation adopted it.
        let fitted_sobol_index = if let DynStrategyInner::Auto(a) = &fitted.inner {
            serde_json::to_value(&a.sobol).unwrap()["index"].clone()
        } else {
            panic!("expected Auto strategy");
        };
        assert_ne!(live_sobol_index, fitted_sobol_index);

        fitted.reconcile_after_refit(&live);

        match &fitted.inner {
            DynStrategyInner::Auto(a) => {
                // Sampling state is taken from the live strategy.
                assert_eq!(a.trial_count, 9);
                assert_eq!(a.issued_count.load(Ordering::Relaxed), 12);
                // The reconciled Sobol sampler adopts the live (advanced) index,
                // not the stale snapshot index.
                let reconciled_sobol_index =
                    serde_json::to_value(&a.sobol).unwrap()["index"].clone();
                assert_eq!(reconciled_sobol_index, live_sobol_index);
                // The freshly fitted GMM model is kept (two components, not the
                // single-component prior that `live` still held).
                assert_eq!(a.gmm.params().n_components(), 2);
            }
            _ => panic!("expected Auto strategy"),
        }
    }

    #[test]
    fn test_sobol_strategy_rejects_high_dimensional_space() {
        // Each Real/Integer/Categorical param is exactly one unit-cube
        // dimension, so a space of N one-dimensional params has dimensionality N.
        let space_of = |n: usize| -> BTreeMap<String, ParamConfig> {
            (0..n)
                .map(|i| {
                    (
                        format!("p{i}"),
                        ParamConfig::Real {
                            min: 0.0,
                            max: 1.0,
                            scale: "linear".to_string(),
                        },
                    )
                })
                .collect()
        };
        let objectives = vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }];
        let strategy = |ty: &str| {
            Some(StrategyConfig {
                strategy_type: ty.to_string(),
                refit_interval: 20,
                total_budget: None,
                exploration_budget: None,
                seed: None,
                elite_fraction: None,
            })
        };
        let study = |n: usize, ty: &str| StudyConfig {
            space: space_of(n),
            objectives: objectives.clone(),
            strategy: strategy(ty),
            checkpoint: None,
            max_trials: None,
            max_leaderboard_size: None,
        };

        // A pure 'sobol' strategy above the 256-dimension limit is rejected with
        // a clear error, before any sampling can panic the backend.
        let err = HolaEngine::from_config(study(MAX_SOBOL_DIMS + 1, "sobol"))
            .err()
            .expect("sobol on a >256-dim space must be rejected");
        assert!(
            err.contains("the 'sobol' strategy supports at most 256 dimensions")
                && err.contains("this space has 257"),
            "unexpected error for sobol: {err}"
        );

        // 'auto' and 'gmm' use Sobol only for exploration (which falls back to
        // random above the limit while GMM exploitation is unaffected), so they
        // are accepted on a >256-dim space.
        for ty in ["auto", "gmm"] {
            assert!(
                HolaEngine::from_config(study(MAX_SOBOL_DIMS + 1, ty)).is_ok(),
                "{ty} on a >256-dim space must be accepted (random-exploration fallback)"
            );
        }

        // sobol exactly at the 256-dimension limit is accepted; random never
        // touches the Sobol backend so it is accepted above the limit.
        assert!(
            HolaEngine::from_config(study(MAX_SOBOL_DIMS, "sobol")).is_ok(),
            "sobol on a 256-dim space must be accepted"
        );
        assert!(
            HolaEngine::from_config(study(MAX_SOBOL_DIMS + 1, "random")).is_ok(),
            "random on a >256-dim space must be accepted"
        );
    }

    fn single_objective_config(strategy_type: &str) -> StudyConfig {
        StudyConfig {
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
            strategy: Some(StrategyConfig {
                strategy_type: strategy_type.to_string(),
                refit_interval: default_refit_interval(),
                total_budget: None,
                exploration_budget: None,
                seed: Some(7),
                elite_fraction: None,
            }),
            checkpoint: None,
            max_trials: None,
            max_leaderboard_size: None,
        }
    }

    #[tokio::test]
    async fn test_concurrent_tells_and_update_objectives_consistent() {
        // Objectives + leaderboard live behind one lock, so concurrent
        // tell()s racing an update_objectives must never panic, deadlock, lose a
        // trial, or leave a half-updated state. This crosses the scalar<->vector
        // boundary: the study starts single-group (scalar leaderboard) and the
        // concurrent updater installs two priority groups, forcing a real
        // observation-kind migration to run while tell()s are in flight.
        let engine = HolaEngine::from_config(single_objective_config("gmm")).unwrap();
        assert!(
            matches!(
                &engine.state.read().await.leaderboard,
                DynLeaderboard::Scalar(_)
            ),
            "study must start scalar so the update crosses into vector"
        );

        let n = 80usize;
        let mut handles = Vec::new();
        for i in 0..n {
            let eng = engine.clone();
            handles.push(tokio::spawn(async move {
                let trial = eng.ask().await.expect("ask should succeed");
                // Provide both metric fields so the post-migration vector
                // topology scores every trial as feasible.
                let metrics = serde_json::json!({
                    "loss": (i as f64) / (n as f64),
                    "latency": (n - i) as f64,
                });
                eng.tell(trial.trial_id, metrics)
                    .await
                    .expect("tell should succeed");
            }));
        }

        // Swap to two priority groups partway through the concurrent tell()s.
        // Crossing from one group to two flips the leaderboard from scalar to
        // vector, so update_objectives performs an observation-kind migration
        // concurrently with the in-flight tells.
        let updater = {
            let eng = engine.clone();
            tokio::spawn(async move {
                let new_objectives = vec![
                    ObjectiveConfig {
                        field: "loss".to_string(),
                        obj_type: "minimize".to_string(),
                        target: None,
                        limit: None,
                        priority: 1.0,
                        group: Some("quality".to_string()),
                    },
                    ObjectiveConfig {
                        field: "latency".to_string(),
                        obj_type: "minimize".to_string(),
                        target: None,
                        limit: None,
                        priority: 1.0,
                        group: Some("speed".to_string()),
                    },
                ];
                eng.update_objectives(new_objectives)
                    .await
                    .expect("update_objectives should succeed");
            })
        };

        for h in handles {
            h.await.expect("tell task must not panic or deadlock");
        }
        updater
            .await
            .expect("update task must not panic or deadlock");

        // Every issued trial was recorded exactly once, even though some tell()s
        // landed before the migration (scalar) and some after (vector).
        assert_eq!(
            engine.trial_count().await,
            n,
            "all concurrent trials must be recorded across the migration"
        );

        // Final topology is vector (two priority groups won the race to be last).
        let final_objectives = engine.objectives().await;
        assert_eq!(final_objectives.len(), 2);
        assert!(
            matches!(
                &engine.state.read().await.leaderboard,
                DynLeaderboard::Vector(_)
            ),
            "final leaderboard must reflect the two-group vector topology"
        );

        // The leaderboard is internally consistent under the final topology: it
        // ranks (NSGA-II) without error and returns the full set of trials, with
        // valid 0-indexed ranks across every migrated trial.
        let trials = engine.trials("rank", true).await;
        assert_eq!(
            trials.len(),
            n,
            "ranked view must include all trials after migration"
        );
        let mut ranks: Vec<usize> = trials.iter().map(|t| t.rank).collect();
        ranks.sort_unstable();
        assert_eq!(
            ranks,
            (0..n).collect::<Vec<_>>(),
            "ranks must be a valid 0..n permutation under the final topology"
        );
    }

    #[tokio::test]
    async fn test_auto_checkpoint_roundtrip_preserves_strategy_state() {
        // The auto-checkpoint path saves a full checkpoint, so resuming
        // continues the strategy/exploration state instead of resetting it. The
        // issued counter (next suggested trial) must be continuous after resume.
        let dir = tempfile::tempdir().unwrap();
        let mut config = single_objective_config("auto");
        config.strategy.as_mut().unwrap().exploration_budget = Some(4);
        config.checkpoint = Some(CheckpointConfig {
            directory: dir.path().to_string_lossy().to_string(),
            interval: 5,
            max_checkpoints: None,
            load_from: None,
        });

        let engine = HolaEngine::from_config(config).unwrap();
        for i in 0..5 {
            let trial = engine.ask().await.unwrap();
            engine
                .tell(trial.trial_id, serde_json::json!({ "loss": i as f64 }))
                .await
                .unwrap();
        }

        // The interval-5 auto-checkpoint fired on the 5th tell.
        let checkpoint_path = dir
            .path()
            .join("checkpoint_000005.json")
            .to_string_lossy()
            .to_string();
        assert!(
            std::path::Path::new(&checkpoint_path).exists(),
            "auto-checkpoint file should exist at {checkpoint_path}"
        );

        // The auto-checkpoint is a full checkpoint, so it can be reloaded as one
        // (a leaderboard-only checkpoint would lack strategy_state).
        let resumed = HolaEngine::load_from_checkpoint(&checkpoint_path)
            .await
            .expect("auto-checkpoint must be a full checkpoint reconstructable from config");
        assert_eq!(resumed.trial_count().await, 5);

        // Exploration progress is continuous: the resumed engine has already
        // issued 5 trials, so its first ask() lands on trial 5 (past the
        // exploration budget of 4, i.e. in the GMM exploitation phase) rather
        // than restarting exploration from 0.
        let next = resumed.ask().await.unwrap();
        assert_eq!(
            next.trial_id, 5,
            "resumed engine must continue trial numbering, not reset it"
        );

        // Prove the strategy state (Sobol index, GMM model, exploration/refit
        // progress) was restored, not just the trial count: a control
        // engine that ran the same seeded trials end-to-end without any
        // checkpointing must propose the identical next candidate. Both are in
        // the GMM exploitation phase, so this only matches if the resumed GMM
        // and counters were restored from the checkpoint.
        let mut control_config = single_objective_config("auto");
        control_config.strategy.as_mut().unwrap().exploration_budget = Some(4);
        let control = HolaEngine::from_config(control_config).unwrap();
        for i in 0..5 {
            let trial = control.ask().await.unwrap();
            control
                .tell(trial.trial_id, serde_json::json!({ "loss": i as f64 }))
                .await
                .unwrap();
        }
        let control_next = control.ask().await.unwrap();
        assert_eq!(
            control_next.trial_id, 5,
            "control engine numbering must also reach trial 5"
        );
        assert_eq!(
            next.params, control_next.params,
            "resumed strategy must propose the same candidate as an uninterrupted \
             seeded run; a mismatch means strategy state was not truly restored"
        );
    }

    #[tokio::test]
    async fn test_leaderboard_checkpoint_carries_and_respects_observation_kind() {
        // A leaderboard checkpoint records its observation_kind tag and
        // load honors it, erroring when the tag conflicts with the current
        // objective topology rather than mis-deserializing.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lb.json").to_string_lossy().to_string();

        // Build a scalar (single-group) study and save a leaderboard checkpoint.
        let scalar_engine = HolaEngine::from_config(single_objective_config("random")).unwrap();
        for i in 0..3 {
            let trial = scalar_engine.ask().await.unwrap();
            scalar_engine
                .tell(trial.trial_id, serde_json::json!({ "loss": i as f64 }))
                .await
                .unwrap();
        }
        scalar_engine
            .save_leaderboard_checkpoint_to(&path, None)
            .await
            .unwrap();

        // The saved file carries the scalar observation_kind tag.
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            raw.get("observation_kind"),
            Some(&serde_json::Value::from("scalar")),
            "leaderboard checkpoint must carry the scalar observation_kind tag"
        );

        // Loading into a matching scalar study succeeds and restores all trials.
        let scalar_loader = HolaEngine::from_config(single_objective_config("random")).unwrap();
        scalar_loader
            .load_leaderboard_checkpoint(&path)
            .await
            .expect("scalar checkpoint loads into a scalar study");
        assert_eq!(scalar_loader.trial_count().await, 3);

        // Loading into a vector (multi-group) study conflicts with the tag and
        // must error clearly instead of mis-deserializing.
        let mut vector_config = single_objective_config("random");
        vector_config.objectives = vec![
            ObjectiveConfig {
                field: "loss".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("a".to_string()),
            },
            ObjectiveConfig {
                field: "latency".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("b".to_string()),
            },
        ];
        let vector_loader = HolaEngine::from_config(vector_config).unwrap();
        let err = vector_loader
            .load_leaderboard_checkpoint(&path)
            .await
            .expect_err("scalar checkpoint must not load into a vector study");
        assert!(
            err.to_string().contains("observation_kind"),
            "conflict error should mention observation_kind, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_tell_returns_correct_completed_trial_without_full_clone() {
        // tell() builds the returned view without cloning the whole leaderboard.
        // We cannot assert "did not clone" directly, so assert the observable
        // contract instead: the returned CompletedTrial has the same fields and
        // rank it would get from the independent full-board ranking path
        // (completed_trial / trials), across a sequence of tells.
        let engine = HolaEngine::from_config(single_objective_config("random")).unwrap();

        // Feed losses in a non-monotonic order so ranks are not just insertion
        // order: the best trial is told last.
        let losses = [0.5_f64, 0.9, 0.3, 0.7, 0.1];
        let mut told: Vec<(u64, CompletedTrial)> = Vec::new();
        for loss in losses {
            let trial = engine.ask().await.unwrap();
            let completed = engine
                .tell(trial.trial_id, serde_json::json!({ "loss": loss }))
                .await
                .unwrap();
            assert_eq!(completed.trial_id, trial.trial_id);
            assert_eq!(completed.params, trial.params);
            told.push((trial.trial_id, completed));
        }

        // The view returned by tell() must match the same trial recomputed from
        // the full leaderboard ranking path (which does rank the whole board).
        for (id, from_tell) in &told {
            let from_board = engine
                .completed_trial(*id, true)
                .await
                .expect("trial must be present");
            // Rank in the final board can differ from the rank at tell()-time
            // (later, better trials shift it), so compare the non-rank fields
            // here and verify rank consistency below at the final snapshot.
            assert_eq!(from_tell.trial_id, from_board.trial_id);
            assert_eq!(from_tell.params, from_board.params);
            assert_eq!(from_tell.metrics, from_board.metrics);
            assert_eq!(from_tell.scores, from_board.scores);
            assert_eq!(from_tell.score_vector, from_board.score_vector);
            assert_eq!(from_tell.completed_at, from_board.completed_at);
        }

        // The last tell() saw the full board, so its reported rank must equal the
        // final ranking. Loss 0.1 (told last) is the best, so rank 0.
        let (last_id, last_view) = told.last().unwrap();
        assert_eq!(last_view.rank, 0, "best trial told last must rank first");
        let final_best = engine.top_k(1, true).await;
        assert_eq!(final_best[0].trial_id, *last_id);
        assert_eq!(final_best[0].rank, 0);
    }

    #[tokio::test]
    async fn test_max_leaderboard_size_caps_retained_trials() {
        // Opt-in bounded mode caps the stored trial count, while the default
        // (unbounded) study retains every trial.
        let cap = 5usize;
        let n = 20usize;

        let mut bounded_cfg = single_objective_config("random");
        bounded_cfg.max_leaderboard_size = Some(cap);
        let bounded = HolaEngine::from_config(bounded_cfg).unwrap();

        let unbounded = HolaEngine::from_config(single_objective_config("random")).unwrap();

        for engine in [&bounded, &unbounded] {
            for i in 0..n {
                let trial = engine.ask().await.unwrap();
                // Feed strictly improving losses so the just-completed trial is
                // never the eviction victim, regardless of whether the bounded
                // policy drops the oldest or the worst trial. tell() must always
                // return that trial's view, so it must remain retained.
                let loss = (n - i) as f64;
                engine
                    .tell(trial.trial_id, serde_json::json!({ "loss": loss }))
                    .await
                    .expect("tell must succeed under both bounded and unbounded modes");
            }
        }

        assert_eq!(
            bounded.trial_count().await,
            cap,
            "bounded study must retain at most max_leaderboard_size trials"
        );
        assert_eq!(
            unbounded.trial_count().await,
            n,
            "default (unbounded) study must retain every trial"
        );
    }

    #[tokio::test]
    async fn test_default_study_is_unbounded() {
        // Back-compat: max_leaderboard_size defaults to None (unbounded) and a
        // config that omits it deserializes to None.
        let cfg: StudyConfig = serde_json::from_value(serde_json::json!({
            "space": { "x": { "type": "real", "min": 0.0, "max": 1.0 } },
            "objectives": [ { "field": "loss", "type": "minimize" } ],
        }))
        .unwrap();
        assert_eq!(
            cfg.max_leaderboard_size, None,
            "omitted max_leaderboard_size must default to None"
        );

        let engine = HolaEngine::from_config(cfg).unwrap();
        let n = 12usize;
        for i in 0..n {
            let trial = engine.ask().await.unwrap();
            engine
                .tell(trial.trial_id, serde_json::json!({ "loss": i as f64 }))
                .await
                .unwrap();
        }
        assert_eq!(engine.trial_count().await, n);
    }

    #[tokio::test]
    async fn test_ask_tell_produce_unique_contiguous_ids() {
        // ask()/tell() must still hand out unique, contiguous trial ids. The
        // O(1) id-probe must not skip or reuse ids on the hot path.
        let engine = HolaEngine::from_config(single_objective_config("random")).unwrap();
        let n = 30usize;
        let mut ids = Vec::new();
        for i in 0..n {
            let trial = engine.ask().await.unwrap();
            ids.push(trial.trial_id);
            engine
                .tell(trial.trial_id, serde_json::json!({ "loss": i as f64 }))
                .await
                .unwrap();
        }
        assert_eq!(
            ids,
            (0..n as u64).collect::<Vec<_>>(),
            "ask()/tell() must yield contiguous ids 0..n with no gaps or reuse"
        );
    }

    /// Two priority-group objectives that force a vector leaderboard.
    fn two_group_objectives() -> Vec<ObjectiveConfig> {
        vec![
            ObjectiveConfig {
                field: "loss".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("quality".to_string()),
            },
            ObjectiveConfig {
                field: "latency".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("speed".to_string()),
            },
        ]
    }

    #[tokio::test]
    async fn test_bounded_study_terminates_and_stays_bounded_across_migration() {
        // A bounded study (max_leaderboard_size = cap) with a max_trials
        // limit must (a) stay capped even after update_objectives flips the
        // priority-group count and rebuilds the leaderboard, and (b) still
        // terminate at max_trials. The stopping check counts total completed
        // trials rather than the capped leaderboard length (which never reaches
        // max_trials once full), and migration carries the cap across an
        // objective swap so the board stays bounded.
        let cap = 5usize;
        let max_trials = 12usize;

        let mut cfg = single_objective_config("random");
        cfg.max_leaderboard_size = Some(cap);
        cfg.max_trials = Some(max_trials);
        let engine = HolaEngine::from_config(cfg).unwrap();

        // Complete enough trials to fill and exceed the cap while still scalar.
        for i in 0..(cap + 1) {
            let trial = engine.ask().await.unwrap();
            engine
                .tell(
                    trial.trial_id,
                    serde_json::json!({ "loss": i as f64, "latency": (cap + 1 - i) as f64 }),
                )
                .await
                .unwrap();
        }
        assert_eq!(
            engine.trial_count().await,
            cap,
            "scalar bounded study must not exceed the cap"
        );

        // Flip to two priority groups, forcing a scalar -> vector migration that
        // rebuilds the leaderboard. The cap must survive the rebuild.
        engine
            .update_objectives(two_group_objectives())
            .await
            .unwrap();
        assert!(
            matches!(
                &engine.state.read().await.leaderboard,
                DynLeaderboard::Vector(_)
            ),
            "study must be vector after flipping to two priority groups"
        );
        assert!(
            engine.trial_count().await <= cap,
            "migration must carry the cap; the rebuilt board must stay bounded"
        );

        // Drive the study until ask() refuses further trials. The monotonic
        // completed-count check guarantees termination; bound the loop anyway so
        // a regression fails loudly instead of hanging.
        let mut completed = cap + 1;
        let mut terminated = false;
        for _ in 0..(max_trials * 4) {
            match engine.ask().await {
                Ok(trial) => {
                    engine
                        .tell(
                            trial.trial_id,
                            serde_json::json!({ "loss": completed as f64, "latency": 1.0 }),
                        )
                        .await
                        .unwrap();
                    completed += 1;
                    assert!(
                        engine.trial_count().await <= cap,
                        "bounded study must never exceed the cap mid-run"
                    );
                }
                Err(_) => {
                    terminated = true;
                    break;
                }
            }
        }
        assert!(
            terminated,
            "bounded study must terminate at max_trials, not run forever"
        );
        assert_eq!(
            completed, max_trials,
            "study must terminate exactly when the monotonic completed count reaches max_trials"
        );
        assert!(
            engine.trial_count().await <= cap,
            "final bounded study must still respect the cap"
        );
    }

    #[tokio::test]
    async fn test_bounded_vector_study_with_infeasible_terminates_across_migration() {
        // Exercises the migration counter carry-over on a VECTOR board that
        // holds an infeasible (infinite-observation) trial. An over-limit latency
        // maps to +inf in the vector observation; serde_json renders +inf as
        // `null` and cannot read it back as f64, so restoring total_completed via
        // a JSON round-trip would fail and fall back to the retained (capped)
        // length, dropping the evicted history and leaving the max_trials stopping
        // check (driven by total_completed) unable to fire. set_total_completed
        // carries the counter directly, with no round-trip, so the study
        // terminates exactly at max_trials.
        let cap = 5usize;
        let max_trials = 12usize;

        let mut cfg = single_objective_config("random");
        cfg.max_leaderboard_size = Some(cap);
        cfg.max_trials = Some(max_trials);
        // Two priority groups force a vector leaderboard from the start; the speed
        // objective carries a limit so an over-limit latency maps to +inf and the
        // trial is infeasible.
        cfg.objectives = vec![
            ObjectiveConfig {
                field: "loss".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("quality".to_string()),
            },
            ObjectiveConfig {
                field: "latency".to_string(),
                obj_type: "minimize".to_string(),
                target: Some(0.0),
                limit: Some(10.0),
                priority: 1.0,
                group: Some("speed".to_string()),
            },
        ];
        let engine = HolaEngine::from_config(cfg).unwrap();
        assert!(
            matches!(
                &engine.state.read().await.leaderboard,
                DynLeaderboard::Vector(_)
            ),
            "two priority groups must produce a vector leaderboard"
        );

        // Fill and exceed the cap while still vector, completing at least one
        // infeasible trial (latency over the limit -> +inf observation) so the
        // retained board carries a non-finite, non-round-trippable observation.
        for i in 0..(cap + 1) {
            let trial = engine.ask().await.unwrap();
            // Make the last trial infeasible (latency far over the limit) and
            // ensure it is retained, then push one more to exceed the cap.
            let latency = if i == cap { 500.0 } else { 1.0 };
            engine
                .tell(
                    trial.trial_id,
                    serde_json::json!({ "loss": i as f64, "latency": latency }),
                )
                .await
                .unwrap();
        }
        assert_eq!(
            engine.trial_count().await,
            cap,
            "vector bounded study must not exceed the cap"
        );

        // Collapse to a single priority group, forcing a vector -> scalar
        // migration that rebuilds the leaderboard. The rebuilt board must carry
        // the prior total_completed counter even though it held an infinite
        // observation; the JSON round-trip could not.
        let mut collapsed = two_group_objectives();
        collapsed[1].group = Some("quality".to_string());
        engine.update_objectives(collapsed).await.unwrap();
        assert!(
            matches!(
                &engine.state.read().await.leaderboard,
                DynLeaderboard::Scalar(_)
            ),
            "study must be scalar after collapsing to one priority group"
        );

        // Drive the study until ask() refuses further trials. The carried
        // counter guarantees termination at max_trials; bound the loop anyway so
        // a regression fails loudly instead of hanging.
        let mut completed = cap + 1;
        let mut terminated = false;
        for _ in 0..(max_trials * 4) {
            match engine.ask().await {
                Ok(trial) => {
                    engine
                        .tell(
                            trial.trial_id,
                            serde_json::json!({ "loss": completed as f64 }),
                        )
                        .await
                        .unwrap();
                    completed += 1;
                    assert!(
                        engine.trial_count().await <= cap,
                        "bounded study must never exceed the cap mid-run"
                    );
                }
                Err(_) => {
                    terminated = true;
                    break;
                }
            }
        }
        assert!(
            terminated,
            "bounded vector study with infeasible trial must terminate at max_trials, not run forever"
        );
        assert_eq!(
            completed, max_trials,
            "study must terminate exactly when total_completed reaches max_trials, \
             proving the counter was carried across migration despite the +inf observation"
        );
    }

    #[tokio::test]
    async fn test_bounded_checkpoint_roundtrip_preserves_cap() {
        // Loading a checkpoint must re-apply the engine's configured cap so
        // the leaderboard stays bounded after a load. The saved board may have
        // been capped (or, for legacy files, uncapped); either way the loaded
        // engine must enforce its own configured cap.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bounded.json");
        let cap = 4usize;

        let mut cfg = single_objective_config("random");
        cfg.max_leaderboard_size = Some(cap);
        let engine = HolaEngine::from_config(cfg).unwrap();
        for i in 0..(cap * 3) {
            let trial = engine.ask().await.unwrap();
            engine
                .tell(trial.trial_id, serde_json::json!({ "loss": i as f64 }))
                .await
                .unwrap();
        }
        assert_eq!(engine.trial_count().await, cap);
        engine.save(&path).await.unwrap();

        // A fresh bounded engine loads the checkpoint and must keep the cap.
        let mut cfg2 = single_objective_config("random");
        cfg2.max_leaderboard_size = Some(cap);
        let loaded = HolaEngine::from_config(cfg2).unwrap();
        loaded.load(&path).await.unwrap();
        assert_eq!(
            loaded.trial_count().await,
            cap,
            "loaded board must be capped down to the configured max_leaderboard_size"
        );
        assert_eq!(
            loaded.state.read().await.leaderboard.max_size(),
            Some(cap),
            "loaded board's recorded cap must match the engine's configured cap"
        );

        // Continuing to push must keep the loaded study bounded.
        for i in 0..cap {
            let trial = loaded.ask().await.unwrap();
            loaded
                .tell(
                    trial.trial_id,
                    serde_json::json!({ "loss": (100 + i) as f64 }),
                )
                .await
                .unwrap();
        }
        assert_eq!(
            loaded.trial_count().await,
            cap,
            "post-load pushes must continue to respect the cap"
        );
    }

    #[test]
    fn test_rotate_checkpoints_deletes_oldest_across_digit_boundary() {
        // Files span the 6-to-7-digit counter boundary, where a lexicographic
        // filename sort mis-orders ("checkpoint_1000000" sorts before
        // "checkpoint_0999998"). The newest-by-mtime files must be retained and
        // the oldest deleted regardless of filename ordering.
        let dir = tempfile::tempdir().unwrap();
        let prefix = "checkpoint";

        // Counters around the 6->7 digit boundary. Listed oldest-first; we set
        // mtimes in this order so the last entry is the newest.
        let counters = [999_997usize, 999_998, 999_999, 1_000_000, 1_000_001];
        let mut paths = Vec::new();
        let base = std::time::SystemTime::UNIX_EPOCH;
        for (i, c) in counters.iter().enumerate() {
            let path = dir.path().join(format!("{prefix}_{c:06}.json"));
            std::fs::write(&path, b"{}").unwrap();
            // Give each file a strictly increasing mtime in oldest-first order so
            // age, not filename lexicography, drives rotation.
            let mtime = base + std::time::Duration::from_secs((i as u64 + 1) * 100);
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(mtime)
                .unwrap();
            paths.push((path, *c));
        }

        // Keep only the 2 newest; the 3 oldest must be deleted.
        HolaEngine::rotate_checkpoints(dir.path(), prefix, 2);

        // The two newest-by-mtime counters (1_000_000, 1_000_001) survive; the
        // three oldest are gone, even though "checkpoint_1000000" would sort
        // before "checkpoint_0999998" lexicographically.
        for (path, c) in &paths {
            let should_exist = *c == 1_000_000 || *c == 1_000_001;
            assert_eq!(
                path.exists(),
                should_exist,
                "counter {c}: expected exists={should_exist}, got {}",
                path.exists()
            );
        }
    }

    #[tokio::test]
    async fn test_max_trials_out_of_order_completion_does_not_under_deliver() {
        // A next_trial_id()-based budget check would double-count a pending id
        // that sits below an already-completed id (once in the id span
        // next_trial_id() reports, once in pending.len()), so a parallel
        // ask-many / tell-out-of-order study would under-deliver and stop early.
        // total_completed() counts only successful pushes, so the study must
        // admit and complete exactly max_trials distinct trials.
        let max_trials = 10usize;
        let mut cfg = single_objective_config("random");
        cfg.max_trials = Some(max_trials);
        let engine = HolaEngine::from_config(cfg).unwrap();

        // Ask a batch up front so several trials are pending at once, then tell
        // them in reverse id order (a higher id completes before lower pending
        // ids). This is exactly the shape that trips the id-span double-count.
        let batch = 4usize;
        let mut ids = Vec::new();
        for _ in 0..batch {
            ids.push(engine.ask().await.unwrap().trial_id);
        }
        for (i, id) in ids.iter().rev().enumerate() {
            engine
                .tell(*id, serde_json::json!({ "loss": i as f64 }))
                .await
                .unwrap();
        }

        // Drive the remaining budget one at a time, again interleaving an
        // out-of-order pair to keep a low pending id outstanding past a higher
        // completed id.
        let mut completed = batch;
        while completed < max_trials {
            let first = engine.ask().await.unwrap();
            if completed + 1 < max_trials {
                let second = engine.ask().await.unwrap();
                engine
                    .tell(second.trial_id, serde_json::json!({ "loss": 0.0 }))
                    .await
                    .unwrap();
                engine
                    .tell(first.trial_id, serde_json::json!({ "loss": 0.0 }))
                    .await
                    .unwrap();
                completed += 2;
            } else {
                engine
                    .tell(first.trial_id, serde_json::json!({ "loss": 0.0 }))
                    .await
                    .unwrap();
                completed += 1;
            }
        }

        assert_eq!(
            engine.trial_count().await,
            max_trials,
            "out-of-order completion must admit and complete exactly max_trials trials"
        );
        // The budget is now exhausted: the next ask() must be refused.
        assert!(
            engine.ask().await.is_err(),
            "study must stop once max_trials distinct trials have completed"
        );
    }

    #[tokio::test]
    async fn test_max_trials_cancelled_trials_do_not_consume_budget() {
        // Cancelled trials are removed from pending and never pushed, so
        // they must not consume max_trials budget. A next_trial_id()-based check
        // would advance past cancelled-id gaps and charge them, stopping the
        // study early; total_completed() counts only pushes, so a cancelled trial
        // leaves the budget untouched.
        let max_trials = 8usize;
        let mut cfg = single_objective_config("random");
        cfg.max_trials = Some(max_trials);
        let engine = HolaEngine::from_config(cfg).unwrap();

        let mut completed = 0usize;
        while completed < max_trials {
            // Ask and cancel a trial, then ask and complete one. The cancelled
            // trial burns an id but must not count toward the budget.
            let doomed = engine.ask().await.unwrap();
            engine.cancel(doomed.trial_id).await.unwrap();

            let keep = engine.ask().await.unwrap();
            engine
                .tell(
                    keep.trial_id,
                    serde_json::json!({ "loss": completed as f64 }),
                )
                .await
                .unwrap();
            completed += 1;
        }

        assert_eq!(
            engine.trial_count().await,
            max_trials,
            "cancelled trials must not consume budget; exactly max_trials must complete"
        );
        assert!(
            engine.ask().await.is_err(),
            "study must stop only after max_trials non-cancelled trials complete"
        );
    }

    #[tokio::test]
    async fn test_cancelled_set_is_bounded() {
        // Cancelling many trials without any checkpoint reload must not grow the
        // cancelled set without bound: it is pruned to MAX_CANCELLED_RETAINED.
        let engine = HolaEngine::from_config(single_objective_config("random")).unwrap();

        let n = MAX_CANCELLED_RETAINED * 2 + 100;
        for _ in 0..n {
            let t = engine.ask().await.unwrap();
            engine.cancel(t.trial_id).await.unwrap();
        }

        {
            let state = engine.state.read().await;
            assert!(
                state.cancelled.len() <= MAX_CANCELLED_RETAINED,
                "cancelled set must stay bounded, got {}",
                state.cancelled.len()
            );
        }

        // ask/tell behavior is preserved: a fresh trial can still be asked,
        // completed, and a recently cancelled trial is still rejected by tell.
        let doomed = engine.ask().await.unwrap();
        engine.cancel(doomed.trial_id).await.unwrap();
        let err = engine
            .tell(doomed.trial_id, serde_json::json!({"loss": 1.0}))
            .await
            .unwrap_err();
        assert!(
            err.contains("cancelled"),
            "tell on a recently cancelled trial must report it as cancelled, got: {err}"
        );

        let keep = engine.ask().await.unwrap();
        let completed = engine
            .tell(keep.trial_id, serde_json::json!({"loss": 0.5}))
            .await
            .unwrap();
        assert_eq!(completed.trial_id, keep.trial_id);
    }

    #[tokio::test]
    async fn test_max_trials_bounded_mode_still_terminates() {
        // Re-confirm with the total_completed() counter that a capped
        // leaderboard (len() frozen at the cap) still terminates at max_trials.
        // total_completed() keeps growing past the cap, so the stopping check
        // fires; a len()-based check would loop forever once full.
        let cap = 4usize;
        let max_trials = 11usize;
        let mut cfg = single_objective_config("random");
        cfg.max_leaderboard_size = Some(cap);
        cfg.max_trials = Some(max_trials);
        let engine = HolaEngine::from_config(cfg).unwrap();

        let mut completed = 0usize;
        let mut terminated = false;
        for _ in 0..(max_trials * 4) {
            match engine.ask().await {
                Ok(trial) => {
                    engine
                        .tell(
                            trial.trial_id,
                            serde_json::json!({ "loss": completed as f64 }),
                        )
                        .await
                        .unwrap();
                    completed += 1;
                    assert!(
                        engine.trial_count().await <= cap,
                        "bounded study must never exceed the cap mid-run"
                    );
                }
                Err(_) => {
                    terminated = true;
                    break;
                }
            }
        }
        assert!(
            terminated,
            "bounded study must terminate at max_trials, not run forever"
        );
        assert_eq!(
            completed, max_trials,
            "bounded study must terminate exactly at max_trials via total_completed()"
        );
    }

    #[tokio::test]
    async fn test_vector_global_rank_matches_canonical_ranking() {
        // Pin the off-lock vector_global_rank against the canonical
        // leaderboard ranking. For a vector board with multiple multi-member
        // fronts (including ties and infeasible trials), vector_global_rank
        // must equal the position of each id in ranked_trials_all() (include
        // infeasible) and ranked_trials() (feasible only), for EVERY id.
        let mut cfg = single_objective_config("random");
        // Two priority groups force a vector leaderboard. The speed objective
        // carries a target/limit so an over-limit latency maps to +inf and the
        // trial is infeasible; both transforms are monotonic, so the feasible
        // front geometry is preserved.
        cfg.objectives = vec![
            ObjectiveConfig {
                field: "loss".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("quality".to_string()),
            },
            ObjectiveConfig {
                field: "latency".to_string(),
                obj_type: "minimize".to_string(),
                target: Some(0.0),
                limit: Some(10.0),
                priority: 1.0,
                group: Some("speed".to_string()),
            },
        ];
        let engine = HolaEngine::from_config(cfg).unwrap();
        assert!(
            matches!(
                &engine.state.read().await.leaderboard,
                DynLeaderboard::Vector(_)
            ),
            "two priority groups must produce a vector leaderboard"
        );

        // A spread of (loss, latency) points building several multi-member
        // fronts, with a deliberate tie (two identical observations) and two
        // infeasible trials (latency above the limit).
        let points = [
            (1.0, 5.0),   // front 0
            (2.0, 4.0),   // front 0
            (3.0, 3.0),   // front 0
            (2.0, 4.0),   // front 0, tie with (2.0, 4.0)
            (5.0, 9.0),   // dominated, later front
            (6.0, 8.0),   // dominated, later front
            (4.0, 6.0),   // middle front
            (7.0, 200.0), // infeasible (latency over limit)
            (8.0, 300.0), // infeasible (latency over limit)
        ];
        for (loss, latency) in points {
            let trial = engine.ask().await.unwrap();
            engine
                .tell(
                    trial.trial_id,
                    serde_json::json!({ "loss": loss, "latency": latency }),
                )
                .await
                .unwrap();
        }

        let state = engine.state.read().await;
        let lb = match &state.leaderboard {
            DynLeaderboard::Vector(lb) => lb,
            _ => unreachable!("verified vector above"),
        };

        // include_infeasible == true: compare against ranked_trials_all().
        let ranked_all = lb.ranked_trials_all();
        let snapshot_all: Vec<(u64, BTreeMap<String, f64>)> = lb
            .iter()
            .map(|t| (t.trial_id, t.observation.clone()))
            .collect();
        for (canonical_pos, rt) in ranked_all.iter().enumerate() {
            let id = rt.trial.trial_id;
            let computed = vector_global_rank(&snapshot_all, id)
                .expect("every present id must rank in the all-trials snapshot");
            assert_eq!(
                computed, canonical_pos,
                "vector_global_rank (all) for id {id} must match ranked_trials_all position"
            );
        }

        // include_infeasible == false: compare against ranked_trials(), using a
        // feasible-only snapshot exactly as completed_for_tell builds it.
        let ranked_feasible = lb.ranked_trials();
        let snapshot_feasible: Vec<(u64, BTreeMap<String, f64>)> = lb
            .iter()
            .filter(|t| {
                Leaderboard::<serde_json::Value, BTreeMap<String, f64>>::trial_is_feasible(t)
            })
            .map(|t| (t.trial_id, t.observation.clone()))
            .collect();
        for (canonical_pos, rt) in ranked_feasible.iter().enumerate() {
            let id = rt.trial.trial_id;
            let computed = vector_global_rank(&snapshot_feasible, id)
                .expect("every feasible id must rank in the feasible snapshot");
            assert_eq!(
                computed, canonical_pos,
                "vector_global_rank (feasible) for id {id} must match ranked_trials position"
            );
        }
    }
}
