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

use opt_engine::leaderboard::{Leaderboard, Trial, is_feasible_multi};
use opt_engine::persistence::{
    AutoCheckpointConfig, Checkpoint, LeaderboardCheckpoint, ObservationKind, atomic_write_json,
    check_format_version_bytes, read_checkpoint_capped,
};
use opt_engine::scales::{LinearScale, Log10Scale, LogScale, Scale};
use opt_engine::spaces::{CategoricalSpace, ContinuousSpace, DiscreteSpace};
use opt_engine::strategies::{GmmRefitConfig, GmmStrategy, RandomStrategy, SobolStrategy};
use opt_engine::traits::{RefitConfig, SampleSpace, StandardizedSpace, Strategy};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
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

/// Hard safety bound for in-flight work when no smaller `max_trials` budget is
/// configured. This prevents crashed or malicious workers from growing pending
/// state without limit.
const MAX_PENDING_TRIALS: usize = 10_000;

/// Every ask retry key refers to one live pending trial, so its capacity must
/// cover the entire pending-work bound. Evicting a key while its trial is still
/// pending would let a retry allocate duplicate work.
const MAX_ASK_IDEMPOTENCY_KEYS: usize = MAX_PENDING_TRIALS;

/// Recent completion receipts retained after leaderboard eviction. They make a
/// tell retry safe when the original success response was lost, without turning
/// the idempotency ledger into unbounded study history.
const MAX_COMPLETION_RECEIPTS: usize = 4096;

/// Legacy cadence for low-discrepancy exploration after the initial warm-up.
/// This prevents a fitted GMM from permanently collapsing around sparse early
/// elites.
pub const DEFAULT_ONGOING_EXPLORATION_PERIOD: usize = 5;
/// Legacy upper bound on fitted GMM components.
pub const DEFAULT_MAX_COMPONENTS: usize = 3;
/// Legacy lower bound on the elite workset passed to a GMM refit.
pub const DEFAULT_MIN_ELITE_SAMPLES: usize = 1;
/// Variance of the neutral GMM placeholder used before the first empirical fit.
const AUTO_GMM_PRIOR_VARIANCE: f64 = 0.1;
/// Default bound on the number of elite samples passed to full-covariance EM.
/// This is an implementation safeguard, not part of the abstract method, and
/// can be overridden in [`StrategyConfig`].
pub const DEFAULT_MAX_REFIT_SAMPLES: usize = 4096;
/// Default bound on the candidate workset ranked to choose those samples.
/// Histories beyond this size are covered by deterministic chronological
/// strata rather than a newest-only window.
pub const DEFAULT_MAX_REFIT_CANDIDATES: usize = 16_384;

fn unix_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

fn fresh_legacy_trial_id_floor() -> u64 {
    // Legacy leaderboard-only files cannot encode IDs of work that was pending
    // when they were written. Start the resumed allocator in a fresh high-bit
    // epoch so a late pre-restart worker cannot collide with a newly issued ID.
    const EPOCH_BIT: u64 = 1 << 62;
    EPOCH_BIT | (rand::random::<u64>() & (EPOCH_BIT - 1))
}

fn lease_deadline(duration: Duration) -> Result<u64, String> {
    if duration.is_zero() {
        return Err("trial lease duration must be greater than zero".to_string());
    }
    let millis = duration.as_millis().min(u128::from(u64::MAX)) as u64;
    Ok(unix_time_millis().saturating_add(millis.max(1)))
}

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
                min: s.min(),
                max: s.max(),
                scale: "linear".to_string(),
            },
            Self::RealLog(s) => ParamConfig::Real {
                min: s.min(),
                max: s.max(),
                scale: "log".to_string(),
            },
            Self::RealLog10(s) => ParamConfig::Real {
                min: s.min(),
                max: s.max(),
                scale: "log10".to_string(),
            },
            Self::Integer(s) => ParamConfig::Integer {
                min: s.min(),
                max: s.max(),
            },
            Self::Categorical(s) => ParamConfig::Categorical {
                choices: s.choices().to_vec(),
            },
        }
    }

    fn param_info(&self) -> ParamInfo {
        match self {
            Self::RealLinear(s) => ParamInfo {
                param_type: "real".into(),
                min: s.min(),
                max: s.max(),
                scale: LinearScale::name().to_string(),
                choices: None,
            },
            Self::RealLog(s) => ParamInfo {
                param_type: "real".into(),
                min: s.min(),
                max: s.max(),
                scale: LogScale::name().to_string(),
                choices: None,
            },
            Self::RealLog10(s) => ParamInfo {
                param_type: "real".into(),
                min: s.min(),
                max: s.max(),
                scale: Log10Scale::name().to_string(),
                choices: None,
            },
            Self::Integer(s) => ParamInfo {
                param_type: "integer".into(),
                min: s.min() as f64,
                max: s.max() as f64,
                scale: "linear".into(),
                choices: None,
            },
            Self::Categorical(s) => ParamInfo {
                param_type: "categorical".into(),
                min: 0.0,
                max: (s.cardinality() - 1) as f64,
                scale: "linear".into(),
                choices: Some(s.choices().to_vec()),
            },
        }
    }
}

// =============================================================================
// DynSpace: named parameter space built from DynDimension variants
// =============================================================================

/// A flat, named parameter space built from `DynDimension` variants.
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
        obj.len() == self.dims.len()
            && self
                .dims
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
/// model once its first empirical fit is available; the model is periodically
/// refit to elite trials.
///
/// The default exploration budget follows the formula from the paper:
/// `min(floor(S / 5), 50 + 2n)`, where `S` is the intended number of
/// simulations and `n` is the dimensionality.
#[derive(Debug)]
pub struct AutoStrategy {
    sobol: SobolStrategy<DynSpace>,
    gmm: GmmStrategy<DynSpace>,
    exploration_budget: usize,
    ongoing_exploration_period: usize,
    /// Whether exploitation may use the GMM rather than the uninformed prior.
    /// This is distinct from the GMM's sampling epoch so legacy checkpoints can
    /// preserve their historical route without pretending an epoch existed.
    gmm_sampling_ready: bool,
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
        Self::new_with_exploration_period(
            dim,
            exploration_budget,
            DEFAULT_ONGOING_EXPLORATION_PERIOD,
            seed,
        )
    }

    fn new_with_exploration_period(
        dim: usize,
        exploration_budget: usize,
        ongoing_exploration_period: usize,
        seed: Option<u64>,
    ) -> Self {
        let (sobol_seed, gmm_seed) = match seed {
            // Fold the high 32 bits into the low 32 instead of truncating, so two
            // u64 seeds that differ only in their high bits yield distinct Sobol
            // seeds. Deterministic: the same u64 always folds to the same u32.
            Some(s) => ((s ^ (s >> 32)) as u32, s),
            None => (42, rand::random()),
        };
        Self {
            sobol: SobolStrategy::new(sobol_seed),
            gmm: GmmStrategy::uniform_prior(gmm_seed, dim, AUTO_GMM_PRIOR_VARIANCE)
                .expect("AutoStrategy dimensions and prior variance are validated"),
            exploration_budget,
            ongoing_exploration_period,
            gmm_sampling_ready: false,
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
            ongoing_exploration_period: self.ongoing_exploration_period,
            gmm_sampling_ready: self.gmm_sampling_ready,
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

        let mut state = serializer.serialize_struct("AutoStrategy", 7)?;
        state.serialize_field("sobol", &self.sobol)?;
        state.serialize_field("gmm", &self.gmm)?;
        state.serialize_field("exploration_budget", &self.exploration_budget)?;
        state.serialize_field(
            "ongoing_exploration_period",
            &self.ongoing_exploration_period,
        )?;
        state.serialize_field("gmm_sampling_ready", &self.gmm_sampling_ready)?;
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
            #[serde(default = "default_ongoing_exploration_period")]
            ongoing_exploration_period: usize,
            #[serde(default)]
            gmm_sampling_ready: Option<bool>,
            trial_count: usize,
            #[serde(default)]
            issued_count: Option<usize>,
        }

        let state = AutoStrategySerde::deserialize(deserializer)?;
        let issued_count = state.issued_count.unwrap_or(state.trial_count);
        if issued_count < state.trial_count {
            return Err(serde::de::Error::custom(format!(
                "auto strategy issued_count {issued_count} is smaller than trial_count {}",
                state.trial_count
            )));
        }
        if state.ongoing_exploration_period == 1 {
            return Err(serde::de::Error::custom(
                "auto strategy ongoing_exploration_period must be 0 or at least 2",
            ));
        }
        // Before this marker existed, Auto switched to its GMM strictly at the
        // issued-suggestion boundary—even if that model was still the uniform
        // prior. Preserve that route for old checkpoints. Epoch-aware fitted
        // checkpoints can be recognized directly; the boundary inference also
        // covers the older integer-cursor format whose epoch is necessarily 0.
        let gmm_sampling_ready = state.gmm_sampling_ready.unwrap_or_else(|| {
            state.gmm.refit_epoch() > 0 || issued_count >= state.exploration_budget
        });
        Ok(Self {
            sobol: state.sobol,
            gmm: state.gmm,
            exploration_budget: state.exploration_budget,
            ongoing_exploration_period: state.ongoing_exploration_period,
            gmm_sampling_ready,
            trial_count: state.trial_count,
            issued_count: AtomicUsize::new(issued_count),
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
                let periodic_exploration = s.ongoing_exploration_period >= 2
                    && issued >= s.exploration_budget
                    && (issued - s.exploration_budget + 1)
                        .is_multiple_of(s.ongoing_exploration_period);
                // Until the first empirical fit is installed, the uniform GMM
                // prior contains no information beyond global exploration. A
                // burst of concurrent asks can outrun tells at the warm-up
                // boundary, so keep those asks on Sobol rather than sampling
                // the arbitrary prior.
                if issued < s.exploration_budget || !s.gmm_sampling_ready || periodic_exploration {
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
    /// Epoch of an `auto` strategy that is still waiting for its first
    /// empirical GMM fit. Capturing this before queued maintenance and
    /// comparing it again under `refit_lock` coalesces concurrent retries once
    /// another completion has installed the model.
    fn pending_initial_fit_epoch(&self) -> Option<u64> {
        match &self.inner {
            DynStrategyInner::Auto(strategy) if !strategy.gmm_sampling_ready => {
                Some(strategy.gmm.refit_epoch())
            }
            _ => None,
        }
    }

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

    fn validate_for_study(
        &self,
        config: &StrategyConfig,
        space: &DynSpace,
        completed_count: usize,
        minimum_issued_count: usize,
    ) -> Result<(), String> {
        let expected_kind = match config.strategy_type.as_str() {
            "random" => "random",
            "sobol" => "sobol",
            "gmm" | "auto" => "auto",
            other => return Err(format!("unsupported checkpoint strategy kind '{other}'")),
        };
        let actual_kind = match &self.inner {
            DynStrategyInner::Random(_) => "random",
            DynStrategyInner::Sobol(_) => "sobol",
            DynStrategyInner::Gmm(_) => "gmm",
            DynStrategyInner::Auto(_) => "auto",
        };
        if actual_kind != expected_kind {
            return Err(format!(
                "checkpoint strategy state is '{actual_kind}' but study configuration requires '{expected_kind}'"
            ));
        }

        let expected_dim = space.dimensionality();
        let minimum_issued_u64 =
            u64::try_from(minimum_issued_count).map_err(|_| "issued count exceeds u64")?;
        match &self.inner {
            DynStrategyInner::Gmm(strategy) => {
                let params = strategy
                    .params()
                    .map_err(|error| format!("invalid checkpoint GMM state: {error}"))?;
                strategy
                    .epoch_index()
                    .map_err(|error| format!("invalid checkpoint GMM sampling state: {error}"))?;
                if params.dim() != expected_dim {
                    return Err(format!(
                        "checkpoint GMM dimension {} does not match search-space dimension {expected_dim}",
                        params.dim()
                    ));
                }
                if let Some(seed) = config.seed {
                    if strategy.seed() != seed {
                        return Err(format!(
                            "checkpoint GMM seed {} does not match configured seed {seed}",
                            strategy.seed()
                        ));
                    }
                }
                if strategy.counter() < minimum_issued_u64 || strategy.counter() == u64::MAX {
                    return Err(format!(
                        "checkpoint GMM cursor {} is smaller than minimum issued count {minimum_issued_count}",
                        strategy.counter()
                    ));
                }
            }
            DynStrategyInner::Auto(strategy) => {
                let params = strategy
                    .gmm
                    .params()
                    .map_err(|error| format!("invalid checkpoint GMM state: {error}"))?;
                strategy
                    .gmm
                    .epoch_index()
                    .map_err(|error| format!("invalid checkpoint GMM sampling state: {error}"))?;
                if params.dim() != expected_dim {
                    return Err(format!(
                        "checkpoint GMM dimension {} does not match search-space dimension {expected_dim}",
                        params.dim()
                    ));
                }
                if strategy.trial_count != completed_count {
                    return Err(format!(
                        "checkpoint strategy trial_count {} does not match completed count {completed_count}",
                        strategy.trial_count
                    ));
                }
                let issued_count = strategy.issued_count.load(Ordering::Relaxed);
                if issued_count < minimum_issued_count || issued_count == usize::MAX {
                    return Err(format!(
                        "checkpoint auto issued_count {issued_count} is outside the valid range {minimum_issued_count}..{}",
                        usize::MAX - 1
                    ));
                }
                if let Some(seed) = config.seed {
                    let expected_sobol_seed = (seed ^ (seed >> 32)) as u32;
                    if strategy.gmm.seed() != seed || strategy.sobol.seed() != expected_sobol_seed {
                        return Err(format!(
                            "checkpoint auto strategy seeds do not match configured seed {seed}"
                        ));
                    }
                }
                let initial_sobol = issued_count.min(strategy.exploration_budget);
                let post_exploration = issued_count.saturating_sub(strategy.exploration_budget);
                let periodic_sobol = if strategy.ongoing_exploration_period >= 2 {
                    post_exploration / strategy.ongoing_exploration_period
                } else {
                    0
                };
                let expected_sobol = initial_sobol.saturating_add(periodic_sobol);
                let expected_sobol_u32 = u32::try_from(expected_sobol).map_err(|_| {
                    "checkpoint auto Sobol cursor exceeds the supported u32 range".to_string()
                })?;
                if strategy.sobol.index() < expected_sobol_u32 || strategy.sobol.index() == u32::MAX
                {
                    return Err(format!(
                        "checkpoint auto Sobol cursor {} is smaller than expected {expected_sobol}",
                        strategy.sobol.index()
                    ));
                }
                if strategy.gmm.counter() == u64::MAX {
                    return Err(format!(
                        "checkpoint auto GMM cursor {} is exhausted",
                        strategy.gmm.counter()
                    ));
                }
                // Concurrent asks can cross the warm-up boundary before the
                // first fit completes. Those requests deliberately stay on
                // Sobol, so the nominal cadence cannot provide a lower bound
                // for the GMM cursor. Every issued request must nevertheless
                // be represented by one of the two sampler cursors. Imported
                // legacy history may conservatively advance both, hence `>=`.
                let routed_count = u128::from(strategy.sobol.index())
                    .saturating_add(u128::from(strategy.gmm.counter()));
                if routed_count < issued_count as u128 {
                    return Err(format!(
                        "checkpoint auto sampler cursors account for {routed_count} suggestions, fewer than issued_count {issued_count}"
                    ));
                }
                if let Some(expected_budget) = config.exploration_budget {
                    if strategy.exploration_budget != expected_budget {
                        return Err(format!(
                            "checkpoint exploration budget {} does not match configured budget {expected_budget}",
                            strategy.exploration_budget
                        ));
                    }
                }
                let expected_period = config
                    .ongoing_exploration_period
                    .unwrap_or(DEFAULT_ONGOING_EXPLORATION_PERIOD);
                if strategy.ongoing_exploration_period != expected_period {
                    return Err(format!(
                        "checkpoint ongoing exploration period {} does not match configured period {expected_period}",
                        strategy.ongoing_exploration_period
                    ));
                }
                let expected_components = config.max_components.unwrap_or(DEFAULT_MAX_COMPONENTS);
                let checkpoint_components = strategy.gmm.get_refit_config().n_components();
                if checkpoint_components != expected_components {
                    return Err(format!(
                        "checkpoint maximum GMM components {checkpoint_components} does not match configured maximum {expected_components}"
                    ));
                }
            }
            DynStrategyInner::Random(strategy) => {
                if let Some(seed) = config.seed {
                    if strategy.seed() != seed {
                        return Err(format!(
                            "checkpoint random seed {} does not match configured seed {seed}",
                            strategy.seed()
                        ));
                    }
                }
                if strategy.counter() < minimum_issued_u64 || strategy.counter() == u64::MAX {
                    return Err(format!(
                        "checkpoint random cursor {} is smaller than minimum issued count {minimum_issued_count}",
                        strategy.counter()
                    ));
                }
            }
            DynStrategyInner::Sobol(strategy) => {
                if let Some(seed) = config.seed {
                    let expected_seed = (seed ^ (seed >> 32)) as u32;
                    if strategy.seed() != expected_seed {
                        return Err(format!(
                            "checkpoint Sobol seed {} does not match configured folded seed {expected_seed}",
                            strategy.seed()
                        ));
                    }
                }
                let minimum_index = u32::try_from(minimum_issued_count).map_err(|_| {
                    "checkpoint Sobol issued count exceeds the supported u32 range".to_string()
                })?;
                if strategy.index() < minimum_index || strategy.index() == u32::MAX {
                    return Err(format!(
                        "checkpoint Sobol cursor {} is outside the valid range {minimum_index}..{}",
                        strategy.index(),
                        u32::MAX - 1
                    ));
                }
            }
        }
        Ok(())
    }

    fn resolved_seed(&self) -> u64 {
        match &self.inner {
            DynStrategyInner::Random(strategy) => strategy.seed(),
            DynStrategyInner::Sobol(strategy) => u64::from(strategy.seed()),
            DynStrategyInner::Gmm(strategy) => strategy.seed(),
            DynStrategyInner::Auto(strategy) => strategy.gmm.seed(),
        }
    }

    fn try_refit(
        &mut self,
        space: &DynSpace,
        trials: &[(serde_json::Value, f64)],
    ) -> Result<(), String> {
        match &mut self.inner {
            DynStrategyInner::Gmm(strategy) => strategy
                .try_refit(space, trials)
                .map_err(|error| error.to_string()),
            DynStrategyInner::Auto(strategy) => {
                strategy
                    .gmm
                    .try_refit(space, trials)
                    .map_err(|error| error.to_string())?;
                if !trials.is_empty() {
                    strategy.gmm_sampling_ready = true;
                }
                Ok(())
            }
            DynStrategyInner::Random(_) | DynStrategyInner::Sobol(_) => Ok(()),
        }
    }

    fn reconcile_imported_history(
        &mut self,
        space: &DynSpace,
        completed_count: usize,
        trials: &[(serde_json::Value, f64)],
    ) -> Result<(), String> {
        // Trial IDs can come from a deliberately sparse, high restart epoch and
        // are not sampler cursors. A leaderboard-only checkpoint has no pending
        // jobs, so the monotonic completion count is the faithful cursor.
        let sample_count = u64::try_from(completed_count).unwrap_or(u64::MAX);
        let sobol_index = u32::try_from(completed_count).unwrap_or(u32::MAX);
        match &mut self.inner {
            DynStrategyInner::Random(strategy) => strategy.advance_to(sample_count),
            DynStrategyInner::Sobol(strategy) => strategy.advance_to(sobol_index),
            DynStrategyInner::Gmm(strategy) => {
                strategy.advance_to(sample_count);
                strategy
                    .try_refit(space, trials)
                    .map_err(|error| error.to_string())?;
            }
            DynStrategyInner::Auto(strategy) => {
                strategy.sobol.advance_to(sobol_index);
                strategy.trial_count = completed_count;
                strategy
                    .issued_count
                    .fetch_max(completed_count, Ordering::Relaxed);
                let gmm_counter = strategy.gmm.counter().max(sample_count);
                if trials.is_empty() {
                    // A leaderboard-only import replaces the history. An old
                    // fitted model cannot remain eligible when the replacement
                    // has no adequate feasible workset, so restore the neutral
                    // prior while preserving configuration and monotonic cursor
                    // state. Auto will remain on Sobol until a later valid fit.
                    let refit_config = strategy.gmm.get_refit_config().clone();
                    let mut gmm = GmmStrategy::uniform_prior(
                        strategy.gmm.seed(),
                        space.dimensionality(),
                        AUTO_GMM_PRIOR_VARIANCE,
                    )
                    .map_err(|error| error.to_string())?;
                    gmm.set_refit_config(refit_config);
                    gmm.advance_to(gmm_counter);
                    strategy.gmm = gmm;
                    strategy.gmm_sampling_ready = false;
                } else {
                    strategy.gmm.advance_to(gmm_counter);
                    strategy
                        .gmm
                        .try_refit(space, trials)
                        .map_err(|error| error.to_string())?;
                    strategy.gmm_sampling_ready = true;
                }
            }
        }
        Ok(())
    }
}

impl opt_engine::traits::RefittableStrategy for DynStrategy {
    fn refit(&mut self, space: &DynSpace, trials: &[(serde_json::Value, f64)]) {
        if let Err(error) = self.try_refit(space, trials) {
            eprintln!("DynStrategy::refit rejected invalid input: {error}");
        }
    }

    fn reconcile_after_refit(&mut self, live: &Self) {
        match (&mut self.inner, &live.inner) {
            (DynStrategyInner::Auto(s), DynStrategyInner::Auto(l)) => {
                // refit rebuilt only the GMM model; everything else is live
                // sampling state that advanced while the refit ran off-lock.
                s.sobol = l.sobol.clone();
                opt_engine::traits::RefittableStrategy::reconcile_after_refit(&mut s.gmm, &l.gmm);
                s.trial_count = l.trial_count;
                // A successful nonempty refit marks the fitted snapshot ready;
                // an empty no-op clone carries the live value already.
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
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveConfig {
    pub field: String,
    #[serde(alias = "type")]
    pub obj_type: String,
    #[serde(default)]
    pub target: Option<f64>,
    #[serde(default)]
    pub limit: Option<f64>,
    /// TLP score at the limit (when target and limit are configured) and
    /// relative weight within a priority group. The TLP segment's slope is
    /// `priority / (limit - target)`.
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
#[serde(deny_unknown_fields)]
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
    /// Cadence for ongoing Sobol exploration after the initial warm-up.
    /// Missing resolves to the legacy cadence of 5; 0 disables it.
    #[serde(default)]
    pub ongoing_exploration_period: Option<usize>,
    /// Optional seed for reproducible runs. When `None`, strategies use their
    /// default seeding (Sobol=42, others use random seeds).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Fraction of top trials used for GMM refitting (default: 0.25).
    /// Must be in (0.0, 1.0].
    #[serde(default)]
    pub elite_fraction: Option<f64>,
    /// Maximum number of GMM components considered during fitting. Missing
    /// resolves to the legacy cap of 3.
    #[serde(default)]
    pub max_components: Option<usize>,
    /// Minimum feasible elite workset size required for refitting. A scheduled
    /// refit is skipped until selection can supply this many trials. Missing
    /// resolves to the legacy floor of 1.
    #[serde(default)]
    pub min_elite_samples: Option<usize>,
    /// Maximum elite samples used by one GMM fit. This bounds EM cost but does
    /// not change the abstract elite definition.
    #[serde(default = "default_max_refit_samples")]
    pub max_refit_samples: usize,
    /// Maximum retained trials ranked during one elite-selection pass. Longer
    /// histories are covered by deterministic chronological strata.
    #[serde(default = "default_max_refit_candidates")]
    pub max_refit_candidates: usize,
}

fn default_refit_interval() -> usize {
    20
}

fn default_ongoing_exploration_period() -> usize {
    DEFAULT_ONGOING_EXPLORATION_PERIOD
}

fn default_max_refit_samples() -> usize {
    DEFAULT_MAX_REFIT_SAMPLES
}

fn default_max_refit_candidates() -> usize {
    DEFAULT_MAX_REFIT_CANDIDATES
}

impl StrategyConfig {
    /// Resolve controls added after the original checkpoint schema. Keeping
    /// these as optional on input lets old YAML and embedded checkpoint configs
    /// retain their historical behavior, while exported configs record the
    /// concrete values that actually govern sampling.
    fn resolve_calibration_control_defaults(&mut self) {
        self.ongoing_exploration_period
            .get_or_insert(DEFAULT_ONGOING_EXPLORATION_PERIOD);
        self.max_components.get_or_insert(DEFAULT_MAX_COMPONENTS);
        self.min_elite_samples
            .get_or_insert(DEFAULT_MIN_ELITE_SAMPLES);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    if let Some(checkpoint) = &config.checkpoint {
        if checkpoint.interval == 0 {
            return Err("checkpoint.interval must be at least 1".to_string());
        }
        if checkpoint.max_checkpoints == Some(0) {
            return Err("checkpoint.max_checkpoints must be at least 1 when set".to_string());
        }
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
                let validation = match scale.as_str() {
                    "linear" => ContinuousSpace::try_new(*min, *max).map(|_| ()),
                    "log" | "ln" => {
                        ContinuousSpace::try_with_scale(*min, *max, LogScale).map(|_| ())
                    }
                    "log10" => ContinuousSpace::try_with_scale(*min, *max, Log10Scale).map(|_| ()),
                    other => {
                        return Err(format!(
                            "Parameter '{name}': unknown real scale '{other}'. Expected one of: linear, log, ln, log10",
                        ));
                    }
                };
                validation.map_err(|error| format!("Parameter '{name}': {error}"))?;
            }
            ParamConfig::Integer { min, max } => {
                DiscreteSpace::try_new(*min, *max)
                    .map_err(|error| format!("Parameter '{name}': {error}"))?;
            }
            ParamConfig::Categorical { choices } => {
                CategoricalSpace::try_new(choices.clone())
                    .map_err(|error| format!("Parameter '{name}': {error}"))?;
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

    let mut fields = HashSet::with_capacity(objectives.len());
    for obj in objectives {
        if obj.field.trim().is_empty() {
            return Err("Objective field names must not be empty".to_string());
        }
        if !fields.insert(obj.field.as_str()) {
            return Err(format!(
                "Objective field '{}' is configured more than once",
                obj.field
            ));
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
        if let Some(target) = obj.target {
            if !target.is_finite() {
                return Err(format!(
                    "Objective '{}': target must be finite, got {}",
                    obj.field, target
                ));
            }
        }
        if let Some(limit) = obj.limit {
            if !limit.is_finite() {
                return Err(format!(
                    "Objective '{}': limit must be finite, got {}",
                    obj.field, limit
                ));
            }
        }

        if obj.target.is_some() != obj.limit.is_some() {
            return Err(format!(
                "Objective '{}': target and limit must either both be set or both be omitted",
                obj.field
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
    if strategy.ongoing_exploration_period == Some(1) {
        return Err("strategy.ongoing_exploration_period must be 0 or at least 2".to_string());
    }
    if let Some(elite_fraction) = strategy.elite_fraction {
        if !elite_fraction.is_finite() || elite_fraction <= 0.0 || elite_fraction > 1.0 {
            return Err(format!(
                "strategy.elite_fraction must be finite and in (0, 1], got {elite_fraction}",
            ));
        }
    }
    if strategy.max_refit_samples == 0 {
        return Err("strategy.max_refit_samples must be at least 1".to_string());
    }
    if strategy.max_components == Some(0) {
        return Err("strategy.max_components must be at least 1".to_string());
    }
    if strategy.min_elite_samples == Some(0) {
        return Err("strategy.min_elite_samples must be at least 1".to_string());
    }
    if let Some(min_elite_samples) = strategy.min_elite_samples {
        if min_elite_samples > strategy.max_refit_samples {
            return Err(format!(
                "strategy.min_elite_samples must not exceed max_refit_samples ({}), got {min_elite_samples}",
                strategy.max_refit_samples
            ));
        }
    }
    if strategy.max_refit_candidates < strategy.max_refit_samples {
        return Err(format!(
            "strategy.max_refit_candidates must be at least max_refit_samples ({}), got {}",
            strategy.max_refit_samples, strategy.max_refit_candidates,
        ));
    }

    Ok(())
}

// =============================================================================
// HolaEngine: the top-level Ask/Tell interface
// =============================================================================

/// A trial returned by `ask()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// For TLP fields: 0 = target met, `priority` = at the limit, and inf =
    /// infeasible beyond it.
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

/// Result metadata for callers that must distinguish a newly committed tell
/// from an idempotent replay of an earlier success.
#[derive(Clone, Debug)]
pub struct TellOutcome {
    pub completed: CompletedTrial,
    pub trial_count: usize,
    pub newly_committed: bool,
    /// Failures in post-commit maintenance. The trial is durable even when this
    /// is non-empty, so callers must not retry it as an ingestion failure.
    pub post_commit_warnings: Vec<String>,
}

/// Internal lifecycle view used to reconcile distributed callback workers.
///
/// `NotPending` deliberately combines expired, cancelled, and unknown trials:
/// none of those states should trigger another cancellation attempt. A recent
/// completion remains distinguishable after bounded-leaderboard eviction via
/// the engine's completion-receipt ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(feature = "server")]
pub(crate) enum TrialLifecycle {
    Completed,
    Pending,
    NotPending,
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

    fn raw_metrics(&self, trial_id: u64) -> Option<&serde_json::Value> {
        match self {
            DynLeaderboard::Scalar(lb) => lb.get(trial_id)?.raw_metrics.as_ref(),
            DynLeaderboard::Vector(lb) => lb.get(trial_id)?.raw_metrics.as_ref(),
        }
    }

    fn candidate_and_timestamp(&self, trial_id: u64) -> Option<(&serde_json::Value, u64)> {
        match self {
            DynLeaderboard::Scalar(lb) => {
                let trial = lb.get(trial_id)?;
                Some((&trial.candidate, trial.timestamp))
            }
            DynLeaderboard::Vector(lb) => {
                let trial = lb.get(trial_id)?;
                Some((&trial.candidate, trial.timestamp))
            }
        }
    }

    /// Validate completed state before a parsed checkpoint can replace live
    /// state. Serde checks the leaderboard's structural counters; this layer
    /// checks the study-specific candidate and observation contracts that the
    /// generic leaderboard cannot know about.
    fn validate_for_study(
        &self,
        space: &DynSpace,
        objectives: &[ObjectiveConfig],
    ) -> Result<(), String> {
        // IEEE-754 permits multiple NaN payload/sign encodings. Persistence
        // canonicalizes NaN, so invariant validation compares all NaNs as the
        // same semantic invalid observation while retaining bit equality for
        // every finite value and infinity sign.
        let same_float = |left: f64, right: f64| {
            (left.is_nan() && right.is_nan()) || left.to_bits() == right.to_bits()
        };
        match self {
            DynLeaderboard::Scalar(leaderboard) => {
                for trial in leaderboard.iter() {
                    if !space.contains(&trial.candidate) {
                        return Err(format!(
                            "completed trial {} has a candidate outside the configured space",
                            trial.trial_id
                        ));
                    }
                    if let Some(raw) = &trial.raw_metrics {
                        let expected = scalarize_raw(raw, objectives);
                        if !same_float(trial.observation, expected) {
                            return Err(format!(
                                "completed trial {} observation conflicts with its raw metrics",
                                trial.trial_id
                            ));
                        }
                    }
                }
            }
            DynLeaderboard::Vector(leaderboard) => {
                let expected_keys: Vec<String> =
                    vectorize_raw(&serde_json::Value::Null, objectives)
                        .into_keys()
                        .collect();
                for trial in leaderboard.iter() {
                    if !space.contains(&trial.candidate) {
                        return Err(format!(
                            "completed trial {} has a candidate outside the configured space",
                            trial.trial_id
                        ));
                    }
                    if trial.observation.len() != expected_keys.len()
                        || !expected_keys
                            .iter()
                            .all(|key| trial.observation.contains_key(key))
                    {
                        return Err(format!(
                            "completed trial {} observation has the wrong objective-group schema",
                            trial.trial_id
                        ));
                    }
                    if let Some(raw) = &trial.raw_metrics {
                        let expected = vectorize_raw(raw, objectives);
                        let matches = expected.len() == trial.observation.len()
                            && expected.iter().all(|(key, expected_value)| {
                                trial
                                    .observation
                                    .get(key)
                                    .is_some_and(|actual| same_float(*actual, *expected_value))
                            });
                        if !matches {
                            return Err(format!(
                                "completed trial {} observation conflicts with its raw metrics",
                                trial.trial_id
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
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

    /// Monotonic completion count in the engine's native count type.
    ///
    /// Cadence decisions must use this rather than [`Self::len`]: a bounded
    /// leaderboard's retained length stops growing at its cap, while refits,
    /// checkpoints, and study budgets must continue advancing with every
    /// committed result.
    fn completed_count(&self) -> usize {
        usize::try_from(self.total_completed()).unwrap_or(usize::MAX)
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

    /// Return elite trials as (candidate, strategy observation) for refitting.
    ///
    /// A scalar leaderboard uses its ordinary score order. A multi-group
    /// leaderboard uses the same NSGA-II order as the public leaderboard:
    /// lower non-domination rank first, then higher crowding distance. Both
    /// paths rank the full retained history while it fits the configured work
    /// bound; longer histories use deterministic chronological strata so no
    /// newest-only bias is introduced.
    fn top_k_for_refit(
        &self,
        k: usize,
        max_candidates: usize,
        objectives: &[ObjectiveConfig],
    ) -> Vec<(serde_json::Value, f64)> {
        match self {
            DynLeaderboard::Scalar(lb) => lb
                .top_k_stratified(k, max_candidates)
                .into_iter()
                .map(|t| (t.candidate, t.observation))
                .collect(),
            DynLeaderboard::Vector(lb) => lb
                .select_nsga2_stratified(k, max_candidates)
                .into_iter()
                .map(|ranked| {
                    let t = ranked.trial;
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

                // Delegate to the leaderboard's canonical total-order policy.
                // In particular, finite values sort before infinities and NaN
                // sorts last. Reimplementing this with `partial_cmp` makes NaN
                // compare equal to every value and can therefore disagree with
                // `completed_trials()` for a lossless non-finite checkpoint.
                let rank = lb.rank_of(trial_id, include_infeasible)?;

                Some(build_completed_scalar(trial, rank, objectives))
            }
            DynLeaderboard::Vector(_) => {
                let all = self.completed_trials("rank", include_infeasible, objectives);
                all.into_iter().find(|ct| ct.trial_id == trial_id)
            }
        }
    }

    /// Build the just-told trial's view and the owned inputs needed to finish
    /// vector ranking without borrowing the leaderboard.
    ///
    /// The scalar path computes its O(n) `rank_of` directly and returns a fully
    /// populated `CompletedTrial`. The vector path fills `pareto_front` via the
    /// front-peeling `pareto_rank_of` (no trial clones) and returns a cheap
    /// `(trial_id, observation)` snapshot of the participating trials plus the
    /// target id. The caller finalizes the response before exposing the
    /// idempotency receipt, without cloning every candidate or raw-metrics DTO.
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
                if !snapshot.iter().any(|(id, _)| *id == trial_id) {
                    return None;
                }
                // Ranking is filled from `snapshot` by vector_rank.
                let completed = build_completed_vector(trial, 0, objectives);
                Some((completed, Some((snapshot, trial_id))))
            }
        }
    }

    /// Build a completion payload without computing leaderboard-wide rank.
    ///
    /// This is only used by the local `Study.run` batch path, which never
    /// exposes the per-completion return value. The private placeholder rank is
    /// replaced before a receipt can be returned by the public `tell` API.
    fn completed_without_ranking(
        &self,
        trial_id: u64,
        objectives: &[ObjectiveConfig],
    ) -> Option<CompletedTrial> {
        match self {
            DynLeaderboard::Scalar(lb) => Some(build_completed_scalar(
                lb.get(trial_id)?.clone(),
                0,
                objectives,
            )),
            DynLeaderboard::Vector(lb) => Some(build_completed_vector(
                lb.get(trial_id)?.clone(),
                0,
                objectives,
            )),
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
                // Rank only feasible trials into Pareto fronts. Infeasible
                // observations are still returned when requested, but they are
                // placed after every feasible front and can never masquerade as
                // Pareto-front zero when all observations violate a constraint.
                let ranked = lb.ranked_trials();
                let infeasible_front = ranked
                    .iter()
                    .map(|ranked_trial| ranked_trial.rank)
                    .max()
                    .unwrap_or(1);
                let mut results: Vec<CompletedTrial> = ranked
                    .into_iter()
                    .map(|rt| {
                        let pareto_front = rt.rank.saturating_sub(1); // 1-indexed → 0-indexed
                        build_completed_vector(rt.trial, pareto_front, objectives)
                    })
                    .collect();
                if include_infeasible {
                    results.extend(
                        lb.iter()
                            .filter(|trial| {
                                !Leaderboard::<
                                    serde_json::Value,
                                    BTreeMap<String, f64>,
                                >::trial_is_feasible(trial)
                            })
                            .cloned()
                            .map(|trial| {
                                build_completed_vector(trial, infeasible_front, objectives)
                            }),
                    );
                }
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
    let feasible = t.observation.is_finite();
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
        // Pareto fronts do not apply to scalar studies, but the dashboard uses
        // this field defensively. Reserve zero for a feasible best trial.
        pareto_front: if feasible { rank } else { rank.max(1) },
        completed_at: t.timestamp,
    }
}

/// Build a CompletedTrial from a vector leaderboard trial.
fn build_completed_vector(
    t: Trial<serde_json::Value, BTreeMap<String, f64>>,
    pareto_front: usize,
    objectives: &[ObjectiveConfig],
) -> CompletedTrial {
    let feasible = is_feasible_multi(&t.observation);
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
        pareto_front: if feasible {
            pareto_front
        } else {
            pareto_front.max(1)
        },
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

/// Compute every front rank in O(N log N) when a snapshot has exactly two
/// consistent, finite objectives. A Fenwick tree stores the best chain depth
/// among compressed y coordinates while points are swept by x. Exact duplicate
/// points are queried as a group before updating so they do not dominate one
/// another.
fn two_objective_front_ranks(participants: &[(u64, BTreeMap<String, f64>)]) -> Option<Vec<usize>> {
    let first = participants.first()?;
    let keys: Vec<&String> = first.1.keys().collect();
    if keys.len() != 2 {
        return None;
    }
    let (x_key, y_key) = (keys[0], keys[1]);
    let normalize_zero = |value: f64| if value == 0.0 { 0.0 } else { value };
    let mut points = Vec::with_capacity(participants.len());
    for (index, (_, observation)) in participants.iter().enumerate() {
        if observation.len() != 2 {
            return None;
        }
        let x = normalize_zero(*observation.get(x_key)?);
        let y = normalize_zero(*observation.get(y_key)?);
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        points.push((index, x, y));
    }
    points.sort_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut y_values: Vec<f64> = points.iter().map(|point| point.2).collect();
    y_values.sort_by(f64::total_cmp);
    y_values.dedup_by(|a, b| *a == *b);

    let mut tree = vec![0usize; y_values.len() + 1];
    let query = |tree: &[usize], mut index: usize| {
        let mut best = 0;
        while index > 0 {
            best = best.max(tree[index]);
            index &= index - 1;
        }
        best
    };
    let update = |tree: &mut [usize], mut index: usize, value: usize| {
        while index < tree.len() {
            tree[index] = tree[index].max(value);
            index += index & index.wrapping_neg();
        }
    };

    let mut fronts = vec![0usize; participants.len()];
    let mut start = 0;
    while start < points.len() {
        let mut end = start + 1;
        while end < points.len()
            && points[end].1 == points[start].1
            && points[end].2 == points[start].2
        {
            end += 1;
        }
        let y_index = y_values
            .binary_search_by(|value| value.total_cmp(&points[start].2))
            .expect("compressed coordinate came from the same finite values")
            + 1;
        let front = query(&tree, y_index);
        for point in &points[start..end] {
            fronts[point.0] = front;
        }
        update(&mut tree, y_index, front + 1);
        start = end;
    }
    Some(fronts)
}

/// Compute every trial's 0-indexed Pareto front from a cheap observation
/// snapshot, preserving iteration order within each front.
fn vector_front_ranks(participants: &[(u64, BTreeMap<String, f64>)]) -> Vec<usize> {
    if let Some(fronts) = two_objective_front_ranks(participants) {
        return fronts;
    }

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
    let mut fronts = vec![0usize; n];
    let mut front = 0usize;
    while !current.is_empty() {
        for &index in &current {
            fronts[index] = front;
        }
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
        front += 1;
    }
    fronts
}

/// Compute a single trial's 0-indexed NSGA-II global rank and front from a cheap
/// snapshot.
fn vector_rank(
    participants: &[(u64, BTreeMap<String, f64>)],
    target: u64,
) -> Option<(usize, usize)> {
    let target_index = participants.iter().position(|(id, _)| *id == target)?;
    let fronts = vector_front_ranks(participants);
    let target_front = fronts[target_index];
    let rank_base = fronts.iter().filter(|&&front| front < target_front).count();
    let front_trials: Vec<Trial<u64, BTreeMap<String, f64>>> = participants
        .iter()
        .enumerate()
        .filter(|(index, _)| fronts[*index] == target_front)
        .map(|(index, (trial_id, observation))| Trial {
            candidate: *trial_id,
            observation: observation.clone(),
            raw_metrics: None,
            trial_id: *trial_id,
            timestamp: index as u64,
        })
        .collect();
    let mut crowded = Leaderboard::<u64, BTreeMap<String, f64>>::crowding_distance(&front_trials);
    // Stable sorting preserves snapshot order for equal crowding distances,
    // matching Leaderboard::select_nsga2's canonical full-front ordering.
    crowded.sort_by(|left, right| right.1.total_cmp(&left.1));
    let position = crowded
        .iter()
        .position(|(trial, _)| trial.trial_id == target)?;
    Some((rank_base + position, target_front))
}

/// Dashboard/API ranking keeps infeasible observations after every feasible
/// front. This differs deliberately from the generic leaderboard's `*_all`
/// ranking, where an all-infinite population is mutually non-dominating and
/// would therefore be labelled front zero.
fn vector_dashboard_rank(
    participants: &[(u64, BTreeMap<String, f64>)],
    target: u64,
) -> Option<(usize, usize)> {
    let target_index = participants.iter().position(|(id, _)| *id == target)?;
    let feasible: Vec<(u64, BTreeMap<String, f64>)> = participants
        .iter()
        .filter(|(_, observation)| is_feasible_multi(observation))
        .cloned()
        .collect();
    if is_feasible_multi(&participants[target_index].1) {
        return vector_rank(&feasible, target);
    }

    let next_front = vector_front_ranks(&feasible)
        .into_iter()
        .max()
        .map_or(1, |front| front + 1);
    let position = participants[..target_index]
        .iter()
        .filter(|(_, observation)| !is_feasible_multi(observation))
        .count();
    Some((feasible.len() + position, next_front))
}

/// Compute a single trial's 0-indexed NSGA-II global rank from a cheap snapshot
/// of `(trial_id, observation)` pairs, without cloning trials or building the
/// full ranked view.
///
/// `participants` must already be filtered to the same set the leaderboard's
/// ranked view would use (feasible-only or all). The returned rank reproduces
/// `Leaderboard::ranked_trials`/`ranked_trials_all`: fronts are concatenated in
/// non-domination order and each front is ordered by descending crowding
/// distance. Returns `None` if the target is not present in `participants`.
#[cfg(test)]
fn vector_global_rank(participants: &[(u64, BTreeMap<String, f64>)], target: u64) -> Option<usize> {
    vector_rank(participants, target).map(|(rank, _)| rank)
}

/// Convert one floating-point value to its lossless public JSON representation.
///
/// JSON numbers cannot represent IEEE-754 non-finite values, so use the same
/// stable string sentinels as checkpoint persistence and the Python bindings.
fn f64_to_json(value: f64) -> serde_json::Value {
    if value.is_nan() {
        serde_json::Value::from("nan")
    } else if value == f64::INFINITY {
        serde_json::Value::from("inf")
    } else if value == f64::NEG_INFINITY {
        serde_json::Value::from("-inf")
    } else {
        serde_json::Value::from(value)
    }
}

/// Decode a worker metric from JSON without collapsing IEEE-754 values that
/// JSON numbers cannot represent. Python and other strict JSON clients encode
/// these values with the same sentinels used by public score responses.
fn metric_f64(value: &serde_json::Value) -> Option<f64> {
    value.as_f64().or_else(|| match value.as_str() {
        Some("inf") => Some(f64::INFINITY),
        Some("-inf") => Some(f64::NEG_INFINITY),
        Some("nan") => Some(f64::NAN),
        _ => None,
    })
}

fn raw_metric_f64(raw: &serde_json::Value, field: &str) -> Option<f64> {
    raw.get(field).and_then(metric_f64)
}

/// Convert a `BTreeMap<String, f64>` to a lossless JSON object.
fn f64_map_to_json(map: &BTreeMap<String, f64>) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    for (k, v) in map {
        obj.insert(k.clone(), f64_to_json(*v));
    }
    serde_json::Value::Object(obj)
}

/// Compute per-objective TLP scores φ_i from raw metrics.
///
/// Each score is P_i × normalized_distance, so P_i is its score at the limit
/// and its relative weight within the group.
fn compute_scores(raw: &serde_json::Value, objectives: &[ObjectiveConfig]) -> serde_json::Value {
    let mut scores = serde_json::Map::new();
    for obj in objectives {
        let val = raw_metric_f64(raw, &obj.field);
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
        scores.insert(obj.field.clone(), f64_to_json(score));
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
            .map(group_key)
            .unwrap_or_else(|| "score".to_string());
        let mut map = serde_json::Map::new();
        map.insert(key, f64_to_json(score));
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

fn should_attempt_post_commit_refit(
    scheduled_refit: bool,
    captured_initial_fit_epoch: Option<u64>,
    current_initial_fit_epoch: Option<u64>,
    refit_is_eligible: bool,
    initial_fit_has_new_history: bool,
) -> bool {
    if !refit_is_eligible {
        return false;
    }

    match current_initial_fit_epoch {
        Some(epoch) => {
            (scheduled_refit || captured_initial_fit_epoch == Some(epoch))
                && initial_fit_has_new_history
        }
        // Another queued task installed the initial model. A cadence boundary
        // still needs to run if it represents completions beyond that fitted
        // snapshot, but must not immediately refit identical history.
        None if captured_initial_fit_epoch.is_some() => {
            scheduled_refit && initial_fit_has_new_history
        }
        None => scheduled_refit,
    }
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
    /// Implementation bound for samples passed to one GMM fit.
    max_refit_samples: usize,
    /// Implementation bound for trials ranked during elite selection.
    max_refit_candidates: usize,
    /// Required lower bound for a refit's feasible elite workset.
    min_elite_samples: usize,
    /// Latest completed-history count already tried while the auto strategy
    /// was waiting for its first empirical model. This coalesces queued
    /// retries after both successful and unsuccessful attempts.
    initial_fit_attempted_completed: Arc<AtomicUsize>,
    auto_checkpoint: Option<AutoCheckpointConfig>,
    /// Failures from unattended auto-checkpoint writes and rotation. Shared by
    /// clones and exposed to the server metrics endpoint.
    checkpoint_failures: Arc<AtomicU64>,
    /// Failed periodic or objective-change strategy refits. The committed trial
    /// or objective transition remains valid; operators can alert on this
    /// counter and inspect the accompanying warning log.
    refit_failures: Arc<AtomicU64>,
    #[cfg(test)]
    force_refit_failure: Arc<std::sync::atomic::AtomicBool>,
    #[cfg(test)]
    refit_attempts: Arc<AtomicU64>,
    max_trials: Option<usize>,
    /// Opt-in leaderboard retention cap (`None` = unbounded). Recorded here so
    /// `study_config()` can emit it into checkpoints and a resumed study rebuilds
    /// with the same bound.
    max_leaderboard_size: Option<usize>,
}

struct HolaEngineState {
    strategy: DynStrategy,
    /// Effective strategy settings, including the resolved seed. This lives
    /// beside the replaceable sampler state so loading a checkpoint updates
    /// both atomically and subsequent saves describe the sampler actually in
    /// use.
    strategy_template: Option<StrategyConfig>,
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
    ask_idempotency: BTreeMap<String, DynTrial>,
    lease_deadlines: BTreeMap<u64, u64>,
    /// Keyed by trial id for bounded logarithmic retry lookup.
    completion_receipts: BTreeMap<u64, CompletionReceipt>,
    /// Trial ids in commit order, used to prune the oldest receipt in O(1).
    completion_receipt_order: VecDeque<u64>,
    /// Number of retained receipts whose public rank/front has not yet been
    /// materialized. Local batch runners can defer that work and rank the
    /// leaderboard once at the end instead of once per completion.
    deferred_completion_receipts: usize,
}

/// Retry receipt for a committed tell. Public tells retain the exact response
/// view immediately. A local batch may defer its never-exposed rank/front until
/// batch exit, while retaining the full completion payload durably.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompletionReceipt {
    commit_sequence: u64,
    completed: CompletedTrial,
    committed_count: usize,
    #[serde(default)]
    post_commit_warnings: Vec<String>,
    /// The completion payload is durable, but `rank`/`pareto_front` are private
    /// placeholders until the first public observation of this receipt.
    #[serde(default)]
    ranking_deferred: bool,
}

/// Transient job state persisted alongside a full checkpoint.
///
/// Older full checkpoints do not contain this object; those loads retain the
/// legacy behavior of invalidating all pending work. New checkpoints preserve
/// issued candidates and the monotonic ID cursor so a worker response arriving
/// after a restart is still correlated with the candidate it evaluated.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct RuntimeCheckpointState {
    next_pending_id: u64,
    pending: BTreeMap<u64, serde_json::Value>,
    cancelled: Vec<u64>,
    #[serde(default)]
    ask_idempotency: BTreeMap<String, DynTrial>,
    #[serde(default)]
    lease_deadlines: BTreeMap<u64, u64>,
    #[serde(default)]
    completion_receipts: BTreeMap<u64, CompletionReceipt>,
}

enum LeaderboardSnapshot {
    Scalar(Leaderboard<serde_json::Value, f64>),
    Vector(Leaderboard<serde_json::Value, BTreeMap<String, f64>>),
}

struct FullCheckpointSnapshot {
    config: StudyConfig,
    leaderboard: LeaderboardSnapshot,
    strategy: DynStrategy,
    runtime_state: RuntimeCheckpointState,
    description: Option<String>,
    /// Number of records retained in the serialized leaderboard.
    n_trials: usize,
    /// Monotonic number of results committed over the study lifetime.
    total_completed: usize,
}

enum LoadedFullCheckpoint {
    Scalar(Checkpoint<serde_json::Value, f64, DynStrategy>),
    Vector(Checkpoint<serde_json::Value, BTreeMap<String, f64>, DynStrategy>),
}

/// Metadata captured from the exact state snapshot written to a checkpoint.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct SavedCheckpoint {
    pub n_trials: usize,
    pub created_at: u64,
}

impl HolaEngineState {
    fn reset_transient_trial_state_after_load(&mut self) {
        self.next_pending_id = self.leaderboard.normalize_next_trial_id();
        self.pending.clear();
        self.cancelled.clear();
        self.ask_idempotency.clear();
        self.lease_deadlines.clear();
        self.completion_receipts.clear();
        self.completion_receipt_order.clear();
        self.deferred_completion_receipts = 0;
    }

    fn runtime_checkpoint_state(&self) -> RuntimeCheckpointState {
        let mut cancelled: Vec<u64> = self.cancelled.iter().copied().collect();
        cancelled.sort_unstable();
        RuntimeCheckpointState {
            next_pending_id: self.next_pending_id,
            pending: self.pending.clone(),
            cancelled,
            ask_idempotency: self.ask_idempotency.clone(),
            lease_deadlines: self.lease_deadlines.clone(),
            completion_receipts: self.completion_receipts.clone(),
        }
    }

    fn restore_runtime_checkpoint_state(
        &mut self,
        runtime: Option<RuntimeCheckpointState>,
        space: &DynSpace,
    ) -> Result<(), String> {
        let Some(runtime) = runtime else {
            self.reset_transient_trial_state_after_load();
            return Ok(());
        };

        let mut minimum_next = self.leaderboard.next_trial_id();
        if runtime.pending.len() > MAX_PENDING_TRIALS {
            return Err(format!(
                "checkpoint contains {} pending trials (maximum {MAX_PENDING_TRIALS})",
                runtime.pending.len()
            ));
        }
        for (&trial_id, candidate) in &runtime.pending {
            if self.leaderboard.contains_trial_id(trial_id) {
                return Err(format!(
                    "checkpoint pending trial_id {trial_id} is already completed"
                ));
            }
            if !space.contains(candidate) {
                return Err(format!(
                    "checkpoint pending trial_id {trial_id} contains a candidate outside the configured space"
                ));
            }
            minimum_next = minimum_next.max(trial_id.checked_add(1).ok_or_else(|| {
                "checkpoint pending trial_id u64::MAX leaves no assignable ID".to_string()
            })?);
        }

        if runtime.cancelled.len() > MAX_CANCELLED_RETAINED {
            return Err(format!(
                "checkpoint contains {} cancelled trial ids (maximum {MAX_CANCELLED_RETAINED})",
                runtime.cancelled.len()
            ));
        }
        let mut cancelled = HashSet::with_capacity(runtime.cancelled.len());
        for trial_id in runtime.cancelled {
            if self.leaderboard.contains_trial_id(trial_id)
                || runtime.pending.contains_key(&trial_id)
                || !cancelled.insert(trial_id)
            {
                return Err(format!(
                    "checkpoint cancelled trial_id {trial_id} overlaps another job state"
                ));
            }
            minimum_next = minimum_next.max(trial_id.checked_add(1).ok_or_else(|| {
                "checkpoint cancelled trial_id u64::MAX leaves no assignable ID".to_string()
            })?);
        }

        if runtime.ask_idempotency.len() > MAX_ASK_IDEMPOTENCY_KEYS {
            return Err(format!(
                "checkpoint contains {} ask idempotency keys (maximum {MAX_ASK_IDEMPOTENCY_KEYS})",
                runtime.ask_idempotency.len()
            ));
        }
        let mut keyed_trial_ids = HashSet::with_capacity(runtime.ask_idempotency.len());
        for (key, trial) in &runtime.ask_idempotency {
            if key.is_empty() || key.len() > 128 || !key.is_ascii() {
                return Err("checkpoint contains an invalid ask idempotency key".to_string());
            }
            if runtime.pending.get(&trial.trial_id) != Some(&trial.params) {
                return Err(format!(
                    "checkpoint ask idempotency key '{key}' does not match pending trial {}",
                    trial.trial_id
                ));
            }
            if !keyed_trial_ids.insert(trial.trial_id) {
                return Err(format!(
                    "checkpoint contains multiple ask idempotency keys for trial {}",
                    trial.trial_id
                ));
            }
        }
        for (&trial_id, &deadline) in &runtime.lease_deadlines {
            if deadline == 0 || !runtime.pending.contains_key(&trial_id) {
                return Err(format!(
                    "checkpoint lease for trial {trial_id} does not match a pending trial"
                ));
            }
        }
        if runtime.completion_receipts.len() > MAX_COMPLETION_RECEIPTS {
            return Err(format!(
                "checkpoint contains {} completion receipts (maximum {MAX_COMPLETION_RECEIPTS})",
                runtime.completion_receipts.len()
            ));
        }
        let total_completed = self.leaderboard.total_completed();
        let mut receipt_sequences = HashSet::with_capacity(runtime.completion_receipts.len());
        let mut receipt_order = Vec::with_capacity(runtime.completion_receipts.len());
        for (&trial_id, receipt) in &runtime.completion_receipts {
            let sequence = receipt.commit_sequence;
            if sequence == 0 || sequence > total_completed {
                return Err(format!(
                    "checkpoint completion receipt {sequence} is outside completed sequence 1..={total_completed}"
                ));
            }
            if receipt.completed.trial_id != trial_id || !receipt_sequences.insert(sequence) {
                return Err(format!(
                    "checkpoint completion receipt for trial_id {trial_id} has inconsistent identity or sequence"
                ));
            }
            let receipt_count = u64::try_from(receipt.committed_count).unwrap_or(u64::MAX);
            if receipt_count < sequence || receipt_count > total_completed {
                return Err(format!(
                    "checkpoint completion receipt {sequence} has mismatched committed_count {} (expected {sequence}..={total_completed})",
                    receipt.committed_count
                ));
            }
            if runtime.pending.contains_key(&trial_id) || cancelled.contains(&trial_id) {
                return Err(format!(
                    "checkpoint completion receipt for trial_id {trial_id} overlaps another job state"
                ));
            }
            if !space.contains(&receipt.completed.params) {
                return Err(format!(
                    "checkpoint completion receipt for trial_id {trial_id} contains a candidate outside the configured space"
                ));
            }
            let expected_scores = compute_scores(&receipt.completed.metrics, &self.objectives);
            let expected_score_vector =
                compute_score_vector(&receipt.completed.metrics, &self.objectives);
            if receipt.completed.scores != expected_scores
                || receipt.completed.score_vector != expected_score_vector
            {
                return Err(format!(
                    "checkpoint completion receipt for trial_id {trial_id} conflicts with its raw metrics"
                ));
            }
            if receipt.completed.rank >= receipt.committed_count
                // Front zero is reserved for feasible best trials. With no
                // feasible observations, an infeasible trial legitimately uses
                // the sentinel front immediately after all possible fronts,
                // which can equal committed_count.
                || receipt.completed.pareto_front > receipt.committed_count
            {
                return Err(format!(
                    "checkpoint completion receipt for trial_id {trial_id} has an out-of-range rank/front"
                ));
            }
            if receipt.ranking_deferred && !self.leaderboard.contains_trial_id(trial_id) {
                return Err(format!(
                    "checkpoint deferred completion receipt for trial_id {trial_id} has no retained leaderboard trial"
                ));
            }
            if let Some(stored_metrics) = self.leaderboard.raw_metrics(trial_id) {
                if stored_metrics != &receipt.completed.metrics {
                    return Err(format!(
                        "checkpoint completion receipt for trial_id {trial_id} conflicts with leaderboard metrics"
                    ));
                }
            }
            if let Some((stored_candidate, stored_timestamp)) =
                self.leaderboard.candidate_and_timestamp(trial_id)
            {
                if stored_candidate != &receipt.completed.params
                    || stored_timestamp != receipt.completed.completed_at
                {
                    return Err(format!(
                        "checkpoint completion receipt for trial_id {trial_id} conflicts with leaderboard identity"
                    ));
                }
            }
            minimum_next = minimum_next.max(trial_id.checked_add(1).ok_or_else(|| {
                "checkpoint completion receipt trial_id u64::MAX leaves no assignable ID"
                    .to_string()
            })?);
            receipt_order.push((sequence, trial_id));
        }
        receipt_order.sort_unstable();
        if runtime.next_pending_id < minimum_next {
            return Err(format!(
                "checkpoint next_pending_id {} is stale (minimum {minimum_next})",
                runtime.next_pending_id
            ));
        }

        self.next_pending_id = runtime.next_pending_id;
        self.pending = runtime.pending;
        self.cancelled = cancelled;
        self.ask_idempotency = runtime.ask_idempotency;
        self.lease_deadlines = runtime.lease_deadlines;
        self.deferred_completion_receipts = runtime
            .completion_receipts
            .values()
            .filter(|receipt| receipt.ranking_deferred)
            .count();
        self.completion_receipts = runtime.completion_receipts;
        self.completion_receipt_order = receipt_order
            .into_iter()
            .map(|(_, trial_id)| trial_id)
            .collect();
        self.expire_leases(unix_time_millis());
        Ok(())
    }

    fn record_ask_idempotency(&mut self, key: String, trial: DynTrial) {
        self.ask_idempotency.insert(key, trial);
        debug_assert!(self.ask_idempotency.len() <= MAX_ASK_IDEMPOTENCY_KEYS);
        debug_assert!(self.ask_idempotency.len() <= self.pending.len());
    }

    fn remove_ask_idempotency_for_trial(&mut self, trial_id: u64) {
        self.ask_idempotency
            .retain(|_, trial| trial.trial_id != trial_id);
    }

    fn completion_receipt(&self, trial_id: u64) -> Option<&CompletionReceipt> {
        self.completion_receipts.get(&trial_id)
    }

    fn record_completion_receipt(
        &mut self,
        sequence: u64,
        completed: CompletedTrial,
        committed_count: usize,
        ranking_deferred: bool,
    ) {
        let trial_id = completed.trial_id;
        let replaced = self.completion_receipts.insert(
            trial_id,
            CompletionReceipt {
                commit_sequence: sequence,
                completed,
                committed_count,
                post_commit_warnings: Vec::new(),
                ranking_deferred,
            },
        );
        debug_assert!(replaced.is_none());
        if ranking_deferred {
            self.deferred_completion_receipts += 1;
        }
        self.completion_receipt_order.push_back(trial_id);
        while self.completion_receipts.len() > MAX_COMPLETION_RECEIPTS {
            if let Some(oldest_trial_id) = self.completion_receipt_order.pop_front() {
                if self
                    .completion_receipts
                    .remove(&oldest_trial_id)
                    .is_some_and(|receipt| receipt.ranking_deferred)
                {
                    self.deferred_completion_receipts -= 1;
                }
            }
        }
    }

    /// Materialize every private batch receipt against one canonical ranking
    /// snapshot. Deferred receipts are guaranteed to remain in the leaderboard:
    /// the batch commit path stops deferring before a bounded push can evict.
    fn finalize_deferred_completion_receipts(&mut self) -> Result<(), String> {
        if self.deferred_completion_receipts == 0 {
            return Ok(());
        }

        let view_count = self.leaderboard.completed_count();
        let current: BTreeMap<u64, CompletedTrial> = self
            .leaderboard
            .completed_trials("rank", true, &self.objectives)
            .into_iter()
            .map(|trial| (trial.trial_id, trial))
            .collect();
        let expected = self.deferred_completion_receipts;
        if let Some(trial_id) = self
            .completion_receipts
            .iter()
            .find_map(|(&trial_id, receipt)| {
                (receipt.ranking_deferred && !current.contains_key(&trial_id)).then_some(trial_id)
            })
        {
            return Err(format!(
                "Deferred completion receipt for trial {trial_id} has no retained leaderboard trial"
            ));
        }
        let mut finalized = 0usize;
        for (&trial_id, receipt) in &mut self.completion_receipts {
            if !receipt.ranking_deferred {
                continue;
            }
            let completed = current
                .get(&trial_id)
                .expect("deferred receipt backing was checked above");
            receipt.completed = completed.clone();
            receipt.committed_count = view_count;
            receipt.ranking_deferred = false;
            finalized += 1;
        }
        debug_assert_eq!(finalized, expected);
        self.deferred_completion_receipts = 0;
        Ok(())
    }

    fn record_post_commit_warnings(&mut self, sequence: u64, trial_id: u64, warnings: &[String]) {
        if let Some(receipt) = self.completion_receipts.get_mut(&trial_id) {
            if receipt.commit_sequence == sequence {
                receipt.post_commit_warnings = warnings.to_vec();
            }
        }
    }

    fn rescore_completion_receipts(&mut self, objectives: &[ObjectiveConfig]) {
        let view_count = self.leaderboard.completed_count();
        let current: BTreeMap<u64, CompletedTrial> = self
            .leaderboard
            .completed_trials("rank", true, objectives)
            .into_iter()
            .map(|trial| (trial.trial_id, trial))
            .collect();
        // An objective change starts a new ranking epoch. Retained receipts can
        // be rebuilt exactly from the canonical leaderboard. An evicted trial's
        // new global rank is unknowable, so invalidate that receipt instead of
        // returning a stale or fabricated rank/front after the epoch change.
        self.completion_receipts.retain(|trial_id, receipt| {
            if let Some(completed) = current.get(trial_id) {
                receipt.completed = completed.clone();
                // This view now belongs to the current objective/ranking epoch,
                // not the original commit prefix. Carry the epoch's count so
                // every rebuilt rank/front remains internally consistent.
                receipt.committed_count = view_count;
                receipt.ranking_deferred = false;
                true
            } else {
                false
            }
        });
        self.completion_receipt_order
            .retain(|trial_id| self.completion_receipts.contains_key(trial_id));
        self.deferred_completion_receipts = 0;
    }

    fn expire_leases(&mut self, now: u64) -> usize {
        let expired: Vec<u64> = self
            .lease_deadlines
            .iter()
            .filter_map(|(&trial_id, &deadline)| (deadline <= now).then_some(trial_id))
            .collect();
        for trial_id in &expired {
            self.lease_deadlines.remove(trial_id);
            if self.pending.remove(trial_id).is_some() {
                self.remove_ask_idempotency_for_trial(*trial_id);
                self.record_cancelled(*trial_id);
            }
        }
        expired.len()
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
                    "log" | "ln" => space.add_real_log(name, *min, *max),
                    "log10" => space.add_real_log10(name, *min, *max),
                    "linear" => space.add_real(name, *min, *max),
                    _ => unreachable!("real scale was validated before construction"),
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
        let max_refit_samples = strategy_cfg
            .map(|strategy| strategy.max_refit_samples)
            .unwrap_or(DEFAULT_MAX_REFIT_SAMPLES);
        let max_refit_candidates = strategy_cfg
            .map(|strategy| strategy.max_refit_candidates)
            .unwrap_or(DEFAULT_MAX_REFIT_CANDIDATES);
        let ongoing_exploration_period = strategy_cfg
            .and_then(|strategy| strategy.ongoing_exploration_period)
            .unwrap_or(DEFAULT_ONGOING_EXPLORATION_PERIOD);
        let max_components = strategy_cfg
            .and_then(|strategy| strategy.max_components)
            .unwrap_or(DEFAULT_MAX_COMPONENTS);
        let min_elite_samples = strategy_cfg
            .and_then(|strategy| strategy.min_elite_samples)
            .unwrap_or(DEFAULT_MIN_ELITE_SAMPLES);
        // Resolve an omitted seed exactly once. The concrete value is used by
        // every strategy and persisted in study_config/checkpoints so an
        // auto-seeded run can be reproduced rather than recording `None`.
        let seed = strategy_cfg
            .and_then(|s| s.seed)
            .unwrap_or_else(rand::random);
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
                    inner: DynStrategyInner::Random(RandomStrategy::new(seed)),
                },
                None,
            ),
            "sobol" => (
                DynStrategy {
                    inner: DynStrategyInner::Sobol(SobolStrategy::new(
                        // Fold high bits into low instead of truncating so seeds
                        // differing only in bits >= 32 produce distinct sequences.
                        (seed ^ (seed >> 32)) as u32,
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
                // Anchor the cadence at the first statistically permitted fit.
                // With the legacy floor of one this is exactly the historical
                // K0 schedule; an explicit larger floor waits until that many
                // completed observations exist and then refits at the requested
                // interval from that boundary.
                let first_refit_trials = exploration_budget.max(min_elite_samples);
                effective_exploration_budget = Some(exploration_budget);
                effective_elite_fraction = Some(elite_fraction);
                let mut auto = AutoStrategy::new_with_exploration_period(
                    dim,
                    exploration_budget,
                    ongoing_exploration_period,
                    Some(seed),
                );
                let default_gmm_refit = GmmRefitConfig::default();
                auto.gmm.set_refit_config(
                    GmmRefitConfig::new(
                        max_components,
                        default_gmm_refit.max_iters(),
                        default_gmm_refit.tolerance(),
                        default_gmm_refit.regularization(),
                    )
                    .map_err(|error| format!("Invalid GMM refit configuration: {error}"))?,
                );
                (
                    DynStrategy {
                        inner: DynStrategyInner::Auto(auto),
                    },
                    Some(
                        RefitConfig::try_with_quantile(
                            first_refit_trials,
                            refit_interval,
                            elite_fraction,
                        )
                        .map_err(|error| {
                            format!("Invalid strategy refit configuration: {error}")
                        })?,
                    ),
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
            ongoing_exploration_period: Some(ongoing_exploration_period),
            seed: Some(seed),
            elite_fraction: effective_elite_fraction,
            max_components: Some(max_components),
            min_elite_samples: Some(min_elite_samples),
            max_refit_samples,
            max_refit_candidates,
        });

        let auto_checkpoint = if let Some(c) = config.checkpoint.as_ref() {
            std::fs::create_dir_all(&c.directory).map_err(|error| {
                format!(
                    "failed to create checkpoint directory '{}': {error}",
                    c.directory
                )
            })?;
            let mut ac = AutoCheckpointConfig::new(&c.directory, c.interval)?;
            ac.max_checkpoints = c.max_checkpoints;
            Some(ac)
        } else {
            None
        };

        let mut leaderboard = DynLeaderboard::for_objectives(&config.objectives);
        // Opt-in bounded mode. None (default) keeps the leaderboard unbounded
        // and retains every completed trial.
        leaderboard.set_max_size(config.max_leaderboard_size);

        Ok(Self {
            space,
            state: Arc::new(RwLock::new(HolaEngineState {
                strategy,
                strategy_template,
                leaderboard,
                objectives: config.objectives,
                next_pending_id: 0,
                pending: BTreeMap::new(),
                cancelled: HashSet::new(),
                ask_idempotency: BTreeMap::new(),
                lease_deadlines: BTreeMap::new(),
                completion_receipts: BTreeMap::new(),
                completion_receipt_order: VecDeque::new(),
                deferred_completion_receipts: 0,
            })),
            refit_lock: Arc::new(Mutex::new(())),
            refit_config,
            max_refit_samples,
            max_refit_candidates,
            min_elite_samples,
            initial_fit_attempted_completed: Arc::new(AtomicUsize::new(0)),
            auto_checkpoint,
            checkpoint_failures: Arc::new(AtomicU64::new(0)),
            refit_failures: Arc::new(AtomicU64::new(0)),
            #[cfg(test)]
            force_refit_failure: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(test)]
            refit_attempts: Arc::new(AtomicU64::new(0)),
            max_trials,
            max_leaderboard_size: config.max_leaderboard_size,
        })
    }

    /// Ask for the next trial to evaluate.
    ///
    /// Returns an error if `max_trials` has been reached.
    pub async fn ask(&self) -> Result<DynTrial, String> {
        let mut state = self.state.write().await;
        state.expire_leases(unix_time_millis());
        self.ask_locked(&mut state)
    }

    /// Request a trial with a server-managed lease.
    pub async fn ask_with_lease(&self, lease: Duration) -> Result<DynTrial, String> {
        let deadline = lease_deadline(lease)?;
        let mut state = self.state.write().await;
        state.expire_leases(unix_time_millis());
        let trial = self.ask_locked(&mut state)?;
        state.lease_deadlines.insert(trial.trial_id, deadline);
        Ok(trial)
    }

    /// Request a trial with retry-safe allocation semantics.
    ///
    /// Repeating the same key while its trial remains pending returns the exact
    /// same ID and parameters. The bounded key map is included in full
    /// checkpoints, so retries remain deterministic after a checkpointed
    /// restart. Keys are removed when their trial completes or is cancelled.
    pub async fn ask_idempotent(&self, key: &str) -> Result<DynTrial, String> {
        if key.is_empty() || key.len() > 128 || !key.is_ascii() {
            return Err("idempotency key must contain 1 to 128 ASCII characters".to_string());
        }
        let mut state = self.state.write().await;
        state.expire_leases(unix_time_millis());
        if let Some(trial) = state.ask_idempotency.get(key) {
            return Ok(trial.clone());
        }
        let trial = self.ask_locked(&mut state)?;
        state.record_ask_idempotency(key.to_string(), trial.clone());
        Ok(trial)
    }

    /// Retry-safe ask with a renewable server-managed lease.
    pub async fn ask_idempotent_with_lease(
        &self,
        key: &str,
        lease: Duration,
    ) -> Result<DynTrial, String> {
        if key.is_empty() || key.len() > 128 || !key.is_ascii() {
            return Err("idempotency key must contain 1 to 128 ASCII characters".to_string());
        }
        let deadline = lease_deadline(lease)?;
        let mut state = self.state.write().await;
        state.expire_leases(unix_time_millis());
        if let Some(trial) = state.ask_idempotency.get(key).cloned() {
            state.lease_deadlines.insert(trial.trial_id, deadline);
            return Ok(trial);
        }
        let trial = self.ask_locked(&mut state)?;
        state.record_ask_idempotency(key.to_string(), trial.clone());
        state.lease_deadlines.insert(trial.trial_id, deadline);
        Ok(trial)
    }

    /// Renew a pending trial lease and return its absolute Unix deadline.
    pub async fn heartbeat(&self, trial_id: u64, lease: Duration) -> Result<u64, String> {
        let deadline = lease_deadline(lease)?;
        let mut state = self.state.write().await;
        state.expire_leases(unix_time_millis());
        if !state.pending.contains_key(&trial_id) {
            return Err(format!(
                "Trial {trial_id} is not pending or its lease expired"
            ));
        }
        state.lease_deadlines.insert(trial_id, deadline);
        Ok(deadline)
    }

    /// Lazily reclaim expired distributed jobs. Returns the number reclaimed.
    pub async fn expire_pending_leases(&self) -> usize {
        self.state.write().await.expire_leases(unix_time_millis())
    }

    fn ask_locked(&self, state: &mut HolaEngineState) -> Result<DynTrial, String> {
        if state.pending.len() >= MAX_PENDING_TRIALS {
            return Err(format!(
                "maximum pending trial limit ({MAX_PENDING_TRIALS}) reached; complete or cancel existing trials"
            ));
        }
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
            let completed = state.leaderboard.completed_count();
            let total = completed.saturating_add(state.pending.len());
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
        self.tell_with_outcome(trial_id, raw_metrics)
            .await
            .map(|outcome| outcome.completed)
    }

    /// Tell the engine and return the completed trial plus the completed-count
    /// captured by the same commit. Servers use this to avoid reporting a later
    /// concurrent count as metadata for this operation.
    pub async fn tell_with_count(
        &self,
        trial_id: u64,
        raw_metrics: serde_json::Value,
    ) -> Result<(CompletedTrial, usize), String> {
        self.tell_with_outcome(trial_id, raw_metrics)
            .await
            .map(|outcome| (outcome.completed, outcome.trial_count))
    }

    /// Tell the engine and report whether this call committed new state or
    /// replayed a retained completion receipt.
    pub async fn tell_with_outcome(
        &self,
        trial_id: u64,
        raw_metrics: serde_json::Value,
    ) -> Result<TellOutcome, String> {
        self.tell_with_outcome_on_commit(trial_id, raw_metrics, |_, _| {})
            .await
    }

    /// Tell the engine and synchronously notify a caller at the exact commit
    /// boundary. The callback runs after the completion receipt is durable in
    /// engine state and before any post-commit async maintenance can yield.
    ///
    /// The HTTP server uses this boundary to publish its SSE event without a
    /// cancellation window between a successful commit and observability.
    pub(crate) async fn tell_with_outcome_on_commit<F>(
        &self,
        trial_id: u64,
        raw_metrics: serde_json::Value,
        on_commit: F,
    ) -> Result<TellOutcome, String>
    where
        F: FnOnce(&CompletedTrial, usize),
    {
        self.tell_with_outcome_mode(trial_id, raw_metrics, false, on_commit)
            .await
    }

    /// Commit a result for a local batch runner without eagerly constructing a
    /// ranked response that the runner will discard.
    ///
    /// This is intentionally hidden from the ordinary Rust API. Call
    /// [`Self::finalize_deferred_rankings`] once the batch ends. Public
    /// `tell()` calls remain eagerly ranked, and automatically materialize any
    /// overlapping deferred receipts before returning one.
    #[doc(hidden)]
    pub async fn tell_without_ranking(
        &self,
        trial_id: u64,
        raw_metrics: serde_json::Value,
    ) -> Result<(), String> {
        self.tell_with_outcome_mode(trial_id, raw_metrics, true, |_, _| {})
            .await
            .map(|_| ())
    }

    /// Materialize private batch receipts using one canonical leaderboard
    /// ranking snapshot. This is a no-op when no batch completion is pending.
    #[doc(hidden)]
    pub async fn finalize_deferred_rankings(&self) -> Result<(), String> {
        self.state
            .write()
            .await
            .finalize_deferred_completion_receipts()
    }

    async fn tell_with_outcome_mode<F>(
        &self,
        trial_id: u64,
        raw_metrics: serde_json::Value,
        mut defer_ranking: bool,
        on_commit: F,
    ) -> Result<TellOutcome, String>
    where
        F: FnOnce(&CompletedTrial, usize),
    {
        let mut state = self.state.write().await;
        state.expire_leases(unix_time_millis());

        if state.cancelled.contains(&trial_id) {
            return Err(format!("Trial {trial_id} has been cancelled"));
        }

        if let Some(receipt) = state.completion_receipt(trial_id) {
            if receipt.completed.metrics != raw_metrics {
                return Err(format!(
                    "Trial {trial_id} has already been completed with different metrics"
                ));
            }
            if !defer_ranking && receipt.ranking_deferred {
                state.finalize_deferred_completion_receipts()?;
            }
            let receipt = state
                .completion_receipt(trial_id)
                .expect("finalizing receipts must preserve the requested receipt");
            return Ok(TellOutcome {
                completed: receipt.completed.clone(),
                trial_count: receipt.committed_count,
                newly_committed: false,
                post_commit_warnings: receipt.post_commit_warnings.clone(),
            });
        }

        if state.leaderboard.contains_trial_id(trial_id) {
            if state.leaderboard.raw_metrics(trial_id) != Some(&raw_metrics) {
                return Err(format!(
                    "Trial {trial_id} has already been completed with different metrics"
                ));
            }
            let objectives = state.objectives.clone();
            let committed_count = state.leaderboard.completed_count();
            let (mut completed, vector_rank_inputs) = state
                .leaderboard
                .completed_for_tell(trial_id, true, &objectives)
                .ok_or_else(|| format!("Failed to rebuild CompletedTrial for {trial_id}"))?;
            drop(state);
            if let Some((participants, target)) = vector_rank_inputs {
                let (rank, front) = vector_dashboard_rank(&participants, target)
                    .ok_or_else(|| format!("Failed to rank CompletedTrial for {trial_id}"))?;
                completed.rank = rank;
                completed.pareto_front = front;
            }
            return Ok(TellOutcome {
                completed,
                trial_count: committed_count,
                newly_committed: false,
                post_commit_warnings: Vec::new(),
            });
        }

        // A bounded board must never evict the backing trial for a deferred
        // receipt. Stop deferring at the capacity boundary and materialize the
        // prior batch before the push can evict its oldest member.
        if !state.pending.contains_key(&trial_id) {
            return Err(format!("Unknown trial_id: {trial_id}"));
        }
        let would_evict = state
            .leaderboard
            .max_size()
            .is_some_and(|cap| state.leaderboard.len() >= cap);
        if !defer_ranking || would_evict {
            state.finalize_deferred_completion_receipts()?;
        }
        if would_evict {
            defer_ranking = false;
        }

        let candidate = state
            .pending
            .remove(&trial_id)
            .expect("pending membership was checked above");
        state.remove_ask_idempotency_for_trial(trial_id);
        state.lease_deadlines.remove(&trial_id);

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

        let completed_trials = state.leaderboard.completed_count();
        let commit_sequence = state.leaderboard.total_completed();

        // Public tells build their exact response before releasing the commit
        // lock. The local batch path stores only the completion payload; its
        // private placeholder rank/front is materialized before any public
        // replay and once at normal batch exit.
        let completed = if defer_ranking {
            state
                .leaderboard
                .completed_without_ranking(stored_trial_id, &objectives)
                .ok_or_else(|| format!("Failed to build CompletedTrial for {stored_trial_id}"))?
        } else {
            let (mut completed, vector_rank_inputs) = state
                .leaderboard
                .completed_for_tell(stored_trial_id, true, &objectives)
                .ok_or_else(|| format!("Failed to build CompletedTrial for {stored_trial_id}"))?;
            if let Some((participants, target)) = vector_rank_inputs {
                let (rank, front) =
                    vector_dashboard_rank(&participants, target).ok_or_else(|| {
                        format!("Failed to rank CompletedTrial for {stored_trial_id}")
                    })?;
                completed.rank = rank;
                completed.pareto_front = front;
            }
            completed
        };
        state.record_completion_receipt(
            commit_sequence,
            completed.clone(),
            completed_trials,
            defer_ranking,
        );
        drop(state);

        // There is deliberately no `.await` between recording the receipt and
        // this hook. Cancellation therefore observes either neither operation
        // or both the commit and its externally visible event.
        if !defer_ranking {
            on_commit(&completed, completed_trials);
        }

        // Own post-commit maintenance in a spawned task. Awaiting it preserves
        // the synchronous API's warnings on the normal path, while dropping or
        // timing out this tell future detaches the task instead of cancelling a
        // scheduled refit/checkpoint after the trial has committed.
        let post_commit_warnings = if self.refit_config.is_none() && self.auto_checkpoint.is_none()
        {
            Vec::new()
        } else {
            let maintenance_engine = self.clone();
            let maintenance_trial_id = completed.trial_id;
            let maintenance_task = tokio::spawn(async move {
                maintenance_engine
                    .run_post_commit_maintenance(
                        completed_trials,
                        commit_sequence,
                        maintenance_trial_id,
                    )
                    .await
            });
            match maintenance_task.await {
                Ok(warnings) => warnings,
                Err(error) => {
                    let warning = format!("post-commit maintenance task failed: {error}");
                    eprintln!("[hola] Warning: {warning}");
                    self.state.write().await.record_post_commit_warnings(
                        commit_sequence,
                        completed.trial_id,
                        std::slice::from_ref(&warning),
                    );
                    vec![warning]
                }
            }
        };

        Ok(TellOutcome {
            completed,
            trial_count: completed_trials,
            newly_committed: true,
            post_commit_warnings,
        })
    }

    async fn run_post_commit_maintenance(
        &self,
        completed_trials: usize,
        commit_sequence: u64,
        trial_id: u64,
    ) -> Vec<String> {
        let mut post_commit_warnings = Vec::new();

        if let Some(ref config) = self.refit_config {
            let scheduled_refit =
                completed_trials >= self.min_elite_samples && config.should_refit(completed_trials);
            let pending_initial_fit_epoch = if completed_trials >= config.min_trials()
                && completed_trials >= self.min_elite_samples
            {
                self.state.read().await.strategy.pending_initial_fit_epoch()
            } else {
                None
            };

            if scheduled_refit || pending_initial_fit_epoch.is_some() {
                // Serialize refits and take the leaderboard snapshot only after
                // earlier work finishes. A cadence boundary that arrives while
                // fitting must coalesce into a fit of the latest history rather
                // than being silently dropped. Initial-fit retries additionally
                // compare the model epoch and latest attempted history after
                // acquiring the lock, so queued retries do not repeat either a
                // successful fit or an unsuccessful attempt on identical data.
                let _refit_guard = self.refit_lock.lock().await;
                let state_guard = self.state.read().await;
                let refit_completed = state_guard.leaderboard.completed_count();
                let current_initial_fit_epoch = state_guard.strategy.pending_initial_fit_epoch();
                let refit_is_eligible = refit_completed >= config.min_trials()
                    && refit_completed >= self.min_elite_samples;
                let initial_fit_has_new_history =
                    refit_completed > self.initial_fit_attempted_completed.load(Ordering::Relaxed);
                let should_attempt = should_attempt_post_commit_refit(
                    scheduled_refit,
                    pending_initial_fit_epoch,
                    current_initial_fit_epoch,
                    refit_is_eligible,
                    initial_fit_has_new_history,
                );
                if !should_attempt {
                    drop(state_guard);
                    drop(_refit_guard);
                } else {
                    if current_initial_fit_epoch.is_some() {
                        self.initial_fit_attempted_completed
                            .store(refit_completed, Ordering::Relaxed);
                    }
                    #[cfg(test)]
                    self.refit_attempts.fetch_add(1, Ordering::Relaxed);
                    let k = config
                        .selection_count(refit_completed)
                        .max(self.min_elite_samples)
                        .min(refit_completed)
                        .min(self.max_refit_samples)
                        .min(self.max_refit_candidates);
                    let refit_objectives = state_guard.objectives.clone();
                    let mut trials = state_guard.leaderboard.top_k_for_refit(
                        k,
                        self.max_refit_candidates,
                        &refit_objectives,
                    );
                    // Infeasible trials are deliberately excluded from model
                    // fitting. If that leaves fewer than the requested adequacy
                    // floor, turn this cadence point into a no-op and keep the
                    // current model (or the pre-fit Sobol route) until a later one.
                    if trials.len() < self.min_elite_samples {
                        trials.clear();
                    }
                    let mut strategy_snapshot = state_guard.strategy.clone();
                    let space_clone = self.space.clone();
                    drop(state_guard);

                    let force_refit_failure = {
                        #[cfg(test)]
                        {
                            self.force_refit_failure.swap(false, Ordering::SeqCst)
                        }
                        #[cfg(not(test))]
                        {
                            false
                        }
                    };

                    let fitted_result = tokio::task::spawn_blocking(move || {
                        if force_refit_failure {
                            Err("forced refit failure for observability test".to_string())
                        } else {
                            strategy_snapshot
                                .try_refit(&space_clone, &trials)
                                .map(|()| strategy_snapshot)
                        }
                    })
                    .await;

                    match fitted_result {
                        Ok(Ok(mut fitted)) => {
                            use opt_engine::traits::RefittableStrategy;
                            let mut guard = self.state.write().await;
                            fitted.reconcile_after_refit(&guard.strategy);
                            guard.strategy = fitted;
                        }
                        Ok(Err(error)) => {
                            self.refit_failures.fetch_add(1, Ordering::Relaxed);
                            let warning = format!("post-commit refit failed: {error}");
                            eprintln!("[hola] Warning: {warning}");
                            post_commit_warnings.push(warning);
                        }
                        Err(error) => {
                            self.refit_failures.fetch_add(1, Ordering::Relaxed);
                            let warning = format!("post-commit refit task failed: {error}");
                            eprintln!("[hola] Warning: {warning}");
                            post_commit_warnings.push(warning);
                        }
                    }
                }
            }
        }

        if let Some(ref config) = self.auto_checkpoint {
            if config.should_checkpoint(completed_trials) {
                let mut snapshot = self.checkpoint_snapshot(None).await;
                let snapshot_completed = snapshot.total_completed;
                snapshot.description = Some(format!(
                    "Auto-checkpoint after {snapshot_completed} completed trials ({} retained)",
                    snapshot.n_trials
                ));
                let path = config.filename(snapshot_completed);
                if let Err(error) = Self::persist_checkpoint_snapshot(snapshot, path).await {
                    self.checkpoint_failures.fetch_add(1, Ordering::Relaxed);
                    let warning = format!("post-commit auto-checkpoint failed: {error}");
                    eprintln!("[hola] Warning: {warning}");
                    post_commit_warnings.push(warning);
                } else {
                    eprintln!(
                        "[hola] Auto-checkpoint saved after {snapshot_completed} completed trials"
                    );
                    if let Some(max) = config.max_checkpoints {
                        let directory = config.directory.clone();
                        let prefix = config.prefix.clone();
                        match tokio::task::spawn_blocking(move || {
                            Self::rotate_checkpoints(&directory, &prefix, max)
                        })
                        .await
                        {
                            Ok(failures) if failures > 0 => {
                                self.checkpoint_failures
                                    .fetch_add(failures as u64, Ordering::Relaxed);
                                post_commit_warnings.push(format!(
                                    "post-commit checkpoint rotation failed for {failures} file(s)"
                                ));
                            }
                            Ok(_) => {}
                            Err(error) => {
                                self.checkpoint_failures.fetch_add(1, Ordering::Relaxed);
                                let warning =
                                    format!("post-commit checkpoint rotation task failed: {error}");
                                eprintln!("[hola] Warning: {warning}");
                                post_commit_warnings.push(warning);
                            }
                        }
                    }
                }
            }
        }

        if !post_commit_warnings.is_empty() {
            self.state.write().await.record_post_commit_warnings(
                commit_sequence,
                trial_id,
                &post_commit_warnings,
            );
        }
        post_commit_warnings
    }

    /// Cancel a pending trial.
    pub async fn cancel(&self, trial_id: u64) -> Result<(), String> {
        let mut state = self.state.write().await;
        state.expire_leases(unix_time_millis());
        if state.pending.remove(&trial_id).is_some() {
            state.remove_ask_idempotency_for_trial(trial_id);
            state.lease_deadlines.remove(&trial_id);
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
        self.state.read().await.leaderboard.completed_count()
    }

    /// Get the number of completed trials currently retained in memory.
    pub async fn retained_trial_count(&self) -> usize {
        self.state.read().await.leaderboard.len()
    }

    /// Number of unattended auto-checkpoint or rotation failures observed by
    /// this engine process.
    pub fn checkpoint_failure_count(&self) -> u64 {
        self.checkpoint_failures.load(Ordering::Relaxed)
    }

    /// Number of failed unattended strategy refits observed by this process.
    pub fn refit_failure_count(&self) -> u64 {
        self.refit_failures.load(Ordering::Relaxed)
    }

    /// Number of issued trials still awaiting a result or cancellation.
    pub async fn pending_count(&self) -> usize {
        let mut state = self.state.write().await;
        state.expire_leases(unix_time_millis());
        state.pending.len()
    }

    /// Return the current lifecycle of one distributed trial atomically.
    ///
    /// Completion receipts make this a stronger completion oracle than the
    /// ranked single-trial view: a bounded leaderboard may evict a completed
    /// trial while its exact retry receipt is still retained. Expired,
    /// cancelled, and unknown trials intentionally share `NotPending`, because
    /// none should be cancelled again.
    #[cfg(feature = "server")]
    pub(crate) async fn trial_lifecycle(&self, trial_id: u64) -> TrialLifecycle {
        let mut state = self.state.write().await;
        state.expire_leases(unix_time_millis());
        if state.completion_receipt(trial_id).is_some()
            || state.leaderboard.contains_trial_id(trial_id)
        {
            TrialLifecycle::Completed
        } else if state.pending.contains_key(&trial_id) {
            TrialLifecycle::Pending
        } else {
            TrialLifecycle::NotPending
        }
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
        let state = self.state.read().await;
        self.study_config_from_state(&state)
    }

    fn study_config_from_state(&self, state: &HolaEngineState) -> StudyConfig {
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
        let strategy = state.strategy_template.clone().map(|mut tmpl| {
            if let Some(budget) = state.strategy.exploration_budget() {
                tmpl.exploration_budget = Some(budget);
            }
            tmpl
        });
        let objectives = state.objectives.clone();
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
        state.rescore_completion_receipts(&objectives);
    }

    /// Update objectives and re-scalarize (for mid-run dashboard adjustments).
    ///
    /// Persists the new objectives so that subsequent `tell()` calls use the
    /// updated scalarization. If a refittable strategy (e.g., GMM) is configured,
    /// a refit is triggered immediately so the sampling distribution reflects
    /// the new objective weights.
    pub async fn update_objectives(&self, objectives: Vec<ObjectiveConfig>) -> Result<(), String> {
        self.update_objectives_on_commit(objectives, |_, _| {})
            .await
    }

    /// Update objectives and synchronously notify at the committed ranking
    /// epoch before any asynchronous refit work can yield.
    pub(crate) async fn update_objectives_on_commit<F>(
        &self,
        objectives: Vec<ObjectiveConfig>,
        on_commit: F,
    ) -> Result<(), String>
    where
        F: FnOnce(usize, usize),
    {
        validate_objectives(&objectives)?;

        // Serialize the objective transition with every periodic refit before
        // changing state. Otherwise an old-objective fit could finish and
        // commit after the leaderboard had already been re-scalarized.
        let _refit_guard = if self.refit_config.is_some() {
            Some(self.refit_lock.lock().await)
        } else {
            None
        };

        // Swap objectives and migrate the leaderboard atomically under one write
        // lock so no concurrent tell() observes a half-updated state (new
        // objectives but an un-migrated leaderboard, or vice versa).
        let (completed_trials, retained_trials) = {
            let mut state = self.state.write().await;
            state.objectives = objectives.clone();
            state.leaderboard.migrate_for_objectives(&objectives);
            state.rescore_completion_receipts(&objectives);
            (state.leaderboard.completed_count(), state.leaderboard.len())
        };

        // As with tell publication, keep the external ranking-epoch signal in
        // the same cancellation-free synchronous boundary as the state commit.
        on_commit(completed_trials, retained_trials);
        drop(_refit_guard);

        // Shield the committed transition's refit from caller cancellation.
        // The task re-reads the latest objectives after acquiring the refit
        // lock, so queued objective changes coalesce safely to the newest epoch.
        let refit_engine = self.clone();
        let refit_task = tokio::spawn(async move {
            refit_engine.run_objective_update_refit().await;
        });
        if let Err(error) = refit_task.await {
            self.refit_failures.fetch_add(1, Ordering::Relaxed);
            eprintln!("[hola] Warning: objective-update maintenance task failed: {error}");
        }
        Ok(())
    }

    async fn run_objective_update_refit(&self) {
        let Some(config) = &self.refit_config else {
            return;
        };
        let _refit_guard = self.refit_lock.lock().await;
        let state_guard = self.state.read().await;
        let current_completed = state_guard.leaderboard.completed_count();
        if current_completed < config.min_trials() || current_completed < self.min_elite_samples {
            return;
        }
        let k = config
            .selection_count(current_completed)
            .max(self.min_elite_samples)
            .min(current_completed)
            .min(self.max_refit_samples)
            .min(self.max_refit_candidates);
        let objectives = state_guard.objectives.clone();
        let mut trials =
            state_guard
                .leaderboard
                .top_k_for_refit(k, self.max_refit_candidates, &objectives);
        if trials.len() < self.min_elite_samples {
            trials.clear();
        }
        let mut strategy_snapshot = state_guard.strategy.clone();
        let space_clone = self.space.clone();
        if state_guard.strategy.pending_initial_fit_epoch().is_some() {
            self.initial_fit_attempted_completed
                .store(current_completed, Ordering::Relaxed);
        }
        #[cfg(test)]
        self.refit_attempts.fetch_add(1, Ordering::Relaxed);
        drop(state_guard);

        let fitted_result = tokio::task::spawn_blocking(move || {
            strategy_snapshot
                .try_refit(&space_clone, &trials)
                .map(|()| strategy_snapshot)
        })
        .await;
        match fitted_result {
            Ok(Ok(mut fitted)) => {
                use opt_engine::traits::RefittableStrategy;
                let mut guard = self.state.write().await;
                fitted.reconcile_after_refit(&guard.strategy);
                guard.strategy = fitted;
            }
            Ok(Err(error)) => {
                self.refit_failures.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "[hola] Warning: objectives were updated but strategy refit failed: {error}"
                );
            }
            Err(error) => {
                self.refit_failures.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "[hola] Warning: objectives were updated but strategy refit task failed: {error}"
                );
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

    /// Load a checkpoint, preferring full checkpoints and falling back to
    /// legacy leaderboard-only files.
    ///
    /// This is used by CLI config `checkpoint.load_from`, which historically
    /// accepted leaderboard-only checkpoints. Full checkpoints preserve search
    /// strategy and runtime state. Leaderboard-only checkpoints preserve
    /// completed trials, invalidate unknown outstanding work, begin a fresh ID
    /// epoch, and reconcile the configured strategy with imported history.
    pub async fn load_checkpoint_with_fallback(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<CheckpointLoadKind> {
        let path = path.as_ref().to_path_buf();
        let raw = tokio::task::spawn_blocking(move || read_checkpoint_document(&path))
            .await
            .map_err(|error| std::io::Error::other(format!("checkpoint task failed: {error}")))??;

        let has_strategy_state = raw
            .get("checkpoint")
            .unwrap_or(&raw)
            .get("strategy_state")
            .is_some();
        if has_strategy_state {
            self.load_full_checkpoint_document(raw).await?;
            Ok(CheckpointLoadKind::Full)
        } else {
            self.load_leaderboard_checkpoint_document(raw).await?;
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
        let leaderboard = match &state.leaderboard {
            DynLeaderboard::Scalar(lb) => LeaderboardSnapshot::Scalar(lb.clone()),
            DynLeaderboard::Vector(lb) => LeaderboardSnapshot::Vector(lb.clone()),
        };
        drop(state);
        let path = path.as_ref().to_path_buf();
        let description = description.map(str::to_owned);
        tokio::task::spawn_blocking(move || match leaderboard {
            LeaderboardSnapshot::Scalar(lb) => {
                let mut cp = LeaderboardCheckpoint::new(lb, description.as_deref());
                cp.observation_kind = kind;
                cp.save_json(path)
            }
            LeaderboardSnapshot::Vector(lb) => {
                let mut cp = LeaderboardCheckpoint::new(lb, description.as_deref());
                cp.observation_kind = kind;
                cp.save_json(path)
            }
        })
        .await
        .map_err(|error| std::io::Error::other(format!("checkpoint task failed: {error}")))?
    }

    /// Load a leaderboard-only checkpoint, replacing the current trial history.
    ///
    /// The original runtime and sampler state are unavailable, so HOLA
    /// invalidates outstanding jobs, begins a fresh trial-ID epoch, advances
    /// sampling counters, and refits GMM state from the retained history.
    pub async fn load_leaderboard_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let path = path.as_ref().to_path_buf();
        let raw = tokio::task::spawn_blocking(move || read_checkpoint_document(&path))
            .await
            .map_err(|error| std::io::Error::other(format!("checkpoint task failed: {error}")))??;
        self.load_leaderboard_checkpoint_document(raw).await
    }

    async fn load_leaderboard_checkpoint_document(
        &self,
        raw: serde_json::Value,
    ) -> std::io::Result<()> {
        // Snapshot only the small pieces needed to prepare the replacement.
        // Parsing, O(N) validation, and GMM fitting all run off the async worker
        // pool and without holding the engine state lock.
        let (objectives, strategy, config_snapshot) = {
            let state = self.state.read().await;
            (
                state.objectives.clone(),
                state.strategy.clone(),
                self.study_config_from_state(&state),
            )
        };
        let config_snapshot_json = serde_json::to_value(&config_snapshot)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let current_is_vector = count_priority_groups(&objectives) > 1;
        let space = self.space.clone();
        let max_leaderboard_size = self.max_leaderboard_size;
        let max_refit_samples = self.max_refit_samples;
        let max_refit_candidates = self.max_refit_candidates;
        let min_elite_samples = self.min_elite_samples;
        let (leaderboard, strategy, n) = tokio::task::spawn_blocking(move || {
            let mut leaderboard = parse_leaderboard_checkpoint(raw, current_is_vector)?;
            leaderboard
                .validate_for_study(&space, &objectives)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

            let n = leaderboard.len();
            leaderboard.set_max_size(max_leaderboard_size);
            let completed_count = leaderboard.completed_count();
            let mut trials = leaderboard.top_k_for_refit(
                completed_count.min(max_refit_samples),
                max_refit_candidates,
                &objectives,
            );
            if trials.len() < min_elite_samples {
                trials.clear();
            }
            let mut strategy = strategy;
            strategy
                .reconcile_imported_history(&space, completed_count, &trials)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
            Ok::<_, std::io::Error>((leaderboard, strategy, n))
        })
        .await
        .map_err(|error| std::io::Error::other(format!("checkpoint task failed: {error}")))??;

        // The final swap is short and serialized with strategy installation. If
        // a concurrent objective update or another load changed the study
        // configuration while preparation ran, fail without mutating state.
        let _refit_guard = self.refit_lock.lock().await;
        let mut state = self.state.write().await;
        let live_config_json = serde_json::to_value(self.study_config_from_state(&state))
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        if live_config_json != config_snapshot_json {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "study configuration changed while preparing checkpoint load; retry the load",
            ));
        }
        state.strategy = strategy;
        state.leaderboard = leaderboard;
        state.reset_transient_trial_state_after_load();
        self.initial_fit_attempted_completed
            .store(state.leaderboard.completed_count(), Ordering::Relaxed);
        state.next_pending_id = state.next_pending_id.max(fresh_legacy_trial_id_floor());
        eprintln!("[hola] Loaded leaderboard checkpoint with {n} trials");
        Ok(())
    }

    /// Save a full checkpoint (leaderboard + strategy state + config).
    ///
    /// The saved JSON has the format:
    /// ```json
    /// {
    ///   "config": { ...StudyConfig... },
    ///   "checkpoint": { "leaderboard": ..., "strategy_state": ..., "metadata": ... },
    ///   "runtime_state": { "next_pending_id": ..., "pending": ..., "cancelled": ... }
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
        self.save_full_checkpoint_with_metadata(path, description)
            .await
            .map(|_| ())
    }

    /// Save a full checkpoint and return metadata from the exact snapshot that
    /// was written. This avoids recounting live state after the write, when
    /// concurrent tells may already have advanced the study.
    pub async fn save_full_checkpoint_with_metadata(
        &self,
        path: impl AsRef<std::path::Path>,
        description: Option<&str>,
    ) -> std::io::Result<SavedCheckpoint> {
        let snapshot = self.checkpoint_snapshot(description).await;
        Self::persist_checkpoint_snapshot(snapshot, path.as_ref().to_path_buf()).await
    }

    async fn checkpoint_snapshot(&self, description: Option<&str>) -> FullCheckpointSnapshot {
        let state = self.state.read().await;
        let leaderboard = match &state.leaderboard {
            DynLeaderboard::Scalar(lb) => LeaderboardSnapshot::Scalar(lb.clone()),
            DynLeaderboard::Vector(lb) => LeaderboardSnapshot::Vector(lb.clone()),
        };
        FullCheckpointSnapshot {
            config: self.study_config_from_state(&state),
            leaderboard,
            strategy: state.strategy.clone(),
            runtime_state: state.runtime_checkpoint_state(),
            description: description.map(str::to_owned),
            n_trials: state.leaderboard.len(),
            total_completed: state.leaderboard.completed_count(),
        }
    }

    async fn persist_checkpoint_snapshot(
        snapshot: FullCheckpointSnapshot,
        path: std::path::PathBuf,
    ) -> std::io::Result<SavedCheckpoint> {
        tokio::task::spawn_blocking(move || {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }

            let (checkpoint_json, created_at) = match snapshot.leaderboard {
                LeaderboardSnapshot::Scalar(leaderboard) => {
                    let checkpoint = Checkpoint::new(
                        leaderboard,
                        snapshot.strategy,
                        snapshot.description.as_deref(),
                    );
                    let created_at = checkpoint.metadata.created_at;
                    let value = serde_json::to_value(checkpoint)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    (value, created_at)
                }
                LeaderboardSnapshot::Vector(leaderboard) => {
                    let checkpoint = Checkpoint::new(
                        leaderboard,
                        snapshot.strategy,
                        snapshot.description.as_deref(),
                    );
                    let created_at = checkpoint.metadata.created_at;
                    let value = serde_json::to_value(checkpoint)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    (value, created_at)
                }
            };

            let wrapper = serde_json::json!({
                "config": snapshot.config,
                "checkpoint": checkpoint_json,
                "runtime_state": snapshot.runtime_state,
            });
            atomic_write_json(&path, |writer| {
                serde_json::to_writer_pretty(writer, &wrapper)
            })?;

            Ok(SavedCheckpoint {
                n_trials: snapshot.n_trials,
                created_at,
            })
        })
        .await
        .map_err(|error| std::io::Error::other(format!("checkpoint task failed: {error}")))?
    }

    /// Load a full checkpoint, restoring both leaderboard and strategy state.
    ///
    /// Handles both the new format (with `"config"` + `"checkpoint"` wrapper)
    /// and the legacy format (direct checkpoint without config).
    pub async fn load_full_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let path = path.as_ref().to_path_buf();
        let raw = tokio::task::spawn_blocking(move || read_checkpoint_document(&path))
            .await
            .map_err(|error| std::io::Error::other(format!("checkpoint task failed: {error}")))??;
        self.load_full_checkpoint_document(raw).await
    }

    async fn load_full_checkpoint_document(&self, raw: serde_json::Value) -> std::io::Result<()> {
        // Capture a small compatibility snapshot, then prepare the complete
        // replacement off-lock. The final state swap remains the single
        // linearization point without blocking asks/tells on O(N) parsing.
        let (current_study_config, current_strategy_template) = {
            let state = self.state.read().await;
            (
                self.study_config_from_state(&state),
                state.strategy_template.clone(),
            )
        };
        let current_study_config_json = serde_json::to_value(&current_study_config)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let has_embedded_config = raw.get("config").is_some();
        let mut loaded_strategy_template = None;
        if let Some(config_value) = raw.get("config") {
            let mut saved_config: StudyConfig = serde_json::from_value(config_value.clone())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            if let Some(strategy) = &mut saved_config.strategy {
                strategy.resolve_calibration_control_defaults();
            }
            loaded_strategy_template = saved_config.strategy.clone();
            let mut current_config = current_study_config.clone();
            // A loaded strategy replaces its sampler state wholesale. Its seed
            // therefore need not match the temporary target engine's seed—most
            // importantly, two engines built from the same omitted-seed config
            // resolve distinct random seeds before one loads the other. Keep
            // validating strategy kind/refit settings and every study schema
            // field, but normalize this non-compatibility field.
            if let Some(strategy) = &mut saved_config.strategy {
                strategy.seed = None;
            }
            if let Some(strategy) = &mut current_config.strategy {
                strategy.seed = None;
            }
            let saved_value = serde_json::to_value(saved_config)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let current_value = serde_json::to_value(current_config)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            if saved_value != current_value {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "checkpoint study configuration does not match the target engine",
                ));
            }
        }

        let vector = count_priority_groups(&current_study_config.objectives) > 1;
        let objectives = current_study_config.objectives.clone();
        let strategy_template = loaded_strategy_template.or(current_strategy_template);
        let space = self.space.clone();
        let max_leaderboard_size = self.max_leaderboard_size;
        let (replacement, n_loaded) = tokio::task::spawn_blocking(move || {
            let (loaded, runtime_state) = parse_full_checkpoint(raw, vector)?;
            let (leaderboard, strategy, n_loaded) = match loaded {
                LoadedFullCheckpoint::Scalar(checkpoint) => (
                    DynLeaderboard::Scalar(checkpoint.leaderboard),
                    checkpoint.strategy_state,
                    checkpoint.metadata.n_trials,
                ),
                LoadedFullCheckpoint::Vector(checkpoint) => (
                    DynLeaderboard::Vector(checkpoint.leaderboard),
                    checkpoint.strategy_state,
                    checkpoint.metadata.n_trials,
                ),
            };
            let mut strategy_template = strategy_template;
            let resolved_seed = strategy.resolved_seed();
            let strategy_config = strategy_template.as_mut().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "target engine has no resolved strategy configuration",
                )
            })?;
            // Older direct full checkpoints did not carry config metadata. Use
            // the loaded sampler's real seed so subsequent exports never claim
            // the discarded target engine's seed.
            if !has_embedded_config || strategy_config.seed.is_none() {
                strategy_config.seed = Some(resolved_seed);
            }
            let strategy_config = strategy_config.clone();
            let mut replacement = HolaEngineState {
                strategy,
                strategy_template,
                leaderboard,
                objectives,
                next_pending_id: 0,
                pending: BTreeMap::new(),
                cancelled: HashSet::new(),
                ask_idempotency: BTreeMap::new(),
                lease_deadlines: BTreeMap::new(),
                completion_receipts: BTreeMap::new(),
                completion_receipt_order: VecDeque::new(),
                deferred_completion_receipts: 0,
            };
            replacement.leaderboard.set_max_size(max_leaderboard_size);
            replacement
                .leaderboard
                .validate_for_study(&space, &replacement.objectives)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
            replacement
                .restore_runtime_checkpoint_state(runtime_state, &space)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
            let completed_count = replacement.leaderboard.completed_count();
            let minimum_issued_count = completed_count
                .saturating_add(replacement.pending.len())
                .saturating_add(replacement.cancelled.len());
            replacement
                .strategy
                .validate_for_study(
                    &strategy_config,
                    &space,
                    completed_count,
                    minimum_issued_count,
                )
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
            Ok::<_, std::io::Error>((replacement, n_loaded))
        })
        .await
        .map_err(|error| std::io::Error::other(format!("checkpoint task failed: {error}")))??;

        // Serialize only the short swap with refit installation. A concurrent
        // configuration transition makes this prepared replacement stale, so
        // reject it atomically and ask the caller to retry.
        let _refit_guard = self.refit_lock.lock().await;
        let mut state = self.state.write().await;
        let live_config_json = serde_json::to_value(self.study_config_from_state(&state))
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        if live_config_json != current_study_config_json {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "study configuration changed while preparing checkpoint load; retry the load",
            ));
        }
        *state = replacement;
        self.initial_fit_attempted_completed
            .store(state.leaderboard.completed_count(), Ordering::Relaxed);
        eprintln!("[hola] Loaded full checkpoint with {n_loaded} trials");
        Ok(())
    }

    /// Load a study from a checkpoint file, reconstructing the engine from the
    /// embedded `StudyConfig`.
    ///
    /// The checkpoint must have been saved with the new format that includes
    /// the `"config"` key. Returns an error if the config is missing (i.e.,
    /// the file was saved with an older version of HOLA).
    pub async fn load_from_checkpoint(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let raw = tokio::task::spawn_blocking(move || read_checkpoint_document(&path))
            .await
            .map_err(|error| format!("Checkpoint task failed: {error}"))?
            .map_err(|error| format!("Failed to read checkpoint file: {error}"))?;

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
            .load_full_checkpoint_document(raw)
            .await
            .map_err(|e| format!("Failed to load checkpoint data: {e}"))?;
        Ok(engine)
    }

    /// Delete oldest checkpoint files to keep at most `max` files.
    fn rotate_checkpoints(directory: &std::path::Path, prefix: &str, max: usize) -> usize {
        let pattern = format!("{prefix}_");
        // Parse the numeric snapshot count rather than relying on lexical names
        // or mutable filesystem timestamps. This remains correctly ordered when
        // the counter grows beyond the six-digit zero-padding width.
        let mut checkpoints: Vec<(usize, std::path::PathBuf)> = std::fs::read_dir(directory)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter_map(|p| {
                if p.extension().is_none_or(|ext| ext != "json") {
                    return None;
                }
                p.file_stem()
                    .and_then(|stem| stem.to_str())
                    .and_then(|stem| stem.strip_prefix(&pattern))
                    .and_then(|suffix| suffix.parse::<usize>().ok())
                    .map(|sequence| (sequence, p))
            })
            .collect();

        if checkpoints.len() <= max {
            return 0;
        }

        // Oldest first, so the leading `to_delete` entries are the ones to evict.
        checkpoints.sort_by_key(|(sequence, _)| *sequence);
        let to_delete = checkpoints.len() - max;
        let mut failures = 0;
        for (_, path) in checkpoints.into_iter().take(to_delete) {
            if let Err(e) = std::fs::remove_file(&path) {
                failures += 1;
                eprintln!("[hola] Warning: failed to delete old checkpoint {path:?}: {e}");
            }
        }
        failures
    }
}

fn read_checkpoint_document(path: &std::path::Path) -> std::io::Result<serde_json::Value> {
    let bytes = read_checkpoint_capped(path)?;
    check_format_version_bytes(&bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn parse_full_checkpoint(
    raw: serde_json::Value,
    vector: bool,
) -> std::io::Result<(LoadedFullCheckpoint, Option<RuntimeCheckpointState>)> {
    let runtime_state = raw
        .get("runtime_state")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let checkpoint_json = raw.get("checkpoint").cloned().unwrap_or(raw);
    let checkpoint = if vector {
        LoadedFullCheckpoint::Vector(
            serde_json::from_value(checkpoint_json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        )
    } else {
        LoadedFullCheckpoint::Scalar(
            serde_json::from_value(checkpoint_json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        )
    };
    Ok((checkpoint, runtime_state))
}

fn parse_leaderboard_checkpoint(
    raw: serde_json::Value,
    current_is_vector: bool,
) -> std::io::Result<DynLeaderboard> {
    let stored_kind = raw
        .get("observation_kind")
        .cloned()
        .map(serde_json::from_value::<ObservationKind>)
        .transpose()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if let Some(kind) = stored_kind {
        let stored_is_vector = matches!(kind, ObservationKind::Vector);
        if stored_is_vector != current_is_vector {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "leaderboard checkpoint observation_kind ({}) does not match the current objective topology ({})",
                    if stored_is_vector { "vector" } else { "scalar" },
                    if current_is_vector {
                        "vector"
                    } else {
                        "scalar"
                    },
                ),
            ));
        }
    }

    if current_is_vector {
        let checkpoint: LeaderboardCheckpoint<serde_json::Value, BTreeMap<String, f64>> =
            serde_json::from_value(raw)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(DynLeaderboard::Vector(checkpoint.leaderboard))
    } else {
        let checkpoint: LeaderboardCheckpoint<serde_json::Value, f64> = serde_json::from_value(raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(DynLeaderboard::Scalar(checkpoint.leaderboard))
    }
}

/// Collapse multi-field raw metrics into a single scalar cost F(x).
///
/// Implements the paper's formula: F(x) = Σ φ_i(f_i(x)) where
/// φ_i(u) = P_i × (u − T_i)/(L_i − T_i) for the in-range case.
/// Each objective's normalized TLP score is multiplied by its priority P_i,
/// which is the resulting score at the limit and relative weight in its group.
fn scalarize_raw(raw: &serde_json::Value, objectives: &[ObjectiveConfig]) -> f64 {
    let mut total = 0.0;
    for obj in objectives {
        let val = match raw_metric_f64(raw, &obj.field) {
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
/// the group cost is C_g(x) = Σ_{i ∈ G_g} φ_i(f_i(x)), where each
/// paper-defined TLP score φ_i already includes its priority P_i. The code
/// below forms that score as P_i times the unweighted normalized distance.
fn vectorize_raw(raw: &serde_json::Value, objectives: &[ObjectiveConfig]) -> BTreeMap<String, f64> {
    let mut groups: BTreeMap<String, f64> = BTreeMap::new();
    for obj in objectives {
        let val = match raw_metric_f64(raw, &obj.field) {
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

/// Compute normalized TLP score for a single value (shared by `scalarize_raw`
/// and `vectorize_raw`).
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
    fn test_scalar_score_vector_uses_the_priority_group_name() {
        let one = vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: Some("quality".to_string()),
        }];
        assert_eq!(
            compute_score_vector(&serde_json::json!({"loss": 2.0}), &one),
            serde_json::json!({"quality": 2.0})
        );

        let shared = vec![
            one[0].clone(),
            ObjectiveConfig {
                field: "calibration".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("quality".to_string()),
            },
        ];
        assert_eq!(
            compute_score_vector(
                &serde_json::json!({"loss": 2.0, "calibration": 3.0}),
                &shared,
            ),
            serde_json::json!({"quality": 5.0})
        );
    }

    #[test]
    fn test_vector_refit_elites_use_group_cost_pareto_rank_and_crowding() {
        let objectives = vec![
            ObjectiveConfig {
                field: "error".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 2.0,
                group: Some("quality".to_string()),
            },
            ObjectiveConfig {
                field: "calibration".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 0.5,
                group: Some("quality".to_string()),
            },
            ObjectiveConfig {
                field: "latency".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("cost".to_string()),
            },
        ];
        let mut leaderboard = DynLeaderboard::for_objectives(&objectives);

        for (trial_id, candidate, metrics) in [
            (
                0,
                "quality-boundary",
                serde_json::json!({"error": 0.0, "calibration": 0.0, "latency": 10.0}),
            ),
            (
                1,
                "summed-cost-winner",
                serde_json::json!({"error": 1.0, "calibration": 4.0, "latency": 4.0}),
            ),
            (
                2,
                "cost-boundary",
                serde_json::json!({"error": 5.0, "calibration": 0.0, "latency": 0.0}),
            ),
        ] {
            leaderboard.push_with_raw(
                trial_id,
                serde_json::json!({"candidate": candidate}),
                metrics,
                &objectives,
            );
        }

        let middle = vectorize_raw(
            &serde_json::json!({"error": 1.0, "calibration": 4.0, "latency": 4.0}),
            &objectives,
        );
        assert_eq!(middle["quality"], 4.0);
        assert_eq!(middle["cost"], 4.0);

        let elites = leaderboard.top_k_for_refit(2, DEFAULT_MAX_REFIT_CANDIDATES, &objectives);
        let names: Vec<&str> = elites
            .iter()
            .map(|(candidate, _)| candidate["candidate"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["quality-boundary", "cost-boundary"]);
        assert!(
            !names.contains(&"summed-cost-winner"),
            "the summed-cost winner is an interior Pareto point and must lose the two-slot crowding tie-break"
        );
    }

    #[test]
    fn test_public_score_json_preserves_all_non_finite_values() {
        assert_eq!(f64_to_json(f64::INFINITY), serde_json::json!("inf"));
        assert_eq!(f64_to_json(f64::NEG_INFINITY), serde_json::json!("-inf"));
        assert_eq!(f64_to_json(f64::NAN), serde_json::json!("nan"));
        assert_eq!(f64_to_json(1.25), serde_json::json!(1.25));

        let observation = BTreeMap::from([
            ("finite".to_string(), 1.25),
            ("nan".to_string(), f64::NAN),
            ("negative".to_string(), f64::NEG_INFINITY),
            ("positive".to_string(), f64::INFINITY),
        ]);
        assert_eq!(
            f64_map_to_json(&observation),
            serde_json::json!({
                "finite": 1.25,
                "nan": "nan",
                "negative": "-inf",
                "positive": "inf",
            })
        );

        // Finite inputs and weights can overflow in either direction. When
        // opposite infinities share one group, their aggregate is NaN. Every
        // public scoring helper must preserve those three distinct outcomes.
        let objectives = vec![
            ObjectiveConfig {
                field: "positive".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: f64::MAX,
                group: Some("combined".to_string()),
            },
            ObjectiveConfig {
                field: "negative".to_string(),
                obj_type: "maximize".to_string(),
                target: None,
                limit: None,
                priority: f64::MAX,
                group: Some("combined".to_string()),
            },
        ];
        let raw = serde_json::json!({"positive": f64::MAX, "negative": f64::MAX});
        assert_eq!(
            compute_scores(&raw, &objectives),
            serde_json::json!({"positive": "inf", "negative": "-inf"})
        );
        assert_eq!(
            compute_score_vector(&raw, &objectives),
            serde_json::json!({"combined": "nan"})
        );

        // Strict JSON transports (including the Python binding) use string
        // sentinels for non-finite raw metrics. Decode them before objective
        // direction/scalarization instead of treating them as missing fields.
        let sentinel_objectives = vec![
            ObjectiveConfig {
                field: "min_loss".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("min".to_string()),
            },
            ObjectiveConfig {
                field: "max_reward".to_string(),
                obj_type: "maximize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("max".to_string()),
            },
            ObjectiveConfig {
                field: "unstable".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("nan".to_string()),
            },
        ];
        let sentinel_raw = serde_json::json!({
            "min_loss": "-inf",
            "max_reward": "inf",
            "unstable": "nan",
        });
        assert_eq!(
            compute_scores(&sentinel_raw, &sentinel_objectives),
            serde_json::json!({
                "min_loss": "-inf",
                "max_reward": "-inf",
                "unstable": "nan",
            })
        );
        assert_eq!(
            compute_score_vector(&sentinel_raw, &sentinel_objectives),
            serde_json::json!({"max": "-inf", "min": "-inf", "nan": "nan"})
        );
    }

    #[test]
    fn test_scalar_single_trial_rank_matches_full_total_order() {
        let objectives = vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }];
        let mut leaderboard = Leaderboard::<serde_json::Value, f64>::new();

        // Put NaN first so a partial_cmp(None) -> Equal implementation would
        // incorrectly rank it ahead of every later value by trial ID.
        for (trial_id, observation) in [
            (0, f64::NAN),
            (1, f64::INFINITY),
            (2, 2.0),
            (3, f64::NEG_INFINITY),
            (4, -1.0),
        ] {
            leaderboard.push_with_raw_trial_id(
                serde_json::json!({"x": trial_id}),
                observation,
                serde_json::json!({"loss": trial_id}),
                trial_id,
            );
        }

        let leaderboard = DynLeaderboard::Scalar(leaderboard);
        let full = leaderboard.completed_trials("rank", true, &objectives);
        assert_eq!(
            full.iter().map(|trial| trial.trial_id).collect::<Vec<_>>(),
            vec![4, 2, 3, 1, 0],
            "canonical order is finite numeric values, infinities, then NaN"
        );

        for expected in full {
            let single = leaderboard
                .get_completed(expected.trial_id, true, &objectives)
                .expect("every stored trial must have a single-trial view");
            assert_eq!(
                single.rank, expected.rank,
                "single-trial and full-board ranks diverged for trial {}",
                expected.trial_id
            );
        }
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

        // Fixed dimensions are intentional constants and map to the unit-cube
        // midpoint rather than dividing by a zero span.
        assert!(validate_space_config(&study(0.5, 0.5, "linear").space).is_ok());
    }

    #[test]
    fn test_validate_space_rejects_duplicate_categories_and_inexact_integer_range() {
        let duplicate = BTreeMap::from([(
            "optimizer".to_string(),
            ParamConfig::Categorical {
                choices: vec!["adam".into(), "sgd".into(), "adam".into()],
            },
        )]);
        let error = validate_space_config(&duplicate).unwrap_err();
        assert!(error.contains("optimizer"));
        assert!(error.contains("duplicate choice 'adam'"));

        let too_wide = BTreeMap::from([(
            "integer".to_string(),
            ParamConfig::Integer {
                min: 0,
                max: 1i64 << 52,
            },
        )]);
        let error = validate_space_config(&too_wide).unwrap_err();
        assert!(error.contains("integer"));
        assert!(error.contains("exact unit-cube mapping limit"));
    }

    #[test]
    fn test_validate_objectives_rejects_incomplete_and_duplicate_contracts() {
        let objective = |field: &str, target, limit| ObjectiveConfig {
            field: field.to_string(),
            obj_type: "minimize".to_string(),
            target,
            limit,
            priority: 1.0,
            group: None,
        };

        let incomplete = [objective("loss", Some(0.0), None)];
        let error = validate_objectives(&incomplete).unwrap_err();
        assert!(error.contains("target and limit must either both be set"));

        let duplicate = [objective("loss", None, None), objective("loss", None, None)];
        let error = validate_objectives(&duplicate).unwrap_err();
        assert!(error.contains("configured more than once"));
    }

    #[test]
    fn test_config_deserialization_rejects_unknown_fields() {
        let param = serde_json::json!({
            "type": "real",
            "min": 0.0,
            "max": 1.0,
            "typo_scale": "log10"
        });
        assert!(serde_json::from_value::<ParamConfig>(param).is_err());

        let objective = serde_json::json!({
            "field": "loss",
            "type": "minimize",
            "prioirty": 1.0
        });
        assert!(serde_json::from_value::<ObjectiveConfig>(objective).is_err());

        let strategy = serde_json::json!({
            "type": "gmm",
            "refit_interval": 10,
            "elite_fracton": 0.25
        });
        assert!(serde_json::from_value::<StrategyConfig>(strategy).is_err());

        let checkpoint = serde_json::json!({
            "directory": ".",
            "interval": 10,
            "max_checkpoint": 3
        });
        assert!(serde_json::from_value::<CheckpointConfig>(checkpoint).is_err());

        let study = serde_json::json!({
            "space": {"x": {"type": "real", "min": 0.0, "max": 1.0}},
            "objectives": [{"field": "loss", "type": "minimize"}],
            "max_trial": 10
        });
        assert!(serde_json::from_value::<StudyConfig>(study).is_err());
    }

    #[test]
    fn test_strategy_config_legacy_json_uses_refit_limit_and_control_defaults() {
        let strategy: StrategyConfig = serde_json::from_value(serde_json::json!({
            "type": "gmm",
            "refit_interval": 20
        }))
        .unwrap();
        assert_eq!(strategy.max_refit_samples, DEFAULT_MAX_REFIT_SAMPLES);
        assert_eq!(strategy.max_refit_candidates, DEFAULT_MAX_REFIT_CANDIDATES);
        assert_eq!(strategy.ongoing_exploration_period, None);
        assert_eq!(strategy.max_components, None);
        assert_eq!(strategy.min_elite_samples, None);
        validate_strategy_config(&strategy).unwrap();
    }

    #[test]
    fn test_strategy_config_rejects_invalid_calibration_controls() {
        let mut strategy: StrategyConfig = serde_json::from_value(serde_json::json!({
            "type": "gmm"
        }))
        .unwrap();
        strategy.ongoing_exploration_period = Some(1);
        assert!(
            validate_strategy_config(&strategy)
                .unwrap_err()
                .contains("ongoing_exploration_period")
        );

        strategy.ongoing_exploration_period = Some(0);
        strategy.max_components = Some(0);
        assert!(
            validate_strategy_config(&strategy)
                .unwrap_err()
                .contains("max_components")
        );

        strategy.max_components = Some(1);
        strategy.min_elite_samples = Some(0);
        assert!(
            validate_strategy_config(&strategy)
                .unwrap_err()
                .contains("min_elite_samples")
        );
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
            a.gmm
                .set_params(GmmParams::uniform_prior(1, 0.1).unwrap())
                .unwrap();
            a.trial_count = 9;
            a.issued_count.store(12, Ordering::Relaxed);
        }

        // `fitted` is the off-lock snapshot: it carries the freshly fitted GMM
        // (two components) but stale sampling counters from before the refit.
        let two_cluster_samples: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![if i < 25 { 0.2 } else { 0.8 }])
            .collect();
        let fitted_model = GmmParams::fit(&two_cluster_samples, 2, 100, 1e-6, 1e-4, 1).unwrap();
        assert_eq!(fitted_model.n_components(), 2);

        let mut fitted = live.clone();
        if let DynStrategyInner::Auto(a) = &mut fitted.inner {
            a.gmm.set_params(fitted_model).unwrap();
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

        // Exploitation suggestions advance a separate GMM seed counter. It is
        // just as important as the Sobol index: rewinding it would repeat a
        // previously issued candidate after the refit commits.
        let live_gmm_counter = if let DynStrategyInner::Auto(a) = &live.inner {
            a.gmm.suggest(&space);
            a.gmm.suggest(&space);
            serde_json::to_value(&a.gmm).unwrap()["counter"].clone()
        } else {
            panic!("expected Auto strategy");
        };
        let fitted_gmm_counter = if let DynStrategyInner::Auto(a) = &fitted.inner {
            serde_json::to_value(&a.gmm).unwrap()["counter"].clone()
        } else {
            panic!("expected Auto strategy");
        };
        assert_ne!(live_gmm_counter, fitted_gmm_counter);

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
                let reconciled_gmm_counter =
                    serde_json::to_value(&a.gmm).unwrap()["counter"].clone();
                assert_eq!(reconciled_gmm_counter, live_gmm_counter);
                // The freshly fitted GMM model is kept (two components, not the
                // single-component prior that `live` still held).
                assert_eq!(a.gmm.params().unwrap().n_components(), 2);
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
                ongoing_exploration_period: None,
                seed: None,
                elite_fraction: None,
                max_components: None,
                min_elite_samples: None,
                max_refit_samples: 4096,
                max_refit_candidates: 16_384,
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
                ongoing_exploration_period: None,
                seed: Some(7),
                elite_fraction: None,
                max_components: None,
                min_elite_samples: None,
                max_refit_samples: 4096,
                max_refit_candidates: 16_384,
            }),
            checkpoint: None,
            max_trials: None,
            max_leaderboard_size: None,
        }
    }

    #[test]
    fn auto_strategy_period_has_no_off_by_one_and_zero_disables_it() {
        use opt_engine::strategies::GmmParams;

        let space = DynSpace::new().add_real("x", 0.0, 1.0);
        let routed = |period: usize| {
            let mut auto = AutoStrategy::new_with_exploration_period(1, 2, period, Some(7));
            // Mark the model as empirically fitted so this test isolates the
            // initial/periodic routing schedule from the pre-fit Sobol guard.
            auto.gmm
                .set_params(GmmParams::uniform_prior(1, 0.1).unwrap())
                .unwrap();
            auto.gmm_sampling_ready = true;
            let strategy = DynStrategy {
                inner: DynStrategyInner::Auto(auto),
            };
            let mut routes = Vec::new();
            for _ in 0..8 {
                let (sobol_before, gmm_before) = match &strategy.inner {
                    DynStrategyInner::Auto(auto) => (auto.sobol.index(), auto.gmm.counter()),
                    _ => unreachable!(),
                };
                strategy.suggest(&space);
                let (sobol_after, gmm_after) = match &strategy.inner {
                    DynStrategyInner::Auto(auto) => (auto.sobol.index(), auto.gmm.counter()),
                    _ => unreachable!(),
                };
                routes.push(match (sobol_after - sobol_before, gmm_after - gmm_before) {
                    (1, 0) => 's',
                    (0, 1) => 'g',
                    other => panic!("one and only one sampler must advance, got {other:?}"),
                });
            }
            routes
        };

        // With period 3, the third and sixth post-warm-up requests are Sobol.
        assert_eq!(routed(3), vec!['s', 's', 'g', 'g', 's', 'g', 'g', 's']);
        assert_eq!(routed(0), vec!['s', 's', 'g', 'g', 'g', 'g', 'g', 'g']);
    }

    #[test]
    fn auto_strategy_legacy_period_is_used_when_serde_field_is_missing() {
        let auto = AutoStrategy::new(1, 4, Some(7));
        let mut encoded = serde_json::to_value(auto).unwrap();
        encoded
            .as_object_mut()
            .unwrap()
            .remove("ongoing_exploration_period");

        let restored: AutoStrategy = serde_json::from_value(encoded).unwrap();
        assert_eq!(
            restored.ongoing_exploration_period,
            DEFAULT_ONGOING_EXPLORATION_PERIOD
        );
    }

    #[tokio::test]
    async fn omitted_calibration_controls_resolve_to_legacy_values() {
        let config = single_objective_config("gmm");
        let explicit_legacy = {
            let mut config = config.clone();
            let strategy = config.strategy.as_mut().unwrap();
            strategy.ongoing_exploration_period = Some(DEFAULT_ONGOING_EXPLORATION_PERIOD);
            strategy.max_components = Some(DEFAULT_MAX_COMPONENTS);
            strategy.min_elite_samples = Some(DEFAULT_MIN_ELITE_SAMPLES);
            config
        };

        let implicit = HolaEngine::from_config(config).unwrap();
        let explicit = HolaEngine::from_config(explicit_legacy).unwrap();
        let exported = implicit.study_config().await;
        let exported = exported.strategy.unwrap();
        assert_eq!(
            exported.ongoing_exploration_period,
            Some(DEFAULT_ONGOING_EXPLORATION_PERIOD)
        );
        assert_eq!(exported.max_components, Some(DEFAULT_MAX_COMPONENTS));
        assert_eq!(exported.min_elite_samples, Some(DEFAULT_MIN_ELITE_SAMPLES));

        // The omitted and explicit legacy forms must remain behaviorally
        // identical, including the first refit and periodic exploration.
        for _ in 0..40 {
            let implicit_trial = implicit.ask().await.unwrap();
            let explicit_trial = explicit.ask().await.unwrap();
            assert_eq!(implicit_trial.params, explicit_trial.params);
            let loss = implicit_trial.params["x"].as_f64().unwrap();
            implicit
                .tell(implicit_trial.trial_id, serde_json::json!({"loss": loss}))
                .await
                .unwrap();
            explicit
                .tell(explicit_trial.trial_id, serde_json::json!({"loss": loss}))
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn min_elite_samples_delays_first_refit_and_max_components_is_applied() {
        let mut config = single_objective_config("gmm");
        let strategy = config.strategy.as_mut().unwrap();
        strategy.exploration_budget = Some(1);
        strategy.refit_interval = 1;
        strategy.min_elite_samples = Some(4);
        strategy.max_components = Some(2);
        let engine = HolaEngine::from_config(config).unwrap();

        {
            let state = engine.state.read().await;
            let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
                panic!("expected auto strategy");
            };
            assert_eq!(auto.gmm.get_refit_config().n_components(), 2);
        }

        for completed in 1..=4 {
            let trial = engine.ask().await.unwrap();
            let loss = trial.params["x"].as_f64().unwrap();
            engine
                .tell(trial.trial_id, serde_json::json!({"loss": loss}))
                .await
                .unwrap();
            let state = engine.state.read().await;
            let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
                panic!("expected auto strategy");
            };
            assert_eq!(
                auto.gmm.refit_epoch(),
                u64::from(completed == 4),
                "the first refit must wait for the requested elite floor"
            );
            assert!(auto.gmm.params().unwrap().n_components() <= 2);
        }
    }

    #[tokio::test]
    async fn min_elite_samples_counts_only_feasible_fit_inputs() {
        let mut config = single_objective_config("gmm");
        config.objectives[0].target = Some(0.0);
        config.objectives[0].limit = Some(1.0);
        let strategy = config.strategy.as_mut().unwrap();
        strategy.exploration_budget = Some(1);
        strategy.refit_interval = 1;
        strategy.min_elite_samples = Some(4);
        let engine = HolaEngine::from_config(config).unwrap();

        for completed in 1..=5 {
            let trial = engine.ask().await.unwrap();
            let loss = if completed == 1 { 2.0 } else { 0.5 };
            engine
                .tell(trial.trial_id, serde_json::json!({"loss": loss}))
                .await
                .unwrap();
            let state = engine.state.read().await;
            let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
                panic!("expected auto strategy");
            };
            assert_eq!(
                auto.gmm.refit_epoch(),
                u64::from(completed == 5),
                "an infeasible observation must not satisfy the elite floor"
            );
        }
    }

    #[tokio::test]
    async fn initial_fit_retries_before_the_next_periodic_cadence() {
        let mut config = single_objective_config("gmm");
        config.objectives[0].target = Some(0.0);
        config.objectives[0].limit = Some(1.0);
        let strategy = config.strategy.as_mut().unwrap();
        strategy.exploration_budget = Some(10);
        strategy.refit_interval = 20;
        strategy.min_elite_samples = Some(10);
        let engine = HolaEngine::from_config(config).unwrap();

        for completed in 1..=10 {
            let trial = engine.ask().await.unwrap();
            let loss = if completed == 1 { 2.0 } else { 0.5 };
            engine
                .tell(trial.trial_id, serde_json::json!({"loss": loss}))
                .await
                .unwrap();
        }
        {
            let state = engine.state.read().await;
            let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
                panic!("expected auto strategy");
            };
            assert_eq!(auto.gmm.refit_epoch(), 0);
            assert!(!auto.gmm_sampling_ready);
        }

        let trial = engine.ask().await.unwrap();
        engine
            .tell(trial.trial_id, serde_json::json!({"loss": 0.5}))
            .await
            .unwrap();
        let state = engine.state.read().await;
        let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
            panic!("expected auto strategy");
        };
        assert_eq!(auto.gmm.refit_epoch(), 1);
        assert!(auto.gmm_sampling_ready);
    }

    #[tokio::test]
    async fn successful_initial_retry_restores_periodic_cadence() {
        let mut config = single_objective_config("gmm");
        config.objectives[0].target = Some(0.0);
        config.objectives[0].limit = Some(1.0);
        let strategy = config.strategy.as_mut().unwrap();
        strategy.exploration_budget = Some(4);
        strategy.refit_interval = 3;
        strategy.min_elite_samples = Some(4);
        let engine = HolaEngine::from_config(config).unwrap();

        for completed in 1..=7 {
            let trial = engine.ask().await.unwrap();
            let loss = if completed == 1 { 2.0 } else { 0.5 };
            engine
                .tell(trial.trial_id, serde_json::json!({"loss": loss}))
                .await
                .unwrap();
            let state = engine.state.read().await;
            let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
                panic!("expected auto strategy");
            };
            let expected_epoch = match completed {
                1..=4 => 0,
                5..=6 => 1,
                7 => 2,
                _ => unreachable!(),
            };
            assert_eq!(auto.gmm.refit_epoch(), expected_epoch);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_initial_fit_retries_do_not_repeat_failed_history() {
        let mut config = single_objective_config("gmm");
        config.objectives[0].target = Some(0.0);
        config.objectives[0].limit = Some(1.0);
        let strategy = config.strategy.as_mut().unwrap();
        strategy.exploration_budget = Some(4);
        strategy.refit_interval = 20;
        strategy.min_elite_samples = Some(4);
        let engine = HolaEngine::from_config(config).unwrap();

        for completed in 1..=4 {
            let trial = engine.ask().await.unwrap();
            let loss = if completed == 1 { 2.0 } else { 0.5 };
            engine
                .tell(trial.trial_id, serde_json::json!({"loss": loss}))
                .await
                .unwrap();
        }
        assert_eq!(engine.refit_attempts.load(Ordering::Relaxed), 1);

        // Keep every new completion's maintenance task queued until all eight
        // tells have committed. The first task will see the complete shared
        // history; the others must not repeat that same attempt after it fails.
        let refit_guard = engine.refit_lock.lock().await;
        engine.force_refit_failure.store(true, Ordering::SeqCst);

        let mut trials = Vec::new();
        for _ in 0..8 {
            trials.push(engine.ask().await.unwrap());
        }
        let mut handles = Vec::new();
        for trial in trials {
            let engine = engine.clone();
            handles.push(tokio::spawn(async move {
                engine
                    .tell(trial.trial_id, serde_json::json!({"loss": 0.5}))
                    .await
                    .unwrap();
            }));
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            while engine.trial_count().await < 12 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("all tells must commit before refit maintenance is released");
        drop(refit_guard);
        for handle in handles {
            handle.await.unwrap();
        }

        {
            let state = engine.state.read().await;
            let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
                panic!("expected auto strategy");
            };
            assert_eq!(auto.gmm.refit_epoch(), 0);
            assert!(!auto.gmm_sampling_ready);
        }
        assert_eq!(engine.refit_attempts.load(Ordering::Relaxed), 2);
        assert_eq!(engine.refit_failure_count(), 1);

        // One genuinely new completion supplies a new history generation and
        // therefore gets one fresh retry, which installs the first model.
        let trial = engine.ask().await.unwrap();
        engine
            .tell(trial.trial_id, serde_json::json!({"loss": 0.5}))
            .await
            .unwrap();
        let state = engine.state.read().await;
        let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
            panic!("expected auto strategy");
        };
        assert_eq!(auto.gmm.refit_epoch(), 1);
        assert!(auto.gmm_sampling_ready);
        assert_eq!(engine.refit_attempts.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn concurrent_prefit_sobol_checkpoint_roundtrip_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prefit-pending.json");
        let mut config = single_objective_config("gmm");
        let strategy = config.strategy.as_mut().unwrap();
        strategy.exploration_budget = Some(2);
        strategy.ongoing_exploration_period = Some(3);
        let source = HolaEngine::from_config(config).unwrap();

        // No tells have arrived, so every request beyond K0 must remain on
        // Sobol rather than use the uninformed uniform GMM prior.
        for _ in 0..8 {
            source.ask().await.unwrap();
        }
        {
            let state = source.state.read().await;
            let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
                panic!("expected auto strategy");
            };
            assert_eq!(auto.sobol.index(), 8);
            assert_eq!(auto.gmm.counter(), 0);
            assert_eq!(auto.gmm.refit_epoch(), 0);
        }

        source.save_full_checkpoint(&path, None).await.unwrap();
        let resumed = HolaEngine::load_from_checkpoint(&path).await.unwrap();
        let source_next = source.ask().await.unwrap();
        let resumed_next = resumed.ask().await.unwrap();
        assert_eq!(source_next.trial_id, resumed_next.trial_id);
        assert_eq!(source_next.params, resumed_next.params);

        // The route-accounting validation must still reject a checkpoint whose
        // sampler cursors cannot account for its issued pending requests.
        let mut forged: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        // Four Sobol points meet the nominal K0+periodic lower bound for eight
        // requests (K0=2, period=3), but cannot account for all eight requests
        // when the GMM cursor is also zero.
        forged["checkpoint"]["strategy_state"]["inner"]["sobol"]["index"] = serde_json::json!(4);
        forged["checkpoint"]["strategy_state"]["inner"]["gmm"]["counter"]["value"] =
            serde_json::json!(0);
        let target = HolaEngine::from_config(source.study_config().await).unwrap();
        let error = target
            .load_full_checkpoint_document(forged)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("sampler cursors account"));
    }

    #[tokio::test]
    async fn checkpoint_missing_calibration_fields_loads_with_legacy_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy-controls.json");
        let source = HolaEngine::from_config(single_objective_config("gmm")).unwrap();
        let trial = source.ask().await.unwrap();
        source
            .tell(trial.trial_id, serde_json::json!({"loss": 0.5}))
            .await
            .unwrap();
        source.save_full_checkpoint(&path, None).await.unwrap();

        let mut legacy: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let strategy_config = legacy["config"]["strategy"].as_object_mut().unwrap();
        strategy_config.remove("ongoing_exploration_period");
        strategy_config.remove("max_components");
        strategy_config.remove("min_elite_samples");
        legacy["checkpoint"]["strategy_state"]["inner"]
            .as_object_mut()
            .unwrap()
            .remove("ongoing_exploration_period");
        std::fs::write(&path, serde_json::to_vec_pretty(&legacy).unwrap()).unwrap();

        let restored = HolaEngine::load_from_checkpoint(&path).await.unwrap();
        let strategy = restored.study_config().await.strategy.unwrap();
        assert_eq!(
            strategy.ongoing_exploration_period,
            Some(DEFAULT_ONGOING_EXPLORATION_PERIOD)
        );
        assert_eq!(strategy.max_components, Some(DEFAULT_MAX_COMPONENTS));
        assert_eq!(strategy.min_elite_samples, Some(DEFAULT_MIN_ELITE_SAMPLES));
    }

    #[tokio::test]
    async fn legacy_auto_checkpoint_preserves_saved_gmm_route() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy-auto-route.json");
        let mut config = single_objective_config("gmm");
        let strategy = config.strategy.as_mut().unwrap();
        strategy.exploration_budget = Some(1);
        strategy.refit_interval = 1;
        let source = HolaEngine::from_config(config).unwrap();

        let warmup = source.ask().await.unwrap();
        source
            .tell(warmup.trial_id, serde_json::json!({"loss": 0.5}))
            .await
            .unwrap();
        let _pending_gmm = source.ask().await.unwrap();
        source.save_full_checkpoint(&path, None).await.unwrap();

        let mut legacy: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let auto = legacy["checkpoint"]["strategy_state"]["inner"]
            .as_object_mut()
            .unwrap();
        auto.remove("gmm_sampling_ready");
        let gmm = auto["gmm"].as_object_mut().unwrap();
        let counter = gmm["counter"]["value"].as_u64().unwrap();
        gmm.insert("counter".to_string(), serde_json::json!(counter));
        gmm.remove("epoch_start");
        gmm.remove("refit_epoch");
        std::fs::write(&path, serde_json::to_vec_pretty(&legacy).unwrap()).unwrap();

        let restored = HolaEngine::load_from_checkpoint(&path).await.unwrap();
        let (sobol_before, gmm_before) = {
            let state = restored.state.read().await;
            let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
                panic!("expected auto strategy");
            };
            assert!(auto.gmm_sampling_ready);
            (auto.sobol.index(), auto.gmm.counter())
        };
        restored.ask().await.unwrap();
        let state = restored.state.read().await;
        let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
            panic!("expected auto strategy");
        };
        assert_eq!(auto.sobol.index(), sobol_before);
        assert_eq!(auto.gmm.counter(), gmm_before + 1);
    }

    #[tokio::test]
    async fn leaderboard_import_honors_feasible_elite_floor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("three-trials.json");
        let source = HolaEngine::from_config(single_objective_config("random")).unwrap();
        for loss in [0.1, 0.2, 0.3] {
            let trial = source.ask().await.unwrap();
            source
                .tell(trial.trial_id, serde_json::json!({"loss": loss}))
                .await
                .unwrap();
        }
        source
            .save_leaderboard_checkpoint_to(&path, None)
            .await
            .unwrap();

        let mut config = single_objective_config("gmm");
        let strategy = config.strategy.as_mut().unwrap();
        strategy.exploration_budget = Some(1);
        strategy.min_elite_samples = Some(4);
        let restored = HolaEngine::from_config(config).unwrap();
        for loss in [0.9, 0.8, 0.7, 0.6] {
            let trial = restored.ask().await.unwrap();
            restored
                .tell(trial.trial_id, serde_json::json!({"loss": loss}))
                .await
                .unwrap();
        }
        let old_gmm_counter = {
            let state = restored.state.read().await;
            let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
                panic!("expected auto strategy");
            };
            assert_eq!(auto.gmm.refit_epoch(), 1);
            assert!(auto.gmm_sampling_ready);
            auto.gmm.counter()
        };

        restored.load_leaderboard_checkpoint(&path).await.unwrap();

        let state = restored.state.read().await;
        let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
            panic!("expected auto strategy");
        };
        assert_eq!(auto.gmm.refit_epoch(), 0);
        assert!(auto.gmm.counter() >= old_gmm_counter);
        assert!(!auto.gmm_sampling_ready);
    }

    #[tokio::test]
    async fn test_auto_generated_seed_is_exported_and_reproducible() {
        let mut config = single_objective_config("random");
        config.strategy.as_mut().unwrap().seed = None;
        let engine = HolaEngine::from_config(config).unwrap();

        let exported = engine.study_config().await;
        let resolved_seed = exported
            .strategy
            .as_ref()
            .and_then(|strategy| strategy.seed);
        assert!(resolved_seed.is_some(), "resolved seed must be exported");

        let reproduced = HolaEngine::from_config(exported).unwrap();
        let original_trial = engine.ask().await.unwrap();
        let reproduced_trial = reproduced.ask().await.unwrap();
        assert_eq!(original_trial.params, reproduced_trial.params);
    }

    #[tokio::test]
    async fn test_refit_limits_are_exported_and_persisted_in_full_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint_path = dir.path().join("refit-limits.json");
        let mut config = single_objective_config("gmm");
        let strategy = config.strategy.as_mut().unwrap();
        strategy.max_refit_samples = 17;
        strategy.max_refit_candidates = 53;
        strategy.ongoing_exploration_period = Some(0);
        strategy.max_components = Some(2);
        strategy.min_elite_samples = Some(7);

        let engine = HolaEngine::from_config(config).unwrap();
        let exported = engine.study_config().await;
        let exported_strategy = exported.strategy.as_ref().unwrap();
        assert_eq!(exported_strategy.max_refit_samples, 17);
        assert_eq!(exported_strategy.max_refit_candidates, 53);
        assert_eq!(exported_strategy.ongoing_exploration_period, Some(0));
        assert_eq!(exported_strategy.max_components, Some(2));
        assert_eq!(exported_strategy.min_elite_samples, Some(7));

        engine
            .save_full_checkpoint(&checkpoint_path, None)
            .await
            .unwrap();
        let checkpoint: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&checkpoint_path).unwrap()).unwrap();
        assert_eq!(checkpoint["config"]["strategy"]["max_refit_samples"], 17);
        assert_eq!(checkpoint["config"]["strategy"]["max_refit_candidates"], 53);
        assert_eq!(
            checkpoint["config"]["strategy"]["ongoing_exploration_period"],
            0
        );
        assert_eq!(checkpoint["config"]["strategy"]["max_components"], 2);
        assert_eq!(checkpoint["config"]["strategy"]["min_elite_samples"], 7);
        assert_eq!(
            checkpoint["checkpoint"]["strategy_state"]["inner"]["ongoing_exploration_period"],
            0
        );
        assert_eq!(
            checkpoint["checkpoint"]["strategy_state"]["inner"]["gmm"]["refit_config"]["n_components"],
            2
        );

        let restored = HolaEngine::load_from_checkpoint(&checkpoint_path)
            .await
            .unwrap();
        let restored_config = restored.study_config().await;
        let restored_strategy = restored_config.strategy.unwrap();
        assert_eq!(restored_strategy.max_refit_samples, 17);
        assert_eq!(restored_strategy.max_refit_candidates, 53);
        assert_eq!(restored_strategy.ongoing_exploration_period, Some(0));
        assert_eq!(restored_strategy.max_components, Some(2));
        assert_eq!(restored_strategy.min_elite_samples, Some(7));
    }

    #[tokio::test]
    async fn test_full_load_adopts_the_loaded_strategy_seed_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.json");
        let resaved_path = dir.path().join("resaved.json");

        let mut source_config = single_objective_config("random");
        source_config.strategy.as_mut().unwrap().seed = Some(11);
        let source = HolaEngine::from_config(source_config).unwrap();
        source
            .save_full_checkpoint(&source_path, None)
            .await
            .unwrap();

        let mut target_config = single_objective_config("random");
        target_config.strategy.as_mut().unwrap().seed = Some(22);
        let target = HolaEngine::from_config(target_config).unwrap();
        target.load_full_checkpoint(&source_path).await.unwrap();

        assert_eq!(
            target
                .study_config()
                .await
                .strategy
                .and_then(|strategy| strategy.seed),
            Some(11),
            "metadata must describe the loaded sampler, not the discarded target seed"
        );
        assert_eq!(
            source.ask().await.unwrap().params,
            target.ask().await.unwrap().params
        );

        target
            .save_full_checkpoint(&resaved_path, None)
            .await
            .unwrap();
        let resaved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(resaved_path).unwrap()).unwrap();
        assert_eq!(resaved["config"]["strategy"]["seed"], 11);

        // Legacy direct full checkpoints have no embedded config. The loaded
        // sampler is still authoritative; re-export metadata must not retain
        // the temporary target engine's seed.
        let mut direct: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&source_path).unwrap()).unwrap();
        direct.as_object_mut().unwrap().remove("config");
        let mut direct_target_config = single_objective_config("random");
        direct_target_config.strategy.as_mut().unwrap().seed = Some(22);
        let direct_target = HolaEngine::from_config(direct_target_config).unwrap();
        direct_target
            .load_full_checkpoint_document(direct)
            .await
            .unwrap();
        assert_eq!(
            direct_target
                .study_config()
                .await
                .strategy
                .and_then(|strategy| strategy.seed),
            Some(11)
        );
    }

    #[tokio::test]
    async fn test_objective_update_rescores_retry_receipts_and_remains_loadable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("updated-objectives.json");
        let engine = HolaEngine::from_config(single_objective_config("random")).unwrap();
        let trial = engine.ask().await.unwrap();
        let metrics = serde_json::json!({"loss": 0.4, "accuracy": 0.9});
        engine.tell(trial.trial_id, metrics.clone()).await.unwrap();
        let better = engine.ask().await.unwrap();
        engine
            .tell(
                better.trial_id,
                serde_json::json!({"loss": 0.5, "accuracy": 0.95}),
            )
            .await
            .unwrap();

        engine
            .update_objectives(vec![ObjectiveConfig {
                field: "accuracy".to_string(),
                obj_type: "maximize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("quality".to_string()),
            }])
            .await
            .unwrap();
        let retry = engine
            .tell_with_outcome(trial.trial_id, metrics.clone())
            .await
            .unwrap();
        assert!(!retry.newly_committed);
        assert_eq!(retry.trial_count, 2);
        assert_eq!(retry.completed.rank, 1);
        assert_eq!(
            retry.completed.score_vector,
            serde_json::json!({"quality": -0.9})
        );

        engine.save_full_checkpoint(&path, None).await.unwrap();
        let restored = HolaEngine::load_from_checkpoint(&path).await.unwrap();
        let retry_after_restore = restored
            .tell_with_outcome(trial.trial_id, metrics)
            .await
            .unwrap();
        assert!(!retry_after_restore.newly_committed);
        assert_eq!(
            retry_after_restore.completed.score_vector,
            retry.completed.score_vector
        );
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
        let start = Arc::new(tokio::sync::Barrier::new(n + 2));
        let mut handles = Vec::new();
        for i in 0..n {
            let eng = engine.clone();
            let start = Arc::clone(&start);
            handles.push(tokio::spawn(async move {
                let trial = eng.ask().await.expect("ask should succeed");
                // Provide both metric fields so the post-migration vector
                // topology scores every trial as feasible.
                let metrics = serde_json::json!({
                    "loss": (i as f64) / (n as f64),
                    "latency": (n - i) as f64,
                });
                start.wait().await;
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
            let start = Arc::clone(&start);
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
                start.wait().await;
                eng.update_objectives(new_objectives)
                    .await
                    .expect("update_objectives should succeed");
            })
        };

        // Release every tell and the topology migration from the same barrier,
        // forcing real lock contention instead of relying on scheduler timing.
        start.wait().await;

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
    async fn test_full_checkpoint_roundtrip_preserves_infeasible_observation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("infeasible.json");
        let mut config = single_objective_config("random");
        config.objectives[0].target = Some(0.0);
        config.objectives[0].limit = Some(1.0);

        let engine = HolaEngine::from_config(config).unwrap();
        let trial = engine.ask().await.unwrap();
        engine
            .tell(trial.trial_id, serde_json::json!({ "loss": 2.0 }))
            .await
            .unwrap();

        engine.save_full_checkpoint(&path, None).await.unwrap();
        let json = std::fs::read_to_string(&path).unwrap();
        assert!(json.contains("$hola.float"));
        assert!(!json.contains("\"observation\": null"));

        let resumed = HolaEngine::load_from_checkpoint(&path)
            .await
            .expect("a checkpoint containing an infeasible trial must remain loadable");
        assert_eq!(resumed.trial_count().await, 1);
        let restored = resumed
            .completed_trial(trial.trial_id, true)
            .await
            .expect("the infeasible trial must survive the round trip");
        assert_eq!(restored.trial_id, trial.trial_id);
    }

    #[tokio::test]
    async fn full_checkpoint_roundtrip_accepts_canonicalized_scalar_and_vector_nan() {
        let dir = tempfile::tempdir().unwrap();

        let scalar = HolaEngine::from_config(single_objective_config("random")).unwrap();
        let scalar_trial = scalar.ask().await.unwrap();
        scalar
            .tell(scalar_trial.trial_id, serde_json::json!({"loss": "nan"}))
            .await
            .unwrap();
        let scalar_path = dir.path().join("scalar-nan.json");
        scalar
            .save_full_checkpoint(&scalar_path, None)
            .await
            .unwrap();
        let restored_scalar = HolaEngine::load_from_checkpoint(&scalar_path)
            .await
            .expect("canonicalized scalar NaN must remain loadable");
        assert_eq!(restored_scalar.trial_count().await, 1);

        let mut vector_config = single_objective_config("random");
        vector_config.objectives[0].group = Some("quality".to_string());
        vector_config.objectives.push(ObjectiveConfig {
            field: "latency".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: Some("speed".to_string()),
        });
        let vector = HolaEngine::from_config(vector_config).unwrap();
        let vector_trial = vector.ask().await.unwrap();
        vector
            .tell(
                vector_trial.trial_id,
                serde_json::json!({"loss": "nan", "latency": 1.0}),
            )
            .await
            .unwrap();
        let vector_path = dir.path().join("vector-nan.json");
        vector
            .save_full_checkpoint(&vector_path, None)
            .await
            .unwrap();
        let restored_vector = HolaEngine::load_from_checkpoint(&vector_path)
            .await
            .expect("canonicalized vector NaN must remain loadable");
        assert_eq!(restored_vector.trial_count().await, 1);
    }

    #[tokio::test]
    async fn test_full_checkpoint_roundtrip_preserves_pending_jobs_and_id_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pending.json");
        let engine = HolaEngine::from_config(single_objective_config("sobol")).unwrap();

        let completed = engine.ask().await.unwrap();
        engine
            .tell(completed.trial_id, serde_json::json!({ "loss": 0.5 }))
            .await
            .unwrap();
        let pending = engine.ask().await.unwrap();
        engine.save_full_checkpoint(&path, None).await.unwrap();

        let resumed = HolaEngine::load_from_checkpoint(&path).await.unwrap();
        let restored = resumed
            .tell(pending.trial_id, serde_json::json!({ "loss": 0.25 }))
            .await
            .expect("a late worker result must match the pending job saved before restart");
        assert_eq!(restored.trial_id, pending.trial_id);
        let restored_x = restored.params["x"].as_f64().unwrap();
        let pending_x = pending.params["x"].as_f64().unwrap();
        assert!((restored_x - pending_x).abs() <= f64::EPSILON);

        let next = resumed.ask().await.unwrap();
        assert!(
            next.trial_id > pending.trial_id,
            "resumed ID allocation must not reuse a persisted pending ID"
        );
    }

    #[tokio::test]
    async fn test_full_checkpoint_rejects_incompatible_engine_without_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("configured.json");
        let source = HolaEngine::from_config(single_objective_config("random")).unwrap();
        let trial = source.ask().await.unwrap();
        source
            .tell(trial.trial_id, serde_json::json!({ "loss": 0.5 }))
            .await
            .unwrap();
        source.save_full_checkpoint(&path, None).await.unwrap();

        let mut incompatible = single_objective_config("random");
        incompatible.space.insert(
            "y".to_string(),
            ParamConfig::Real {
                min: -1.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        );
        let target = HolaEngine::from_config(incompatible).unwrap();
        let error = target.load_full_checkpoint(&path).await.unwrap_err();
        assert!(error.to_string().contains("does not match"));
        assert_eq!(target.trial_count().await, 0);
    }

    #[tokio::test]
    async fn test_checkpoint_load_rejects_forged_completed_state_without_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let full_path = dir.path().join("full.json");
        let leaderboard_path = dir.path().join("leaderboard.json");
        let source = HolaEngine::from_config(single_objective_config("random")).unwrap();
        let trial = source.ask().await.unwrap();
        source
            .tell(trial.trial_id, serde_json::json!({"loss": 0.5}))
            .await
            .unwrap();
        source.save_full_checkpoint(&full_path, None).await.unwrap();
        source
            .save_leaderboard_checkpoint_to(&leaderboard_path, None)
            .await
            .unwrap();

        let target = HolaEngine::from_config(single_objective_config("random")).unwrap();
        let retained = target.ask().await.unwrap();
        target
            .tell(retained.trial_id, serde_json::json!({"loss": 0.25}))
            .await
            .unwrap();

        let mut forged_candidate: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&full_path).unwrap()).unwrap();
        forged_candidate["checkpoint"]["leaderboard"]["trials"][0]["candidate"]["unexpected"] =
            serde_json::json!(true);
        let error = target
            .load_full_checkpoint_document(forged_candidate)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("configured space"));
        assert_eq!(target.trial_count().await, 1);
        assert_eq!(target.top_k(1, true).await[0].metrics["loss"], 0.25);

        let mut forged_observation: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&full_path).unwrap()).unwrap();
        forged_observation["checkpoint"]["leaderboard"]["trials"][0]["observation"] =
            serde_json::json!(999.0);
        let error = target
            .load_full_checkpoint_document(forged_observation)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("conflicts with its raw metrics"));
        assert_eq!(target.trial_count().await, 1);

        let mut forged_leaderboard: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&leaderboard_path).unwrap()).unwrap();
        forged_leaderboard["leaderboard"]["trials"][0]["candidate"]["unexpected"] =
            serde_json::json!(true);
        let error = target
            .load_leaderboard_checkpoint_document(forged_leaderboard)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("configured space"));
        assert_eq!(target.trial_count().await, 1);
        assert_eq!(target.top_k(1, true).await[0].metrics["loss"], 0.25);

        let mut forged_receipt_count: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&full_path).unwrap()).unwrap();
        forged_receipt_count["runtime_state"]["completion_receipts"]["0"]["committed_count"] =
            serde_json::json!(0);
        let error = target
            .load_full_checkpoint_document(forged_receipt_count)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("mismatched committed_count"));
        assert_eq!(target.trial_count().await, 1);

        let mut forged_receipt_candidate: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&full_path).unwrap()).unwrap();
        forged_receipt_candidate["runtime_state"]["completion_receipts"]["0"]["completed"]["params"]
            ["x"] = serde_json::json!(0.75);
        let error = target
            .load_full_checkpoint_document(forged_receipt_candidate)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("leaderboard identity"));
        assert_eq!(target.trial_count().await, 1);

        let mut oversized_cancelled: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&full_path).unwrap()).unwrap();
        oversized_cancelled["runtime_state"]["cancelled"] = serde_json::Value::Array(
            (0..=MAX_CANCELLED_RETAINED)
                .map(|id| serde_json::json!(10_000 + id))
                .collect(),
        );
        let error = target
            .load_full_checkpoint_document(oversized_cancelled)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cancelled trial ids"));
        assert_eq!(target.trial_count().await, 1);
    }

    #[tokio::test]
    async fn test_full_checkpoint_rejects_strategy_variant_conflicting_with_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sobol.json");
        let source = HolaEngine::from_config(single_objective_config("sobol")).unwrap();
        source.save_full_checkpoint(&path, None).await.unwrap();

        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        raw["config"]["strategy"]["strategy_type"] = serde_json::json!("random");

        let target = HolaEngine::from_config(single_objective_config("random")).unwrap();
        let error = target.load_full_checkpoint_document(raw).await.unwrap_err();
        assert!(error.to_string().contains("strategy state is 'sobol'"));
        assert_eq!(target.trial_count().await, 0);
    }

    #[tokio::test]
    async fn test_full_checkpoint_rejects_strategy_dimension_conflicting_with_space() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("one-dimensional.json");
        let source = HolaEngine::from_config(single_objective_config("gmm")).unwrap();
        source.save_full_checkpoint(&path, None).await.unwrap();

        let mut target_config = single_objective_config("gmm");
        target_config.space.insert(
            "y".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        );
        let target = HolaEngine::from_config(target_config).unwrap();
        let target_space =
            serde_json::to_value(target.study_config().await).unwrap()["space"].clone();

        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        raw["config"]["space"] = target_space;
        let error = target.load_full_checkpoint_document(raw).await.unwrap_err();
        assert!(error.to_string().contains("GMM dimension"));
        assert_eq!(target.trial_count().await, 0);
    }

    #[tokio::test]
    async fn full_checkpoint_rejects_forged_strategy_seed_and_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("random-pending.json");
        let mut config = single_objective_config("random");
        config.strategy.as_mut().unwrap().seed = Some(11);
        let source = HolaEngine::from_config(config.clone()).unwrap();
        let _pending = source.ask().await.unwrap();
        source.save_full_checkpoint(&path, None).await.unwrap();
        let pristine: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();

        let mut forged_cursor = pristine.clone();
        forged_cursor["checkpoint"]["strategy_state"]["inner"]["counter"] = serde_json::json!(0);
        let target = HolaEngine::from_config(config.clone()).unwrap();
        let error = target
            .load_full_checkpoint_document(forged_cursor)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("random cursor"));
        assert_eq!(target.pending_count().await, 0);

        let mut forged_seed = pristine;
        forged_seed["checkpoint"]["strategy_state"]["inner"]["seed"] = serde_json::json!(12);
        let target = HolaEngine::from_config(config).unwrap();
        let error = target
            .load_full_checkpoint_document(forged_seed)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("random seed"));
        assert_eq!(target.pending_count().await, 0);
    }

    #[tokio::test]
    async fn full_checkpoint_rejects_gmm_epoch_start_after_suggestion_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gmm-pending.json");
        let config = single_objective_config("gmm");
        let source = HolaEngine::from_config(config.clone()).unwrap();
        let _pending = source.ask().await.unwrap();
        source.save_full_checkpoint(&path, None).await.unwrap();

        let pristine: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let mut missing_epoch = pristine.clone();
        let gmm = missing_epoch["checkpoint"]["strategy_state"]["inner"]["gmm"]
            .as_object_mut()
            .unwrap();
        gmm.remove("epoch_start");
        gmm.remove("refit_epoch");
        let target = HolaEngine::from_config(config.clone()).unwrap();
        let error = target
            .load_full_checkpoint_document(missing_epoch)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("require epoch_start"));
        assert_eq!(target.pending_count().await, 0);

        let mut forged = pristine;
        let counter = forged["checkpoint"]["strategy_state"]["inner"]["gmm"]["counter"]["value"]
            .as_u64()
            .unwrap();
        forged["checkpoint"]["strategy_state"]["inner"]["gmm"]["epoch_start"] =
            serde_json::json!(counter + 1);
        forged["checkpoint"]["strategy_state"]["inner"]["gmm"]["refit_epoch"] =
            serde_json::json!(1);

        let target = HolaEngine::from_config(config).unwrap();
        let error = target
            .load_full_checkpoint_document(forged)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("GMM epoch start"));
        assert_eq!(target.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_full_checkpoint_rejects_stale_runtime_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stale-runtime.json");
        let engine = HolaEngine::from_config(single_objective_config("random")).unwrap();
        let _pending = engine.ask().await.unwrap();
        engine.save_full_checkpoint(&path, None).await.unwrap();

        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        raw["runtime_state"]["next_pending_id"] = serde_json::json!(0);
        std::fs::write(&path, serde_json::to_vec_pretty(&raw).unwrap()).unwrap();

        let error = HolaEngine::load_from_checkpoint(&path)
            .await
            .err()
            .expect("stale runtime cursor must be rejected");
        assert!(error.contains("next_pending_id"));
    }

    #[test]
    fn test_auto_checkpoint_directory_is_created_at_startup() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint_dir = dir.path().join("nested/checkpoints");
        let mut config = single_objective_config("random");
        config.checkpoint = Some(CheckpointConfig {
            directory: checkpoint_dir.to_string_lossy().into_owned(),
            interval: 10,
            max_checkpoints: Some(2),
            load_from: None,
        });

        HolaEngine::from_config(config).unwrap();
        assert!(checkpoint_dir.is_dir());
    }

    #[tokio::test]
    async fn test_bounded_auto_checkpoint_uses_total_completed_cadence() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = single_objective_config("random");
        config.max_leaderboard_size = Some(3);
        config.checkpoint = Some(CheckpointConfig {
            directory: dir.path().to_string_lossy().into_owned(),
            interval: 2,
            max_checkpoints: None,
            load_from: None,
        });

        let engine = HolaEngine::from_config(config).unwrap();
        for loss in 0..6 {
            let trial = engine.ask().await.unwrap();
            engine
                .tell(trial.trial_id, serde_json::json!({"loss": loss as f64}))
                .await
                .unwrap();
        }

        assert_eq!(
            engine.retained_trial_count().await,
            3,
            "retained history stays capped"
        );
        let mut checkpoint_names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".json"))
            .collect();
        checkpoint_names.sort();
        assert_eq!(
            checkpoint_names,
            vec![
                "checkpoint_000002.json",
                "checkpoint_000004.json",
                "checkpoint_000006.json",
            ],
            "checkpoint cadence must keep advancing after retained length reaches its cap"
        );

        let latest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.path().join("checkpoint_000006.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(latest["checkpoint"]["metadata"]["n_trials"], 3);
        assert_eq!(latest["checkpoint"]["leaderboard"]["total_completed"], 6);
        assert!(
            latest["checkpoint"]["metadata"]["description"]
                .as_str()
                .is_some_and(|description| description.contains("6 completed")
                    && description.contains("3 retained"))
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn commit_hook_is_exactly_once_when_post_commit_work_is_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = single_objective_config("random");
        config.checkpoint = Some(CheckpointConfig {
            directory: dir.path().to_string_lossy().into_owned(),
            interval: 1,
            max_checkpoints: None,
            load_from: None,
        });
        let engine = HolaEngine::from_config(config).unwrap();
        let trial = engine.ask().await.unwrap();

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_for_task = Arc::clone(&callback_count);
        let gate = Arc::new(std::sync::Barrier::new(2));
        let task_gate = Arc::clone(&gate);
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let task_engine = engine.clone();
        let tell_task = tokio::spawn(async move {
            task_engine
                .tell_with_outcome_on_commit(
                    trial.trial_id,
                    serde_json::json!({"loss": 0.25}),
                    move |_, _| {
                        callback_count_for_task.fetch_add(1, Ordering::SeqCst);
                        let _ = entered_tx.send(());
                        // Keep the task inside the synchronous commit boundary
                        // until the test has requested cancellation.
                        task_gate.wait();
                    },
                )
                .await
        });

        entered_rx
            .await
            .expect("commit callback should run before checkpoint I/O");
        tell_task.abort();
        tokio::task::spawn_blocking(move || gate.wait())
            .await
            .unwrap();
        assert!(
            tell_task.await.unwrap_err().is_cancelled(),
            "the request-side future should be cancellable after publication"
        );
        assert_eq!(engine.trial_count().await, 1);
        assert_eq!(callback_count.load(Ordering::SeqCst), 1);
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let saved = std::fs::read_dir(dir.path())
                    .unwrap()
                    .filter_map(Result::ok)
                    .any(|entry| entry.path().extension().is_some_and(|ext| ext == "json"));
                if saved {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("auto-checkpoint maintenance must survive tell-future cancellation");

        let replay_count = Arc::clone(&callback_count);
        let replay = engine
            .tell_with_outcome_on_commit(
                trial.trial_id,
                serde_json::json!({"loss": 0.25}),
                move |_, _| {
                    replay_count.fetch_add(1, Ordering::SeqCst);
                },
            )
            .await
            .unwrap();
        assert!(!replay.newly_committed);
        assert_eq!(replay.completed.trial_id, trial.trial_id);
        assert_eq!(callback_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_auto_checkpoint_failure_is_counted_after_tell_commits() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint_dir = dir.path().join("checkpoints");
        let mut config = single_objective_config("random");
        config.checkpoint = Some(CheckpointConfig {
            directory: checkpoint_dir.to_string_lossy().into_owned(),
            interval: 1,
            max_checkpoints: None,
            load_from: None,
        });
        let engine = HolaEngine::from_config(config).unwrap();

        // Construction creates the directory. Replace it with a regular file so
        // the unattended write fails deterministically on every platform.
        std::fs::remove_dir(&checkpoint_dir).unwrap();
        std::fs::write(&checkpoint_dir, b"not a directory").unwrap();

        let trial = engine.ask().await.unwrap();
        let outcome = engine
            .tell_with_outcome(trial.trial_id, serde_json::json!({"loss": 0.5}))
            .await
            .expect("checkpoint post-processing must not make a committed tell retryable");
        assert_eq!(outcome.completed.trial_id, trial.trial_id);
        assert_eq!(outcome.post_commit_warnings.len(), 1);
        assert!(outcome.post_commit_warnings[0].contains("auto-checkpoint failed"));
        assert_eq!(engine.trial_count().await, 1);
        assert_eq!(engine.checkpoint_failure_count(), 1);

        let retry = engine
            .tell_with_outcome(trial.trial_id, serde_json::json!({"loss": 0.5}))
            .await
            .unwrap();
        assert!(!retry.newly_committed);
        assert_eq!(retry.post_commit_warnings, outcome.post_commit_warnings);
    }

    #[tokio::test]
    async fn post_commit_refit_failure_is_counted_warned_and_retry_safe() {
        let mut config = single_objective_config("gmm");
        let strategy = config.strategy.as_mut().unwrap();
        strategy.exploration_budget = Some(1);
        strategy.refit_interval = 20;
        let engine = HolaEngine::from_config(config).unwrap();
        engine.force_refit_failure.store(true, Ordering::SeqCst);

        let trial = engine.ask().await.unwrap();
        let metrics = serde_json::json!({"loss": 0.25});
        let outcome = engine
            .tell_with_outcome(trial.trial_id, metrics.clone())
            .await
            .unwrap();
        assert!(outcome.newly_committed);
        assert_eq!(engine.refit_failure_count(), 1);
        assert_eq!(outcome.post_commit_warnings.len(), 1);
        assert!(outcome.post_commit_warnings[0].contains("refit failed"));

        let retry = engine
            .tell_with_outcome(trial.trial_id, metrics)
            .await
            .unwrap();
        assert!(!retry.newly_committed);
        assert_eq!(retry.post_commit_warnings, outcome.post_commit_warnings);
        assert_eq!(engine.refit_failure_count(), 1);

        // A genuinely new completion retries the missing initial fit even
        // though the next periodic cadence is still far away.
        let next_trial = engine.ask().await.unwrap();
        engine
            .tell(next_trial.trial_id, serde_json::json!({"loss": 0.125}))
            .await
            .unwrap();
        let state = engine.state.read().await;
        let DynStrategyInner::Auto(auto) = &state.strategy.inner else {
            panic!("expected auto strategy");
        };
        assert_eq!(auto.gmm.refit_epoch(), 1);
        assert!(auto.gmm_sampling_ready);
        assert_eq!(engine.refit_failure_count(), 1);
    }

    #[test]
    fn test_bounded_refit_schedule_uses_total_completed() {
        let config = RefitConfig::with_top_k(2, 2, 2);
        let objectives = vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }];
        let mut leaderboard = DynLeaderboard::Scalar(Leaderboard::new());
        leaderboard.set_max_size(Some(3));
        let mut refit_at = Vec::new();

        for completed in 1..=6 {
            match &mut leaderboard {
                DynLeaderboard::Scalar(inner) => {
                    inner.push(serde_json::json!({"x": completed}), completed as f64);
                }
                DynLeaderboard::Vector(_) => unreachable!(),
            }
            assert!(leaderboard.len() <= 3);
            let cadence = leaderboard.completed_count();
            if config.should_refit(cadence) {
                refit_at.push(cadence);
                assert_eq!(
                    leaderboard
                        .top_k_for_refit(
                            config.selection_count(cadence),
                            DEFAULT_MAX_REFIT_CANDIDATES,
                            &objectives,
                        )
                        .len(),
                    2
                );
            }
        }

        assert_eq!(refit_at, vec![2, 4, 6]);
        assert_eq!(leaderboard.len(), 3);
        assert_eq!(leaderboard.completed_count(), 6);
    }

    #[test]
    fn queued_cadence_does_not_repeat_a_successful_initial_retry() {
        // The cadence task observed the pre-fit epoch, then a retry installed
        // the model from the same completed history before the cadence task
        // acquired `refit_lock`.
        assert!(!should_attempt_post_commit_refit(
            true,
            Some(0),
            None,
            true,
            false,
        ));

        // If additional completions arrived after that fitted snapshot, the
        // cadence task remains meaningful and must refit the newer history.
        assert!(should_attempt_post_commit_refit(
            true,
            Some(0),
            None,
            true,
            true,
        ));
    }

    #[test]
    #[ignore = "performance probe; run explicitly with --ignored --nocapture"]
    fn bounded_refit_workset_scaling_probe() {
        use std::hint::black_box;
        use std::time::Instant;

        let objectives = vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }];

        for history_size in [1_000usize, 10_000, 100_000] {
            let mut inner = Leaderboard::with_capacity(history_size);
            for trial_id in 0..history_size {
                inner.push(
                    serde_json::json!({"x": trial_id as f64 / history_size as f64}),
                    ((trial_id.wrapping_mul(37)) % history_size) as f64,
                );
            }
            let leaderboard = DynLeaderboard::Scalar(inner);
            let started = Instant::now();
            let workset = black_box(leaderboard.top_k_for_refit(
                DEFAULT_MAX_REFIT_SAMPLES,
                DEFAULT_MAX_REFIT_CANDIDATES,
                &objectives,
            ));
            let elapsed = started.elapsed();

            assert_eq!(workset.len(), history_size.min(DEFAULT_MAX_REFIT_SAMPLES));
            assert!(workset.len() <= DEFAULT_MAX_REFIT_SAMPLES);
            assert!(
                elapsed < std::time::Duration::from_secs(5),
                "bounded refit selection exceeded the 5s debug budget at history={history_size}: {elapsed:?}"
            );
            eprintln!(
                "history={history_size}, covered_at_most={DEFAULT_MAX_REFIT_CANDIDATES}, workset={}, selection={elapsed:?}",
                workset.len()
            );
        }
    }

    #[test]
    #[ignore = "performance probe; run explicitly with --ignored --nocapture"]
    fn multiobjective_refit_selection_scaling_probe() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const HISTORY_SIZE: usize = 1_000;
        const ELITE_COUNT: usize = 250;
        for group_count in [2usize, 3, 5] {
            let objectives: Vec<ObjectiveConfig> = (0..group_count)
                .map(|group| ObjectiveConfig {
                    field: format!("metric-{group}"),
                    obj_type: "minimize".to_string(),
                    target: None,
                    limit: None,
                    priority: 1.0,
                    group: Some(format!("group-{group}")),
                })
                .collect();
            let mut inner = Leaderboard::with_capacity(HISTORY_SIZE);
            for trial in 0..HISTORY_SIZE {
                let mut observation = BTreeMap::new();
                observation.insert("group-0".to_string(), trial as f64);
                observation.insert("group-1".to_string(), (HISTORY_SIZE - trial) as f64);
                for group in 2..group_count {
                    observation.insert(
                        format!("group-{group}"),
                        ((trial.wrapping_mul(37 + group)) % HISTORY_SIZE) as f64,
                    );
                }
                inner.push(serde_json::json!({"trial": trial}), observation);
            }
            let leaderboard = DynLeaderboard::Vector(inner);

            let started = Instant::now();
            let elites =
                black_box(leaderboard.top_k_for_refit(ELITE_COUNT, HISTORY_SIZE, &objectives));
            let elapsed = started.elapsed();
            assert_eq!(elites.len(), ELITE_COUNT);
            assert!(
                elapsed < Duration::from_secs(5),
                "{group_count}-group NSGA-II selection exceeded the debug-build budget: {elapsed:?}"
            );
            eprintln!(
                "N={HISTORY_SIZE}, groups={group_count}, elites={ELITE_COUNT}, selection={elapsed:?}"
            );
        }
    }

    #[tokio::test]
    async fn deferred_batch_rankings_materialize_and_replay_exactly() {
        let mut config = single_objective_config("random");
        config.objectives = (0..3)
            .map(|group| ObjectiveConfig {
                field: format!("metric-{group}"),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some(format!("group-{group}")),
            })
            .collect();
        let engine = HolaEngine::from_config(config).unwrap();
        let mut completions = Vec::new();
        for index in 0..24u64 {
            let trial = engine.ask().await.unwrap();
            let metrics = serde_json::json!({
                "metric-0": (index * 17 % 23) as f64,
                "metric-1": ((23 - index) * 11 % 29) as f64,
                "metric-2": (index * index % 31) as f64,
            });
            engine
                .tell_without_ranking(trial.trial_id, metrics.clone())
                .await
                .unwrap();
            completions.push((trial.trial_id, metrics));
        }

        {
            let state = engine.state.read().await;
            assert_eq!(state.deferred_completion_receipts, completions.len());
            assert!(
                state
                    .completion_receipts
                    .values()
                    .all(|receipt| receipt.ranking_deferred)
            );
        }

        engine.finalize_deferred_rankings().await.unwrap();
        let final_view: BTreeMap<u64, CompletedTrial> = engine
            .trials("rank", true)
            .await
            .into_iter()
            .map(|trial| (trial.trial_id, trial))
            .collect();
        {
            let state = engine.state.read().await;
            assert_eq!(state.deferred_completion_receipts, 0);
            assert!(
                state
                    .completion_receipts
                    .values()
                    .all(|receipt| !receipt.ranking_deferred)
            );
        }

        for (trial_id, metrics) in completions {
            let replay = engine.tell_with_outcome(trial_id, metrics).await.unwrap();
            let expected = &final_view[&trial_id];
            assert!(!replay.newly_committed);
            assert_eq!(replay.completed.rank, expected.rank);
            assert_eq!(replay.completed.pareto_front, expected.pareto_front);
            assert_eq!(replay.completed.score_vector, expected.score_vector);
        }
    }

    #[tokio::test]
    async fn public_tell_materializes_a_deferred_batch_before_replay() {
        let mut config = single_objective_config("random");
        config.objectives = vec![
            ObjectiveConfig {
                field: "a".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("a".to_string()),
            },
            ObjectiveConfig {
                field: "b".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("b".to_string()),
            },
            ObjectiveConfig {
                field: "c".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("c".to_string()),
            },
        ];
        let engine = HolaEngine::from_config(config).unwrap();
        let mut completions = Vec::new();
        for index in 0..5u64 {
            let trial = engine.ask().await.unwrap();
            let metrics = serde_json::json!({
                "a": index as f64,
                "b": (5 - index) as f64,
                "c": (index * 3 % 5) as f64,
            });
            engine
                .tell_without_ranking(trial.trial_id, metrics.clone())
                .await
                .unwrap();
            completions.push((trial.trial_id, metrics));
        }

        let (trial_id, metrics) = completions[2].clone();
        let replay = engine.tell_with_outcome(trial_id, metrics).await.unwrap();
        assert!(!replay.newly_committed);
        assert_eq!(engine.state.read().await.deferred_completion_receipts, 0);
        let current = engine.completed_trial(trial_id, true).await.unwrap();
        assert_eq!(replay.completed.rank, current.rank);
        assert_eq!(replay.completed.pareto_front, current.pareto_front);
    }

    #[tokio::test]
    async fn bounded_batch_never_evicts_an_unranked_receipt() {
        let mut config = single_objective_config("random");
        config.max_leaderboard_size = Some(2);
        let engine = HolaEngine::from_config(config).unwrap();
        let mut completions = Vec::new();
        for loss in [0.4, 0.2, 0.3] {
            let trial = engine.ask().await.unwrap();
            let metrics = serde_json::json!({"loss": loss});
            engine
                .tell_without_ranking(trial.trial_id, metrics.clone())
                .await
                .unwrap();
            completions.push((trial.trial_id, metrics));
        }

        let state = engine.state.read().await;
        assert_eq!(state.leaderboard.len(), 2);
        assert_eq!(state.deferred_completion_receipts, 0);
        assert!(
            state
                .completion_receipts
                .values()
                .all(|receipt| !receipt.ranking_deferred)
        );
        drop(state);

        // The first trial has left the bounded leaderboard, so this successful
        // replay proves its exact receipt was materialized before eviction.
        let replay = engine
            .tell_with_outcome(completions[0].0, completions[0].1.clone())
            .await
            .unwrap();
        assert!(!replay.newly_committed);
        assert_eq!(replay.completed.trial_id, completions[0].0);
    }

    #[tokio::test]
    async fn deferred_receipts_round_trip_through_a_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deferred.json");
        let engine = HolaEngine::from_config(single_objective_config("random")).unwrap();
        let mut completions = Vec::new();
        for loss in [0.8, 0.1, 0.5] {
            let trial = engine.ask().await.unwrap();
            let metrics = serde_json::json!({"loss": loss});
            engine
                .tell_without_ranking(trial.trial_id, metrics.clone())
                .await
                .unwrap();
            completions.push((trial.trial_id, metrics));
        }
        engine.save(path.to_str().unwrap()).await.unwrap();

        let restored = HolaEngine::load_from_checkpoint(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            restored.state.read().await.deferred_completion_receipts,
            completions.len()
        );
        restored.finalize_deferred_rankings().await.unwrap();
        let replay = restored
            .tell_with_outcome(completions[1].0, completions[1].1.clone())
            .await
            .unwrap();
        assert!(!replay.newly_committed);
        assert_eq!(replay.completed.rank, 0);
    }

    #[tokio::test]
    #[ignore = "performance probe; run explicitly with --ignored --nocapture"]
    async fn deferred_multiobjective_batch_scaling_probe() {
        use std::time::Instant;

        for group_count in [3usize, 5] {
            for sample_count in [100usize, 200, 500, 1_000] {
                let mut config = single_objective_config("random");
                config.objectives = (0..group_count)
                    .map(|group| ObjectiveConfig {
                        field: format!("metric-{group}"),
                        obj_type: "minimize".to_string(),
                        target: None,
                        limit: None,
                        priority: 1.0,
                        group: Some(format!("group-{group}")),
                    })
                    .collect();
                let engine = HolaEngine::from_config(config).unwrap();
                let started = Instant::now();
                for index in 0..sample_count {
                    let trial = engine.ask().await.unwrap();
                    let metrics = serde_json::Value::Object(
                        (0..group_count)
                            .map(|group| {
                                (
                                    format!("metric-{group}"),
                                    serde_json::json!(
                                        (index.wrapping_mul(37 + group) % sample_count) as f64
                                    ),
                                )
                            })
                            .collect(),
                    );
                    engine
                        .tell_without_ranking(trial.trial_id, metrics)
                        .await
                        .unwrap();
                }
                let commits = started.elapsed();
                let finalize_started = Instant::now();
                engine.finalize_deferred_rankings().await.unwrap();
                let finalize = finalize_started.elapsed();
                eprintln!(
                    "N={sample_count}, groups={group_count}, commits={commits:?}, finalize={finalize:?}, total={:?}",
                    started.elapsed()
                );
            }
        }
    }

    #[tokio::test]
    #[ignore = "performance probe; run explicitly with --ignored --nocapture"]
    async fn tell_hot_path_scaling_probe() {
        use std::time::{Duration, Instant};

        for vector in [false, true] {
            let mut config = single_objective_config("random");
            if vector {
                config.objectives = vec![
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
            }
            let engine = HolaEngine::from_config(config).unwrap();
            {
                let mut state = engine.state.write().await;
                let objectives = state.objectives.clone();
                for trial_id in 0..100_000u64 {
                    let metrics = if vector {
                        serde_json::json!({
                            "loss": (trial_id % 997) as f64,
                            "latency": ((100_000 - trial_id) % 991) as f64,
                        })
                    } else {
                        serde_json::json!({"loss": (trial_id % 997) as f64})
                    };
                    state.leaderboard.push_with_raw(
                        trial_id,
                        serde_json::json!({"x": trial_id as f64 / 100_000.0}),
                        metrics,
                        &objectives,
                    );
                }
            }

            let trial = engine.ask().await.unwrap();
            let metrics = if vector {
                serde_json::json!({"loss": 0.25, "latency": 0.75})
            } else {
                serde_json::json!({"loss": 0.25})
            };
            let started = Instant::now();
            engine.tell(trial.trial_id, metrics).await.unwrap();
            let elapsed = started.elapsed();
            assert!(
                elapsed < Duration::from_secs(5),
                "{} tell exceeded the 5s debug budget: {elapsed:?}",
                if vector { "two-objective" } else { "scalar" }
            );
            eprintln!(
                "{} tell at 100k history: {elapsed:?}",
                if vector { "two-objective" } else { "scalar" }
            );
        }
    }

    #[test]
    fn test_checkpoint_config_rejects_zero_retention() {
        let mut config = single_objective_config("random");
        config.checkpoint = Some(CheckpointConfig {
            directory: ".".to_string(),
            interval: 1,
            max_checkpoints: Some(0),
            load_from: None,
        });
        let error = HolaEngine::from_config(config)
            .err()
            .expect("zero retained checkpoints must fail validation");
        assert!(error.contains("checkpoint.max_checkpoints"));
        assert!(error.contains("at least 1"));
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
            bounded.retained_trial_count().await,
            cap,
            "bounded study must retain at most max_leaderboard_size trials"
        );
        assert_eq!(
            bounded.trial_count().await,
            n,
            "public completed count must remain monotonic past the retention cap"
        );
        assert_eq!(
            unbounded.retained_trial_count().await,
            n,
            "default (unbounded) study must retain every trial"
        );
    }

    #[tokio::test]
    async fn trial_lifecycle_uses_receipts_after_leaderboard_eviction() {
        let mut config = single_objective_config("random");
        config.max_leaderboard_size = Some(1);
        let engine = HolaEngine::from_config(config).unwrap();

        let evicted = engine.ask().await.unwrap();
        engine
            .tell(evicted.trial_id, serde_json::json!({"loss": 2.0}))
            .await
            .unwrap();
        let retained = engine.ask().await.unwrap();
        engine
            .tell(retained.trial_id, serde_json::json!({"loss": 1.0}))
            .await
            .unwrap();

        assert!(
            engine
                .completed_trial(evicted.trial_id, true)
                .await
                .is_none(),
            "the stronger lifecycle result must come from the receipt, not the bounded leaderboard"
        );
        assert_eq!(
            engine.trial_lifecycle(evicted.trial_id).await,
            TrialLifecycle::Completed
        );
        assert_eq!(
            engine.trial_lifecycle(retained.trial_id).await,
            TrialLifecycle::Completed
        );

        let pending = engine.ask().await.unwrap();
        assert_eq!(
            engine.trial_lifecycle(pending.trial_id).await,
            TrialLifecycle::Pending
        );
        engine.cancel(pending.trial_id).await.unwrap();
        assert_eq!(
            engine.trial_lifecycle(pending.trial_id).await,
            TrialLifecycle::NotPending
        );
        assert_eq!(
            engine.trial_lifecycle(u64::MAX).await,
            TrialLifecycle::NotPending
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
            engine.retained_trial_count().await,
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
            engine.retained_trial_count().await <= cap,
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
                        engine.retained_trial_count().await <= cap,
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
            engine.retained_trial_count().await <= cap,
            "final bounded study must still respect the cap"
        );
    }

    #[tokio::test]
    async fn test_bounded_vector_study_with_infeasible_terminates_across_migration() {
        // Exercises the migration counter carry-over on a VECTOR board that
        // holds an infeasible (infinite-observation) trial. An over-limit latency
        // maps to +inf in the vector observation. The migration carries the
        // monotonic counter directly rather than deriving it from the retained
        // (capped) length, so evicted history cannot prevent the max_trials
        // stopping check from firing.
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
            engine.retained_trial_count().await,
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
                        engine.retained_trial_count().await <= cap,
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
        assert_eq!(engine.retained_trial_count().await, cap);
        engine.save(&path).await.unwrap();

        // A fresh bounded engine loads the checkpoint and must keep the cap.
        let mut cfg2 = single_objective_config("random");
        cfg2.max_leaderboard_size = Some(cap);
        let loaded = HolaEngine::from_config(cfg2).unwrap();
        loaded.load(&path).await.unwrap();
        assert_eq!(
            loaded.retained_trial_count().await,
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
            loaded.retained_trial_count().await,
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

        // Counters around the 6->7 digit boundary. Give newer sequences older
        // mtimes deliberately: rotation must trust the checkpoint sequence,
        // not mutable filesystem timestamps.
        let counters = [999_997usize, 999_998, 999_999, 1_000_000, 1_000_001];
        let mut paths = Vec::new();
        let base = std::time::SystemTime::UNIX_EPOCH;
        for (i, c) in counters.iter().enumerate() {
            let path = dir.path().join(format!("{prefix}_{c:06}.json"));
            std::fs::write(&path, b"{}").unwrap();
            let mtime = base + std::time::Duration::from_secs(((counters.len() - i) as u64) * 100);
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(mtime)
                .unwrap();
            paths.push((path, *c));
        }

        // Keep only the 2 highest snapshot sequences; the 3 oldest must be deleted.
        HolaEngine::rotate_checkpoints(dir.path(), prefix, 2);

        // The two newest sequence counters (1_000_000, 1_000_001) survive; the
        // three oldest are gone despite both lexical-width and mtime traps.
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
                        engine.retained_trial_count().await <= cap,
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
