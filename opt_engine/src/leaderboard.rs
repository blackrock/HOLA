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

//! Leaderboard: append-only trial history with lazy ranking.
//!
//! The leaderboard stores all `(candidate, observation)` pairs without maintaining
//! any particular ordering. Ranking operations (`top_k`, `pareto_front`) are computed
//! on-demand, making this efficient for persistence-only use cases.
//!
//! # Design Principles
//!
//! - **Append-only storage**: Trials are never reordered or removed
//! - **Lazy ranking**: Sorting/selection computed only when requested
//! - **Type-generic**: Works with any `Domain` and `Observation` types
//! - **Serializable**: Full history can be persisted and restored

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

// =============================================================================
// Core Data Structures
// =============================================================================

/// A single trial record.
///
/// Captures the candidate configuration, the observed result, and metadata.
/// Optionally stores the raw metrics (pre-scalarization) to support mid-run
/// objective weight changes and lazy re-scalarization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trial<D, Obs> {
    /// The candidate configuration that was evaluated.
    pub candidate: D,
    /// The observation/result from evaluating the candidate.
    pub observation: Obs,
    /// Raw metrics before scalarization (e.g., `{"loss": 0.3, "latency": 100}`).
    /// Stored when the engine receives JSON metrics from workers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_metrics: Option<serde_json::Value>,
    /// Unique identifier for this trial (monotonically increasing).
    pub trial_id: u64,
    /// Unix timestamp (seconds) when the trial was recorded.
    pub timestamp: u64,
}

impl<D, Obs> Trial<D, Obs> {
    fn new(candidate: D, observation: Obs, trial_id: u64) -> Self {
        Self {
            candidate,
            observation,
            raw_metrics: None,
            trial_id,
            timestamp: Utc::now().timestamp() as u64,
        }
    }

    fn with_raw_metrics(
        candidate: D,
        observation: Obs,
        raw_metrics: serde_json::Value,
        trial_id: u64,
    ) -> Self {
        Self {
            candidate,
            observation,
            raw_metrics: Some(raw_metrics),
            trial_id,
            timestamp: Utc::now().timestamp() as u64,
        }
    }
}

/// Append-only trial history with lazy ranking.
///
/// The leaderboard stores all trials in insertion order. No sorting or ranking
/// is performed until explicitly requested, making storage operations O(1).
///
/// # Example
///
/// ```
/// use opt_engine::leaderboard::Leaderboard;
///
/// let mut lb: Leaderboard<(f64, f64), f64> = Leaderboard::new();
///
/// // Append trials (O(1))
/// lb.push((0.1, 0.5), 0.42);
/// lb.push((0.2, 0.3), 0.35);
/// lb.push((0.15, 0.4), 0.38);
///
/// // Lazy ranking (computed on-demand)
/// let best = lb.top_k(2);  // Returns 2 trials with lowest observations
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    from = "LeaderboardData<D, Obs>",
    bound(deserialize = "D: Deserialize<'de>, Obs: Deserialize<'de>")
)]
pub struct Leaderboard<D, Obs> {
    /// Trials stored in insertion order (append-only).
    trials: Vec<Trial<D, Obs>>,
    /// Next trial ID to assign.
    next_id: u64,
    /// Optional cap on the number of stored trials. `None` (default) keeps the
    /// leaderboard unbounded.
    #[serde(default)]
    max_size: Option<usize>,
    /// Count of trials ever stored, incremented on every push and never
    /// decremented by eviction. For an unbounded leaderboard this equals
    /// `len()`; for a capped one it keeps growing past the cap. This is the
    /// correct basis for a `max_trials` stopping check.
    #[serde(default)]
    total_completed: u64,
    /// Maps `trial_id -> index` into `trials` for O(1) id lookups.
    ///
    /// Derived state: never serialized, and rebuilt from `trials` whenever the
    /// leaderboard is deserialized (see the `From<LeaderboardData>` impl) or
    /// mutated. It is always kept consistent with `trials`.
    #[serde(skip)]
    id_index: HashMap<u64, usize>,
}

/// Serialized shape of a [`Leaderboard`].
///
/// Deserialization targets this type and then rebuilds the derived
/// `id_index` via `From`. This keeps the on-disk format identical to a plain
/// `{ trials, next_id, max_size }` object and avoids persisting the index.
#[derive(Deserialize)]
#[serde(bound(deserialize = "D: Deserialize<'de>, Obs: Deserialize<'de>"))]
struct LeaderboardData<D, Obs> {
    trials: Vec<Trial<D, Obs>>,
    #[serde(default)]
    next_id: u64,
    #[serde(default)]
    max_size: Option<usize>,
    #[serde(default)]
    total_completed: u64,
}

impl<D, Obs> From<LeaderboardData<D, Obs>> for Leaderboard<D, Obs> {
    fn from(data: LeaderboardData<D, Obs>) -> Self {
        let id_index = build_id_index(&data.trials);
        // Old checkpoints lack the counter (default 0) and were always
        // unbounded, so total pushed equals stored len. Newer checkpoints
        // restore the real count, which can exceed len under a cap.
        let total_completed = data.total_completed.max(data.trials.len() as u64);
        Self {
            trials: data.trials,
            next_id: data.next_id,
            // Normalize a persisted Some(0) to unbounded, matching set_max_size.
            max_size: data.max_size.filter(|&cap| cap > 0),
            total_completed,
            id_index,
        }
    }
}

/// Build a fresh `trial_id -> index` map from a slice of trials.
///
/// On duplicate trial IDs the last occurrence wins, matching how `get` would
/// observe the most recently pushed trial.
fn build_id_index<D, Obs>(trials: &[Trial<D, Obs>]) -> HashMap<u64, usize> {
    let mut index = HashMap::with_capacity(trials.len());
    for (i, trial) in trials.iter().enumerate() {
        index.insert(trial.trial_id, i);
    }
    index
}

impl<D, Obs> Default for Leaderboard<D, Obs> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D, Obs> Leaderboard<D, Obs> {
    /// Create an empty leaderboard.
    pub fn new() -> Self {
        Self {
            trials: Vec::new(),
            next_id: 0,
            max_size: None,
            total_completed: 0,
            id_index: HashMap::new(),
        }
    }

    /// Create a leaderboard with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            trials: Vec::with_capacity(capacity),
            next_id: 0,
            max_size: None,
            total_completed: 0,
            id_index: HashMap::with_capacity(capacity),
        }
    }

    /// Set an optional maximum number of stored trials (builder form).
    ///
    /// `None` (the default) leaves the leaderboard unbounded. `Some(0)` is
    /// normalized to unbounded; see
    /// [`set_max_size`](Self::set_max_size) for the rationale and for the
    /// eviction policy applied when a positive cap is set.
    pub fn with_max_size(mut self, max_size: Option<usize>) -> Self {
        self.set_max_size(max_size);
        self
    }

    /// Set or clear the optional cap on the number of stored trials.
    ///
    /// # Eviction policy
    ///
    /// When a cap is set and a `push` would exceed it, the leaderboard evicts
    /// the single oldest trial (lowest index, i.e. earliest inserted). Oldest
    /// eviction is chosen because it is O(1) amortized for the scalar and
    /// vector cases alike and keeps `push` cheap; ranking helpers that the
    /// optimizer relies on (`top_k`, `pareto_front`, `rank_of`) all recompute
    /// from the retained set, so they stay correct over whatever window
    /// survives. Callers that need the global best preserved should leave the
    /// leaderboard unbounded.
    ///
    /// Setting a cap smaller than the current length immediately evicts the
    /// oldest trials down to the cap. Passing `None` clears the cap and never
    /// evicts. `next_id` is preserved so trial IDs remain unique across
    /// evictions.
    ///
    /// A cap of `Some(0)` is not a valid bound: a zero-capacity leaderboard
    /// could not even hold the trial a caller just pushed. It is therefore
    /// normalized to `None` (unbounded) here so every code path treats it the
    /// same way. This matches the `cap > 0` guard in
    /// [`enforce_max_size`](Self::enforce_max_size) and keeps `max_size()` from
    /// ever reporting a `Some(0)` that would not actually be enforced. The
    /// product layer rejects `Some(0)` during validation; this normalization is
    /// the library-level backstop.
    pub fn set_max_size(&mut self, max_size: Option<usize>) {
        // Treat Some(0) as unbounded (see method docs): a zero cap is invalid
        // and must never evict the just-pushed trial.
        let max_size = max_size.filter(|&cap| cap > 0);
        self.max_size = max_size;
        if let Some(cap) = max_size {
            self.evict_to(cap);
        }
    }

    /// Current cap on stored trials, if any.
    pub fn max_size(&self) -> Option<usize> {
        self.max_size
    }

    /// Evict the oldest trials until at most `cap` remain.
    ///
    /// Rebuilds the id index only when at least one trial is removed. With a
    /// single eviction per push the common path drains one element.
    fn evict_to(&mut self, cap: usize) {
        if self.trials.len() <= cap {
            return;
        }
        let remove = self.trials.len() - cap;
        self.trials.drain(0..remove);
        self.id_index = build_id_index(&self.trials);
    }

    /// Enforce `max_size` after a push, evicting the oldest trial if needed.
    ///
    /// A push adds exactly one trial, so at most one eviction is required.
    ///
    /// The `cap > 0` guard is a defensive backstop. `set_max_size` already
    /// normalizes `Some(0)` to `None`, so `self.max_size` should never be
    /// `Some(0)` here; the guard makes the no-evict-on-zero contract explicit
    /// and keeps a zero cap from ever dropping the just-pushed trial.
    fn enforce_max_size(&mut self) {
        if let Some(cap) = self.max_size
            && cap > 0
            && self.trials.len() > cap
        {
            self.evict_to(cap);
        }
    }

    /// Append a trial record, updating the id index and enforcing `max_size`.
    ///
    /// Index update is O(1); eviction (only when a cap is set and exceeded)
    /// removes one trial and rebuilds the index over the retained set.
    fn store_trial(&mut self, trial: Trial<D, Obs>) {
        let index = self.trials.len();
        self.id_index.insert(trial.trial_id, index);
        self.trials.push(trial);
        // Count every stored trial. Eviction below must not lower this, so it
        // tracks trials ever pushed rather than currently stored.
        self.total_completed += 1;
        self.enforce_max_size();
    }

    /// Number of trials ever stored, incremented on every push and never
    /// decremented by eviction.
    ///
    /// For an unbounded leaderboard this equals [`len`](Self::len). For a capped
    /// one it keeps growing past the cap as pushes continue. This is the correct
    /// basis for a `max_trials` stopping check, which must not be fooled by
    /// eviction shrinking the stored set.
    pub fn total_completed(&self) -> u64 {
        self.total_completed
    }

    /// Set the monotonic completed counter, clamping up to the number of
    /// currently stored trials.
    ///
    /// This is the mechanism for carrying the completed count across a
    /// leaderboard rebuild (e.g. an objective-topology migration that
    /// reconstructs the board with a different observation type). The rebuilt
    /// board starts with a fresh counter, so the caller transplants the prior
    /// count here rather than relying on a serialization round-trip.
    ///
    /// The value is clamped to `self.trials.len()` so the counter can never
    /// report fewer trials than are actually stored, preserving the invariant
    /// that `total_completed() >= len()`.
    pub fn set_total_completed(&mut self, value: u64) {
        self.total_completed = value.max(self.trials.len() as u64);
    }

    /// Append a new trial. Returns the assigned trial ID.
    ///
    /// This is an O(1) operation - no sorting is performed.
    pub fn push(&mut self, candidate: D, observation: Obs) -> u64 {
        let trial_id = self.next_id;
        self.next_id += 1;
        self.store_trial(Trial::new(candidate, observation, trial_id));
        trial_id
    }

    /// Append a trial using an externally assigned trial ID.
    ///
    /// This is useful when trial IDs are issued before completion. The internal
    /// auto-assignment counter is advanced so future generated IDs do not reuse
    /// the provided ID.
    pub fn push_with_trial_id(&mut self, candidate: D, observation: Obs, trial_id: u64) -> u64 {
        self.next_id = self.next_id.max(trial_id.saturating_add(1));
        self.store_trial(Trial::new(candidate, observation, trial_id));
        trial_id
    }

    /// Append a trial with raw metrics preserved for lazy re-scalarization.
    pub fn push_with_raw(
        &mut self,
        candidate: D,
        observation: Obs,
        raw_metrics: serde_json::Value,
    ) -> u64 {
        let trial_id = self.next_id;
        self.next_id += 1;
        self.store_trial(Trial::with_raw_metrics(
            candidate,
            observation,
            raw_metrics,
            trial_id,
        ));
        trial_id
    }

    /// Append a trial with an externally assigned trial ID and raw metrics
    /// preserved for lazy re-scalarization.
    pub fn push_with_raw_trial_id(
        &mut self,
        candidate: D,
        observation: Obs,
        raw_metrics: serde_json::Value,
        trial_id: u64,
    ) -> u64 {
        self.next_id = self.next_id.max(trial_id.saturating_add(1));
        self.store_trial(Trial::with_raw_metrics(
            candidate,
            observation,
            raw_metrics,
            trial_id,
        ));
        trial_id
    }

    /// Append an existing trial record, preserving its ID and timestamp.
    ///
    /// This supports rebuilding a leaderboard with a different observation type
    /// while preserving public trial identity and completion metadata.
    pub fn push_existing_trial(&mut self, trial: Trial<D, Obs>) -> u64 {
        let trial_id = trial.trial_id;
        self.next_id = self.next_id.max(trial_id.saturating_add(1));
        self.store_trial(trial);
        trial_id
    }

    /// Return the next ID that can be assigned without reusing a stored trial ID.
    pub fn next_trial_id(&self) -> u64 {
        let next_from_trials = self
            .trials
            .iter()
            .map(|trial| trial.trial_id.saturating_add(1))
            .max()
            .unwrap_or(0);
        self.next_id.max(next_from_trials)
    }

    /// Repair the internal next-ID counter after deserializing or manually
    /// modifying a leaderboard.
    pub fn normalize_next_trial_id(&mut self) -> u64 {
        let next_id = self.next_trial_id();
        self.next_id = next_id;
        next_id
    }

    pub fn len(&self) -> usize {
        self.trials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trials.is_empty()
    }

    /// Access all trials in insertion order (no sorting).
    pub fn trials(&self) -> &[Trial<D, Obs>] {
        &self.trials
    }

    /// Re-compute observations for all trials that have raw_metrics.
    ///
    /// This enables mid-run objective changes: update the scoring function,
    /// call `rescalarize`, and the leaderboard reflects the new priorities.
    pub fn rescalarize<F>(&mut self, scalarizer: F)
    where
        F: Fn(&serde_json::Value) -> Option<Obs>,
    {
        for trial in &mut self.trials {
            if let Some(ref raw) = trial.raw_metrics
                && let Some(score) = scalarizer(raw)
            {
                trial.observation = score;
            }
        }
    }

    /// Get a trial by its ID.
    ///
    /// O(1): backed by an internal `trial_id -> index` map kept in sync with
    /// `trials` on every push, eviction, and after deserialization.
    pub fn get(&self, trial_id: u64) -> Option<&Trial<D, Obs>> {
        let &index = self.id_index.get(&trial_id)?;
        self.trials.get(index)
    }

    /// Return whether a trial with the given ID is stored.
    ///
    /// O(1): backed by the same internal id index as [`get`](Self::get).
    pub fn contains_trial_id(&self, trial_id: u64) -> bool {
        self.id_index.contains_key(&trial_id)
    }

    /// Iterator over all trials in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &Trial<D, Obs>> {
        self.trials.iter()
    }

    pub fn last(&self) -> Option<&Trial<D, Obs>> {
        self.trials.last()
    }

    pub fn clear(&mut self) {
        self.trials.clear();
        self.id_index.clear();
        self.total_completed = 0;
        // Note: we don't reset next_id to preserve uniqueness across clears
    }
}

// =============================================================================
// Feasibility Helpers
// =============================================================================

/// Check if a scalar observation is feasible (finite).
///
/// An observation of `f64::INFINITY` or `f64::NEG_INFINITY` indicates
/// an infeasible solution (e.g., constraint violation in TLP objectives).
#[inline]
pub fn is_feasible_scalar(observation: f64) -> bool {
    observation.is_finite()
}

/// Check if a multi-objective observation is feasible (all values finite).
///
/// Any objective with `f64::INFINITY` indicates an infeasible solution.
#[inline]
pub fn is_feasible_multi(observation: &BTreeMap<String, f64>) -> bool {
    observation.values().all(|v| v.is_finite())
}

// =============================================================================
// Scalar Ranking (f64 observations)
// =============================================================================

impl<D: Clone> Leaderboard<D, f64> {
    /// Check if a trial is feasible (finite observation).
    #[inline]
    pub fn trial_is_feasible(trial: &Trial<D, f64>) -> bool {
        is_feasible_scalar(trial.observation)
    }

    fn compare_best(a: &Trial<D, f64>, b: &Trial<D, f64>) -> Ordering {
        a.observation
            .partial_cmp(&b.observation)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.trial_id.cmp(&b.trial_id))
    }

    fn compare_worst(a: &Trial<D, f64>, b: &Trial<D, f64>) -> Ordering {
        b.observation
            .partial_cmp(&a.observation)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.trial_id.cmp(&b.trial_id))
    }

    /// Return only feasible trials (those with finite observations).
    pub fn feasible_trials(&self) -> Vec<Trial<D, f64>> {
        self.trials
            .iter()
            .filter(|t| Self::trial_is_feasible(t))
            .cloned()
            .collect()
    }

    pub fn feasible_count(&self) -> usize {
        self.trials
            .iter()
            .filter(|t| Self::trial_is_feasible(t))
            .count()
    }

    /// Return the k trials with the lowest (best) observations.
    ///
    /// Infeasible trials (those with infinite observations) are excluded.
    /// Use `top_k_all()` to include them.
    ///
    /// Ties are broken by trial_id (earlier trials preferred).
    pub fn top_k(&self, k: usize) -> Vec<Trial<D, f64>> {
        if self.trials.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut feasible: Vec<&Trial<D, f64>> = self
            .trials
            .iter()
            .filter(|t| Self::trial_is_feasible(t))
            .collect();

        let limit = k.min(feasible.len());
        if limit < feasible.len() {
            feasible.select_nth_unstable_by(limit, |a, b| Self::compare_best(a, b));
            feasible.truncate(limit);
        }
        feasible.sort_by(|a, b| Self::compare_best(a, b));

        feasible.into_iter().cloned().collect()
    }

    /// Return the k trials with the lowest observations, including infeasible.
    pub fn top_k_all(&self, k: usize) -> Vec<Trial<D, f64>> {
        if self.trials.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut trials: Vec<&Trial<D, f64>> = self.trials.iter().collect();
        let limit = k.min(trials.len());
        if limit < trials.len() {
            trials.select_nth_unstable_by(limit, |a, b| Self::compare_best(a, b));
            trials.truncate(limit);
        }
        trials.sort_by(|a, b| Self::compare_best(a, b));

        trials.into_iter().cloned().collect()
    }

    /// Return feasible trials sorted by observation (ascending).
    pub fn sorted(&self) -> Vec<Trial<D, f64>> {
        self.top_k(self.feasible_count())
    }

    /// Return all trials sorted by observation, including infeasible.
    pub fn sorted_all(&self) -> Vec<Trial<D, f64>> {
        self.top_k_all(self.len())
    }

    /// Return the trial with the best (lowest) observation.
    ///
    /// Returns `None` if no feasible trials exist.
    pub fn best(&self) -> Option<Trial<D, f64>> {
        self.top_k(1).into_iter().next()
    }

    /// Return the best trial including infeasible ones.
    pub fn best_all(&self) -> Option<Trial<D, f64>> {
        self.top_k_all(1).into_iter().next()
    }

    /// Return the k trials with the highest (worst) observations.
    ///
    /// Excludes infeasible trials. Use `bottom_k_all()` to include them.
    pub fn bottom_k(&self, k: usize) -> Vec<Trial<D, f64>> {
        if self.trials.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut feasible: Vec<&Trial<D, f64>> = self
            .trials
            .iter()
            .filter(|t| Self::trial_is_feasible(t))
            .collect();

        let limit = k.min(feasible.len());
        if limit < feasible.len() {
            feasible.select_nth_unstable_by(limit, |a, b| Self::compare_worst(a, b));
            feasible.truncate(limit);
        }
        feasible.sort_by(|a, b| Self::compare_worst(a, b));

        feasible.into_iter().cloned().collect()
    }

    /// Return the k worst trials, including infeasible.
    pub fn bottom_k_all(&self, k: usize) -> Vec<Trial<D, f64>> {
        if self.trials.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut trials: Vec<&Trial<D, f64>> = self.trials.iter().collect();
        let limit = k.min(trials.len());
        if limit < trials.len() {
            trials.select_nth_unstable_by(limit, |a, b| Self::compare_worst(a, b));
            trials.truncate(limit);
        }
        trials.sort_by(|a, b| Self::compare_worst(a, b));

        trials.into_iter().cloned().collect()
    }

    /// Compute the quantile threshold for feasible observations.
    ///
    /// `quantile` should be in [0, 1]. Returns the observation value at that quantile.
    /// E.g., `quantile_threshold(0.25)` returns the value below which ~25% of observations fall.
    ///
    /// This is a **nearest-rank** (non-interpolated) estimator: it rounds
    /// `quantile * (n - 1)` to the closest integer rank and returns the
    /// observation at that rank. It does not interpolate between adjacent
    /// observations, so the result may differ by one rank from a
    /// linear-interpolation quantile estimator (e.g. NumPy's default).
    pub fn quantile_threshold(&self, quantile: f64) -> Option<f64> {
        let mut observations: Vec<f64> = self
            .trials
            .iter()
            .filter(|t| Self::trial_is_feasible(t))
            .map(|t| t.observation)
            .collect();

        if observations.is_empty() {
            return None;
        }

        observations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((observations.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
        Some(observations[index])
    }

    /// Return feasible trials in the top quantile (e.g., top 25% if quantile = 0.25).
    pub fn top_quantile(&self, quantile: f64) -> Vec<Trial<D, f64>> {
        let threshold = match self.quantile_threshold(quantile) {
            Some(t) => t,
            None => return Vec::new(),
        };

        self.trials
            .iter()
            .filter(|t| Self::trial_is_feasible(t) && t.observation <= threshold)
            .cloned()
            .collect()
    }

    /// Compute the 0-indexed rank of a single trial without sorting or cloning
    /// the whole leaderboard.
    ///
    /// The rank is the number of stored trials that sort strictly before the
    /// target under the same ordering as [`top_k`](Self::top_k) /
    /// [`sorted`](Self::sorted): observation ascending, ties broken by
    /// `trial_id` (earlier first). A return of `0` means best.
    ///
    /// When `include_infeasible` is `false`, infeasible trials are ignored when
    /// counting and the method returns `None` if the target itself is
    /// infeasible. When `true`, all trials participate (matching
    /// [`sorted_all`](Self::sorted_all)).
    ///
    /// Complexity: O(n) over a single pass, with no allocation and no clones.
    /// Returns `None` if no trial with `trial_id` exists.
    pub fn rank_of(&self, trial_id: u64, include_infeasible: bool) -> Option<usize> {
        let target = self.get(trial_id)?;
        if !include_infeasible && !Self::trial_is_feasible(target) {
            return None;
        }
        let rank = self
            .trials
            .iter()
            .filter(|other| include_infeasible || Self::trial_is_feasible(other))
            .filter(|other| Self::compare_best(other, target) == Ordering::Less)
            .count();
        Some(rank)
    }
}

// =============================================================================
// Multi-Objective Ranking (BTreeMap<String, f64> observations)
// =============================================================================

type ScoredMultiTrial<'a, D> = (&'a Trial<D, BTreeMap<String, f64>>, f64);

impl<D: Clone> Leaderboard<D, BTreeMap<String, f64>> {
    /// Check if a trial is feasible (all objective values are finite).
    ///
    /// A trial with any `f64::INFINITY` value is considered infeasible
    /// (typically indicating constraint violation).
    #[inline]
    pub fn trial_is_feasible(trial: &Trial<D, BTreeMap<String, f64>>) -> bool {
        is_feasible_multi(&trial.observation)
    }

    /// Return only feasible trials (those with all finite objective values).
    pub fn feasible_trials(&self) -> Vec<Trial<D, BTreeMap<String, f64>>> {
        self.trials
            .iter()
            .filter(|t| Self::trial_is_feasible(t))
            .cloned()
            .collect()
    }

    pub fn feasible_count(&self) -> usize {
        self.trials
            .iter()
            .filter(|t| Self::trial_is_feasible(t))
            .count()
    }

    /// Check if trial `a` dominates trial `b` (all objectives better or equal, at least one strictly better).
    ///
    /// Assumes minimization for all objectives.
    ///
    /// NaN objective values are treated as strictly worst (worse than any
    /// finite value or infinity), so that ordering stays deterministic:
    /// - a NaN in `a` but finite in `b` makes `a` worse in that objective, so
    ///   `a` cannot dominate;
    /// - a NaN in `b` but finite in `a` counts as `a` being strictly better;
    /// - both NaN is a tie in that objective.
    fn dominates(a: &BTreeMap<String, f64>, b: &BTreeMap<String, f64>) -> bool {
        let mut dominated_some = false;
        for key in a.keys() {
            let va = a.get(key).copied().unwrap_or(f64::INFINITY);
            let vb = b.get(key).copied().unwrap_or(f64::INFINITY);
            // Deterministic NaN policy: NaN is strictly worst. Handle the cases
            // involving NaN explicitly because `>` / `<` are always false for
            // NaN and would otherwise be silently treated as a tie.
            match (va.is_nan(), vb.is_nan()) {
                (true, false) => return false, // a is worst here, cannot dominate
                (false, true) => {
                    dominated_some = true; // a is strictly better than NaN
                    continue;
                }
                (true, true) => continue, // both NaN: tie in this objective
                (false, false) => {}
            }
            if va > vb {
                return false; // a is worse in this objective
            }
            if va < vb {
                dominated_some = true;
            }
        }
        // Check if b has keys that a doesn't (a would have infinity there)
        for key in b.keys() {
            if !a.contains_key(key) {
                return false; // a is worse (infinity) in this objective
            }
        }
        dominated_some
    }

    /// Compute the Pareto front (non-dominated feasible trials).
    ///
    /// Infeasible trials (those with any infinite objective) are excluded.
    /// Use `pareto_front_all()` to include them.
    ///
    /// A trial is on the Pareto front if no other trial dominates it
    /// (i.e., is better or equal in all objectives and strictly better in at least one).
    pub fn pareto_front(&self) -> Vec<Trial<D, BTreeMap<String, f64>>> {
        let feasible = self.feasible_trials();
        if feasible.is_empty() {
            return Vec::new();
        }

        let mut front = Vec::new();

        for trial in &feasible {
            let dominated_by_front = front.iter().any(|f: &Trial<D, BTreeMap<String, f64>>| {
                Self::dominates(&f.observation, &trial.observation)
            });

            if dominated_by_front {
                continue;
            }

            front.retain(|f: &Trial<D, BTreeMap<String, f64>>| {
                !Self::dominates(&trial.observation, &f.observation)
            });
            front.push(trial.clone());
        }

        front
    }

    /// Compute the Pareto front including infeasible trials.
    pub fn pareto_front_all(&self) -> Vec<Trial<D, BTreeMap<String, f64>>> {
        if self.trials.is_empty() {
            return Vec::new();
        }

        let mut front = Vec::new();

        for trial in &self.trials {
            let dominated_by_front = front.iter().any(|f: &Trial<D, BTreeMap<String, f64>>| {
                Self::dominates(&f.observation, &trial.observation)
            });

            if dominated_by_front {
                continue;
            }

            front.retain(|f: &Trial<D, BTreeMap<String, f64>>| {
                !Self::dominates(&trial.observation, &f.observation)
            });
            front.push(trial.clone());
        }

        front
    }

    /// Compute the Pareto front considering only specific objectives.
    ///
    /// Only considers feasible trials. Useful when you want to focus on a subset of objectives.
    pub fn pareto_front_for(&self, groups: &[&str]) -> Vec<Trial<D, BTreeMap<String, f64>>> {
        let feasible = self.feasible_trials();
        if feasible.is_empty() || groups.is_empty() {
            return Vec::new();
        }

        let project = |obs: &BTreeMap<String, f64>| -> BTreeMap<String, f64> {
            groups
                .iter()
                .filter_map(|&g| obs.get(g).map(|&v| (g.to_string(), v)))
                .collect()
        };

        let mut front = Vec::new();

        for trial in &feasible {
            let proj = project(&trial.observation);

            let dominated_by_front = front
                .iter()
                .any(|(_, f_proj): &(_, BTreeMap<String, f64>)| Self::dominates(f_proj, &proj));

            if dominated_by_front {
                continue;
            }

            front
                .retain(|(_, f_proj): &(_, BTreeMap<String, f64>)| !Self::dominates(&proj, f_proj));
            front.push((trial.clone(), proj));
        }

        front.into_iter().map(|(t, _)| t).collect()
    }

    /// Scalarize multi-objective observations and return top-k feasible trials.
    ///
    /// The scalarizer function converts `BTreeMap<String, f64>` to a single `f64`.
    /// Common choices: weighted sum, Chebyshev distance to ideal point, etc.
    pub fn top_k_scalarized<F>(
        &self,
        k: usize,
        scalarizer: F,
    ) -> Vec<Trial<D, BTreeMap<String, f64>>>
    where
        F: Fn(&BTreeMap<String, f64>) -> f64,
    {
        if self.trials.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut scored: Vec<ScoredMultiTrial<'_, D>> = self
            .trials
            .iter()
            .filter(|t| Self::trial_is_feasible(t))
            .map(|t| (t, scalarizer(&t.observation)))
            .collect();

        if scored.is_empty() {
            return Vec::new();
        }

        let compare = |a: &ScoredMultiTrial<'_, D>, b: &ScoredMultiTrial<'_, D>| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.trial_id.cmp(&b.0.trial_id))
        };

        let limit = k.min(scored.len());
        if limit < scored.len() {
            scored.select_nth_unstable_by(limit, compare);
            scored.truncate(limit);
        }
        scored.sort_by(compare);

        scored
            .into_iter()
            .map(|(trial, _)| (*trial).clone())
            .collect()
    }

    /// Return trials on the Pareto front, sorted by a scalarizer.
    ///
    /// Useful for getting an ordered list of Pareto-optimal solutions.
    pub fn pareto_front_sorted<F>(&self, scalarizer: F) -> Vec<Trial<D, BTreeMap<String, f64>>>
    where
        F: Fn(&BTreeMap<String, f64>) -> f64,
    {
        let mut front = self.pareto_front();
        front.sort_by(|a, b| {
            let sa = scalarizer(&a.observation);
            let sb = scalarizer(&b.observation);
            sa.partial_cmp(&sb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.trial_id.cmp(&b.trial_id))
        });
        front
    }

    /// Get the best feasible trial for a single objective (ignoring others).
    pub fn best_for_objective(&self, group: &str) -> Option<Trial<D, BTreeMap<String, f64>>> {
        self.feasible_trials()
            .into_iter()
            .filter(|t| t.observation.contains_key(group))
            .min_by(|a, b| {
                let va = a.observation.get(group).unwrap();
                let vb = b.observation.get(group).unwrap();
                va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Compute the 0-indexed Pareto front membership of a single trial without
    /// cloning trials or running a full non-dominated sort of every front.
    ///
    /// The result is the front index the trial belongs to under the same
    /// non-domination relation as [`non_dominated_sort`](Self::non_dominated_sort)
    /// (`0` = Pareto front, `1` = second front, etc.), which equals the
    /// `rank - 1` that `ranked_trials` would assign. It is computed by peeling
    /// fronts and stopping as soon as the target is placed, so it never has to
    /// rank trials in later fronts and clones nothing (it works on observation
    /// references and indices only).
    ///
    /// When `include_infeasible` is `false`, only feasible trials participate
    /// and the method returns `None` if the target is infeasible. When `true`,
    /// all trials participate (matching the `*_all` variants).
    ///
    /// Complexity: O(M * N^2) worst case like the full sort, but typically far
    /// less since front peeling halts once the target lands. Returns `None` if
    /// no trial with `trial_id` exists.
    pub fn pareto_rank_of(&self, trial_id: u64, include_infeasible: bool) -> Option<usize> {
        let target = self.get(trial_id)?;
        if !include_infeasible && !Self::trial_is_feasible(target) {
            return None;
        }

        // Work on references to the participating observations to avoid clones.
        let participants: Vec<(u64, &BTreeMap<String, f64>)> = self
            .trials
            .iter()
            .filter(|t| include_infeasible || Self::trial_is_feasible(t))
            .map(|t| (t.trial_id, &t.observation))
            .collect();

        let n = participants.len();
        let mut domination_count: Vec<usize> = vec![0; n];
        let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            for j in (i + 1)..n {
                if Self::dominates(participants[i].1, participants[j].1) {
                    dominated_by[i].push(j);
                    domination_count[j] += 1;
                } else if Self::dominates(participants[j].1, participants[i].1) {
                    dominated_by[j].push(i);
                    domination_count[i] += 1;
                }
            }
        }

        let mut current: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();
        let mut front = 0usize;
        while !current.is_empty() {
            if current.iter().any(|&i| participants[i].0 == trial_id) {
                return Some(front);
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
        None
    }

    // =========================================================================
    // NSGA-II: Non-dominated Sorting and Crowding Distance
    // =========================================================================

    /// Perform NSGA-II non-dominated sorting on feasible trials.
    ///
    /// Returns trials grouped by rank (front). `fronts[0]` is the Pareto front (rank 1),
    /// `fronts[1]` is rank 2, etc. Infeasible trials are excluded.
    ///
    /// Use `non_dominated_sort_all()` to include infeasible trials.
    ///
    /// # Algorithm
    ///
    /// Fast non-dominated sort from Deb et al. (2002):
    /// 1. For each trial, count how many trials dominate it
    /// 2. Trials with count 0 form the first front
    /// 3. Remove first front, repeat to find subsequent fronts
    ///
    /// Complexity: O(M * N²) where M = objectives, N = trials
    pub fn non_dominated_sort(&self) -> Vec<Vec<Trial<D, BTreeMap<String, f64>>>> {
        let feasible = self.feasible_trials();
        if feasible.is_empty() {
            return Vec::new();
        }

        let n = feasible.len();

        let mut domination_count: Vec<usize> = vec![0; n];
        let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            for j in (i + 1)..n {
                let obs_i = &feasible[i].observation;
                let obs_j = &feasible[j].observation;

                if Self::dominates(obs_i, obs_j) {
                    dominated_by[i].push(j);
                    domination_count[j] += 1;
                } else if Self::dominates(obs_j, obs_i) {
                    dominated_by[j].push(i);
                    domination_count[i] += 1;
                }
            }
        }

        let mut fronts: Vec<Vec<Trial<D, BTreeMap<String, f64>>>> = Vec::new();
        let mut current_front_indices: Vec<usize> =
            (0..n).filter(|&i| domination_count[i] == 0).collect();

        while !current_front_indices.is_empty() {
            let front: Vec<Trial<D, BTreeMap<String, f64>>> = current_front_indices
                .iter()
                .map(|&i| feasible[i].clone())
                .collect();
            fronts.push(front);

            let mut next_front_indices = Vec::new();
            for &i in &current_front_indices {
                for &j in &dominated_by[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front_indices.push(j);
                    }
                }
            }
            current_front_indices = next_front_indices;
        }

        fronts
    }

    /// Perform NSGA-II non-dominated sorting including infeasible trials.
    pub fn non_dominated_sort_all(&self) -> Vec<Vec<Trial<D, BTreeMap<String, f64>>>> {
        if self.trials.is_empty() {
            return Vec::new();
        }

        let n = self.trials.len();

        let mut domination_count: Vec<usize> = vec![0; n];
        let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            for j in (i + 1)..n {
                let obs_i = &self.trials[i].observation;
                let obs_j = &self.trials[j].observation;

                if Self::dominates(obs_i, obs_j) {
                    dominated_by[i].push(j);
                    domination_count[j] += 1;
                } else if Self::dominates(obs_j, obs_i) {
                    dominated_by[j].push(i);
                    domination_count[i] += 1;
                }
            }
        }

        let mut fronts: Vec<Vec<Trial<D, BTreeMap<String, f64>>>> = Vec::new();
        let mut current_front_indices: Vec<usize> =
            (0..n).filter(|&i| domination_count[i] == 0).collect();

        while !current_front_indices.is_empty() {
            let front: Vec<Trial<D, BTreeMap<String, f64>>> = current_front_indices
                .iter()
                .map(|&i| self.trials[i].clone())
                .collect();
            fronts.push(front);

            let mut next_front_indices = Vec::new();
            for &i in &current_front_indices {
                for &j in &dominated_by[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front_indices.push(j);
                    }
                }
            }
            current_front_indices = next_front_indices;
        }

        fronts
    }

    /// Calculate crowding distance for a set of trials.
    ///
    /// Crowding distance measures how spread out a trial is relative to its neighbors
    /// in objective space. Higher values indicate more isolated (diverse) solutions.
    ///
    /// # Algorithm
    ///
    /// For each objective:
    /// 1. Sort trials by that objective
    /// 2. Boundary trials get infinite distance
    /// 3. Interior trials get distance = (neighbor_above - neighbor_below) / range
    /// 4. Sum distances across all objectives
    ///
    /// # Returns
    ///
    /// Vector of (trial, crowding_distance) pairs in the same order as input.
    #[allow(clippy::type_complexity)]
    pub fn crowding_distance(
        trials: &[Trial<D, BTreeMap<String, f64>>],
    ) -> Vec<(Trial<D, BTreeMap<String, f64>>, f64)> {
        if trials.is_empty() {
            return Vec::new();
        }

        let n = trials.len();
        let mut distances: Vec<f64> = vec![0.0; n];

        let objectives: Vec<&String> = trials
            .first()
            .map(|t| t.observation.keys().collect())
            .unwrap_or_default();

        if objectives.is_empty() {
            return trials.iter().cloned().map(|t| (t, 0.0)).collect();
        }

        for obj in &objectives {
            let mut indexed: Vec<(usize, f64)> = trials
                .iter()
                .enumerate()
                .map(|(i, t)| (i, t.observation.get(*obj).copied().unwrap_or(f64::INFINITY)))
                .collect();

            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Compute range from only finite values to avoid NaN from Inf - Inf
            let finite_vals: Vec<f64> = indexed
                .iter()
                .map(|(_, v)| *v)
                .filter(|v| v.is_finite())
                .collect();
            let min_val = finite_vals.first().copied().unwrap_or(0.0);
            let max_val = finite_vals.last().copied().unwrap_or(0.0);
            let range = max_val - min_val;

            if let Some(&(idx, _)) = indexed.first() {
                distances[idx] = f64::INFINITY;
            }
            if let Some(&(idx, _)) = indexed.last() {
                distances[idx] = f64::INFINITY;
            }

            if range.is_finite() && range > 0.0 && n > 2 {
                for i in 1..(indexed.len() - 1) {
                    let (idx, val) = indexed[i];
                    let (_, prev_val) = indexed[i - 1];
                    let (_, next_val) = indexed[i + 1];

                    // Assign zero distance to infeasible trials instead of computing NaN
                    if !val.is_finite() || !prev_val.is_finite() || !next_val.is_finite() {
                        continue;
                    }

                    if distances[idx].is_finite() {
                        distances[idx] += (next_val - prev_val) / range;
                    }
                }
            }
        }

        trials.iter().cloned().zip(distances).collect()
    }

    /// Select top-k feasible trials using NSGA-II criteria.
    ///
    /// Selection is based on:
    /// 1. **Rank** (lower is better) - trials from earlier fronts are preferred
    /// 2. **Crowding distance** (higher is better) - more diverse trials are preferred
    ///
    /// Infeasible trials are excluded. Use `select_nsga2_all()` to include them.
    pub fn select_nsga2(&self, k: usize) -> Vec<RankedTrial<D>> {
        if self.feasible_count() == 0 || k == 0 {
            return Vec::new();
        }

        let fronts = self.non_dominated_sort();
        let mut selected: Vec<RankedTrial<D>> = Vec::with_capacity(k.min(self.feasible_count()));

        for (rank, front) in fronts.into_iter().enumerate() {
            if selected.len() >= k {
                break;
            }

            let remaining = k - selected.len();

            if front.len() <= remaining {
                let with_distance = Self::crowding_distance(&front);
                for (trial, distance) in with_distance {
                    selected.push(RankedTrial {
                        trial,
                        rank: rank + 1,
                        crowding_distance: distance,
                    });
                }
            } else {
                let mut with_distance = Self::crowding_distance(&front);

                with_distance
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (trial, distance) in with_distance.into_iter().take(remaining) {
                    selected.push(RankedTrial {
                        trial,
                        rank: rank + 1,
                        crowding_distance: distance,
                    });
                }
            }
        }

        selected
    }

    /// Select top-k trials using NSGA-II criteria, including infeasible.
    pub fn select_nsga2_all(&self, k: usize) -> Vec<RankedTrial<D>> {
        if self.trials.is_empty() || k == 0 {
            return Vec::new();
        }

        let fronts = self.non_dominated_sort_all();
        let mut selected: Vec<RankedTrial<D>> = Vec::with_capacity(k.min(self.trials.len()));

        for (rank, front) in fronts.into_iter().enumerate() {
            if selected.len() >= k {
                break;
            }

            let remaining = k - selected.len();

            if front.len() <= remaining {
                let with_distance = Self::crowding_distance(&front);
                for (trial, distance) in with_distance {
                    selected.push(RankedTrial {
                        trial,
                        rank: rank + 1,
                        crowding_distance: distance,
                    });
                }
            } else {
                let mut with_distance = Self::crowding_distance(&front);

                with_distance
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (trial, distance) in with_distance.into_iter().take(remaining) {
                    selected.push(RankedTrial {
                        trial,
                        rank: rank + 1,
                        crowding_distance: distance,
                    });
                }
            }
        }

        selected
    }

    /// Get the Pareto front with crowding distances.
    ///
    /// Returns feasible trials on the Pareto front along with their crowding distances,
    /// sorted by crowding distance (most isolated first).
    pub fn pareto_front_with_crowding(&self) -> Vec<RankedTrial<D>> {
        let front = self.pareto_front();
        if front.is_empty() {
            return Vec::new();
        }

        let mut with_distance = Self::crowding_distance(&front);

        with_distance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        with_distance
            .into_iter()
            .map(|(trial, distance)| RankedTrial {
                trial,
                rank: 1,
                crowding_distance: distance,
            })
            .collect()
    }

    /// Get all feasible trials with NSGA-II ranking information.
    pub fn ranked_trials(&self) -> Vec<RankedTrial<D>> {
        self.select_nsga2(self.feasible_count())
    }

    /// Get all trials (including infeasible) with NSGA-II ranking information.
    pub fn ranked_trials_all(&self) -> Vec<RankedTrial<D>> {
        self.select_nsga2_all(self.trials.len())
    }
}

// =============================================================================
// NSGA-II Support Types
// =============================================================================

/// A trial annotated with NSGA-II ranking information.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankedTrial<D> {
    /// The underlying trial.
    pub trial: Trial<D, BTreeMap<String, f64>>,
    /// Non-domination rank (1 = Pareto front, 2 = second front, etc.).
    pub rank: usize,
    /// Crowding distance (higher = more isolated/diverse).
    /// Boundary points have `f64::INFINITY`.
    pub crowding_distance: f64,
}

impl<D> RankedTrial<D> {
    /// Compare two ranked trials using NSGA-II crowded comparison.
    ///
    /// Returns `Ordering::Less` if `self` is preferred over `other`.
    /// Prefers lower rank, then higher crowding distance.
    pub fn crowded_compare(&self, other: &Self) -> std::cmp::Ordering {
        // Lower rank is better
        match self.rank.cmp(&other.rank) {
            std::cmp::Ordering::Less => std::cmp::Ordering::Less,
            std::cmp::Ordering::Greater => std::cmp::Ordering::Greater,
            std::cmp::Ordering::Equal => {
                // Same rank: higher crowding distance is better
                other
                    .crowding_distance
                    .partial_cmp(&self.crowding_distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_len() {
        let mut lb: Leaderboard<f64, f64> = Leaderboard::new();
        assert!(lb.is_empty());

        lb.push(0.5, 0.1);
        lb.push(0.3, 0.2);
        lb.push(0.7, 0.05);

        assert_eq!(lb.len(), 3);
        assert!(!lb.is_empty());
    }

    #[test]
    fn test_trial_ids_monotonic() {
        let mut lb: Leaderboard<f64, f64> = Leaderboard::new();

        let id1 = lb.push(0.1, 0.1);
        let id2 = lb.push(0.2, 0.2);
        let id3 = lb.push(0.3, 0.3);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
    }

    #[test]
    fn test_externally_assigned_trial_ids_advance_next_id() {
        let mut lb: Leaderboard<f64, f64> = Leaderboard::new();

        assert_eq!(lb.push_with_trial_id(0.1, 0.1, 7), 7);
        assert_eq!(lb.push(0.2, 0.2), 8);
        assert_eq!(lb.next_trial_id(), 9);
    }

    #[test]
    fn test_normalize_next_trial_id_repairs_stale_counter() {
        let mut lb: Leaderboard<f64, f64> = Leaderboard::new();
        lb.push(0.1, 0.1);
        lb.push(0.2, 0.2);
        lb.next_id = 0;

        assert_eq!(lb.next_trial_id(), 2);
        assert_eq!(lb.normalize_next_trial_id(), 2);
        assert_eq!(lb.push(0.3, 0.3), 2);
    }

    #[test]
    fn test_push_existing_trial_preserves_timestamp_and_advances_next_id() {
        let mut lb: Leaderboard<f64, f64> = Leaderboard::new();
        let trial = Trial {
            candidate: 0.1,
            observation: 0.2,
            raw_metrics: Some(serde_json::json!({"loss": 0.2})),
            trial_id: 4,
            timestamp: 123,
        };

        assert_eq!(lb.push_existing_trial(trial), 4);
        let stored = lb.get(4).unwrap();
        assert_eq!(stored.timestamp, 123);
        assert_eq!(lb.push(0.3, 0.3), 5);
    }

    #[test]
    fn test_set_total_completed_sets_and_clamps() {
        let mut lb: Leaderboard<f64, f64> = Leaderboard::new();
        lb.push(0.1, 0.1);
        lb.push(0.2, 0.2);
        assert_eq!(lb.total_completed(), 2);

        // Carries a larger prior count onto the board (the rebuild case).
        lb.set_total_completed(10);
        assert_eq!(lb.total_completed(), 10);

        // Clamps up to len() when given a value below the stored count,
        // never reporting fewer trials than are actually stored.
        lb.set_total_completed(1);
        assert_eq!(lb.total_completed(), 2);
    }

    #[test]
    fn test_top_k_scalar() {
        let mut lb: Leaderboard<&str, f64> = Leaderboard::new();

        lb.push("worst", 1.0);
        lb.push("best", 0.1);
        lb.push("middle", 0.5);

        let top2 = lb.top_k(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].candidate, "best");
        assert_eq!(top2[1].candidate, "middle");
    }

    #[test]
    fn test_top_k_with_ties() {
        let mut lb: Leaderboard<&str, f64> = Leaderboard::new();

        lb.push("first", 0.5);
        lb.push("second", 0.5);
        lb.push("third", 0.5);

        // Ties broken by trial_id (earlier first)
        let top2 = lb.top_k(2);
        assert_eq!(top2[0].candidate, "first");
        assert_eq!(top2[1].candidate, "second");
    }

    #[test]
    fn test_best() {
        let mut lb: Leaderboard<&str, f64> = Leaderboard::new();

        lb.push("a", 0.5);
        lb.push("b", 0.1);
        lb.push("c", 0.3);

        let best = lb.best().unwrap();
        assert_eq!(best.candidate, "b");
        assert_eq!(best.observation, 0.1);
    }

    #[test]
    fn test_bottom_k() {
        let mut lb: Leaderboard<&str, f64> = Leaderboard::new();

        lb.push("best", 0.1);
        lb.push("worst", 1.0);
        lb.push("middle", 0.5);

        let bottom2 = lb.bottom_k(2);
        assert_eq!(bottom2[0].candidate, "worst");
        assert_eq!(bottom2[1].candidate, "middle");
    }

    #[test]
    fn test_quantile() {
        let mut lb: Leaderboard<i32, f64> = Leaderboard::new();

        for i in 0..100 {
            lb.push(i, i as f64);
        }

        let q25 = lb.quantile_threshold(0.25).unwrap();
        assert!((q25 - 25.0).abs() < 1.0);

        let q50 = lb.quantile_threshold(0.50).unwrap();
        assert!((q50 - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_pareto_front_simple() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // Point A: good in both
        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.1)].into());
        // Point B: dominated by A
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        // Point C: good in x, bad in y (not dominated by A)
        lb.push("C", [("x".into(), 0.05), ("y".into(), 0.8)].into());

        let front = lb.pareto_front();
        assert_eq!(front.len(), 2);

        let candidates: Vec<_> = front.iter().map(|t| t.candidate).collect();
        assert!(candidates.contains(&"A"));
        assert!(candidates.contains(&"C"));
        assert!(!candidates.contains(&"B"));
    }

    #[test]
    fn test_dominates_nan_is_strictly_worst() {
        // A NaN objective must be treated as strictly worst, not a tie.
        // A finite trial that is better in the NaN objective and equal/better
        // elsewhere must dominate the NaN trial, and the NaN trial must never
        // spuriously dominate the finite one.
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // "good": finite and better in y, equal in x.
        lb.push("good", [("x".into(), 0.1), ("y".into(), 0.1)].into());
        // "nan": NaN in y (strictly worst there), equal in x.
        lb.push("nan", [("x".into(), 0.1), ("y".into(), f64::NAN)].into());

        // pareto_front_all includes infeasible trials but still applies
        // dominance, so the NaN trial must be dropped as dominated by "good".
        let front = lb.pareto_front_all();
        let candidates: Vec<_> = front.iter().map(|t| t.candidate).collect();
        assert!(
            candidates.contains(&"good"),
            "finite-better trial must be on the front"
        );
        assert!(
            !candidates.contains(&"nan"),
            "NaN-objective trial must be dominated, not tie"
        );

        // Direct dominance checks: NaN must not spuriously dominate, and finite
        // must dominate NaN.
        let good_obs: BTreeMap<String, f64> = [("x".into(), 0.1), ("y".into(), 0.1)].into();
        let nan_obs: BTreeMap<String, f64> = [("x".into(), 0.1), ("y".into(), f64::NAN)].into();
        assert!(
            Leaderboard::<&str, BTreeMap<String, f64>>::dominates(&good_obs, &nan_obs),
            "finite-better must dominate NaN"
        );
        assert!(
            !Leaderboard::<&str, BTreeMap<String, f64>>::dominates(&nan_obs, &good_obs),
            "NaN must never dominate a finite trial"
        );
    }

    #[test]
    fn test_pareto_front_all_dominated() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        lb.push("best", [("x".into(), 0.1), ("y".into(), 0.1)].into());
        lb.push("worse1", [("x".into(), 0.2), ("y".into(), 0.2)].into());
        lb.push("worse2", [("x".into(), 0.3), ("y".into(), 0.3)].into());

        let front = lb.pareto_front();
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].candidate, "best");
    }

    #[test]
    fn test_pareto_front_for_subset() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // A: best overall
        lb.push(
            "A",
            [("x".into(), 0.1), ("y".into(), 0.1), ("z".into(), 0.5)].into(),
        );
        // B: best in z only
        lb.push(
            "B",
            [("x".into(), 0.5), ("y".into(), 0.5), ("z".into(), 0.1)].into(),
        );

        // Full Pareto front includes both
        let full = lb.pareto_front();
        assert_eq!(full.len(), 2);

        // Pareto front for just x and y: only A
        let xy_front = lb.pareto_front_for(&["x", "y"]);
        assert_eq!(xy_front.len(), 1);
        assert_eq!(xy_front[0].candidate, "A");
    }

    #[test]
    fn test_top_k_scalarized() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.9)].into());
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        lb.push("C", [("x".into(), 0.9), ("y".into(), 0.1)].into());

        // Equal weight scalarizer: sum
        let top = lb.top_k_scalarized(2, |obs| obs.values().sum());
        assert_eq!(top.len(), 2);
        // A and C both sum to 1.0, B sums to 1.0 too - all equal
        // Ties broken by trial_id
        assert_eq!(top[0].candidate, "A");
        assert_eq!(top[1].candidate, "B");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut lb: Leaderboard<(f64, f64), f64> = Leaderboard::new();
        lb.push((0.1, 0.2), 0.5);
        lb.push((0.3, 0.4), 0.3);

        let json = serde_json::to_string(&lb).unwrap();
        let restored: Leaderboard<(f64, f64), f64> = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.len(), 2);
        assert_eq!(restored.trials()[0].observation, 0.5);
        assert_eq!(restored.trials()[1].observation, 0.3);
    }

    #[test]
    fn test_get_by_id() {
        let mut lb: Leaderboard<&str, f64> = Leaderboard::new();

        lb.push("a", 0.1);
        let id = lb.push("b", 0.2);
        lb.push("c", 0.3);

        let trial = lb.get(id).unwrap();
        assert_eq!(trial.candidate, "b");
    }

    // =========================================================================
    // NSGA-II Tests
    // =========================================================================

    #[test]
    fn test_non_dominated_sort_single_front() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // All points on the Pareto front (none dominate each other)
        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.9)].into());
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        lb.push("C", [("x".into(), 0.9), ("y".into(), 0.1)].into());

        let fronts = lb.non_dominated_sort();
        assert_eq!(fronts.len(), 1);
        assert_eq!(fronts[0].len(), 3);
    }

    #[test]
    fn test_non_dominated_sort_multiple_fronts() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // Front 1: Pareto optimal
        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.1)].into());
        // Front 2: dominated by A
        lb.push("B", [("x".into(), 0.3), ("y".into(), 0.3)].into());
        // Front 3: dominated by B
        lb.push("C", [("x".into(), 0.5), ("y".into(), 0.5)].into());

        let fronts = lb.non_dominated_sort();
        assert_eq!(fronts.len(), 3);
        assert_eq!(fronts[0].len(), 1);
        assert_eq!(fronts[0][0].candidate, "A");
        assert_eq!(fronts[1].len(), 1);
        assert_eq!(fronts[1][0].candidate, "B");
        assert_eq!(fronts[2].len(), 1);
        assert_eq!(fronts[2][0].candidate, "C");
    }

    #[test]
    fn test_non_dominated_sort_mixed() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // Front 1: Two non-dominated points
        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.5)].into());
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.1)].into());
        // Front 2: Dominated by both A and B
        lb.push("C", [("x".into(), 0.6), ("y".into(), 0.6)].into());

        let fronts = lb.non_dominated_sort();
        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0].len(), 2);
        assert_eq!(fronts[1].len(), 1);
        assert_eq!(fronts[1][0].candidate, "C");
    }

    #[test]
    fn test_crowding_distance_boundary_infinite() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.9)].into());
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        lb.push("C", [("x".into(), 0.9), ("y".into(), 0.1)].into());

        let front = lb.pareto_front();
        let with_distance = Leaderboard::<&str, BTreeMap<String, f64>>::crowding_distance(&front);

        // Boundary points (best in any objective) should have infinite distance
        let distances: Vec<f64> = with_distance.iter().map(|(_, d)| *d).collect();

        // At least 2 points should have infinite distance (the extremes)
        let infinite_count = distances.iter().filter(|d| d.is_infinite()).count();
        assert!(infinite_count >= 2);
    }

    #[test]
    fn test_crowding_distance_interior_finite() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // Create a front with clear interior point
        lb.push("left", [("x".into(), 0.0), ("y".into(), 1.0)].into());
        lb.push("middle", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        lb.push("right", [("x".into(), 1.0), ("y".into(), 0.0)].into());

        let front = lb.pareto_front();
        let with_distance = Leaderboard::<&str, BTreeMap<String, f64>>::crowding_distance(&front);

        // Find the middle point's distance
        let middle_distance = with_distance
            .iter()
            .find(|(t, _)| t.candidate == "middle")
            .map(|(_, d)| *d);

        // Middle point should have finite distance
        assert!(middle_distance.is_some());
        assert!(middle_distance.unwrap().is_finite());
        assert!(middle_distance.unwrap() > 0.0);
    }

    #[test]
    fn test_select_nsga2_respects_rank() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // Front 1
        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.1)].into());
        // Front 2
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        // Front 3
        lb.push("C", [("x".into(), 0.9), ("y".into(), 0.9)].into());

        let selected = lb.select_nsga2(2);
        assert_eq!(selected.len(), 2);

        // Should select A (rank 1) and B (rank 2), not C
        let candidates: Vec<&str> = selected.iter().map(|r| r.trial.candidate).collect();
        assert!(candidates.contains(&"A"));
        assert!(candidates.contains(&"B"));
        assert!(!candidates.contains(&"C"));
    }

    #[test]
    fn test_select_nsga2_uses_crowding_within_rank() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // All on Pareto front (rank 1)
        lb.push("extreme1", [("x".into(), 0.0), ("y".into(), 1.0)].into());
        lb.push("middle", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        lb.push("extreme2", [("x".into(), 1.0), ("y".into(), 0.0)].into());

        // Select 2 from 3 - should prefer extremes (infinite crowding distance)
        let selected = lb.select_nsga2(2);
        assert_eq!(selected.len(), 2);

        let candidates: Vec<&str> = selected.iter().map(|r| r.trial.candidate).collect();
        // Both extremes should be selected (they have infinite crowding distance)
        assert!(candidates.contains(&"extreme1"));
        assert!(candidates.contains(&"extreme2"));
    }

    #[test]
    fn test_pareto_front_with_crowding() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.9)].into());
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        lb.push("C", [("x".into(), 0.9), ("y".into(), 0.1)].into());

        let ranked = lb.pareto_front_with_crowding();
        assert_eq!(ranked.len(), 3);

        // All should be rank 1
        for r in &ranked {
            assert_eq!(r.rank, 1);
        }

        // Should be sorted by crowding distance (descending)
        // First elements should have infinite distance
        assert!(ranked[0].crowding_distance.is_infinite());
    }

    #[test]
    fn test_crowded_compare() {
        let trial_a: Trial<&str, BTreeMap<String, f64>> =
            Trial::new("A", [("x".into(), 0.1)].into(), 0);
        let trial_b: Trial<&str, BTreeMap<String, f64>> =
            Trial::new("B", [("x".into(), 0.2)].into(), 1);

        let ranked_a = RankedTrial {
            trial: trial_a.clone(),
            rank: 1,
            crowding_distance: 0.5,
        };
        let ranked_b = RankedTrial {
            trial: trial_b.clone(),
            rank: 2,
            crowding_distance: 1.0,
        };

        // A (rank 1) should be preferred over B (rank 2)
        assert_eq!(
            ranked_a.crowded_compare(&ranked_b),
            std::cmp::Ordering::Less
        );

        // Same rank: higher crowding distance wins
        let ranked_c = RankedTrial {
            trial: trial_a,
            rank: 1,
            crowding_distance: 0.3,
        };
        let ranked_d = RankedTrial {
            trial: trial_b,
            rank: 1,
            crowding_distance: 0.8,
        };

        // D has higher crowding distance, so D is preferred
        assert_eq!(
            ranked_d.crowded_compare(&ranked_c),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_ranked_trials() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.1)].into());
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.5)].into());

        let ranked = lb.ranked_trials();
        assert_eq!(ranked.len(), 2);

        // A should be rank 1, B should be rank 2
        let a_rank = ranked.iter().find(|r| r.trial.candidate == "A").unwrap();
        let b_rank = ranked.iter().find(|r| r.trial.candidate == "B").unwrap();

        assert_eq!(a_rank.rank, 1);
        assert_eq!(b_rank.rank, 2);
    }

    // =========================================================================
    // Feasibility Tests
    // =========================================================================

    #[test]
    fn test_is_feasible_scalar() {
        assert!(is_feasible_scalar(0.5));
        assert!(is_feasible_scalar(-1.0));
        assert!(is_feasible_scalar(0.0));
        assert!(!is_feasible_scalar(f64::INFINITY));
        assert!(!is_feasible_scalar(f64::NEG_INFINITY));
        assert!(!is_feasible_scalar(f64::NAN));
    }

    #[test]
    fn test_is_feasible_multi() {
        let feasible: BTreeMap<String, f64> = [("x".into(), 0.5), ("y".into(), 0.3)].into();
        assert!(is_feasible_multi(&feasible));

        let infeasible: BTreeMap<String, f64> =
            [("x".into(), 0.5), ("y".into(), f64::INFINITY)].into();
        assert!(!is_feasible_multi(&infeasible));

        let all_inf: BTreeMap<String, f64> =
            [("x".into(), f64::INFINITY), ("y".into(), f64::INFINITY)].into();
        assert!(!is_feasible_multi(&all_inf));
    }

    #[test]
    fn test_scalar_feasible_count() {
        let mut lb: Leaderboard<&str, f64> = Leaderboard::new();

        lb.push("good1", 0.1);
        lb.push("infeasible", f64::INFINITY);
        lb.push("good2", 0.2);

        assert_eq!(lb.len(), 3);
        assert_eq!(lb.feasible_count(), 2);
    }

    #[test]
    fn test_scalar_top_k_excludes_infinity() {
        let mut lb: Leaderboard<&str, f64> = Leaderboard::new();

        lb.push("good", 0.5);
        lb.push("infeasible", f64::INFINITY);
        lb.push("best", 0.1);

        // top_k excludes infinity by default
        let feasible = lb.top_k(3);
        assert_eq!(feasible.len(), 2);
        assert_eq!(feasible[0].candidate, "best");
        assert_eq!(feasible[1].candidate, "good");

        // top_k_all includes infinity
        let all = lb.top_k_all(3);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_scalar_best() {
        let mut lb: Leaderboard<&str, f64> = Leaderboard::new();

        lb.push("infeasible", f64::INFINITY);
        lb.push("good", 0.5);

        // best() excludes infeasible
        let best = lb.best().unwrap();
        assert_eq!(best.candidate, "good");

        // best_all() includes infeasible (but inf sorts last)
        let best_all = lb.best_all().unwrap();
        assert_eq!(best_all.candidate, "good");
    }

    #[test]
    fn test_scalar_all_infeasible() {
        let mut lb: Leaderboard<&str, f64> = Leaderboard::new();

        lb.push("inf1", f64::INFINITY);
        lb.push("inf2", f64::INFINITY);

        assert_eq!(lb.feasible_count(), 0);
        assert!(lb.top_k(5).is_empty());
        assert!(lb.best().is_none());
    }

    #[test]
    fn test_multi_feasible_count() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        lb.push("good", [("x".into(), 0.1), ("y".into(), 0.2)].into());
        lb.push(
            "infeasible",
            [("x".into(), 0.1), ("y".into(), f64::INFINITY)].into(),
        );
        lb.push("also_good", [("x".into(), 0.3), ("y".into(), 0.4)].into());

        assert_eq!(lb.len(), 3);
        assert_eq!(lb.feasible_count(), 2);
    }

    #[test]
    fn test_pareto_front_excludes_infinity() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // Feasible Pareto front
        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.9)].into());
        lb.push("B", [("x".into(), 0.9), ("y".into(), 0.1)].into());

        // Infeasible trial
        lb.push(
            "infeasible",
            [("x".into(), 0.05), ("y".into(), f64::INFINITY)].into(),
        );

        // pareto_front excludes infeasible by default
        let front = lb.pareto_front();
        assert_eq!(front.len(), 2);

        let candidates: Vec<&str> = front.iter().map(|t| t.candidate).collect();
        assert!(candidates.contains(&"A"));
        assert!(candidates.contains(&"B"));
        assert!(!candidates.contains(&"infeasible"));

        // pareto_front_all includes infeasible
        let all_front = lb.pareto_front_all();
        assert_eq!(all_front.len(), 3);
    }

    #[test]
    fn test_non_dominated_sort_excludes_infeasible() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        // Feasible front 1
        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.1)].into());
        // Feasible front 2
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        // Infeasible - should be excluded
        lb.push(
            "inf",
            [("x".into(), 0.0), ("y".into(), f64::INFINITY)].into(),
        );

        let fronts = lb.non_dominated_sort();
        assert_eq!(fronts.len(), 2);

        // Should only have A and B across both fronts
        let total_trials: usize = fronts.iter().map(|f: &Vec<_>| f.len()).sum();
        assert_eq!(total_trials, 2);
    }

    #[test]
    fn test_select_nsga2_excludes_infeasible() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        lb.push("good1", [("x".into(), 0.1), ("y".into(), 0.1)].into());
        lb.push("good2", [("x".into(), 0.2), ("y".into(), 0.2)].into());
        lb.push(
            "infeasible",
            [("x".into(), 0.0), ("y".into(), f64::INFINITY)].into(),
        );

        let selected = lb.select_nsga2(10);
        assert_eq!(selected.len(), 2);

        let candidates: Vec<&str> = selected.iter().map(|r| r.trial.candidate).collect();
        assert!(candidates.contains(&"good1"));
        assert!(candidates.contains(&"good2"));
        assert!(!candidates.contains(&"infeasible"));
    }

    #[test]
    fn test_pareto_front_with_crowding_excludes_infeasible() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.9)].into());
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        lb.push("C", [("x".into(), 0.9), ("y".into(), 0.1)].into());
        lb.push(
            "inf",
            [("x".into(), 0.0), ("y".into(), f64::INFINITY)].into(),
        );

        let ranked = lb.pareto_front_with_crowding();
        assert_eq!(ranked.len(), 3);

        // All should be rank 1
        for r in &ranked {
            assert_eq!(r.rank, 1);
        }

        // None should be the infeasible one
        for r in &ranked {
            assert_ne!(r.trial.candidate, "inf");
        }
    }

    #[test]
    fn test_ranked_trials_excludes_infeasible() {
        let mut lb: Leaderboard<&str, BTreeMap<String, f64>> = Leaderboard::new();

        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.1)].into());
        lb.push("B", [("x".into(), 0.5), ("y".into(), 0.5)].into());
        lb.push(
            "inf",
            [("x".into(), f64::INFINITY), ("y".into(), 0.0)].into(),
        );

        let ranked = lb.ranked_trials();
        assert_eq!(ranked.len(), 2);

        // A should be rank 1, B should be rank 2
        let a_rank = ranked.iter().find(|r| r.trial.candidate == "A").unwrap();
        let b_rank = ranked.iter().find(|r| r.trial.candidate == "B").unwrap();

        assert_eq!(a_rank.rank, 1);
        assert_eq!(b_rank.rank, 2);
    }

    // ==================================================================
    // Edge cases from integration tests (unique scenarios)
    // ==================================================================

    #[test]
    fn test_is_feasible_scalar_neg_infinity() {
        assert!(!is_feasible_scalar(f64::NEG_INFINITY));
    }

    #[test]
    fn test_is_feasible_multi_nan() {
        let obs: BTreeMap<String, f64> = [("a".into(), f64::NAN)].into();
        assert!(!is_feasible_multi(&obs));
    }

    #[test]
    fn test_empty_leaderboard_accessors() {
        let lb = Leaderboard::<f64, f64>::new();
        assert!(lb.best().is_none());
        assert!(lb.top_k(5).is_empty());
        assert!(lb.bottom_k(5).is_empty());
        assert!(lb.quantile_threshold(0.5).is_none());
        assert!(lb.top_quantile(0.5).is_empty());
        assert!(lb.last().is_none());
    }

    #[test]
    fn test_all_infeasible_with_neg_infinity() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("a", f64::INFINITY);
        lb.push("b", f64::NEG_INFINITY);

        assert_eq!(lb.feasible_count(), 0);
        assert!(lb.best().is_none());
        assert!(lb.top_k(10).is_empty());
        assert_eq!(lb.top_k_all(10).len(), 2);
        assert!(lb.best_all().is_some());
    }

    #[test]
    fn test_nan_observation() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("ok", 0.5);
        lb.push("nan", f64::NAN);
        lb.push("ok2", 0.3);

        assert_eq!(lb.feasible_count(), 2);
        let top = lb.top_k(10);
        assert_eq!(top.len(), 2);
        for t in &top {
            assert!(t.observation.is_finite());
        }
        assert!((lb.best().unwrap().observation - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_quantile_threshold_edges() {
        let mut lb = Leaderboard::<i32, f64>::new();
        for i in 0..10 {
            lb.push(i, i as f64);
        }
        let q0 = lb.quantile_threshold(0.0).unwrap();
        assert!((q0 - 0.0).abs() < 1e-9);
        let q1 = lb.quantile_threshold(1.0).unwrap();
        assert!((q1 - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_top_k_larger_than_len() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("a", 0.1);
        lb.push("b", 0.2);
        assert_eq!(lb.top_k(100).len(), 2);
    }

    #[test]
    fn test_top_k_zero() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("a", 0.1);
        assert!(lb.top_k(0).is_empty());
    }

    #[test]
    fn test_bottom_k_zero() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("a", 0.1);
        assert!(lb.bottom_k(0).is_empty());
    }

    #[test]
    fn test_top_k_all_includes_infeasible() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("a", 0.1);
        lb.push("b", f64::INFINITY);
        lb.push("c", 0.3);

        let top_all = lb.top_k_all(10);
        assert_eq!(top_all.len(), 3);
        assert!(top_all[0].observation.is_finite());
        assert!(top_all[1].observation.is_finite());
    }

    #[test]
    fn test_bottom_k_all_includes_infeasible() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("a", 0.1);
        lb.push("b", f64::INFINITY);
        lb.push("c", 0.3);

        let bottom_all = lb.bottom_k_all(10);
        assert_eq!(bottom_all.len(), 3);
        assert!(bottom_all[0].observation.is_infinite());
    }

    #[test]
    fn test_sorted_and_sorted_all() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("c", 0.3);
        lb.push("inf", f64::INFINITY);
        lb.push("a", 0.1);
        lb.push("b", 0.2);

        let sorted = lb.sorted();
        assert_eq!(sorted.len(), 3);
        assert!((sorted[0].observation - 0.1).abs() < 1e-9);
        assert!((sorted[1].observation - 0.2).abs() < 1e-9);
        assert!((sorted[2].observation - 0.3).abs() < 1e-9);

        let sorted_all = lb.sorted_all();
        assert_eq!(sorted_all.len(), 4);
    }

    #[test]
    fn test_clear_preserves_id_uniqueness() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("a", 0.1);
        lb.push("b", 0.2);
        lb.clear();
        assert!(lb.is_empty());

        let id = lb.push("c", 0.3);
        assert!(id >= 2, "After clear, new IDs should be >= 2, got {id}");
    }

    #[test]
    fn test_clear_then_accessors() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("a", 0.1);
        lb.clear();

        assert!(lb.best().is_none());
        assert!(lb.top_k(5).is_empty());
        assert!(lb.last().is_none());
        assert_eq!(lb.feasible_count(), 0);
    }

    #[test]
    fn test_large_1000_trials() {
        let mut lb = Leaderboard::<u32, f64>::new();
        for i in 0..1000 {
            lb.push(i, (i as f64) * 0.001);
        }

        assert_eq!(lb.len(), 1000);
        let top_5 = lb.top_k(5);
        assert_eq!(top_5[0].trial_id, 0);
        assert!((top_5[0].observation - 0.0).abs() < 1e-9);

        let bottom_5 = lb.bottom_k(5);
        assert_eq!(bottom_5[0].trial_id, 999);
    }

    #[test]
    fn test_iter_insertion_order() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("c", 0.3);
        lb.push("a", 0.1);
        lb.push("b", 0.2);

        let ids: Vec<u64> = lb.iter().map(|t| t.trial_id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(lb.trials()[0].candidate, "c");
    }

    #[test]
    fn test_last() {
        let mut lb = Leaderboard::<&str, f64>::new();
        assert!(lb.last().is_none());
        lb.push("a", 0.1);
        assert_eq!(lb.last().unwrap().candidate, "a");
        lb.push("b", 0.2);
        assert_eq!(lb.last().unwrap().candidate, "b");
    }

    #[test]
    fn test_with_capacity() {
        let mut lb = Leaderboard::<&str, f64>::with_capacity(100);
        assert!(lb.is_empty());
        lb.push("a", 0.1);
        assert_eq!(lb.len(), 1);
    }

    #[test]
    fn test_rescalarize_updates_observations() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push_with_raw("a", 0.5, serde_json::json!({"loss": 0.5, "acc": 0.9}));
        lb.push_with_raw("b", 0.3, serde_json::json!({"loss": 0.3, "acc": 0.7}));

        lb.rescalarize(|raw| raw.get("acc")?.as_f64().map(|v| -v));

        let best = lb.best().unwrap();
        assert!((best.observation - (-0.9)).abs() < 1e-9);
    }

    #[test]
    fn test_rescalarize_no_raw_metrics_unchanged() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("a", 0.5);
        lb.rescalarize(|_| Some(999.0));
        assert!((lb.best().unwrap().observation - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_multi_objective_empty() {
        let lb = Leaderboard::<&str, BTreeMap<String, f64>>::new();
        assert!(lb.pareto_front().is_empty());
        assert!(lb.feasible_trials().is_empty());
    }

    #[test]
    fn test_non_dominated_sort_all_variant() {
        let mut lb = Leaderboard::<&str, BTreeMap<String, f64>>::new();
        lb.push(
            "feasible",
            [("loss".into(), 0.5), ("lat".into(), 50.0)].into(),
        );
        lb.push(
            "infeasible",
            [("loss".into(), 0.1), ("lat".into(), f64::INFINITY)].into(),
        );

        let fronts = lb.non_dominated_sort();
        let total: usize = fronts.iter().map(|f| f.len()).sum();
        assert_eq!(total, 1);

        let fronts_all = lb.non_dominated_sort_all();
        let total_all: usize = fronts_all.iter().map(|f| f.len()).sum();
        assert_eq!(total_all, 2);
    }

    #[test]
    fn test_select_nsga2_zero() {
        let mut lb = Leaderboard::<&str, BTreeMap<String, f64>>::new();
        lb.push("a", [("loss".into(), 0.5), ("lat".into(), 50.0)].into());
        assert!(lb.select_nsga2(0).is_empty());
    }

    #[test]
    fn test_select_nsga2_all_variant() {
        let mut lb = Leaderboard::<&str, BTreeMap<String, f64>>::new();
        lb.push(
            "feasible",
            [("loss".into(), 0.5), ("lat".into(), 50.0)].into(),
        );
        lb.push(
            "infeasible",
            [("loss".into(), 0.1), ("lat".into(), f64::INFINITY)].into(),
        );

        assert_eq!(lb.select_nsga2(10).len(), 1);
        assert_eq!(lb.select_nsga2_all(10).len(), 2);
    }

    #[test]
    fn test_feasibility_excludes_infinite_priority_groups() {
        let mut lb = Leaderboard::<&str, BTreeMap<String, f64>>::new();

        lb.push(
            "a",
            [("perf".into(), 0.3), ("resources".into(), 0.5)].into(),
        );
        lb.push(
            "b",
            [("perf".into(), 0.2), ("resources".into(), f64::INFINITY)].into(),
        );
        lb.push(
            "c",
            [("perf".into(), f64::INFINITY), ("resources".into(), 0.1)].into(),
        );
        lb.push(
            "d",
            [("perf".into(), 0.1), ("resources".into(), 0.2)].into(),
        );

        assert_eq!(lb.feasible_count(), 2);
        let feasible_ids: Vec<u64> = lb.feasible_trials().iter().map(|t| t.trial_id).collect();
        assert!(feasible_ids.contains(&0));
        assert!(feasible_ids.contains(&3));

        let front = lb.pareto_front();
        for trial in &front {
            for &val in trial.observation.values() {
                assert!(val.is_finite());
            }
        }

        let fronts = lb.non_dominated_sort();
        let total: usize = fronts.iter().map(|f| f.len()).sum();
        assert_eq!(total, 2);
    }

    // ==================================================================
    // Id index, single-trial rank, and bounded mode
    // ==================================================================

    #[test]
    fn test_contains_trial_id_after_push() {
        let mut lb = Leaderboard::<&str, f64>::new();
        let a = lb.push("a", 0.1);
        let b = lb.push_with_trial_id("b", 0.2, 42);

        assert!(lb.contains_trial_id(a));
        assert!(lb.contains_trial_id(b));
        assert!(!lb.contains_trial_id(7));
        assert_eq!(lb.get(b).unwrap().candidate, "b");
        assert!(lb.get(7).is_none());
    }

    #[test]
    fn test_id_index_survives_serde_roundtrip() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("a", 0.1);
        lb.push_with_trial_id("b", 0.2, 100);

        let json = serde_json::to_string(&lb).unwrap();
        // The derived index must not be persisted.
        assert!(!json.contains("id_index"));

        let restored: Leaderboard<String, f64> = serde_json::from_str(&json).unwrap();
        // O(1) lookup works without any explicit rebuild call.
        assert!(restored.contains_trial_id(0));
        assert!(restored.contains_trial_id(100));
        assert!(!restored.contains_trial_id(1));
        assert_eq!(restored.get(100).unwrap().candidate, "b");
        // next_id is preserved across the roundtrip.
        assert_eq!(restored.next_trial_id(), 101);
    }

    #[test]
    fn test_legacy_json_without_max_size_deserializes() {
        // Legacy checkpoints have only trials and next_id.
        let json = r#"{"trials":[{"candidate":"a","observation":0.5,"trial_id":3,"timestamp":7}],"next_id":4}"#;
        let lb: Leaderboard<String, f64> = serde_json::from_str(json).unwrap();
        assert_eq!(lb.len(), 1);
        assert_eq!(lb.max_size(), None);
        assert!(lb.contains_trial_id(3));
        assert_eq!(lb.next_trial_id(), 4);
    }

    #[test]
    fn test_rank_of_matches_full_ranking_scalar() {
        let mut lb = Leaderboard::<&str, f64>::new();
        lb.push("c", 0.3);
        lb.push("a", 0.1);
        lb.push("inf", f64::INFINITY);
        lb.push("b", 0.2);
        lb.push("a_tie", 0.1); // ties with "a" by observation, later trial_id

        // Build the ground-truth rank from the full sorted order.
        let sorted = lb.sorted();
        for (expected_rank, t) in sorted.iter().enumerate() {
            assert_eq!(
                lb.rank_of(t.trial_id, false),
                Some(expected_rank),
                "feasible rank mismatch for trial {}",
                t.trial_id
            );
        }

        // Infeasible target is excluded unless include_infeasible is set.
        let inf_id = lb.iter().find(|t| t.candidate == "inf").unwrap().trial_id;
        assert_eq!(lb.rank_of(inf_id, false), None);

        let sorted_all = lb.sorted_all();
        for (expected_rank, t) in sorted_all.iter().enumerate() {
            assert_eq!(
                lb.rank_of(t.trial_id, true),
                Some(expected_rank),
                "all rank mismatch for trial {}",
                t.trial_id
            );
        }

        assert_eq!(lb.rank_of(9999, false), None);
    }

    #[test]
    fn test_pareto_rank_of_matches_full_sort() {
        let mut lb = Leaderboard::<&str, BTreeMap<String, f64>>::new();
        lb.push("A", [("x".into(), 0.1), ("y".into(), 0.1)].into()); // front 0
        lb.push("B", [("x".into(), 0.3), ("y".into(), 0.3)].into()); // front 1
        lb.push("C", [("x".into(), 0.5), ("y".into(), 0.5)].into()); // front 2
        lb.push("D", [("x".into(), 0.1), ("y".into(), 0.9)].into()); // front 0 (non-dominated)
        lb.push(
            "inf",
            [("x".into(), 0.0), ("y".into(), f64::INFINITY)].into(),
        );

        // Ground truth: front index from the full non-dominated sort.
        let fronts = lb.non_dominated_sort();
        for (front_idx, front) in fronts.iter().enumerate() {
            for t in front {
                assert_eq!(
                    lb.pareto_rank_of(t.trial_id, false),
                    Some(front_idx),
                    "front mismatch for trial {}",
                    t.trial_id
                );
            }
        }

        let inf_id = lb.iter().find(|t| t.candidate == "inf").unwrap().trial_id;
        assert_eq!(lb.pareto_rank_of(inf_id, false), None);
        assert!(lb.pareto_rank_of(inf_id, true).is_some());
        assert_eq!(lb.pareto_rank_of(123, false), None);
    }

    #[test]
    fn test_unbounded_default_keeps_everything() {
        let mut lb = Leaderboard::<u32, f64>::new();
        for i in 0..50 {
            lb.push(i, i as f64);
        }
        assert_eq!(lb.max_size(), None);
        assert_eq!(lb.len(), 50);
        // First and last still present.
        assert!(lb.contains_trial_id(0));
        assert!(lb.contains_trial_id(49));
    }

    #[test]
    fn test_bounded_mode_caps_and_evicts_oldest() {
        let mut lb = Leaderboard::<u32, f64>::new().with_max_size(Some(3));
        for i in 0..6 {
            lb.push(i, i as f64);
        }
        // Only the 3 most recent (oldest evicted) remain.
        assert_eq!(lb.len(), 3);
        let ids: Vec<u64> = lb.iter().map(|t| t.trial_id).collect();
        assert_eq!(ids, vec![3, 4, 5]);

        // Evicted ids are gone from the index; survivors resolve correctly.
        assert!(!lb.contains_trial_id(0));
        assert!(!lb.contains_trial_id(2));
        assert!(lb.contains_trial_id(3));
        assert_eq!(lb.get(5).unwrap().candidate, 5);
        for t in lb.iter() {
            assert_eq!(lb.get(t.trial_id).unwrap().candidate, t.candidate);
        }

        // Ranking still consistent over the retained window.
        assert_eq!(lb.rank_of(3, false), Some(0));
        assert_eq!(lb.rank_of(5, false), Some(2));

        // next_id keeps growing so ids stay unique after eviction.
        assert_eq!(lb.push(99, 99.0), 6);
    }

    #[test]
    fn test_set_max_size_shrinks_immediately_then_clears() {
        let mut lb = Leaderboard::<u32, f64>::new();
        for i in 0..5 {
            lb.push(i, i as f64);
        }
        lb.set_max_size(Some(2));
        assert_eq!(lb.len(), 2);
        let ids: Vec<u64> = lb.iter().map(|t| t.trial_id).collect();
        assert_eq!(ids, vec![3, 4]);

        // Clearing the cap stops eviction; further pushes accumulate.
        lb.set_max_size(None);
        for i in 5..8 {
            lb.push(i, i as f64);
        }
        assert_eq!(lb.len(), 5);
    }

    #[test]
    fn test_cap_of_one_keeps_only_newest() {
        let mut lb = Leaderboard::<u32, f64>::new().with_max_size(Some(1));
        // Each push evicts the previous trial; the newest survives.
        for i in 0..5 {
            let id = lb.push(i, i as f64);
            assert_eq!(lb.len(), 1, "len must stay 1 after pushing {i}");
            // The trial just pushed is readable and is the only survivor.
            let only = lb.last().unwrap();
            assert_eq!(only.candidate, i, "newest trial must survive push {i}");
            assert_eq!(only.trial_id, id);
            assert_eq!(lb.get(id).unwrap().candidate, i);
            // Earlier trials have been evicted from the index.
            if i > 0 {
                assert!(!lb.contains_trial_id(id - 1));
            }
        }
    }

    #[test]
    fn test_total_completed_unbounded_equals_len() {
        let mut lb = Leaderboard::<u32, f64>::new();
        assert_eq!(lb.total_completed(), 0);
        // Exercise every push route; each must bump the counter by one.
        lb.push(0, 0.0);
        lb.push_with_trial_id(1, 1.0, 10);
        lb.push_with_raw(2, 2.0, serde_json::json!({"loss": 2.0}));
        lb.push_with_raw_trial_id(3, 3.0, serde_json::json!({"loss": 3.0}), 20);
        lb.push_existing_trial(Trial {
            candidate: 4,
            observation: 4.0,
            raw_metrics: None,
            trial_id: 30,
            timestamp: 1,
        });
        assert_eq!(lb.total_completed(), 5);
        // Unbounded: counter and stored length stay in lockstep.
        assert_eq!(lb.total_completed(), lb.len() as u64);
    }

    #[test]
    fn test_total_completed_grows_past_cap() {
        let mut lb = Leaderboard::<u32, f64>::new().with_max_size(Some(3));
        for i in 0..10 {
            lb.push(i, i as f64);
            // Counter tracks pushes, not the evicted stored set.
            assert_eq!(lb.total_completed(), (i + 1) as u64);
        }
        // Eviction held len at the cap but never lowered the counter.
        assert_eq!(lb.len(), 3);
        assert_eq!(lb.total_completed(), 10);
        assert!(lb.total_completed() > lb.len() as u64);
    }

    #[test]
    fn test_total_completed_reset_by_clear() {
        let mut lb = Leaderboard::<u32, f64>::new().with_max_size(Some(2));
        for i in 0..5 {
            lb.push(i, i as f64);
        }
        assert_eq!(lb.total_completed(), 5);
        lb.clear();
        assert_eq!(lb.total_completed(), 0);
        // Counting resumes from zero after a clear.
        lb.push(99, 9.0);
        assert_eq!(lb.total_completed(), 1);
    }

    #[test]
    fn test_total_completed_survives_serde_roundtrip() {
        let mut lb = Leaderboard::<u32, f64>::new().with_max_size(Some(3));
        for i in 0..7 {
            lb.push(i, i as f64);
        }
        assert_eq!(lb.total_completed(), 7);
        assert_eq!(lb.len(), 3);

        let json = serde_json::to_string(&lb).unwrap();
        let restored: Leaderboard<u32, f64> = serde_json::from_str(&json).unwrap();
        // The real count survives even though only 3 trials are stored.
        assert_eq!(restored.total_completed(), 7);
        assert_eq!(restored.len(), 3);
        assert_eq!(restored.max_size(), Some(3));
    }

    #[test]
    fn test_total_completed_legacy_json_defaults_to_len() {
        // Pre-counter checkpoints lack total_completed and were unbounded, so
        // total pushed equals stored len.
        let json = r#"{"trials":[
            {"candidate":1,"observation":0.1,"trial_id":0,"timestamp":1},
            {"candidate":2,"observation":0.2,"trial_id":1,"timestamp":2}
        ],"next_id":2}"#;
        let lb: Leaderboard<u32, f64> = serde_json::from_str(json).unwrap();
        assert_eq!(lb.len(), 2);
        assert_eq!(lb.total_completed(), 2);
    }

    #[test]
    fn test_total_completed_new_checkpoint_count_beats_len() {
        // A new checkpoint can record a count above stored len (post-cap). The
        // max() reconstruction must keep that real count, not clamp to len.
        let json = r#"{"trials":[
            {"candidate":1,"observation":0.1,"trial_id":5,"timestamp":1}
        ],"next_id":6,"max_size":1,"total_completed":42}"#;
        let lb: Leaderboard<u32, f64> = serde_json::from_str(json).unwrap();
        assert_eq!(lb.len(), 1);
        assert_eq!(lb.total_completed(), 42);
    }

    #[test]
    fn test_cap_zero_is_normalized_to_unbounded() {
        // Some(0) is invalid and treated as unbounded: it must never evict the
        // just-pushed trial. Builder, setter, and deserialization all agree.
        let mut lb = Leaderboard::<u32, f64>::new().with_max_size(Some(0));
        assert_eq!(lb.max_size(), None, "with_max_size(Some(0)) must normalize");
        for i in 0..4 {
            lb.push(i, i as f64);
        }
        assert_eq!(lb.len(), 4, "cap 0 must behave as unbounded");
        assert!(lb.contains_trial_id(0));
        assert!(lb.contains_trial_id(3));

        // set_max_size(Some(0)) on a populated leaderboard must not drop trials.
        lb.set_max_size(Some(0));
        assert_eq!(lb.max_size(), None, "set_max_size(Some(0)) must normalize");
        assert_eq!(lb.len(), 4, "cap 0 must not evict existing trials");

        // A persisted Some(0) deserializes to unbounded as well.
        let json = r#"{"trials":[{"candidate":1,"observation":0.5,"trial_id":0,"timestamp":7}],"next_id":1,"max_size":0}"#;
        let restored: Leaderboard<u32, f64> = serde_json::from_str(json).unwrap();
        assert_eq!(restored.max_size(), None);
        assert_eq!(restored.len(), 1);
        assert!(restored.contains_trial_id(0));
    }
}
