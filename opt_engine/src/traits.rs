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

//! Core trait definitions for the optimization engine.
//!
//! These traits define the composable abstractions that make up an optimization loop:
//!
//! 1. **[`SampleSpace`]** — defines what a valid hyperparameter configuration
//!    looks like (bounds checking including clamping).
//! 2. **[`StandardizedSpace`]** — *optional* extension that maps a space
//!    to/from the unit hypercube `[0, 1]^n`. Required by strategies that
//!    operate in the latent space (e.g., random, Sobol, GMM), but not by
//!    all spaces or strategies in general.
//! 3. **[`Strategy`]** — the search algorithm. Proposes candidates and
//!    incorporates observed results.
//! 4. **[`RefittableStrategy`]** — optional extension for strategies that can
//!    rebuild their internal model from a batch of historical trials (e.g., GMM
//!    refitting from the top quantile of the leaderboard).

use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Debug;

/// Defines the structure and validity rules for a hyperparameter search space.
///
/// Every optimization problem starts with a `SampleSpace` that describes what
/// a valid configuration looks like. Callers use `contains` to validate
/// strategy proposals and `clamp` to snap out-of-bounds candidates back into
/// the feasible region.
pub trait SampleSpace: Send + Sync + 'static {
    /// A single point in this space — the type that gets serialized to JSON
    /// and sent to the worker process (e.g., `f64`, `(f64, i64)`, or a
    /// custom struct).
    type Domain: Serialize + DeserializeOwned + Send + Sync + Clone + Debug + PartialEq;

    fn contains(&self, point: &Self::Domain) -> bool;

    /// Snap an out-of-bounds point to the nearest valid value.
    ///
    /// The default implementation is a no-op; override this for spaces where
    /// clamping is meaningful (e.g., continuous ranges, integer ranges).
    fn clamp(&self, point: &Self::Domain) -> Self::Domain
    where
        Self::Domain: Clone,
    {
        point.clone()
    }
}

/// Optional extension of [`SampleSpace`] that provides a bijection to the
/// unit hypercube `[0, 1]^n`.
///
/// Strategies like [`RandomStrategy`](crate::strategies::RandomStrategy) and
/// [`SobolStrategy`](crate::strategies::SobolStrategy) operate in `[0, 1]^n`
/// and require this mapping. A space that does *not* implement
/// `StandardizedSpace` can still be used with strategies that work directly
/// in the domain (all built-in spaces currently implement this trait, but
/// custom spaces are not required to).
pub trait StandardizedSpace: SampleSpace {
    /// Number of continuous dimensions this space occupies in `[0, 1]^n`.
    fn dimensionality(&self) -> usize;

    /// Map a domain point to its `[0, 1]^n` representation.
    fn to_unit_cube(&self, point: &Self::Domain) -> Vec<f64>;

    /// Map a `[0, 1]^n` vector back to a domain point, or `None` if the
    /// vector has the wrong length.
    #[allow(clippy::wrong_self_convention)] // `self` is the space; name matches `to_unit_cube`
    fn from_unit_cube(&self, vec: &[f64]) -> Option<Self::Domain>;
}

/// A search algorithm that proposes candidate configurations and learns from results.
///
/// The two associated types wire the strategy into the rest of the system
/// at compile time: `Space` determines which parameter space the strategy
/// operates on, and `Observation` determines the result type fed back via
/// `update` (typically `f64` for scalar optimization, or
/// `BTreeMap<String, f64>` for multi-objective).
pub trait Strategy: Send + Sync + 'static {
    type Space: SampleSpace;

    /// The result type the strategy consumes (typically `f64` for scalar
    /// optimization, or a map of named objectives for multi-objective).
    type Observation: Serialize + DeserializeOwned + Send + Sync + Clone + Debug;

    fn suggest(&self, space: &Self::Space) -> <Self::Space as SampleSpace>::Domain;

    fn update(
        &mut self,
        candidate: &<Self::Space as SampleSpace>::Domain,
        observation: Self::Observation,
    );
}

// =============================================================================
// Refittable Strategy
// =============================================================================

/// Extension trait for strategies that can refit their internal distribution
/// from historical trial data.
///
/// This is separate from `Strategy` to:
/// - Not burden non-refitting strategies with extra requirements
/// - Allow refitting logic to be optional and composable
/// - Keep the core `Strategy` trait simple
///
/// # Design
///
/// The trait takes a slice of `(candidate, observation)` tuples rather than
/// a `Leaderboard` directly. This allows the caller to:
/// - Select which trials to use (top-k, Pareto front, all, etc.)
/// - Pre-filter or transform trials as needed
/// - Work with any selection method without coupling to the leaderboard
///
/// The strategy receives candidates in **domain space** and must convert them
/// to its internal representation (e.g., unit hypercube) using the provided space.
///
/// # Example
///
/// ```no_run
/// use opt_engine::{ContinuousSpace, GmmStrategy, RefittableStrategy};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let space = ContinuousSpace::new(0.0, 1.0);
/// let mut strategy = GmmStrategy::<ContinuousSpace>::uniform_prior(42, 1, 0.1)?;
/// let completed_trials = vec![(0.2, 0.8), (0.4, 0.3), (0.7, 0.5)];
///
/// // Each tuple contains a domain-space candidate and its observation.
/// strategy.refit(&space, &completed_trials);
/// # Ok(())
/// # }
/// ```
pub trait RefittableStrategy: Strategy {
    /// Rebuild the strategy's internal model from selected trials.
    ///
    /// Candidates arrive in **domain space**; the strategy must convert them
    /// to its internal representation (e.g., unit hypercube) using `space`.
    /// The caller decides which trials to pass (top-k, Pareto front, etc.).
    fn refit(
        &mut self,
        space: &Self::Space,
        trials: &[(<Self::Space as SampleSpace>::Domain, Self::Observation)],
    );

    /// Reconcile a strategy produced by an off-lock refit with the strategy
    /// currently installed in the engine, before it is swapped back in.
    ///
    /// `self` is the freshly refit strategy carrying the new model; `live` is
    /// the strategy in the engine, which may have advanced its sampling
    /// counters via concurrent `suggest`/`update` calls while the refit ran on
    /// a blocking thread. Implementations copy that volatile sampling state
    /// forward from `live` into `self`, so swapping `self` back in does not
    /// rewind trial counters or replay an already-issued sample.
    ///
    /// The default keeps `self` unchanged, which is correct for strategies that
    /// carry no concurrently-mutated sampling state.
    fn reconcile_after_refit(&mut self, live: &Self)
    where
        Self: Sized,
    {
        let _ = live;
    }
}

/// Controls when and how a [`RefittableStrategy`] is automatically refit
/// during optimization.
///
/// The engine checks `should_refit(n)` after each ingestion. When it fires,
/// `selection_count(n)` determines how many of the best trials are fed to
/// [`RefittableStrategy::refit`].
#[derive(Clone, Debug)]
pub struct RefitConfig {
    min_trials: usize,
    refit_interval: usize,
    top_k: Option<usize>,
    top_quantile: Option<f64>,
}

impl Default for RefitConfig {
    fn default() -> Self {
        Self {
            min_trials: 20,
            refit_interval: 10,
            top_k: None,
            top_quantile: Some(0.25),
        }
    }
}

impl RefitConfig {
    /// Minimum completed-trial count before refitting can begin.
    pub fn min_trials(&self) -> usize {
        self.min_trials
    }

    /// Number of completed trials between refits.
    pub fn refit_interval(&self) -> usize {
        self.refit_interval
    }

    /// Fixed number of elite trials selected, when configured.
    pub fn top_k(&self) -> Option<usize> {
        self.top_k
    }

    /// Elite quantile selected, when configured.
    pub fn top_quantile(&self) -> Option<f64> {
        self.top_quantile
    }

    /// Validate this refit configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.refit_interval == 0 {
            return Err("refit_interval must be at least 1".to_string());
        }
        if self.top_k == Some(0) {
            return Err("top_k must be at least 1".to_string());
        }
        if let Some(quantile) = self.top_quantile {
            if !quantile.is_finite() || quantile <= 0.0 || quantile > 1.0 {
                return Err(format!(
                    "top_quantile must be finite and in (0, 1], got {quantile}"
                ));
            }
        }
        if self.top_k.is_some() && self.top_quantile.is_some() {
            return Err("top_k and top_quantile are mutually exclusive".to_string());
        }
        Ok(())
    }

    /// Construct a validated top-k refit policy.
    pub fn try_with_top_k(
        min_trials: usize,
        refit_interval: usize,
        top_k: usize,
    ) -> Result<Self, String> {
        let config = Self {
            min_trials,
            refit_interval,
            top_k: Some(top_k),
            top_quantile: None,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_top_k(min_trials: usize, refit_interval: usize, top_k: usize) -> Self {
        Self::try_with_top_k(min_trials, refit_interval, top_k)
            .unwrap_or_else(|error| panic!("RefitConfig::with_top_k: {error}"))
    }

    /// Construct a validated top-quantile refit policy.
    pub fn try_with_quantile(
        min_trials: usize,
        refit_interval: usize,
        quantile: f64,
    ) -> Result<Self, String> {
        let config = Self {
            min_trials,
            refit_interval,
            top_k: None,
            top_quantile: Some(quantile),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_quantile(min_trials: usize, refit_interval: usize, quantile: f64) -> Self {
        Self::try_with_quantile(min_trials, refit_interval, quantile)
            .unwrap_or_else(|error| panic!("RefitConfig::with_quantile: {error}"))
    }

    pub fn should_refit(&self, n_trials: usize) -> bool {
        // Defend against corrupt in-memory state as well as constructor input.
        if self.validate().is_err() {
            return false;
        }
        n_trials >= self.min_trials
            && (n_trials - self.min_trials).is_multiple_of(self.refit_interval)
    }

    pub fn selection_count(&self, n_trials: usize) -> usize {
        if let Some(k) = self.top_k {
            if n_trials == 0 {
                0
            } else {
                k.max(1).min(n_trials)
            }
        } else if let Some(q) = self.top_quantile {
            let q = if q.is_finite() && q > 0.0 && q <= 1.0 {
                q
            } else {
                1.0
            };
            let count = ((n_trials as f64) * q).ceil() as usize;
            // Ensure at least one trial is selected when there are trials to
            // select from, so a tiny quantile never yields an empty refit set.
            if n_trials > 0 { count.max(1) } else { count }
        } else {
            n_trials
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refit_config_default() {
        let config = RefitConfig::default();
        assert_eq!(config.min_trials, 20);
        assert_eq!(config.refit_interval, 10);
        assert!(config.top_k.is_none());
        assert_eq!(config.top_quantile, Some(0.25));
    }

    #[test]
    fn test_refit_config_with_top_k() {
        let config = RefitConfig::with_top_k(10, 5, 7);
        assert_eq!(config.min_trials, 10);
        assert_eq!(config.refit_interval, 5);
        assert_eq!(config.top_k, Some(7));
        assert!(config.top_quantile.is_none());

        assert!(!config.should_refit(0));
        assert!(!config.should_refit(9));
        assert!(config.should_refit(10));
        assert!(!config.should_refit(11));
        assert!(config.should_refit(15));
        assert!(config.should_refit(20));

        assert_eq!(config.selection_count(100), 7);
        assert_eq!(config.selection_count(3), 3);
    }

    #[test]
    fn test_refit_config_with_quantile() {
        let config = RefitConfig::with_quantile(5, 3, 0.1);
        assert_eq!(config.min_trials, 5);
        assert_eq!(config.refit_interval, 3);
        assert!(config.top_k.is_none());
        assert_eq!(config.top_quantile, Some(0.1));

        assert!(!config.should_refit(4));
        assert!(config.should_refit(5));
        assert!(config.should_refit(8));
        assert!(!config.should_refit(6));

        assert_eq!(config.selection_count(100), 10);
        assert_eq!(config.selection_count(7), 1);
    }

    #[test]
    #[should_panic(expected = "top_quantile must be finite and in (0, 1]")]
    fn test_with_quantile_rejects_zero() {
        RefitConfig::with_quantile(5, 3, 0.0);
    }

    #[test]
    #[should_panic(expected = "top_quantile must be finite and in (0, 1]")]
    fn test_with_quantile_rejects_above_one() {
        RefitConfig::with_quantile(5, 3, 1.5);
    }

    #[test]
    fn test_selection_count_floors_to_one() {
        // A tiny quantile must still select at least one trial when there are
        // trials available, so a refit never gets an empty selection.
        let config = RefitConfig::with_quantile(1, 1, 1e-6);
        assert_eq!(config.selection_count(10), 1);
        assert_eq!(config.selection_count(1), 1);
        // With no trials, the count is zero (nothing to floor to).
        assert_eq!(config.selection_count(0), 0);
    }

    #[test]
    fn test_try_constructors_reject_invalid_values() {
        assert!(RefitConfig::try_with_top_k(5, 0, 2).is_err());
        assert!(RefitConfig::try_with_top_k(5, 1, 0).is_err());
        assert!(RefitConfig::try_with_quantile(5, 0, 0.25).is_err());
        for quantile in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(RefitConfig::try_with_quantile(5, 1, quantile).is_err());
        }
    }

    #[test]
    fn test_defensive_handling_of_invalid_internal_values_never_panics() {
        let zero_interval = RefitConfig {
            min_trials: 0,
            refit_interval: 0,
            top_k: Some(1),
            top_quantile: None,
        };
        assert!(!zero_interval.should_refit(1));

        let zero_top_k = RefitConfig {
            min_trials: 0,
            refit_interval: 1,
            top_k: Some(0),
            top_quantile: None,
        };
        assert!(!zero_top_k.should_refit(1));
        assert_eq!(zero_top_k.selection_count(10), 1);

        let nan_quantile = RefitConfig {
            min_trials: 0,
            refit_interval: 1,
            top_k: None,
            top_quantile: Some(f64::NAN),
        };
        assert!(!nan_quantile.should_refit(1));
        assert_eq!(nan_quantile.selection_count(10), 10);
    }
}
