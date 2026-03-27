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
//! 4. **[`Transformer`]** — the trust boundary between raw external results
//!    (e.g., JSON from a worker) and the typed observations the strategy expects.
//! 5. **[`RefittableStrategy`]** — optional extension for strategies that can
//!    rebuild their internal model from a batch of historical trials (e.g., GMM
//!    refitting from the top quantile of the leaderboard).

use crate::leaderboard::{Leaderboard, Trial};
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Debug;

/// Defines the structure and validity rules for a hyperparameter search space.
///
/// Every optimization problem starts with a `SampleSpace` that describes what
/// a valid configuration looks like. The [`Engine`](crate::engine::Engine)
/// uses `contains` to validate strategy proposals and `clamp` to snap
/// out-of-bounds candidates back into the feasible region.
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

    /// The result type the strategy consumes (must match the
    /// [`Transformer::Output`] used by the engine).
    type Observation: Serialize + DeserializeOwned + Send + Sync + Clone + Debug;

    fn suggest(&self, space: &Self::Space) -> <Self::Space as SampleSpace>::Domain;

    fn update(
        &mut self,
        candidate: &<Self::Space as SampleSpace>::Domain,
        observation: Self::Observation,
    );
}

/// Converts raw external results into the typed observation the strategy expects.
///
/// A transformer has two responsibilities:
///
/// 1. **Schema validation** — parse untrusted input (typically JSON from a
///    worker), extract the relevant fields, and reject malformed data. This
///    is the only component in the pipeline that can fail at runtime; spaces
///    and strategies are statically verified.
/// 2. **Objective processing** — reduce the parsed fields into the
///    observation type the strategy consumes. This includes negating values
///    for maximization, computing weighted sums, or applying
///    target-limit-priority (TLP) scalarization.
///
/// The built-in implementations (e.g., [`JsonFieldTransformer`],
/// [`JsonWeightedTransformer`], [`JsonTlpTransformer`]) bundle both
/// concerns into a single step.
pub trait Transformer: Send + Sync + 'static {
    /// Raw input from the outside world (usually `serde_json::Value`).
    type ForeignInput: Debug + Send + Sync;

    /// Validated output (must equal the strategy's `Observation` type).
    type Output;

    fn transform(&self, input: Self::ForeignInput) -> Result<Self::Output, String>;
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
/// ```ignore
/// use opt_engine::traits::{RefittableStrategy, StandardizedSpace};
///
/// impl<S: StandardizedSpace> RefittableStrategy for GmmStrategy<S> {
///     fn refit(&mut self, space: &Self::Space, trials: &[(S::Domain, Self::Observation)]) {
///         // Convert candidates to unit cube
///         let samples: Vec<Vec<f64>> = trials
///             .iter()
///             .map(|(c, _)| space.to_unit_cube(c))
///             .collect();
///
///         // Fit GMM to the samples
///         self.fit_from_samples(&samples, 3, 100, 1e-6, 1e-4);
///     }
/// }
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

    /// Refit from a leaderboard using a custom selector.
    ///
    /// This is a convenience method that extracts trials from a leaderboard
    /// using the provided selector function.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Refit from top 20% of trials
    /// strategy.refit_from_leaderboard(&space, &leaderboard, |lb| {
    ///     lb.top_quantile(0.2)
    ///         .into_iter()
    ///         .map(|t| (t.candidate, t.observation))
    ///         .collect()
    /// });
    /// ```
    fn refit_from_leaderboard<F>(
        &mut self,
        space: &Self::Space,
        leaderboard: &Leaderboard<<Self::Space as SampleSpace>::Domain, Self::Observation>,
        selector: F,
    ) where
        F: FnOnce(
            &Leaderboard<<Self::Space as SampleSpace>::Domain, Self::Observation>,
        ) -> Vec<Trial<<Self::Space as SampleSpace>::Domain, Self::Observation>>,
        <Self::Space as SampleSpace>::Domain: Clone,
        Self::Observation: Clone,
    {
        let selected = selector(leaderboard);
        let trials: Vec<_> = selected
            .into_iter()
            .map(|t| (t.candidate, t.observation))
            .collect();
        self.refit(space, &trials);
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
    pub min_trials: usize,
    pub refit_interval: usize,
    pub top_k: Option<usize>,
    pub top_quantile: Option<f64>,
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
    pub fn with_top_k(min_trials: usize, refit_interval: usize, top_k: usize) -> Self {
        Self {
            min_trials,
            refit_interval,
            top_k: Some(top_k),
            top_quantile: None,
        }
    }

    pub fn with_quantile(min_trials: usize, refit_interval: usize, quantile: f64) -> Self {
        Self {
            min_trials,
            refit_interval,
            top_k: None,
            top_quantile: Some(quantile),
        }
    }

    pub fn should_refit(&self, n_trials: usize) -> bool {
        n_trials >= self.min_trials
            && (n_trials - self.min_trials).is_multiple_of(self.refit_interval)
    }

    pub fn selection_count(&self, n_trials: usize) -> usize {
        if let Some(k) = self.top_k {
            k.min(n_trials)
        } else if let Some(q) = self.top_quantile {
            ((n_trials as f64) * q).ceil() as usize
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
}
