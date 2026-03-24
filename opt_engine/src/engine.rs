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

//! The optimization engine that orchestrates spaces, strategies, and transformers.

use crate::leaderboard::Leaderboard;
use crate::persistence::{Checkpoint, LeaderboardCheckpoint};
use crate::traits::{RefitConfig, RefittableStrategy, SampleSpace, Strategy, Transformer};
use serde::{Serialize, de::DeserializeOwned};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Strategy and leaderboard bundled under a single lock so that concurrent
/// ingestions cannot interleave and leave the two out of sync.
struct EngineState<S, Sp>
where
    Sp: SampleSpace,
    S: Strategy<Space = Sp>,
{
    strategy: S,
    leaderboard: Option<Leaderboard<Sp::Domain, S::Observation>>,
}

/// The core optimization engine.
///
/// Coordinates the interaction between:
/// - A `SampleSpace` defining the parameter domain
/// - A `Strategy` for selecting candidates
/// - A `Transformer` for processing results
/// - An optional `Leaderboard` for trial history
///
/// # Type Safety
/// The compiler statically verifies that `Transformer::Output == Strategy::Observation`.
///
/// # Atomicity
/// The strategy and leaderboard are protected by a single `RwLock`, ensuring that
/// concurrent ingestions cannot interleave and diverge the two.
///
/// # Example
///
/// ```ignore
/// // Basic usage (no leaderboard)
/// let engine = Engine::new(space, strategy, transformer);
///
/// // With leaderboard for persistence
/// let engine = Engine::with_leaderboard(space, strategy, transformer);
///
/// // With custom configuration
/// let engine = Engine::builder(space, strategy, transformer)
///     .with_leaderboard()
///     .with_refit_config(RefitConfig::default())
///     .build();
/// ```
pub struct Engine<S, T, Sp>
where
    Sp: SampleSpace,
    S: Strategy<Space = Sp>,
    T: Transformer,
    T: Transformer<Output = S::Observation>,
{
    space: Sp,
    state: Arc<RwLock<EngineState<S, Sp>>>,
    transformer: T,
    refit_config: Option<RefitConfig>,
    has_leaderboard: bool,
}

impl<S, T, Sp> Engine<S, T, Sp>
where
    Sp: SampleSpace,
    S: Strategy<Space = Sp>,
    T: Transformer<Output = S::Observation>,
{
    /// Create a new engine with the given space, strategy, and transformer.
    ///
    /// This creates an engine without a leaderboard. Use `with_leaderboard()`
    /// or `builder()` if you need trial history tracking.
    pub fn new(space: Sp, strategy: S, transformer: T) -> Self {
        Self {
            space,
            state: Arc::new(RwLock::new(EngineState {
                strategy,
                leaderboard: None,
            })),
            transformer,
            refit_config: None,
            has_leaderboard: false,
        }
    }

    /// Create a new engine with leaderboard tracking enabled.
    pub fn with_leaderboard(space: Sp, strategy: S, transformer: T) -> Self {
        Self {
            space,
            state: Arc::new(RwLock::new(EngineState {
                strategy,
                leaderboard: Some(Leaderboard::new()),
            })),
            transformer,
            refit_config: None,
            has_leaderboard: true,
        }
    }

    /// Create a builder for more complex engine configuration.
    pub fn builder(space: Sp, strategy: S, transformer: T) -> EngineBuilder<S, T, Sp> {
        EngineBuilder::new(space, strategy, transformer)
    }

    /// Outbound pipeline: strategy proposes a candidate, the engine validates
    /// and clamps it, then serializes to JSON for the worker.
    pub async fn dispatch_job(&self) -> serde_json::Value {
        let state = self.state.read().await;

        // 1. Suggest Candidate
        let candidate = state.strategy.suggest(&self.space);

        // 2. Validate and clamp if needed
        let candidate = if !self.space.contains(&candidate) {
            eprintln!(
                "Warning: Strategy suggested out-of-bounds candidate, clamping to valid range"
            );
            self.space.clamp(&candidate)
        } else {
            candidate
        };

        // 3. Serialize (Strict Type -> JSON)
        serde_json::to_value(&candidate).expect("Serialization failed")
    }

    /// Inbound pipeline: the engine deserializes the candidate, transforms the
    /// raw result, and atomically updates both the leaderboard and the strategy
    /// under a single lock.
    pub async fn ingest_result(
        &self,
        candidate_json: serde_json::Value,
        raw_result: T::ForeignInput,
    ) -> Result<(), String> {
        let candidate: Sp::Domain =
            serde_json::from_value(candidate_json).map_err(|e| format!("DB Error: {e}"))?;

        let observation = self.transformer.transform(raw_result)?;

        // Atomic update: single lock for both leaderboard and strategy
        let mut state = self.state.write().await;
        if let Some(ref mut lb) = state.leaderboard {
            lb.push(candidate.clone(), observation.clone());
        }
        state.strategy.update(&candidate, observation);

        Ok(())
    }

    pub fn has_leaderboard(&self) -> bool {
        self.has_leaderboard
    }

    pub async fn trial_count(&self) -> usize {
        let state = self.state.read().await;
        state.leaderboard.as_ref().map_or(0, |lb| lb.len())
    }

    pub fn space(&self) -> &Sp {
        &self.space
    }
}

// =============================================================================
// Leaderboard Access
// =============================================================================

impl<S, T, Sp> Engine<S, T, Sp>
where
    Sp: SampleSpace,
    S: Strategy<Space = Sp>,
    T: Transformer<Output = S::Observation>,
    Sp::Domain: Clone,
    S::Observation: Clone,
{
    pub async fn leaderboard(&self) -> Option<Leaderboard<Sp::Domain, S::Observation>> {
        self.state.read().await.leaderboard.clone()
    }

    /// Execute a function with read access to the leaderboard.
    pub async fn with_leaderboard_ref<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&Leaderboard<Sp::Domain, S::Observation>) -> R,
    {
        let state = self.state.read().await;
        state.leaderboard.as_ref().map(f)
    }
}

// =============================================================================
// Persistence
// =============================================================================

impl<S, T, Sp> Engine<S, T, Sp>
where
    Sp: SampleSpace,
    S: Strategy<Space = Sp>,
    T: Transformer<Output = S::Observation>,
    Sp::Domain: Serialize + DeserializeOwned + Clone,
    S::Observation: Serialize + DeserializeOwned + Clone,
{
    /// Save a leaderboard-only checkpoint (no strategy state).
    pub async fn save_leaderboard_checkpoint(
        &self,
        path: impl AsRef<Path>,
        description: Option<&str>,
    ) -> std::io::Result<()> {
        let state = self.state.read().await;
        let lb = state.leaderboard.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No leaderboard configured",
            )
        })?;

        LeaderboardCheckpoint::new(lb.clone(), description).save_json(path)
    }

    /// Load trials from a leaderboard checkpoint into this engine.
    pub async fn load_leaderboard_checkpoint(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let checkpoint: LeaderboardCheckpoint<Sp::Domain, S::Observation> =
            LeaderboardCheckpoint::load_json(path)?;

        let mut state = self.state.write().await;
        if state.leaderboard.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No leaderboard configured",
            ));
        }
        state.leaderboard = Some(checkpoint.leaderboard);
        Ok(())
    }
}

impl<S, T, Sp> Engine<S, T, Sp>
where
    Sp: SampleSpace,
    S: Strategy<Space = Sp> + Serialize + DeserializeOwned + Clone,
    T: Transformer<Output = S::Observation>,
    Sp::Domain: Serialize + DeserializeOwned + Clone,
    S::Observation: Serialize + DeserializeOwned + Clone,
{
    /// Save a full checkpoint (leaderboard + strategy state).
    pub async fn save_checkpoint(
        &self,
        path: impl AsRef<Path>,
        description: Option<&str>,
    ) -> std::io::Result<()> {
        let state = self.state.read().await;
        let lb = state
            .leaderboard
            .as_ref()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "No leaderboard configured",
                )
            })?
            .clone();

        let strategy_state = state.strategy.clone();

        Checkpoint::new(lb, strategy_state, description).save_json(path)
    }

    /// Load a full checkpoint, restoring both leaderboard and strategy state.
    pub async fn load_checkpoint(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let checkpoint: Checkpoint<Sp::Domain, S::Observation, S> = Checkpoint::load_json(path)?;

        let mut state = self.state.write().await;
        if state.leaderboard.is_some() {
            state.leaderboard = Some(checkpoint.leaderboard);
        }
        state.strategy = checkpoint.strategy_state;

        Ok(())
    }
}

// =============================================================================
// Refitting Support
// =============================================================================

impl<S, T, Sp> Engine<S, T, Sp>
where
    Sp: SampleSpace + Clone + Send + 'static,
    S: Strategy<Space = Sp> + RefittableStrategy + Clone + Send + 'static,
    T: Transformer<Output = S::Observation>,
    Sp::Domain: Clone + Send + 'static,
    S::Observation: Clone + Send + 'static,
{
    /// Manually trigger a refit of the strategy from the leaderboard.
    /// The refit is offloaded to a blocking thread to avoid starving the async executor.
    pub async fn refit<F>(&self, selector: F) -> Result<(), String>
    where
        F: FnOnce(
            &Leaderboard<Sp::Domain, S::Observation>,
        ) -> Vec<crate::leaderboard::Trial<Sp::Domain, S::Observation>>,
    {
        let state = self.state.read().await;
        let lb = state
            .leaderboard
            .as_ref()
            .ok_or("No leaderboard configured")?;

        let selected = selector(lb);
        let strategy_snapshot = state.strategy.clone();
        let space_clone = self.space.clone();
        drop(state);

        let trials: Vec<(Sp::Domain, S::Observation)> = selected
            .into_iter()
            .map(|t| (t.candidate, t.observation))
            .collect();

        let fitted = tokio::task::spawn_blocking(move || {
            let mut s = strategy_snapshot;
            s.refit(&space_clone, &trials);
            s
        })
        .await
        .map_err(|e| e.to_string())?;

        self.state.write().await.strategy = fitted;

        Ok(())
    }
}

// Specialization for scalar observations (f64) with auto-refit
impl<S, T, Sp> Engine<S, T, Sp>
where
    Sp: SampleSpace + Clone + Send + 'static,
    S: Strategy<Space = Sp, Observation = f64> + RefittableStrategy + Clone + Send + 'static,
    T: Transformer<Output = f64>,
    Sp::Domain: Clone + Send + 'static,
{
    /// Ingest a result and automatically refit if configured.
    pub async fn ingest_result_with_refit(
        &self,
        candidate_json: serde_json::Value,
        raw_result: T::ForeignInput,
    ) -> Result<(), String> {
        let candidate: Sp::Domain =
            serde_json::from_value(candidate_json).map_err(|e| format!("DB Error: {e}"))?;

        let observation = self.transformer.transform(raw_result)?;

        // Atomic ingestion: single lock for leaderboard + strategy update
        let n_trials = {
            let mut state = self.state.write().await;
            let n = if let Some(ref mut lb) = state.leaderboard {
                lb.push(candidate.clone(), observation);
                lb.len()
            } else {
                0
            };
            state.strategy.update(&candidate, observation);
            n
        };

        // Check if we should refit (offloaded to blocking thread)
        if let Some(ref config) = self.refit_config
            && config.should_refit(n_trials)
        {
            let state = self.state.read().await;
            if let Some(ref lb) = state.leaderboard {
                let k = config.selection_count(n_trials);
                let top_trials = lb.top_k(k);

                let trials: Vec<(Sp::Domain, f64)> = top_trials
                    .into_iter()
                    .map(|t| (t.candidate, t.observation))
                    .collect();

                let strategy_snapshot = state.strategy.clone();
                let space_clone = self.space.clone();
                drop(state);

                let fitted = tokio::task::spawn_blocking(move || {
                    let mut s = strategy_snapshot;
                    s.refit(&space_clone, &trials);
                    s
                })
                .await
                .map_err(|e| e.to_string())?;

                self.state.write().await.strategy = fitted;
            }
        }

        Ok(())
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for configuring an Engine with optional features.
pub struct EngineBuilder<S, T, Sp>
where
    Sp: SampleSpace,
    S: Strategy<Space = Sp>,
    T: Transformer<Output = S::Observation>,
{
    space: Sp,
    strategy: S,
    transformer: T,
    leaderboard: Option<Leaderboard<Sp::Domain, S::Observation>>,
    refit_config: Option<RefitConfig>,
}

impl<S, T, Sp> EngineBuilder<S, T, Sp>
where
    Sp: SampleSpace,
    S: Strategy<Space = Sp>,
    T: Transformer<Output = S::Observation>,
{
    pub fn new(space: Sp, strategy: S, transformer: T) -> Self {
        Self {
            space,
            strategy,
            transformer,
            leaderboard: None,
            refit_config: None,
        }
    }

    /// Enable leaderboard tracking with an empty leaderboard.
    pub fn with_leaderboard(mut self) -> Self {
        self.leaderboard = Some(Leaderboard::new());
        self
    }

    /// Enable leaderboard tracking with a pre-populated leaderboard.
    pub fn with_existing_leaderboard(
        mut self,
        leaderboard: Leaderboard<Sp::Domain, S::Observation>,
    ) -> Self {
        self.leaderboard = Some(leaderboard);
        self
    }

    /// Configure automatic refitting (requires leaderboard).
    pub fn with_refit_config(mut self, config: RefitConfig) -> Self {
        self.refit_config = Some(config);
        if self.leaderboard.is_none() {
            self.leaderboard = Some(Leaderboard::new());
        }
        self
    }

    pub fn build(self) -> Engine<S, T, Sp> {
        let has_lb = self.leaderboard.is_some();
        Engine {
            space: self.space,
            state: Arc::new(RwLock::new(EngineState {
                strategy: self.strategy,
                leaderboard: self.leaderboard,
            })),
            transformer: self.transformer,
            refit_config: self.refit_config,
            has_leaderboard: has_lb,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{ConstantStrategy, IdentityTransformer, UnitInterval};
    use std::marker::PhantomData;

    #[tokio::test]
    async fn test_engine_without_leaderboard() {
        let engine = Engine::new(
            UnitInterval,
            ConstantStrategy {
                value: 0.5,
                _marker: PhantomData,
            },
            IdentityTransformer,
        );

        assert!(!engine.has_leaderboard());
        assert_eq!(engine.trial_count().await, 0);
    }

    #[tokio::test]
    async fn test_engine_with_leaderboard() {
        let engine = Engine::with_leaderboard(
            UnitInterval,
            ConstantStrategy {
                value: 0.5,
                _marker: PhantomData,
            },
            IdentityTransformer,
        );

        assert!(engine.has_leaderboard());

        let job = engine.dispatch_job().await;
        engine.ingest_result(job, 0.42).await.unwrap();

        assert_eq!(engine.trial_count().await, 1);

        let lb = engine.leaderboard().await.unwrap();
        assert_eq!(lb.len(), 1);
        assert_eq!(lb.trials()[0].observation, 0.42);
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let engine = Engine::builder(
            UnitInterval,
            ConstantStrategy {
                value: 0.5,
                _marker: PhantomData,
            },
            IdentityTransformer,
        )
        .with_leaderboard()
        .build();

        assert!(engine.has_leaderboard());
    }
}
