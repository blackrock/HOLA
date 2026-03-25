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

//! A type-safe hyperparameter optimization engine.
//!
//! `opt_engine` provides the core building blocks for black-box optimization:
//! define a parameter space, choose a search strategy, and run an ask/tell loop
//! that proposes candidate configurations and incorporates observed results.
//!
//! # Key types
//!
//! - **Spaces** define the parameter domain (`ContinuousSpace`, `DiscreteSpace`,
//!   `CategoricalSpace`, and combinators like `ProductSpace` / `BranchingSpace`).
//! - **Scales** transform continuous ranges (e.g., `LogScale`, `Log10Scale`).
//! - **Strategies** are the search algorithms (`RandomStrategy`,
//!   `SobolStrategy`, `GmmStrategy`).
//! - **Transformers** convert raw worker output into typed observations.
//! - **[`Engine`]** orchestrates the loop with full compile-time type checking.
//!
//! For the type-erased HOLA frontend (`DynEngine`, REST server), see the `hola`
//! crate which builds on top of `opt_engine`.
//!
//! # Quick start
//!
//! ```ignore
//! use opt_engine::prelude::*;
//!
//! let space = ProductSpace {
//!     a: ContinuousSpace::with_scale(1e-4, 0.1, Log10Scale), // learning rate
//!     b: DiscreteSpace::new(1, 10), // num layers
//! };
//!
//! let engine = Engine::new(space, RandomStrategy::auto_seed(), JsonFieldTransformer::default());
//! ```

pub mod engine;
pub mod leaderboard;
pub mod objectives;
pub mod persistence;
pub mod scales;
pub mod spaces;
pub mod strategies;
pub mod traits;
pub mod transformers;

#[cfg(test)]
pub(crate) mod test_helpers;

/// Commonly used types, re-exported for convenience.
pub mod prelude {
    pub use crate::engine::Engine;
    pub use crate::leaderboard::{Leaderboard, RankedTrial, Trial};
    pub use crate::persistence::{
        AutoCheckpointConfig, Checkpoint, CheckpointMetadata, LeaderboardCheckpoint,
    };
    pub use crate::scales::{LinearScale, Log10Scale, LogScale, Scale};
    pub use crate::spaces::{
        BranchingSpace, CategoricalSpace, ContinuousSpace, DiscreteSpace, EitherDomain,
        ProductSpace,
    };
    pub use crate::strategies::{GmmRefitConfig, GmmStrategy, RandomStrategy, SobolStrategy};
    pub use crate::traits::{
        RefitConfig, RefittableStrategy, SampleSpace, StandardizedSpace, Strategy, Transformer,
    };
    pub use crate::transformers::{
        GroupedTlpField, JsonFieldTransformer, JsonGroupedTlpTransformer, JsonTlpTransformer,
        JsonWeightedTransformer, TlpField, WeightedField,
    };
}

// Re-export at crate root for convenience
pub use engine::Engine;
pub use leaderboard::{Leaderboard, RankedTrial, Trial, is_feasible_multi, is_feasible_scalar};
pub use persistence::{
    AutoCheckpointConfig, Checkpoint, CheckpointMetadata, LeaderboardCheckpoint,
};
pub use scales::{LinearScale, Log10Scale, LogScale, Scale};
pub use spaces::{
    BranchingSpace, CategoricalSpace, ContinuousSpace, DiscreteSpace, EitherDomain, ProductSpace,
};
pub use strategies::{GmmRefitConfig, GmmStrategy, RandomStrategy, SobolStrategy};
pub use traits::{
    RefitConfig, RefittableStrategy, SampleSpace, StandardizedSpace, Strategy, Transformer,
};
pub use transformers::{
    GroupedTlpField, JsonFieldTransformer, JsonGroupedTlpTransformer, JsonTlpTransformer,
    JsonWeightedTransformer, TlpField, WeightedField,
};
