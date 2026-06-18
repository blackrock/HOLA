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

//! A type-safe hyperparameter optimization toolkit.
//!
//! `opt_engine` provides the core building blocks for black-box optimization:
//! define a parameter space and choose a search strategy that proposes
//! candidate configurations and incorporates observed results.
//!
//! # Key types
//!
//! - **Spaces** define the parameter domain (`ContinuousSpace`, `DiscreteSpace`,
//!   `CategoricalSpace`, and combinators like `ProductSpace` / `BranchingSpace`).
//! - **Scales** transform continuous ranges (e.g., `LogScale`, `Log10Scale`).
//! - **Strategies** are the search algorithms (`RandomStrategy`,
//!   `SobolStrategy`, `GmmStrategy`).
//!
//! For the type-erased HOLA frontend (`HolaEngine`, Python bindings, CLI, and
//! REST server), see the `hola` crate which builds on top of `opt_engine`.

pub mod leaderboard;
pub mod objectives;
pub mod persistence;
pub mod scales;
pub mod spaces;
pub mod strategies;
pub mod traits;

#[cfg(test)]
pub(crate) mod test_helpers;

/// Commonly used types, re-exported for convenience.
pub mod prelude {
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
        RefitConfig, RefittableStrategy, SampleSpace, StandardizedSpace, Strategy,
    };
}

// Re-export at crate root for convenience
pub use leaderboard::{Leaderboard, RankedTrial, Trial};
pub use persistence::{
    AutoCheckpointConfig, Checkpoint, CheckpointMetadata, LeaderboardCheckpoint,
};
pub use scales::{LinearScale, Log10Scale, LogScale, Scale};
pub use spaces::{
    BranchingSpace, CategoricalSpace, ContinuousSpace, DiscreteSpace, EitherDomain, ProductSpace,
};
pub use strategies::{GmmRefitConfig, GmmStrategy, RandomStrategy, SobolStrategy};
pub use traits::{RefitConfig, RefittableStrategy, SampleSpace, StandardizedSpace, Strategy};
