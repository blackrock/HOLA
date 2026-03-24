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

//! Search strategies for suggesting candidate configurations.
//!
//! Three built-in strategies cover the typical optimization lifecycle:
//!
//! - [`RandomStrategy`] — uniform random sampling; useful as a baseline.
//! - [`SobolStrategy`] — quasi-random Sobol’ sequences for space-filling
//!   exploration with better coverage than pseudo-random sampling.
//! - [`GmmStrategy`] — samples from a Gaussian mixture model fit to elite
//!   trials; enables informed search after an initial exploration phase.

mod gmm;
mod random;
mod sobol;

pub use gmm::{GaussianComponent, GmmParams, GmmRefitConfig, GmmStrategy};
pub use random::RandomStrategy;
pub use sobol::SobolStrategy;
