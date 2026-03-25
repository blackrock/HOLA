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

//! Parameter space definitions and combinators.
//!
//! A *space* defines the feasible region of hyperparameters: what values are
//! valid, how to clamp out-of-bounds proposals, and how to map between the
//! user's domain types and the `[0, 1]^n` unit hypercube that strategies
//! operate in.
//!
//! Primitive spaces:
//! - [`ContinuousSpace`] — a real-valued range with optional scale transform
//! - [`DiscreteSpace`] — an integer range
//! - [`CategoricalSpace`] — one of N string labels
//!
//! Combinators for composing primitives:
//! - [`ProductSpace`] — Cartesian product `A × B`
//! - [`BranchingSpace`] — disjoint union `A | B`

mod categorical;
mod combinators;
mod continuous;
mod discrete;

pub use categorical::CategoricalSpace;
pub use combinators::{BranchingSpace, EitherDomain, ProductSpace};
pub use continuous::ContinuousSpace;
pub use discrete::DiscreteSpace;
