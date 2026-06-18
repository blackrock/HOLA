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

//! Shared mock types for unit tests across the crate.
//!
//! This module is only compiled in `#[cfg(test)]` mode.
//! Import with `use crate::test_helpers::*;` in any `mod tests` block.

use crate::traits::{SampleSpace, StandardizedSpace};

// =============================================================================
// Mock Spaces
// =============================================================================

/// A 1-D test space representing [0, 1].
pub struct UnitInterval;

impl SampleSpace for UnitInterval {
    type Domain = f64;

    fn contains(&self, point: &Self::Domain) -> bool {
        *point >= 0.0 && *point <= 1.0
    }
}

impl StandardizedSpace for UnitInterval {
    fn dimensionality(&self) -> usize {
        1
    }

    fn to_unit_cube(&self, point: &Self::Domain) -> Vec<f64> {
        vec![*point]
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<Self::Domain> {
        vec.first().copied()
    }
}

/// A 2-D test space representing [0, 1] × [0, 1].
pub struct UnitSquare;

impl SampleSpace for UnitSquare {
    type Domain = (f64, f64);

    fn contains(&self, point: &Self::Domain) -> bool {
        point.0 >= 0.0 && point.0 <= 1.0 && point.1 >= 0.0 && point.1 <= 1.0
    }
}

impl StandardizedSpace for UnitSquare {
    fn dimensionality(&self) -> usize {
        2
    }

    fn to_unit_cube(&self, point: &Self::Domain) -> Vec<f64> {
        vec![point.0, point.1]
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<Self::Domain> {
        if vec.len() == 2 {
            Some((vec[0], vec[1]))
        } else {
            None
        }
    }
}
