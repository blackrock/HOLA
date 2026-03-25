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

//! Categorical parameter space (pick one of N string labels).

use crate::traits::{SampleSpace, StandardizedSpace};
use serde::{Deserialize, Serialize};

/// Categorical space representing a choice among named options.
///
/// Internally maps to a single `[0, 1]` dimension using uniform quantile bins,
/// identical to `DiscreteSpace` but with string labels instead of integer values.
///
/// For example, with choices `["adam", "sgd", "rmsprop"]` (3 choices):
/// - "adam"    ↔ bucket [0, 1/3),    center at 1/6
/// - "sgd"     ↔ bucket [1/3, 2/3),  center at 1/2
/// - "rmsprop" ↔ bucket [2/3, 1],    center at 5/6
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CategoricalSpace {
    pub choices: Vec<String>,
}

impl CategoricalSpace {
    /// # Panics
    /// Panics if `choices` is empty.
    pub fn new(choices: Vec<String>) -> Self {
        assert!(
            !choices.is_empty(),
            "CategoricalSpace: choices must not be empty"
        );
        Self { choices }
    }

    /// Convenience constructor from string slices.
    pub fn from_strs(choices: &[&str]) -> Self {
        Self::new(choices.iter().map(|s| s.to_string()).collect())
    }

    pub fn cardinality(&self) -> usize {
        self.choices.len()
    }
}

impl SampleSpace for CategoricalSpace {
    type Domain = String;

    fn contains(&self, point: &String) -> bool {
        self.choices.contains(point)
    }

    fn clamp(&self, point: &String) -> String {
        if self.choices.contains(point) {
            point.clone()
        } else {
            self.choices[0].clone()
        }
    }
}

impl StandardizedSpace for CategoricalSpace {
    fn dimensionality(&self) -> usize {
        1
    }

    fn to_unit_cube(&self, point: &String) -> Vec<f64> {
        let n = self.cardinality() as f64;
        let index = self.choices.iter().position(|c| c == point).unwrap_or(0) as f64;
        // Map to the center of the bucket, same pattern as DiscreteSpace
        vec![(index + 0.5) / n]
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<String> {
        if vec.len() != 1 {
            return None;
        }
        let val = vec[0].clamp(0.0, 1.0);
        let n = self.cardinality() as f64;
        let index = (val * n).floor() as usize;
        // Clamp to valid range (handles edge case where val = 1.0 exactly)
        let index = index.min(self.choices.len() - 1);
        Some(self.choices[index].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let space = CategoricalSpace::from_strs(&["adam", "sgd", "rmsprop"]);
        assert_eq!(space.cardinality(), 3);
        assert!(space.contains(&"adam".to_string()));
        assert!(!space.contains(&"unknown".to_string()));
    }

    #[test]
    fn test_clamp() {
        let space = CategoricalSpace::from_strs(&["a", "b", "c"]);
        assert_eq!(space.clamp(&"b".to_string()), "b");
        assert_eq!(space.clamp(&"unknown".to_string()), "a");
    }

    #[test]
    fn test_unit_cube_roundtrip() {
        let space = CategoricalSpace::from_strs(&["adam", "sgd", "rmsprop"]);
        for choice in &space.choices {
            let unit = space.to_unit_cube(choice);
            assert_eq!(unit.len(), 1);
            assert!(unit[0] >= 0.0 && unit[0] <= 1.0);
            let restored = space.from_unit_cube(&unit).unwrap();
            assert_eq!(&restored, choice);
        }
    }

    #[test]
    fn test_from_unit_cube_boundaries() {
        let space = CategoricalSpace::from_strs(&["a", "b", "c"]);
        assert_eq!(space.from_unit_cube(&[0.0]).unwrap(), "a");
        assert_eq!(space.from_unit_cube(&[0.5]).unwrap(), "b");
        assert_eq!(space.from_unit_cube(&[1.0]).unwrap(), "c"); // edge case
    }

    #[test]
    fn test_from_unit_cube_wrong_dims() {
        let space = CategoricalSpace::from_strs(&["a", "b"]);
        assert!(space.from_unit_cube(&[]).is_none());
        assert!(space.from_unit_cube(&[0.5, 0.5]).is_none());
    }

    #[test]
    #[should_panic(expected = "choices must not be empty")]
    fn test_empty_choices_panics() {
        CategoricalSpace::new(vec![]);
    }

    #[test]
    fn test_large_space() {
        let choices: Vec<String> = (0..100).map(|i| format!("choice_{i}")).collect();
        let space = CategoricalSpace::new(choices.clone());
        assert_eq!(space.cardinality(), 100);

        for choice in &choices {
            assert!(space.contains(choice));
            let unit = space.to_unit_cube(choice);
            assert_eq!(unit.len(), 1);
            assert!(unit[0] >= 0.0 && unit[0] <= 1.0);
            let restored = space.from_unit_cube(&unit).unwrap();
            assert_eq!(&restored, choice, "roundtrip failed for {choice}");
        }
    }

    #[test]
    fn test_single_choice() {
        let space = CategoricalSpace::from_strs(&["only"]);
        assert_eq!(space.cardinality(), 1);
        assert!(space.contains(&"only".to_string()));

        let unit = space.to_unit_cube(&"only".to_string());
        let restored = space.from_unit_cube(&unit).unwrap();
        assert_eq!(restored, "only");

        assert_eq!(space.from_unit_cube(&[0.0]).unwrap(), "only");
        assert_eq!(space.from_unit_cube(&[1.0]).unwrap(), "only");
    }

    #[test]
    fn test_clamp_unknown() {
        let space = CategoricalSpace::from_strs(&["first", "second", "third"]);
        assert_eq!(space.clamp(&"nonexistent".to_string()), "first");
    }

    #[test]
    fn test_from_strs_convenience() {
        let space = CategoricalSpace::from_strs(&["a", "b", "c"]);
        assert_eq!(space.cardinality(), 3);
        assert!(space.contains(&"a".to_string()));
        assert!(space.contains(&"b".to_string()));
        assert!(space.contains(&"c".to_string()));
    }
}
