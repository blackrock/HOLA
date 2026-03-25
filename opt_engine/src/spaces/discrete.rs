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

//! Discrete integer parameter space.

use crate::traits::{SampleSpace, StandardizedSpace};
use serde::{Deserialize, Serialize};

/// Discrete integer space over [min, max] inclusive.
///
/// Each integer in the range gets an equally-sized bucket in [0, 1] for standardization.
/// For example, with min=0, max=2 (3 integers):
/// - Integer 0 ↔ bucket [0, 1/3), center at 1/6
/// - Integer 1 ↔ bucket [1/3, 2/3), center at 1/2
/// - Integer 2 ↔ bucket [2/3, 1], center at 5/6
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscreteSpace {
    pub min: i64,
    pub max: i64,
}

impl DiscreteSpace {
    pub fn new(min: i64, max: i64) -> Self {
        assert!(min <= max, "DiscreteSpace: min must be <= max");
        Self { min, max }
    }

    pub fn cardinality(&self) -> usize {
        (self.max - self.min + 1) as usize
    }
}

impl SampleSpace for DiscreteSpace {
    type Domain = i64;

    fn contains(&self, point: &i64) -> bool {
        *point >= self.min && *point <= self.max
    }

    fn clamp(&self, point: &i64) -> i64 {
        (*point).clamp(self.min, self.max)
    }
}

impl StandardizedSpace for DiscreteSpace {
    fn dimensionality(&self) -> usize {
        1
    }

    fn to_unit_cube(&self, point: &i64) -> Vec<f64> {
        let n = self.cardinality() as f64;
        // Map integer to the center of its bucket
        // Integer i gets bucket [(i-min)/n, (i-min+1)/n), center at (i-min+0.5)/n
        let bucket_center = (*point - self.min) as f64 + 0.5;
        vec![bucket_center / n]
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<i64> {
        if vec.len() != 1 {
            return None;
        }
        let val = vec[0].clamp(0.0, 1.0);
        let n = self.cardinality() as f64;
        // Map from [0,1] to bucket index, then to actual integer
        // val * n gives us a value in [0, n], floor gives bucket index
        let index = (val * n).floor() as i64;
        // Clamp to valid range (handles edge case where val = 1.0 exactly)
        let index = index.min(self.max - self.min);
        Some(self.min + index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let space = DiscreteSpace::new(1, 10);
        assert_eq!(space.cardinality(), 10);
        assert!(space.contains(&5));
        assert!(!space.contains(&11));
        assert!(!space.contains(&0));
        assert_eq!(space.clamp(&15), 10);
        assert_eq!(space.clamp(&-5), 1);
    }

    #[test]
    #[should_panic(expected = "min must be <= max")]
    fn test_panics_min_gt_max() {
        DiscreteSpace::new(10, 1);
    }

    #[test]
    fn test_single_value() {
        let space = DiscreteSpace::new(5, 5);
        assert_eq!(space.cardinality(), 1);
        assert!(space.contains(&5));
        assert!(!space.contains(&4));
        assert!(!space.contains(&6));
        assert_eq!(space.clamp(&100), 5);
        assert_eq!(space.clamp(&-100), 5);

        let unit = space.to_unit_cube(&5);
        assert_eq!(unit.len(), 1);
        let restored = space.from_unit_cube(&unit).unwrap();
        assert_eq!(restored, 5);
    }

    #[test]
    fn test_unit_cube_roundtrip_all_values() {
        let space = DiscreteSpace::new(0, 9);
        for i in 0..=9 {
            let unit = space.to_unit_cube(&i);
            assert!(
                unit[0] >= 0.0 && unit[0] <= 1.0,
                "unit value out of range for {i}"
            );
            let restored = space.from_unit_cube(&unit).unwrap();
            assert_eq!(restored, i, "roundtrip failed for {i}");
        }
    }

    #[test]
    fn test_from_unit_cube_wrong_dims() {
        let space = DiscreteSpace::new(0, 9);
        assert!(space.from_unit_cube(&[]).is_none());
        assert!(space.from_unit_cube(&[0.5, 0.5]).is_none());
    }

    #[test]
    fn test_negative_range() {
        let space = DiscreteSpace::new(-10, -1);
        assert_eq!(space.cardinality(), 10);
        assert!(space.contains(&-5));
        assert!(!space.contains(&0));
        assert_eq!(space.clamp(&0), -1);
        assert_eq!(space.clamp(&-20), -10);

        for i in -10..=-1 {
            let unit = space.to_unit_cube(&i);
            let restored = space.from_unit_cube(&unit).unwrap();
            assert_eq!(restored, i, "roundtrip failed for {i}");
        }
    }

    #[test]
    fn test_unit_cube_boundary_values() {
        let space = DiscreteSpace::new(0, 2);
        assert_eq!(space.from_unit_cube(&[0.0]).unwrap(), 0);
        assert_eq!(space.from_unit_cube(&[1.0]).unwrap(), 2);
    }
}
