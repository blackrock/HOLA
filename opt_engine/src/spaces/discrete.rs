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
use serde::{Deserialize, Deserializer, Serialize, de};

/// Largest inclusive cardinality supported by the bucket mapping.
///
/// Bucket centers contain a half-integer. Binary64 represents every such
/// center exactly while the zero-based index is below `2^52`; wider spaces
/// cannot promise an integer -> unit cube -> integer round trip.
const MAX_EXACT_DISCRETE_CARDINALITY: u128 = 1u128 << 52;

/// Discrete integer space over [min, max] inclusive.
///
/// Each integer in the range gets an equally-sized bucket in [0, 1] for standardization.
/// For example, with min=0, max=2 (3 integers):
/// - Integer 0 ↔ bucket [0, 1/3), center at 1/6
/// - Integer 1 ↔ bucket [1/3, 2/3), center at 1/2
/// - Integer 2 ↔ bucket [2/3, 1], center at 5/6
#[derive(Clone, Debug, Serialize)]
pub struct DiscreteSpace {
    min: i64,
    max: i64,
}

impl DiscreteSpace {
    /// Maximum cardinality that preserves exact integer round trips through
    /// the binary64 unit-cube representation.
    pub const MAX_EXACT_CARDINALITY: u128 = MAX_EXACT_DISCRETE_CARDINALITY;

    /// Inclusive lower bound.
    pub fn min(&self) -> i64 {
        self.min
    }

    /// Inclusive upper bound.
    pub fn max(&self) -> i64 {
        self.max
    }

    /// Construct a discrete space whose unit-cube mapping is exact.
    pub fn try_new(min: i64, max: i64) -> Result<Self, String> {
        if min > max {
            return Err(format!(
                "min must be less than or equal to max, got min={min}, max={max}"
            ));
        }

        let width = ((max as i128) - (min as i128) + 1) as u128;
        if width > MAX_EXACT_DISCRETE_CARDINALITY {
            return Err(format!(
                "cardinality {width} exceeds the exact unit-cube mapping limit of {MAX_EXACT_DISCRETE_CARDINALITY}"
            ));
        }
        if width > usize::MAX as u128 {
            return Err(format!(
                "cardinality {width} exceeds usize::MAX on this platform"
            ));
        }

        Ok(Self { min, max })
    }

    /// Construct a discrete space, panicking immediately for unsupported
    /// bounds. Prefer [`Self::try_new`] for user-provided configuration.
    pub fn new(min: i64, max: i64) -> Self {
        Self::try_new(min, max).unwrap_or_else(|error| panic!("DiscreteSpace: {error}"))
    }

    pub fn cardinality(&self) -> usize {
        // Widen to i128 before subtracting to avoid i64 overflow for ranges
        // wider than i64::MAX. new() guarantees this fits in a usize.
        ((self.max as i128) - (self.min as i128) + 1) as usize
    }
}

impl<'de> Deserialize<'de> for DiscreteSpace {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DiscreteSpaceSerde {
            min: i64,
            max: i64,
        }

        let raw = DiscreteSpaceSerde::deserialize(deserializer)?;
        Self::try_new(raw.min, raw.max).map_err(de::Error::custom)
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
        // Widen to i128 so the offset does not overflow for ranges wider than i64::MAX.
        let bucket_center = ((*point as i128) - (self.min as i128)) as f64 + 0.5;
        vec![bucket_center / n]
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<i64> {
        if vec.len() != 1 {
            return None;
        }
        let val = vec[0].clamp(0.0, 1.0);
        let n = self.cardinality() as f64;
        // Map from [0,1] to bucket index, then to actual integer
        // val * n gives us a value in [0, n], floor gives bucket index.
        // Compute in i128 so adding the index to min cannot overflow for ranges
        // wider than i64::MAX.
        let index = (val * n).floor() as i128;
        // Clamp to valid range (handles edge case where val = 1.0 exactly)
        let max_index = (self.max as i128) - (self.min as i128);
        let index = index.min(max_index);
        Some(((self.min as i128) + index) as i64)
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
    #[should_panic(expected = "min must be less than or equal to max")]
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

    #[test]
    #[should_panic(expected = "exceeds the exact unit-cube mapping limit")]
    fn test_full_i64_range_panics_clean() {
        // [i64::MIN, i64::MAX] has width 2^64 which exceeds usize::MAX; new()
        // must panic with a clear message rather than overflow silently.
        DiscreteSpace::new(i64::MIN, i64::MAX);
    }

    #[test]
    fn test_maximum_supported_cardinality_roundtrips() {
        let min = -(1i64 << 51);
        let max = (1i64 << 51) - 1;
        let space = DiscreteSpace::try_new(min, max).unwrap();
        assert_eq!(space.cardinality() as u128, MAX_EXACT_DISCRETE_CARDINALITY);

        for &pt in &[min, min + 1, -1, 0, 1, max - 1, max] {
            let unit = space.to_unit_cube(&pt);
            let restored = space.from_unit_cube(&unit).unwrap();
            assert_eq!(restored, pt, "roundtrip failed for {pt}");
        }
    }

    #[test]
    fn test_try_new_rejects_inexact_cardinality() {
        let max = 1i64 << 52;
        let error = DiscreteSpace::try_new(0, max).unwrap_err();
        assert!(error.contains("exact unit-cube mapping limit"));
    }

    #[test]
    fn test_deserialization_validates_bounds_and_cardinality() {
        assert!(serde_json::from_str::<DiscreteSpace>(r#"{"min":10,"max":1}"#).is_err());
        let too_wide = format!(r#"{{"min":0,"max":{}}}"#, 1i64 << 52);
        assert!(serde_json::from_str::<DiscreteSpace>(&too_wide).is_err());
    }

    proptest::proptest! {
        #[test]
        fn prop_maximum_supported_range_roundtrips(offset in 0u64..(1u64 << 52)) {
            let min = -(1i64 << 51);
            let max = (1i64 << 51) - 1;
            let point = min + offset as i64;
            let space = DiscreteSpace::try_new(min, max).unwrap();
            let unit = space.to_unit_cube(&point);
            proptest::prop_assert_eq!(space.from_unit_cube(&unit), Some(point));
        }
    }
}
