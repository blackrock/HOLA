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

//! Continuous parameter space with optional scale transformation.

use crate::scales::{LinearScale, Scale};
use crate::traits::{SampleSpace, StandardizedSpace};
use serde::{Deserialize, Serialize};

/// A continuous range `[min, max]` with an optional scale transformation.
///
/// Bounds are specified in the *actual* (user-facing) space — the values you
/// care about. For example, to search learning rates between `1e-4` and `0.1`
/// on a log10 scale:
///
/// ```ignore
/// ContinuousSpace::with_scale(1e-4, 0.1, Log10Scale)
/// ```
///
/// Internally, the scale's `inverse` maps actual values to a linear internal
/// space where unit-cube normalization is performed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContinuousSpace<S: Scale = LinearScale> {
    pub min: f64,
    pub max: f64,
    pub scale: S,
}

impl ContinuousSpace<LinearScale> {
    pub fn new(min: f64, max: f64) -> Self {
        Self {
            min,
            max,
            scale: LinearScale,
        }
    }
}

impl<S: Scale> ContinuousSpace<S> {
    /// Create a continuous space with a custom scale.
    ///
    /// `min` and `max` are specified in actual (user-facing) space.
    pub fn with_scale(min: f64, max: f64, scale: S) -> Self {
        Self { min, max, scale }
    }
}

impl<S: Scale> SampleSpace for ContinuousSpace<S> {
    /// The domain is the actual (user-facing) value.
    type Domain = f64;

    fn contains(&self, point: &f64) -> bool {
        let eps = 1e-9;
        *point >= self.min - eps && *point <= self.max + eps
    }

    fn clamp(&self, point: &f64) -> f64 {
        (*point).clamp(self.min, self.max)
    }
}

impl<S: Scale> StandardizedSpace for ContinuousSpace<S> {
    fn dimensionality(&self) -> usize {
        1
    }

    fn to_unit_cube(&self, point: &f64) -> Vec<f64> {
        // Map actual value to internal space, then normalize to [0, 1]
        let internal = self.scale.inverse(*point);
        let internal_min = self.scale.inverse(self.min);
        let internal_max = self.scale.inverse(self.max);
        vec![(internal - internal_min) / (internal_max - internal_min)]
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<f64> {
        if vec.len() != 1 {
            return None;
        }
        // Clamp for numerical safety
        let val = vec[0].clamp(0.0, 1.0);
        // Map from unit cube to internal space, then apply forward transformation
        let internal_min = self.scale.inverse(self.min);
        let internal_max = self.scale.inverse(self.max);
        let internal = internal_min + val * (internal_max - internal_min);
        Some(self.scale.forward(internal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scales::{Log10Scale, LogScale};

    #[test]
    fn test_invariants() {
        let space = ContinuousSpace::new(-1.0, 1.0);
        assert_eq!(space.min, -1.0);
        assert_eq!(space.max, 1.0);
        assert!(space.contains(&0.0));
        assert!(!space.contains(&2.0));
        assert_eq!(space.clamp(&2.0), 1.0);
        assert_eq!(space.clamp(&-5.0), -1.0);
        assert_eq!(space.dimensionality(), 1);

        let unit = space.to_unit_cube(&0.5);
        assert!(unit[0] >= 0.0 && unit[0] <= 1.0);
        let recon = space.from_unit_cube(&unit).unwrap();
        assert!((recon - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_log10_roundtrip() {
        let space = ContinuousSpace::with_scale(1e-4, 0.1, Log10Scale);
        assert!((space.min - 1e-4).abs() < 1e-12);
        assert!((space.max - 0.1).abs() < 1e-12);

        let val = 0.01;
        let unit = space.to_unit_cube(&val);
        assert!(unit[0] >= 0.0 && unit[0] <= 1.0);
        let recon = space.from_unit_cube(&unit).unwrap();
        assert!((recon - val).abs() / val < 1e-6);
    }

    #[test]
    fn test_degenerate_min_eq_max() {
        let space = ContinuousSpace::new(5.0, 5.0);
        assert!(space.contains(&5.0));
        assert_eq!(space.clamp(&100.0), 5.0);
        assert_eq!(space.clamp(&-100.0), 5.0);
        assert_eq!(space.dimensionality(), 1);
        let restored = space.from_unit_cube(&[0.5]).unwrap();
        assert_eq!(restored, 5.0);
    }

    #[test]
    fn test_boundary_precision() {
        let space = ContinuousSpace::new(-1.0, 1.0);
        assert!(space.contains(&-1.0));
        assert!(space.contains(&1.0));
        assert!(!space.contains(&(1.0 + 1e-6)));
        assert!(!space.contains(&(-1.0 - 1e-6)));
    }

    #[test]
    fn test_clamp_preserves_valid() {
        let space = ContinuousSpace::new(0.0, 10.0);
        assert_eq!(space.clamp(&5.0), 5.0);
        assert_eq!(space.clamp(&0.0), 0.0);
        assert_eq!(space.clamp(&10.0), 10.0);
    }

    #[test]
    fn test_log_scale_roundtrip() {
        let space = ContinuousSpace::with_scale(0.001, 1.0, LogScale);
        let test_values = [0.001, 0.01, 0.1, 0.5, 1.0];
        for &val in &test_values {
            let unit = space.to_unit_cube(&val);
            assert!(
                unit[0] >= -1e-9 && unit[0] <= 1.0 + 1e-9,
                "unit value out of range for {val}"
            );
            let restored = space.from_unit_cube(&unit).unwrap();
            assert!(
                (restored - val).abs() / val < 1e-9,
                "roundtrip failed for {val}: got {restored}"
            );
        }
    }

    #[test]
    fn test_log10_boundary_values() {
        let space = ContinuousSpace::with_scale(1e-4, 0.1, Log10Scale);
        let unit_min = space.to_unit_cube(&1e-4);
        let unit_max = space.to_unit_cube(&0.1);
        assert!((unit_min[0] - 0.0).abs() < 1e-9, "min should map to ~0.0");
        assert!((unit_max[0] - 1.0).abs() < 1e-9, "max should map to ~1.0");
    }

    #[test]
    fn test_from_unit_cube_wrong_dims() {
        let space = ContinuousSpace::new(0.0, 1.0);
        assert!(space.from_unit_cube(&[]).is_none());
        assert!(space.from_unit_cube(&[0.5, 0.5]).is_none());
    }
}
