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
        // Scale the tolerance relative to the magnitude of the bounds so that
        // a fixed absolute eps is neither over-tolerant for tiny ranges nor
        // too strict for very large ones.
        let eps = 1e-9 * (1.0 + self.min.abs().max(self.max.abs()));
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
        let span = internal_max - internal_min;
        if span == 0.0 || !span.is_finite() {
            // Degenerate fixed parameter (min == max), or a span that is not
            // finite because the scale maps the bounds to non-finite internal
            // values (e.g. a log scale with non-positive bounds). Normalization
            // would yield NaN/inf, so map to the cube midpoint instead.
            // from_unit_cube applies the same guard and returns the fixed actual
            // value, so the degenerate case round-trips to that fixed value.
            return vec![0.5];
        }
        let normalized = (internal - internal_min) / span;
        if !normalized.is_finite() {
            // Guard against a non-finite quotient (e.g. internal being
            // non-finite) escaping into the unit cube.
            return vec![0.5];
        }
        vec![normalized]
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
        let span = internal_max - internal_min;
        if span == 0.0 || !span.is_finite() {
            // Degenerate fixed parameter (min == max), or a span that is not
            // finite because the scale maps the bounds to non-finite internal
            // values (e.g. a log scale with non-positive bounds). Computing with
            // such a span would yield a degenerate/non-finite result, so return
            // the well-defined fixed actual value at internal_min instead. This
            // mirrors to_unit_cube collapsing these cases to the cube midpoint.
            return Some(self.scale.forward(internal_min));
        }
        let internal = internal_min + val * span;
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
        // to_unit_cube must not divide by a zero span (which would yield NaN).
        let unit = space.to_unit_cube(&5.0);
        assert_eq!(unit.len(), 1);
        assert!(
            unit[0].is_finite(),
            "degenerate min==max must not produce NaN"
        );

        // A log-scale space with non-positive bounds maps to non-finite
        // internal values, making the span NaN. to_unit_cube must still return
        // a finite midpoint rather than letting NaN escape.
        let log_degenerate = ContinuousSpace::with_scale(0.0, 0.0, LogScale);
        let unit = log_degenerate.to_unit_cube(&0.0);
        assert_eq!(unit.len(), 1);
        assert_eq!(
            unit[0], 0.5,
            "log-scale degenerate space must return finite 0.5, not NaN"
        );
        // from_unit_cube must symmetrically guard the non-finite span: without
        // the guard the NaN span would make the reconstructed value non-finite.
        let restored = log_degenerate.from_unit_cube(&[0.5]).unwrap();
        assert!(
            restored.is_finite(),
            "log-scale degenerate from_unit_cube must return finite value, got {restored}"
        );
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

    #[test]
    fn test_contains_relative_eps_tiny_range() {
        // For a tiny range the scale-relative eps is essentially the fixed
        // 1e-9 floor, so a value beyond max by more than that must be rejected.
        let space = ContinuousSpace::new(1e-12, 1e-9);
        // eps = 1e-9 * (1 + 1e-9) ~= 1e-9. A point above max by ~1e-6 (>> eps)
        // must not be considered contained.
        assert!(!space.contains(&(1e-9 + 1e-6)));
        // The exact boundaries are still contained.
        assert!(space.contains(&1e-12));
        assert!(space.contains(&1e-9));
    }

    #[test]
    fn test_contains_relative_eps_large_range() {
        // For a large range the eps scales up with magnitude, so boundary
        // values (and their floating-point neighbors) still register as
        // contained instead of being rejected by an over-strict absolute eps.
        let space = ContinuousSpace::new(1e9, 2e9);
        assert!(space.contains(&1e9));
        assert!(space.contains(&2e9));
        // eps ~= 1e-9 * (1 + 2e9) = ~2.0, so values a hair past the bounds are
        // still tolerated at this scale.
        assert!(space.contains(&(2e9 + 1.0)));
        assert!(space.contains(&(1e9 - 1.0)));
    }

    proptest::proptest! {
        /// from_unit_cube(to_unit_cube(x)) recovers x within tolerance
        /// across Linear / Log / Log10 scales over swept continuous values.
        #[test]
        fn prop_continuous_roundtrip_linear(x in -1e6f64..1e6f64) {
            let space = ContinuousSpace::new(-1e6, 1e6);
            let unit = space.to_unit_cube(&x);
            proptest::prop_assert_eq!(unit.len(), 1);
            proptest::prop_assert!(unit[0] >= -1e-9 && unit[0] <= 1.0 + 1e-9);
            let recon = space.from_unit_cube(&unit).unwrap();
            proptest::prop_assert!((recon - x).abs() < 1e-3, "linear roundtrip failed: {} -> {}", x, recon);
        }

        #[test]
        fn prop_continuous_roundtrip_log(x in 1e-6f64..1e3f64) {
            let space = ContinuousSpace::with_scale(1e-6, 1e3, LogScale);
            let unit = space.to_unit_cube(&x);
            proptest::prop_assert_eq!(unit.len(), 1);
            let recon = space.from_unit_cube(&unit).unwrap();
            proptest::prop_assert!((recon - x).abs() / x < 1e-6, "log roundtrip failed: {} -> {}", x, recon);
        }

        #[test]
        fn prop_continuous_roundtrip_log10(x in 1e-6f64..1e3f64) {
            let space = ContinuousSpace::with_scale(1e-6, 1e3, Log10Scale);
            let unit = space.to_unit_cube(&x);
            proptest::prop_assert_eq!(unit.len(), 1);
            let recon = space.from_unit_cube(&unit).unwrap();
            proptest::prop_assert!((recon - x).abs() / x < 1e-6, "log10 roundtrip failed: {} -> {}", x, recon);
        }
    }
}
