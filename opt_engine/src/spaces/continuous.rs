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
use serde::{Deserialize, Deserializer, Serialize, de};

/// A continuous range `[min, max]` with an optional scale transformation.
///
/// Bounds are specified in the *actual* (user-facing) space — the values you
/// care about. For example, to search learning rates between `1e-4` and `0.1`
/// on a log10 scale:
///
/// ```
/// use opt_engine::{ContinuousSpace, Log10Scale};
///
/// let space = ContinuousSpace::with_scale(1e-4, 0.1, Log10Scale);
/// assert_eq!(space.min(), 1e-4);
/// ```
///
/// Internally, the scale's `inverse` maps actual values to a linear internal
/// space where unit-cube normalization is performed.
#[derive(Clone, Debug, Serialize)]
pub struct ContinuousSpace<S: Scale = LinearScale> {
    min: f64,
    max: f64,
    scale: S,
}

impl ContinuousSpace<LinearScale> {
    /// Construct a validated linear space.
    ///
    /// Fixed ranges (`min == max`) are supported. Reversed, non-finite, and
    /// ranges whose internal span overflows are rejected.
    pub fn try_new(min: f64, max: f64) -> Result<Self, String> {
        Self::try_with_scale(min, max, LinearScale)
    }

    /// Construct a linear space, panicking immediately if its bounds are
    /// invalid. Prefer [`Self::try_new`] for user-provided configuration.
    pub fn new(min: f64, max: f64) -> Self {
        Self::try_new(min, max).unwrap_or_else(|error| panic!("ContinuousSpace: {error}"))
    }
}

impl<S: Scale> ContinuousSpace<S> {
    /// Lower bound in user-facing space.
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Upper bound in user-facing space.
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Scale transformation used by this space.
    pub fn scale(&self) -> &S {
        &self.scale
    }

    /// Create a validated continuous space with a custom scale.
    ///
    /// Bounds live in user-facing space. Both bounds and their transformed
    /// values must be finite. Fixed ranges are represented by the unit-cube
    /// midpoint; non-fixed ranges must have a finite, non-zero internal span.
    pub fn try_with_scale(min: f64, max: f64, scale: S) -> Result<Self, String> {
        if !min.is_finite() || !max.is_finite() {
            return Err(format!("bounds must be finite, got min={min}, max={max}"));
        }
        if min > max {
            return Err(format!(
                "min must be less than or equal to max, got min={min}, max={max}"
            ));
        }

        let internal_min = scale.inverse(min);
        let internal_max = scale.inverse(max);
        if !internal_min.is_finite() || !internal_max.is_finite() {
            return Err(format!(
                "{} scale requires bounds in its finite domain, got min={min}, max={max}",
                S::name()
            ));
        }

        if min < max {
            let span = internal_max - internal_min;
            if !span.is_finite() || span == 0.0 {
                return Err(format!(
                    "{} scale produces an invalid internal span for min={min}, max={max}",
                    S::name()
                ));
            }
        }

        Ok(Self { min, max, scale })
    }

    /// Create a continuous space with a custom scale.
    ///
    /// `min` and `max` are specified in actual (user-facing) space.
    pub fn with_scale(min: f64, max: f64, scale: S) -> Self {
        Self::try_with_scale(min, max, scale)
            .unwrap_or_else(|error| panic!("ContinuousSpace: {error}"))
    }
}

impl<'de, S> Deserialize<'de> for ContinuousSpace<S>
where
    S: Scale + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ContinuousSpaceSerde<S> {
            min: f64,
            max: f64,
            scale: S,
        }

        let raw = ContinuousSpaceSerde::<S>::deserialize(deserializer)?;
        Self::try_with_scale(raw.min, raw.max, raw.scale).map_err(de::Error::custom)
    }
}

impl<S: Scale> SampleSpace for ContinuousSpace<S> {
    /// The domain is the actual (user-facing) value.
    type Domain = f64;

    fn contains(&self, point: &f64) -> bool {
        if !point.is_finite() {
            return false;
        }
        // A handful of ULPs tolerates normal floating-point boundary noise
        // without introducing an absolute floor that can exceed a tiny range.
        let magnitude = self.min.abs().max(self.max.abs());
        let eps = 16.0 * f64::EPSILON * magnitude;
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
        if self.min == self.max {
            return vec![0.5];
        }
        // Map actual value to internal space, then normalize to [0, 1]
        let internal = self.scale.inverse(self.clamp(point));
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
        if !vec[0].is_finite() {
            return None;
        }
        if self.min == self.max {
            return Some(self.min);
        }
        // Clamp for numerical safety
        let val = vec[0].clamp(0.0, 1.0);
        // Transcendental inverse/forward pairs can overflow or move one ULP at
        // extreme finite bounds. Preserve the declared endpoints exactly.
        if val == 0.0 {
            return Some(self.min);
        }
        if val == 1.0 {
            return Some(self.max);
        }
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
        let actual = self.scale.forward(internal);
        actual.is_finite().then(|| actual.clamp(self.min, self.max))
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
        assert_eq!(unit, vec![0.5]);
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
        let space = ContinuousSpace::new(1e-12, 2e-12);
        // This is only one range-width beyond max. The old absolute 1e-9
        // tolerance accepted it even though it is materially out of range.
        assert!(!space.contains(&3e-12));
        // The exact boundaries are still contained.
        assert!(space.contains(&1e-12));
        assert!(space.contains(&2e-12));
    }

    #[test]
    fn test_contains_relative_eps_large_range() {
        // For a large range the eps scales up with magnitude, so boundary
        // values (and their floating-point neighbors) still register as
        // contained instead of being rejected by an over-strict absolute eps.
        let space = ContinuousSpace::new(1e9, 2e9);
        assert!(space.contains(&1e9));
        assert!(space.contains(&2e9));
        // One adjacent representable value is boundary noise and is accepted.
        let above = f64::from_bits(2e9f64.to_bits() + 1);
        let below = f64::from_bits(1e9f64.to_bits() - 1);
        assert!(space.contains(&above));
        assert!(space.contains(&below));
    }

    #[test]
    fn test_contains_rejects_non_finite_points_at_extreme_fixed_bounds() {
        let positive = ContinuousSpace::new(f64::MAX, f64::MAX);
        assert!(positive.contains(&f64::MAX));
        assert!(!positive.contains(&f64::INFINITY));
        assert!(!positive.contains(&f64::NAN));

        let negative = ContinuousSpace::new(-f64::MAX, -f64::MAX);
        assert!(negative.contains(&-f64::MAX));
        assert!(!negative.contains(&f64::NEG_INFINITY));
    }

    #[test]
    fn test_extreme_log10_cube_endpoint_stays_finite_and_exact() {
        let space = ContinuousSpace::with_scale(1.0, f64::MAX, Log10Scale);
        assert_eq!(space.from_unit_cube(&[0.0]), Some(1.0));
        assert_eq!(space.from_unit_cube(&[1.0]), Some(f64::MAX));
        let midpoint = space.from_unit_cube(&[0.5]).unwrap();
        assert!(midpoint.is_finite());
        assert!(space.contains(&midpoint));
        assert!(space.from_unit_cube(&[f64::NAN]).is_none());
        assert!(space.from_unit_cube(&[f64::INFINITY]).is_none());
    }

    #[test]
    fn test_try_constructors_reject_invalid_bounds() {
        assert!(ContinuousSpace::try_new(2.0, 1.0).is_err());
        assert!(ContinuousSpace::try_new(f64::NAN, 1.0).is_err());
        assert!(ContinuousSpace::try_new(0.0, f64::INFINITY).is_err());
        assert!(ContinuousSpace::try_new(-f64::MAX, f64::MAX).is_err());
        assert!(ContinuousSpace::try_new(4.0, 4.0).is_ok());

        assert!(ContinuousSpace::try_with_scale(0.0, 1.0, LogScale).is_err());
        assert!(ContinuousSpace::try_with_scale(-1.0, 1.0, Log10Scale).is_err());
        assert!(ContinuousSpace::try_with_scale(0.5, 0.5, LogScale).is_ok());
    }

    #[test]
    fn test_deserialization_validates_bounds_and_scale_domain() {
        let reversed = r#"{"min":2.0,"max":1.0,"scale":null}"#;
        assert!(serde_json::from_str::<ContinuousSpace>(reversed).is_err());

        let invalid_log = r#"{"min":0.0,"max":1.0,"scale":null}"#;
        assert!(serde_json::from_str::<ContinuousSpace<LogScale>>(invalid_log).is_err());

        let fixed = r#"{"min":5.0,"max":5.0,"scale":null}"#;
        let restored: ContinuousSpace = serde_json::from_str(fixed).unwrap();
        assert_eq!(restored.to_unit_cube(&5.0), vec![0.5]);
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
