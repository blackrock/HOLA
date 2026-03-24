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

//! Scale transformations for continuous spaces.
//!
//! A scale defines a bijection between a linear *internal* space (where
//! unit-cube normalization is performed) and the *actual* space of user-facing
//! values. Users specify bounds in actual space; the scale handles the rest.

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A bijective transformation between a linear internal space and the
/// user-facing actual space.
///
/// `forward` maps internal → actual; `inverse` maps actual → internal.
pub trait Scale: Send + Sync + Clone + Debug + 'static {
    /// Short identifier used for dashboard axis labels and config serialization.
    fn name() -> &'static str;

    /// Transform from internal space to actual space.
    /// e.g., for LogScale: forward(x) = exp(x)
    fn forward(&self, x: f64) -> f64;

    /// Transform from actual space to internal space.
    /// e.g., for LogScale: inverse(x) = ln(x)
    fn inverse(&self, x: f64) -> f64;
}

/// Identity scale — the internal and actual spaces are the same.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LinearScale;

impl Scale for LinearScale {
    fn name() -> &'static str {
        "linear"
    }
    #[inline]
    fn forward(&self, x: f64) -> f64 {
        x
    }
    #[inline]
    fn inverse(&self, x: f64) -> f64 {
        x
    }
}

/// Natural logarithmic scale: actual = exp(internal).
///
/// Useful for parameters that vary over orders of magnitude.
///
/// # Example
/// ```ignore
/// // Learning rate between 1e-4 and 0.1, sampled on a natural log scale
/// let space = ContinuousSpace::with_scale(1e-4, 0.1, LogScale);
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LogScale;

impl Scale for LogScale {
    fn name() -> &'static str {
        "log"
    }
    #[inline]
    fn forward(&self, x: f64) -> f64 {
        x.exp()
    }
    #[inline]
    fn inverse(&self, x: f64) -> f64 {
        x.ln()
    }
}

/// Base-10 logarithmic scale: actual = 10^internal.
///
/// Convenient when thinking in orders of magnitude.
///
/// # Example
/// ```ignore
/// // Learning rate between 1e-4 and 0.1, sampled on a log10 scale
/// let space = ContinuousSpace::with_scale(1e-4, 0.1, Log10Scale);
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Log10Scale;

impl Scale for Log10Scale {
    fn name() -> &'static str {
        "log10"
    }
    #[inline]
    fn forward(&self, x: f64) -> f64 {
        (10.0_f64).powf(x)
    }
    #[inline]
    fn inverse(&self, x: f64) -> f64 {
        x.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_scale_identity() {
        assert_eq!(LinearScale.forward(5.0), 5.0);
        assert_eq!(LinearScale.inverse(5.0), 5.0);
    }

    #[test]
    fn test_log_scale() {
        assert!((LogScale.forward(0.0) - 1.0).abs() < 1e-12); // e^0 = 1
        assert!((LogScale.inverse(1.0) - 0.0).abs() < 1e-12); // ln(1) = 0
    }

    #[test]
    fn test_log10_scale() {
        assert!((Log10Scale.forward(2.0) - 100.0).abs() < 1e-9); // 10^2 = 100
        assert!((Log10Scale.inverse(100.0) - 2.0).abs() < 1e-9); // log10(100) = 2
    }
}
