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

//! Format-agnostic objective scalarization math.
//!
//! These are the pure mathematical operations that map raw metric values to
//! optimization scores. They have no knowledge of JSON, wire formats, or
//! field names — that responsibility belongs to [`Transformer`](crate::traits::Transformer)
//! implementations.

/// Flip sign for maximization.
///
/// Convention: the optimizer always minimizes. To maximize a metric,
/// negate it before feeding it to the strategy.
pub fn directed_value(value: f64, negate: bool) -> f64 {
    if negate { -value } else { value }
}

/// Target-Limit-Priority (TLP) scalarization for a single value.
///
/// Direction is inferred from the relationship between `target` and `limit`:
/// - `target < limit` → **minimize** (lower values are better)
/// - `target > limit` → **maximize** (higher values are better)
/// - `target == limit` → values at or beyond target score 0, otherwise +∞
///
/// Returns:
/// - `0.0` if the value meets or exceeds the target
/// - `f64::INFINITY` if the value exceeds the limit (infeasible)
/// - Linear interpolation in `[0, 1]` between target and limit
///
/// Note: this returns a **normalized** score (0 to 1). Multiply by a
/// priority weight after calling if needed.
pub fn tlp_score(value: f64, target: f64, limit: f64) -> f64 {
    if target < limit {
        // Minimize: lower is better
        if value <= target {
            0.0
        } else if value > limit {
            f64::INFINITY
        } else {
            (value - target) / (limit - target)
        }
    } else if target > limit {
        // Maximize: higher is better
        if value >= target {
            0.0
        } else if value < limit {
            f64::INFINITY
        } else {
            (target - value) / (target - limit)
        }
    } else {
        // Degenerate: target == limit
        if value >= target { 0.0 } else { f64::INFINITY }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directed_value_minimize() {
        assert_eq!(directed_value(0.5, false), 0.5);
        assert_eq!(directed_value(-1.0, false), -1.0);
    }

    #[test]
    fn test_directed_value_maximize() {
        assert_eq!(directed_value(0.5, true), -0.5);
        assert_eq!(directed_value(-1.0, true), 1.0);
    }

    #[test]
    fn test_tlp_minimize_at_target() {
        assert_eq!(tlp_score(0.1, 0.1, 1.0), 0.0);
    }

    #[test]
    fn test_tlp_minimize_below_target() {
        assert_eq!(tlp_score(0.0, 0.1, 1.0), 0.0);
    }

    #[test]
    fn test_tlp_minimize_at_limit() {
        assert!((tlp_score(1.0, 0.1, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tlp_minimize_beyond_limit() {
        assert!(tlp_score(1.5, 0.1, 1.0).is_infinite());
    }

    #[test]
    fn test_tlp_minimize_midpoint() {
        // target=0, limit=1, value=0.5 → (0.5 - 0) / (1 - 0) = 0.5
        assert!((tlp_score(0.5, 0.0, 1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tlp_maximize_at_target() {
        assert_eq!(tlp_score(0.9, 0.9, 0.5), 0.0);
    }

    #[test]
    fn test_tlp_maximize_above_target() {
        assert_eq!(tlp_score(1.0, 0.9, 0.5), 0.0);
    }

    #[test]
    fn test_tlp_maximize_at_limit() {
        assert!((tlp_score(0.5, 0.9, 0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tlp_maximize_below_limit() {
        assert!(tlp_score(0.3, 0.9, 0.5).is_infinite());
    }

    #[test]
    fn test_tlp_maximize_midpoint() {
        // target=1.0, limit=0.0, value=0.5 → (1.0 - 0.5) / (1.0 - 0.0) = 0.5
        assert!((tlp_score(0.5, 1.0, 0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tlp_degenerate_equal() {
        assert_eq!(tlp_score(1.0, 1.0, 1.0), 0.0);
        assert_eq!(tlp_score(2.0, 1.0, 1.0), 0.0);
        assert!(tlp_score(0.5, 1.0, 1.0).is_infinite());
    }
}
