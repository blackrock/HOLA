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

//! JSON TLP (Target-Limit-Priority) scalarizer transformer.

use crate::objectives::tlp_score;
use crate::traits::Transformer;

/// Specification for a single field in the TLP scalarizer.
#[derive(Debug, Clone)]
pub struct TlpField {
    /// The field name to extract from JSON.
    pub field: String,
    /// The target value at/beyond which the user is satisfied (output = 0).
    pub target: f64,
    /// The limit value beyond which the user is infinitely unsatisfied (output = +inf).
    pub limit: f64,
    /// The priority weight at the limit boundary.
    pub priority: f64,
}

impl TlpField {
    /// Create a new TLP field specification.
    ///
    /// - If `target < limit`: minimization (lower values are better)
    /// - If `target > limit`: maximization (higher values are better)
    ///
    /// # Panics
    /// Panics if `priority` is negative.
    pub fn new(field: impl Into<String>, target: f64, limit: f64, priority: f64) -> Self {
        assert!(
            priority >= 0.0,
            "Priority must be non-negative (got {priority})"
        );
        Self {
            field: field.into(),
            target,
            limit,
            priority,
        }
    }

    /// Create a minimization field (target < limit).
    ///
    /// Values at or below `target` score 0; values at or above `limit` score infinity.
    pub fn minimize(field: impl Into<String>, target: f64, limit: f64, priority: f64) -> Self {
        debug_assert!(target < limit, "For minimize, target should be < limit");
        Self::new(field, target, limit, priority)
    }

    /// Create a maximization field (target > limit).
    ///
    /// Values at or above `target` score 0; values at or below `limit` score infinity.
    pub fn maximize(field: impl Into<String>, target: f64, limit: f64, priority: f64) -> Self {
        debug_assert!(target > limit, "For maximize, target should be > limit");
        Self::new(field, target, limit, priority)
    }

    /// Compute the scalarized value for this field.
    fn scalarize(&self, value: f64) -> f64 {
        self.priority * tlp_score(value, self.target, self.limit)
    }
}

/// A TLP (Target-Limit-Priority) scalarizer transformer.
///
/// For each field, the user specifies:
/// - **target**: the value at/beyond which they are satisfied (contributes 0)
/// - **limit**: the value beyond which they are infinitely unsatisfied (contributes +∞)
/// - **priority**: the weight at the limit boundary
///
/// The direction (minimize vs maximize) is inferred from target vs limit:
/// - `target < limit`: minimize (want lower values)
/// - `target > limit`: maximize (want higher values)
///
/// Between target and limit, the contribution is linearly interpolated from 0 to priority.
/// The final output is the sum of all field contributions.
///
/// # Example
///
/// ```
/// use opt_engine::transformers::{JsonTlpTransformer, TlpField};
///
/// let transformer = JsonTlpTransformer::new([
///     TlpField::minimize("loss", 0.01, 1.0, 1.0),      // Want loss ≤ 0.01, unacceptable ≥ 1.0
///     TlpField::minimize("latency_ms", 100.0, 500.0, 0.5), // Want latency ≤ 100ms
///     TlpField::maximize("accuracy", 0.95, 0.5, 2.0), // Want accuracy ≥ 95%, unacceptable ≤ 50%
/// ]);
/// ```
pub struct JsonTlpTransformer {
    fields: Vec<TlpField>,
}

impl JsonTlpTransformer {
    /// Create a new TLP transformer with the specified field specifications.
    pub fn new<I>(fields: I) -> Self
    where
        I: IntoIterator<Item = TlpField>,
    {
        Self {
            fields: fields.into_iter().collect(),
        }
    }

    /// Builder method to add a field specification.
    pub fn with_field(mut self, field: TlpField) -> Self {
        self.fields.push(field);
        self
    }
}

impl Transformer for JsonTlpTransformer {
    type ForeignInput = serde_json::Value;
    type Output = f64;

    fn transform(&self, input: serde_json::Value) -> Result<f64, String> {
        let mut sum = 0.0;

        for spec in &self.fields {
            let value = input
                .get(&spec.field)
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    format!(
                        "Invalid Schema: Missing or non-numeric '{}' field",
                        spec.field
                    )
                })?;

            sum += spec.scalarize(value);
        }

        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_minimize_at_target() {
        let transformer = JsonTlpTransformer::new([TlpField::minimize("loss", 0.1, 1.0, 1.0)]);
        let input = json!({"loss": 0.1});
        assert_eq!(transformer.transform(input).unwrap(), 0.0);
    }

    #[test]
    fn test_minimize_below_target() {
        let transformer = JsonTlpTransformer::new([TlpField::minimize("loss", 0.1, 1.0, 1.0)]);
        let input = json!({"loss": 0.05});
        assert_eq!(transformer.transform(input).unwrap(), 0.0);
    }

    #[test]
    fn test_minimize_at_limit() {
        let transformer = JsonTlpTransformer::new([TlpField::minimize("loss", 0.1, 1.0, 1.0)]);
        let input = json!({"loss": 1.0});
        // At limit, value equals priority
        assert_eq!(transformer.transform(input).unwrap(), 1.0);
    }

    #[test]
    fn test_minimize_beyond_limit_is_infinite() {
        let transformer = JsonTlpTransformer::new([TlpField::minimize("loss", 0.1, 1.0, 1.0)]);
        let input = json!({"loss": 1.1});
        assert!(transformer.transform(input).unwrap().is_infinite());
    }

    #[test]
    fn test_minimize_well_beyond_limit() {
        let transformer = JsonTlpTransformer::new([TlpField::minimize("loss", 0.1, 1.0, 1.0)]);
        let input = json!({"loss": 2.0});
        assert!(transformer.transform(input).unwrap().is_infinite());
    }

    #[test]
    fn test_minimize_midpoint() {
        // At midpoint between target=0.0 and limit=1.0, should be priority/2
        let transformer = JsonTlpTransformer::new([TlpField::minimize("loss", 0.0, 1.0, 2.0)]);
        let input = json!({"loss": 0.5});
        assert_eq!(transformer.transform(input).unwrap(), 1.0); // 2.0 * 0.5
    }

    #[test]
    fn test_maximize_at_target() {
        let transformer = JsonTlpTransformer::new([TlpField::maximize("accuracy", 0.9, 0.5, 1.0)]);
        let input = json!({"accuracy": 0.9});
        assert_eq!(transformer.transform(input).unwrap(), 0.0);
    }

    #[test]
    fn test_maximize_above_target() {
        let transformer = JsonTlpTransformer::new([TlpField::maximize("accuracy", 0.9, 0.5, 1.0)]);
        let input = json!({"accuracy": 0.95});
        assert_eq!(transformer.transform(input).unwrap(), 0.0);
    }

    #[test]
    fn test_maximize_at_limit() {
        let transformer = JsonTlpTransformer::new([TlpField::maximize("accuracy", 0.9, 0.5, 1.0)]);
        let input = json!({"accuracy": 0.5});
        // At limit, value equals priority
        assert_eq!(transformer.transform(input).unwrap(), 1.0);
    }

    #[test]
    fn test_maximize_beyond_limit_is_infinite() {
        let transformer = JsonTlpTransformer::new([TlpField::maximize("accuracy", 0.9, 0.5, 1.0)]);
        let input = json!({"accuracy": 0.4});
        assert!(transformer.transform(input).unwrap().is_infinite());
    }

    #[test]
    fn test_maximize_well_below_limit() {
        let transformer = JsonTlpTransformer::new([TlpField::maximize("accuracy", 0.9, 0.5, 1.0)]);
        let input = json!({"accuracy": 0.3});
        assert!(transformer.transform(input).unwrap().is_infinite());
    }

    #[test]
    fn test_maximize_midpoint() {
        // target=1.0, limit=0.0, value=0.5 → halfway → priority/2
        let transformer = JsonTlpTransformer::new([TlpField::maximize("acc", 1.0, 0.0, 2.0)]);
        let input = json!({"acc": 0.5});
        assert_eq!(transformer.transform(input).unwrap(), 1.0); // 2.0 * 0.5
    }

    #[test]
    fn test_multiple_fields_sum() {
        let transformer = JsonTlpTransformer::new([
            TlpField::minimize("loss", 0.0, 1.0, 1.0),
            TlpField::maximize("accuracy", 1.0, 0.0, 1.0),
        ]);
        // loss=0.5 → 0.5, accuracy=0.5 → 0.5, total = 1.0
        let input = json!({"loss": 0.5, "accuracy": 0.5});
        assert_eq!(transformer.transform(input).unwrap(), 1.0);
    }

    #[test]
    fn test_one_field_infinite_makes_sum_infinite() {
        let transformer = JsonTlpTransformer::new([
            TlpField::minimize("loss", 0.0, 1.0, 1.0),
            TlpField::maximize("accuracy", 1.0, 0.5, 1.0),
        ]);
        // loss=0.5 is fine, but accuracy=0.3 is below limit
        let input = json!({"loss": 0.5, "accuracy": 0.3});
        assert!(transformer.transform(input).unwrap().is_infinite());
    }

    #[test]
    fn test_missing_field_error() {
        let transformer = JsonTlpTransformer::new([TlpField::minimize("missing", 0.0, 1.0, 1.0)]);
        let input = json!({"other": 0.5});
        assert!(transformer.transform(input).is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let transformer = JsonTlpTransformer::new([])
            .with_field(TlpField::minimize("loss", 0.0, 1.0, 1.0))
            .with_field(TlpField::maximize("accuracy", 1.0, 0.0, 1.0));

        let input = json!({"loss": 0.0, "accuracy": 1.0});
        assert_eq!(transformer.transform(input).unwrap(), 0.0);
    }

    #[test]
    fn test_target_equals_limit() {
        let field = TlpField::new("x", 1.0, 1.0, 1.0);
        let transformer = JsonTlpTransformer::new([field]);

        // At target/limit: 0.0
        assert_eq!(transformer.transform(json!({"x": 1.0})).unwrap(), 0.0);
        // Above: 0.0
        assert_eq!(transformer.transform(json!({"x": 2.0})).unwrap(), 0.0);
        // Below: INFINITY
        assert!(
            transformer
                .transform(json!({"x": 0.5}))
                .unwrap()
                .is_infinite()
        );
    }
}
