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

//! JSON weighted multi-field transformer.

use crate::objectives::directed_value;
use crate::traits::Transformer;

/// Specification for a single field in the weighted transformer.
#[derive(Debug, Clone)]
pub struct WeightedField {
    /// The field name to extract from JSON.
    pub field: String,
    /// The weight to apply to this field.
    pub weight: f64,
    /// If true, negate the value before weighting (for maximization).
    pub negate: bool,
}

impl WeightedField {
    /// Create a field to minimize with the given weight.
    ///
    /// Contributes `weight * value` to the sum.
    ///
    /// # Panics
    /// Panics if `weight` is negative. Use [`maximize`](Self::maximize) to flip direction.
    pub fn minimize(field: impl Into<String>, weight: f64) -> Self {
        assert!(
            weight >= 0.0,
            "Weight must be non-negative (got {weight}); use maximize() to flip direction"
        );
        Self {
            field: field.into(),
            weight,
            negate: false,
        }
    }

    /// Create a field to maximize with the given weight.
    ///
    /// Contributes `weight * (-value)` to the sum.
    ///
    /// # Panics
    /// Panics if `weight` is negative. Use [`minimize`](Self::minimize) to flip direction.
    pub fn maximize(field: impl Into<String>, weight: f64) -> Self {
        assert!(
            weight >= 0.0,
            "Weight must be non-negative (got {weight}); use minimize() to flip direction"
        );
        Self {
            field: field.into(),
            weight,
            negate: true,
        }
    }
}

/// A Transformer that extracts multiple numeric fields and returns their weighted sum.
///
/// Each field has an associated weight and direction (minimize/maximize).
/// The output is computed as: `sum(weight_i * value_i)` for minimized fields
/// and `sum(weight_i * -value_i)` for maximized fields.
///
/// # Example
///
/// ```
/// use opt_engine::transformers::{JsonWeightedTransformer, WeightedField};
///
/// let transformer = JsonWeightedTransformer::new([
///     WeightedField::minimize("loss", 1.0),
///     WeightedField::minimize("latency_ms", 0.01),
///     WeightedField::maximize("accuracy", 1.0),  // negated
/// ]);
/// ```
pub struct JsonWeightedTransformer {
    fields: Vec<WeightedField>,
}

impl JsonWeightedTransformer {
    /// Create a new transformer with the specified weighted fields.
    pub fn new<I>(fields: I) -> Self
    where
        I: IntoIterator<Item = WeightedField>,
    {
        Self {
            fields: fields.into_iter().collect(),
        }
    }

    /// Create a transformer with a single minimized field of weight 1.0.
    pub fn single(field: impl Into<String>) -> Self {
        Self::new([WeightedField::minimize(field, 1.0)])
    }

    /// Builder method to add a minimized field.
    pub fn with_minimize(mut self, field: impl Into<String>, weight: f64) -> Self {
        self.fields.push(WeightedField::minimize(field, weight));
        self
    }

    /// Builder method to add a maximized field.
    pub fn with_maximize(mut self, field: impl Into<String>, weight: f64) -> Self {
        self.fields.push(WeightedField::maximize(field, weight));
        self
    }
}

impl Transformer for JsonWeightedTransformer {
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

            sum += spec.weight * directed_value(value, spec.negate);
        }

        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_single_field() {
        let transformer = JsonWeightedTransformer::single("loss");
        let input = json!({"loss": 0.5});
        assert_eq!(transformer.transform(input).unwrap(), 0.5);
    }

    #[test]
    fn test_minimize_weighted() {
        let transformer = JsonWeightedTransformer::new([
            WeightedField::minimize("loss", 1.0),
            WeightedField::minimize("penalty", 0.5),
        ]);
        let input = json!({"loss": 1.0, "penalty": 2.0});
        // 1.0 * 1.0 + 0.5 * 2.0 = 2.0
        assert_eq!(transformer.transform(input).unwrap(), 2.0);
    }

    #[test]
    fn test_maximize_field() {
        let transformer = JsonWeightedTransformer::new([WeightedField::maximize("accuracy", 1.0)]);
        let input = json!({"accuracy": 0.9});
        // 1.0 * (-0.9) = -0.9
        assert_eq!(transformer.transform(input).unwrap(), -0.9);
    }

    #[test]
    fn test_mixed_minimize_maximize() {
        let transformer = JsonWeightedTransformer::new([
            WeightedField::minimize("loss", 1.0),
            WeightedField::maximize("accuracy", 1.0),
        ]);
        let input = json!({"loss": 0.3, "accuracy": 0.9});
        // 1.0 * 0.3 + 1.0 * (-0.9) = -0.6
        let result = transformer.transform(input).unwrap();
        assert!(
            (result - (-0.6)).abs() < 1e-10,
            "Expected -0.6, got {result}"
        );
    }

    #[test]
    fn test_builder_pattern() {
        let transformer = JsonWeightedTransformer::single("loss")
            .with_minimize("latency", 0.01)
            .with_maximize("accuracy", 1.0);

        let input = json!({"loss": 0.5, "latency": 100.0, "accuracy": 0.9});
        // 1.0 * 0.5 + 0.01 * 100.0 + 1.0 * (-0.9) = 0.5 + 1.0 - 0.9 = 0.6
        let result = transformer.transform(input).unwrap();
        assert!((result - 0.6).abs() < 1e-10, "Expected 0.6, got {result}");
    }

    #[test]
    fn test_missing_field_error() {
        let transformer = JsonWeightedTransformer::new([WeightedField::minimize("missing", 1.0)]);
        let input = json!({"loss": 0.5});
        let err = transformer.transform(input).unwrap_err();
        assert!(err.contains("missing"));
    }

    #[test]
    fn test_empty_fields_returns_zero() {
        let transformer = JsonWeightedTransformer::new([]);
        let input = json!({"anything": 123});
        assert_eq!(transformer.transform(input).unwrap(), 0.0);
    }

    #[test]
    fn test_zero_weight() {
        let transformer = JsonWeightedTransformer::new([
            WeightedField::minimize("loss", 1.0),
            WeightedField::minimize("noise", 0.0),
        ]);
        let input = json!({"loss": 0.5, "noise": 999999.0});
        assert!((transformer.transform(input).unwrap() - 0.5).abs() < 1e-15);
    }
}
