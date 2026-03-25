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

//! JSON single-field transformer.

use crate::objectives::directed_value;
use crate::traits::Transformer;

/// A Transformer that extracts a single numeric field from JSON.
///
/// Supports both minimization (return value as-is) and maximization (return negated value).
pub struct JsonFieldTransformer {
    field: String,
    /// If true, negate the value (for maximization).
    negate: bool,
}

impl JsonFieldTransformer {
    pub fn minimize(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            negate: false,
        }
    }

    /// Create a transformer that maximizes the specified field.
    ///
    /// The extracted value is negated (multiplied by -1).
    pub fn maximize(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            negate: true,
        }
    }

    pub fn loss() -> Self {
        Self::minimize("loss")
    }
}

impl Default for JsonFieldTransformer {
    fn default() -> Self {
        Self::loss()
    }
}

impl Transformer for JsonFieldTransformer {
    type ForeignInput = serde_json::Value;
    type Output = f64;

    fn transform(&self, input: serde_json::Value) -> Result<f64, String> {
        let value = input
            .get(&self.field)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| {
                format!(
                    "Invalid Schema: Missing or non-numeric '{}' field",
                    self.field
                )
            })?;

        Ok(directed_value(value, self.negate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_minimize_field() {
        let transformer = JsonFieldTransformer::minimize("loss");
        let input = json!({"loss": 0.5});
        assert_eq!(transformer.transform(input).unwrap(), 0.5);
    }

    #[test]
    fn test_maximize_field() {
        let transformer = JsonFieldTransformer::maximize("accuracy");
        let input = json!({"accuracy": 0.9});
        assert_eq!(transformer.transform(input).unwrap(), -0.9);
    }

    #[test]
    fn test_loss_is_minimize() {
        let transformer = JsonFieldTransformer::loss();
        let input = json!({"loss": 0.5});
        assert_eq!(transformer.transform(input).unwrap(), 0.5);
    }

    #[test]
    fn test_default_is_loss() {
        let transformer = JsonFieldTransformer::default();
        let input = json!({"loss": 0.5});
        assert_eq!(transformer.transform(input).unwrap(), 0.5);
    }

    #[test]
    fn test_missing_field_error() {
        let transformer = JsonFieldTransformer::minimize("missing");
        let input = json!({"loss": 0.5});
        let err = transformer.transform(input).unwrap_err();
        assert!(err.contains("missing"));
    }

    #[test]
    fn test_non_numeric_field_error() {
        let transformer = JsonFieldTransformer::minimize("name");
        let input = json!({"name": "not a number"});
        let err = transformer.transform(input).unwrap_err();
        assert!(err.contains("name"));
    }

    #[test]
    fn test_null_value_error() {
        let transformer = JsonFieldTransformer::minimize("loss");
        let result = transformer.transform(json!({"loss": null}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("loss"));
    }

    #[test]
    fn test_nested_object_error() {
        let transformer = JsonFieldTransformer::minimize("loss");
        let result = transformer.transform(json!({"loss": {"value": 0.5}}));
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_fields_ignored() {
        let transformer = JsonFieldTransformer::minimize("loss");
        let input = json!({"loss": 0.5, "extra": "stuff", "more": [1, 2, 3]});
        assert!((transformer.transform(input).unwrap() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_very_large_value() {
        let transformer = JsonFieldTransformer::minimize("loss");
        let result = transformer.transform(json!({"loss": 1e308})).unwrap();
        assert!((result - 1e308).abs() < 1e293);
    }
}
