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

//! JSON Grouped TLP (Target-Limit-Priority) scalarizer transformer.

use crate::objectives::tlp_score;
use crate::traits::Transformer;
use std::collections::BTreeMap;

/// Specification for a single field in the grouped TLP scalarizer.
#[derive(Debug, Clone)]
pub struct GroupedTlpField {
    /// The field name to extract from JSON.
    pub field: String,
    /// The priority group this field belongs to.
    pub group: String,
    /// The target value at/beyond which the user is satisfied (output = 0).
    pub target: f64,
    /// The limit value beyond which the user is infinitely unsatisfied (output = +inf).
    pub limit: f64,
    /// The priority weight within the group.
    pub priority: f64,
}

impl GroupedTlpField {
    /// Create a new grouped TLP field specification.
    ///
    /// - If `target < limit`: minimization (lower values are better)
    /// - If `target > limit`: maximization (higher values are better)
    ///
    /// # Panics
    /// Panics if `priority` is negative.
    pub fn new(
        field: impl Into<String>,
        group: impl Into<String>,
        target: f64,
        limit: f64,
        priority: f64,
    ) -> Self {
        assert!(
            priority >= 0.0,
            "Priority must be non-negative (got {priority})"
        );
        Self {
            field: field.into(),
            group: group.into(),
            target,
            limit,
            priority,
        }
    }

    /// Create a minimization field (target < limit).
    pub fn minimize(
        field: impl Into<String>,
        group: impl Into<String>,
        target: f64,
        limit: f64,
        priority: f64,
    ) -> Self {
        debug_assert!(target < limit, "For minimize, target should be < limit");
        Self::new(field, group, target, limit, priority)
    }

    /// Create a maximization field (target > limit).
    pub fn maximize(
        field: impl Into<String>,
        group: impl Into<String>,
        target: f64,
        limit: f64,
        priority: f64,
    ) -> Self {
        debug_assert!(target > limit, "For maximize, target should be > limit");
        Self::new(field, group, target, limit, priority)
    }

    /// Compute the scalarized value for this field (before weighting).
    fn scalarize(&self, value: f64) -> f64 {
        tlp_score(value, self.target, self.limit)
    }
}

/// A Grouped TLP (Target-Limit-Priority) scalarizer transformer.
///
/// Fields are organized into priority groups. Each group produces its own
/// scalarized value, and the output is a mapping from group name to value.
///
/// Within each group, fields are combined using weighted TLP scalarization.
/// Optionally, weights can be normalized to sum to 1 within each group.
///
/// # Example
///
/// ```
/// use opt_engine::transformers::{JsonGroupedTlpTransformer, GroupedTlpField};
///
/// let transformer = JsonGroupedTlpTransformer::new([
///     // Performance group
///     GroupedTlpField::minimize("loss", "performance", 0.01, 1.0, 1.0),
///     GroupedTlpField::maximize("accuracy", "performance", 0.95, 0.5, 2.0),
///     // Resource group
///     GroupedTlpField::minimize("latency_ms", "resources", 100.0, 500.0, 1.0),
///     GroupedTlpField::minimize("memory_mb", "resources", 512.0, 2048.0, 0.5),
/// ]);
/// ```
pub struct JsonGroupedTlpTransformer {
    fields: Vec<GroupedTlpField>,
    /// If true, normalize weights within each group to sum to 1.
    normalize_weights: bool,
}

impl JsonGroupedTlpTransformer {
    /// Create a new grouped TLP transformer with the specified field specifications.
    ///
    /// Weights are not normalized by default.
    pub fn new<I>(fields: I) -> Self
    where
        I: IntoIterator<Item = GroupedTlpField>,
    {
        Self {
            fields: fields.into_iter().collect(),
            normalize_weights: false,
        }
    }

    /// Create a new grouped TLP transformer with normalized weights.
    ///
    /// Within each group, weights are normalized to sum to 1.
    pub fn normalized<I>(fields: I) -> Self
    where
        I: IntoIterator<Item = GroupedTlpField>,
    {
        Self {
            fields: fields.into_iter().collect(),
            normalize_weights: true,
        }
    }

    /// Builder method to set weight normalization.
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize_weights = normalize;
        self
    }

    /// Builder method to add a field specification.
    pub fn with_field(mut self, field: GroupedTlpField) -> Self {
        self.fields.push(field);
        self
    }

    /// Get the list of unique group names.
    pub fn groups(&self) -> Vec<&str> {
        let mut groups: Vec<&str> = self.fields.iter().map(|f| f.group.as_str()).collect();
        groups.sort();
        groups.dedup();
        groups
    }
}

impl Transformer for JsonGroupedTlpTransformer {
    type ForeignInput = serde_json::Value;
    type Output = BTreeMap<String, f64>;

    fn transform(&self, input: serde_json::Value) -> Result<BTreeMap<String, f64>, String> {
        // Group fields by their priority group
        let mut groups: BTreeMap<&str, Vec<&GroupedTlpField>> = BTreeMap::new();
        for field in &self.fields {
            groups.entry(&field.group).or_default().push(field);
        }

        // Compute weight sums for normalization if needed
        let weight_sums: BTreeMap<&str, f64> = if self.normalize_weights {
            groups
                .iter()
                .map(|(group, fields)| {
                    let sum: f64 = fields.iter().map(|f| f.priority).sum();
                    (*group, sum)
                })
                .collect()
        } else {
            BTreeMap::new()
        };

        let mut result = BTreeMap::new();

        for (group, fields) in groups {
            let mut group_sum = 0.0;
            let weight_sum = weight_sums.get(group).copied().unwrap_or(1.0);

            for spec in fields {
                let value = input
                    .get(&spec.field)
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| {
                        format!(
                            "Invalid Schema: Missing or non-numeric '{}' field",
                            spec.field
                        )
                    })?;

                let scalarized = spec.scalarize(value);

                // Apply weight (normalized if requested)
                let weight = if self.normalize_weights && weight_sum > 0.0 {
                    spec.priority / weight_sum
                } else {
                    spec.priority
                };

                group_sum += weight * scalarized;
            }

            result.insert(group.to_string(), group_sum);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_single_group() {
        let transformer = JsonGroupedTlpTransformer::new([GroupedTlpField::minimize(
            "loss", "perf", 0.0, 1.0, 1.0,
        )]);
        let input = json!({"loss": 0.5});
        let result = transformer.transform(input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result["perf"], 0.5); // priority * scalarized = 1.0 * 0.5
    }

    #[test]
    fn test_multiple_groups() {
        let transformer = JsonGroupedTlpTransformer::new([
            GroupedTlpField::minimize("loss", "perf", 0.0, 1.0, 1.0),
            GroupedTlpField::minimize("memory", "resources", 0.0, 100.0, 1.0),
        ]);
        let input = json!({"loss": 0.5, "memory": 50.0});
        let result = transformer.transform(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result["perf"], 0.5);
        assert_eq!(result["resources"], 0.5);
    }

    #[test]
    fn test_multiple_fields_same_group() {
        let transformer = JsonGroupedTlpTransformer::new([
            GroupedTlpField::minimize("loss", "perf", 0.0, 1.0, 1.0),
            GroupedTlpField::minimize("error", "perf", 0.0, 1.0, 2.0),
        ]);
        let input = json!({"loss": 0.5, "error": 0.5});
        let result = transformer.transform(input).unwrap();
        // 1.0 * 0.5 + 2.0 * 0.5 = 1.5
        assert_eq!(result["perf"], 1.5);
    }

    #[test]
    fn test_normalized_weights() {
        let transformer = JsonGroupedTlpTransformer::normalized([
            GroupedTlpField::minimize("loss", "perf", 0.0, 1.0, 1.0),
            GroupedTlpField::minimize("error", "perf", 0.0, 1.0, 3.0),
        ]);
        let input = json!({"loss": 1.0, "error": 1.0}); // Both at limit = scalarized 1.0
        let result = transformer.transform(input).unwrap();
        // Normalized: (1/4) * 1.0 + (3/4) * 1.0 = 1.0
        assert!((result["perf"] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_different_values() {
        let transformer = JsonGroupedTlpTransformer::normalized([
            GroupedTlpField::minimize("a", "g", 0.0, 1.0, 1.0), // weight 0.25
            GroupedTlpField::minimize("b", "g", 0.0, 1.0, 3.0), // weight 0.75
        ]);
        let input = json!({"a": 1.0, "b": 0.0}); // a at limit (1.0), b at target (0.0)
        let result = transformer.transform(input).unwrap();
        // (1/4) * 1.0 + (3/4) * 0.0 = 0.25
        assert!((result["g"] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_maximize_field() {
        let transformer = JsonGroupedTlpTransformer::new([GroupedTlpField::maximize(
            "accuracy", "perf", 1.0, 0.0, 1.0,
        )]);
        let input = json!({"accuracy": 0.5}); // Midpoint
        let result = transformer.transform(input).unwrap();
        assert_eq!(result["perf"], 0.5);
    }

    #[test]
    fn test_infinity_propagates() {
        let transformer = JsonGroupedTlpTransformer::new([
            GroupedTlpField::minimize("loss", "perf", 0.0, 1.0, 1.0),
            GroupedTlpField::minimize("error", "perf", 0.0, 1.0, 1.0),
        ]);
        let input = json!({"loss": 0.5, "error": 1.5}); // error beyond limit (> 1.0)
        let result = transformer.transform(input).unwrap();
        assert!(result["perf"].is_infinite());
    }

    #[test]
    fn test_missing_field_error() {
        let transformer = JsonGroupedTlpTransformer::new([GroupedTlpField::minimize(
            "missing", "perf", 0.0, 1.0, 1.0,
        )]);
        let input = json!({"other": 0.5});
        assert!(transformer.transform(input).is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let transformer = JsonGroupedTlpTransformer::new([])
            .with_field(GroupedTlpField::minimize("loss", "perf", 0.0, 1.0, 1.0))
            .with_field(GroupedTlpField::minimize(
                "mem",
                "resources",
                0.0,
                100.0,
                1.0,
            ))
            .with_normalization(true);

        let input = json!({"loss": 0.5, "mem": 50.0});
        let result = transformer.transform(input).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_groups_method() {
        let transformer = JsonGroupedTlpTransformer::new([
            GroupedTlpField::minimize("a", "zebra", 0.0, 1.0, 1.0),
            GroupedTlpField::minimize("b", "alpha", 0.0, 1.0, 1.0),
            GroupedTlpField::minimize("c", "alpha", 0.0, 1.0, 1.0),
        ]);
        let groups = transformer.groups();
        assert_eq!(groups, vec!["alpha", "zebra"]);
    }
}
