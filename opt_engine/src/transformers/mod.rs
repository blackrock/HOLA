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

//! Result transformers (the trust boundary).
//!
//! A transformer sits between raw worker output (typically JSON) and the
//! engine's typed observation. It enforces schema validation, extracts the
//! relevant fields, and handles direction (minimize vs. maximize).
//!
//! Built-in transformers:
//!
//! - [`JsonFieldTransformer`] — extracts a single numeric field.
//! - [`JsonWeightedTransformer`] — weighted sum of multiple fields.
//! - [`JsonTlpTransformer`] — Target-Limit-Priority scalarizer (scalar output).
//! - [`JsonGroupedTlpTransformer`] — grouped TLP scalarizer (multi-objective
//!   output as `BTreeMap<String, f64>`).

mod json_field;
mod json_tlp;
mod json_tlp_grouped;
mod json_weighted;

pub use json_field::JsonFieldTransformer;
pub use json_tlp::{JsonTlpTransformer, TlpField};
pub use json_tlp_grouped::{GroupedTlpField, JsonGroupedTlpTransformer};
pub use json_weighted::{JsonWeightedTransformer, WeightedField};
