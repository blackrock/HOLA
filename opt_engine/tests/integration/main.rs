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

//! Integration tests for the opt_engine public API.
//!
//! Engine cycles and end-to-end optimization with generic spaces.

mod end_to_end;
mod engine;
mod leaderboard_scalability;

#[test]
fn integration_test_files_are_referenced_by_harness() {
    let integration_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("integration");
    let allowed = [
        "end_to_end.rs",
        "engine.rs",
        "leaderboard_scalability.rs",
        "main.rs",
    ];

    let mut unreferenced = Vec::new();
    for entry in std::fs::read_dir(&integration_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if !allowed.contains(&name.as_str()) {
                unreferenced.push(name);
            }
        }
    }

    assert!(
        unreferenced.is_empty(),
        "opt_engine integration test files must be referenced by tests/integration/main.rs: {unreferenced:?}"
    );
}
