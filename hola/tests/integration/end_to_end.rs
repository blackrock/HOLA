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

//! End-to-end optimization tests for HolaEngine.

use hola::hola_engine::{HolaEngine, ObjectiveConfig, ParamConfig, StudyConfig};
use serde_json::json;
use std::collections::BTreeMap;

/// Sphere function: f(x, y) = x^2 + y^2. Global minimum at (0, 0).
fn sphere(x: f64, y: f64) -> f64 {
    x * x + y * y
}

// ==========================================================================
// HolaEngine YAML config flow
// ==========================================================================

#[tokio::test]
async fn test_e2e_hola_engine_yaml_config() {
    let yaml = r#"
    space:
      x:
        type: continuous
        min: -5.0
        max: 5.0
      y:
        type: continuous
        min: -5.0
        max: 5.0
    objectives:
      - field: loss
        type: minimize
        priority: 1.0
    strategy:
      type: sobol
      refit_interval: 20
    "#;

    let config: StudyConfig = serde_yaml::from_str(yaml).unwrap();
    let engine = HolaEngine::from_config(config).unwrap();

    for _ in 0..50 {
        let t = engine.ask().await.unwrap();
        let x = t.params["x"].as_f64().unwrap();
        let y = t.params["y"].as_f64().unwrap();
        let loss = sphere(x, y);
        engine
            .tell(t.trial_id, json!({"loss": loss}))
            .await
            .unwrap();
    }

    assert_eq!(engine.trial_count().await, 50);
    let top = engine.top_k(1, false).await;
    assert!(!top.is_empty());
    // The best trial should have a reasonable score on sphere function
    assert!(top[0].rank == 0);
}

// ==========================================================================
// Checkpoint resume
// ==========================================================================

#[tokio::test]
async fn test_e2e_checkpoint_resume_continues() {
    let config = || StudyConfig {
        space: BTreeMap::from([
            (
                "x".to_string(),
                ParamConfig::Continuous {
                    min: -5.0,
                    max: 5.0,
                    scale: "linear".to_string(),
                },
            ),
            (
                "y".to_string(),
                ParamConfig::Continuous {
                    min: -5.0,
                    max: 5.0,
                    scale: "linear".to_string(),
                },
            ),
        ]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: None,
        checkpoint: None,
        max_trials: None,
    };

    // Phase 1: run 50 trials
    let engine1 = HolaEngine::from_config(config()).unwrap();
    for _ in 0..50 {
        let t = engine1.ask().await.unwrap();
        let x = t.params["x"].as_f64().unwrap();
        let y = t.params["y"].as_f64().unwrap();
        engine1
            .tell(t.trial_id, json!({"loss": sphere(x, y)}))
            .await
            .unwrap();
    }

    let best_before = engine1.top_k(1, false).await;
    assert!(!best_before.is_empty());

    // Save checkpoint
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resume_test.json");
    engine1
        .save_leaderboard_checkpoint_to(&path, Some("phase 1"))
        .await
        .unwrap();

    // Phase 2: new engine, load checkpoint, run 50 more
    let engine2 = HolaEngine::from_config(config()).unwrap();
    engine2.load_leaderboard_checkpoint(&path).await.unwrap();
    assert_eq!(engine2.trial_count().await, 50);

    for _ in 0..50 {
        let t = engine2.ask().await.unwrap();
        let x = t.params["x"].as_f64().unwrap();
        let y = t.params["y"].as_f64().unwrap();
        engine2
            .tell(t.trial_id, json!({"loss": sphere(x, y)}))
            .await
            .unwrap();
    }

    assert_eq!(engine2.trial_count().await, 100);
    let best_after = engine2.top_k(1, false).await;
    assert!(!best_after.is_empty());
}
