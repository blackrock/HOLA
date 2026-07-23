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

//! Integration tests for HolaEngine (type-erased layer).
//!
//! Exercises config parsing, ask/tell flows, strategy types, scalarization,
//! objectives, checkpoints, refit, and all parameter types.

use hola::hola_engine::{
    CheckpointLoadKind, HolaEngine, ObjectiveConfig, ParamConfig, StrategyConfig, StudyConfig,
};
use opt_engine::traits::SampleSpace;
use serde_json::json;
use std::collections::BTreeMap;

// ==========================================================================
// Config parsing
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_config_parsing() {
    let yaml_config = r#"
    space:
      learning_rate:
        type: real
        min: 0.0001
        max: 0.1
        scale: log10
      num_layers:
        type: integer
        min: 1
        max: 10
    objectives:
      - field: loss
        type: minimize
        priority: 1.0
      - field: latency
        type: minimize
        target: 100
        limit: 500
        priority: 0.5
    strategy:
      type: sobol
      refit_interval: 20
    "#;

    let config: StudyConfig = serde_yaml::from_str(yaml_config).unwrap();
    assert_eq!(config.space.len(), 2);
    assert_eq!(config.objectives.len(), 2);
    assert!(config.strategy.is_some());
    assert!(config.checkpoint.is_none());

    let engine = HolaEngine::from_config(config).unwrap();
    assert_eq!(engine.trial_count().await, 0);
}

#[tokio::test]
async fn test_dyn_engine_config_with_checkpoint() {
    let yaml = r#"
    space:
      x:
        type: real
        min: 0.0
        max: 1.0
    objectives:
      - field: loss
        type: minimize
    checkpoint:
      directory: "/tmp/hola_test_ckpts"
      interval: 10
      max_checkpoints: 3
    "#;

    let config: StudyConfig = serde_yaml::from_str(yaml).unwrap();
    assert!(config.checkpoint.is_some());
    let ckpt = config.checkpoint.as_ref().unwrap();
    assert_eq!(ckpt.directory, "/tmp/hola_test_ckpts");
    assert_eq!(ckpt.interval, 10);
    assert_eq!(ckpt.max_checkpoints, Some(3));

    let engine = HolaEngine::from_config(config).unwrap();
    let _t = engine.ask().await.unwrap();
}

fn valid_config_for_validation() -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: Some(StrategyConfig {
            strategy_type: "gmm".to_string(),
            refit_interval: 20,
            total_budget: None,
            exploration_budget: None,
            ongoing_exploration_period: None,
            seed: None,
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    }
}

fn assert_config_error(config: StudyConfig, expected: &[&str]) {
    let err = match HolaEngine::from_config(config) {
        Ok(_) => panic!("expected config validation to fail"),
        Err(err) => err,
    };
    for needle in expected {
        assert!(
            err.contains(needle),
            "expected error {err:?} to contain {needle:?}"
        );
    }
}

#[test]
fn test_dyn_engine_config_validation_rejects_invalid_scale() {
    let mut config = valid_config_for_validation();
    config.space.insert(
        "lr".to_string(),
        ParamConfig::Real {
            min: 1.0e-4,
            max: 1.0e-1,
            scale: "log2".to_string(),
        },
    );
    assert_config_error(config, &["Parameter 'lr'", "unknown real scale", "log2"]);
}

#[test]
fn test_dyn_engine_config_validation_rejects_invalid_strategy() {
    let mut config = valid_config_for_validation();
    config.strategy.as_mut().unwrap().strategy_type = "soboll".to_string();
    assert_config_error(config, &["Unknown strategy type", "soboll"]);
}

#[test]
fn test_dyn_engine_config_validation_rejects_invalid_objective_type() {
    let mut config = valid_config_for_validation();
    config.objectives[0].obj_type = "minimise".to_string();
    assert_config_error(
        config,
        &["Objective 'loss'", "unknown objective type", "minimise"],
    );
}

#[test]
fn test_dyn_engine_config_validation_rejects_invalid_space_shapes() {
    let mut real = valid_config_for_validation();
    real.space.insert(
        "x".to_string(),
        ParamConfig::Real {
            min: 2.0,
            max: 1.0,
            scale: "linear".to_string(),
        },
    );
    assert_config_error(
        real,
        &["Parameter 'x'", "min must be less than or equal to max"],
    );

    let mut integer = valid_config_for_validation();
    integer.space.insert(
        "layers".to_string(),
        ParamConfig::Integer { min: 10, max: 1 },
    );
    assert_config_error(
        integer,
        &[
            "Parameter 'layers'",
            "min must be less than or equal to max",
        ],
    );

    let mut categorical = valid_config_for_validation();
    categorical.space.insert(
        "optimizer".to_string(),
        ParamConfig::Categorical { choices: vec![] },
    );
    assert_config_error(
        categorical,
        &["Parameter 'optimizer'", "choices must not be empty"],
    );
}

#[test]
fn test_dyn_engine_config_validation_rejects_non_finite_real_bounds() {
    let mut nan = valid_config_for_validation();
    nan.space.insert(
        "x".to_string(),
        ParamConfig::Real {
            min: f64::NAN,
            max: 1.0,
            scale: "linear".to_string(),
        },
    );
    assert_config_error(nan, &["Parameter 'x'", "bounds must be finite"]);

    let mut inf = valid_config_for_validation();
    inf.space.insert(
        "x".to_string(),
        ParamConfig::Real {
            min: 0.0,
            max: f64::INFINITY,
            scale: "linear".to_string(),
        },
    );
    assert_config_error(inf, &["Parameter 'x'", "bounds must be finite"]);
}

#[test]
fn test_dyn_engine_config_validation_rejects_invalid_refit_and_priority() {
    let mut refit = valid_config_for_validation();
    refit.strategy.as_mut().unwrap().refit_interval = 0;
    assert_config_error(refit, &["strategy.refit_interval", "at least 1"]);

    let mut priority = valid_config_for_validation();
    priority.objectives[0].priority = -1.0;
    assert_config_error(priority, &["Objective 'loss'", "priority"]);

    let mut elite_fraction = valid_config_for_validation();
    elite_fraction.strategy.as_mut().unwrap().elite_fraction = Some(f64::NAN);
    assert_config_error(elite_fraction, &["strategy.elite_fraction", "finite"]);

    let mut zero_fit_samples = valid_config_for_validation();
    zero_fit_samples
        .strategy
        .as_mut()
        .unwrap()
        .max_refit_samples = 0;
    assert_config_error(
        zero_fit_samples,
        &["strategy.max_refit_samples", "at least 1"],
    );

    let mut undersized_candidate_workset = valid_config_for_validation();
    let strategy = undersized_candidate_workset.strategy.as_mut().unwrap();
    strategy.max_refit_samples = 100;
    strategy.max_refit_candidates = 99;
    assert_config_error(
        undersized_candidate_workset,
        &[
            "strategy.max_refit_candidates",
            "at least max_refit_samples (100)",
        ],
    );
}

// ==========================================================================
// Ask/Tell flow
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_ask_tell_flow() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    assert_eq!(engine.trial_count().await, 0);
    assert!(engine.top_k(1, false).await.is_empty());

    let t0 = engine.ask().await.unwrap();
    assert_eq!(t0.trial_id, 0);
    let t1 = engine.ask().await.unwrap();
    assert_eq!(t1.trial_id, 1);

    engine
        .tell(t0.trial_id, json!({"loss": 0.8}))
        .await
        .unwrap();
    assert_eq!(engine.trial_count().await, 1);

    engine
        .tell(t1.trial_id, json!({"loss": 0.2}))
        .await
        .unwrap();
    assert_eq!(engine.trial_count().await, 2);

    let best = engine.top_k(1, false).await.into_iter().next().unwrap();
    assert_eq!(best.trial_id, 1);
    assert_eq!(engine.trial_count().await, 2);
}

#[tokio::test]
async fn test_dyn_engine_unknown_trial_error() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let result = engine.tell(999, json!({"loss": 0.5})).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("999"));
}

#[tokio::test]
async fn test_dyn_engine_double_tell_error() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let t = engine.ask().await.unwrap();
    engine.tell(t.trial_id, json!({"loss": 0.5})).await.unwrap();
    assert!(engine.tell(t.trial_id, json!({"loss": 0.3})).await.is_err());
}

#[tokio::test]
async fn test_dyn_engine_out_of_order_tell_preserves_public_trial_ids() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let t0 = engine.ask().await.unwrap();
    let t1 = engine.ask().await.unwrap();

    let completed_1 = engine
        .tell(t1.trial_id, json!({"loss": 0.2}))
        .await
        .unwrap();
    assert_eq!(completed_1.trial_id, t1.trial_id);
    assert_eq!(completed_1.params, t1.params);

    let completed_0 = engine
        .tell(t0.trial_id, json!({"loss": 0.8}))
        .await
        .unwrap();
    assert_eq!(completed_0.trial_id, t0.trial_id);
    assert_eq!(completed_0.params, t0.params);

    let ids: Vec<u64> = engine
        .trials("index", true)
        .await
        .into_iter()
        .map(|trial| trial.trial_id)
        .collect();
    assert_eq!(ids, vec![0, 1]);
}

// ==========================================================================
// All parameter types
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_all_param_types() {
    let config = StudyConfig {
        space: BTreeMap::from([
            (
                "lr".to_string(),
                ParamConfig::Real {
                    min: 1e-4,
                    max: 0.1,
                    scale: "log10".to_string(),
                },
            ),
            (
                "layers".to_string(),
                ParamConfig::Integer { min: 1, max: 10 },
            ),
            (
                "optimizer".to_string(),
                ParamConfig::Categorical {
                    choices: vec!["adam".into(), "sgd".into(), "rmsprop".into()],
                },
            ),
            (
                "dropout".to_string(),
                ParamConfig::Real {
                    min: 0.0,
                    max: 0.5,
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    for _ in 0..10 {
        let t = engine.ask().await.unwrap();
        assert!(engine.space().contains(&t.params));
        engine.tell(t.trial_id, json!({"loss": 0.5})).await.unwrap();
    }
    assert_eq!(engine.trial_count().await, 10);
}

#[tokio::test]
async fn test_dyn_engine_categorical_params() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "optimizer".to_string(),
            ParamConfig::Categorical {
                choices: vec!["adam".into(), "sgd".into(), "rmsprop".into()],
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let valid_choices: Vec<String> = vec!["adam".into(), "sgd".into(), "rmsprop".into()];
    for _ in 0..20 {
        let t = engine.ask().await.unwrap();
        let opt = t.params.get("optimizer").unwrap().as_str().unwrap();
        assert!(valid_choices.contains(&opt.to_string()));
    }
}

#[tokio::test]
async fn test_dyn_engine_ask_returns_valid_params() {
    let config = StudyConfig {
        space: BTreeMap::from([
            (
                "lr".to_string(),
                ParamConfig::Real {
                    min: 0.001,
                    max: 1.0,
                    scale: "log10".to_string(),
                },
            ),
            (
                "batch".to_string(),
                ParamConfig::Integer { min: 16, max: 256 },
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    for _ in 0..20 {
        let t = engine.ask().await.unwrap();
        assert!(engine.space().contains(&t.params));
    }
}

// ==========================================================================
// Param info
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_param_info() {
    let config = StudyConfig {
        space: BTreeMap::from([
            (
                "lr".to_string(),
                ParamConfig::Real {
                    min: 1e-4,
                    max: 0.1,
                    scale: "log10".to_string(),
                },
            ),
            (
                "layers".to_string(),
                ParamConfig::Integer { min: 1, max: 10 },
            ),
            (
                "opt".to_string(),
                ParamConfig::Categorical {
                    choices: vec!["adam".into(), "sgd".into()],
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let info = engine.space_config();
    assert_eq!(info.len(), 3);

    let info_map: BTreeMap<String, _> = info.into_iter().collect();
    assert_eq!(info_map["lr"].param_type, "real");
    assert_eq!(info_map["lr"].scale, "log10");
    assert_eq!(info_map["layers"].param_type, "integer");
    assert_eq!(info_map["opt"].param_type, "categorical");
    assert_eq!(info_map["opt"].choices.as_ref().unwrap().len(), 2);
}

// ==========================================================================
// Strategy types
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_strategy_types() {
    // Random
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: Some(StrategyConfig {
            strategy_type: "random".to_string(),
            refit_interval: 20,
            total_budget: None,
            exploration_budget: None,
            ongoing_exploration_period: None,
            seed: None,
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };
    let engine = HolaEngine::from_config(config).unwrap();
    assert!(engine.space().contains(&engine.ask().await.unwrap().params));

    // GMM
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: Some(StrategyConfig {
            strategy_type: "gmm".to_string(),
            refit_interval: 20,
            total_budget: None,
            exploration_budget: None,
            ongoing_exploration_period: None,
            seed: None,
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };
    let engine = HolaEngine::from_config(config).unwrap();
    assert!(engine.space().contains(&engine.ask().await.unwrap().params));

    // Default (Sobol)
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };
    let engine = HolaEngine::from_config(config).unwrap();
    assert!(engine.space().contains(&engine.ask().await.unwrap().params));
}

// ==========================================================================
// Scalarization
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_scalarize_missing_field_infinity() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let t = engine.ask().await.unwrap();
    engine
        .tell(t.trial_id, json!({"accuracy": 0.9}))
        .await
        .unwrap();

    // Missing field → infeasible trial → no feasible best
    assert!(engine.top_k(1, false).await.is_empty());
}

#[tokio::test]
async fn test_dyn_engine_scalarize_maximize() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "accuracy".to_string(),
            obj_type: "maximize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: None,
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let t1 = engine.ask().await.unwrap();
    engine
        .tell(t1.trial_id, json!({"accuracy": 0.9}))
        .await
        .unwrap();
    let t2 = engine.ask().await.unwrap();
    engine
        .tell(t2.trial_id, json!({"accuracy": 0.5}))
        .await
        .unwrap();

    let best = engine.top_k(1, false).await.into_iter().next().unwrap();
    // The score for "maximize" direction should be negated internally
    let score = best
        .scores
        .get("accuracy")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!(score < 0.0, "Maximized field should be negated");
}

// ==========================================================================
// TLP objectives
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_tlp_objectives() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: Some(0.0),
            limit: Some(1.0),
            priority: 1.0,
            group: None,
        }],
        strategy: None,
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let t1 = engine.ask().await.unwrap();
    engine
        .tell(t1.trial_id, json!({"loss": 0.5}))
        .await
        .unwrap();
    let t2 = engine.ask().await.unwrap();
    engine
        .tell(t2.trial_id, json!({"loss": 2.0}))
        .await
        .unwrap();

    // Two trials told, but one is infeasible (loss >= limit=1.5)
    assert_eq!(engine.trial_count().await, 2);
    let top = engine.top_k(1, false).await;
    assert!(!top.is_empty());
}

// ==========================================================================
// Objectives
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_update_objectives() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let t = engine.ask().await.unwrap();
    engine
        .tell(t.trial_id, json!({"loss": 0.5, "accuracy": 0.9}))
        .await
        .unwrap();

    engine
        .update_objectives(vec![ObjectiveConfig {
            field: "accuracy".to_string(),
            obj_type: "maximize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }])
        .await
        .unwrap();

    assert!(!engine.top_k(1, false).await.is_empty());
}

#[tokio::test]
async fn test_dyn_engine_update_objectives_rejects_invalid_config() {
    let engine = HolaEngine::from_config(valid_config_for_validation()).unwrap();
    let before = engine.objectives().await;

    let err = engine
        .update_objectives(vec![ObjectiveConfig {
            field: "accuracy".to_string(),
            obj_type: "larger_is_better".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }])
        .await
        .unwrap_err();

    assert!(err.contains("Objective 'accuracy'"));
    assert_eq!(engine.objectives().await[0].field, before[0].field);
}

#[tokio::test]
async fn test_dyn_engine_objectives_accessor() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![
            ObjectiveConfig {
                field: "loss".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: None,
            },
            ObjectiveConfig {
                field: "acc".to_string(),
                obj_type: "maximize".to_string(),
                target: None,
                limit: None,
                priority: 0.5,
                group: None,
            },
        ],
        strategy: None,
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let objs = engine.objectives().await;
    assert_eq!(objs.len(), 2);
    assert_eq!(objs[0].field, "loss");
    assert_eq!(objs[1].field, "acc");
}

#[tokio::test]
async fn test_dyn_engine_update_objectives_rescalarizes() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();

    let metrics = vec![
        json!({"loss": 0.1, "accuracy": 0.3}),
        json!({"loss": 0.5, "accuracy": 0.9}),
        json!({"loss": 0.3, "accuracy": 0.5}),
        json!({"loss": 0.8, "accuracy": 0.95}),
        json!({"loss": 0.2, "accuracy": 0.4}),
    ];
    for m in metrics {
        let t = engine.ask().await.unwrap();
        engine.tell(t.trial_id, m).await.unwrap();
    }

    let best_before = engine.top_k(1, false).await.into_iter().next().unwrap();
    assert_eq!(best_before.trial_id, 0);

    engine
        .update_objectives(vec![ObjectiveConfig {
            field: "accuracy".to_string(),
            obj_type: "maximize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }])
        .await
        .unwrap();

    let best_after = engine.top_k(1, false).await.into_iter().next().unwrap();
    assert_ne!(best_before.trial_id, best_after.trial_id);
}

#[tokio::test]
async fn test_dyn_engine_update_objectives_migrates_scalar_to_vector() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "f1".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: None,
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    for metrics in [
        json!({"f1": 1.0, "f2": 5.0}),
        json!({"f1": 5.0, "f2": 1.0}),
        json!({"f1": 3.0, "f2": 3.0}),
        json!({"f1": 4.0, "f2": 4.0}),
    ] {
        let trial = engine.ask().await.unwrap();
        engine.tell(trial.trial_id, metrics).await.unwrap();
    }
    assert!(engine.pareto_front(0, false).await.is_empty());

    engine
        .update_objectives(vec![
            ObjectiveConfig {
                field: "f1".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: None,
            },
            ObjectiveConfig {
                field: "f2".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: None,
            },
        ])
        .await
        .unwrap();

    let mut front_ids: Vec<u64> = engine
        .pareto_front(0, false)
        .await
        .into_iter()
        .map(|trial| trial.trial_id)
        .collect();
    front_ids.sort_unstable();
    assert_eq!(front_ids, vec![0, 1, 2]);

    let migrated = engine
        .trials("index", true)
        .await
        .into_iter()
        .find(|trial| trial.trial_id == 0)
        .unwrap();
    assert!(migrated.score_vector.get("f1").is_some());
    assert!(migrated.score_vector.get("f2").is_some());
}

#[tokio::test]
async fn test_dyn_engine_update_objectives_migrates_vector_to_scalar() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![
            ObjectiveConfig {
                field: "f1".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: None,
            },
            ObjectiveConfig {
                field: "f2".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: None,
            },
        ],
        strategy: None,
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    for metrics in [
        json!({"f1": 10.0, "f2": 0.0}),
        json!({"f1": 1.0, "f2": 10.0}),
        json!({"f1": 5.0, "f2": 5.0}),
    ] {
        let trial = engine.ask().await.unwrap();
        engine.tell(trial.trial_id, metrics).await.unwrap();
    }
    assert!(!engine.pareto_front(0, false).await.is_empty());

    engine
        .update_objectives(vec![ObjectiveConfig {
            field: "f1".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }])
        .await
        .unwrap();

    assert!(engine.pareto_front(0, false).await.is_empty());
    let best = engine.top_k(1, false).await.into_iter().next().unwrap();
    assert_eq!(best.trial_id, 1);
    assert_eq!(best.score_vector.as_object().unwrap().len(), 1);
}

// ==========================================================================
// Rescalarize
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_rescalarize() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let t = engine.ask().await.unwrap();
    engine
        .tell(t.trial_id, json!({"loss": 0.5, "acc": 0.9}))
        .await
        .unwrap();

    engine.rescalarize().await;
    assert_eq!(engine.trial_count().await, 1);
}

// ==========================================================================
// GMM with refit
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_gmm_with_refit() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: Some(StrategyConfig {
            strategy_type: "gmm".to_string(),
            refit_interval: 5,
            total_budget: None,
            exploration_budget: None,
            ongoing_exploration_period: None,
            seed: None,
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    for i in 0..30 {
        let t = engine.ask().await.unwrap();
        engine
            .tell(t.trial_id, json!({"loss": (i as f64) * 0.03}))
            .await
            .unwrap();
    }

    assert_eq!(engine.trial_count().await, 30);
    assert!(!engine.top_k(1, false).await.is_empty());
}

#[tokio::test]
async fn test_refit_excludes_infeasible_scalar() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: Some(0.0),
            limit: Some(1.0),
            priority: 1.0,
            group: None,
        }],
        strategy: Some(StrategyConfig {
            strategy_type: "gmm".to_string(),
            refit_interval: 1,
            total_budget: None,
            exploration_budget: None,
            ongoing_exploration_period: None,
            seed: None,
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();

    for i in 0..25 {
        let t = engine.ask().await.unwrap();
        let loss_val = if i % 5 == 4 { 2.0 } else { (i as f64) * 0.03 };
        engine
            .tell(t.trial_id, json!({"loss": loss_val}))
            .await
            .unwrap();
    }

    // 25 trials: 5 infeasible (loss=2.0), 20 feasible
    assert_eq!(engine.trial_count().await, 25);
    assert!(!engine.top_k(1, false).await.is_empty());

    let t = engine.ask().await.unwrap();
    assert!(engine.space().contains(&t.params));
}

#[tokio::test]
async fn test_update_objectives_triggers_refit() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: Some(StrategyConfig {
            strategy_type: "gmm".to_string(),
            refit_interval: 10,
            total_budget: None,
            exploration_budget: None,
            ongoing_exploration_period: None,
            seed: None,
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();

    for i in 0..30 {
        let t = engine.ask().await.unwrap();
        let x = (i as f64) / 29.0;
        engine
            .tell(t.trial_id, json!({"loss": x, "accuracy": x}))
            .await
            .unwrap();
    }

    let best_before = engine.top_k(1, false).await.into_iter().next().unwrap();

    engine
        .update_objectives(vec![ObjectiveConfig {
            field: "accuracy".to_string(),
            obj_type: "maximize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }])
        .await
        .unwrap();

    let best_after = engine.top_k(1, false).await.into_iter().next().unwrap();
    assert!(best_after.rank < best_before.rank || best_after.trial_id != best_before.trial_id);

    let t = engine.ask().await.unwrap();
    assert!(engine.space().contains(&t.params));
}

// ==========================================================================
// Concurrent refit serialization
// ==========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_dyn_engine_concurrent_refit_serialization() {
    // Guards against regressions in the refit_lock serialization in
    // HolaEngine: concurrent tell()s crossing the refit threshold (which take
    // refit_lock with try_lock-skip) must not deadlock or corrupt state when
    // racing a concurrent update_objectives() (which takes refit_lock with
    // lock().await so its model is built against the new objectives and is not
    // clobbered by an in-flight periodic refit).
    //
    // Exact one-at-a-time serialization is asserted via final-state consistency
    // rather than timing: the test completing at all is the deadlock guard, the
    // recorded trial_count must match every tell() that ran, and after the
    // concurrent update_objectives the engine.objectives() must reflect the new
    // objectives (a clobber/skip would leave the stale objectives in place).
    //
    // If update_objectives used a try_lock-skip instead of lock().await, this
    // test would either hang (no progress under contention) or surface stale
    // objectives rather than the swapped-in ones.
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        // auto/gmm refit so tell()s past the threshold trigger real refits.
        strategy: Some(StrategyConfig {
            strategy_type: "auto".to_string(),
            refit_interval: 5,
            total_budget: None,
            exploration_budget: Some(4),
            ongoing_exploration_period: None,
            seed: Some(7),
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();

    // Stop immediately before the first periodic refit threshold so the
    // barrier-released workload must race a real refit with objective migration.
    for i in 0..4 {
        let trial = engine.ask().await.unwrap();
        engine
            .tell(
                trial.trial_id,
                json!({"loss": i as f64 / 100.0, "accuracy": 1.0 - i as f64 / 100.0}),
            )
            .await
            .unwrap();
    }

    // Bounded, deterministic workload: a few writer tasks each driving a mix of
    // ask()/tell() across the refit threshold, plus one update_objectives().
    const WRITERS: usize = 4;
    const TELLS_PER_WRITER: usize = 10;

    let start = std::sync::Arc::new(tokio::sync::Barrier::new(WRITERS + 2));
    let mut handles = Vec::new();
    for w in 0..WRITERS {
        let engine = engine.clone();
        let start = std::sync::Arc::clone(&start);
        handles.push(tokio::spawn(async move {
            start.wait().await;
            for i in 0..TELLS_PER_WRITER {
                let t = engine.ask().await.unwrap();
                let v = ((w * TELLS_PER_WRITER + i) as f64) / 100.0;
                engine
                    .tell(t.trial_id, json!({"loss": v, "accuracy": 1.0 - v}))
                    .await
                    .unwrap();
            }
        }));
    }

    // Concurrent objectives swap racing the refit-triggering tell()s.
    let updater = {
        let engine = engine.clone();
        let start = std::sync::Arc::clone(&start);
        tokio::spawn(async move {
            start.wait().await;
            engine
                .update_objectives(vec![ObjectiveConfig {
                    field: "accuracy".to_string(),
                    obj_type: "maximize".to_string(),
                    target: None,
                    limit: None,
                    priority: 1.0,
                    group: None,
                }])
                .await
                .unwrap();
        })
    };

    start.wait().await;

    for h in handles {
        h.await.unwrap();
    }
    updater.await.unwrap();

    // Final-state consistency: every tell() was recorded exactly once.
    assert_eq!(engine.trial_count().await, 4 + WRITERS * TELLS_PER_WRITER);

    // The concurrent update_objectives must win the final state: a clobber or a
    // skipped/serialization-lost update would leave the stale "loss" objective.
    let objectives = engine.objectives().await;
    assert_eq!(objectives.len(), 1);
    assert_eq!(objectives[0].field, "accuracy");
    assert_eq!(objectives[0].obj_type, "maximize");

    // Engine remains usable after the concurrent storm.
    let t = engine.ask().await.unwrap();
    assert!(engine.space().contains(&t.params));
}

// ==========================================================================
// Checkpoints
// ==========================================================================

fn scalar_checkpoint_config(max_trials: Option<usize>) -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_trials,
        max_leaderboard_size: None,
    }
}

fn vector_checkpoint_config() -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![
            ObjectiveConfig {
                field: "f1".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: None,
            },
            ObjectiveConfig {
                field: "f2".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 2.0,
                group: None,
            },
        ],
        strategy: None,
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    }
}

#[tokio::test]
async fn test_dyn_engine_leaderboard_checkpoint() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config.clone()).unwrap();
    let t0 = engine.ask().await.unwrap();
    engine
        .tell(t0.trial_id, json!({"loss": 0.5}))
        .await
        .unwrap();
    let t1 = engine.ask().await.unwrap();
    engine
        .tell(t1.trial_id, json!({"loss": 0.3}))
        .await
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lb.json");
    engine
        .save_leaderboard_checkpoint_to(&path, Some("2 trials"))
        .await
        .unwrap();

    let engine2 = HolaEngine::from_config(config).unwrap();
    engine2.load_leaderboard_checkpoint(&path).await.unwrap();
    assert_eq!(engine2.trial_count().await, 2);
}

#[tokio::test]
async fn test_dyn_engine_leaderboard_checkpoint_resume_uses_fresh_trial_id() {
    let config = scalar_checkpoint_config(Some(3));
    let engine = HolaEngine::from_config(config.clone()).unwrap();

    for (expected_id, loss) in [0.5, 0.3].into_iter().enumerate() {
        let trial = engine.ask().await.unwrap();
        assert_eq!(trial.trial_id, expected_id as u64);
        let completed = engine
            .tell(trial.trial_id, json!({"loss": loss}))
            .await
            .unwrap();
        assert_eq!(completed.trial_id, expected_id as u64);
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lb.json");
    engine
        .save_leaderboard_checkpoint_to(&path, Some("2 trials"))
        .await
        .unwrap();

    let restored = HolaEngine::from_config(config).unwrap();
    let stale_pending = restored.ask().await.unwrap();
    let stale_cancelled = restored.ask().await.unwrap();
    restored.cancel(stale_cancelled.trial_id).await.unwrap();

    restored.load_leaderboard_checkpoint(&path).await.unwrap();
    assert!(
        restored
            .tell(stale_pending.trial_id, json!({"loss": 0.0}))
            .await
            .is_err(),
        "pending trials from the pre-load engine state must not survive checkpoint load"
    );

    let trial = restored.ask().await.unwrap();
    assert!(
        trial.trial_id >= (1_u64 << 62),
        "leaderboard-only restore must allocate from a fresh high ID epoch"
    );
    let resumed_id = trial.trial_id;
    let completed = restored
        .tell(trial.trial_id, json!({"loss": 0.1}))
        .await
        .unwrap();
    assert_eq!(completed.trial_id, resumed_id);

    let ids: Vec<u64> = restored
        .trials("index", true)
        .await
        .into_iter()
        .map(|trial| trial.trial_id)
        .collect();
    assert_eq!(ids, vec![0, 1, resumed_id]);
}

#[tokio::test]
async fn test_dyn_engine_full_checkpoint() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config.clone()).unwrap();
    let t = engine.ask().await.unwrap();
    engine.tell(t.trial_id, json!({"loss": 0.5})).await.unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full.json");
    engine
        .save_full_checkpoint(&path, Some("full checkpoint test"))
        .await
        .unwrap();

    let engine2 = HolaEngine::from_config(config).unwrap();
    engine2.load_full_checkpoint(&path).await.unwrap();
    assert_eq!(engine2.trial_count().await, 1);
}

#[tokio::test]
async fn test_dyn_engine_full_checkpoint_resume_returns_new_completed_trial() {
    let config = scalar_checkpoint_config(None);
    let engine = HolaEngine::from_config(config.clone()).unwrap();

    for loss in [0.5, 0.3] {
        let trial = engine.ask().await.unwrap();
        engine
            .tell(trial.trial_id, json!({"loss": loss}))
            .await
            .unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full.json");
    engine
        .save_full_checkpoint(&path, Some("2 trials"))
        .await
        .unwrap();

    let restored = HolaEngine::from_config(config).unwrap();
    restored.load_full_checkpoint(&path).await.unwrap();

    let trial = restored.ask().await.unwrap();
    assert_eq!(trial.trial_id, 2);
    let completed = restored
        .tell(trial.trial_id, json!({"loss": 0.1}))
        .await
        .unwrap();
    assert_eq!(completed.trial_id, 2);
    assert_eq!(completed.params, trial.params);

    let ids: Vec<u64> = restored
        .trials("index", true)
        .await
        .into_iter()
        .map(|trial| trial.trial_id)
        .collect();
    assert_eq!(ids, vec![0, 1, 2]);
}

#[tokio::test]
async fn test_full_checkpoint_preserves_pending_ask_idempotency() {
    let engine = HolaEngine::from_config(scalar_checkpoint_config(Some(41))).unwrap();
    let first = engine
        .ask_idempotent_with_lease("worker-3-request-8", std::time::Duration::from_secs(60))
        .await
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idempotent-ask.json");
    engine.save_full_checkpoint(&path, None).await.unwrap();

    let restored = HolaEngine::load_from_checkpoint(&path).await.unwrap();
    let replay = restored
        .ask_idempotent_with_lease("worker-3-request-8", std::time::Duration::from_secs(60))
        .await
        .unwrap();
    let next = restored
        .ask_idempotent_with_lease("worker-3-request-9", std::time::Duration::from_secs(60))
        .await
        .unwrap();

    assert_eq!(replay, first);
    assert_eq!(next.trial_id, first.trial_id + 1);
}

#[tokio::test]
async fn test_full_version_one_checkpoint_migrates_without_runtime_state() {
    let mut config = scalar_checkpoint_config(Some(10));
    config.strategy = Some(StrategyConfig {
        strategy_type: "sobol".to_string(),
        refit_interval: 20,
        total_budget: None,
        exploration_budget: None,
        ongoing_exploration_period: None,
        seed: Some(17),
        elite_fraction: None,
        max_components: None,
        min_elite_samples: None,
        max_refit_samples: 4096,
        max_refit_candidates: 16_384,
    });
    let engine = HolaEngine::from_config(config).unwrap();
    let first = engine.ask().await.unwrap();
    engine
        .tell(first.trial_id, json!({"loss": 0.25}))
        .await
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy-v1-full.json");
    engine.save_full_checkpoint(&path, None).await.unwrap();
    let mut document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    document["checkpoint"]["metadata"]["format_version"] = json!(1);
    document.as_object_mut().unwrap().remove("runtime_state");
    std::fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

    let restored = HolaEngine::load_from_checkpoint(&path).await.unwrap();
    assert_eq!(restored.trial_count().await, 1);
    let next = restored.ask().await.unwrap();
    assert_eq!(next.trial_id, 1);
    let completed = restored
        .tell(next.trial_id, json!({"loss": 0.1}))
        .await
        .unwrap();
    assert_eq!(completed.trial_id, 1);
}

#[tokio::test]
async fn test_live_ask_idempotency_keys_are_not_evicted_at_the_old_capacity() {
    let engine = HolaEngine::from_config(scalar_checkpoint_config(None)).unwrap();
    let first = engine
        .ask_idempotent("worker-key-0")
        .await
        .expect("first keyed ask should succeed");

    // The old key ledger retained only 4096 entries while pending work allowed
    // 10,000. Crossing that boundary evicted worker-key-0 even though its trial
    // was still pending, and replaying it allocated duplicate work.
    for index in 1..=4096 {
        engine
            .ask_idempotent(&format!("worker-key-{index}"))
            .await
            .expect("keyed ask within pending bound should succeed");
    }

    let replay = engine
        .ask_idempotent("worker-key-0")
        .await
        .expect("oldest live key should still replay");
    assert_eq!(replay, first);
    assert_eq!(engine.pending_count().await, 4097);
}

#[tokio::test]
async fn test_completion_receipt_survives_leaderboard_eviction_and_checkpoint_restart() {
    let mut config = scalar_checkpoint_config(None);
    config.max_leaderboard_size = Some(1);
    let engine = HolaEngine::from_config(config).unwrap();

    let first_trial = engine.ask().await.unwrap();
    let first = engine
        .tell_with_outcome(first_trial.trial_id, json!({"loss": 0.4}))
        .await
        .unwrap();
    assert!(first.newly_committed);

    let second_trial = engine.ask().await.unwrap();
    engine
        .tell(second_trial.trial_id, json!({"loss": 0.2}))
        .await
        .unwrap();
    assert_eq!(engine.trials("index", true).await.len(), 1);
    assert!(
        engine
            .trials("index", true)
            .await
            .iter()
            .all(|trial| trial.trial_id != first_trial.trial_id),
        "the first trial must actually be evicted for this regression"
    );

    let retry = engine
        .tell_with_outcome(first_trial.trial_id, json!({"loss": 0.4}))
        .await
        .expect("an exact retry must replay its completion receipt");
    assert!(!retry.newly_committed);
    assert_eq!(retry.trial_count, first.trial_count);
    assert_eq!(
        serde_json::to_value(&retry.completed).unwrap(),
        serde_json::to_value(&first.completed).unwrap()
    );
    assert!(
        engine
            .tell(first_trial.trial_id, json!({"loss": 9.0}))
            .await
            .unwrap_err()
            .contains("different metrics")
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completion-receipt.json");
    engine.save_full_checkpoint(&path, None).await.unwrap();
    let restored = HolaEngine::load_from_checkpoint(&path).await.unwrap();
    let retry_after_restart = restored
        .tell_with_outcome(first_trial.trial_id, json!({"loss": 0.4}))
        .await
        .expect("completion receipt must survive a checkpointed restart");
    assert!(!retry_after_restart.newly_committed);
    assert_eq!(
        serde_json::to_value(retry_after_restart.completed).unwrap(),
        serde_json::to_value(first.completed).unwrap()
    );
}

#[tokio::test]
async fn test_dyn_engine_full_checkpoint_resume_preserves_vector_trial_ids() {
    let config = vector_checkpoint_config();
    let engine = HolaEngine::from_config(config.clone()).unwrap();

    for metrics in [json!({"f1": 1.0, "f2": 3.0}), json!({"f1": 2.0, "f2": 1.0})] {
        let trial = engine.ask().await.unwrap();
        let completed = engine.tell(trial.trial_id, metrics).await.unwrap();
        assert_eq!(completed.trial_id, trial.trial_id);
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vector-full.json");
    engine
        .save_full_checkpoint(&path, Some("vector checkpoint"))
        .await
        .unwrap();

    let restored = HolaEngine::from_config(config).unwrap();
    restored.load_full_checkpoint(&path).await.unwrap();

    let trial = restored.ask().await.unwrap();
    assert_eq!(trial.trial_id, 2);
    let completed = restored
        .tell(trial.trial_id, json!({"f1": 0.5, "f2": 2.5}))
        .await
        .unwrap();
    assert_eq!(completed.trial_id, 2);

    let ids: Vec<u64> = restored
        .trials("index", true)
        .await
        .into_iter()
        .map(|trial| trial.trial_id)
        .collect();
    assert_eq!(ids, vec![0, 1, 2]);
}

#[tokio::test]
async fn test_configured_checkpoint_load_supports_full_and_leaderboard() {
    let config = scalar_checkpoint_config(None);
    let engine = HolaEngine::from_config(config.clone()).unwrap();
    let trial = engine.ask().await.unwrap();
    engine
        .tell(trial.trial_id, json!({"loss": 0.5}))
        .await
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let full_path = dir.path().join("full.json");
    engine
        .save_full_checkpoint(&full_path, Some("full"))
        .await
        .unwrap();

    let (restored_full, kind) = HolaEngine::load_configured_checkpoint(config.clone(), &full_path)
        .await
        .unwrap();
    assert_eq!(kind, CheckpointLoadKind::Full);
    assert_eq!(restored_full.trial_count().await, 1);
    assert_eq!(restored_full.ask().await.unwrap().trial_id, 1);

    let strict_full = HolaEngine::from_config(config.clone()).unwrap();
    let strict_kind = strict_full
        .load_checkpoint_with_fallback(&full_path)
        .await
        .unwrap();
    assert_eq!(strict_kind, CheckpointLoadKind::Full);
    assert_eq!(strict_full.trial_count().await, 1);

    let leaderboard_path = dir.path().join("leaderboard.json");
    engine
        .save_leaderboard_checkpoint_to(&leaderboard_path, Some("leaderboard"))
        .await
        .unwrap();

    let (restored_leaderboard, kind) =
        HolaEngine::load_configured_checkpoint(config, &leaderboard_path)
            .await
            .unwrap();
    assert_eq!(kind, CheckpointLoadKind::Leaderboard);
    assert_eq!(restored_leaderboard.trial_count().await, 1);
    assert!(restored_leaderboard.ask().await.unwrap().trial_id >= (1_u64 << 62));

    let strict_leaderboard = HolaEngine::from_config(scalar_checkpoint_config(None)).unwrap();
    let strict_kind = strict_leaderboard
        .load_checkpoint_with_fallback(&leaderboard_path)
        .await
        .unwrap();
    assert_eq!(strict_kind, CheckpointLoadKind::Leaderboard);
    assert_eq!(strict_leaderboard.trial_count().await, 1);
}

#[tokio::test]
async fn leaderboard_reimport_uses_completed_count_not_sparse_trial_ids_as_sampler_cursor() {
    for strategy_type in ["sobol", "auto"] {
        let mut config = scalar_checkpoint_config(None);
        config.strategy = Some(StrategyConfig {
            strategy_type: strategy_type.to_string(),
            refit_interval: 20,
            total_budget: Some(100),
            exploration_budget: Some(10),
            ongoing_exploration_period: None,
            seed: Some(73),
            elite_fraction: Some(0.5),
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        });

        // Establish the deterministic fifth sample for this strategy after four
        // ordinary completed trials.
        let baseline = HolaEngine::from_config(config.clone()).unwrap();
        for loss in [0.4, 0.3, 0.2, 0.1] {
            let trial = baseline.ask().await.unwrap();
            baseline
                .tell(trial.trial_id, json!({"loss": loss}))
                .await
                .unwrap();
        }
        let expected_fifth = baseline.ask().await.unwrap().params;

        let source = HolaEngine::from_config(config.clone()).unwrap();
        for loss in [0.4, 0.3, 0.2] {
            let trial = source.ask().await.unwrap();
            source
                .tell(trial.trial_id, json!({"loss": loss}))
                .await
                .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join(format!("{strategy_type}-first.json"));
        source
            .save_leaderboard_checkpoint_to(&first_path, None)
            .await
            .unwrap();

        // This restore deliberately assigns the fourth trial a sparse high ID.
        let first_restore = HolaEngine::from_config(config.clone()).unwrap();
        first_restore
            .load_leaderboard_checkpoint(&first_path)
            .await
            .unwrap();
        let fourth = first_restore.ask().await.unwrap();
        assert!(fourth.trial_id >= (1_u64 << 62));
        first_restore
            .tell(fourth.trial_id, json!({"loss": 0.1}))
            .await
            .unwrap();

        let second_path = dir.path().join(format!("{strategy_type}-second.json"));
        first_restore
            .save_leaderboard_checkpoint_to(&second_path, None)
            .await
            .unwrap();

        // Reimport must advance from total_completed=4, not saturate/wrap a
        // Sobol cursor derived from the sparse high trial ID.
        let second_restore = HolaEngine::from_config(config).unwrap();
        second_restore
            .load_leaderboard_checkpoint(&second_path)
            .await
            .unwrap();
        let actual_fifth = second_restore.ask().await.unwrap();
        assert_eq!(
            actual_fifth.params, expected_fifth,
            "{strategy_type} sampler cursor changed after sparse-ID reimport"
        );
    }
}

// ==========================================================================
// Auto strategy (Sobol -> GMM switching)
// ==========================================================================

fn auto_strategy_test_config(exploration_budget: usize, seed: u64) -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: Some(StrategyConfig {
            strategy_type: "auto".to_string(),
            refit_interval: 5,
            total_budget: None,
            exploration_budget: Some(exploration_budget),
            ongoing_exploration_period: None,
            seed: Some(seed),
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    }
}

fn sobol_strategy_test_config(seed: u64) -> StudyConfig {
    let mut config = auto_strategy_test_config(0, seed);
    config.strategy = Some(StrategyConfig {
        strategy_type: "sobol".to_string(),
        refit_interval: 20,
        total_budget: None,
        exploration_budget: None,
        ongoing_exploration_period: None,
        seed: Some(seed),
        elite_fraction: None,
        max_components: None,
        min_elite_samples: None,
        max_refit_samples: 4096,
        max_refit_candidates: 16_384,
    });
    config
}

#[tokio::test]
async fn test_auto_strategy_default() {
    // With no strategy config, should use "auto" and work correctly
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: None, // should default to "auto"
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();

    // Run enough trials to cross the exploration threshold and trigger refit
    for i in 0..60 {
        let t = engine.ask().await.unwrap();
        assert!(engine.space().contains(&t.params));
        let loss = (i as f64) / 59.0;
        engine
            .tell(t.trial_id, json!({"loss": loss}))
            .await
            .unwrap();
    }

    assert_eq!(engine.trial_count().await, 60);
    let top = engine.top_k(1, false).await;
    assert!(!top.is_empty());
}

#[tokio::test]
async fn test_auto_strategy_with_explicit_exploration_budget() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: Some(StrategyConfig {
            strategy_type: "auto".to_string(),
            refit_interval: 5,
            total_budget: None,
            exploration_budget: Some(10), // switch to GMM after 10 trials
            ongoing_exploration_period: None,
            seed: None,
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();

    for i in 0..30 {
        let t = engine.ask().await.unwrap();
        assert!(engine.space().contains(&t.params));
        let loss = (i as f64) / 29.0;
        engine
            .tell(t.trial_id, json!({"loss": loss}))
            .await
            .unwrap();
    }

    assert_eq!(engine.trial_count().await, 30);
}

#[test]
fn test_auto_strategy_default_exploration_budget() {
    use hola::hola_engine::AutoStrategy;

    // 2 * min(40, 56) = 80 -> round down to 64
    assert_eq!(AutoStrategy::default_exploration_budget(200, 3), 64);

    // 2 * min(20, 52) = 40 -> round down to 32
    assert_eq!(AutoStrategy::default_exploration_budget(100, 1), 32);

    // 2 * min(200, 60) = 120 -> round down to 64
    assert_eq!(AutoStrategy::default_exploration_budget(1000, 5), 64);

    // 2 * min(10, 70) = 20 -> round down to 16
    assert_eq!(AutoStrategy::default_exploration_budget(50, 10), 16);

    // Edge cases
    assert_eq!(AutoStrategy::default_exploration_budget(10, 1), 4);
    assert_eq!(AutoStrategy::default_exploration_budget(5, 1), 2);
}

#[tokio::test]
async fn test_auto_strategy_counts_pending_asks_against_exploration_budget() {
    let auto = HolaEngine::from_config(auto_strategy_test_config(2, 17)).unwrap();
    let sobol = HolaEngine::from_config(sobol_strategy_test_config(17)).unwrap();

    let auto_trials = [
        auto.ask().await.unwrap(),
        auto.ask().await.unwrap(),
        auto.ask().await.unwrap(),
        auto.ask().await.unwrap(),
    ];

    for auto_trial in auto_trials {
        assert_eq!(auto_trial.params, sobol.ask().await.unwrap().params);
    }
    assert_eq!(auto.trial_count().await, 0);
}

#[tokio::test]
async fn test_auto_strategy_full_checkpoint_preserves_pending_ask_accounting() {
    let engine = HolaEngine::from_config(auto_strategy_test_config(2, 17)).unwrap();
    for _ in 0..3 {
        engine.ask().await.unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("auto-full.json");
    engine
        .save_full_checkpoint(&path, Some("auto pending asks"))
        .await
        .unwrap();

    let restored = HolaEngine::load_from_checkpoint(&path).await.unwrap();

    let sobol = HolaEngine::from_config(sobol_strategy_test_config(17)).unwrap();
    for _ in 0..3 {
        sobol.ask().await.unwrap();
    }
    let expected = sobol.ask().await.unwrap();
    let resumed = restored.ask().await.unwrap();

    assert_eq!(resumed.params, expected.params);
}

// ==========================================================================
// Seed determinism tests
// ==========================================================================

#[tokio::test]
async fn test_seed_determinism_sobol() {
    let make_engine = |seed| {
        HolaEngine::from_config(StudyConfig {
            space: BTreeMap::from([(
                "x".to_string(),
                ParamConfig::Real {
                    min: 0.0,
                    max: 1.0,
                    scale: "linear".to_string(),
                },
            )]),
            objectives: vec![ObjectiveConfig {
                field: "loss".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: None,
            }],
            strategy: Some(StrategyConfig {
                strategy_type: "sobol".to_string(),
                refit_interval: 20,
                total_budget: None,
                exploration_budget: None,
                ongoing_exploration_period: None,
                seed: Some(seed),
                elite_fraction: None,
                max_components: None,
                min_elite_samples: None,
                max_refit_samples: 4096,
                max_refit_candidates: 16_384,
            }),
            checkpoint: None,
            max_trials: None,
            max_leaderboard_size: None,
        })
        .unwrap()
    };

    let e1 = make_engine(123);
    let e2 = make_engine(123);

    for _ in 0..10 {
        let t1 = e1.ask().await.unwrap();
        let t2 = e2.ask().await.unwrap();
        assert_eq!(
            t1.params, t2.params,
            "Same seed should produce same candidates"
        );
    }
}

#[tokio::test]
async fn test_seed_determinism_random() {
    let make_engine = |seed| {
        HolaEngine::from_config(StudyConfig {
            space: BTreeMap::from([(
                "x".to_string(),
                ParamConfig::Real {
                    min: 0.0,
                    max: 1.0,
                    scale: "linear".to_string(),
                },
            )]),
            objectives: vec![ObjectiveConfig {
                field: "loss".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: None,
            }],
            strategy: Some(StrategyConfig {
                strategy_type: "random".to_string(),
                refit_interval: 20,
                total_budget: None,
                exploration_budget: None,
                ongoing_exploration_period: None,
                seed: Some(seed),
                elite_fraction: None,
                max_components: None,
                min_elite_samples: None,
                max_refit_samples: 4096,
                max_refit_candidates: 16_384,
            }),
            checkpoint: None,
            max_trials: None,
            max_leaderboard_size: None,
        })
        .unwrap()
    };

    let e1 = make_engine(42);
    let e2 = make_engine(42);

    for _ in 0..10 {
        let t1 = e1.ask().await.unwrap();
        let t2 = e2.ask().await.unwrap();
        assert_eq!(
            t1.params, t2.params,
            "Same seed should produce same candidates"
        );
    }
}

// ==========================================================================
// Pareto front tests
// ==========================================================================

#[tokio::test]
async fn test_pareto_front_multi_objective() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![
            ObjectiveConfig {
                field: "f1".to_string(),
                obj_type: "minimize".to_string(),
                target: Some(0.0),
                limit: Some(10.0),
                priority: 1.0,
                group: None,
            },
            ObjectiveConfig {
                field: "f2".to_string(),
                obj_type: "minimize".to_string(),
                target: Some(0.0),
                limit: Some(10.0),
                priority: 2.0,
                group: None,
            },
        ],
        strategy: Some(StrategyConfig {
            strategy_type: "random".to_string(),
            refit_interval: 20,
            total_budget: None,
            exploration_budget: None,
            ongoing_exploration_period: None,
            seed: Some(0),
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();

    // Tell trials with known Pareto structure:
    // (1,5) and (5,1) are non-dominated; (3,3) dominated by neither but (4,4) dominated by (3,3)
    let trials_data = vec![
        json!({"f1": 1.0, "f2": 5.0}), // Pareto-optimal
        json!({"f1": 5.0, "f2": 1.0}), // Pareto-optimal
        json!({"f1": 3.0, "f2": 3.0}), // Pareto-optimal
        json!({"f1": 4.0, "f2": 4.0}), // Dominated by (3,3)
    ];

    for data in trials_data {
        let t = engine.ask().await.unwrap();
        engine.tell(t.trial_id, data).await.unwrap();
    }

    let front = engine.pareto_front(0, false).await;
    assert_eq!(front.len(), 3, "Should have 3 non-dominated trials");
}

#[tokio::test]
async fn test_live_vector_ranks_respect_direction_and_never_promote_infeasible_trials() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![
            ObjectiveConfig {
                field: "loss".to_string(),
                obj_type: "minimize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("loss".to_string()),
            },
            ObjectiveConfig {
                field: "accuracy".to_string(),
                obj_type: "maximize".to_string(),
                target: None,
                limit: None,
                priority: 1.0,
                group: Some("accuracy".to_string()),
            },
            ObjectiveConfig {
                field: "latency".to_string(),
                obj_type: "minimize".to_string(),
                target: Some(0.0),
                limit: Some(10.0),
                priority: 1.0,
                group: Some("constraint".to_string()),
            },
        ],
        strategy: Some(StrategyConfig {
            strategy_type: "random".to_string(),
            refit_interval: 20,
            total_budget: None,
            exploration_budget: None,
            ongoing_exploration_period: None,
            seed: Some(7),
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config.clone()).unwrap();
    let cases = [
        json!({"loss": 1.0, "accuracy": 0.8, "latency": 5.0}),
        json!({"loss": 2.0, "accuracy": 0.9, "latency": 5.0}),
        json!({"loss": 2.0, "accuracy": 0.7, "latency": 5.0}),
        // Excellent unconstrained metrics, but latency violates the limit.
        json!({"loss": 0.5, "accuracy": 0.95, "latency": 20.0}),
    ];
    let mut completed = Vec::new();
    for metrics in cases {
        let trial = engine.ask().await.unwrap();
        completed.push(engine.tell(trial.trial_id, metrics).await.unwrap());
    }

    // Maximize is oriented as a negative cost, so the first two trials trade
    // off and the third is dominated by trial 0.
    assert_eq!(completed[0].score_vector["accuracy"], json!(-0.8));
    assert_eq!(completed[1].score_vector["accuracy"], json!(-0.9));
    assert_eq!(completed[0].pareto_front, 0);
    assert_eq!(completed[1].pareto_front, 0);
    assert!(completed[2].pareto_front > 0);

    // The tell response is the exact DTO sent in the live SSE event.
    assert_eq!(completed[3].score_vector["constraint"], "inf");
    assert!(completed[3].pareto_front > 0);

    let all = engine.trials("rank", true).await;
    let infeasible = all.iter().find(|trial| trial.trial_id == 3).unwrap();
    assert!(infeasible.pareto_front > 0);
    assert!(infeasible.rank >= 3);
    let front_ids: Vec<u64> = engine
        .pareto_front(0, true)
        .await
        .into_iter()
        .map(|trial| trial.trial_id)
        .collect();
    assert_eq!(front_ids, vec![0, 1]);

    // Even an all-infeasible live study reserves front zero for feasible work.
    let all_infeasible = HolaEngine::from_config(config).unwrap();
    let trial = all_infeasible.ask().await.unwrap();
    let event_trial = all_infeasible
        .tell(
            trial.trial_id,
            json!({"loss": 0.1, "accuracy": 0.99, "latency": 100.0}),
        )
        .await
        .unwrap();
    assert!(event_trial.pareto_front > 0);
    assert!(all_infeasible.pareto_front(0, true).await.is_empty());
    assert!(
        all_infeasible
            .trials("rank", true)
            .await
            .iter()
            .all(|trial| trial.pareto_front > 0)
    );
}

#[tokio::test]
async fn test_pareto_front_scalar_study_errors() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
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
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    // Scalar studies return empty pareto front (no completed trials yet)
    assert!(engine.pareto_front(0, false).await.is_empty());
}

#[tokio::test]
async fn test_pareto_front_empty() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![
            ObjectiveConfig {
                field: "f1".to_string(),
                obj_type: "minimize".to_string(),
                target: Some(0.0),
                limit: Some(10.0),
                priority: 1.0,
                group: None,
            },
            ObjectiveConfig {
                field: "f2".to_string(),
                obj_type: "minimize".to_string(),
                target: Some(0.0),
                limit: Some(10.0),
                priority: 2.0,
                group: None,
            },
        ],
        strategy: None,
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    };

    let engine = HolaEngine::from_config(config).unwrap();
    let front = engine.pareto_front(0, false).await;
    assert!(front.is_empty());
}

// ==========================================================================
// Concurrency: interleaved ask/tell/cancel/update_objectives stress test
// ==========================================================================

/// Single-objective config with a deterministically seeded auto strategy, used
/// by the concurrency stress tests so refit/exploration paths are exercised too.
fn concurrency_test_config() -> StudyConfig {
    StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Real {
                min: 0.0,
                max: 1.0,
                scale: "linear".to_string(),
            },
        )]),
        objectives: vec![ObjectiveConfig {
            field: "loss".to_string(),
            obj_type: "minimize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
            group: None,
        }],
        strategy: Some(StrategyConfig {
            strategy_type: "auto".to_string(),
            refit_interval: 5,
            total_budget: None,
            exploration_budget: Some(8),
            ongoing_exploration_period: None,
            seed: Some(7),
            elite_fraction: None,
            max_components: None,
            min_elite_samples: None,
            max_refit_samples: 4096,
            max_refit_candidates: 16_384,
        }),
        checkpoint: None,
        max_trials: None,
        max_leaderboard_size: None,
    }
}

/// Hammer one shared `Arc<HolaEngine>` with many concurrent tasks that interleave
/// ask()/tell() plus a racing cancel() and update_objectives(). Asserts:
///   * every trial id returned by ask() is UNIQUE (no duplicate id allocation),
///   * the final trial_count() equals the number of completed (successful) tells,
///   * the whole storm finishes without deadlock/timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_hola_engine_concurrent_ask_tell_unique_ids_no_deadlock() {
    use std::collections::HashSet;
    use std::sync::Arc;

    let engine = Arc::new(HolaEngine::from_config(concurrency_test_config()).unwrap());

    const TASKS: usize = 16;
    const OPS_PER_TASK: usize = 20;

    let body = async {
        // Each ask()->tell() task returns (asked_ids, completed_tells).
        let mut handles = Vec::new();
        for t in 0..TASKS {
            let engine = Arc::clone(&engine);
            handles.push(tokio::spawn(async move {
                let mut asked_ids: Vec<u64> = Vec::new();
                let mut completed: u64 = 0;
                for i in 0..OPS_PER_TASK {
                    let trial = engine.ask().await.unwrap();
                    asked_ids.push(trial.trial_id);
                    let v = ((t * OPS_PER_TASK + i) as f64) / 1000.0;
                    // tell() can only fail here if a racing cancel() targeted this
                    // exact id; this task does not cancel its own ids, so every
                    // tell must succeed. A failure would surface as a panic.
                    engine
                        .tell(trial.trial_id, json!({"loss": v}))
                        .await
                        .unwrap();
                    completed += 1;
                }
                (asked_ids, completed)
            }));
        }

        // One task that races by asking and immediately cancelling its own ids,
        // so it contributes ids (which must still be unique) but no completed
        // tells. Its cancelled ids must never be reissued to another task.
        let canceller = {
            let engine = Arc::clone(&engine);
            tokio::spawn(async move {
                let mut asked_ids: Vec<u64> = Vec::new();
                for _ in 0..OPS_PER_TASK {
                    let trial = engine.ask().await.unwrap();
                    asked_ids.push(trial.trial_id);
                    engine.cancel(trial.trial_id).await.unwrap();
                }
                asked_ids
            })
        };

        // One task that races update_objectives() against the storm.
        let updater = {
            let engine = Arc::clone(&engine);
            tokio::spawn(async move {
                engine
                    .update_objectives(vec![ObjectiveConfig {
                        field: "loss".to_string(),
                        obj_type: "minimize".to_string(),
                        target: None,
                        limit: None,
                        priority: 2.0,
                        group: None,
                    }])
                    .await
                    .unwrap();
            })
        };

        let mut all_ids: Vec<u64> = Vec::new();
        let mut total_completed: u64 = 0;
        for h in handles {
            let (ids, completed) = h.await.unwrap();
            all_ids.extend(ids);
            total_completed += completed;
        }
        all_ids.extend(canceller.await.unwrap());
        updater.await.unwrap();

        (all_ids, total_completed)
    };

    // Generous bound: assert the whole concurrent workload completes without
    // deadlocking. A hang would otherwise surface as this timeout firing.
    let (all_ids, total_completed) =
        match tokio::time::timeout(std::time::Duration::from_secs(30), body).await {
            Ok(result) => result,
            Err(_) => panic!("concurrent ask/tell/cancel storm timed out (possible deadlock)"),
        };

    // Every id ever handed out by ask() must be unique across all tasks,
    // including the cancelled ones.
    let unique: HashSet<u64> = all_ids.iter().copied().collect();
    assert_eq!(
        unique.len(),
        all_ids.len(),
        "ask() returned duplicate trial ids: {} total vs {} unique",
        all_ids.len(),
        unique.len()
    );

    // The leaderboard must hold exactly the trials that were successfully told.
    assert_eq!(
        engine.trial_count().await,
        total_completed as usize,
        "trial_count must equal the number of completed tells"
    );

    // Engine remains usable after the storm.
    let trial = engine.ask().await.unwrap();
    assert!(engine.space().contains(&trial.params));
}

// ==========================================================================
// Concurrency: data integrity of stored observations
// ==========================================================================

/// Across a large number of concurrent ask()+tell() tasks where each task tells
/// a DISTINCT, known loss value, read back ALL stored trials afterward and
/// assert that:
///   * the multiset of recorded loss values equals exactly the set sent
///     (nothing dropped, duplicated, or corrupted),
///   * the stored trial ids are unique,
///   * the trial count matches N.
/// A larger N amplifies races between concurrent tell()s pushing to the shared
/// leaderboard.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_hola_engine_concurrent_tell_preserves_all_observations() {
    use std::collections::HashSet;
    use std::sync::Arc;

    let engine = Arc::new(HolaEngine::from_config(concurrency_test_config()).unwrap());

    const N: usize = 50;

    let body = async {
        let mut handles = Vec::new();
        for i in 0..N {
            let engine = Arc::clone(&engine);
            handles.push(tokio::spawn(async move {
                let trial = engine.ask().await.unwrap();
                // Distinct, known observation per task: loss = i.
                engine
                    .tell(trial.trial_id, json!({"loss": i as f64}))
                    .await
                    .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    };

    match tokio::time::timeout(std::time::Duration::from_secs(30), body).await {
        Ok(()) => {}
        Err(_) => panic!("concurrent tell storm timed out (possible deadlock)"),
    }

    assert_eq!(
        engine.trial_count().await,
        N,
        "all N tells must be recorded"
    );

    // Read back every stored trial and collect their loss metrics and ids.
    let trials = engine.trials("index", true).await;
    assert_eq!(trials.len(), N, "every told trial must be retrievable");

    let mut recorded_losses: Vec<u64> = Vec::with_capacity(N);
    let mut ids: Vec<u64> = Vec::with_capacity(N);
    for t in &trials {
        ids.push(t.trial_id);
        let loss = t
            .metrics
            .get("loss")
            .and_then(|v| v.as_f64())
            .expect("each stored trial must retain its loss metric");
        // Values were exact integers cast to f64; recover them losslessly.
        recorded_losses.push(loss as u64);
    }

    // Trial ids must be unique.
    let unique_ids: HashSet<u64> = ids.iter().copied().collect();
    assert_eq!(unique_ids.len(), N, "stored trial ids must be unique");

    // The multiset of recorded losses must equal exactly {0, 1, ..., N-1}: no
    // value dropped, duplicated, or corrupted under concurrency.
    recorded_losses.sort_unstable();
    let expected: Vec<u64> = (0..N as u64).collect();
    assert_eq!(
        recorded_losses, expected,
        "recorded observation multiset must equal the set of sent values"
    );
}

#[tokio::test]
async fn test_concurrent_tell_returns_each_atomic_commit_count() {
    let engine =
        std::sync::Arc::new(HolaEngine::from_config(scalar_checkpoint_config(None)).unwrap());
    let first = engine.ask().await.unwrap();
    let second = engine.ask().await.unwrap();

    let a = {
        let engine = engine.clone();
        tokio::spawn(async move {
            engine
                .tell_with_count(first.trial_id, json!({"loss": 0.4}))
                .await
                .unwrap()
                .1
        })
    };
    let b = {
        let engine = engine.clone();
        tokio::spawn(async move {
            engine
                .tell_with_count(second.trial_id, json!({"loss": 0.2}))
                .await
                .unwrap()
                .1
        })
    };
    let mut counts = vec![a.await.unwrap(), b.await.unwrap()];
    counts.sort_unstable();
    assert_eq!(counts, vec![1, 2]);
}
