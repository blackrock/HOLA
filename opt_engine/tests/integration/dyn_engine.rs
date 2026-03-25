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

//! Integration tests for DynEngine (type-erased layer).
//!
//! Exercises config parsing, ask/tell flows, strategy types, scalarization,
//! objectives, checkpoints, refit, and all parameter types.

use opt_engine::dyn_engine::{
    DynEngine, ObjectiveConfig, ParamConfig, StrategyConfig, StudyConfig,
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
        type: continuous
        min: 0.0001
        max: 0.1
        scale: log10
      num_layers:
        type: discrete
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

    let engine = DynEngine::from_config(config).unwrap();
    assert_eq!(engine.trial_count().await, 0);
}

#[tokio::test]
async fn test_dyn_engine_config_with_checkpoint() {
    let yaml = r#"
    space:
      x:
        type: continuous
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

    let engine = DynEngine::from_config(config).unwrap();
    let _t = engine.ask().await;
}

// ==========================================================================
// Ask/Tell flow
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_ask_tell_flow() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    assert_eq!(engine.trial_count().await, 0);
    assert!(engine.best().await.is_none());

    let t0 = engine.ask().await;
    assert_eq!(t0.trial_id, 0);
    let t1 = engine.ask().await;
    assert_eq!(t1.trial_id, 1);

    engine.tell(t0.trial_id, json!({"loss": 0.8})).await.unwrap();
    assert_eq!(engine.trial_count().await, 1);

    engine.tell(t1.trial_id, json!({"loss": 0.2})).await.unwrap();
    assert_eq!(engine.trial_count().await, 2);

    let best = engine.best().await.unwrap();
    assert_eq!(best.trial_id, 1);
    assert_eq!(engine.trial_count().await, 2);
}

#[tokio::test]
async fn test_dyn_engine_unknown_trial_error() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    let result = engine.tell(999, json!({"loss": 0.5})).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("999"));
}

#[tokio::test]
async fn test_dyn_engine_double_tell_error() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    let t = engine.ask().await;
    engine.tell(t.trial_id, json!({"loss": 0.5})).await.unwrap();
    assert!(engine.tell(t.trial_id, json!({"loss": 0.3})).await.is_err());
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
                ParamConfig::Continuous {
                    min: 1e-4,
                    max: 0.1,
                    scale: "log10".to_string(),
                },
            ),
            (
                "layers".to_string(),
                ParamConfig::Discrete { min: 1, max: 10 },
            ),
            (
                "optimizer".to_string(),
                ParamConfig::Categorical {
                    choices: vec!["adam".into(), "sgd".into(), "rmsprop".into()],
                },
            ),
            (
                "dropout".to_string(),
                ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    for _ in 0..10 {
        let t = engine.ask().await;
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    let valid_choices: Vec<String> = vec!["adam".into(), "sgd".into(), "rmsprop".into()];
    for _ in 0..20 {
        let t = engine.ask().await;
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
                ParamConfig::Continuous {
                    min: 0.001,
                    max: 1.0,
                    scale: "log10".to_string(),
                },
            ),
            (
                "batch".to_string(),
                ParamConfig::Discrete { min: 16, max: 256 },
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    for _ in 0..20 {
        let t = engine.ask().await;
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
                ParamConfig::Continuous {
                    min: 1e-4,
                    max: 0.1,
                    scale: "log10".to_string(),
                },
            ),
            (
                "layers".to_string(),
                ParamConfig::Discrete { min: 1, max: 10 },
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    let info = engine.space_config();
    assert_eq!(info.len(), 3);

    let info_map: BTreeMap<String, _> = info.into_iter().collect();
    assert_eq!(info_map["lr"].param_type, "continuous");
    assert_eq!(info_map["lr"].scale, "log10");
    assert_eq!(info_map["layers"].param_type, "discrete");
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
            ParamConfig::Continuous {
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
            seed: None,
            elite_fraction: None,
        }),
        checkpoint: None,
    };
    let engine = DynEngine::from_config(config).unwrap();
    assert!(engine.space().contains(&engine.ask().await.params));

    // GMM
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
            seed: None,
            elite_fraction: None,
        }),
        checkpoint: None,
    };
    let engine = DynEngine::from_config(config).unwrap();
    assert!(engine.space().contains(&engine.ask().await.params));

    // Default (Sobol)
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };
    let engine = DynEngine::from_config(config).unwrap();
    assert!(engine.space().contains(&engine.ask().await.params));
}

// ==========================================================================
// Scalarization
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_scalarize_missing_field_infinity() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    let t = engine.ask().await;
    engine.tell(t.trial_id, json!({"accuracy": 0.9})).await.unwrap();

    // Missing field → infeasible trial → no feasible best
    assert!(engine.best().await.is_none());
}

#[tokio::test]
async fn test_dyn_engine_scalarize_maximize() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    let t1 = engine.ask().await;
    engine.tell(t1.trial_id, json!({"accuracy": 0.9})).await.unwrap();
    let t2 = engine.ask().await;
    engine.tell(t2.trial_id, json!({"accuracy": 0.5})).await.unwrap();

    let best = engine.best().await.unwrap();
    assert!(
        best.observation.as_f64().unwrap() < 0.0,
        "Maximized field should be negated"
    );
}

// ==========================================================================
// TLP objectives
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_tlp_objectives() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    let t1 = engine.ask().await;
    engine.tell(t1.trial_id, json!({"loss": 0.5})).await.unwrap();
    let t2 = engine.ask().await;
    engine.tell(t2.trial_id, json!({"loss": 2.0})).await.unwrap();

    // Two trials told, but one is infeasible (loss >= limit=1.5)
    assert_eq!(engine.trial_count().await, 2);
    assert!(engine.best().await.unwrap().observation.as_f64().unwrap().is_finite());
}

// ==========================================================================
// Objectives
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_update_objectives() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    let t = engine.ask().await;
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
        .await;

    assert!(engine.best().await.is_some());
}

#[tokio::test]
async fn test_dyn_engine_objectives_accessor() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
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
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();

    let metrics = vec![
        json!({"loss": 0.1, "accuracy": 0.3}),
        json!({"loss": 0.5, "accuracy": 0.9}),
        json!({"loss": 0.3, "accuracy": 0.5}),
        json!({"loss": 0.8, "accuracy": 0.95}),
        json!({"loss": 0.2, "accuracy": 0.4}),
    ];
    for m in metrics {
        let t = engine.ask().await;
        engine.tell(t.trial_id, m).await.unwrap();
    }

    let best_before = engine.best().await.unwrap();
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
        .await;

    let best_after = engine.best().await.unwrap();
    assert_ne!(best_before.trial_id, best_after.trial_id);
}

// ==========================================================================
// Rescalarize
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_rescalarize() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    let t = engine.ask().await;
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
            ParamConfig::Continuous {
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
            seed: None,
            elite_fraction: None,
        }),
        checkpoint: None,
    };

    let engine = DynEngine::from_config(config).unwrap();
    for i in 0..30 {
        let t = engine.ask().await;
        engine
            .tell(t.trial_id, json!({"loss": (i as f64) * 0.03}))
            .await
            .unwrap();
    }

    assert_eq!(engine.trial_count().await, 30);
    assert!(engine.best().await.is_some());
}

#[tokio::test]
async fn test_refit_excludes_infeasible_scalar() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
            seed: None,
            elite_fraction: None,
        }),
        checkpoint: None,
    };

    let engine = DynEngine::from_config(config).unwrap();

    for i in 0..25 {
        let t = engine.ask().await;
        let loss_val = if i % 5 == 4 { 2.0 } else { (i as f64) * 0.03 };
        engine.tell(t.trial_id, json!({"loss": loss_val})).await.unwrap();
    }

    // 25 trials: 5 infeasible (loss=2.0), 20 feasible
    assert_eq!(engine.trial_count().await, 25);
    assert!(engine.best().await.is_some());

    let t = engine.ask().await;
    assert!(engine.space().contains(&t.params));
}

#[tokio::test]
async fn test_update_objectives_triggers_refit() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
            seed: None,
            elite_fraction: None,
        }),
        checkpoint: None,
    };

    let engine = DynEngine::from_config(config).unwrap();

    for i in 0..30 {
        let t = engine.ask().await;
        let x = (i as f64) / 29.0;
        engine
            .tell(t.trial_id, json!({"loss": x, "accuracy": x}))
            .await
            .unwrap();
    }

    let best_before = engine.best().await.unwrap();

    engine
        .update_objectives(vec![ObjectiveConfig {
            field: "accuracy".to_string(),
            obj_type: "maximize".to_string(),
            target: None,
            limit: None,
            priority: 1.0,
                group: None,
        }])
        .await;

    let best_after = engine.best().await.unwrap();
    assert!(
        best_after.observation.as_f64().unwrap()
            < best_before.observation.as_f64().unwrap()
    );

    let t = engine.ask().await;
    assert!(engine.space().contains(&t.params));
}

// ==========================================================================
// Checkpoints
// ==========================================================================

#[tokio::test]
async fn test_dyn_engine_leaderboard_checkpoint() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config.clone()).unwrap();
    let t0 = engine.ask().await;
    engine.tell(t0.trial_id, json!({"loss": 0.5})).await.unwrap();
    let t1 = engine.ask().await;
    engine.tell(t1.trial_id, json!({"loss": 0.3})).await.unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lb.json");
    engine
        .save_leaderboard_checkpoint_to(&path, Some("2 trials"))
        .await
        .unwrap();

    let engine2 = DynEngine::from_config(config).unwrap();
    engine2.load_leaderboard_checkpoint(&path).await.unwrap();
    assert_eq!(engine2.trial_count().await, 2);
}

#[tokio::test]
async fn test_dyn_engine_full_checkpoint() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config.clone()).unwrap();
    let t = engine.ask().await;
    engine.tell(t.trial_id, json!({"loss": 0.5})).await.unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full.json");
    engine
        .save_full_checkpoint(&path, Some("full checkpoint test"))
        .await
        .unwrap();

    let engine2 = DynEngine::from_config(config).unwrap();
    engine2.load_full_checkpoint(&path).await.unwrap();
    assert_eq!(engine2.trial_count().await, 1);
}

// ==========================================================================
// Auto strategy (Sobol -> GMM switching)
// ==========================================================================

#[tokio::test]
async fn test_auto_strategy_default() {
    // With no strategy config, should use "auto" and work correctly
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();

    // Run enough trials to cross the exploration threshold and trigger refit
    for i in 0..60 {
        let t = engine.ask().await;
        assert!(engine.space().contains(&t.params));
        let loss = (i as f64) / 59.0;
        engine.tell(t.trial_id, json!({"loss": loss})).await.unwrap();
    }

    assert_eq!(engine.trial_count().await, 60);
    let best = engine.best().await;
    assert!(best.is_some());
}

#[tokio::test]
async fn test_auto_strategy_with_explicit_exploration_budget() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
            seed: None,
            elite_fraction: None,
        }),
        checkpoint: None,
    };

    let engine = DynEngine::from_config(config).unwrap();

    for i in 0..30 {
        let t = engine.ask().await;
        assert!(engine.space().contains(&t.params));
        let loss = (i as f64) / 29.0;
        engine.tell(t.trial_id, json!({"loss": loss})).await.unwrap();
    }

    assert_eq!(engine.trial_count().await, 30);
}

#[test]
fn test_auto_strategy_default_exploration_budget() {
    use opt_engine::dyn_engine::AutoStrategy;

    // min(40, 56) = 40 -> round down to 32
    assert_eq!(AutoStrategy::default_exploration_budget(200, 3), 32);

    // min(20, 52) = 20 -> round down to 16
    assert_eq!(AutoStrategy::default_exploration_budget(100, 1), 16);

    // min(200, 60) = 60 -> round down to 32
    assert_eq!(AutoStrategy::default_exploration_budget(1000, 5), 32);

    // min(10, 70) = 10 -> round down to 8
    assert_eq!(AutoStrategy::default_exploration_budget(50, 10), 8);

    // Edge cases
    assert_eq!(AutoStrategy::default_exploration_budget(10, 1), 2); // min(2, 52) = 2
    assert_eq!(AutoStrategy::default_exploration_budget(5, 1), 1);  // min(1, 52) = 1
}

// ==========================================================================
// Seed determinism tests
// ==========================================================================

#[tokio::test]
async fn test_seed_determinism_sobol() {
    let make_engine = |seed| {
        DynEngine::from_config(StudyConfig {
            space: BTreeMap::from([(
                "x".to_string(),
                ParamConfig::Continuous {
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
                seed: Some(seed),
            elite_fraction: None,
            }),
            checkpoint: None,
        })
        .unwrap()
    };

    let e1 = make_engine(123);
    let e2 = make_engine(123);

    for _ in 0..10 {
        let t1 = e1.ask().await;
        let t2 = e2.ask().await;
        assert_eq!(t1.params, t2.params, "Same seed should produce same candidates");
    }
}

#[tokio::test]
async fn test_seed_determinism_random() {
    let make_engine = |seed| {
        DynEngine::from_config(StudyConfig {
            space: BTreeMap::from([(
                "x".to_string(),
                ParamConfig::Continuous {
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
                seed: Some(seed),
            elite_fraction: None,
            }),
            checkpoint: None,
        })
        .unwrap()
    };

    let e1 = make_engine(42);
    let e2 = make_engine(42);

    for _ in 0..10 {
        let t1 = e1.ask().await;
        let t2 = e2.ask().await;
        assert_eq!(t1.params, t2.params, "Same seed should produce same candidates");
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
            ParamConfig::Continuous {
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
            seed: Some(0),
            elite_fraction: None,
        }),
        checkpoint: None,
    };

    let engine = DynEngine::from_config(config).unwrap();

    // Tell trials with known Pareto structure:
    // (1,5) and (5,1) are non-dominated; (3,3) dominated by neither but (4,4) dominated by (3,3)
    let trials_data = vec![
        json!({"f1": 1.0, "f2": 5.0}), // Pareto-optimal
        json!({"f1": 5.0, "f2": 1.0}), // Pareto-optimal
        json!({"f1": 3.0, "f2": 3.0}), // Pareto-optimal
        json!({"f1": 4.0, "f2": 4.0}), // Dominated by (3,3)
    ];

    for data in trials_data {
        let t = engine.ask().await;
        engine.tell(t.trial_id, data).await.unwrap();
    }

    let front = engine.pareto_front().await.unwrap();
    assert_eq!(front.len(), 3, "Should have 3 non-dominated trials");
}

#[tokio::test]
async fn test_pareto_front_scalar_study_errors() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    assert!(engine.pareto_front().await.is_err());
}

#[tokio::test]
async fn test_pareto_front_empty() {
    let config = StudyConfig {
        space: BTreeMap::from([(
            "x".to_string(),
            ParamConfig::Continuous {
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
    };

    let engine = DynEngine::from_config(config).unwrap();
    let front = engine.pareto_front().await.unwrap();
    assert!(front.is_empty());
}
