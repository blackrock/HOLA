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

//! Integration tests for the generic Engine (typed layer).
//!
//! Exercises dispatch/ingest cycles, builder patterns, checkpoints, concurrency,
//! refit, error paths, and multiple transformer types.

use opt_engine::engine::Engine;
use opt_engine::leaderboard::Leaderboard;
use opt_engine::scales::Log10Scale;
use opt_engine::spaces::{ContinuousSpace, DiscreteSpace, ProductSpace};
use opt_engine::strategies::{GmmStrategy, RandomStrategy, SobolStrategy};
use opt_engine::traits::{RefitConfig, SampleSpace};
use opt_engine::transformers::{
    JsonFieldTransformer, JsonTlpTransformer, JsonWeightedTransformer, TlpField, WeightedField,
};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;

// ==========================================================================
// Basic dispatch/ingest
// ==========================================================================

#[tokio::test]
async fn test_engine_dispatch_ingest_cycle() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    );

    assert!(engine.has_leaderboard());
    assert_eq!(engine.trial_count().await, 0);

    let job = engine.dispatch_job().await;
    assert!(job.is_number());

    let val = job.as_f64().unwrap();
    assert!((0.0..=1.0).contains(&val));

    engine
        .ingest_result(job, json!({"loss": 0.42}))
        .await
        .unwrap();

    assert_eq!(engine.trial_count().await, 1);

    let lb = engine.leaderboard().await.unwrap();
    assert_eq!(lb.len(), 1);
    assert!((lb.best().unwrap().observation - 0.42).abs() < 1e-12);
}

#[tokio::test]
async fn test_engine_without_leaderboard() {
    let engine = Engine::new(
        ContinuousSpace::new(0.0, 1.0),
        SobolStrategy::new(42),
        JsonFieldTransformer::loss(),
    );

    assert!(!engine.has_leaderboard());
    assert_eq!(engine.trial_count().await, 0);

    let job = engine.dispatch_job().await;
    engine
        .ingest_result(job, json!({"loss": 0.5}))
        .await
        .unwrap();

    assert_eq!(engine.trial_count().await, 0);
    assert!(engine.leaderboard().await.is_none());
}

// ==========================================================================
// Builder pattern
// ==========================================================================

#[tokio::test]
async fn test_engine_builder_all_options() {
    let engine = Engine::builder(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    )
    .with_leaderboard()
    .with_refit_config(RefitConfig::with_top_k(5, 5, 3))
    .build();

    assert!(engine.has_leaderboard());

    for _ in 0..10 {
        let job = engine.dispatch_job().await;
        engine
            .ingest_result(job.clone(), json!({"loss": rand::random::<f64>()}))
            .await
            .unwrap();
    }

    assert_eq!(engine.trial_count().await, 10);
}

#[tokio::test]
async fn test_engine_builder_with_existing_leaderboard() {
    let mut lb = Leaderboard::<f64, f64>::new();
    lb.push(0.3, 0.2);
    lb.push(0.7, 0.8);

    let engine = Engine::builder(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    )
    .with_existing_leaderboard(lb)
    .build();

    assert!(engine.has_leaderboard());
    assert_eq!(engine.trial_count().await, 2);

    let lb = engine.leaderboard().await.unwrap();
    assert!((lb.best().unwrap().observation - 0.2).abs() < 1e-12);
}

#[tokio::test]
async fn test_engine_builder_refit_config_enables_leaderboard() {
    let engine = Engine::builder(
        ContinuousSpace::new(0.0, 1.0),
        GmmStrategy::uniform_prior(42, 1, 1.0),
        JsonFieldTransformer::loss(),
    )
    .with_refit_config(RefitConfig::default())
    .build();

    assert!(engine.has_leaderboard());
}

// ==========================================================================
// Leaderboard access
// ==========================================================================

#[tokio::test]
async fn test_engine_with_leaderboard_ref() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    );

    let job = engine.dispatch_job().await;
    engine
        .ingest_result(job, json!({"loss": 0.5}))
        .await
        .unwrap();

    let count = engine.with_leaderboard_ref(|lb| lb.len()).await;
    assert_eq!(count, Some(1));

    let engine2 = Engine::new(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    );
    assert!(engine2.with_leaderboard_ref(|lb| lb.len()).await.is_none());
}

#[tokio::test]
async fn test_engine_space_accessor() {
    let engine = Engine::new(
        ContinuousSpace::new(-5.0, 5.0),
        SobolStrategy::new(0),
        JsonFieldTransformer::loss(),
    );

    let sp = engine.space();
    assert_eq!(sp.min, -5.0);
    assert_eq!(sp.max, 5.0);
    assert!(sp.contains(&0.0));
    assert!(!sp.contains(&10.0));
}

#[tokio::test]
async fn test_engine_trial_ids_monotonic() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    );

    for i in 0..10 {
        let job = engine.dispatch_job().await;
        engine
            .ingest_result(job, json!({"loss": i as f64 * 0.1}))
            .await
            .unwrap();
    }

    let lb = engine.leaderboard().await.unwrap();
    let ids: Vec<u64> = lb.iter().map(|t| t.trial_id).collect();
    for i in 1..ids.len() {
        assert!(ids[i] > ids[i - 1]);
    }
}

// ==========================================================================
// Checkpoints
// ==========================================================================

#[tokio::test]
async fn test_engine_save_load_leaderboard_checkpoint() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        SobolStrategy::new(42),
        JsonFieldTransformer::loss(),
    );

    let job = engine.dispatch_job().await;
    engine
        .ingest_result(job, json!({"loss": 0.3}))
        .await
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("engine_lb.json");

    engine
        .save_leaderboard_checkpoint(&path, Some("engine test"))
        .await
        .unwrap();

    let engine2 = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        SobolStrategy::new(0),
        JsonFieldTransformer::loss(),
    );
    engine2.load_leaderboard_checkpoint(&path).await.unwrap();
    assert_eq!(engine2.trial_count().await, 1);
}

#[tokio::test]
async fn test_engine_save_load_full_checkpoint() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        SobolStrategy::new(42),
        JsonFieldTransformer::loss(),
    );

    let job = engine.dispatch_job().await;
    engine
        .ingest_result(job, json!({"loss": 0.5}))
        .await
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("engine_full.json");

    engine
        .save_checkpoint(&path, Some("full engine"))
        .await
        .unwrap();

    let engine2 = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        SobolStrategy::new(0),
        JsonFieldTransformer::loss(),
    );
    engine2.load_checkpoint(&path).await.unwrap();
    assert_eq!(engine2.trial_count().await, 1);
}

#[tokio::test]
async fn test_engine_checkpoint_roundtrip_preserves_strategy() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        SobolStrategy::new(42),
        JsonFieldTransformer::loss(),
    );

    for _ in 0..5 {
        let job = engine.dispatch_job().await;
        engine
            .ingest_result(job, json!({"loss": rand::random::<f64>()}))
            .await
            .unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full.json");
    engine
        .save_checkpoint(&path, Some("strategy test"))
        .await
        .unwrap();

    let engine2 = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        SobolStrategy::new(0),
        JsonFieldTransformer::loss(),
    );
    engine2.load_checkpoint(&path).await.unwrap();

    let next1 = engine.dispatch_job().await;
    let next2 = engine2.dispatch_job().await;
    assert_eq!(next1.as_f64().unwrap(), next2.as_f64().unwrap());
    assert_eq!(engine2.trial_count().await, 5);
}

#[tokio::test]
async fn test_engine_random_checkpoint_roundtrip() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::new(42),
        JsonFieldTransformer::loss(),
    );

    for _ in 0..5 {
        let job = engine.dispatch_job().await;
        engine
            .ingest_result(job, json!({"loss": rand::random::<f64>()}))
            .await
            .unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("random_full.json");
    engine.save_checkpoint(&path, None).await.unwrap();

    let engine2 = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::new(0),
        JsonFieldTransformer::loss(),
    );
    engine2.load_checkpoint(&path).await.unwrap();

    let next1 = engine.dispatch_job().await;
    let next2 = engine2.dispatch_job().await;
    assert_eq!(next1.as_f64().unwrap(), next2.as_f64().unwrap());
}

// ==========================================================================
// Concurrency
// ==========================================================================

#[tokio::test]
async fn test_engine_concurrent_dispatch_ingest() {
    let engine = Arc::new(Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    ));

    let mut handles = Vec::new();
    for i in 0..10 {
        let eng = engine.clone();
        handles.push(tokio::spawn(async move {
            let job = eng.dispatch_job().await;
            eng.ingest_result(job, json!({"loss": i as f64 * 0.1}))
                .await
                .unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    assert_eq!(engine.trial_count().await, 10);
}

// ==========================================================================
// Refit
// ==========================================================================

#[tokio::test]
async fn test_engine_ingest_with_refit_triggers() {
    let engine = Engine::builder(
        ContinuousSpace::new(0.0, 1.0),
        GmmStrategy::uniform_prior(42, 1, 1.0),
        JsonFieldTransformer::loss(),
    )
    .with_refit_config(RefitConfig::with_top_k(5, 5, 3))
    .build();

    for i in 0..25 {
        let job = engine.dispatch_job().await;
        engine
            .ingest_result_with_refit(job, json!({"loss": (i as f64) * 0.04}))
            .await
            .unwrap();
    }

    assert_eq!(engine.trial_count().await, 25);
}

#[tokio::test]
async fn test_engine_refit_no_leaderboard_returns_err() {
    let engine = Engine::new(
        ContinuousSpace::new(0.0, 1.0),
        GmmStrategy::uniform_prior(42, 1, 1.0),
        JsonFieldTransformer::loss(),
    );

    assert!(engine.refit(|lb| lb.top_k(5)).await.is_err());
}

#[tokio::test]
async fn test_engine_refit_manual_selector() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        GmmStrategy::uniform_prior(42, 1, 1.0),
        JsonFieldTransformer::loss(),
    );

    for i in 0..10 {
        let job = engine.dispatch_job().await;
        engine
            .ingest_result(job, json!({"loss": (i as f64) * 0.1}))
            .await
            .unwrap();
    }

    engine.refit(|lb| lb.top_k(5)).await.unwrap();

    let job = engine.dispatch_job().await;
    let val = job.as_f64().unwrap();
    assert!((0.0..=1.0).contains(&val));
}

// ==========================================================================
// Error paths
// ==========================================================================

#[tokio::test]
async fn test_engine_ingest_result_bad_json() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    );

    let result = engine
        .ingest_result(json!("not_a_number"), json!({"loss": 0.5}))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_engine_ingest_bad_transformer_input() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    );

    let job = engine.dispatch_job().await;
    let result = engine.ingest_result(job, json!({"accuracy": 0.9})).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("loss"));
}

#[tokio::test]
async fn test_engine_save_leaderboard_no_leaderboard_err() {
    let engine = Engine::new(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("should_fail.json");
    assert!(
        engine
            .save_leaderboard_checkpoint(&path, None)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn test_engine_load_nonexistent_file() {
    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        SobolStrategy::new(42),
        JsonFieldTransformer::loss(),
    );

    assert!(
        engine
            .load_leaderboard_checkpoint("/nonexistent/path/to/file.json")
            .await
            .is_err()
    );
}

// ==========================================================================
// Multiple transformer types
// ==========================================================================

#[tokio::test]
async fn test_engine_with_weighted_transformer() {
    let transformer = JsonWeightedTransformer::new(vec![
        WeightedField::minimize("loss", 1.0),
        WeightedField::minimize("latency", 0.5),
    ]);

    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        transformer,
    );

    for i in 0..5 {
        let job = engine.dispatch_job().await;
        engine
            .ingest_result(
                job,
                json!({"loss": i as f64 * 0.2, "latency": 100.0 - i as f64 * 10.0}),
            )
            .await
            .unwrap();
    }

    assert_eq!(engine.trial_count().await, 5);
    assert!(engine.leaderboard().await.unwrap().best().is_some());
}

#[tokio::test]
async fn test_engine_with_tlp_transformer() {
    let transformer = JsonTlpTransformer::new(vec![TlpField::minimize("loss", 0.0, 1.0, 1.0)]);

    let engine = Engine::with_leaderboard(
        ContinuousSpace::new(0.0, 1.0),
        RandomStrategy::auto_seed(),
        transformer,
    );

    let job = engine.dispatch_job().await;
    engine
        .ingest_result(job, json!({"loss": 0.5}))
        .await
        .unwrap();

    let job = engine.dispatch_job().await;
    engine
        .ingest_result(job, json!({"loss": 2.0}))
        .await
        .unwrap();

    let lb = engine.leaderboard().await.unwrap();
    assert_eq!(lb.feasible_count(), 1);
    assert!(lb.best().unwrap().observation.is_finite());
}

// ==========================================================================
// Sobol unique points
// ==========================================================================

#[tokio::test]
async fn test_engine_dispatch_many_sobol_unique() {
    let space = ProductSpace {
        a: ContinuousSpace::new(0.0, 1.0),
        b: ContinuousSpace::new(0.0, 1.0),
    };
    let engine =
        Engine::with_leaderboard(space, SobolStrategy::new(42), JsonFieldTransformer::loss());

    let mut points = Vec::new();
    for _ in 0..100 {
        let job = engine.dispatch_job().await;
        let arr: (f64, f64) = serde_json::from_value(job.clone()).unwrap();
        assert!(arr.0 >= 0.0 && arr.0 <= 1.0);
        assert!(arr.1 >= 0.0 && arr.1 <= 1.0);
        points.push(job.to_string());

        engine
            .ingest_result(job, json!({"loss": rand::random::<f64>()}))
            .await
            .unwrap();
    }

    let unique: HashSet<&String> = points.iter().collect();
    assert_eq!(unique.len(), 100);
}

// ==========================================================================
// Multi-dimensional space with log10
// ==========================================================================

#[tokio::test]
async fn test_engine_with_log10_product_space() {
    let space = ProductSpace {
        a: ContinuousSpace::with_scale(1e-4, 0.1, Log10Scale),
        b: DiscreteSpace::new(1, 10),
    };

    let engine = Engine::with_leaderboard(
        space,
        RandomStrategy::auto_seed(),
        JsonFieldTransformer::loss(),
    );

    for _ in 0..10 {
        let job = engine.dispatch_job().await;
        let candidate: (f64, i64) = serde_json::from_value(job.clone()).unwrap();

        assert!(candidate.0 >= 1e-4 - 1e-9 && candidate.0 <= 0.1 + 1e-9);
        assert!(candidate.1 >= 1 && candidate.1 <= 10);

        engine
            .ingest_result(job, json!({"loss": rand::random::<f64>()}))
            .await
            .unwrap();
    }

    assert_eq!(engine.trial_count().await, 10);
}
