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

//! End-to-end optimization tests.
//!
//! Validates that the system actually finds good solutions for known objective
//! functions. Fast versions run by default; thorough versions are `#[ignore]`.

use opt_engine::engine::Engine;
use opt_engine::scales::Log10Scale;
use opt_engine::spaces::{CategoricalSpace, ContinuousSpace, DiscreteSpace, ProductSpace};
use opt_engine::strategies::{GmmStrategy, RandomStrategy, SobolStrategy};
use opt_engine::traits::RefitConfig;
use opt_engine::transformers::JsonFieldTransformer;
use serde_json::json;

/// Sphere function: f(x, y) = x^2 + y^2. Global minimum at (0, 0).
fn sphere(x: f64, y: f64) -> f64 {
    x * x + y * y
}

// ==========================================================================
// Sphere function with different strategies (fast versions)
// ==========================================================================

#[tokio::test]
async fn test_e2e_sphere_random() {
    let space = ProductSpace {
        a: ContinuousSpace::new(-5.0, 5.0),
        b: ContinuousSpace::new(-5.0, 5.0),
    };
    let engine =
        Engine::with_leaderboard(space, RandomStrategy::new(42), JsonFieldTransformer::loss());

    for _ in 0..200 {
        let job = engine.dispatch_job().await;
        let (x, y): (f64, f64) = serde_json::from_value(job.clone()).unwrap();
        let loss = sphere(x, y);
        engine
            .ingest_result(job, json!({"loss": loss}))
            .await
            .unwrap();
    }

    let lb = engine.leaderboard().await.unwrap();
    let best = lb.best().unwrap();
    assert!(
        best.observation < 2.0,
        "Random search on sphere should find best < 2.0 in 200 trials, got {}",
        best.observation
    );
}

#[tokio::test]
async fn test_e2e_sphere_sobol() {
    let space = ProductSpace {
        a: ContinuousSpace::new(-5.0, 5.0),
        b: ContinuousSpace::new(-5.0, 5.0),
    };
    let engine =
        Engine::with_leaderboard(space, SobolStrategy::new(42), JsonFieldTransformer::loss());

    for _ in 0..200 {
        let job = engine.dispatch_job().await;
        let (x, y): (f64, f64) = serde_json::from_value(job.clone()).unwrap();
        let loss = sphere(x, y);
        engine
            .ingest_result(job, json!({"loss": loss}))
            .await
            .unwrap();
    }

    let lb = engine.leaderboard().await.unwrap();
    let best = lb.best().unwrap();
    assert!(
        best.observation < 1.0,
        "Sobol search on sphere should find best < 1.0 in 200 trials, got {}",
        best.observation
    );
}

#[tokio::test]
async fn test_e2e_sphere_gmm() {
    let space = ProductSpace {
        a: ContinuousSpace::new(-5.0, 5.0),
        b: ContinuousSpace::new(-5.0, 5.0),
    };

    let engine = Engine::builder(
        space,
        GmmStrategy::uniform_prior(42, 2, 1.0),
        JsonFieldTransformer::loss(),
    )
    .with_refit_config(RefitConfig::with_quantile(10, 5, 0.25))
    .build();

    for _ in 0..200 {
        let job = engine.dispatch_job().await;
        let (x, y): (f64, f64) = serde_json::from_value(job.clone()).unwrap();
        let loss = sphere(x, y);
        engine
            .ingest_result_with_refit(job, json!({"loss": loss}))
            .await
            .unwrap();
    }

    let lb = engine.leaderboard().await.unwrap();
    let best = lb.best().unwrap();
    assert!(
        best.observation < 15.0,
        "GMM search on sphere should find best < 15.0 in 200 trials, got {}",
        best.observation
    );
}

// ==========================================================================
// Thorough versions (slow, marked #[ignore])
// ==========================================================================

#[tokio::test]
#[ignore]
async fn test_e2e_sphere_gmm_thorough() {
    let space = ProductSpace {
        a: ContinuousSpace::new(-5.0, 5.0),
        b: ContinuousSpace::new(-5.0, 5.0),
    };

    let engine = Engine::builder(
        space,
        GmmStrategy::uniform_prior(42, 2, 1.0),
        JsonFieldTransformer::loss(),
    )
    .with_refit_config(RefitConfig::with_quantile(10, 5, 0.25))
    .build();

    for _ in 0..500 {
        let job = engine.dispatch_job().await;
        let (x, y): (f64, f64) = serde_json::from_value(job.clone()).unwrap();
        let loss = sphere(x, y);
        engine
            .ingest_result_with_refit(job, json!({"loss": loss}))
            .await
            .unwrap();
    }

    let lb = engine.leaderboard().await.unwrap();
    let best = lb.best().unwrap();
    assert!(
        best.observation < 0.1,
        "GMM search on sphere should find best < 0.1 in 500 trials, got {}",
        best.observation
    );
}

#[tokio::test]
#[ignore]
async fn test_e2e_rosenbrock_sobol_thorough() {
    let space = ProductSpace {
        a: ContinuousSpace::new(-2.0, 2.0),
        b: ContinuousSpace::new(-2.0, 2.0),
    };
    let engine =
        Engine::with_leaderboard(space, SobolStrategy::new(42), JsonFieldTransformer::loss());

    for _ in 0..500 {
        let job = engine.dispatch_job().await;
        let (x, y): (f64, f64) = serde_json::from_value(job.clone()).unwrap();
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, min at (1,1)
        let loss = (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2);
        engine
            .ingest_result(job, json!({"loss": loss}))
            .await
            .unwrap();
    }

    let lb = engine.leaderboard().await.unwrap();
    let best = lb.best().unwrap();
    assert!(
        best.observation < 10.0,
        "Sobol search on Rosenbrock should find best < 10.0 in 500 trials, got {}",
        best.observation
    );
}

// ==========================================================================
// Mixed space optimization
// ==========================================================================

#[tokio::test]
async fn test_e2e_mixed_space_optimization() {
    let space = ProductSpace {
        a: ProductSpace {
            a: ContinuousSpace::with_scale(1e-4, 0.1, Log10Scale),
            b: DiscreteSpace::new(1, 10),
        },
        b: CategoricalSpace::from_strs(&["small", "medium", "large"]),
    };

    let engine =
        Engine::with_leaderboard(space, SobolStrategy::new(42), JsonFieldTransformer::loss());

    // Synthetic objective: prefers lr near 0.01, layers near 5, and "medium"
    for _ in 0..100 {
        let job = engine.dispatch_job().await;
        let ((lr, layers), opt): ((f64, i64), String) =
            serde_json::from_value(job.clone()).unwrap();

        let loss = (lr - 0.01).powi(2) * 1000.0
            + (layers as f64 - 5.0).powi(2)
            + if opt == "medium" { 0.0 } else { 10.0 };

        engine
            .ingest_result(job, json!({"loss": loss}))
            .await
            .unwrap();
    }

    let lb = engine.leaderboard().await.unwrap();
    let best = lb.best().unwrap();
    // Should find something reasonable
    assert!(
        best.observation < 20.0,
        "Mixed space search should find reasonable solution, got {}",
        best.observation
    );
}
