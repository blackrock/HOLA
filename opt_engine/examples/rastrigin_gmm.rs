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

//! GMM-based optimization on the Rastrigin function.
//!
//! Demonstrates the full optimization pipeline:
//! 1. Sobol exploration phase (20 trials)
//! 2. GMM exploitation phase with refitting (80 trials)

use nalgebra::DVector;
use opt_engine::strategies::{GaussianComponent, GmmParams};
use opt_engine::{
    ContinuousSpace, Engine, GmmStrategy, JsonFieldTransformer, ProductSpace, RefitConfig,
    SobolStrategy,
};

#[tokio::main]
async fn main() {
    println!("Minimizing Rastrigin function with adaptive GMM\n");

    fn rastrigin(x: &[f64]) -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        a * n
            + x.iter()
                .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    type Space2D = ProductSpace<ContinuousSpace, ContinuousSpace>;
    let space: Space2D = ProductSpace {
        a: ContinuousSpace::new(-5.12, 5.12),
        b: ContinuousSpace::new(-5.12, 5.12),
    };

    // Phase 1: Sobol exploration
    println!("Phase 1: Sobol sequence exploration (20 trials)");
    let sobol_engine = Engine::with_leaderboard(
        space.clone(),
        SobolStrategy::<Space2D>::new(42),
        JsonFieldTransformer::default(),
    );

    for _ in 0..20 {
        let job: serde_json::Value = sobol_engine.dispatch_job().await;
        let candidate: (f64, f64) = serde_json::from_value(job.clone()).unwrap();
        let loss = rastrigin(&[candidate.0, candidate.1]);
        sobol_engine
            .ingest_result(job, serde_json::json!({"loss": loss}))
            .await
            .unwrap();
    }

    let sobol_lb = sobol_engine.leaderboard().await.unwrap();
    let sobol_best = sobol_lb.best().unwrap();
    println!(
        "  Best after Sobol: {:.4} at ({:.3}, {:.3})\n",
        sobol_best.observation, sobol_best.candidate.0, sobol_best.candidate.1
    );

    // Phase 2: GMM with refitting
    println!("Phase 2: GMM strategy with refitting (80 trials)");
    let best_x = (sobol_best.candidate.0 + 5.12) / 10.24;
    let best_y = (sobol_best.candidate.1 + 5.12) / 10.24;
    let initial_gmm = GmmParams::single(GaussianComponent::isotropic(
        DVector::from_vec(vec![best_x, best_y]),
        0.1,
    ));

    let gmm_engine = Engine::builder(
        space.clone(),
        GmmStrategy::<Space2D>::new(42, initial_gmm),
        JsonFieldTransformer::default(),
    )
    .with_existing_leaderboard(sobol_lb.clone())
    .with_refit_config(RefitConfig::with_quantile(10, 10, 0.25))
    .build();

    for i in 0..80 {
        let job: serde_json::Value = gmm_engine.dispatch_job().await;
        let candidate: (f64, f64) = serde_json::from_value(job.clone()).unwrap();
        let loss = rastrigin(&[candidate.0, candidate.1]);
        gmm_engine
            .ingest_result(job, serde_json::json!({"loss": loss}))
            .await
            .unwrap();

        if (i + 1) % 20 == 0 {
            let lb = gmm_engine.leaderboard().await.unwrap();
            let best = lb.best().unwrap();
            println!(
                "  Trial {:3}: best = {:.4} at ({:.3}, {:.3})",
                i + 21,
                best.observation,
                best.candidate.0,
                best.candidate.1
            );
        }
    }

    let final_lb = gmm_engine.leaderboard().await.unwrap();
    let final_best = final_lb.best().unwrap();
    println!("\nFinal Results:");
    println!("  Total trials: {}", final_lb.len());
    println!(
        "  Best loss: {:.6} (global optimum = 0)",
        final_best.observation
    );
    println!(
        "  Best point: ({:.4}, {:.4})",
        final_best.candidate.0, final_best.candidate.1
    );

    println!("\nTop 5 trials:");
    for (i, trial) in final_lb.top_k(5).iter().enumerate() {
        println!(
            "  {}. loss={:.4} at ({:.3}, {:.3})",
            i + 1,
            trial.observation,
            trial.candidate.0,
            trial.candidate.1
        );
    }
}
