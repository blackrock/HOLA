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

//! Multi-objective optimization with Pareto front analysis.

use opt_engine::Leaderboard;
use std::collections::BTreeMap;

fn main() {
    println!("Multi-objective optimization with Pareto front\n");

    let mut lb: Leaderboard<String, BTreeMap<String, f64>> = Leaderboard::new();

    // Add trials with two objectives (lower is better for both)
    lb.push(
        "balanced".into(),
        [("loss".into(), 0.3), ("latency".into(), 100.0)].into(),
    );
    lb.push(
        "fast_but_poor".into(),
        [("loss".into(), 0.5), ("latency".into(), 50.0)].into(),
    );
    lb.push(
        "accurate_slow".into(),
        [("loss".into(), 0.1), ("latency".into(), 200.0)].into(),
    );
    lb.push(
        "dominated".into(),
        [("loss".into(), 0.4), ("latency".into(), 150.0)].into(),
    );
    lb.push(
        "best_loss".into(),
        [("loss".into(), 0.05), ("latency".into(), 300.0)].into(),
    );

    println!("Added {} trials with 2 objectives\n", lb.len());

    // Pareto front
    let front = lb.pareto_front();
    println!("Pareto front ({} trials):", front.len());
    for trial in &front {
        let loss = trial.observation.get("loss").unwrap();
        let latency = trial.observation.get("latency").unwrap();
        println!(
            "  {} -> loss={:.2}, latency={:.0}",
            trial.candidate, loss, latency
        );
    }

    // Scalarized ranking
    println!("\nTop 3 by weighted sum (loss + latency*0.01):");
    let top = lb.top_k_scalarized(3, |obs| {
        obs.get("loss").unwrap_or(&1.0) + obs.get("latency").unwrap_or(&1000.0) * 0.01
    });
    for trial in top {
        let score = trial.observation.get("loss").unwrap()
            + trial.observation.get("latency").unwrap() * 0.01;
        println!("  {} -> score={:.3}", trial.candidate, score);
    }

    // Best for single objective
    let best_loss = lb.best_for_objective("loss").unwrap();
    println!(
        "\nBest for 'loss': {} -> {:.2}",
        best_loss.candidate,
        best_loss.observation.get("loss").unwrap()
    );
}
