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

//! Ignored scalability probes for leaderboard ranking hot paths.
//!
//! Run with:
//! `RUN_EXPENSIVE_BENCHMARKS=1 cargo test -p opt_engine --test integration leaderboard_scalability -- --ignored --nocapture`

use opt_engine::leaderboard::Leaderboard;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

const REPRESENTATIVE_SIZES: &[usize] = &[1_000, 10_000, 50_000];

fn benchmark_sizes() -> Vec<usize> {
    if std::env::var_os("RUN_EXPENSIVE_BENCHMARKS").is_some() {
        REPRESENTATIVE_SIZES.to_vec()
    } else {
        vec![1_000]
    }
}

fn scalar_leaderboard(n: usize) -> Leaderboard<usize, f64> {
    let mut lb = Leaderboard::with_capacity(n);
    for i in 0..n {
        let score = ((i.wrapping_mul(37) % n.max(1)) as f64) / n.max(1) as f64;
        lb.push(i, score);
    }
    lb
}

fn vector_leaderboard(n: usize) -> Leaderboard<usize, BTreeMap<String, f64>> {
    let mut lb = Leaderboard::with_capacity(n);
    for i in 0..n {
        let mut obs = BTreeMap::new();
        obs.insert("loss".to_string(), (i % 997) as f64 / 997.0);
        obs.insert("latency".to_string(), ((n - i) % 991) as f64 / 991.0);
        lb.push(i, obs);
    }
    lb
}

fn time_it<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let result = f();
    eprintln!("{label}: {:?}", start.elapsed());
    result
}

#[test]
#[ignore = "performance probe; run explicitly with --ignored --nocapture"]
fn leaderboard_scalability_scalar_ranking() {
    for n in benchmark_sizes() {
        let lb = scalar_leaderboard(n);
        let top = time_it(&format!("scalar n={n} top_k(100)"), || {
            black_box(lb.top_k(100))
        });
        assert_eq!(top.len(), 100.min(n));

        let all = time_it(&format!("scalar n={n} sorted_all()"), || {
            black_box(lb.sorted_all())
        });
        assert_eq!(all.len(), n);
    }
}

#[test]
#[ignore = "performance probe; run explicitly with --ignored --nocapture"]
fn leaderboard_scalability_vector_ranking() {
    for n in benchmark_sizes() {
        let lb = vector_leaderboard(n);
        let top = time_it(&format!("vector n={n} top_k_scalarized(100)"), || {
            black_box(lb.top_k_scalarized(100, |obs| obs.values().sum()))
        });
        assert_eq!(top.len(), 100.min(n));

        let front = time_it(&format!("vector n={n} pareto_front()"), || {
            black_box(lb.pareto_front())
        });
        assert!(!front.is_empty());
    }
}
