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

//! Checkpoint save and load with the persistence module.

use opt_engine::{Leaderboard, LeaderboardCheckpoint};

#[tokio::main]
async fn main() {
    println!("Checkpoint save/load demo\n");

    // Create and populate a leaderboard
    let mut lb: Leaderboard<f64, f64> = Leaderboard::new();
    lb.push(0.1, 0.5);
    lb.push(0.3, 0.3);
    lb.push(0.2, 0.4);

    // Save to JSON string
    let checkpoint = LeaderboardCheckpoint::new(lb.clone(), Some("Demo checkpoint"));
    let json = checkpoint.to_json().unwrap();

    println!("Checkpoint JSON preview:");
    let preview: String = json.chars().take(500).collect();
    println!("  {}\n", preview.replace('\n', "\n  "));

    // Restore from JSON string
    let restored: LeaderboardCheckpoint<f64, f64> =
        LeaderboardCheckpoint::from_json(&json).unwrap();

    println!("Restored {} trials", restored.leaderboard.len());
    println!(
        "Metadata: {} trials at {}",
        restored.metadata.n_trials, restored.metadata.created_at_iso
    );

    // You can also save/load from files:
    //   checkpoint.save_json("checkpoint.json").unwrap();
    //   let loaded: LeaderboardCheckpoint<f64, f64> =
    //       LeaderboardCheckpoint::load_json("checkpoint.json").unwrap();
}
