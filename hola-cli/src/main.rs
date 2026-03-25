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

//! HOLA CLI — serve optimization studies and run workers.

use clap::{Parser, Subcommand};
use hola::hola_engine::{HolaEngine, StudyConfig};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "hola",
    version,
    about = "Distributed optimization engine (HOLA)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the optimization server from a YAML config.
    Serve {
        /// Path to the study YAML config file.
        config: PathBuf,
        /// Port to listen on.
        #[arg(long, default_value = "8000")]
        port: u16,
        /// Serve the dashboard UI from this directory.
        #[arg(long)]
        dashboard: Option<PathBuf>,
    },
    /// Run a worker that polls the server for trials.
    ///
    /// In "callback" mode (the default), the worker sets HOLA_SERVER,
    /// HOLA_TRIAL_ID, and HOLA_PARAMS environment variables, then runs
    /// your --exec command. The command is responsible for calling
    /// POST /api/tell to report results. If the command exits with
    /// non-zero status, the worker cancels the trial.
    ///
    /// In "exec" mode, the worker runs the command, parses its stdout
    /// as a JSON metrics object, and reports the result on the
    /// command's behalf.
    Worker {
        /// URL of the HOLA server (e.g. http://localhost:8000).
        #[arg(long)]
        server: String,
        /// Command to execute for each trial.
        #[arg(long)]
        exec: String,
        /// Worker mode: "callback" (default) or "exec".
        #[arg(long, default_value = "callback")]
        mode: String,
    },
}

fn load_config(path: &PathBuf) -> Result<StudyConfig, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    let config: StudyConfig = serde_yaml::from_str(&contents)?;
    Ok(config)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            config,
            port,
            dashboard,
        } => {
            let study_config = load_config(&config)?;
            let load_from = study_config
                .checkpoint
                .as_ref()
                .and_then(|c| c.load_from.clone());
            let engine = HolaEngine::from_config(study_config)
                .map_err(|e| format!("Failed to create engine: {e}"))?;

            if let Some(path) = load_from {
                engine
                    .load_leaderboard_checkpoint(&path)
                    .await
                    .map_err(|e| format!("Failed to load checkpoint '{path}': {e}"))?;
            }

            hola::server::serve(engine, port, dashboard.as_deref()).await?;
        }
        Commands::Worker { server, exec, mode } => {
            let exec_mode = mode == "exec";
            eprintln!("Worker connecting to {server} ({mode} mode)...");
            eprintln!("Will execute: {exec}");

            let client = reqwest::Client::new();

            loop {
                let resp = client.post(format!("{server}/api/ask")).send().await;

                match resp {
                    Ok(r) => {
                        let trial: serde_json::Value = r.json().await?;
                        let params = trial.get("params").cloned().unwrap_or_default();
                        let trial_id = trial.get("trial_id").and_then(|v| v.as_u64()).unwrap_or(0);

                        if exec_mode {
                            // Exec mode: run command, parse stdout as
                            // JSON metrics, report on the script's behalf.
                            let output = std::process::Command::new("sh")
                                .arg("-c")
                                .arg(&exec)
                                .env("HOLA_PARAMS", params.to_string())
                                .output()?;

                            let stdout = String::from_utf8_lossy(&output.stdout);
                            let metrics: serde_json::Value = serde_json::from_str(stdout.trim())
                                .unwrap_or_else(|_| serde_json::json!({"error": "parse_failed"}));

                            let tell_resp = client
                                .post(format!("{server}/api/tell"))
                                .json(&serde_json::json!({
                                    "trial_id": trial_id,
                                    "metrics": metrics,
                                }))
                                .send()
                                .await;

                            match tell_resp {
                                Ok(_) => {
                                    eprintln!("Completed trial {trial_id}: {metrics}")
                                }
                                Err(e) => {
                                    eprintln!("Failed to report trial {trial_id}: {e}")
                                }
                            }
                        } else {
                            // Callback mode (default): script calls
                            // POST /api/tell itself via HOLA_SERVER.
                            let status = std::process::Command::new("sh")
                                .arg("-c")
                                .arg(&exec)
                                .env("HOLA_SERVER", &server)
                                .env("HOLA_TRIAL_ID", trial_id.to_string())
                                .env("HOLA_PARAMS", params.to_string())
                                .status()?;

                            if status.success() {
                                eprintln!("Trial {trial_id}: script exited successfully");
                            } else {
                                eprintln!(
                                    "Trial {trial_id}: script failed (exit {}), canceling",
                                    status.code().unwrap_or(-1)
                                );
                                let _ = client
                                    .post(format!("{server}/api/cancel"))
                                    .json(&serde_json::json!({"trial_id": trial_id}))
                                    .send()
                                    .await;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to connect to server: {e}. Retrying in 5s...");
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    }
                }
            }
        }
    }

    Ok(())
}
