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
use hola::server::ServerOptions;
use std::net::IpAddr;
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
        /// Host/interface to bind. Defaults to localhost; use 0.0.0.0 explicitly for network access.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on.
        #[arg(long, default_value = "8000")]
        port: u16,
        /// Serve the dashboard UI from this directory.
        #[arg(long)]
        dashboard: Option<PathBuf>,
        /// Bearer token required for write-capable API endpoints.
        #[arg(long)]
        auth_token: Option<String>,
        /// Directory where dashboard/API checkpoint saves are allowed.
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        /// Allowed CORS origin. May be provided multiple times.
        #[arg(long = "cors-origin")]
        cors_origins: Vec<String>,
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
        /// Bearer token for servers started with --auth-token.
        #[arg(long)]
        token: Option<String>,
    },
}

fn load_config(path: &PathBuf) -> Result<StudyConfig, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    let config: StudyConfig = serde_yaml::from_str(&contents)?;
    Ok(config)
}

fn is_local_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn configured_token(cli_token: Option<String>) -> Option<String> {
    cli_token
        .or_else(|| std::env::var("HOLA_API_TOKEN").ok())
        .filter(|token| !token.is_empty())
}

fn with_bearer_auth(
    request: reqwest::RequestBuilder,
    token: Option<&str>,
) -> reqwest::RequestBuilder {
    match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            config,
            host,
            port,
            dashboard,
            auth_token,
            checkpoint_dir,
            cors_origins,
        } => {
            let study_config = load_config(&config)?;
            let load_from = study_config
                .checkpoint
                .as_ref()
                .and_then(|c| c.load_from.clone());
            let config_checkpoint_dir = study_config
                .checkpoint
                .as_ref()
                .map(|checkpoint| PathBuf::from(&checkpoint.directory));
            let engine = HolaEngine::from_config(study_config)
                .map_err(|e| format!("Failed to create engine: {e}"))?;

            if let Some(path) = load_from {
                let checkpoint_kind = engine
                    .load_checkpoint_with_fallback(&path)
                    .await
                    .map_err(|e| format!("Failed to load checkpoint '{path}': {e}"))?;
                eprintln!("Loaded {} checkpoint from {path}", checkpoint_kind.as_str());
            }

            let auth_token = configured_token(auth_token);
            if !is_local_host(&host) && auth_token.is_none() {
                return Err(
                    "--auth-token or HOLA_API_TOKEN is required when --host is not localhost"
                        .into(),
                );
            }

            let mut options = ServerOptions::new(port);
            options.host = host;
            options.dashboard_dir = dashboard;
            options.auth_token = auth_token;
            options.checkpoint_dir = checkpoint_dir
                .or(config_checkpoint_dir)
                .or_else(|| config.parent().map(|path| path.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."));
            options.cors_allowed_origins = cors_origins;

            hola::server::serve_with_options(engine, options).await?;
        }
        Commands::Worker {
            server,
            exec,
            mode,
            token,
        } => {
            let exec_mode = mode == "exec";
            let token = configured_token(token);
            eprintln!("Worker connecting to {server} ({mode} mode)...");
            eprintln!("Will execute: {exec}");

            let client = reqwest::Client::new();

            loop {
                let resp =
                    with_bearer_auth(client.post(format!("{server}/api/ask")), token.as_deref())
                        .send()
                        .await;

                match resp {
                    Ok(r) => {
                        if !r.status().is_success() {
                            let status = r.status();
                            let body = r
                                .text()
                                .await
                                .unwrap_or_else(|_| "unknown error".to_string());
                            eprintln!("Server returned {status}: {body}. Retrying in 5s...");
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                            continue;
                        }

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

                            let tell_resp = client.post(format!("{server}/api/tell"));
                            let tell_resp = with_bearer_auth(tell_resp, token.as_deref())
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
                            let mut command = std::process::Command::new("sh");
                            command
                                .arg("-c")
                                .arg(&exec)
                                .env("HOLA_SERVER", &server)
                                .env("HOLA_TRIAL_ID", trial_id.to_string())
                                .env("HOLA_PARAMS", params.to_string());
                            if let Some(token) = &token {
                                command.env("HOLA_API_TOKEN", token);
                            }
                            let status = command.status()?;

                            if status.success() {
                                eprintln!("Trial {trial_id}: script exited successfully");
                            } else {
                                eprintln!(
                                    "Trial {trial_id}: script failed (exit {}), canceling",
                                    status.code().unwrap_or(-1)
                                );
                                let _ = with_bearer_auth(
                                    client.post(format!("{server}/api/cancel")),
                                    token.as_deref(),
                                )
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
