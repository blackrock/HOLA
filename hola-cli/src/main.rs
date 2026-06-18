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

use clap::{Parser, Subcommand, ValueEnum};
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
        /// Also require the bearer token for read-only endpoints and the SSE
        /// stream. Only has an effect together with --auth-token; off by
        /// default, so reads stay open while writes remain protected.
        #[arg(long)]
        require_read_auth: bool,
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
        #[arg(long, value_enum, default_value_t = WorkerMode::Callback)]
        mode: WorkerMode,
        /// Bearer token for servers started with --auth-token.
        #[arg(long)]
        token: Option<String>,
    },
}

/// Strategy a worker uses to report trial results.
#[derive(Clone, ValueEnum)]
enum WorkerMode {
    /// The executed command calls POST /api/tell itself via HOLA_SERVER.
    Callback,
    /// The worker parses the command's stdout as metrics and reports them.
    Exec,
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

/// Validate and normalize a server base URL.
///
/// Requires an http/https scheme and a host, and strips any trailing slash so
/// that paths joined as "{server}/api/..." never produce a double slash. A
/// scheme-less or otherwise malformed value is rejected with a clear error so
/// the worker fails fast instead of retrying forever against a bad address.
fn normalize_server_url(server: &str) -> Result<String, Box<dyn std::error::Error>> {
    let url =
        reqwest::Url::parse(server).map_err(|e| format!("invalid --server URL '{server}': {e}"))?;
    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(format!(
                "invalid --server URL '{server}': unsupported scheme '{other}', expected http or https"
            )
            .into());
        }
    }
    if url.host_str().is_none() {
        return Err(format!("invalid --server URL '{server}': missing host").into());
    }
    Ok(server.trim_end_matches('/').to_string())
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

/// Maximum number of bytes captured from a child's stdout/stderr in exec
/// mode. A runaway command can produce unbounded output, so we cap each
/// stream to avoid exhausting worker memory. Metrics objects are small, so
/// this is generous for legitimate use.
const MAX_CAPTURE_BYTES: usize = 1 << 20; // 1 MiB

/// Maximum number of characters of stderr included in a failure log line.
const STDERR_SNIPPET_CHARS: usize = 512;

/// Result of running a child command with bounded output capture.
struct CappedOutput {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
    /// True when stdout exceeded MAX_CAPTURE_BYTES and was truncated.
    stdout_truncated: bool,
}

/// Output of a single capped stream read: the captured (lossy UTF-8) text and
/// whether the source exceeded MAX_CAPTURE_BYTES and was truncated.
struct CappedRead {
    text: String,
    truncated: bool,
}

/// A short, single-line snippet of `text` suitable for a log message.
fn log_snippet(text: &str) -> String {
    let trimmed = text.trim();
    let snippet: String = trimmed.chars().take(STDERR_SNIPPET_CHARS).collect();
    if snippet.len() < trimmed.len() {
        format!("{snippet}...")
    } else {
        snippet
    }
}

/// Read up to `MAX_CAPTURE_BYTES` from `reader`, discarding the rest so a
/// runaway producer cannot stall the worker. Returns the captured text and a
/// flag indicating whether the source exceeded the cap and was truncated.
fn read_capped<R: std::io::Read>(mut reader: R) -> std::io::Result<CappedRead> {
    use std::io::Read;
    let mut buf = Vec::new();
    // Read one byte past the cap so we can distinguish "exactly at the cap"
    // from "larger than the cap and therefore truncated".
    reader
        .by_ref()
        .take(MAX_CAPTURE_BYTES as u64 + 1)
        .read_to_end(&mut buf)?;
    let truncated = buf.len() > MAX_CAPTURE_BYTES;
    buf.truncate(MAX_CAPTURE_BYTES);
    // Drain any remainder so the child is not blocked on a full pipe.
    let drained = std::io::copy(&mut reader, &mut std::io::sink())?;
    Ok(CappedRead {
        text: String::from_utf8_lossy(&buf).into_owned(),
        truncated: truncated || drained > 0,
    })
}

/// The decision an exec-mode worker makes after a command completes.
enum ExecOutcome {
    /// Command succeeded and produced parseable metrics; report via /api/tell.
    Tell(serde_json::Value),
    /// Command should be canceled; carries a human-readable reason.
    Cancel(String),
}

/// Decide what to do with a finished exec-mode command, independent of any
/// network I/O so the decision can be unit-tested directly.
///
/// A zero exit with valid metrics JSON yields `Tell`. A non-zero exit, invalid
/// JSON, or stdout truncated by the capture cap all yield `Cancel` with a
/// distinct reason.
fn decide_exec_outcome(output: &CappedOutput) -> ExecOutcome {
    if !output.status.success() {
        return ExecOutcome::Cancel(format!(
            "command failed (exit {})",
            output.status.code().unwrap_or(-1)
        ));
    }
    if output.stdout_truncated {
        return ExecOutcome::Cancel(format!(
            "exec stdout exceeded the capture limit of {MAX_CAPTURE_BYTES} bytes"
        ));
    }
    match serde_json::from_str::<serde_json::Value>(output.stdout.trim()) {
        Ok(metrics) => ExecOutcome::Tell(metrics),
        Err(_) => ExecOutcome::Cancel("command produced invalid JSON".to_string()),
    }
}

/// Run a child command, capturing stdout/stderr with a per-stream byte cap.
fn run_capped(mut command: std::process::Command) -> std::io::Result<CappedOutput> {
    use std::process::Stdio;
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();

    // Read stderr on a separate thread so a child filling both pipes does
    // not deadlock against us draining stdout.
    let stderr_thread = std::thread::spawn(move || stderr_handle.map(read_capped).transpose());

    // Capture the stdout read result without `?` so that, even on an I/O
    // error, we still join the stderr thread and reap the child below rather
    // than orphaning the thread and leaking a zombie.
    let stdout_result = stdout_handle.map(read_capped).transpose();

    let stderr_result = stderr_thread
        .join()
        .map_err(|_| std::io::Error::other("stderr capture thread panicked"));

    // Always wait on the child so it does not become a zombie. On the stdout
    // error path the child may still be running, so kill it first to avoid
    // blocking on a child that is itself blocked writing to our drained pipe.
    let status_result = if stdout_result.is_err() {
        let _ = child.kill();
        child.wait()
    } else {
        child.wait()
    };

    let stdout = stdout_result?;
    let stderr = stderr_result??;
    let status = status_result?;

    let (stdout_text, stdout_truncated) = stdout.map(|r| (r.text, r.truncated)).unwrap_or_default();
    let stderr_text = stderr.map(|r| r.text).unwrap_or_default();

    Ok(CappedOutput {
        status,
        stdout: stdout_text,
        stderr: stderr_text,
        stdout_truncated,
    })
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
            require_read_auth,
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
            options.require_read_auth = require_read_auth;

            hola::server::serve_with_options(engine, options).await?;
        }
        Commands::Worker {
            server,
            exec,
            mode,
            token,
        } => {
            let exec_mode = matches!(mode, WorkerMode::Exec);
            let token = configured_token(token);

            // Validate and normalize the server URL up front so a scheme-less
            // or typo'd value fails fast rather than entering the retry loop,
            // and so the joined "{server}/api/..." paths never produce a
            // double slash.
            let server = normalize_server_url(&server)?;

            let mode_label = if exec_mode { "exec" } else { "callback" };
            eprintln!("Worker connecting to {server} ({mode_label} mode)...");
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

                        let trial: serde_json::Value = match r.json().await {
                            Ok(trial) => trial,
                            Err(e) => {
                                eprintln!("Failed to parse trial response: {e}. Retrying in 5s...");
                                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                                continue;
                            }
                        };
                        let params = trial.get("params").cloned().unwrap_or_default();
                        let trial_id = match trial.get("trial_id").and_then(|v| v.as_u64()) {
                            Some(trial_id) => trial_id,
                            None => {
                                eprintln!(
                                    "Trial response missing a valid trial_id, skipping. Retrying in 5s..."
                                );
                                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                                continue;
                            }
                        };

                        if exec_mode {
                            // Exec mode: run command, parse stdout as
                            // JSON metrics, report on the script's behalf.
                            let mut command = std::process::Command::new("sh");
                            command
                                .arg("-c")
                                .arg(&exec)
                                .env("HOLA_PARAMS", params.to_string());

                            match run_capped(command) {
                                Ok(output) => match decide_exec_outcome(&output) {
                                    ExecOutcome::Tell(metrics) => {
                                        let tell_resp = client.post(format!("{server}/api/tell"));
                                        let tell_resp =
                                            with_bearer_auth(tell_resp, token.as_deref())
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
                                    }
                                    ExecOutcome::Cancel(reason) => {
                                        // Cancel rather than reporting a fake result,
                                        // mirroring callback mode's failure path.
                                        eprintln!(
                                            "Trial {trial_id}: {reason}, canceling. stderr: {}",
                                            log_snippet(&output.stderr)
                                        );
                                        let _ = with_bearer_auth(
                                            client.post(format!("{server}/api/cancel")),
                                            token.as_deref(),
                                        )
                                        .json(&serde_json::json!({"trial_id": trial_id}))
                                        .send()
                                        .await;
                                    }
                                },
                                Err(e) => {
                                    eprintln!(
                                        "Trial {trial_id}: failed to run command ({e}), canceling"
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
                            let status = match command.status() {
                                Ok(status) => status,
                                Err(e) => {
                                    eprintln!(
                                        "Trial {trial_id}: failed to run command ({e}), canceling"
                                    );
                                    let _ = with_bearer_auth(
                                        client.post(format!("{server}/api/cancel")),
                                        token.as_deref(),
                                    )
                                    .json(&serde_json::json!({"trial_id": trial_id}))
                                    .send()
                                    .await;
                                    continue;
                                }
                            };

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// A reader that yields `total` bytes of 'a' without ever blocking, used to
    /// exercise the capture cap. Tracks how many bytes were actually consumed
    /// so a test can confirm the reader was fully drained.
    struct CountingReader {
        remaining: usize,
        consumed: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl Read for CountingReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.remaining == 0 {
                return Ok(0);
            }
            let n = buf.len().min(self.remaining);
            for b in &mut buf[..n] {
                *b = b'a';
            }
            self.remaining -= n;
            self.consumed.set(self.consumed.get() + n);
            Ok(n)
        }
    }

    #[test]
    fn read_capped_detects_truncation_and_drains() {
        let total = MAX_CAPTURE_BYTES + 4096;
        let consumed = std::rc::Rc::new(std::cell::Cell::new(0));
        let reader = CountingReader {
            remaining: total,
            consumed: consumed.clone(),
        };

        let result = read_capped(reader).expect("read_capped should succeed");
        assert!(result.truncated, "oversized input should be flagged");
        assert_eq!(
            result.text.len(),
            MAX_CAPTURE_BYTES,
            "captured text is bounded by the cap"
        );
        assert_eq!(
            consumed.get(),
            total,
            "the reader must be fully drained even past the cap"
        );
    }

    #[test]
    fn read_capped_no_truncation_when_under_cap() {
        let reader = std::io::Cursor::new(b"hello".to_vec());
        let result = read_capped(reader).expect("read_capped should succeed");
        assert!(!result.truncated);
        assert_eq!(result.text, "hello");
    }

    #[test]
    fn run_capped_floods_both_pipes_without_deadlock() {
        // Emit far more than the cap on both stdout and stderr. Without the
        // separate stderr-draining thread this would deadlock; both streams
        // must come back bounded by the cap.
        let bytes = MAX_CAPTURE_BYTES + (1 << 16);
        let script = format!(
            "head -c {bytes} /dev/zero | tr '\\0' a; \
             head -c {bytes} /dev/zero | tr '\\0' b 1>&2"
        );
        let mut command = std::process::Command::new("sh");
        command.arg("-c").arg(&script);

        let output = run_capped(command).expect("run_capped should not deadlock");
        assert!(output.status.success());
        assert_eq!(output.stdout.len(), MAX_CAPTURE_BYTES);
        assert_eq!(output.stderr.len(), MAX_CAPTURE_BYTES);
        assert!(output.stdout_truncated);
    }

    /// Run a tiny shell command through run_capped to obtain a real
    /// CappedOutput with a process-supplied ExitStatus for decision testing.
    fn capped_from_sh(script: &str) -> CappedOutput {
        let mut command = std::process::Command::new("sh");
        command.arg("-c").arg(script);
        run_capped(command).expect("command should run")
    }

    #[test]
    fn normalize_server_url_strips_trailing_slash() {
        let normalized = normalize_server_url("http://localhost:8000/")
            .expect("valid http URL should normalize");
        assert_eq!(normalized, "http://localhost:8000");
    }

    #[test]
    fn normalize_server_url_keeps_clean_url_unchanged() {
        let normalized = normalize_server_url("https://example.com:8000")
            .expect("valid https URL should normalize");
        assert_eq!(normalized, "https://example.com:8000");
    }

    #[test]
    fn normalize_server_url_rejects_scheme_less() {
        let err =
            normalize_server_url("localhost:8000").expect_err("a scheme-less URL must be rejected");
        // url parses "localhost:8000" with scheme "localhost", which we reject.
        assert!(err.to_string().contains("invalid --server URL"));
    }

    #[test]
    fn normalize_server_url_rejects_non_http_scheme() {
        let err = normalize_server_url("ftp://example.com")
            .expect_err("a non-http(s) scheme must be rejected");
        assert!(err.to_string().contains("unsupported scheme"));
    }

    #[test]
    fn normalize_server_url_rejects_garbage() {
        let err = normalize_server_url("not a url").expect_err("garbage must be rejected");
        assert!(err.to_string().contains("invalid --server URL"));
    }

    #[test]
    fn decide_exec_outcome_zero_exit_valid_json_tells() {
        let output = capped_from_sh("printf '{\"loss\": 1.5}'");
        match decide_exec_outcome(&output) {
            ExecOutcome::Tell(metrics) => {
                assert_eq!(metrics["loss"], serde_json::json!(1.5));
            }
            ExecOutcome::Cancel(reason) => panic!("expected Tell, got Cancel: {reason}"),
        }
    }

    #[test]
    fn decide_exec_outcome_nonzero_exit_cancels() {
        let output = capped_from_sh("printf '{\"loss\": 1.5}'; exit 3");
        match decide_exec_outcome(&output) {
            ExecOutcome::Cancel(reason) => assert!(reason.contains("exit 3")),
            ExecOutcome::Tell(_) => panic!("expected Cancel on non-zero exit"),
        }
    }

    #[test]
    fn decide_exec_outcome_invalid_json_cancels() {
        let output = capped_from_sh("printf 'not json'");
        match decide_exec_outcome(&output) {
            ExecOutcome::Cancel(reason) => assert!(reason.contains("invalid JSON")),
            ExecOutcome::Tell(_) => panic!("expected Cancel on invalid JSON"),
        }
    }

    #[test]
    fn decide_exec_outcome_truncated_stdout_cancels_distinctly() {
        // A successful command whose stdout exceeds the cap must cancel with a
        // truncation-specific reason rather than the generic parse-failed path.
        let bytes = MAX_CAPTURE_BYTES + 4096;
        let output = capped_from_sh(&format!("head -c {bytes} /dev/zero | tr '\\0' a"));
        assert!(output.status.success());
        assert!(output.stdout_truncated);
        match decide_exec_outcome(&output) {
            ExecOutcome::Cancel(reason) => {
                assert!(
                    reason.contains("capture limit"),
                    "reason should be the truncation diagnostic, got: {reason}"
                );
            }
            ExecOutcome::Tell(_) => panic!("expected Cancel on truncated stdout"),
        }
    }
}
