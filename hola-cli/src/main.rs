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
use command_group::{CommandGroup, GroupChild};
use hola::hola_engine::{HolaEngine, StudyConfig};
use hola::server::ServerOptions;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use uuid::Uuid;

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
        /// Bearer token required for write-capable API endpoints. A value given
        /// on the command line is visible to other local users in the process
        /// list; prefer --auth-token-file or the HOLA_API_TOKEN environment variable.
        #[arg(long, conflicts_with = "auth_token_file")]
        auth_token: Option<String>,
        /// Read the write-capable bearer token from this file. One trailing
        /// newline is ignored.
        #[arg(long, value_name = "PATH")]
        auth_token_file: Option<PathBuf>,
        /// Optional read-only bearer token for dashboards and monitoring. A value
        /// given on the command line is visible to other local users in the
        /// process list; prefer --read-token-file or the HOLA_READ_TOKEN
        /// environment variable.
        #[arg(long, conflicts_with = "read_token_file")]
        read_token: Option<String>,
        /// Read the read-only bearer token from this file. One trailing newline
        /// is ignored.
        #[arg(long, value_name = "PATH")]
        read_token_file: Option<PathBuf>,
        /// Directory where dashboard/API checkpoint saves are allowed.
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        /// Allowed CORS origin. May be provided multiple times.
        #[arg(long = "cors-origin")]
        cors_origins: Vec<String>,
        /// Leave read-only endpoints and the SSE stream open when an API token
        /// is configured. By default the token protects both reads and writes.
        #[arg(long)]
        allow_unauthenticated_reads: bool,
        /// Trial lease duration in seconds. Workers must complete, cancel, or
        /// heartbeat a trial before this deadline.
        #[arg(long, default_value_t = 7200, value_parser = clap::value_parser!(u64).range(1..))]
        lease_seconds: u64,
    },
    /// Run a worker that polls the server for trials.
    ///
    /// In "callback" mode (the default), the worker sets HOLA_SERVER,
    /// HOLA_TRIAL_ID, and HOLA_PARAMS environment variables, then runs
    /// your --exec command. The command is responsible for calling
    /// POST /api/tell to report results. If the command exits with
    /// non-zero status, the worker cancels the trial only if the server
    /// still reports it as pending; a completed tell is authoritative.
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
        /// Bearer token for a token-protected server. A value given on the
        /// command line is visible to other local users in the process list;
        /// prefer --token-file or the HOLA_API_TOKEN environment variable.
        #[arg(long, conflicts_with = "token_file")]
        token: Option<String>,
        /// Read the bearer token from this file. One trailing newline is ignored.
        #[arg(long, value_name = "PATH")]
        token_file: Option<PathBuf>,
        /// Allow sending the bearer token to a non-loopback server over
        /// unencrypted http://. Without this flag (or HOLA_ALLOW_INSECURE_HTTP=1)
        /// the worker refuses, because anyone on the network path could read the
        /// token. Prefer an https:// URL served by a TLS-terminating proxy.
        #[arg(long)]
        allow_insecure_http: bool,
        /// Maximum duration of one HTTP request, in seconds.
        #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..))]
        request_timeout: u64,
        /// Maximum duration of one executed command, in seconds.
        #[arg(long, default_value_t = 3600, value_parser = clap::value_parser!(u64).range(1..))]
        command_timeout: u64,
        /// Durable exec-mode result queue. A server-specific subdirectory is
        /// created here and retried before the worker asks for more work.
        #[arg(long, default_value = ".hola-worker-outbox")]
        outbox_dir: PathBuf,
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

/// True for `localhost` and loopback IP literals. Accepts the bracketed IPv6
/// form that URLs use (`[::1]`).
fn is_local_host(host: &str) -> bool {
    let host = host
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .unwrap_or(host);
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<IpAddr>().is_ok_and(|ip| match ip {
        IpAddr::V4(ip) => ip.is_loopback(),
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip
                    .to_ipv4_mapped()
                    .is_some_and(|mapped| mapped.is_loopback())
        }
    })
}

/// Validate and normalize a server base URL.
///
/// Requires an http/https scheme and a host, and strips any trailing slash so
/// that paths joined as "{server}/api/..." never produce a double slash. A
/// scheme-less or otherwise malformed value is rejected with a clear error so
/// the worker fails fast instead of retrying forever against a bad address.
fn normalize_server_url(server: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut url =
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
    if !url.username().is_empty() || url.password().is_some() {
        return Err(format!(
            "invalid --server URL '{server}': embedded userinfo is not supported; use --token for authentication"
        )
        .into());
    }
    if url.query().is_some() {
        return Err(format!(
            "invalid --server URL '{server}': query parameters are not allowed in a server base URL"
        )
        .into());
    }
    if url.fragment().is_some() {
        return Err(format!(
            "invalid --server URL '{server}': fragments are not allowed in a server base URL"
        )
        .into());
    }

    // Preserve reverse-proxy path prefixes, but canonicalize them to exactly
    // one endpoint join boundary. Returning the parsed URL (rather than the raw
    // input) prevents a query/fragment from swallowing a later `/api/...` suffix.
    let normalized_path = url.path().trim_end_matches('/').to_string();
    url.set_path(&normalized_path);
    Ok(url.to_string().trim_end_matches('/').to_string())
}

const API_TOKEN_ENV: &str = "HOLA_API_TOKEN";
const READ_TOKEN_ENV: &str = "HOLA_READ_TOKEN";
const ALLOW_INSECURE_HTTP_ENV: &str = "HOLA_ALLOW_INSECURE_HTTP";
const MAX_TOKEN_FILE_BYTES: u64 = 16 << 10;

/// Read a bearer token from a file so it never appears in a process listing.
/// One trailing newline (LF or CRLF) is ignored; everything else is the token.
fn read_token_file(path: &Path, flag: &str) -> Result<String, String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("{flag}: cannot open '{}': {error}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(metadata) = file.metadata() {
            let mode = metadata.mode() & 0o777;
            if mode & 0o077 != 0 {
                eprintln!(
                    "warning: token file '{}' is accessible to other users (mode {mode:03o}); restrict it with chmod 600",
                    path.display()
                );
            }
        }
    }
    let mut bytes = Vec::new();
    file.take(MAX_TOKEN_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{flag}: cannot read '{}': {error}", path.display()))?;
    if bytes.len() as u64 > MAX_TOKEN_FILE_BYTES {
        return Err(format!(
            "{flag}: '{}' is larger than {MAX_TOKEN_FILE_BYTES} bytes, which is not a token file",
            path.display()
        ));
    }
    let text = String::from_utf8(bytes)
        .map_err(|_| format!("{flag}: '{}' is not valid UTF-8", path.display()))?;
    let token = text
        .strip_suffix("\r\n")
        .or_else(|| text.strip_suffix('\n'))
        .unwrap_or(&text);
    if token.is_empty() {
        return Err(format!("{flag}: '{}' is empty", path.display()));
    }
    if token.chars().any(char::is_control) {
        return Err(format!(
            "{flag}: '{}' must hold a single-line token without control characters",
            path.display()
        ));
    }
    Ok(token.to_string())
}

/// Resolve one bearer token. An explicit command-line value wins, then the
/// token file, then the environment variable's value. Empty values count as
/// unset.
fn resolve_token(
    cli_token: Option<String>,
    token_file: Option<&Path>,
    file_flag: &str,
    env_value: Option<String>,
) -> Result<Option<String>, String> {
    if let Some(token) = cli_token {
        return Ok(Some(token).filter(|token| !token.is_empty()));
    }
    if let Some(path) = token_file {
        return read_token_file(path, file_flag).map(Some);
    }
    Ok(env_value.filter(|token| !token.is_empty()))
}

fn insecure_http_allowed_by_env() -> bool {
    std::env::var(ALLOW_INSECURE_HTTP_ENV)
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// Decide whether a bearer token may be sent to `server`.
///
/// The HOLA server speaks plain HTTP. A token sent to a non-loopback `http://`
/// address crosses the network unencrypted, and whoever observes it gains the
/// token's full API access, so that is refused unless the operator opted in.
/// Returns a warning to print when the opt-in is what allowed the connection.
fn check_token_transport(
    server: &str,
    has_token: bool,
    allow_insecure_http: bool,
) -> Result<Option<String>, String> {
    if !has_token {
        return Ok(None);
    }
    let url = reqwest::Url::parse(server).map_err(|error| error.to_string())?;
    if url.scheme() != "http" || url.host_str().is_some_and(is_local_host) {
        return Ok(None);
    }
    if allow_insecure_http {
        return Ok(Some(format!(
            "warning: sending the bearer token to {server} over unencrypted HTTP; anyone on the network path can read it"
        )));
    }
    Err(format!(
        "refusing to send the bearer token to {server} over unencrypted HTTP: anyone on the \
         network path could read it and use it against the server. The HOLA server does not \
         speak TLS itself, so put a TLS-terminating reverse proxy in front of it and use its \
         https:// URL. On a network you trust, pass --allow-insecure-http or set \
         {ALLOW_INSECURE_HTTP_ENV}=1 to send it anyway."
    ))
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

const RETRY_DELAY: Duration = Duration::from_secs(5);
const COMMAND_FAILURE_DELAY: Duration = Duration::from_secs(1);
const HTTP_ERROR_SNIPPET_CHARS: usize = 4096;
const MAX_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
const HEARTBEAT_RETRY_INTERVAL: Duration = Duration::from_secs(1);

/// Largest successful response body the worker accepts. Every response it
/// parses is an acknowledgement, one trial, or a lifecycle status; the server
/// caps request bodies at 64 KiB, so 1 MiB is far above any legitimate reply.
const MAX_RESPONSE_BODY_BYTES: usize = 1 << 20;
/// Bytes of an error response that are read at all. Only a short excerpt is
/// shown, so the remainder of an oversized body is never downloaded.
const MAX_ERROR_BODY_BYTES: usize = 64 << 10;
/// Number of post-commit warnings echoed for one tell, and the length of each.
const MAX_POST_COMMIT_WARNINGS: usize = 16;
const POST_COMMIT_WARNING_CHARS: usize = 512;
const REQUEST_ERROR_CHARS: usize = 1024;

fn build_http_client(timeout: Duration) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .connect_timeout(timeout.min(Duration::from_secs(10)))
        .timeout(timeout)
        // Never follow a redirect. A followed redirect would replay the request
        // (and, on the same host, the bearer token) at an address the server
        // chose, and could turn an https:// connection into plain http://.
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

/// Make untrusted text safe to print on a terminal or in a log.
///
/// Control characters (C0, DEL and C1, which include ESC, CR and LF) and the
/// Unicode line-separator and bidirectional-override characters are replaced
/// with visible escapes, so a server cannot forge extra log lines, hide text,
/// or drive the terminal. The result holds at most `max_chars` characters of
/// the input, followed by `...` when it was cut.
fn sanitize_for_terminal(text: &str, max_chars: usize) -> String {
    let mut safe = String::with_capacity(text.len().min(max_chars.saturating_mul(2)));
    for (index, character) in text.chars().enumerate() {
        if index == max_chars {
            safe.push_str("...");
            break;
        }
        match character {
            '\n' => safe.push_str("\\n"),
            '\r' => safe.push_str("\\r"),
            '\t' => safe.push_str("\\t"),
            character
                if character.is_control()
                    || matches!(
                        character,
                        '\u{2028}' | '\u{2029}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'
                    ) =>
            {
                let _ = write!(safe, "\\u{{{:04x}}}", u32::from(character));
            }
            character => safe.push(character),
        }
    }
    safe
}

/// Read a response body, failing as soon as it is known to exceed `limit`.
///
/// A declared `Content-Length` above the limit is rejected before any body
/// byte is read, and the bytes actually received are counted as well because
/// a chunked or mislabelled body declares nothing reliable. The client is
/// built without response decompression, so received bytes are the body size.
async fn read_body_capped(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, String> {
    if let Some(declared) = response.content_length() {
        if declared > limit as u64 {
            return Err(format!(
                "response body of {declared} bytes exceeds the {limit}-byte limit"
            ));
        }
    }
    let mut body = Vec::new();
    loop {
        let chunk = response.chunk().await.map_err(|error| {
            format!(
                "failed to read response body: {}",
                sanitize_for_terminal(&error.to_string(), REQUEST_ERROR_CHARS)
            )
        })?;
        let Some(chunk) = chunk else {
            return Ok(body);
        };
        if chunk.len() > limit - body.len() {
            return Err(format!("response body exceeds the {limit}-byte limit"));
        }
        body.extend_from_slice(&chunk);
    }
}

/// Parse a bounded successful response as JSON.
async fn read_json_capped(response: reqwest::Response) -> Result<serde_json::Value, String> {
    let body = read_body_capped(response, MAX_RESPONSE_BODY_BYTES).await?;
    serde_json::from_slice(&body)
        .map_err(|error| sanitize_for_terminal(&error.to_string(), REQUEST_ERROR_CHARS))
}

/// Read at most `MAX_ERROR_BODY_BYTES` of an error body and stop. An error
/// body is only quoted in a diagnostic, so an oversized one is cut, not failed.
async fn read_error_body(mut response: reqwest::Response) -> String {
    let mut body = Vec::new();
    while body.len() < MAX_ERROR_BODY_BYTES {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                let room = MAX_ERROR_BODY_BYTES - body.len();
                body.extend_from_slice(&chunk[..chunk.len().min(room)]);
            }
            Ok(None) => break,
            Err(error) if body.is_empty() => {
                return format!("unable to read error response: {error}");
            }
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&body).into_owned()
}

/// A non-success response: its status, the bounded raw body for protocol
/// classification, and a terminal-safe message for the operator.
struct HttpFailure {
    status: reqwest::StatusCode,
    body: String,
    message: String,
}

async fn http_failure(operation: &str, response: reqwest::Response) -> HttpFailure {
    let status = response.status();
    let location = response
        .headers()
        .get(reqwest::header::LOCATION)
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned());
    let body = read_error_body(response).await;
    let message = if status.is_redirection() {
        format_redirect_error(operation, status, location.as_deref())
    } else {
        format_http_error(operation, status, &body)
    };
    HttpFailure {
        status,
        body,
        message,
    }
}

async fn http_error(operation: &str, response: reqwest::Response) -> String {
    http_failure(operation, response).await.message
}

fn format_http_error(operation: &str, status: reqwest::StatusCode, body: &str) -> String {
    let detail = sanitize_for_terminal(body.trim(), HTTP_ERROR_SNIPPET_CHARS);
    let detail = if detail.is_empty() {
        "empty response body"
    } else {
        &detail
    };
    format!("{operation} returned HTTP {status}: {detail}")
}

fn format_redirect_error(
    operation: &str,
    status: reqwest::StatusCode,
    location: Option<&str>,
) -> String {
    let target = location.map_or_else(String::new, |location| {
        format!(" to '{}'", sanitize_for_terminal(location, 256))
    });
    format!(
        "{operation} was redirected (HTTP {status}{target}); HOLA does not follow redirects, so point --server at the final URL"
    )
}

fn request_error(operation: &str, error: reqwest::Error) -> String {
    let detail = sanitize_for_terminal(&error.to_string(), REQUEST_ERROR_CHARS);
    if error.is_timeout() {
        format!("{operation} request timed out: {detail}")
    } else if error.is_connect() {
        format!("{operation} connection failed: {detail}")
    } else {
        format!("{operation} request failed: {detail}")
    }
}

async fn send_checked(
    operation: &str,
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, String> {
    let response = request
        .send()
        .await
        .map_err(|error| request_error(operation, error))?;
    if response.status().is_success() {
        Ok(response)
    } else {
        Err(http_error(operation, response).await)
    }
}

fn retryable_http_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
}

const OUTBOX_FORMAT_VERSION: u32 = 1;
const MAX_OUTBOX_RECORD_BYTES: u64 = (MAX_CAPTURE_BYTES as u64) + (64 << 10);
static OUTBOX_TEMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct PendingTell {
    format_version: u32,
    server: String,
    trial_id: u64,
    metrics: serde_json::Value,
}

impl PendingTell {
    fn new(server: &str, trial_id: u64, metrics: serde_json::Value) -> Self {
        Self {
            format_version: OUTBOX_FORMAT_VERSION,
            server: server.to_string(),
            trial_id,
            metrics,
        }
    }
}

/// Server-scoped, crash-safe queue for exec-mode tells.
///
/// Every record in the queue is later sent with the worker's bearer token, so
/// the queue must not be writable by anyone else. Both directories are created
/// private and an existing one is used only if [`validate_private_directory`]
/// accepts it; the check is repeated before each use of the queue.
struct TellOutbox {
    root: PathBuf,
    directory: PathBuf,
    server: String,
}

impl TellOutbox {
    fn open(root: &std::path::Path, server: &str) -> std::io::Result<Self> {
        let directory = root.join(format!("{:016x}", stable_server_hash(server)));
        ensure_private_directory(root)?;
        ensure_private_directory(&directory)?;
        Ok(Self {
            root: root.to_path_buf(),
            directory,
            server: server.to_string(),
        })
    }

    /// Confirm that the queue is still the private directory that was opened.
    fn validate(&self) -> std::io::Result<()> {
        validate_private_directory(&self.root)?;
        validate_private_directory(&self.directory)
    }

    fn record_path(&self, trial_id: u64) -> PathBuf {
        self.directory.join(format!("tell-{trial_id}.json"))
    }

    fn read_record(&self, path: &std::path::Path) -> std::io::Result<PendingTell> {
        let not_regular = || {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("outbox entry '{}' is not a regular file", path.display()),
            )
        };
        // Open first and then inspect the handle: a path that was checked and
        // then reopened could be replaced in between. The open never follows a
        // symbolic link and never blocks on a FIFO.
        let file = match open_outbox_record(path) {
            Ok(file) => file,
            Err(error) => {
                let irregular = std::fs::symlink_metadata(path)
                    .is_ok_and(|metadata| !metadata.file_type().is_file());
                return Err(if irregular { not_regular() } else { error });
            }
        };
        let metadata = file.metadata()?;
        if !metadata.file_type().is_file() {
            return Err(not_regular());
        }
        let too_large = || {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("outbox entry '{}' exceeds the size limit", path.display()),
            )
        };
        if metadata.len() > MAX_OUTBOX_RECORD_BYTES {
            return Err(too_large());
        }
        let mut bytes = Vec::new();
        file.take(MAX_OUTBOX_RECORD_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_OUTBOX_RECORD_BYTES {
            return Err(too_large());
        }
        let record: PendingTell = serde_json::from_slice(&bytes).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid outbox entry '{}': {error}", path.display()),
            )
        })?;
        self.validate_record(path, &record)?;
        Ok(record)
    }

    fn validate_record(&self, path: &std::path::Path, record: &PendingTell) -> std::io::Result<()> {
        if record.format_version != OUTBOX_FORMAT_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unsupported outbox format {} in '{}'",
                    record.format_version,
                    path.display()
                ),
            ));
        }
        if record.server != self.server {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "outbox entry '{}' belongs to another server",
                    path.display()
                ),
            ));
        }
        Ok(())
    }

    /// Persist a tell before any network attempt. Re-persisting an identical
    /// trial is harmless; different metrics for the same id are rejected.
    fn persist(&self, record: &PendingTell) -> std::io::Result<PathBuf> {
        self.validate()?;
        self.validate_record(&self.record_path(record.trial_id), record)?;
        let path = self.record_path(record.trial_id);
        if outbox_entry_exists(&path)? {
            let existing = self.read_record(&path)?;
            if existing == *record {
                return Ok(path);
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "outbox already contains different metrics for trial {}",
                    record.trial_id
                ),
            ));
        }

        let bytes = serde_json::to_vec(record).map_err(std::io::Error::other)?;
        if bytes.len() as u64 > MAX_OUTBOX_RECORD_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tell result exceeds the outbox record size limit",
            ));
        }
        let sequence = OUTBOX_TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let temp_path = self.directory.join(format!(
            ".tell-{}-{}-{sequence}.tmp",
            record.trial_id,
            std::process::id()
        ));

        let write_result = (|| -> std::io::Result<()> {
            let mut options = std::fs::OpenOptions::new();
            options.create_new(true).write(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&temp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            // Publish without replacing an existing same-trial record. A hard
            // link is atomic and, because temp and destination share a directory,
            // cannot cross filesystems.
            std::fs::hard_link(&temp_path, &path)?;
            std::fs::remove_file(&temp_path)?;
            sync_directory(&self.directory)?;
            Ok(())
        })();

        if let Err(error) = write_result {
            let _ = std::fs::remove_file(&temp_path);
            // A second worker may have won the same-id race. Accept it only if
            // its durable record is byte-for-byte equivalent after parsing.
            if self.read_record(&path).is_ok_and(|saved| saved == *record) {
                return Ok(path);
            }
            return Err(error);
        }
        Ok(path)
    }

    fn pending(&self) -> std::io::Result<Vec<(PathBuf, PendingTell)>> {
        self.validate()?;
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(&self.directory)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("tell-") && name.ends_with(".json") {
                paths.push(entry.path());
            }
        }
        paths.sort();
        paths
            .into_iter()
            .map(|path| self.read_record(&path).map(|record| (path, record)))
            .collect()
    }

    fn remove(&self, path: &std::path::Path) -> std::io::Result<()> {
        match std::fs::remove_file(path) {
            Ok(()) => sync_directory(&self.directory),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

fn stable_server_hash(server: &str) -> u64 {
    // FNV-1a is sufficient here: the full server URL is also stored and checked
    // inside every record, so a hash collision cannot misdeliver a result.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in server.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn create_private_directory(path: &std::path::Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

/// Create `path` as a private directory if it is missing, then require that
/// whatever is there now is trustworthy. A directory that already existed is
/// never trusted merely because it has the expected name.
fn ensure_private_directory(path: &std::path::Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            create_private_directory(path)?;
        }
        Err(error) => return Err(error),
    }
    validate_private_directory(path)
}

/// Why a directory with this owner and mode cannot hold the outbox, if it
/// cannot. Only its owner (or root) may be able to add, replace or remove
/// entries: a record placed there by anyone else would be sent to the server
/// under this worker's bearer token.
#[cfg(unix)]
fn directory_trust_problem(owner: u32, mode: u32, current_user: u32) -> Option<String> {
    if owner != current_user && owner != 0 {
        return Some(format!(
            "is owned by uid {owner}, not by the user running this worker (uid {current_user})"
        ));
    }
    if mode & 0o022 != 0 {
        return Some(format!(
            "is writable by its group or by other users (mode {:03o}); run chmod go-w on it",
            mode & 0o777
        ));
    }
    None
}

/// Require an outbox directory that nobody else can redirect or write to.
fn validate_private_directory(path: &std::path::Path) -> std::io::Result<()> {
    let refuse = |problem: String| {
        std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "outbox directory '{}' {problem}; choose a private directory with --outbox-dir",
                path.display()
            ),
        )
    };
    // lstat, not stat: a symbolic link planted under the expected name would
    // otherwise point the queue at a directory of someone else's choosing.
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(refuse("is a symbolic link".to_string()));
    }
    if !metadata.is_dir() {
        return Err(refuse("is not a directory".to_string()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // SAFETY: geteuid has no preconditions and cannot fail.
        let current_user = unsafe { libc::geteuid() };
        if let Some(problem) =
            directory_trust_problem(metadata.uid(), metadata.mode(), current_user)
        {
            return Err(refuse(problem));
        }
    }
    Ok(())
}

/// Open an outbox record for reading without following a symbolic link at the
/// final path component and without blocking if the entry is a FIFO.
#[cfg(unix)]
fn open_outbox_record(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
}

#[cfg(not(unix))]
fn open_outbox_record(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    if !std::fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("outbox entry '{}' is not a regular file", path.display()),
        ));
    }
    std::fs::File::open(path)
}

/// Whether a directory entry exists at `path`, without following a link.
fn outbox_entry_exists(path: &std::path::Path) -> std::io::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(unix)]
fn sync_directory(path: &std::path::Path) -> std::io::Result<()> {
    match std::fs::File::open(path)?.sync_all() {
        Ok(()) => Ok(()),
        // Some Unix filesystems do not support fsync on directory handles. The
        // record file itself was already synced, so retain the strongest
        // durability the platform exposes instead of making the outbox unusable.
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::InvalidInput | std::io::ErrorKind::Unsupported
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

#[cfg(not(unix))]
fn sync_directory(_path: &std::path::Path) -> std::io::Result<()> {
    Ok(())
}

#[derive(Debug)]
enum OutboxFlushError {
    /// The record remains durable and a later attempt may succeed.
    Retryable(String),
    /// The server/protocol definitively rejected the record. Retrying the same
    /// bytes forever cannot make progress, so the worker exits and leaves the
    /// record for operator reconciliation.
    Permanent(String),
}

impl OutboxFlushError {
    fn is_permanent(&self) -> bool {
        matches!(self, Self::Permanent(_))
    }
}

impl std::fmt::Display for OutboxFlushError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Retryable(message) | Self::Permanent(message) => formatter.write_str(message),
        }
    }
}

async fn deliver_tell(
    client: &reqwest::Client,
    token: Option<&str>,
    record: &PendingTell,
) -> Result<Vec<String>, OutboxFlushError> {
    let operation = format!("tell for trial {}", record.trial_id);
    let response = with_bearer_auth(client.post(format!("{}/api/tell", record.server)), token)
        .json(&serde_json::json!({
            "trial_id": record.trial_id,
            "metrics": &record.metrics,
        }))
        .send()
        .await
        .map_err(|error| OutboxFlushError::Retryable(request_error(&operation, error)))?;
    if !response.status().is_success() {
        let status = response.status();
        let message = http_error(&operation, response).await;
        return Err(if retryable_http_status(status) {
            OutboxFlushError::Retryable(message)
        } else {
            OutboxFlushError::Permanent(message)
        });
    }

    // A 2xx status alone is not enough to delete an expensive durable result:
    // validate the canonical acknowledgement and ensure it names this trial.
    let acknowledgement = read_json_capped(response).await.map_err(|error| {
        OutboxFlushError::Permanent(format!(
            "{operation} returned invalid acknowledgement JSON: {error}"
        ))
    })?;
    if acknowledgement
        .get("status")
        .and_then(|value| value.as_str())
        != Some("ok")
    {
        return Err(OutboxFlushError::Permanent(format!(
            "{operation} acknowledgement is missing status 'ok'"
        )));
    }
    let acknowledged_trial_id = acknowledgement
        .get("trial")
        .and_then(|trial| trial.get("trial_id"))
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            OutboxFlushError::Permanent(format!(
                "{operation} acknowledgement is missing trial.trial_id"
            ))
        })?;
    if acknowledged_trial_id != record.trial_id {
        return Err(OutboxFlushError::Permanent(format!(
            "{operation} acknowledged trial_id {acknowledged_trial_id} instead"
        )));
    }

    let post_commit_warnings = match acknowledgement.get("post_commit_warnings") {
        None => Vec::new(),
        Some(serde_json::Value::Array(warnings)) => {
            let warnings = warnings
                .iter()
                .map(|warning| {
                    warning.as_str().ok_or_else(|| {
                        OutboxFlushError::Permanent(format!(
                            "{operation} acknowledgement has a non-string post_commit_warnings entry"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            printable_post_commit_warnings(&warnings)
        }
        Some(_) => {
            return Err(OutboxFlushError::Permanent(format!(
                "{operation} acknowledgement has non-array post_commit_warnings"
            )));
        }
    };
    Ok(post_commit_warnings)
}

/// Server-supplied warnings, neutralized for the terminal and bounded in both
/// length and number so one acknowledgement cannot flood the worker's log.
fn printable_post_commit_warnings(warnings: &[&str]) -> Vec<String> {
    let mut printable: Vec<String> = warnings
        .iter()
        .take(MAX_POST_COMMIT_WARNINGS)
        .map(|warning| sanitize_for_terminal(warning, POST_COMMIT_WARNING_CHARS))
        .collect();
    if warnings.len() > MAX_POST_COMMIT_WARNINGS {
        printable.push(format!(
            "{} further post-commit warnings were not shown",
            warnings.len() - MAX_POST_COMMIT_WARNINGS
        ));
    }
    printable
}

async fn flush_outbox(
    outbox: &TellOutbox,
    client: &reqwest::Client,
    token: Option<&str>,
) -> Result<Vec<u64>, OutboxFlushError> {
    flush_outbox_with_warning_sink(outbox, client, token, |trial_id, warning| {
        eprintln!("warning: tell for trial {trial_id} committed: {warning}");
    })
    .await
}

async fn flush_outbox_with_warning_sink<F>(
    outbox: &TellOutbox,
    client: &reqwest::Client,
    token: Option<&str>,
    mut emit_warning: F,
) -> Result<Vec<u64>, OutboxFlushError>
where
    F: FnMut(u64, &str),
{
    let pending = outbox.pending().map_err(|error| {
        OutboxFlushError::Permanent(format!("failed to read tell outbox: {error}"))
    })?;
    let mut delivered = Vec::with_capacity(pending.len());
    for (path, record) in pending {
        let warnings = deliver_tell(client, token, &record).await?;
        for warning in warnings {
            emit_warning(record.trial_id, &warning);
        }
        outbox.remove(&path).map_err(|error| {
            OutboxFlushError::Retryable(format!(
                "tell was accepted but outbox cleanup failed: {error}"
            ))
        })?;
        delivered.push(record.trial_id);
    }
    Ok(delivered)
}

fn ask_request(
    client: &reqwest::Client,
    server: &str,
    token: Option<&str>,
    idempotency_key: &str,
) -> reqwest::RequestBuilder {
    with_bearer_auth(client.post(format!("{server}/api/ask")), token)
        .header("Idempotency-Key", idempotency_key)
}

#[derive(Debug)]
enum CancelTrialError {
    NotPending(String),
    Other(String),
}

impl std::fmt::Display for CancelTrialError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPending(message) | Self::Other(message) => formatter.write_str(message),
        }
    }
}

async fn cancel_trial_classified(
    client: &reqwest::Client,
    server: &str,
    token: Option<&str>,
    trial_id: u64,
) -> Result<(), CancelTrialError> {
    let operation = format!("cancel for trial {trial_id}");
    let response = with_bearer_auth(client.post(format!("{server}/api/cancel")), token)
        .json(&serde_json::json!({"trial_id": trial_id}))
        .send()
        .await
        .map_err(|error| CancelTrialError::Other(request_error(&operation, error)))?;
    if !response.status().is_success() {
        let HttpFailure {
            status,
            body,
            message,
        } = http_failure(&operation, response).await;
        let error_body = serde_json::from_str::<serde_json::Value>(&body).ok();
        let legacy_not_pending =
            format!("Trial {trial_id} is not pending (may be completed or unknown)");
        let canonical_not_pending = status == reqwest::StatusCode::BAD_REQUEST
            && error_body.as_ref().is_some_and(|body| {
                body.get("code").and_then(serde_json::Value::as_str) == Some("cancel_failed")
                    || body.get("error").and_then(serde_json::Value::as_str)
                        == Some(legacy_not_pending.as_str())
            });
        return Err(if canonical_not_pending {
            CancelTrialError::NotPending(message)
        } else {
            CancelTrialError::Other(message)
        });
    }
    let acknowledgement = read_json_capped(response).await.map_err(|error| {
        CancelTrialError::Other(format!(
            "{operation} returned invalid acknowledgement JSON: {error}"
        ))
    })?;
    if acknowledgement
        .get("status")
        .and_then(serde_json::Value::as_str)
        != Some("ok")
    {
        return Err(CancelTrialError::Other(format!(
            "{operation} acknowledgement is missing canonical status 'ok'"
        )));
    }
    if let Some(value) = acknowledgement.get("trial_id") {
        let acknowledged_id = value.as_u64().ok_or_else(|| {
            CancelTrialError::Other(format!(
                "{operation} acknowledgement has a non-integer trial_id"
            ))
        })?;
        if acknowledged_id != trial_id {
            return Err(CancelTrialError::Other(format!(
                "{operation} acknowledged trial_id {acknowledged_id} instead"
            )));
        }
    }
    Ok(())
}

async fn cancel_trial(
    client: &reqwest::Client,
    server: &str,
    token: Option<&str>,
    trial_id: u64,
) -> Result<(), String> {
    cancel_trial_classified(client, server, token, trial_id)
        .await
        .map_err(|error| error.to_string())
}

#[derive(Debug)]
enum HeartbeatError {
    Retryable(String),
    Terminal(String),
    Unsupported(String),
    Rejected(String),
}

impl std::fmt::Display for HeartbeatError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Retryable(message)
            | Self::Terminal(message)
            | Self::Unsupported(message)
            | Self::Rejected(message) => formatter.write_str(message),
        }
    }
}

async fn heartbeat_trial(
    client: &reqwest::Client,
    server: &str,
    token: Option<&str>,
    trial_id: u64,
) -> Result<u64, HeartbeatError> {
    let operation = format!("heartbeat for trial {trial_id}");
    let response = with_bearer_auth(client.post(format!("{server}/api/heartbeat")), token)
        .json(&serde_json::json!({"trial_id": trial_id}))
        .send()
        .await
        .map_err(|error| HeartbeatError::Retryable(request_error(&operation, error)))?;
    if !response.status().is_success() {
        let HttpFailure {
            status,
            body,
            message,
        } = http_failure(&operation, response).await;
        let error_code = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .get("code")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            });
        return Err(if status == reqwest::StatusCode::NOT_FOUND {
            HeartbeatError::Unsupported(message)
        } else if retryable_http_status(status) {
            HeartbeatError::Retryable(message)
        } else if status == reqwest::StatusCode::BAD_REQUEST
            && error_code.as_deref() == Some("heartbeat_failed")
        {
            HeartbeatError::Terminal(message)
        } else {
            HeartbeatError::Rejected(message)
        });
    }
    let body = read_json_capped(response).await.map_err(|error| {
        HeartbeatError::Rejected(format!(
            "{operation} returned invalid acknowledgement JSON: {error}"
        ))
    })?;
    if body.get("status").and_then(serde_json::Value::as_str) != Some("ok") {
        return Err(HeartbeatError::Rejected(format!(
            "{operation} acknowledgement is missing status 'ok'"
        )));
    }
    let acknowledged_id = body
        .get("trial_id")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            HeartbeatError::Rejected(format!("{operation} acknowledgement is missing trial_id"))
        })?;
    if acknowledged_id != trial_id {
        return Err(HeartbeatError::Rejected(format!(
            "{operation} acknowledged trial_id {acknowledged_id} instead"
        )));
    }
    body.get("lease_expires_at_ms")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            HeartbeatError::Rejected(format!(
                "{operation} acknowledgement is missing lease_expires_at_ms"
            ))
        })
}

fn unix_time_millis() -> Result<u64, String> {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| format!("system clock is before the Unix epoch: {error}"))?
        .as_millis();
    u64::try_from(millis).map_err(|_| "system clock exceeds u64 milliseconds".to_string())
}

fn heartbeat_renewal_delay(deadline_ms: u64) -> Result<Duration, String> {
    let remaining_ms = deadline_ms.saturating_sub(unix_time_millis()?);
    if remaining_ms == 0 {
        return Err("server-reported trial lease has already expired".to_string());
    }
    // Renew halfway through short leases and at least every 30 seconds for long
    // leases. The latter leaves multiple retry opportunities during a transient
    // outage instead of waiting until a two-hour default lease is nearly over.
    let delay_ms = (remaining_ms / 2)
        .max(1)
        .min(MAX_HEARTBEAT_INTERVAL.as_millis() as u64);
    Ok(Duration::from_millis(delay_ms))
}

async fn maintain_trial_lease(
    client: &reqwest::Client,
    server: &str,
    token: Option<&str>,
    trial_id: u64,
    mut deadline_ms: u64,
) -> Result<(), String> {
    loop {
        tokio::time::sleep(heartbeat_renewal_delay(deadline_ms)?).await;
        loop {
            let remaining_ms = deadline_ms.saturating_sub(unix_time_millis()?);
            if remaining_ms == 0 {
                return Err(format!(
                    "trial {trial_id} lease expired before a heartbeat was confirmed"
                ));
            }
            let attempt = tokio::time::timeout(
                Duration::from_millis(remaining_ms),
                heartbeat_trial(client, server, token, trial_id),
            )
            .await;
            match attempt {
                Ok(Ok(new_deadline_ms)) => {
                    // Validate now, rather than sleeping zero and spinning if a
                    // malformed/proxied response reports an elapsed deadline.
                    heartbeat_renewal_delay(new_deadline_ms)?;
                    deadline_ms = new_deadline_ms;
                    break;
                }
                Ok(Err(
                    HeartbeatError::Terminal(error)
                    | HeartbeatError::Unsupported(error)
                    | HeartbeatError::Rejected(error),
                )) => return Err(error),
                Ok(Err(HeartbeatError::Retryable(error))) => {
                    let remaining_ms = deadline_ms.saturating_sub(unix_time_millis()?);
                    if remaining_ms == 0 {
                        return Err(format!(
                            "{error}; trial {trial_id} lease expired while retrying heartbeat"
                        ));
                    }
                    eprintln!("{error}. Retrying heartbeat before the lease deadline...");
                    let retry_ms = (remaining_ms / 4)
                        .max(1)
                        .min(HEARTBEAT_RETRY_INTERVAL.as_millis() as u64);
                    tokio::time::sleep(Duration::from_millis(retry_ms)).await;
                }
                Err(_) => {
                    return Err(format!(
                        "trial {trial_id} lease expired while its heartbeat request was pending"
                    ));
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandLeaseMode {
    Strict,
    Callback,
}

struct CommandRun<T> {
    result: std::io::Result<T>,
    callback_completion_confirmed: bool,
}

async fn run_command_with_heartbeat<T, Run>(
    client: &reqwest::Client,
    server: &str,
    token: Option<&str>,
    trial_id: u64,
    lease_mode: CommandLeaseMode,
    run: Run,
) -> CommandRun<T>
where
    T: Send + 'static,
    Run: FnOnce(Arc<AtomicBool>) -> std::io::Result<T> + Send + 'static,
{
    // Confirm renewal before starting expensive work. This both verifies that
    // the endpoint/token are usable and obtains the configured server deadline
    // without hard-coding the server's lease duration in the worker.
    let initial_deadline = match heartbeat_trial(client, server, token, trial_id).await {
        Ok(deadline) => Some(deadline),
        Err(HeartbeatError::Unsupported(error)) => {
            eprintln!(
                "Trial {trial_id}: server has no heartbeat endpoint; running with the command timeout for legacy compatibility ({error})"
            );
            None
        }
        Err(error) => {
            return CommandRun {
                result: Err(std::io::Error::other(format!(
                    "initial lease heartbeat failed: {error}"
                ))),
                callback_completion_confirmed: false,
            };
        }
    };
    if let Some(deadline) = initial_deadline {
        if let Err(error) = heartbeat_renewal_delay(deadline) {
            return CommandRun {
                result: Err(std::io::Error::other(error)),
                callback_completion_confirmed: false,
            };
        }
    }

    let cancellation = Arc::new(AtomicBool::new(false));
    let command_cancellation = Arc::clone(&cancellation);
    let mut command_task = tokio::task::spawn_blocking(move || run(command_cancellation));
    let Some(initial_deadline) = initial_deadline else {
        return CommandRun {
            result: command_task
                .await
                .map_err(|error| {
                    std::io::Error::other(format!("command runner task failed: {error}"))
                })
                .and_then(std::convert::identity),
            callback_completion_confirmed: false,
        };
    };
    let lease_task = maintain_trial_lease(client, server, token, trial_id, initial_deadline);
    tokio::pin!(lease_task);

    tokio::select! {
        biased;
        lease_result = &mut lease_task => {
            let error = lease_result
                .err()
                .unwrap_or_else(|| "lease heartbeat task stopped unexpectedly".to_string());

            if lease_mode == CommandLeaseMode::Callback {
                match callback_trial_state(client, server, token, trial_id).await {
                    Ok(CallbackTrialState::Completed) => {
                        eprintln!(
                            "Trial {trial_id}: callback completion confirmed by server after lease heartbeat stopped; waiting for callback cleanup"
                        );
                        let result = command_task.await.map_err(|join_error| {
                            std::io::Error::other(format!(
                                "command runner task failed: {join_error}"
                            ))
                        }).and_then(std::convert::identity);
                        return CommandRun {
                            result,
                            callback_completion_confirmed: true,
                        };
                    }
                    Ok(CallbackTrialState::Pending | CallbackTrialState::NotPending) => {}
                    Err(verification_error) => {
                        eprintln!(
                            "Trial {trial_id}: lease heartbeat failed ({error}), but callback completion verification was inconclusive ({verification_error}); waiting for the callback before retrying verification"
                        );
                        let result = command_task.await.map_err(|join_error| {
                            std::io::Error::other(format!(
                                "command runner task failed: {join_error}"
                            ))
                        }).and_then(std::convert::identity);
                        return CommandRun {
                            result,
                            callback_completion_confirmed: false,
                        };
                    }
                }
            }

            cancellation.store(true, Ordering::Release);
            // The blocking runner observes cancellation within its 20 ms poll,
            // kills the entire process tree, drains capture pipes, and reaps it.
            let cleanup = command_task.await;
            let cleanup_detail = match cleanup {
                Ok(Ok(_)) => String::new(),
                Ok(Err(cleanup_error)) => format!("; command cleanup: {cleanup_error}"),
                Err(join_error) => format!("; command cleanup task failed: {join_error}"),
            };
            CommandRun {
                result: Err(std::io::Error::other(format!(
                    "lease heartbeat failed: {error}{cleanup_detail}"
                ))),
                callback_completion_confirmed: false,
            }
        }
        command_result = &mut command_task => {
            CommandRun {
                result: command_result.map_err(|error| {
                    std::io::Error::other(format!("command runner task failed: {error}"))
                }).and_then(std::convert::identity),
                callback_completion_confirmed: false,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CallbackTrialState {
    Completed,
    Pending,
    NotPending,
}

async fn callback_trial_state(
    client: &reqwest::Client,
    server: &str,
    token: Option<&str>,
    trial_id: u64,
) -> Result<CallbackTrialState, String> {
    let operation = format!("callback verification for trial {trial_id}");
    let response = with_bearer_auth(
        client.get(format!("{server}/api/trial/{trial_id}/status")),
        token,
    )
    .send()
    .await
    .map_err(|error| request_error(&operation, error))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return legacy_callback_trial_state(client, server, token, trial_id).await;
    }
    if !response.status().is_success() {
        return Err(http_error(&operation, response).await);
    }
    let status = read_json_capped(response)
        .await
        .map_err(|error| format!("{operation} returned invalid JSON: {error}"))?;
    if status.get("status").and_then(serde_json::Value::as_str) != Some("ok") {
        return Err(format!(
            "{operation} response is missing canonical status 'ok'"
        ));
    }
    match status.get("trial_id").and_then(serde_json::Value::as_u64) {
        Some(status_id) if status_id == trial_id => {}
        Some(status_id) => {
            return Err(format!("{operation} returned trial_id {status_id} instead"));
        }
        None => return Err(format!("{operation} response is missing trial_id")),
    }
    match status.get("state").and_then(serde_json::Value::as_str) {
        Some("completed") => Ok(CallbackTrialState::Completed),
        Some("pending") => Ok(CallbackTrialState::Pending),
        Some("not_pending") => Ok(CallbackTrialState::NotPending),
        Some(state) => Err(format!(
            "{operation} returned unsupported trial state '{}'",
            sanitize_for_terminal(state, 64)
        )),
        None => Err(format!("{operation} response is missing trial state")),
    }
}

async fn legacy_callback_trial_state(
    client: &reqwest::Client,
    server: &str,
    token: Option<&str>,
    trial_id: u64,
) -> Result<CallbackTrialState, String> {
    let operation = format!("legacy callback verification for trial {trial_id}");
    let response = with_bearer_auth(
        client.get(format!(
            "{server}/api/trial/{trial_id}?include_infeasible=true"
        )),
        token,
    )
    .send()
    .await
    .map_err(|error| request_error(&operation, error))?;
    if response.status().is_success() {
        let completed = read_json_capped(response)
            .await
            .map_err(|error| format!("{operation} returned invalid JSON: {error}"))?;
        return match completed
            .get("trial_id")
            .and_then(serde_json::Value::as_u64)
        {
            Some(completed_id) if completed_id == trial_id => Ok(CallbackTrialState::Completed),
            Some(completed_id) => Err(format!(
                "{operation} returned trial_id {completed_id} instead"
            )),
            None => Err(format!("{operation} response is missing trial_id")),
        };
    }
    if response.status() != reqwest::StatusCode::NOT_FOUND {
        return Err(http_error(&operation, response).await);
    }

    // Older servers have no lifecycle endpoint, and their ranked trial lookup
    // can return 404 after bounded-leaderboard eviction. A successful heartbeat
    // is therefore the only safe evidence that cancellation is still allowed;
    // any definitive rejection means the id is terminal on that server.
    match heartbeat_trial(client, server, token, trial_id).await {
        Ok(_) => Ok(CallbackTrialState::Pending),
        Err(HeartbeatError::Terminal(_)) => Ok(CallbackTrialState::NotPending),
        Err(HeartbeatError::Unsupported(_)) => Ok(CallbackTrialState::Pending),
        Err(HeartbeatError::Retryable(error)) => Err(format!(
            "{operation} could not distinguish pending from terminal state: {error}"
        )),
        Err(HeartbeatError::Rejected(error)) => Err(format!(
            "{operation} received an untrusted heartbeat rejection: {error}"
        )),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum CallbackDisposition {
    Completed,
    Cancel(String),
    NotPending,
}

async fn callback_disposition(
    client: &reqwest::Client,
    server: &str,
    token: Option<&str>,
    trial_id: u64,
    command_failure: Option<&str>,
) -> Result<CallbackDisposition, String> {
    match callback_trial_state(client, server, token, trial_id).await? {
        CallbackTrialState::Completed => Ok(CallbackDisposition::Completed),
        CallbackTrialState::Pending => {
            Ok(CallbackDisposition::Cancel(command_failure.map_or_else(
                || "script exited successfully without completing its trial".to_string(),
                str::to_string,
            )))
        }
        CallbackTrialState::NotPending => Ok(CallbackDisposition::NotPending),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CallbackCancelOutcome {
    Cancelled,
    Completed,
    NotPending,
}

async fn cancel_callback_trial(
    client: &reqwest::Client,
    server: &str,
    token: Option<&str>,
    trial_id: u64,
) -> Result<CallbackCancelOutcome, String> {
    match cancel_trial_classified(client, server, token, trial_id).await {
        Ok(()) => Ok(CallbackCancelOutcome::Cancelled),
        Err(CancelTrialError::NotPending(_)) => {
            match callback_trial_state(client, server, token, trial_id).await {
                Ok(CallbackTrialState::Completed) => Ok(CallbackCancelOutcome::Completed),
                Ok(CallbackTrialState::Pending | CallbackTrialState::NotPending) | Err(_) => {
                    Ok(CallbackCancelOutcome::NotPending)
                }
            }
        }
        Err(CancelTrialError::Other(cancel_error)) => {
            match callback_trial_state(client, server, token, trial_id).await {
                Ok(CallbackTrialState::Completed) => Ok(CallbackCancelOutcome::Completed),
                Ok(CallbackTrialState::NotPending) => Ok(CallbackCancelOutcome::NotPending),
                Ok(CallbackTrialState::Pending) => Err(format!(
                    "{cancel_error}; trial remains pending after the rejected cancellation"
                )),
                Err(verification_error) => Err(format!(
                    "{cancel_error}; cancellation outcome could not be verified: {verification_error}"
                )),
            }
        }
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
    /// True when the command exceeded its deadline and its process tree was killed.
    timed_out: bool,
}

/// Output of a single capped stream read: the captured (lossy UTF-8) text and
/// whether the source exceeded MAX_CAPTURE_BYTES and was truncated.
struct CappedRead {
    text: String,
    truncated: bool,
}

/// A short, single-line snippet of `text` suitable for a log message. The
/// command's output is not trusted either: it may echo server-chosen
/// parameters, so control characters are escaped like any server text.
fn log_snippet(text: &str) -> String {
    sanitize_for_terminal(text.trim(), STDERR_SNIPPET_CHARS)
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
    if output.timed_out {
        return ExecOutcome::Cancel("command timed out".to_string());
    }
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
        Ok(metrics @ serde_json::Value::Object(_)) => ExecOutcome::Tell(metrics),
        Ok(_) => ExecOutcome::Cancel("command metrics must be a JSON object".to_string()),
        Err(_) => ExecOutcome::Cancel("command produced invalid JSON".to_string()),
    }
}

#[cfg(unix)]
fn shell_command(script: &str) -> Command {
    let mut command = Command::new("sh");
    command.arg("-c").arg(script);
    command
}

#[cfg(windows)]
fn shell_command(script: &str) -> Command {
    use std::os::windows::process::CommandExt;

    let shell = std::env::var_os("COMSPEC").unwrap_or_else(|| "cmd.exe".into());
    let mut command = Command::new(shell);
    command.arg("/D").arg("/S").arg("/C");
    // `cmd.exe` does not use the C argv quoting rules applied by
    // `Command::arg`. With `/S /C`, it strips the first and last quotes from
    // the command string, so supply that outer pair explicitly and leave the
    // user's shell command otherwise untouched. This preserves commands that
    // begin with a quoted executable, such as `"python.exe" "script.py"`.
    command.raw_arg(format!("\"{script}\""));
    command
}

#[cfg(not(any(unix, windows)))]
fn shell_command(script: &str) -> Command {
    let mut command = Command::new("sh");
    command.arg("-c").arg(script);
    command
}

#[cfg(windows)]
fn spawn_process_group(command: &mut Command) -> std::io::Result<GroupChild> {
    command.group().kill_on_drop(true).spawn()
}

#[cfg(not(windows))]
fn spawn_process_group(command: &mut Command) -> std::io::Result<GroupChild> {
    command.group_spawn()
}

fn terminate_process_tree(child: &mut GroupChild) -> std::io::Result<()> {
    match child.kill() {
        Ok(()) => Ok(()),
        Err(error)
            if error.kind() == std::io::ErrorKind::InvalidInput && child.try_wait()?.is_some() =>
        {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

struct TimedStatus {
    status: ExitStatus,
    timed_out: bool,
}

#[cfg(test)]
fn run_timed(mut command: Command, timeout: Duration) -> std::io::Result<TimedStatus> {
    run_timed_cancellable(&mut command, timeout, None)
}

fn run_timed_cancellable(
    command: &mut Command,
    timeout: Duration,
    cancellation: Option<&AtomicBool>,
) -> std::io::Result<TimedStatus> {
    let mut child = spawn_process_group(command)?;
    let deadline = Instant::now() + timeout;

    loop {
        if cancellation.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            terminate_process_tree(&mut child)?;
            let _ = child.wait()?;
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "command terminated because its trial lease could not be renewed",
            ));
        }
        if let Some(status) = child.try_wait()? {
            return Ok(TimedStatus {
                status,
                timed_out: false,
            });
        }
        if Instant::now() >= deadline {
            terminate_process_tree(&mut child)?;
            let status = child.wait()?;
            return Ok(TimedStatus {
                status,
                timed_out: true,
            });
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Run a child command with a deadline, capturing stdout/stderr with a
/// per-stream byte cap. Both pipes are drained concurrently so a noisy command
/// cannot deadlock, and the isolated process group is killed on timeout.
#[cfg(test)]
fn run_capped(mut command: Command, timeout: Duration) -> std::io::Result<CappedOutput> {
    run_capped_cancellable(&mut command, timeout, None)
}

fn run_capped_cancellable(
    command: &mut Command,
    timeout: Duration,
    cancellation: Option<&AtomicBool>,
) -> std::io::Result<CappedOutput> {
    use std::process::Stdio;
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = spawn_process_group(command)?;

    let stdout_handle = child
        .inner()
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("failed to capture child stdout"))?;
    let stderr_handle = child
        .inner()
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("failed to capture child stderr"))?;

    let (stdout_done_tx, stdout_done_rx) = std::sync::mpsc::sync_channel(1);
    let stdout_thread = std::thread::spawn(move || {
        let result = read_capped(stdout_handle);
        let _ = stdout_done_tx.send(());
        result
    });
    let (stderr_done_tx, stderr_done_rx) = std::sync::mpsc::sync_channel(1);
    let stderr_thread = std::thread::spawn(move || {
        let result = read_capped(stderr_handle);
        let _ = stderr_done_tx.send(());
        result
    });

    let deadline = Instant::now() + timeout;
    let mut status = None;
    let mut stdout_done = false;
    let mut stderr_done = false;
    let mut lease_cancelled = false;
    let timed_out = loop {
        if cancellation.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            terminate_process_tree(&mut child)?;
            status = Some(child.wait()?);
            lease_cancelled = true;
            break false;
        }
        if status.is_none() {
            status = child.try_wait()?;
        }
        if !stdout_done {
            stdout_done = stdout_done_rx.try_recv().is_ok();
        }
        if !stderr_done {
            stderr_done = stderr_done_rx.try_recv().is_ok();
        }
        if status.is_some() && stdout_done && stderr_done {
            break false;
        }
        if Instant::now() >= deadline {
            terminate_process_tree(&mut child)?;
            status = Some(child.wait()?);
            break true;
        }
        std::thread::sleep(Duration::from_millis(20));
    };

    // A timeout closes pipes for every descendant in the process group. Joining
    // here guarantees capture threads are reclaimed before returning.
    let stdout = stdout_thread
        .join()
        .map_err(|_| std::io::Error::other("stdout capture thread panicked"))??;
    let stderr_result = stderr_thread
        .join()
        .map_err(|_| std::io::Error::other("stderr capture thread panicked"));
    let stderr = stderr_result??;
    let status = status.ok_or_else(|| std::io::Error::other("child status was not collected"))?;

    if lease_cancelled {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "command terminated because its trial lease could not be renewed",
        ));
    }

    let stdout_text = stdout.text;
    let stdout_truncated = stdout.truncated;
    let stderr_text = stderr.text;

    Ok(CappedOutput {
        status,
        stdout: stdout_text,
        stderr: stderr_text,
        stdout_truncated,
        timed_out,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Install a lightweight subscriber so the server's request spans are
    // visible in CLI deployments. Embedders can install their own subscriber
    // before calling the library API; try_init leaves an existing one intact.
    let _ = tracing_subscriber::fmt().try_init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            config,
            host,
            port,
            dashboard,
            auth_token,
            auth_token_file,
            read_token,
            read_token_file,
            checkpoint_dir,
            cors_origins,
            allow_unauthenticated_reads,
            lease_seconds,
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
            let engine = if let Some(path) = load_from {
                let (engine, checkpoint_kind) =
                    HolaEngine::load_configured_checkpoint(study_config, &path)
                        .await
                        .map_err(|e| format!("Failed to load checkpoint '{path}': {e}"))?;
                eprintln!("Loaded {} checkpoint from {path}", checkpoint_kind.as_str());
                engine
            } else {
                HolaEngine::from_config(study_config)
                    .map_err(|e| format!("Failed to create engine: {e}"))?
            };

            let auth_token = resolve_token(
                auth_token,
                auth_token_file.as_deref(),
                "--auth-token-file",
                std::env::var(API_TOKEN_ENV).ok(),
            )?;
            if !is_local_host(&host) && auth_token.is_none() {
                return Err(format!(
                    "an API token is required when --host is not localhost: set {API_TOKEN_ENV}, or pass --auth-token-file or --auth-token"
                )
                .into());
            }
            let read_token = resolve_token(
                read_token,
                read_token_file.as_deref(),
                "--read-token-file",
                std::env::var(READ_TOKEN_ENV).ok(),
            )?;
            if read_token.is_some() && auth_token.is_none() {
                return Err(format!(
                    "a read-only token ({READ_TOKEN_ENV}, --read-token-file, or --read-token) requires an API token for write endpoints: set {API_TOKEN_ENV} or pass --auth-token-file, or remove the read-only token"
                )
                .into());
            }

            let mut options = ServerOptions::new(port);
            options.host = host;
            options.dashboard_dir = dashboard;
            options.auth_token = auth_token;
            options.read_auth_token = read_token;
            options.checkpoint_dir = checkpoint_dir
                .or(config_checkpoint_dir)
                .or_else(|| config.parent().map(|path| path.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."));
            options.cors_allowed_origins = cors_origins;
            options.require_read_auth = !allow_unauthenticated_reads;
            options.lease_duration = Duration::from_secs(lease_seconds);

            hola::server::serve_with_options(engine, options).await?;
        }
        Commands::Worker {
            server,
            exec,
            mode,
            token,
            token_file,
            allow_insecure_http,
            request_timeout,
            command_timeout,
            outbox_dir,
        } => {
            let exec_mode = matches!(mode, WorkerMode::Exec);
            let token = resolve_token(
                token,
                token_file.as_deref(),
                "--token-file",
                std::env::var(API_TOKEN_ENV).ok(),
            )?;

            // Validate and normalize the server URL up front so a scheme-less
            // or typo'd value fails fast rather than entering the retry loop,
            // and so the joined "{server}/api/..." paths never produce a
            // double slash.
            let server = normalize_server_url(&server)?;

            // Decide before any request is built whether the token may travel
            // to this address at all.
            let insecure_http = allow_insecure_http || insecure_http_allowed_by_env();
            if let Some(warning) = check_token_transport(&server, token.is_some(), insecure_http)? {
                eprintln!("{warning}");
            }

            let mode_label = if exec_mode { "exec" } else { "callback" };
            eprintln!("Worker connecting to {server} ({mode_label} mode)...");
            eprintln!("Will execute: {exec}");

            let request_timeout = Duration::from_secs(request_timeout);
            let command_timeout = Duration::from_secs(command_timeout);
            let client = build_http_client(request_timeout)?;
            let outbox = exec_mode
                .then(|| TellOutbox::open(&outbox_dir, &server))
                .transpose()
                .map_err(|error| format!("failed to initialize tell outbox: {error}"))?;
            // Reuse this key until a trial is unambiguously received. If an ask
            // response is lost after the server creates a trial, the retry gets
            // that same trial instead of allocating and orphaning another one.
            let mut ask_idempotency_key = Uuid::new_v4().to_string();

            loop {
                // Never ask for more work while a prior expensive result is
                // uncertain. Exact duplicate tells are idempotent server-side,
                // so retrying a record after a lost response is safe.
                if let Some(outbox) = &outbox {
                    match flush_outbox(outbox, &client, token.as_deref()).await {
                        Ok(delivered) => {
                            for trial_id in delivered {
                                eprintln!("Completed trial {trial_id} (durable tell confirmed)");
                            }
                        }
                        Err(error) if error.is_permanent() => {
                            return Err(format!(
                                "{error}. Tell remains in outbox; permanent rejection requires operator reconciliation"
                            )
                            .into());
                        }
                        Err(error) => {
                            eprintln!("{error}. Tell remains in outbox; retrying in 5s...");
                            tokio::time::sleep(RETRY_DELAY).await;
                            continue;
                        }
                    }
                }

                let response = match send_checked(
                    "ask",
                    ask_request(&client, &server, token.as_deref(), &ask_idempotency_key),
                )
                .await
                {
                    Ok(response) => response,
                    Err(error) => {
                        eprintln!("{error}. Retrying in 5s...");
                        tokio::time::sleep(RETRY_DELAY).await;
                        continue;
                    }
                };
                let trial = match read_json_capped(response).await {
                    Ok(trial) => trial,
                    Err(error) => {
                        eprintln!("Ask returned invalid JSON: {error}. Retrying in 5s...");
                        tokio::time::sleep(RETRY_DELAY).await;
                        continue;
                    }
                };
                let trial_id = match trial.get("trial_id").and_then(serde_json::Value::as_u64) {
                    Some(trial_id) => trial_id,
                    None => {
                        eprintln!(
                            "Ask response missing a valid trial_id; retrying in 5s without executing"
                        );
                        tokio::time::sleep(RETRY_DELAY).await;
                        continue;
                    }
                };
                let params = match trial.get("params").filter(|params| params.is_object()) {
                    Some(params) => params.clone(),
                    None => {
                        eprintln!(
                            "Trial {trial_id}: ask response is missing an object-valued params field; canceling"
                        );
                        if let Err(error) =
                            cancel_trial(&client, &server, token.as_deref(), trial_id).await
                        {
                            eprintln!("Failed to cancel malformed trial {trial_id}: {error}");
                        }
                        ask_idempotency_key = Uuid::new_v4().to_string();
                        tokio::time::sleep(RETRY_DELAY).await;
                        continue;
                    }
                };
                ask_idempotency_key = Uuid::new_v4().to_string();

                if exec_mode {
                    let mut command = shell_command(&exec);
                    command.env("HOLA_PARAMS", params.to_string());
                    let command_result = run_command_with_heartbeat(
                        &client,
                        &server,
                        token.as_deref(),
                        trial_id,
                        CommandLeaseMode::Strict,
                        move |cancellation| {
                            run_capped_cancellable(
                                &mut command,
                                command_timeout,
                                Some(cancellation.as_ref()),
                            )
                        },
                    )
                    .await
                    .result;
                    match command_result {
                        Ok(output) => match decide_exec_outcome(&output) {
                            ExecOutcome::Tell(metrics) => {
                                let record = PendingTell::new(&server, trial_id, metrics);
                                let outbox = outbox.as_ref().expect("exec mode has an outbox");
                                outbox.persist(&record).map_err(|error| {
                                    format!(
                                        "failed to durably queue result for trial {trial_id}: {error}"
                                    )
                                })?;
                                match flush_outbox(outbox, &client, token.as_deref()).await {
                                    Ok(delivered) => {
                                        for delivered_id in delivered {
                                            eprintln!(
                                                "Completed trial {delivered_id} (durable tell confirmed)"
                                            );
                                        }
                                    }
                                    Err(error) if error.is_permanent() => {
                                        return Err(format!(
                                            "{error}. Trial {trial_id} remains durable; permanent rejection requires operator reconciliation"
                                        )
                                        .into());
                                    }
                                    Err(error) => {
                                        eprintln!(
                                            "{error}. Trial {trial_id} remains durable and will be retried before asking again"
                                        );
                                    }
                                }
                            }
                            ExecOutcome::Cancel(reason) => {
                                eprintln!(
                                    "Trial {trial_id}: {reason}, canceling. stderr: {}",
                                    log_snippet(&output.stderr)
                                );
                                if let Err(error) =
                                    cancel_trial(&client, &server, token.as_deref(), trial_id).await
                                {
                                    eprintln!("Failed to cancel trial {trial_id}: {error}");
                                }
                                tokio::time::sleep(COMMAND_FAILURE_DELAY).await;
                            }
                        },
                        Err(error) => {
                            eprintln!(
                                "Trial {trial_id}: failed to run command ({error}), canceling"
                            );
                            if let Err(error) =
                                cancel_trial(&client, &server, token.as_deref(), trial_id).await
                            {
                                eprintln!("Failed to cancel trial {trial_id}: {error}");
                            }
                            tokio::time::sleep(COMMAND_FAILURE_DELAY).await;
                        }
                    }
                } else {
                    // Callback mode: the script owns tell(), but a zero exit is
                    // not sufficient evidence that it actually completed this id.
                    let mut command = shell_command(&exec);
                    command
                        .env("HOLA_SERVER", &server)
                        .env("HOLA_TRIAL_ID", trial_id.to_string())
                        .env("HOLA_PARAMS", params.to_string());
                    if let Some(token) = &token {
                        command.env(API_TOKEN_ENV, token);
                        if insecure_http {
                            // The callback reports to the same server with the
                            // same token; let it see the operator's decision.
                            command.env(ALLOW_INSECURE_HTTP_ENV, "1");
                        }
                    }
                    let command_run = run_command_with_heartbeat(
                        &client,
                        &server,
                        token.as_deref(),
                        trial_id,
                        CommandLeaseMode::Callback,
                        move |cancellation| {
                            run_timed_cancellable(
                                &mut command,
                                command_timeout,
                                Some(cancellation.as_ref()),
                            )
                        },
                    )
                    .await;
                    let failure = match command_run.result {
                        Ok(result) if result.timed_out => Some("command timed out".to_string()),
                        Ok(result) if !result.status.success() => Some(format!(
                            "script failed (exit {})",
                            result.status.code().unwrap_or(-1)
                        )),
                        Ok(_) => None,
                        Err(error) => Some(format!("failed to run command ({error})")),
                    };

                    if command_run.callback_completion_confirmed {
                        if let Some(reason) = failure {
                            eprintln!(
                                "Trial {trial_id}: callback completion remains authoritative after {reason}; not canceling"
                            );
                        } else {
                            eprintln!(
                                "Trial {trial_id}: callback exited after its server-confirmed completion"
                            );
                        }
                        continue;
                    }

                    loop {
                        match callback_disposition(
                            &client,
                            &server,
                            token.as_deref(),
                            trial_id,
                            failure.as_deref(),
                        )
                        .await
                        {
                            Ok(CallbackDisposition::Completed) => {
                                if let Some(reason) = &failure {
                                    eprintln!(
                                        "Trial {trial_id}: callback completion confirmed by server after {reason}"
                                    );
                                } else {
                                    eprintln!(
                                        "Trial {trial_id}: callback completion confirmed by server"
                                    );
                                }
                                break;
                            }
                            Ok(CallbackDisposition::Cancel(reason)) => {
                                eprintln!("Trial {trial_id}: {reason}, canceling");
                                match cancel_callback_trial(
                                    &client,
                                    &server,
                                    token.as_deref(),
                                    trial_id,
                                )
                                .await
                                {
                                    Ok(CallbackCancelOutcome::Cancelled) => {
                                        tokio::time::sleep(COMMAND_FAILURE_DELAY).await;
                                        break;
                                    }
                                    Ok(CallbackCancelOutcome::Completed) => {
                                        eprintln!(
                                            "Trial {trial_id}: callback completion won the cancellation race and is authoritative"
                                        );
                                        break;
                                    }
                                    Ok(CallbackCancelOutcome::NotPending) => {
                                        eprintln!(
                                            "Trial {trial_id}: cancellation was unnecessary because the trial is no longer pending"
                                        );
                                        break;
                                    }
                                    Err(error) => {
                                        eprintln!(
                                            "Failed to reconcile cancellation for trial {trial_id}: {error}. Retrying in 5s..."
                                        );
                                        tokio::time::sleep(RETRY_DELAY).await;
                                    }
                                }
                            }
                            Ok(CallbackDisposition::NotPending) => {
                                if let Some(reason) = &failure {
                                    eprintln!(
                                        "Trial {trial_id}: {reason}, but the trial is no longer pending; leaving terminal server state unchanged"
                                    );
                                } else {
                                    eprintln!(
                                        "Trial {trial_id}: callback exited without a retained completion, but the trial is no longer pending"
                                    );
                                }
                                break;
                            }
                            Err(error) => {
                                // Completion is uncertain: never cancel based on
                                // a failed verification request, because the
                                // callback's tell may already have succeeded.
                                eprintln!("{error}. Retrying verification in 5s...");
                                tokio::time::sleep(RETRY_DELAY).await;
                            }
                        }
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

    fn test_directory(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("hola-cli-{label}-{}-{unique}", std::process::id()));
        // Private regardless of the umask: the outbox refuses a root that its
        // group or other users can write to.
        create_private_directory(&path).expect("test directory should be created");
        path
    }

    struct MockReply {
        status: Option<u16>,
        body: String,
        delay: Duration,
        /// Extra response header lines, each terminated by CRLF.
        headers: String,
        /// When false the body is delimited by closing the connection and no
        /// Content-Length is declared.
        declare_length: bool,
    }

    impl MockReply {
        fn close_connection() -> Self {
            Self {
                status: None,
                ..Self::response(200, "")
            }
        }

        fn response(status: u16, body: impl Into<String>) -> Self {
            Self {
                status: Some(status),
                body: body.into(),
                delay: Duration::ZERO,
                headers: String::new(),
                declare_length: true,
            }
        }

        fn delayed(status: u16, body: &'static str, delay: Duration) -> Self {
            Self {
                delay,
                ..Self::response(status, body)
            }
        }

        fn redirect(status: u16, location: &str) -> Self {
            Self {
                headers: format!("Location: {location}\r\n"),
                ..Self::response(status, "")
            }
        }

        fn without_content_length(status: u16, body: impl Into<String>) -> Self {
            Self {
                declare_length: false,
                ..Self::response(status, body)
            }
        }
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> std::io::Result<String> {
        stream.set_read_timeout(Some(Duration::from_secs(3)))?;
        let mut bytes = Vec::new();
        let mut buffer = [0u8; 2048];
        let mut expected_len = None;
        loop {
            let count = stream.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            bytes.extend_from_slice(&buffer[..count]);
            if expected_len.is_none() {
                if let Some(headers_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
                {
                    let headers_end = headers_end + 4;
                    let headers = String::from_utf8_lossy(&bytes[..headers_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    expected_len = Some(headers_end + content_length);
                }
            }
            if expected_len.is_some_and(|expected| bytes.len() >= expected) {
                break;
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn mock_http_server(replies: Vec<MockReply>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("mock server should bind a local port");
        let address = listener.local_addr().expect("mock server has an address");
        let handle = std::thread::spawn(move || {
            let mut requests = Vec::with_capacity(replies.len());
            for reply in replies {
                let (mut stream, _) = listener
                    .accept()
                    .expect("mock server should accept request");
                requests.push(
                    read_http_request(&mut stream).expect("mock server should read HTTP request"),
                );
                if !reply.delay.is_zero() {
                    std::thread::sleep(reply.delay);
                }
                let Some(status) = reply.status else {
                    continue;
                };
                let reason = match status {
                    200 => "OK",
                    307 => "Temporary Redirect",
                    404 => "Not Found",
                    409 => "Conflict",
                    503 => "Service Unavailable",
                    _ => "Test Status",
                };
                let length = if reply.declare_length {
                    format!("Content-Length: {}\r\n", reply.body.len())
                } else {
                    String::new()
                };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n{length}{}Connection: close\r\n\r\n{}",
                    reply.headers, reply.body
                );
                // The client may stop reading an oversized body early.
                let _ = stream.write_all(response.as_bytes());
            }
            requests
        });
        (format!("http://{address}"), handle)
    }

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

    #[cfg(unix)]
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
        let output = run_capped(shell_command(&script), Duration::from_secs(10))
            .expect("run_capped should not deadlock");
        assert!(output.status.success());
        assert_eq!(output.stdout.len(), MAX_CAPTURE_BYTES);
        assert_eq!(output.stderr.len(), MAX_CAPTURE_BYTES);
        assert!(output.stdout_truncated);
    }

    #[cfg(unix)]
    fn exit_status(code: i32) -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(code << 8)
    }

    #[cfg(windows)]
    fn exit_status(code: i32) -> ExitStatus {
        use std::os::windows::process::ExitStatusExt;
        ExitStatus::from_raw(code as u32)
    }

    fn capped_output(code: i32, stdout: &str) -> CappedOutput {
        CappedOutput {
            status: exit_status(code),
            stdout: stdout.to_string(),
            stderr: String::new(),
            stdout_truncated: false,
            timed_out: false,
        }
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
    fn normalize_server_url_preserves_and_canonicalizes_path_prefix() {
        let normalized = normalize_server_url("https://example.com/hola/proxy///")
            .expect("a path-prefixed server URL should normalize");
        assert_eq!(normalized, "https://example.com/hola/proxy");
        assert_eq!(
            format!("{normalized}/api/ask"),
            "https://example.com/hola/proxy/api/ask"
        );
    }

    #[test]
    fn normalize_server_url_rejects_query_fragment_and_userinfo() {
        for invalid in [
            "https://example.com?tenant=one",
            "https://example.com/#dashboard",
            "https://user:password@example.com",
        ] {
            let error = normalize_server_url(invalid)
                .expect_err("ambiguous base URL components must be rejected");
            assert!(error.to_string().contains("invalid --server URL"));
        }
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
    fn ask_request_carries_auth_and_idempotency_key() {
        let client = build_http_client(Duration::from_secs(1)).expect("client should build");
        let request = ask_request(
            &client,
            "http://127.0.0.1:8000",
            Some("secret"),
            "stable-ask-key",
        )
        .build()
        .expect("request should build");

        assert_eq!(
            request
                .headers()
                .get("Idempotency-Key")
                .expect("idempotency header")
                .to_str()
                .expect("ASCII header"),
            "stable-ask-key"
        );
        assert_eq!(
            request
                .headers()
                .get(reqwest::header::AUTHORIZATION)
                .expect("authorization header")
                .to_str()
                .expect("ASCII header"),
            "Bearer secret"
        );
    }

    #[test]
    fn worker_timeout_flags_are_positive_and_have_bounded_defaults() {
        let cli = Cli::try_parse_from([
            "hola",
            "worker",
            "--server",
            "http://127.0.0.1:8000",
            "--exec",
            "worker-command",
        ])
        .expect("worker defaults should parse");
        match cli.command {
            Commands::Worker {
                request_timeout,
                command_timeout,
                ..
            } => {
                assert_eq!(request_timeout, 30);
                assert_eq!(command_timeout, 3600);
            }
            Commands::Serve { .. } => panic!("expected worker command"),
        }

        let invalid = Cli::try_parse_from([
            "hola",
            "worker",
            "--server",
            "http://127.0.0.1:8000",
            "--exec",
            "worker-command",
            "--command-timeout",
            "0",
        ]);
        assert!(invalid.is_err(), "zero command timeout must be rejected");
    }

    #[test]
    fn outbox_round_trips_atomically_and_rejects_changed_metrics() {
        let root = test_directory("outbox-roundtrip");
        let server = "http://127.0.0.1:8000";
        let outbox = TellOutbox::open(&root, server).expect("outbox should open");
        let record = PendingTell::new(server, 42, serde_json::json!({"loss": 0.25}));

        let path = outbox.persist(&record).expect("record should persist");
        assert!(path.exists());
        drop(outbox);
        let outbox = TellOutbox::open(&root, server).expect("outbox should reopen after restart");
        assert_eq!(outbox.pending().expect("outbox should load")[0].1, record);
        assert_eq!(
            outbox.persist(&record).expect("exact re-persist is safe"),
            path
        );

        let changed = PendingTell::new(server, 42, serde_json::json!({"loss": 9.0}));
        let error = outbox
            .persist(&changed)
            .expect_err("different metrics for one id must be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);

        outbox.remove(&path).expect("record should be removed");
        assert!(outbox.pending().expect("outbox should be empty").is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn outbox_retries_the_same_tell_after_a_lost_response() {
        let (server, server_thread) = mock_http_server(vec![
            MockReply::close_connection(),
            MockReply::response(200, r#"{"status":"ok","trial":{"trial_id":7}}"#),
        ]);
        let root = test_directory("outbox-lost-response");
        let outbox = TellOutbox::open(&root, &server).expect("outbox should open");
        let record = PendingTell::new(&server, 7, serde_json::json!({"loss": 0.5}));
        outbox
            .persist(&record)
            .expect("tell must be durable before send");
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        let first = flush_outbox(&outbox, &client, Some("secret")).await;
        let first = first.expect_err("a dropped response is an uncertain tell");
        assert!(!first.is_permanent());
        assert_eq!(outbox.pending().expect("record remains").len(), 1);

        let delivered = flush_outbox(&outbox, &client, Some("secret"))
            .await
            .expect("idempotent retry should be confirmed");
        assert_eq!(delivered, vec![7]);
        assert!(outbox.pending().expect("record is removed").is_empty());

        let requests = server_thread.join().expect("mock server should finish");
        assert_eq!(requests.len(), 2);
        for request in requests {
            assert!(request.contains(r#""trial_id":7"#));
            assert!(
                request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer secret")
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn outbox_emits_post_commit_warnings_before_removing_the_record() {
        let (server, server_thread) = mock_http_server(vec![MockReply::response(
            200,
            r#"{"status":"ok","trial":{"trial_id":7},"post_commit_warnings":["auto-checkpoint failed","strategy refit failed"]}"#,
        )]);
        let root = test_directory("outbox-post-commit-warnings");
        let outbox = TellOutbox::open(&root, &server).expect("outbox should open");
        let record = PendingTell::new(&server, 7, serde_json::json!({"loss": 0.5}));
        let path = outbox.persist(&record).expect("tell must be durable");
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");
        let mut emitted = Vec::new();

        let delivered =
            flush_outbox_with_warning_sink(&outbox, &client, None, |trial_id, warning| {
                assert!(path.exists(), "warning must be emitted before cleanup");
                emitted.push((trial_id, warning.to_string()));
            })
            .await
            .expect("valid acknowledgement should confirm the tell");

        assert_eq!(delivered, vec![7]);
        assert_eq!(
            emitted,
            vec![
                (7, "auto-checkpoint failed".to_string()),
                (7, "strategy refit failed".to_string()),
            ]
        );
        assert!(!path.exists(), "confirmed record should be removed");
        server_thread.join().expect("mock server should finish");
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn tell_and_cancel_reject_non_success_http_statuses() {
        let (server, server_thread) = mock_http_server(vec![
            MockReply::response(503, r#"{"error":"tell unavailable"}"#),
            MockReply::response(409, r#"{"error":"cancel conflict"}"#),
        ]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");
        let record = PendingTell::new(&server, 3, serde_json::json!({"loss": 1.0}));

        let tell_error = deliver_tell(&client, None, &record)
            .await
            .expect_err("503 tell must fail");
        assert!(!tell_error.is_permanent());
        let tell_error = tell_error.to_string();
        assert!(tell_error.contains("HTTP 503 Service Unavailable"));
        assert!(tell_error.contains("tell unavailable"));

        let cancel_error = cancel_trial(&client, &server, None, 3)
            .await
            .expect_err("409 cancel must fail");
        assert!(cancel_error.contains("HTTP 409 Conflict"));
        assert!(cancel_error.contains("cancel conflict"));
        server_thread.join().expect("mock server should finish");
    }

    #[tokio::test]
    async fn cancel_requires_canonical_2xx_acknowledgement() {
        let (server, server_thread) = mock_http_server(vec![
            MockReply::response(200, "not-json"),
            MockReply::response(200, r#"{"status":"error"}"#),
            MockReply::response(200, r#"{"status":"ok","trial_id":99}"#),
            MockReply::response(200, r#"{"status":"ok","trial_id":7}"#),
        ]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        let invalid_json = cancel_trial(&client, &server, None, 7)
            .await
            .expect_err("invalid JSON must not confirm cancellation");
        assert!(invalid_json.contains("invalid acknowledgement JSON"));

        let bad_status = cancel_trial(&client, &server, None, 7)
            .await
            .expect_err("status:error must not confirm cancellation");
        assert!(bad_status.contains("canonical status 'ok'"));

        let wrong_id = cancel_trial(&client, &server, None, 7)
            .await
            .expect_err("wrong optional identity must not confirm cancellation");
        assert!(wrong_id.contains("trial_id 99 instead"));

        cancel_trial(&client, &server, None, 7)
            .await
            .expect("canonical acknowledgement should confirm cancellation");
        server_thread.join().expect("mock server should finish");
    }

    #[tokio::test]
    async fn outbox_keeps_records_for_invalid_or_mismatched_2xx_acknowledgements() {
        for (label, body, expected) in [
            ("invalid-json", "not-json", "invalid acknowledgement JSON"),
            (
                "missing-trial",
                r#"{"status":"ok"}"#,
                "missing trial.trial_id",
            ),
            (
                "wrong-trial",
                r#"{"status":"ok","trial":{"trial_id":99}}"#,
                "acknowledged trial_id 99 instead",
            ),
            (
                "non-array-warnings",
                r#"{"status":"ok","trial":{"trial_id":7},"post_commit_warnings":"failed"}"#,
                "non-array post_commit_warnings",
            ),
            (
                "non-string-warning",
                r#"{"status":"ok","trial":{"trial_id":7},"post_commit_warnings":[42]}"#,
                "non-string post_commit_warnings entry",
            ),
        ] {
            let (server, server_thread) = mock_http_server(vec![MockReply::response(200, body)]);
            let root = test_directory(label);
            let outbox = TellOutbox::open(&root, &server).expect("outbox should open");
            let record = PendingTell::new(&server, 7, serde_json::json!({"loss": 0.5}));
            outbox.persist(&record).expect("tell must be durable");
            let client = build_http_client(Duration::from_secs(2)).expect("client should build");

            let error = flush_outbox(&outbox, &client, None)
                .await
                .expect_err("invalid acknowledgement must not delete the record");
            assert!(error.is_permanent());
            assert!(error.to_string().contains(expected));
            assert_eq!(outbox.pending().expect("record remains").len(), 1);

            server_thread.join().expect("mock server should finish");
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[tokio::test]
    async fn permanent_tell_rejection_keeps_outbox_record_and_is_classified() {
        let (server, server_thread) = mock_http_server(vec![MockReply::response(
            400,
            r#"{"error":"Trial 7 lease expired"}"#,
        )]);
        let root = test_directory("outbox-permanent-rejection");
        let outbox = TellOutbox::open(&root, &server).expect("outbox should open");
        let record = PendingTell::new(&server, 7, serde_json::json!({"loss": 0.5}));
        outbox.persist(&record).expect("tell must be durable");
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        let error = flush_outbox(&outbox, &client, None)
            .await
            .expect_err("400 is a permanent rejection");
        assert!(error.is_permanent());
        assert!(error.to_string().contains("lease expired"));
        assert_eq!(outbox.pending().expect("record remains").len(), 1);

        server_thread.join().expect("mock server should finish");
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn heartbeat_validates_acknowledgement_and_sends_auth() {
        let (server, server_thread) = mock_http_server(vec![MockReply::response(
            200,
            r#"{"status":"ok","trial_id":7,"lease_expires_at_ms":4000000000000}"#,
        )]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        let deadline = heartbeat_trial(&client, &server, Some("secret"), 7)
            .await
            .expect("heartbeat should be acknowledged");
        assert_eq!(deadline, 4_000_000_000_000);

        let requests = server_thread.join().expect("mock server should finish");
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /api/heartbeat "));
        assert!(requests[0].contains(r#""trial_id":7"#));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer secret")
        );
    }

    #[tokio::test]
    async fn heartbeat_rejects_wrong_trial_acknowledgement() {
        let (server, server_thread) = mock_http_server(vec![MockReply::response(
            200,
            r#"{"status":"ok","trial_id":8,"lease_expires_at_ms":4000000000000}"#,
        )]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        let error = heartbeat_trial(&client, &server, None, 7)
            .await
            .expect_err("wrong trial acknowledgement must fail");
        assert!(matches!(error, HeartbeatError::Rejected(_)));
        assert!(error.to_string().contains("trial_id 8 instead"));
        server_thread.join().expect("mock server should finish");
    }

    #[tokio::test]
    async fn heartbeat_distinguishes_terminal_unsupported_and_untrusted_rejections() {
        let (server, server_thread) = mock_http_server(vec![
            MockReply::response(
                400,
                r#"{"code":"heartbeat_failed","error":"Trial 7 is not pending"}"#,
            ),
            MockReply::response(404, r#"{"error":"route not found"}"#),
            MockReply::response(401, r#"{"code":"unauthorized","error":"bad token"}"#),
        ]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        assert!(matches!(
            heartbeat_trial(&client, &server, None, 7).await,
            Err(HeartbeatError::Terminal(_))
        ));
        assert!(matches!(
            heartbeat_trial(&client, &server, None, 8).await,
            Err(HeartbeatError::Unsupported(_))
        ));
        assert!(matches!(
            heartbeat_trial(&client, &server, None, 9).await,
            Err(HeartbeatError::Rejected(_))
        ));
        server_thread.join().expect("mock server should finish");
    }

    #[tokio::test]
    async fn missing_legacy_heartbeat_endpoint_does_not_prevent_command_execution() {
        let (server, server_thread) = mock_http_server(vec![MockReply::response(
            404,
            r#"{"error":"route not found"}"#,
        )]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        let command_run =
            run_command_with_heartbeat(&client, &server, None, 7, CommandLeaseMode::Strict, |_| {
                Ok(42_u64)
            })
            .await;
        assert_eq!(command_run.result.expect("legacy command should run"), 42);
        assert!(!command_run.callback_completion_confirmed);
        let requests = server_thread.join().expect("mock server should finish");
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /api/heartbeat "));
    }

    #[tokio::test]
    async fn callback_status_requires_canonical_state_for_the_same_trial() {
        let (server, server_thread) = mock_http_server(vec![
            MockReply::response(200, r#"{"status":"ok","trial_id":11,"state":"completed"}"#),
            MockReply::response(200, r#"{"status":"ok","trial_id":12,"state":"pending"}"#),
            MockReply::response(
                200,
                r#"{"status":"ok","trial_id":13,"state":"not_pending"}"#,
            ),
            MockReply::response(200, r#"{"status":"ok","trial_id":99,"state":"completed"}"#),
        ]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        assert_eq!(
            callback_trial_state(&client, &server, None, 11)
                .await
                .expect("matching completion should verify"),
            CallbackTrialState::Completed
        );
        assert_eq!(
            callback_trial_state(&client, &server, None, 12)
                .await
                .expect("pending state should verify"),
            CallbackTrialState::Pending
        );
        assert_eq!(
            callback_trial_state(&client, &server, None, 13)
                .await
                .expect("terminal state should verify"),
            CallbackTrialState::NotPending
        );
        let error = callback_trial_state(&client, &server, None, 14)
            .await
            .expect_err("wrong status id must fail verification");
        assert!(error.contains("trial_id 99 instead"));
        let requests = server_thread.join().expect("mock server should finish");
        assert!(
            requests
                .iter()
                .all(|request| request.starts_with("GET /api/trial/")
                    && request.contains("/status "))
        );
    }

    #[tokio::test]
    async fn callback_status_falls_back_safely_for_legacy_servers() {
        let (server, server_thread) = mock_http_server(vec![
            MockReply::response(404, r#"{"error":"route not found"}"#),
            MockReply::response(200, r#"{"trial_id":21}"#),
            MockReply::response(404, r#"{"error":"route not found"}"#),
            MockReply::response(
                404,
                r#"{"code":"trial_not_found","error":"Trial 22 not found"}"#,
            ),
            MockReply::response(
                200,
                r#"{"status":"ok","trial_id":22,"lease_expires_at_ms":4000000000000}"#,
            ),
            MockReply::response(404, r#"{"error":"route not found"}"#),
            MockReply::response(
                404,
                r#"{"code":"trial_not_found","error":"Trial 23 not found"}"#,
            ),
            MockReply::response(
                400,
                r#"{"code":"heartbeat_failed","error":"Trial 23 is not pending"}"#,
            ),
            MockReply::response(404, r#"{"error":"route not found"}"#),
            MockReply::response(
                404,
                r#"{"code":"trial_not_found","error":"Trial 24 not found"}"#,
            ),
            MockReply::response(404, r#"{"error":"route not found"}"#),
            MockReply::response(404, r#"{"error":"route not found"}"#),
            MockReply::response(
                404,
                r#"{"code":"trial_not_found","error":"Trial 25 not found"}"#,
            ),
            MockReply::response(401, r#"{"code":"unauthorized","error":"bad token"}"#),
        ]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        assert_eq!(
            callback_trial_state(&client, &server, None, 21)
                .await
                .expect("legacy exact completion should remain authoritative"),
            CallbackTrialState::Completed
        );
        assert_eq!(
            callback_trial_state(&client, &server, None, 22)
                .await
                .expect("legacy heartbeat should prove the trial remains pending"),
            CallbackTrialState::Pending
        );
        assert_eq!(
            callback_trial_state(&client, &server, None, 23)
                .await
                .expect("legacy heartbeat rejection should prove the trial is terminal"),
            CallbackTrialState::NotPending
        );
        assert_eq!(
            callback_trial_state(&client, &server, None, 24)
                .await
                .expect("a missing legacy heartbeat route must preserve old-server operation"),
            CallbackTrialState::Pending
        );
        let error = callback_trial_state(&client, &server, None, 25)
            .await
            .expect_err("an authentication rejection is not terminal-state proof");
        assert!(error.contains("untrusted heartbeat rejection"));

        let requests = server_thread.join().expect("mock server should finish");
        assert!(requests[0].starts_with("GET /api/trial/21/status "));
        assert!(requests[1].starts_with("GET /api/trial/21?include_infeasible=true "));
        assert!(requests[2].starts_with("GET /api/trial/22/status "));
        assert!(requests[3].starts_with("GET /api/trial/22?include_infeasible=true "));
        assert!(requests[4].starts_with("POST /api/heartbeat "));
        assert!(requests[5].starts_with("GET /api/trial/23/status "));
        assert!(requests[6].starts_with("GET /api/trial/23?include_infeasible=true "));
        assert!(requests[7].starts_with("POST /api/heartbeat "));
        assert!(requests[8].starts_with("GET /api/trial/24/status "));
        assert!(requests[9].starts_with("GET /api/trial/24?include_infeasible=true "));
        assert!(requests[10].starts_with("POST /api/heartbeat "));
        assert!(requests[11].starts_with("GET /api/trial/25/status "));
        assert!(requests[12].starts_with("GET /api/trial/25?include_infeasible=true "));
        assert!(requests[13].starts_with("POST /api/heartbeat "));
    }

    #[tokio::test]
    async fn callback_command_failure_defers_to_exact_server_completion() {
        let (server, server_thread) = mock_http_server(vec![
            MockReply::response(200, r#"{"status":"ok","trial_id":7,"state":"completed"}"#),
            MockReply::response(200, r#"{"status":"ok","trial_id":8,"state":"pending"}"#),
            MockReply::response(200, r#"{"status":"ok","trial_id":9,"state":"not_pending"}"#),
            MockReply::response(503, r#"{"error":"unavailable"}"#),
        ]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");
        let failure = "failed to run command (lease heartbeat failed)";

        assert_eq!(
            callback_disposition(&client, &server, None, 7, Some(failure))
                .await
                .expect("matching completed trial should be authoritative"),
            CallbackDisposition::Completed
        );
        assert_eq!(
            callback_disposition(&client, &server, None, 8, Some(failure))
                .await
                .expect("pending state should preserve command failure"),
            CallbackDisposition::Cancel(failure.to_string())
        );
        assert_eq!(
            callback_disposition(&client, &server, None, 9, Some(failure))
                .await
                .expect("terminal state must not choose cancellation"),
            CallbackDisposition::NotPending
        );
        let error = callback_disposition(&client, &server, None, 10, Some(failure))
            .await
            .expect_err("uncertain verification must not choose cancellation");
        assert!(error.contains("HTTP 503 Service Unavailable"));
        server_thread.join().expect("mock server should finish");
    }

    #[tokio::test]
    async fn callback_cancel_rejection_reconciles_a_late_completion() {
        let (server, server_thread) = mock_http_server(vec![
            MockReply::response(400, r#"{"code":"cancel_failed","error":"not pending"}"#),
            MockReply::response(200, r#"{"status":"ok","trial_id":7,"state":"completed"}"#),
        ]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        assert_eq!(
            cancel_callback_trial(&client, &server, None, 7)
                .await
                .expect("late completion should reconcile the rejected cancel"),
            CallbackCancelOutcome::Completed
        );
        let requests = server_thread.join().expect("mock server should finish");
        assert!(requests[0].starts_with("POST /api/cancel "));
        assert!(requests[1].starts_with("GET /api/trial/7/status "));
    }

    #[tokio::test]
    async fn legacy_cancel_rejection_terminates_after_completion_receipt_eviction() {
        let (server, server_thread) = mock_http_server(vec![
            MockReply::response(
                400,
                r#"{"error":"Trial 7 is not pending (may be completed or unknown)"}"#,
            ),
            MockReply::response(404, r#"{"error":"route not found"}"#),
            MockReply::response(404, r#"{"error":"Trial 7 not found"}"#),
            MockReply::response(404, r#"{"error":"route not found"}"#),
        ]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        assert_eq!(
            cancel_callback_trial(&client, &server, None, 7)
                .await
                .expect("canonical legacy cancel rejection is terminal proof"),
            CallbackCancelOutcome::NotPending
        );
        let requests = server_thread.join().expect("mock server should finish");
        assert!(requests[0].starts_with("POST /api/cancel "));
        assert!(requests[1].starts_with("GET /api/trial/7/status "));
        assert!(requests[2].starts_with("GET /api/trial/7?include_infeasible=true "));
        assert!(requests[3].starts_with("POST /api/heartbeat "));
    }

    #[tokio::test]
    async fn configured_request_timeout_bounds_worker_http_calls() {
        let (server, server_thread) = mock_http_server(vec![MockReply::delayed(
            200,
            r#"{"status":"ok"}"#,
            Duration::from_millis(300),
        )]);
        let client = build_http_client(Duration::from_millis(50)).expect("client should build");
        let started = Instant::now();

        let error = cancel_trial(&client, &server, None, 1)
            .await
            .expect_err("slow request should time out");
        assert!(error.contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(1));
        server_thread.join().expect("mock server should finish");
    }

    #[test]
    fn decide_exec_outcome_zero_exit_valid_json_tells() {
        let output = capped_output(0, r#"{"loss": 1.5}"#);
        match decide_exec_outcome(&output) {
            ExecOutcome::Tell(metrics) => {
                assert_eq!(metrics["loss"], serde_json::json!(1.5));
            }
            ExecOutcome::Cancel(reason) => panic!("expected Tell, got Cancel: {reason}"),
        }
    }

    #[test]
    fn decide_exec_outcome_nonzero_exit_cancels() {
        let output = capped_output(3, r#"{"loss": 1.5}"#);
        match decide_exec_outcome(&output) {
            ExecOutcome::Cancel(reason) => assert!(reason.contains("exit 3")),
            ExecOutcome::Tell(_) => panic!("expected Cancel on non-zero exit"),
        }
    }

    #[test]
    fn decide_exec_outcome_invalid_json_cancels() {
        let output = capped_output(0, "not json");
        match decide_exec_outcome(&output) {
            ExecOutcome::Cancel(reason) => assert!(reason.contains("invalid JSON")),
            ExecOutcome::Tell(_) => panic!("expected Cancel on invalid JSON"),
        }
    }

    #[test]
    fn decide_exec_outcome_requires_metrics_object() {
        let output = capped_output(0, "[1, 2, 3]");
        match decide_exec_outcome(&output) {
            ExecOutcome::Cancel(reason) => assert!(reason.contains("JSON object")),
            ExecOutcome::Tell(_) => panic!("expected Cancel for non-object metrics"),
        }
    }

    #[test]
    fn decide_exec_outcome_truncated_stdout_cancels_distinctly() {
        // A successful command whose stdout exceeds the cap must cancel with a
        // truncation-specific reason rather than the generic parse-failed path.
        let mut output = capped_output(0, "truncated");
        output.stdout_truncated = true;
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

    #[test]
    fn decide_exec_outcome_timeout_cancels_distinctly() {
        let mut output = capped_output(0, r#"{"loss": 1.5}"#);
        output.timed_out = true;
        match decide_exec_outcome(&output) {
            ExecOutcome::Cancel(reason) => assert!(reason.contains("timed out")),
            ExecOutcome::Tell(_) => panic!("expected Cancel on command timeout"),
        }
    }

    #[test]
    fn platform_shell_preserves_nonzero_exit_status() {
        #[cfg(unix)]
        let script = "exit 7";
        #[cfg(windows)]
        let script = "exit /b 7";

        let result = run_timed(shell_command(script), Duration::from_secs(5))
            .expect("platform shell should execute");
        assert!(!result.timed_out);
        assert_eq!(result.status.code(), Some(7));
    }

    #[cfg(windows)]
    #[test]
    fn platform_shell_runs_leading_quoted_executable() {
        let test_binary = std::env::current_exe().expect("test binary path should be available");
        let script = format!("\"{}\" \"--list\"", test_binary.display());

        let output = run_capped(shell_command(&script), Duration::from_secs(10))
            .expect("platform shell should launch a quoted executable");
        assert!(output.status.success(), "stderr: {}", output.stderr);
        assert!(
            output
                .stdout
                .contains("platform_shell_runs_leading_quoted_executable"),
            "the quoted test executable should have listed its tests"
        );
    }

    #[test]
    fn run_capped_times_out_and_reaps_command() {
        #[cfg(unix)]
        let script = "sleep 5";
        #[cfg(windows)]
        let script = "ping 127.0.0.1 -n 6 >NUL";

        let started = Instant::now();
        let output = run_capped(shell_command(script), Duration::from_millis(100))
            .expect("timed command should be killed and reaped");
        assert!(output.timed_out);
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[cfg(unix)]
    #[test]
    fn run_timed_kills_descendants_before_they_write_a_sentinel() {
        let directory = test_directory("process-tree");
        let sentinel = directory.join("descendant-finished");
        let script = format!("(sleep 0.5; printf done > '{}') & wait", sentinel.display());

        let result = run_timed(shell_command(&script), Duration::from_millis(100))
            .expect("timed process tree should be terminated");
        assert!(result.timed_out);
        std::thread::sleep(Duration::from_millis(600));
        assert!(!sentinel.exists(), "a timed-out descendant survived");
        let _ = std::fs::remove_dir_all(directory);
    }

    // ── Bearer token transport ──────────────────────────────────────────

    #[test]
    fn token_is_refused_over_plain_http_to_a_non_loopback_server() {
        for server in [
            "http://machine-a:8000",
            "http://192.168.1.20:8000",
            "http://[2001:db8::1]:8000",
            "http://localhost.example.com:8000",
        ] {
            let error = check_token_transport(server, true, false)
                .expect_err("a token must not cross the network in clear text by default");
            assert!(error.contains(server), "{error}");
            assert!(error.contains("--allow-insecure-http"), "{error}");
            assert!(error.contains("HOLA_ALLOW_INSECURE_HTTP=1"), "{error}");
            assert!(error.contains("https://"), "{error}");

            let warning = check_token_transport(server, true, true)
                .expect("the explicit override permits plain HTTP")
                .expect("the override is reported so it is never silent");
            assert!(warning.contains("unencrypted HTTP"), "{warning}");

            assert_eq!(
                check_token_transport(server, false, false),
                Ok(None),
                "without a token there is nothing to protect"
            );
        }
    }

    #[test]
    fn token_is_allowed_over_loopback_http_and_any_https() {
        for server in [
            "http://localhost:8000",
            "http://LOCALHOST:8000",
            "http://127.0.0.1:8000",
            "http://127.8.9.10:8000",
            "http://[::1]:8000",
            "http://[::ffff:127.0.0.1]:8000",
            "https://machine-a:8443",
            "https://hola.example.com/prefix",
        ] {
            assert_eq!(
                check_token_transport(server, true, false),
                Ok(None),
                "{server} should not need the insecure override"
            );
        }
    }

    #[test]
    fn worker_accepts_the_insecure_http_override_flag() {
        let cli = Cli::try_parse_from([
            "hola",
            "worker",
            "--server",
            "http://machine-a:8000",
            "--exec",
            "worker-command",
            "--allow-insecure-http",
        ])
        .expect("override flag should parse");
        match cli.command {
            Commands::Worker {
                allow_insecure_http,
                ..
            } => assert!(allow_insecure_http),
            Commands::Serve { .. } => panic!("expected worker command"),
        }
    }

    // ── Tokens without argv ─────────────────────────────────────────────

    fn write_private_file(directory: &Path, name: &str, contents: &str) -> PathBuf {
        let path = directory.join(name);
        std::fs::write(&path, contents).expect("file should be written");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                .expect("file should be private");
        }
        path
    }

    #[test]
    fn token_file_ignores_one_trailing_newline_and_rejects_unusable_files() {
        let directory = test_directory("token-file");
        let write = |name: &str, contents: &str| write_private_file(&directory, name, contents);

        for (name, contents) in [
            ("lf", "s3cret\n"),
            ("crlf", "s3cret\r\n"),
            ("bare", "s3cret"),
        ] {
            assert_eq!(
                read_token_file(&write(name, contents), "--token-file").as_deref(),
                Ok("s3cret")
            );
        }
        assert_eq!(
            read_token_file(&write("blank-line", "s3cret\n\n"), "--token-file")
                .expect_err("only one trailing newline is ignored"),
            format!(
                "--token-file: '{}' must hold a single-line token without control characters",
                directory.join("blank-line").display()
            )
        );
        for (name, contents) in [("empty", ""), ("newline-only", "\n")] {
            let error = read_token_file(&write(name, contents), "--token-file")
                .expect_err("an empty token file is a configuration error");
            assert!(error.contains("is empty"), "{error}");
        }
        let oversized = "x".repeat(MAX_TOKEN_FILE_BYTES as usize + 1);
        let error = read_token_file(&write("oversized", &oversized), "--token-file")
            .expect_err("an oversized file is not a token");
        assert!(error.contains("not a token file"), "{error}");
        let error = read_token_file(&directory.join("missing"), "--auth-token-file")
            .expect_err("a missing file is reported");
        assert!(
            error.starts_with("--auth-token-file: cannot open"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn token_sources_resolve_in_flag_file_environment_order() {
        let directory = test_directory("token-order");
        let file = write_private_file(&directory, "token", "from-file\n");
        let env = || Some("from-env".to_string());

        assert_eq!(
            resolve_token(Some("from-flag".into()), Some(&file), "--token-file", env()),
            Ok(Some("from-flag".to_string()))
        );
        assert_eq!(
            resolve_token(None, Some(&file), "--token-file", env()),
            Ok(Some("from-file".to_string()))
        );
        assert_eq!(
            resolve_token(None, None, "--token-file", env()),
            Ok(Some("from-env".to_string()))
        );
        assert_eq!(
            resolve_token(None, None, "--token-file", Some(String::new())),
            Ok(None)
        );
        assert_eq!(resolve_token(None, None, "--token-file", None), Ok(None));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn token_flags_and_token_files_are_mutually_exclusive() {
        for arguments in [
            vec![
                "hola",
                "worker",
                "--server",
                "http://127.0.0.1:8000",
                "--exec",
                "cmd",
                "--token",
                "a",
                "--token-file",
                "b",
            ],
            vec![
                "hola",
                "serve",
                "study.yaml",
                "--auth-token",
                "a",
                "--auth-token-file",
                "b",
            ],
            vec![
                "hola",
                "serve",
                "study.yaml",
                "--read-token",
                "a",
                "--read-token-file",
                "b",
            ],
        ] {
            assert!(
                Cli::try_parse_from(arguments).is_err(),
                "a token and its file must not both be given"
            );
        }
        let cli = Cli::try_parse_from([
            "hola",
            "serve",
            "study.yaml",
            "--auth-token-file",
            "write.token",
            "--read-token-file",
            "read.token",
        ])
        .expect("token files should parse");
        match cli.command {
            Commands::Serve {
                auth_token,
                auth_token_file,
                read_token,
                read_token_file,
                ..
            } => {
                assert!(auth_token.is_none() && read_token.is_none());
                assert_eq!(auth_token_file, Some(PathBuf::from("write.token")));
                assert_eq!(read_token_file, Some(PathBuf::from("read.token")));
            }
            Commands::Worker { .. } => panic!("expected serve command"),
        }
    }

    // ── Redirects ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn redirects_are_reported_and_never_followed() {
        // The redirect target must never see a request, with or without the
        // bearer token.
        let target = std::net::TcpListener::bind("127.0.0.1:0").expect("target should bind");
        target
            .set_nonblocking(true)
            .expect("target should be non-blocking");
        let target_url = format!("http://{}/api/cancel", target.local_addr().unwrap());

        let (server, server_thread) = mock_http_server(vec![
            MockReply::redirect(307, &target_url),
            MockReply::redirect(307, &target_url),
            // HTTP forbids C0 controls in a header value, but not the C1 and
            // bidirectional controls that UTF-8 can carry.
            MockReply::redirect(307, "http://evil.example/\u{9b}31m\u{202e}gnp.exe"),
        ]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        let error = cancel_trial(&client, &server, Some("secret"), 3)
            .await
            .expect_err("a redirected mutation must fail");
        assert!(error.contains("was redirected"), "{error}");
        assert!(error.contains("HTTP 307 Temporary Redirect"), "{error}");
        assert!(error.contains(&target_url), "{error}");
        assert!(error.contains("does not follow redirects"), "{error}");

        let record = PendingTell::new(&server, 3, serde_json::json!({"loss": 1.0}));
        let tell_error = deliver_tell(&client, Some("secret"), &record)
            .await
            .expect_err("a redirected tell must fail");
        assert!(
            tell_error.is_permanent(),
            "a redirect is a configuration error, not a transient failure"
        );

        let hostile = callback_trial_state(&client, &server, Some("secret"), 3)
            .await
            .expect_err("a redirected read must fail");
        assert!(hostile.contains("was redirected"), "{hostile}");
        assert!(
            hostile.contains("\\u{009b}31m\\u{202e}gnp.exe"),
            "the Location header is server-controlled: {hostile:?}"
        );
        assert!(!hostile.chars().any(|c| c.is_control() || c == '\u{202e}'));

        server_thread.join().expect("mock server should finish");
        match target.accept() {
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            other => panic!("the redirect target received a connection: {other:?}"),
        }
    }

    // ── Bounded response bodies ─────────────────────────────────────────

    #[tokio::test]
    async fn oversized_acknowledgements_are_rejected_before_and_while_reading() {
        let oversized = format!(
            r#"{{"status":"ok","trial_id":7,"padding":"{}"}}"#,
            "x".repeat(MAX_RESPONSE_BODY_BYTES)
        );
        let (server, server_thread) = mock_http_server(vec![
            MockReply::response(200, oversized.clone()),
            MockReply::without_content_length(200, oversized),
            MockReply::without_content_length(200, r#"{"status":"ok","trial_id":7}"#),
        ]);
        let client = build_http_client(Duration::from_secs(5)).expect("client should build");

        let declared = cancel_trial(&client, &server, None, 7)
            .await
            .expect_err("a declared oversized body must be rejected");
        assert!(
            declared.contains("exceeds the 1048576-byte limit"),
            "{declared}"
        );
        assert!(declared.contains("response body of"), "{declared}");

        let streamed = cancel_trial(&client, &server, None, 7)
            .await
            .expect_err("an undeclared oversized body must be rejected while streaming");
        assert!(
            streamed.contains("exceeds the 1048576-byte limit"),
            "{streamed}"
        );

        cancel_trial(&client, &server, None, 7)
            .await
            .expect("a small undeclared body is still accepted");
        server_thread.join().expect("mock server should finish");
    }

    #[tokio::test]
    async fn oversized_error_bodies_are_cut_instead_of_buffered() {
        let (server, server_thread) = mock_http_server(vec![MockReply::without_content_length(
            503,
            "e".repeat(4 * MAX_ERROR_BODY_BYTES),
        )]);
        let client = build_http_client(Duration::from_secs(5)).expect("client should build");

        let error = cancel_trial(&client, &server, None, 7)
            .await
            .expect_err("503 must fail");
        assert!(error.contains("HTTP 503 Service Unavailable"), "{error}");
        assert!(
            error.len() < HTTP_ERROR_SNIPPET_CHARS + 256,
            "the quoted body is bounded, got {} bytes",
            error.len()
        );
        assert!(error.ends_with("..."), "a cut body is marked: {error}");
        server_thread.join().expect("mock server should finish");
    }

    // ── Terminal-safe output ────────────────────────────────────────────

    #[test]
    fn sanitize_for_terminal_escapes_controls_and_bounds_length() {
        assert_eq!(
            sanitize_for_terminal("plain text é ✓", 64),
            "plain text é ✓"
        );
        assert_eq!(
            sanitize_for_terminal("a\r\nb\tc\u{1b}[2J\u{7}\u{7f}\u{9b}", 64),
            "a\\r\\nb\\tc\\u{001b}[2J\\u{0007}\\u{007f}\\u{009b}"
        );
        assert_eq!(
            sanitize_for_terminal("x\u{202e}y\u{2028}z\u{2066}", 64),
            "x\\u{202e}y\\u{2028}z\\u{2066}"
        );
        assert_eq!(sanitize_for_terminal("abcdef", 3), "abc...");
        assert_eq!(sanitize_for_terminal("abc", 3), "abc");
        assert_eq!(sanitize_for_terminal("", 3), "");
    }

    #[tokio::test]
    async fn server_controlled_error_text_cannot_inject_terminal_controls() {
        let (server, server_thread) = mock_http_server(vec![
            MockReply::response(
                409,
                "{\"error\":\"denied\u{1b}[2J\"}\r\nTrial 7 completed successfully",
            ),
            MockReply::response(
                200,
                r#"{"status":"ok","trial_id":7,"state":"pending\u001b[31m\nCompleted trial 7"}"#,
            ),
        ]);
        let client = build_http_client(Duration::from_secs(2)).expect("client should build");

        let cancel_error = cancel_trial(&client, &server, None, 7)
            .await
            .expect_err("409 cancel must fail");
        assert!(cancel_error.contains("\\u{001b}[2J"), "{cancel_error}");
        assert!(cancel_error.contains("\\r\\nTrial 7"), "{cancel_error}");

        let state_error = callback_trial_state(&client, &server, None, 7)
            .await
            .expect_err("an unknown state must fail");
        assert!(
            state_error.contains("unsupported trial state"),
            "{state_error}"
        );

        for message in [cancel_error, state_error] {
            assert!(
                !message.chars().any(char::is_control),
                "control characters reached the terminal: {message:?}"
            );
        }
        server_thread.join().expect("mock server should finish");
    }

    #[test]
    fn post_commit_warnings_are_neutralized_and_bounded() {
        let long = "w".repeat(POST_COMMIT_WARNING_CHARS + 50);
        let mut warnings = vec!["refit failed\r\n\u{1b}[1mCompleted trial 9", long.as_str()];
        warnings.resize(MAX_POST_COMMIT_WARNINGS + 5, "another");

        let printable = printable_post_commit_warnings(&warnings);
        assert_eq!(printable.len(), MAX_POST_COMMIT_WARNINGS + 1);
        assert_eq!(
            printable[0],
            "refit failed\\r\\n\\u{001b}[1mCompleted trial 9"
        );
        assert_eq!(printable[1].chars().count(), POST_COMMIT_WARNING_CHARS + 3);
        assert_eq!(
            printable.last().map(String::as_str),
            Some("5 further post-commit warnings were not shown")
        );
        assert!(
            printable
                .iter()
                .all(|warning| !warning.chars().any(char::is_control))
        );
        assert_eq!(
            printable_post_commit_warnings(&["only one"]),
            vec!["only one".to_string()]
        );
    }

    // ── Outbox trust ────────────────────────────────────────────────────

    #[cfg(unix)]
    #[test]
    fn outbox_directories_and_records_are_created_private() {
        use std::os::unix::fs::PermissionsExt;
        let parent = test_directory("outbox-modes");
        let root = parent.join("nested").join("outbox");
        let server = "http://127.0.0.1:8000";
        let outbox = TellOutbox::open(&root, server).expect("outbox should open");
        let record = PendingTell::new(server, 5, serde_json::json!({"loss": 0.5}));
        let path = outbox.persist(&record).expect("record should persist");

        let mode = |path: &Path| {
            std::fs::symlink_metadata(path)
                .expect("entry should exist")
                .permissions()
                .mode()
                & 0o777
        };
        assert_eq!(mode(&root), 0o700);
        assert_eq!(mode(&outbox.directory), 0o700);
        assert_eq!(mode(&path), 0o600);
        let _ = std::fs::remove_dir_all(parent);
    }

    #[cfg(unix)]
    #[test]
    fn outbox_refuses_directories_another_user_could_control() {
        use std::os::unix::fs::PermissionsExt;
        let server = "http://127.0.0.1:8000";
        let hashed = format!("{:016x}", stable_server_hash(server));

        // A symbolic link planted under the predictable server directory name.
        let root = test_directory("outbox-symlink");
        let elsewhere = test_directory("outbox-symlink-target");
        std::os::unix::fs::symlink(&elsewhere, root.join(&hashed)).expect("symlink");
        let error = TellOutbox::open(&root, server)
            .err()
            .expect("a symlinked server directory must be refused");
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(error.to_string().contains("is a symbolic link"), "{error}");
        assert!(error.to_string().contains("--outbox-dir"), "{error}");

        // The same for the root the operator named.
        let linked_root = root.join("linked-root");
        std::os::unix::fs::symlink(&elsewhere, &linked_root).expect("symlink");
        let error = TellOutbox::open(&linked_root, server)
            .err()
            .expect("a symlinked root must be refused");
        assert!(error.to_string().contains("is a symbolic link"), "{error}");

        // A pre-existing directory that its group or anyone else can write to.
        for mode in [0o777, 0o770, 0o702] {
            let root = test_directory("outbox-writable");
            let directory = root.join(&hashed);
            std::fs::create_dir(&directory).expect("directory");
            std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(mode))
                .expect("chmod");
            let error = TellOutbox::open(&root, server)
                .err()
                .expect("a directory writable by others must be refused");
            assert!(
                error
                    .to_string()
                    .contains("writable by its group or by other users"),
                "{error}"
            );
            let _ = std::fs::remove_dir_all(root);
        }

        // A queue that was private when opened and loosened afterwards.
        let reopened = test_directory("outbox-loosened");
        let outbox = TellOutbox::open(&reopened, server).expect("outbox should open");
        std::fs::set_permissions(&outbox.directory, std::fs::Permissions::from_mode(0o777))
            .expect("chmod");
        assert!(outbox.pending().is_err(), "every use re-checks the queue");
        let record = PendingTell::new(server, 1, serde_json::json!({"loss": 0.5}));
        assert!(outbox.persist(&record).is_err());

        for directory in [root, elsewhere, reopened] {
            let _ = std::fs::remove_dir_all(directory);
        }
    }

    #[cfg(unix)]
    #[test]
    fn outbox_directory_ownership_policy() {
        assert_eq!(directory_trust_problem(1000, 0o40700, 1000), None);
        assert_eq!(directory_trust_problem(1000, 0o40755, 1000), None);
        assert_eq!(
            directory_trust_problem(0, 0o40755, 1000),
            None,
            "root can already act as any user"
        );
        let foreign = directory_trust_problem(1001, 0o40700, 1000)
            .expect("another user's directory is not trusted");
        assert!(foreign.contains("owned by uid 1001"), "{foreign}");
        assert!(
            directory_trust_problem(1000, 0o40700, 0).is_some(),
            "root must not trust a queue an unprivileged user owns"
        );
        for mode in [0o40720, 0o40702, 0o41777] {
            assert!(directory_trust_problem(1000, mode, 1000).is_some());
        }
    }

    #[cfg(unix)]
    #[test]
    fn outbox_never_reads_records_through_links_or_special_files() {
        let root = test_directory("outbox-record-link");
        let server = "http://127.0.0.1:8000";
        let outbox = TellOutbox::open(&root, server).expect("outbox should open");

        // A well-formed record that lives outside the queue.
        let outside = root.join("outside.json");
        let forged = PendingTell::new(server, 9, serde_json::json!({"loss": -1.0}));
        std::fs::write(&outside, serde_json::to_vec(&forged).unwrap()).expect("forged record");
        let link = outbox.record_path(9);
        std::os::unix::fs::symlink(&outside, &link).expect("symlink");

        let error = outbox
            .read_record(&link)
            .expect_err("a symlinked record must not be read");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("not a regular file"), "{error}");
        assert!(outbox.pending().is_err(), "the queue reports the bad entry");
        assert!(
            outbox.persist(&forged).is_err(),
            "a link is not mistaken for an equal existing record"
        );
        std::fs::remove_file(&link).expect("remove link");

        // A directory under a record name is rejected rather than read.
        std::fs::create_dir(outbox.record_path(10)).expect("directory");
        let error = outbox
            .read_record(&outbox.record_path(10))
            .expect_err("a directory is not a record");
        assert!(error.to_string().contains("not a regular file"), "{error}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn outbox_rejects_oversized_records() {
        let root = test_directory("outbox-oversized");
        let server = "http://127.0.0.1:8000";
        let outbox = TellOutbox::open(&root, server).expect("outbox should open");
        let path = outbox.record_path(4);
        std::fs::write(&path, vec![b' '; MAX_OUTBOX_RECORD_BYTES as usize + 1])
            .expect("oversized record");
        let error = outbox
            .read_record(&path)
            .expect_err("an oversized record must be rejected");
        assert!(
            error.to_string().contains("exceeds the size limit"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
