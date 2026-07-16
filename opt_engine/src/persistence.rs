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

//! Persistence layer for checkpointing optimization state.
//!
//! Provides `Checkpoint` for saving and restoring the complete optimization state:
//! - Trial history (leaderboard)
//! - Strategy state
//! - Metadata (timestamps, trial counts, etc.)
//!
//! # Design Principles
//!
//! - **Self-contained**: Checkpoints depend only on the leaderboard and strategy state
//! - **Flexible serialization**: Uses serde for format-agnostic persistence
//! - **Minimal coupling**: Only requires types to implement Serialize/Deserialize

use crate::leaderboard::Leaderboard;
use chrono::Utc;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Serde adapter that represents non-finite floating-point values with an
/// explicit tagged value instead of silently turning them into JSON `null`.
///
/// The adapter operates recursively so scalar and structured observations use
/// the same lossless representation. Maps containing either reserved marker
/// are escaped, preventing user data from being mistaken for an encoded float.
pub(crate) mod lossless_float {
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
    use serde_value::Value;
    use std::collections::BTreeMap;

    const FLOAT_MARKER: &str = "$hola.float";
    const MAP_MARKER: &str = "$hola.map";

    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: Serializer,
    {
        let value = serde_value::to_value(value).map_err(serde::ser::Error::custom)?;
        encode(value).serialize(serializer)
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
    where
        T: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let encoded = Value::deserialize(deserializer)?;
        let decoded = decode(encoded).map_err(D::Error::custom)?;
        T::deserialize(decoded).map_err(D::Error::custom)
    }

    fn float_tag(name: &str) -> Value {
        Value::Map(BTreeMap::from([(
            Value::String(FLOAT_MARKER.to_string()),
            Value::String(name.to_string()),
        )]))
    }

    fn non_finite_name(value: f64, width: &str) -> String {
        let kind = if value.is_nan() {
            "nan"
        } else if value.is_sign_positive() {
            "+inf"
        } else {
            "-inf"
        };
        format!("{width}:{kind}")
    }

    fn encode(value: Value) -> Value {
        match value {
            Value::F32(value) if !value.is_finite() => {
                float_tag(&non_finite_name(value as f64, "f32"))
            }
            Value::F64(value) if !value.is_finite() => float_tag(&non_finite_name(value, "f64")),
            Value::Option(value) => Value::Option(value.map(|value| Box::new(encode(*value)))),
            Value::Newtype(value) => Value::Newtype(Box::new(encode(*value))),
            Value::Seq(values) => Value::Seq(values.into_iter().map(encode).collect()),
            Value::Map(values) => {
                let needs_escape = values.keys().any(|key| {
                    matches!(key, Value::String(key) if key == FLOAT_MARKER || key == MAP_MARKER)
                });
                let encoded = values
                    .into_iter()
                    .map(|(key, value)| (encode(key), encode(value)))
                    .collect();
                if needs_escape {
                    Value::Map(BTreeMap::from([(
                        Value::String(MAP_MARKER.to_string()),
                        Value::Map(encoded),
                    )]))
                } else {
                    Value::Map(encoded)
                }
            }
            value => value,
        }
    }

    fn decode(value: Value) -> Result<Value, String> {
        match value {
            Value::Option(value) => Ok(Value::Option(
                value
                    .map(|value| decode(*value).map(Box::new))
                    .transpose()?,
            )),
            Value::Newtype(value) => Ok(Value::Newtype(Box::new(decode(*value)?))),
            Value::Seq(values) => Ok(Value::Seq(
                values.into_iter().map(decode).collect::<Result<_, _>>()?,
            )),
            Value::Map(mut values) => {
                if values.len() == 1 {
                    if let Some(tag) = values.remove(&Value::String(FLOAT_MARKER.to_string())) {
                        let tag = match tag {
                            Value::String(tag) => tag,
                            _ => return Err("invalid lossless float tag".to_string()),
                        };
                        return decode_float_tag(&tag);
                    }
                    if let Some(inner) = values.remove(&Value::String(MAP_MARKER.to_string())) {
                        let inner = match inner {
                            Value::Map(inner) => inner,
                            _ => return Err("invalid escaped map value".to_string()),
                        };
                        return decode_map_entries(inner);
                    }
                }
                decode_map_entries(values)
            }
            value => Ok(value),
        }
    }

    fn decode_float_tag(tag: &str) -> Result<Value, String> {
        let value = match tag {
            "f32:nan" => Value::F32(f32::NAN),
            "f32:+inf" => Value::F32(f32::INFINITY),
            "f32:-inf" => Value::F32(f32::NEG_INFINITY),
            "f64:nan" | "nan" => Value::F64(f64::NAN),
            "f64:+inf" | "+inf" => Value::F64(f64::INFINITY),
            "f64:-inf" | "-inf" => Value::F64(f64::NEG_INFINITY),
            _ => return Err(format!("unknown lossless float tag: {tag}")),
        };
        Ok(value)
    }

    fn decode_map_entries(values: BTreeMap<Value, Value>) -> Result<Value, String> {
        Ok(Value::Map(
            values
                .into_iter()
                .map(|(key, value)| Ok((decode(key)?, decode(value)?)))
                .collect::<Result<_, String>>()?,
        ))
    }
}

// =============================================================================
// Load Safety Constants
// =============================================================================

/// The format version this build writes.
///
/// Version 2 introduced the explicit lossless representation for non-finite
/// observations and full-engine runtime state. Version 1 remains readable for
/// migration; values that old JSON writers already collapsed to `null` cannot
/// be reconstructed, but intact finite checkpoints migrate transparently.
///
/// Version 3 adds fitted-model epoch state for GMM Gauss--Sobol' sampling. The
/// version gate prevents older binaries from silently ignoring that state and
/// continuing a checkpoint under incompatible sampling semantics. Versions 1
/// and 2 remain readable; a GMM loaded from either continues its legacy global
/// stream until the next successful model installation.
pub const CURRENT_FORMAT_VERSION: u32 = 3;
const MIN_SUPPORTED_FORMAT_VERSION: u32 = 1;

/// Maximum number of bytes accepted when loading a checkpoint.
///
/// Checkpoint files are local and produced by this server, so this is a
/// defense-in-depth bound rather than a network trust boundary. It caps the
/// memory and CPU a single (possibly corrupt) file can force a load to spend.
/// 512 MiB comfortably exceeds realistic checkpoints while preventing a
/// pathological file from exhausting memory. serde_json enforces its own
/// recursion limit, which guards against deeply nested structures.
pub const MAX_CHECKPOINT_BYTES: u64 = 512 * 1024 * 1024;

// =============================================================================
// Observation Kind
// =============================================================================

/// The kind of observation stored in a leaderboard checkpoint.
///
/// Persisting this explicitly lets a load select the correct concrete
/// leaderboard type instead of guessing from the current objective set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ObservationKind {
    /// Single scalar objective value per trial.
    Scalar,
    /// A vector of objective values per trial.
    Vector,
}

impl Default for ObservationKind {
    /// Back-compat default for checkpoints written before the tag existed,
    /// which stored a single scalar observation per trial.
    fn default() -> Self {
        ObservationKind::Scalar
    }
}

// =============================================================================
// Checkpoint Metadata
// =============================================================================

/// Metadata about a checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Unix timestamp when the checkpoint was created.
    pub created_at: u64,
    /// Human-readable timestamp (ISO 8601).
    pub created_at_iso: String,
    /// Number of trials at checkpoint time.
    pub n_trials: usize,
    /// Optional description or notes.
    pub description: Option<String>,
    /// Version of the checkpoint format.
    pub format_version: u32,
}

impl CheckpointMetadata {
    pub fn new(n_trials: usize, description: Option<String>) -> Self {
        let now = Utc::now();
        let timestamp = now.timestamp() as u64;
        let iso = now.format("%Y-%m-%dT%H:%M:%SZ").to_string();

        Self {
            created_at: timestamp,
            created_at_iso: iso,
            n_trials,
            description,
            format_version: CURRENT_FORMAT_VERSION,
        }
    }
}

// =============================================================================
// Checkpoint
// =============================================================================

/// A checkpoint containing all state needed to resume optimization.
///
/// Generic over:
/// - `D`: The domain type (candidate configurations)
/// - `Obs`: The observation type (results)
/// - `S`: The strategy state type
///
/// # Example
///
/// ```no_run
/// use opt_engine::{Checkpoint, ContinuousSpace, Leaderboard, RandomStrategy};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut leaderboard = Leaderboard::new();
/// leaderboard.push(0.25, 0.8);
/// let strategy_state = RandomStrategy::<ContinuousSpace>::new(42);
///
/// // Create checkpoint from current state
/// let checkpoint = Checkpoint::new(
///     leaderboard,
///     strategy_state,
///     Some("After 100 trials"),
/// );
///
/// // Save to file
/// checkpoint.save_json("checkpoint.json")?;
///
/// // Later, restore
/// let restored: Checkpoint<f64, f64, RandomStrategy<ContinuousSpace>> =
///     Checkpoint::load_json("checkpoint.json")?;
/// assert_eq!(restored.leaderboard.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    try_from = "CheckpointData<D, Obs, S>",
    bound(deserialize = "D: Deserialize<'de>, Obs: Deserialize<'de>, S: Deserialize<'de>")
)]
pub struct Checkpoint<D, Obs, S> {
    /// The trial history.
    pub leaderboard: Leaderboard<D, Obs>,
    /// The strategy state (if serializable).
    pub strategy_state: S,
    /// Checkpoint metadata.
    pub metadata: CheckpointMetadata,
}

#[derive(Deserialize)]
#[serde(bound(deserialize = "D: Deserialize<'de>, Obs: Deserialize<'de>, S: Deserialize<'de>"))]
struct CheckpointData<D, Obs, S> {
    leaderboard: Leaderboard<D, Obs>,
    strategy_state: S,
    metadata: CheckpointMetadata,
}

impl<D, Obs, S> TryFrom<CheckpointData<D, Obs, S>> for Checkpoint<D, Obs, S> {
    type Error = String;

    fn try_from(data: CheckpointData<D, Obs, S>) -> Result<Self, Self::Error> {
        validate_checkpoint_metadata(&data.metadata, data.leaderboard.len())?;
        let mut metadata = data.metadata;
        // Once migrated in memory, every subsequent save must advertise the
        // schema this build writes. Retaining an older label could let an old
        // reader accept newly serialized strategy state it does not understand.
        metadata.format_version = CURRENT_FORMAT_VERSION;
        Ok(Self {
            leaderboard: data.leaderboard,
            strategy_state: data.strategy_state,
            metadata,
        })
    }
}

impl<D, Obs, S> Checkpoint<D, Obs, S>
where
    D: Serialize + DeserializeOwned,
    Obs: Serialize + DeserializeOwned,
    S: Serialize + DeserializeOwned,
{
    pub fn new(
        leaderboard: Leaderboard<D, Obs>,
        strategy_state: S,
        description: Option<&str>,
    ) -> Self {
        let n_trials = leaderboard.len();
        Self {
            leaderboard,
            strategy_state,
            metadata: CheckpointMetadata::new(n_trials, description.map(String::from)),
        }
    }

    /// Save checkpoint as JSON to a file.
    /// Uses atomic write (write-to-temp + fsync + rename) to prevent data loss.
    pub fn save_json(&self, path: impl AsRef<Path>) -> io::Result<()> {
        atomic_write_json(path.as_ref(), |w| serde_json::to_writer_pretty(w, self))
    }

    /// Load checkpoint from a JSON file.
    ///
    /// Enforces a byte-size cap ([`MAX_CHECKPOINT_BYTES`]) and validates the
    /// recorded format version (cheaply, before the full parse) before returning.
    pub fn load_json(path: impl AsRef<Path>) -> io::Result<Self> {
        load_json_capped(path.as_ref())
    }

    /// Save checkpoint as compact JSON (no pretty-printing).
    /// Uses atomic write (write-to-temp + fsync + rename) to prevent data loss.
    pub fn save_json_compact(&self, path: impl AsRef<Path>) -> io::Result<()> {
        atomic_write_json(path.as_ref(), |w| serde_json::to_writer(w, self))
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse a checkpoint from a JSON string, validating the format version
    /// cheaply before the full parse.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        check_format_version_bytes(json.as_bytes()).map_err(serde::de::Error::custom)?;
        serde_json::from_str(json)
    }

    pub fn save_to_writer<W: Write>(&self, writer: W) -> io::Result<()> {
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Load a checkpoint from a reader with a byte-size cap and a cheap-fail
    /// format-version gate. The reader is bounded to [`MAX_CHECKPOINT_BYTES`].
    pub fn load_from_reader<R: Read>(reader: R) -> io::Result<Self> {
        let bytes = read_capped_reader(reader, MAX_CHECKPOINT_BYTES)?;
        deserialize_checked(&bytes)
    }
}

// =============================================================================
// Lightweight Checkpoint (leaderboard only, no strategy state)
// =============================================================================

/// A lightweight checkpoint containing only the trial history.
///
/// Use this when:
/// - The strategy is stateless or will be refit from the leaderboard
/// - You want minimal storage overhead
/// - The strategy state is not serializable
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    try_from = "LeaderboardCheckpointData<D, Obs>",
    bound(deserialize = "D: Deserialize<'de>, Obs: Deserialize<'de>")
)]
pub struct LeaderboardCheckpoint<D, Obs> {
    /// The trial history.
    pub leaderboard: Leaderboard<D, Obs>,
    /// Checkpoint metadata.
    pub metadata: CheckpointMetadata,
    /// The kind of observation stored in the leaderboard.
    ///
    /// Defaults to [`ObservationKind::Scalar`] for checkpoints written before
    /// this tag existed, preserving backward compatibility.
    #[serde(default)]
    pub observation_kind: ObservationKind,
}

#[derive(Deserialize)]
#[serde(bound(deserialize = "D: Deserialize<'de>, Obs: Deserialize<'de>"))]
struct LeaderboardCheckpointData<D, Obs> {
    leaderboard: Leaderboard<D, Obs>,
    metadata: CheckpointMetadata,
    #[serde(default)]
    observation_kind: ObservationKind,
}

impl<D, Obs> TryFrom<LeaderboardCheckpointData<D, Obs>> for LeaderboardCheckpoint<D, Obs> {
    type Error = String;

    fn try_from(data: LeaderboardCheckpointData<D, Obs>) -> Result<Self, Self::Error> {
        validate_checkpoint_metadata(&data.metadata, data.leaderboard.len())?;
        let mut metadata = data.metadata;
        metadata.format_version = CURRENT_FORMAT_VERSION;
        Ok(Self {
            leaderboard: data.leaderboard,
            metadata,
            observation_kind: data.observation_kind,
        })
    }
}

fn validate_checkpoint_metadata(
    metadata: &CheckpointMetadata,
    stored_trials: usize,
) -> Result<(), String> {
    check_format_version_value(metadata.format_version)?;
    if metadata.n_trials != stored_trials {
        return Err(format!(
            "checkpoint metadata n_trials {} does not match the {stored_trials} stored trials",
            metadata.n_trials
        ));
    }
    Ok(())
}

impl<D, Obs> LeaderboardCheckpoint<D, Obs>
where
    D: Serialize + DeserializeOwned,
    Obs: Serialize + DeserializeOwned,
{
    /// Create a checkpoint, defaulting the observation kind to scalar.
    ///
    /// Prefer [`LeaderboardCheckpoint::new_with_kind`] when the observation
    /// kind is known so loads do not rely on the back-compat default.
    pub fn new(leaderboard: Leaderboard<D, Obs>, description: Option<&str>) -> Self {
        Self::new_with_kind(leaderboard, description, ObservationKind::default())
    }

    /// Create a checkpoint and record the observation kind explicitly.
    pub fn new_with_kind(
        leaderboard: Leaderboard<D, Obs>,
        description: Option<&str>,
        observation_kind: ObservationKind,
    ) -> Self {
        let n_trials = leaderboard.len();
        Self {
            leaderboard,
            metadata: CheckpointMetadata::new(n_trials, description.map(String::from)),
            observation_kind,
        }
    }

    /// The observation kind recorded in this checkpoint.
    pub fn observation_kind(&self) -> ObservationKind {
        self.observation_kind
    }

    /// Save to JSON file.
    /// Uses atomic write (write-to-temp + fsync + rename) to prevent data loss.
    pub fn save_json(&self, path: impl AsRef<Path>) -> io::Result<()> {
        atomic_write_json(path.as_ref(), |w| serde_json::to_writer_pretty(w, self))
    }

    /// Load from JSON file.
    ///
    /// Enforces a byte-size cap ([`MAX_CHECKPOINT_BYTES`]) and validates the
    /// recorded format version (cheaply, before the full parse) before returning.
    pub fn load_json(path: impl AsRef<Path>) -> io::Result<Self> {
        load_json_capped(path.as_ref())
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse a checkpoint from a JSON string, validating the format version
    /// cheaply before the full parse.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        check_format_version_bytes(json.as_bytes()).map_err(serde::de::Error::custom)?;
        serde_json::from_str(json)
    }
}

// =============================================================================
// Atomic Write Helper
// =============================================================================

/// Read a checkpoint file into memory with a byte-size cap.
///
/// Rejects files whose reported length exceeds [`MAX_CHECKPOINT_BYTES`] before
/// reading them, and bounds the reader as a backstop against TOCTOU growth or
/// unknown lengths. Callers parse the returned bytes themselves (for example
/// with [`check_format_version_bytes`] followed by `serde_json::from_slice`),
/// so the production load paths share one capped read implementation.
pub fn read_checkpoint_capped(path: &Path) -> io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    if len > MAX_CHECKPOINT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "checkpoint file is {len} bytes, exceeding the {MAX_CHECKPOINT_BYTES}-byte limit"
            ),
        ));
    }
    read_capped_reader(BufReader::new(file), MAX_CHECKPOINT_BYTES)
}

/// Cheaply validate the recorded format version from raw checkpoint bytes.
///
/// On-disk checkpoints come in three shapes, two of which carry `metadata` at
/// the top level (legacy full and leaderboard-only) and one of which nests it
/// under a `checkpoint` key (the full-checkpoint wrapper written for
/// auto-checkpoints). The probe accepts `format_version` from either location,
/// parsing only that field and leaving the full payload untouched, so a
/// wrong-version (or huge-but-wrong-version) file is rejected before the
/// expensive typed deserialization. Returns the version-mismatch message on a
/// mismatch; a JSON shape that exposes the field in neither location is rejected
/// with a clear error.
pub fn check_format_version_bytes(bytes: &[u8]) -> Result<(), String> {
    #[derive(Deserialize)]
    struct VersionProbe {
        #[serde(default)]
        metadata: Option<MetadataProbe>,
        #[serde(default)]
        checkpoint: Option<CheckpointProbe>,
    }
    #[derive(Deserialize)]
    struct CheckpointProbe {
        #[serde(default)]
        metadata: Option<MetadataProbe>,
    }
    #[derive(Deserialize)]
    struct MetadataProbe {
        format_version: u32,
    }

    let probe: VersionProbe = serde_json::from_slice(bytes)
        .map_err(|e| format!("could not read checkpoint format_version: {e}"))?;

    // Prefer top-level metadata (legacy full / leaderboard-only); fall back to
    // metadata nested under the full-checkpoint wrapper.
    let version = probe
        .metadata
        .or_else(|| probe.checkpoint.and_then(|c| c.metadata))
        .map(|m| m.format_version)
        .ok_or_else(|| "could not locate format_version in checkpoint".to_string())?;

    check_format_version_value(version)
}

/// Deserialize a checkpoint JSON file with a byte-size cap and a cheap-fail
/// format-version gate.
///
/// The version is validated from the raw bytes before the expensive typed parse,
/// so a wrong-version file is rejected without fully deserializing the payload.
fn load_json_capped<T: DeserializeOwned>(path: &Path) -> io::Result<T> {
    let bytes = read_checkpoint_capped(path)?;
    deserialize_checked(&bytes)
}

/// Read at most `limit` bytes from a reader, erroring if the source yields more.
///
/// The reader is bounded to `limit + 1` so reaching the extra byte signals the
/// payload exceeded the cap. The limit is a parameter so tests can exercise the
/// over-limit branch with a tiny cap instead of allocating [`MAX_CHECKPOINT_BYTES`].
fn read_capped_reader<R: Read>(reader: R, limit: u64) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    reader.take(limit.saturating_add(1)).read_to_end(&mut buf)?;
    if buf.len() as u64 > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("checkpoint payload exceeds the {limit}-byte limit"),
        ));
    }
    Ok(buf)
}

/// Validate the format version from raw bytes, then fully deserialize them.
///
/// Used by the capped file/reader load paths so the cheap version gate runs
/// before the expensive typed parse.
fn deserialize_checked<T: DeserializeOwned>(bytes: &[u8]) -> io::Result<T> {
    check_format_version_bytes(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    serde_json::from_slice(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Validate that a recorded format version can be migrated by this build.
fn check_format_version_value(version: u32) -> Result<(), String> {
    if !(MIN_SUPPORTED_FORMAT_VERSION..=CURRENT_FORMAT_VERSION).contains(&version) {
        return Err(format!(
            "unsupported checkpoint format_version {version} (supported {MIN_SUPPORTED_FORMAT_VERSION} through {CURRENT_FORMAT_VERSION})"
        ));
    }
    Ok(())
}

/// Build a per-write-unique temp path in the same directory as `path`.
///
/// A shared, deterministic temp name (e.g. `path.with_extension("tmp")`) lets two
/// concurrent writers to the same target clobber each other's temp file. Making
/// the name unique per write (target file name + PID + a process-local atomic
/// counter) keeps concurrent writers isolated while staying in the same
/// directory so the final rename is atomic on the same filesystem.
pub fn unique_temp_path(path: &Path) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let base = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| "checkpoint".to_string());
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    dir.join(format!(".{base}.tmp.{pid}.{n}"))
}

/// Fsync the directory containing `path` so a preceding rename is durable on
/// Unix. Platforms that do not support opening directories (notably Windows)
/// rely on the write-through replacement primitive instead.
pub fn sync_parent_dir(path: &Path) -> io::Result<()> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let dir = dir.unwrap_or_else(|| Path::new("."));
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    std::fs::rename(source, destination)
}

/// Retry an I/O operation only while its error is classified as transient.
///
/// `max_retries` counts retries after the initial attempt. Keeping the retry
/// decision and wait strategy injectable makes the bounded behavior testable
/// without sleeping.
#[cfg(any(windows, test))]
fn retry_transient_io<T, F, P, W>(
    max_retries: usize,
    mut operation: F,
    is_transient: P,
    mut wait: W,
) -> io::Result<T>
where
    F: FnMut() -> io::Result<T>,
    P: Fn(&io::Error) -> bool,
    W: FnMut(usize),
{
    let mut retries = 0;
    loop {
        match operation() {
            Err(error) if retries < max_retries && is_transient(&error) => {
                wait(retries);
                retries += 1;
            }
            result => return result,
        }
    }
}

#[cfg(any(windows, test))]
fn is_transient_windows_replace_error(error: &io::Error) -> bool {
    // MoveFileExW can briefly lose a race with another replacer, an open
    // scanner, or filesystem bookkeeping. Do not retry unrelated failures
    // such as a missing directory, invalid path, or full disk.
    const ERROR_ACCESS_DENIED: i32 = 5;
    const ERROR_SHARING_VIOLATION: i32 = 32;
    const ERROR_LOCK_VIOLATION: i32 = 33;

    matches!(
        error.raw_os_error(),
        Some(ERROR_ACCESS_DENIED | ERROR_SHARING_VIOLATION | ERROR_LOCK_VIOLATION)
    )
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use std::time::Duration;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    const MAX_RETRIES: usize = 7;

    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();

    retry_transient_io(
        MAX_RETRIES,
        || {
            // SAFETY: both paths are encoded as owned, NUL-terminated UTF-16
            // buffers that remain alive for the duration of every call.
            let replaced = unsafe {
                MoveFileExW(
                    source.as_ptr(),
                    destination.as_ptr(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
                )
            };
            if replaced == 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        },
        is_transient_windows_replace_error,
        |retry| {
            // The configured exponential delays total 127 ms across all retries.
            const DELAYS_MS: [u64; MAX_RETRIES] = [1, 2, 4, 8, 16, 32, 64];
            std::thread::sleep(Duration::from_millis(DELAYS_MS[retry]));
        },
    )
}

/// Write JSON to a file atomically: write to a unique temp file, fsync, rename,
/// then fsync the parent directory. Prevents data loss if the process crashes
/// mid-write, and cleans up the temp file on every error path.
pub fn atomic_write_json<F>(path: &Path, write_fn: F) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> Result<(), serde_json::Error>,
{
    let (tmp, file) = (0..100)
        .find_map(|_| {
            let tmp = unique_temp_path(path);
            match OpenOptions::new().write(true).create_new(true).open(&tmp) {
                Ok(file) => Some(Ok((tmp, file))),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error)),
            }
        })
        .transpose()?
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not allocate a unique checkpoint temporary file",
            )
        })?;

    // After the temp file exists, remove it on any error so a failed write never
    // leaks a leftover temp.
    let result = (|| {
        let mut writer = BufWriter::new(file);
        write_fn(&mut writer).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let file = writer.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        replace_file(&tmp, path)
    })();

    match result {
        Ok(()) => {
            // Make the rename durable by fsyncing the parent directory.
            sync_parent_dir(path)?;
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

// =============================================================================
// Auto-Checkpointing Support
// =============================================================================

/// Configuration for automatic checkpointing.
#[derive(Clone, Debug)]
pub struct AutoCheckpointConfig {
    /// Directory to save checkpoints.
    pub directory: std::path::PathBuf,
    /// Checkpoint every N trials.
    interval: usize,
    /// Maximum number of checkpoints to keep (oldest are deleted).
    pub max_checkpoints: Option<usize>,
    /// Filename prefix.
    pub prefix: String,
}

impl Default for AutoCheckpointConfig {
    fn default() -> Self {
        Self {
            directory: std::path::PathBuf::from("."),
            interval: 50,
            max_checkpoints: Some(5),
            prefix: "checkpoint".to_string(),
        }
    }
}

impl AutoCheckpointConfig {
    /// Create a new config with the specified directory and non-zero interval.
    pub fn new(directory: impl Into<std::path::PathBuf>, interval: usize) -> Result<Self, String> {
        if interval == 0 {
            return Err("checkpoint interval must be at least 1".to_string());
        }
        Ok(Self {
            directory: directory.into(),
            interval,
            ..Default::default()
        })
    }

    /// Number of completed trials between checkpoints.
    pub fn interval(&self) -> usize {
        self.interval
    }

    /// Generate the filename for a checkpoint at the given trial count.
    pub fn filename(&self, n_trials: usize) -> std::path::PathBuf {
        self.directory
            .join(format!("{}_{:06}.json", self.prefix, n_trials))
    }

    /// Check if a checkpoint should be created at this trial count.
    pub fn should_checkpoint(&self, n_trials: usize) -> bool {
        n_trials > 0 && n_trials.is_multiple_of(self.interval)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_metadata_creation() {
        let meta = CheckpointMetadata::new(100, Some("test".to_string()));
        assert_eq!(meta.n_trials, 100);
        assert_eq!(meta.description, Some("test".to_string()));
        assert_eq!(meta.format_version, CURRENT_FORMAT_VERSION);
        assert!(meta.created_at > 0);
    }

    #[test]
    fn test_checkpoint_scalar_roundtrip() {
        let mut lb: Leaderboard<(f64, f64), f64> = Leaderboard::new();
        lb.push((0.1, 0.2), 0.5);
        lb.push((0.3, 0.4), 0.3);

        let strategy_state = vec![1.0, 2.0, 3.0]; // Mock strategy state

        let checkpoint = Checkpoint::new(lb, strategy_state, Some("test checkpoint"));

        let json = checkpoint.to_json().unwrap();
        let restored: Checkpoint<(f64, f64), f64, Vec<f64>> = Checkpoint::from_json(&json).unwrap();

        assert_eq!(restored.leaderboard.len(), 2);
        assert_eq!(restored.strategy_state, vec![1.0, 2.0, 3.0]);
        assert_eq!(
            restored.metadata.description,
            Some("test checkpoint".to_string())
        );
    }

    #[test]
    fn test_checkpoint_preserves_finite_float_bits() {
        // These values exercise decimals that serde_json's default
        // best-effort parser can round one ULP away from the emitted f64.
        for value in [0.956_369_757_652_282_7, 0.402_462_840_080_261_23] {
            let mut lb = Leaderboard::<serde_json::Value, f64>::new();
            lb.push(serde_json::json!({ "x": value }), value);
            let checkpoint = LeaderboardCheckpoint::new(lb, None);
            let restored: LeaderboardCheckpoint<serde_json::Value, f64> =
                LeaderboardCheckpoint::from_json(&checkpoint.to_json().unwrap()).unwrap();
            let trial = &restored.leaderboard.trials()[0];

            assert_eq!(
                trial.candidate["x"].as_f64().unwrap().to_bits(),
                value.to_bits()
            );
            assert_eq!(trial.observation.to_bits(), value.to_bits());
        }
    }

    #[test]
    fn test_checkpoint_multi_objective() {
        let mut lb: Leaderboard<String, BTreeMap<String, f64>> = Leaderboard::new();
        lb.push(
            "config_a".to_string(),
            [("loss".into(), 0.1), ("latency".into(), 50.0)].into(),
        );

        let checkpoint = LeaderboardCheckpoint::new(lb, None);
        let json = checkpoint.to_json().unwrap();
        let restored: LeaderboardCheckpoint<String, BTreeMap<String, f64>> =
            LeaderboardCheckpoint::from_json(&json).unwrap();

        assert_eq!(restored.leaderboard.len(), 1);
    }

    #[test]
    fn test_checkpoint_preserves_non_finite_scalar_observations() {
        for score in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let mut lb = Leaderboard::<String, f64>::new();
            lb.push("candidate".to_string(), score);
            let checkpoint = LeaderboardCheckpoint::new(lb, None);

            let json = checkpoint.to_json().unwrap();
            assert!(!json.contains("\"observation\": null"));
            assert!(json.contains("$hola.float"));

            let restored: LeaderboardCheckpoint<String, f64> =
                LeaderboardCheckpoint::from_json(&json).unwrap();
            let actual = restored.leaderboard.trials()[0].observation;
            assert!(
                (score.is_nan() && actual.is_nan()) || score == actual,
                "expected {score:?}, got {actual:?}"
            );
        }
    }

    #[test]
    fn test_checkpoint_file_preserves_non_finite_scalar_observations() {
        let mut lb = Leaderboard::<String, f64>::new();
        lb.push("infeasible".to_string(), f64::INFINITY);
        let checkpoint = LeaderboardCheckpoint::new(lb, None);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("non-finite.json");

        checkpoint.save_json(&path).unwrap();
        let restored: LeaderboardCheckpoint<String, f64> =
            LeaderboardCheckpoint::load_json(&path).unwrap();

        assert!(restored.leaderboard.trials()[0].observation.is_infinite());
    }

    #[test]
    fn test_checkpoint_preserves_non_finite_multi_objective_values() {
        let mut lb = Leaderboard::<String, BTreeMap<String, f64>>::new();
        lb.push(
            "candidate".to_string(),
            [
                ("finite".into(), 1.0),
                ("positive_infinity".into(), f64::INFINITY),
                ("negative_infinity".into(), f64::NEG_INFINITY),
                ("nan".into(), f64::NAN),
            ]
            .into(),
        );

        let checkpoint = LeaderboardCheckpoint::new(lb, None);
        let restored: LeaderboardCheckpoint<String, BTreeMap<String, f64>> =
            LeaderboardCheckpoint::from_json(&checkpoint.to_json().unwrap()).unwrap();
        let observation = &restored.leaderboard.trials()[0].observation;

        assert_eq!(observation["finite"], 1.0);
        assert_eq!(observation["positive_infinity"], f64::INFINITY);
        assert_eq!(observation["negative_infinity"], f64::NEG_INFINITY);
        assert!(observation["nan"].is_nan());
    }

    #[test]
    fn test_checkpoint_preserves_f32_and_reserved_map_keys() {
        let mut f32_lb = Leaderboard::<String, f32>::new();
        f32_lb.push("candidate".to_string(), f32::NEG_INFINITY);
        let f32_checkpoint = LeaderboardCheckpoint::new(f32_lb, None);
        let f32_restored: LeaderboardCheckpoint<String, f32> =
            LeaderboardCheckpoint::from_json(&f32_checkpoint.to_json().unwrap()).unwrap();
        assert_eq!(
            f32_restored.leaderboard.trials()[0].observation,
            f32::NEG_INFINITY
        );

        let mut map_lb = Leaderboard::<String, BTreeMap<String, String>>::new();
        map_lb.push(
            "candidate".to_string(),
            [
                ("$hola.float".to_string(), "not a tag".to_string()),
                ("$hola.map".to_string(), "not an escape".to_string()),
            ]
            .into(),
        );
        let map_checkpoint = LeaderboardCheckpoint::new(map_lb, None);
        let map_restored: LeaderboardCheckpoint<String, BTreeMap<String, String>> =
            LeaderboardCheckpoint::from_json(&map_checkpoint.to_json().unwrap()).unwrap();
        let observation = &map_restored.leaderboard.trials()[0].observation;
        assert_eq!(observation["$hola.float"], "not a tag");
        assert_eq!(observation["$hola.map"], "not an escape");
    }

    #[test]
    fn test_auto_checkpoint_config() {
        let config = AutoCheckpointConfig::new("/tmp/checkpoints", 10).unwrap();

        assert!(!config.should_checkpoint(0));
        assert!(!config.should_checkpoint(5));
        assert!(config.should_checkpoint(10));
        assert!(config.should_checkpoint(100));

        let path = config.filename(50);
        assert!(path.to_string_lossy().contains("checkpoint_000050.json"));
    }

    #[test]
    fn test_auto_checkpoint_rejects_zero_interval() {
        assert!(AutoCheckpointConfig::new("/tmp/checkpoints", 0).is_err());
    }

    #[test]
    fn test_lightweight_checkpoint() {
        let mut lb: Leaderboard<f64, f64> = Leaderboard::new();
        lb.push(0.5, 0.1);

        let checkpoint = LeaderboardCheckpoint::new(lb, Some("lightweight"));
        let json = checkpoint.to_json().unwrap();

        // Should contain leaderboard but no strategy state
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("leaderboard").is_some());
        assert!(parsed.get("strategy_state").is_none());
    }

    #[test]
    fn test_leaderboard_checkpoint_file_roundtrip() {
        let mut lb = Leaderboard::<f64, f64>::new();
        lb.push(1.0, 0.5);
        lb.push(2.0, 0.3);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lb.json");

        let ckpt = LeaderboardCheckpoint::new(lb, Some("test lb"));
        ckpt.save_json(&path).unwrap();

        let loaded: LeaderboardCheckpoint<f64, f64> =
            LeaderboardCheckpoint::load_json(&path).unwrap();
        assert_eq!(loaded.leaderboard.len(), 2);
        assert_eq!(loaded.metadata.description, Some("test lb".to_string()));
    }

    #[test]
    fn test_full_checkpoint_file_roundtrip() {
        let mut lb = Leaderboard::<f64, f64>::new();
        lb.push(1.0, 0.5);

        let strategy_state = vec![1.0, 2.0, 3.0];
        let ckpt = Checkpoint::new(lb, strategy_state, Some("full test"));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("full.json");
        ckpt.save_json(&path).unwrap();

        let loaded: Checkpoint<f64, f64, Vec<f64>> = Checkpoint::load_json(&path).unwrap();
        assert_eq!(loaded.leaderboard.len(), 1);
        assert_eq!(loaded.strategy_state, vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded.metadata.description, Some("full test".to_string()));
    }

    #[test]
    fn test_observation_kind_roundtrips() {
        let mut lb = Leaderboard::<f64, f64>::new();
        lb.push(1.0, 0.5);

        let ckpt =
            LeaderboardCheckpoint::new_with_kind(lb, Some("tagged"), ObservationKind::Vector);
        assert_eq!(ckpt.observation_kind(), ObservationKind::Vector);

        let json = ckpt.to_json().unwrap();
        // The tag serializes with lowercase variant names.
        assert!(json.contains("\"observation_kind\""));
        assert!(json.contains("\"vector\""));

        let restored: LeaderboardCheckpoint<f64, f64> =
            LeaderboardCheckpoint::from_json(&json).unwrap();
        assert_eq!(restored.observation_kind(), ObservationKind::Vector);
    }

    #[test]
    fn test_observation_kind_back_compat_default() {
        // An old checkpoint without the observation_kind tag still loads,
        // defaulting to Scalar.
        let json = r#"{
            "leaderboard": {"trials": [], "next_id": 0},
            "metadata": {
                "created_at": 1,
                "created_at_iso": "1970-01-01T00:00:01Z",
                "n_trials": 0,
                "description": null,
                "format_version": 1
            }
        }"#;
        let restored: LeaderboardCheckpoint<f64, f64> =
            LeaderboardCheckpoint::from_json(json).unwrap();
        assert_eq!(restored.observation_kind(), ObservationKind::Scalar);
        assert_eq!(restored.metadata.format_version, CURRENT_FORMAT_VERSION);
    }

    #[test]
    fn test_version_gate_rejects_wrong_format_version() {
        let mut lb = Leaderboard::<f64, f64>::new();
        lb.push(1.0, 0.5);
        let mut ckpt = LeaderboardCheckpoint::new(lb, None);
        ckpt.metadata.format_version = CURRENT_FORMAT_VERSION + 1;

        let json = serde_json::to_string(&ckpt).unwrap();
        let err = LeaderboardCheckpoint::<f64, f64>::from_json(&json).unwrap_err();
        assert!(err.to_string().contains("format_version"));

        // The file path also rejects the mismatched version.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_version.json");
        std::fs::write(&path, &json).unwrap();
        let err = LeaderboardCheckpoint::<f64, f64>::load_json(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_version_gate_accepts_migratable_versions_and_writes_v3() {
        assert!(check_format_version_value(1).is_ok());
        assert!(check_format_version_value(2).is_ok());
        assert!(check_format_version_value(3).is_ok());
        assert!(check_format_version_value(0).is_err());
        assert!(check_format_version_value(4).is_err());

        let checkpoint = LeaderboardCheckpoint::<f64, f64>::new(Leaderboard::new(), None);
        assert_eq!(checkpoint.metadata.format_version, 3);
    }

    #[test]
    fn test_checkpoint_rejects_mismatched_trial_count() {
        let mut lb = Leaderboard::<String, f64>::new();
        lb.push("candidate".to_string(), 1.0);
        let checkpoint = LeaderboardCheckpoint::new(lb, None);
        let mut value = serde_json::to_value(&checkpoint).unwrap();
        value["metadata"]["n_trials"] = serde_json::json!(2);

        let error = LeaderboardCheckpoint::<String, f64>::from_json(&value.to_string())
            .expect_err("mismatched metadata must be rejected");
        assert!(error.to_string().contains("n_trials"));
    }

    #[test]
    fn test_size_cap_rejects_oversized_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oversized.json");

        // Create a sparse file whose reported length exceeds the cap without
        // actually writing that many bytes.
        let file = File::create(&path).unwrap();
        file.set_len(MAX_CHECKPOINT_BYTES + 1).unwrap();
        drop(file);

        let err = LeaderboardCheckpoint::<f64, f64>::load_json(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("limit"));
    }

    #[test]
    fn test_capped_reader_backstop_rejects_oversized_payload() {
        // Drive the over-limit backstop with a tiny injected cap so the test
        // stays cheap: a reader that endlessly yields bytes must be rejected
        // once it exceeds the limit, without allocating MAX_CHECKPOINT_BYTES.
        let limit = 32u64;
        let err = read_capped_reader(io::repeat(b'x'), limit).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("limit"));
    }

    #[test]
    fn test_capped_reader_accepts_payload_at_limit() {
        // A payload exactly at the injected limit must be accepted; one byte
        // over must be rejected. This pins the boundary so the off-by-one in
        // the limit + 1 bound stays correct.
        let limit = 8u64;
        let exact = read_capped_reader(&b"01234567"[..], limit).unwrap();
        assert_eq!(exact.len(), 8);

        let over = read_capped_reader(&b"012345678"[..], limit).unwrap_err();
        assert_eq!(over.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_check_format_version_bytes_is_cheap_fail() {
        // The version probe must reject a wrong version without the payload
        // being a fully valid checkpoint: here the leaderboard field is garbage
        // that would fail a full typed parse, yet the version gate fires first.
        let bytes = br#"{
            "leaderboard": "not a leaderboard",
            "strategy_state": 12345,
            "metadata": {"format_version": 99}
        }"#;
        let err = check_format_version_bytes(bytes).unwrap_err();
        assert!(err.contains("format_version"));
        assert!(err.contains("99"));

        // A supported legacy version passes the probe even though the
        // surrounding payload is not a valid checkpoint, confirming the probe
        // performs only the cheap migration/version gate.
        let ok = br#"{"leaderboard": "junk", "metadata": {"format_version": 1}}"#;
        assert!(check_format_version_bytes(ok).is_ok());
    }

    #[test]
    fn test_check_format_version_bytes_accepts_nested_full_checkpoint() {
        // The full-checkpoint wrapper written for auto-checkpoints nests
        // metadata under a "checkpoint" key. The probe must find the version
        // there, accept a matching version, and reject a wrong one.
        let current = format!(
            r#"{{"config": "anything", "checkpoint": {{"leaderboard": "junk", "metadata": {{"format_version": {CURRENT_FORMAT_VERSION}}}}}}}"#
        );
        assert!(check_format_version_bytes(current.as_bytes()).is_ok());

        let wrong = format!(
            r#"{{"config": "anything", "checkpoint": {{"leaderboard": "junk", "metadata": {{"format_version": {}}}}}}}"#,
            CURRENT_FORMAT_VERSION + 1
        );
        let err = check_format_version_bytes(wrong.as_bytes()).unwrap_err();
        assert!(err.contains("format_version"));

        // A payload that exposes the version in neither location is rejected
        // with a clear, locate-specific error rather than a serde shape error.
        let missing = br#"{"config": "anything", "checkpoint": {"leaderboard": "junk"}}"#;
        let err = check_format_version_bytes(missing).unwrap_err();
        assert!(err.contains("could not locate format_version"));
    }

    #[test]
    fn test_retry_transient_io_retries_then_succeeds() {
        let mut attempts = 0;
        let mut waits = Vec::new();

        let result = retry_transient_io(
            3,
            || {
                attempts += 1;
                if attempts < 3 {
                    Err(io::Error::new(io::ErrorKind::WouldBlock, "busy"))
                } else {
                    Ok("done")
                }
            },
            |error| error.kind() == io::ErrorKind::WouldBlock,
            |retry| waits.push(retry),
        )
        .unwrap();

        assert_eq!(result, "done");
        assert_eq!(attempts, 3);
        assert_eq!(waits, vec![0, 1]);
    }

    #[test]
    fn test_retry_transient_io_does_not_retry_permanent_error() {
        let mut attempts = 0;
        let mut waits = 0;

        let error = retry_transient_io(
            7,
            || {
                attempts += 1;
                Err::<(), _>(io::Error::new(io::ErrorKind::InvalidInput, "permanent"))
            },
            |error| error.kind() == io::ErrorKind::WouldBlock,
            |_| waits += 1,
        )
        .unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(attempts, 1);
        assert_eq!(waits, 0);
    }

    #[test]
    fn test_retry_transient_io_stops_at_bound() {
        let mut attempts = 0;
        let mut waits = Vec::new();

        let error = retry_transient_io(
            2,
            || {
                attempts += 1;
                Err::<(), _>(io::Error::new(io::ErrorKind::WouldBlock, "still busy"))
            },
            |error| error.kind() == io::ErrorKind::WouldBlock,
            |retry| waits.push(retry),
        )
        .unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
        assert_eq!(attempts, 3, "initial attempt plus two retries");
        assert_eq!(waits, vec![0, 1]);
    }

    #[test]
    fn test_windows_replace_retry_classification_is_narrow() {
        for code in [5, 32, 33] {
            assert!(
                is_transient_windows_replace_error(&io::Error::from_raw_os_error(code)),
                "Windows error {code} should be retried"
            );
        }
        for code in [2, 3, 87, 112] {
            assert!(
                !is_transient_windows_replace_error(&io::Error::from_raw_os_error(code)),
                "permanent Windows error {code} must not be retried"
            );
        }
    }

    #[test]
    fn test_unique_temp_path_distinct_per_call() {
        // Two temp paths for the same target must differ (per-write uniqueness)
        // and live in the same directory so the final rename stays atomic.
        let target = std::path::Path::new("/tmp/checkpoints/ckpt.json");
        let a = unique_temp_path(target);
        let b = unique_temp_path(target);
        assert_ne!(a, b, "temp paths must be unique per call");
        assert_eq!(a.parent(), target.parent());
        assert_eq!(b.parent(), target.parent());
    }

    #[test]
    fn test_atomic_write_concurrent_saves_yield_valid_file() {
        // Two concurrent writers to the same target must each use a private temp
        // (so neither clobbers the other) and the final file must be valid JSON
        // written by exactly one of them.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concurrent.json");

        let p1 = path.clone();
        let p2 = path.clone();
        let h1 = std::thread::spawn(move || {
            for _ in 0..50 {
                atomic_write_json(&p1, |w| {
                    serde_json::to_writer(w, &serde_json::json!({"w": 1}))
                })
                .unwrap();
            }
        });
        let h2 = std::thread::spawn(move || {
            for _ in 0..50 {
                atomic_write_json(&p2, |w| {
                    serde_json::to_writer(w, &serde_json::json!({"w": 2}))
                })
                .unwrap();
            }
        });
        h1.join().unwrap();
        h2.join().unwrap();

        // The final file parses and holds one writer's payload.
        let contents = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&contents).unwrap();
        let w = value.get("w").and_then(|v| v.as_i64()).unwrap();
        assert!(
            w == 1 || w == 2,
            "final file must hold one writer's payload"
        );

        // No leftover temp files remain in the directory.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp."))
            .collect();
        assert!(
            leftovers.is_empty(),
            "no temp files should remain: {leftovers:?}"
        );
    }

    #[test]
    fn test_atomic_write_serialization_failure_leaves_no_temp() {
        // A write_fn that fails (here a forced serde error) must not leave a
        // leftover temp file behind, and must not create the target.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fail.json");

        use serde::ser::Error as _;
        let err = atomic_write_json(&path, |_w| {
            Err::<(), serde_json::Error>(serde_json::Error::custom("boom"))
        })
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        // The target was never created (rename never happened).
        assert!(!path.exists(), "target must not exist after a failed write");

        // No temp files remain in the directory.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            leftovers.is_empty(),
            "no temp files should remain after a failed write: {leftovers:?}"
        );
    }
}
