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
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Load Safety Constants
// =============================================================================

/// The format version this build writes and accepts. Loads validate the
/// checkpoint's recorded `format_version` against this and reject mismatches.
pub const CURRENT_FORMAT_VERSION: u32 = 1;

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
            format_version: 1,
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
/// ```ignore
/// use opt_engine::persistence::Checkpoint;
/// use opt_engine::leaderboard::Leaderboard;
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
/// let restored = Checkpoint::load_json("checkpoint.json")?;
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint<D, Obs, S> {
    /// The trial history.
    pub leaderboard: Leaderboard<D, Obs>,
    /// The strategy state (if serializable).
    pub strategy_state: S,
    /// Checkpoint metadata.
    pub metadata: CheckpointMetadata,
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

/// Validate that a recorded format version matches the version this build
/// understands. Returns a clear error on mismatch.
fn check_format_version_value(version: u32) -> Result<(), String> {
    if version != CURRENT_FORMAT_VERSION {
        return Err(format!(
            "unsupported checkpoint format_version {version} (expected {CURRENT_FORMAT_VERSION})"
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

/// Best-effort fsync of the directory containing `path` so a preceding rename is
/// durable. Opening and syncing a directory is unsupported on some platforms
/// (e.g. Windows), so failures here are ignored rather than propagated.
pub fn sync_parent_dir(path: &Path) {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let dir = dir.unwrap_or_else(|| Path::new("."));
    if let Ok(dir_file) = File::open(dir) {
        let _ = dir_file.sync_all();
    }
}

/// Write JSON to a file atomically: write to a unique temp file, fsync, rename,
/// then fsync the parent directory. Prevents data loss if the process crashes
/// mid-write, and cleans up the temp file on every error path.
fn atomic_write_json<F>(path: &Path, write_fn: F) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> Result<(), serde_json::Error>,
{
    let tmp = unique_temp_path(path);
    let file = File::create(&tmp)?;

    // After the temp file exists, remove it on any error so a failed write never
    // leaks a leftover temp.
    let result = (|| {
        let mut writer = BufWriter::new(file);
        write_fn(&mut writer).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let file = writer.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    })();

    match result {
        Ok(()) => {
            // Make the rename durable by fsyncing the parent directory.
            sync_parent_dir(path);
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
    pub interval: usize,
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
    /// Create a new config with the specified directory and interval.
    pub fn new(directory: impl Into<std::path::PathBuf>, interval: usize) -> Self {
        Self {
            directory: directory.into(),
            interval,
            ..Default::default()
        }
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
        assert_eq!(meta.format_version, 1);
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
    fn test_auto_checkpoint_config() {
        let config = AutoCheckpointConfig::new("/tmp/checkpoints", 10);

        assert!(!config.should_checkpoint(0));
        assert!(!config.should_checkpoint(5));
        assert!(config.should_checkpoint(10));
        assert!(config.should_checkpoint(100));

        let path = config.filename(50);
        assert!(path.to_string_lossy().contains("checkpoint_000050.json"));
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

        // A matching version passes the probe even though the surrounding
        // payload is not a valid checkpoint, confirming the probe ignores it.
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
