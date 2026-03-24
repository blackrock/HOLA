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
//! - **Decoupled from Engine**: Checkpoints are independent of the `Engine` struct
//! - **Flexible serialization**: Uses serde for format-agnostic persistence
//! - **Minimal coupling**: Only requires types to implement Serialize/Deserialize

use crate::leaderboard::Leaderboard;
use chrono::Utc;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

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
    pub fn load_json(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Save checkpoint as compact JSON (no pretty-printing).
    /// Uses atomic write (write-to-temp + fsync + rename) to prevent data loss.
    pub fn save_json_compact(&self, path: impl AsRef<Path>) -> io::Result<()> {
        atomic_write_json(path.as_ref(), |w| serde_json::to_writer(w, self))
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn save_to_writer<W: Write>(&self, writer: W) -> io::Result<()> {
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    pub fn load_from_reader<R: Read>(reader: R) -> io::Result<Self> {
        serde_json::from_reader(reader).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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
}

impl<D, Obs> LeaderboardCheckpoint<D, Obs>
where
    D: Serialize + DeserializeOwned,
    Obs: Serialize + DeserializeOwned,
{
    pub fn new(leaderboard: Leaderboard<D, Obs>, description: Option<&str>) -> Self {
        let n_trials = leaderboard.len();
        Self {
            leaderboard,
            metadata: CheckpointMetadata::new(n_trials, description.map(String::from)),
        }
    }

    /// Save to JSON file.
    /// Uses atomic write (write-to-temp + fsync + rename) to prevent data loss.
    pub fn save_json(&self, path: impl AsRef<Path>) -> io::Result<()> {
        atomic_write_json(path.as_ref(), |w| serde_json::to_writer_pretty(w, self))
    }

    /// Load from JSON file.
    pub fn load_json(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// =============================================================================
// Atomic Write Helper
// =============================================================================

/// Write JSON to a file atomically: write to a temp file, fsync, then rename.
/// Prevents data loss if the process crashes mid-write.
fn atomic_write_json<F>(path: &Path, write_fn: F) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> Result<(), serde_json::Error>,
{
    let tmp = path.with_extension("tmp");
    let file = File::create(&tmp)?;
    let mut writer = BufWriter::new(file);
    write_fn(&mut writer).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let file = writer.into_inner().map_err(|e| e.into_error())?;
    file.sync_all()?;
    std::fs::rename(&tmp, path)
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
}
