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

//! HOLA: Hyperparameter Optimization, Lightweight Asynchronous.
//!
//! This crate implements the HOLA system on top of [`opt_engine`]'s generic
//! optimization building blocks. It provides:
//!
//! - **[`HolaEngine`]** — the type-erased Ask/Tell engine for dynamic frontends
//!   (Python via PyO3, REST API, CLI). Operates on `serde_json::Value` instead
//!   of concrete Rust types.
//! - **REST server** (feature-gated behind `server`) — Axum HTTP server with
//!   SSE, CORS, and dashboard integration.
//!
//! For compile-time type-safe optimization, use [`opt_engine::Engine`] directly.

pub mod hola_engine;
#[cfg(feature = "server")]
pub mod server;

// Re-export at crate root for convenience
pub use hola_engine::{
    AutoStrategy, CompletedTrial, DynSpace, DynTrial, HolaEngine, ObjectiveConfig, ParamConfig,
    ParamInfo, StrategyConfig, StudyConfig,
};
