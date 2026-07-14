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

//! Gaussian Mixture Model (GMM) strategy for informed sampling.
//!
//! This strategy samples from a user-specified GMM in the standardized [0, 1]^n
//! hypercube. Samples are clipped to the unit hypercube (censored GMM).
//!
//! The GMM can be specified directly with parameters, or fitted from observed
//! normalized samples using the EM algorithm.
//!
//! # Performance Optimizations
//!
//! 1. **Fused E-Step/M-Step**: Single pass over data per iteration,
//!    eliminating the need to store the full N×K responsibility matrix.
//! 2. **Zero-Allocation Inner Loops**: Uses in-place Rank-1 updates (BLAS `ger`)
//!    instead of allocating intermediate matrices.
//! 3. **Single-Pass Covariance**: Uses `Cov = E[xx^T] - μμ^T` identity for
//!    efficient covariance computation without storing deviations.
//! 4. **Robust Numerics**: Handles singular covariances via regularization
//!    instead of panicking.
//! 5. **Cached Sampling Distribution**: Builds the component `WeightedIndex`
//!    only when validated parameters are constructed or deserialized.

use crate::traits::{StandardizedSpace, Strategy};
use nalgebra::{DMatrix, DVector, DVectorView};
use rand::SeedableRng;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

const WEIGHT_SUM_TOLERANCE: f64 = 1e-9;
const COVARIANCE_SYMMETRY_TOLERANCE: f64 = 1e-12;

/// Validation or numerical error produced by GMM construction and fitting.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum GmmError {
    EmptyDimension {
        context: &'static str,
    },
    ShapeMismatch {
        context: &'static str,
        expected: usize,
        rows: usize,
        columns: usize,
    },
    LengthMismatch {
        context: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteValue {
        context: &'static str,
        index: usize,
        value: f64,
    },
    OutOfCubeValue {
        context: &'static str,
        index: usize,
        value: f64,
    },
    InvalidPositiveValue {
        parameter: &'static str,
        value: f64,
    },
    InvalidCount {
        parameter: &'static str,
        value: usize,
        maximum: Option<usize>,
    },
    NonSymmetricCovariance {
        row: usize,
        column: usize,
    },
    CovarianceNotPositiveDefinite,
    EmptyMixture,
    InvalidWeight {
        index: usize,
        value: f64,
    },
    WeightsDoNotSumToOne {
        sum: f64,
    },
    ComponentDimensionMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    DeclaredDimensionMismatch {
        declared: usize,
        actual: usize,
    },
    EmptySamples,
    RaggedSample {
        sample: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteSample {
        sample: usize,
        dimension: usize,
        value: f64,
    },
    SampleOutOfCube {
        sample: usize,
        dimension: usize,
        value: f64,
    },
    NumericalFailure(&'static str),
    LockPoisoned(&'static str),
}

impl fmt::Display for GmmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension { context } => write!(f, "{context} must not be empty"),
            Self::ShapeMismatch {
                context,
                expected,
                rows,
                columns,
            } => write!(
                f,
                "{context} must be {expected}x{expected}, got {rows}x{columns}"
            ),
            Self::LengthMismatch {
                context,
                expected,
                actual,
            } => write!(f, "{context} length must be {expected}, got {actual}"),
            Self::NonFiniteValue {
                context,
                index,
                value,
            } => write!(f, "{context}[{index}] must be finite, got {value}"),
            Self::OutOfCubeValue {
                context,
                index,
                value,
            } => write!(f, "{context}[{index}] must be in [0, 1], got {value}"),
            Self::InvalidPositiveValue { parameter, value } => {
                write!(
                    f,
                    "{parameter} must be finite and greater than zero, got {value}"
                )
            }
            Self::InvalidCount {
                parameter,
                value,
                maximum,
            } => match maximum {
                Some(maximum) => write!(f, "{parameter} must be in 1..={maximum}, got {value}"),
                None => write!(f, "{parameter} must be at least 1, got {value}"),
            },
            Self::NonSymmetricCovariance { row, column } => write!(
                f,
                "covariance must be symmetric; entries ({row}, {column}) and ({column}, {row}) differ"
            ),
            Self::CovarianceNotPositiveDefinite => {
                write!(f, "covariance is not positive definite")
            }
            Self::EmptyMixture => write!(f, "a GMM must contain at least one component"),
            Self::InvalidWeight { index, value } => write!(
                f,
                "mixture weight {index} must be finite and greater than zero, got {value}"
            ),
            Self::WeightsDoNotSumToOne { sum } => {
                write!(f, "mixture weights must sum to 1, got {sum}")
            }
            Self::ComponentDimensionMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "component {index} has dimension {actual}, expected {expected}"
            ),
            Self::DeclaredDimensionMismatch { declared, actual } => write!(
                f,
                "serialized GMM declares dimension {declared}, but components have dimension {actual}"
            ),
            Self::EmptySamples => write!(f, "GMM fitting requires at least one sample"),
            Self::RaggedSample {
                sample,
                expected,
                actual,
            } => write!(
                f,
                "sample {sample} has dimension {actual}, expected {expected}"
            ),
            Self::NonFiniteSample {
                sample,
                dimension,
                value,
            } => write!(
                f,
                "sample {sample}, dimension {dimension} must be finite, got {value}"
            ),
            Self::SampleOutOfCube {
                sample,
                dimension,
                value,
            } => write!(
                f,
                "sample {sample}, dimension {dimension} must be in [0, 1], got {value}"
            ),
            Self::NumericalFailure(context) => write!(f, "GMM numerical failure: {context}"),
            Self::LockPoisoned(context) => write!(f, "GMM {context} lock is poisoned"),
        }
    }
}

impl std::error::Error for GmmError {}

// =============================================================================
// Core Structures
// =============================================================================

/// A Gaussian component optimized for fast evaluation.
///
/// Caches the Cholesky decomposition and log-determinant for efficient
/// repeated sampling and density evaluation.
///
/// # Serialization
///
/// Only the mean and covariance are serialized. The Cholesky decomposition
/// and log normalization constant are recomputed on deserialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "GaussianComponentSerde", into = "GaussianComponentSerde")]
pub struct GaussianComponent {
    /// Mean vector (dimensionality = n).
    mean: DVector<f64>,
    /// Covariance matrix (n × n), symmetric positive definite.
    covariance: DMatrix<f64>,
    /// Cached Lower Cholesky factor (L where Σ = L L^T).
    cholesky_l: DMatrix<f64>,
    /// Cached constant term: -0.5 * (d * ln(2π) + ln(det(Σ)))
    log_norm_const: f64,
}

impl GaussianComponent {
    /// Create a Gaussian component from an already positive-definite covariance.
    ///
    /// Returns [`GmmError`] for empty/non-finite/out-of-cube means, malformed
    /// covariance matrices, or covariance matrices that are not positive definite.
    pub fn new(mean: DVector<f64>, covariance: DMatrix<f64>) -> Result<Self, GmmError> {
        Self::build(mean, covariance, None)
    }

    /// Create a Gaussian component, explicitly regularizing a singular covariance.
    ///
    /// The regularization value must be finite and positive. Structural errors
    /// such as shape mismatches and non-symmetric matrices are never repaired.
    pub fn with_regularization(
        mean: DVector<f64>,
        covariance: DMatrix<f64>,
        reg: f64,
    ) -> Result<Self, GmmError> {
        if !reg.is_finite() || reg <= 0.0 {
            return Err(GmmError::InvalidPositiveValue {
                parameter: "regularization",
                value: reg,
            });
        }
        Self::build(mean, covariance, Some(reg))
    }

    fn build(
        mean: DVector<f64>,
        mut covariance: DMatrix<f64>,
        regularization: Option<f64>,
    ) -> Result<Self, GmmError> {
        let dim = mean.len();
        if dim == 0 {
            return Err(GmmError::EmptyDimension {
                context: "Gaussian mean",
            });
        }
        for (index, &value) in mean.iter().enumerate() {
            if !value.is_finite() {
                return Err(GmmError::NonFiniteValue {
                    context: "Gaussian mean",
                    index,
                    value,
                });
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(GmmError::OutOfCubeValue {
                    context: "Gaussian mean",
                    index,
                    value,
                });
            }
        }
        if covariance.nrows() != dim || covariance.ncols() != dim {
            return Err(GmmError::ShapeMismatch {
                context: "covariance",
                expected: dim,
                rows: covariance.nrows(),
                columns: covariance.ncols(),
            });
        }
        for (index, &value) in covariance.iter().enumerate() {
            if !value.is_finite() {
                return Err(GmmError::NonFiniteValue {
                    context: "covariance",
                    index,
                    value,
                });
            }
        }
        for row in 0..dim {
            for column in 0..row {
                let a = covariance[(row, column)];
                let b = covariance[(column, row)];
                let tolerance = COVARIANCE_SYMMETRY_TOLERANCE * a.abs().max(b.abs()).max(1.0);
                if (a - b).abs() > tolerance {
                    return Err(GmmError::NonSymmetricCovariance { row, column });
                }
            }
        }

        // Try Cholesky; if it fails, add regularization
        let chol = match covariance.clone().cholesky() {
            Some(c) => c,
            None => {
                let reg = regularization.ok_or(GmmError::CovarianceNotPositiveDefinite)?;
                for i in 0..dim {
                    covariance[(i, i)] += reg;
                }
                covariance
                    .clone()
                    .cholesky()
                    .ok_or(GmmError::CovarianceNotPositiveDefinite)?
            }
        };

        let l = chol.l();

        // Log determinant = 2 * sum(log(diag(L)))
        let log_det: f64 = 2.0 * l.diagonal().iter().map(|x| x.ln()).sum::<f64>();
        let log_norm_const = -0.5 * (dim as f64 * (2.0 * std::f64::consts::PI).ln() + log_det);
        if !log_norm_const.is_finite() {
            return Err(GmmError::NumericalFailure(
                "Gaussian normalization constant is non-finite",
            ));
        }

        Ok(Self {
            mean,
            covariance,
            cholesky_l: l,
            log_norm_const,
        })
    }

    /// Create an isotropic (spherical) Gaussian component.
    ///
    /// Returns [`GmmError`] when the mean or variance is invalid.
    pub fn isotropic(mean: DVector<f64>, variance: f64) -> Result<Self, GmmError> {
        if !variance.is_finite() || variance <= 0.0 {
            return Err(GmmError::InvalidPositiveValue {
                parameter: "variance",
                value: variance,
            });
        }
        let dim = mean.len();
        let covariance = DMatrix::identity(dim, dim) * variance;
        Self::new(mean, covariance)
    }

    /// Create a diagonal Gaussian component.
    ///
    /// Returns [`GmmError`] when the mean is invalid or the variances are not
    /// finite, positive, and dimensionally aligned with the mean.
    pub fn diagonal(mean: DVector<f64>, variances: DVector<f64>) -> Result<Self, GmmError> {
        let dim = mean.len();
        if variances.len() != dim {
            return Err(GmmError::LengthMismatch {
                context: "diagonal variances",
                expected: dim,
                actual: variances.len(),
            });
        }
        for &value in variances.iter() {
            if !value.is_finite() || value <= 0.0 {
                return Err(GmmError::InvalidPositiveValue {
                    parameter: "diagonal variance",
                    value,
                });
            }
        }
        let covariance = DMatrix::from_diagonal(&variances);
        Self::new(mean, covariance)
    }

    /// Sample from this Gaussian component.
    ///
    /// Uses the reparameterization: x = μ + L * z, where z ~ N(0, I).
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<DVector<f64>, GmmError> {
        let mut sample = Vec::with_capacity(self.dim());
        self.sample_into(rng, &mut sample)?;
        Ok(DVector::from_vec(sample))
    }

    /// Fill a caller-owned buffer with one sample without allocating.
    ///
    /// Rows are evaluated in reverse order so the standard-normal vector and
    /// matrix-product result can share the same storage safely.
    fn sample_into<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        sample: &mut Vec<f64>,
    ) -> Result<(), GmmError> {
        let dim = self.dim();
        sample.resize(dim, 0.0);
        for value in sample.iter_mut() {
            *value = StandardNormal.sample(rng);
        }
        for row in (0..dim).rev() {
            let mut value = self.mean[row];
            for (column, standard_normal) in sample.iter().copied().take(row + 1).enumerate() {
                value += self.cholesky_l[(row, column)] * standard_normal;
            }
            sample[row] = value;
        }
        if sample.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err(GmmError::NumericalFailure(
                "Gaussian sample contains a non-finite value",
            ))
        }
    }

    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    pub fn mean(&self) -> &DVector<f64> {
        &self.mean
    }

    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.covariance
    }

    /// Compute log probability density using a caller-owned solve buffer.
    #[inline]
    fn log_pdf_with_scratch(&self, x: &[f64], scratch: &mut [f64]) -> f64 {
        if x.len() != self.dim() || scratch.len() < self.dim() {
            return f64::NEG_INFINITY;
        }

        let mut mahal_sq = 0.0;
        for (row, &x_value) in x.iter().enumerate() {
            let mut value = x_value - self.mean[row];
            for (column, solved) in scratch.iter().copied().take(row).enumerate() {
                value -= self.cholesky_l[(row, column)] * solved;
            }
            let diagonal = self.cholesky_l[(row, row)];
            if !diagonal.is_finite() || diagonal <= 0.0 {
                return f64::NEG_INFINITY;
            }
            let solved = value / diagonal;
            if !solved.is_finite() {
                return f64::NEG_INFINITY;
            }
            scratch[row] = solved;
            mahal_sq += solved * solved;
        }
        self.log_norm_const - 0.5 * mahal_sq
    }
}

// Serde helper for GaussianComponent - only serializes mean and covariance
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GaussianComponentSerde {
    mean: DVector<f64>,
    covariance: DMatrix<f64>,
}

impl From<GaussianComponent> for GaussianComponentSerde {
    fn from(gc: GaussianComponent) -> Self {
        Self {
            mean: gc.mean,
            covariance: gc.covariance,
        }
    }
}

impl TryFrom<GaussianComponentSerde> for GaussianComponent {
    type Error = GmmError;

    fn try_from(value: GaussianComponentSerde) -> Result<Self, Self::Error> {
        Self::new(value.mean, value.covariance)
    }
}

/// Parameters for a Gaussian Mixture Model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "GmmParamsSerde", into = "GmmParamsSerde")]
pub struct GmmParams {
    /// Mixture weights (must sum to 1, all positive).
    weights: Vec<f64>,
    /// Gaussian components.
    components: Vec<GaussianComponent>,
    #[serde(skip)]
    component_distribution: WeightedIndex<f64>,
}

// Serde helper omits the cached component distribution and validates every
// persisted invariant before rebuilding it.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GmmParamsSerde {
    weights: Vec<f64>,
    components: Vec<GaussianComponent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dim: Option<usize>,
}

impl From<GmmParams> for GmmParamsSerde {
    fn from(value: GmmParams) -> Self {
        Self {
            weights: value.weights,
            components: value.components,
            dim: None,
        }
    }
}

impl TryFrom<GmmParamsSerde> for GmmParams {
    type Error = GmmError;

    fn try_from(value: GmmParamsSerde) -> Result<Self, Self::Error> {
        let params = Self::new(value.weights, value.components)?;
        match value.dim {
            Some(declared) if declared != params.dim() => {
                return Err(GmmError::DeclaredDimensionMismatch {
                    declared,
                    actual: params.dim(),
                });
            }
            _ => {}
        }
        Ok(params)
    }
}

impl GmmParams {
    /// Create a new GMM from weights and components.
    ///
    /// Returns [`GmmError`] unless the mixture is non-empty, all weights are
    /// finite and positive, weights sum to one, and component dimensions agree.
    pub fn new(weights: Vec<f64>, components: Vec<GaussianComponent>) -> Result<Self, GmmError> {
        if components.is_empty() {
            return Err(GmmError::EmptyMixture);
        }
        if weights.len() != components.len() {
            return Err(GmmError::LengthMismatch {
                context: "mixture weights",
                expected: components.len(),
                actual: weights.len(),
            });
        }
        for (index, &value) in weights.iter().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(GmmError::InvalidWeight { index, value });
            }
        }
        let sum: f64 = weights.iter().sum();
        if !sum.is_finite() || (sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
            return Err(GmmError::WeightsDoNotSumToOne { sum });
        }
        let dim = components[0].dim();
        for (index, component) in components.iter().enumerate().skip(1) {
            if component.dim() != dim {
                return Err(GmmError::ComponentDimensionMismatch {
                    index,
                    expected: dim,
                    actual: component.dim(),
                });
            }
        }
        let component_distribution = WeightedIndex::new(&weights)
            .map_err(|_| GmmError::NumericalFailure("invalid component distribution"))?;

        Ok(Self {
            weights,
            components,
            component_distribution,
        })
    }

    /// Create a single-component GMM (just a multivariate normal).
    pub fn single(component: GaussianComponent) -> Result<Self, GmmError> {
        Self::new(vec![1.0], vec![component])
    }

    /// Create a uniform GMM centered in the unit hypercube.
    ///
    /// Returns [`GmmError`] for a zero dimension or invalid variance.
    pub fn uniform_prior(dim: usize, variance: f64) -> Result<Self, GmmError> {
        if dim == 0 {
            return Err(GmmError::EmptyDimension {
                context: "uniform prior",
            });
        }
        let mean = DVector::from_element(dim, 0.5);
        Self::single(GaussianComponent::isotropic(mean, variance)?)
    }

    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    pub fn dim(&self) -> usize {
        self.components[0].dim()
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn components(&self) -> &[GaussianComponent] {
        &self.components
    }

    /// Sample from the GMM (unclamped).
    pub fn sample_unclamped<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<DVector<f64>, GmmError> {
        let idx = self.component_distribution.sample(rng);
        self.components[idx].sample(rng)
    }

    /// Sample from the GMM, clamped to [0, 1]^n.
    pub fn sample_clamped<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>, GmmError> {
        let mut sample = Vec::with_capacity(self.dim());
        self.sample_clamped_into(rng, &mut sample)?;
        Ok(sample)
    }

    /// Fill a caller-owned buffer with a clamped GMM sample.
    ///
    /// Once the buffer has capacity for [`Self::dim`] values, repeated calls
    /// reuse that allocation as well as the cached component distribution.
    pub fn sample_clamped_into<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        sample: &mut Vec<f64>,
    ) -> Result<(), GmmError> {
        let idx = self.component_distribution.sample(rng);
        self.components[idx].sample_into(rng, sample)?;
        for value in sample {
            *value = value.clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// Fit GMM parameters from normalized samples using EM algorithm.
    ///
    /// # Arguments
    /// * `samples` - Normalized samples in [0, 1]^n, each inner Vec is one sample
    /// * `n_components` - Number of mixture components to fit
    /// * `max_iters` - Maximum EM iterations
    /// * `tolerance` - Convergence tolerance for log-likelihood change
    /// * `reg` - Regularization added to covariance diagonal for numerical stability
    /// * `seed` - Seed for K-means++ initialization
    ///
    /// # Errors
    ///
    /// Returns [`GmmError`] for empty, ragged, non-finite, or out-of-cube
    /// samples; invalid algorithm parameters; or a numerical fitting failure.
    pub fn fit(
        samples: &[Vec<f64>],
        n_components: usize,
        max_iters: usize,
        tolerance: f64,
        reg: f64,
        seed: u64,
    ) -> Result<Self, GmmError> {
        if samples.is_empty() {
            return Err(GmmError::EmptySamples);
        }

        let dim = samples[0].len();
        if dim == 0 {
            return Err(GmmError::EmptyDimension {
                context: "GMM samples",
            });
        }
        for (sample_index, sample) in samples.iter().enumerate() {
            if sample.len() != dim {
                return Err(GmmError::RaggedSample {
                    sample: sample_index,
                    expected: dim,
                    actual: sample.len(),
                });
            }
            for (dimension, &value) in sample.iter().enumerate() {
                if !value.is_finite() {
                    return Err(GmmError::NonFiniteSample {
                        sample: sample_index,
                        dimension,
                        value,
                    });
                }
                if !(0.0..=1.0).contains(&value) {
                    return Err(GmmError::SampleOutOfCube {
                        sample: sample_index,
                        dimension,
                        value,
                    });
                }
            }
        }
        if n_components == 0 || n_components > samples.len() {
            return Err(GmmError::InvalidCount {
                parameter: "n_components",
                value: n_components,
                maximum: Some(samples.len()),
            });
        }
        if max_iters == 0 {
            return Err(GmmError::InvalidCount {
                parameter: "max_iters",
                value: max_iters,
                maximum: None,
            });
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(GmmError::InvalidPositiveValue {
                parameter: "tolerance",
                value: tolerance,
            });
        }
        if !reg.is_finite() || reg <= 0.0 {
            return Err(GmmError::InvalidPositiveValue {
                parameter: "regularization",
                value: reg,
            });
        }

        let n_samples = samples.len();

        // Flatten data for cache-friendly access (each sample is contiguous)
        let flat_data: Vec<f64> = samples.iter().flat_map(|s| s.iter().copied()).collect();

        // Initialize with K-means++
        let (mut weights, mut means, mut covs) =
            kmeans_pp_init(&flat_data, dim, n_samples, n_components, seed)?;

        let mut prev_ll = f64::NEG_INFINITY;
        let mut log_probs = vec![0.0f64; n_components]; // Reusable buffer
        let mut solve_scratch = vec![0.0f64; dim];

        for _iter in 0..max_iters {
            // Build components for E-step (computes Cholesky once per iteration)
            let components: Vec<GaussianComponent> = means
                .iter()
                .zip(covs.iter())
                .map(|(m, c)| GaussianComponent::with_regularization(m.clone(), c.clone(), reg))
                .collect::<Result<_, _>>()?;

            // Fused E-Step & M-Step: single pass over data
            let mut stats = SufficientStats::new(n_components, dim);
            let mut total_ll = 0.0;

            for sample_slice in flat_data.chunks_exact(dim) {
                let x = DVectorView::from_slice(sample_slice, dim);
                let mut max_log = f64::NEG_INFINITY;

                // E-Step: compute log responsibilities
                for (k, comp) in components.iter().enumerate() {
                    let log_w = weights[k].max(1e-300).ln();
                    let log_p = comp.log_pdf_with_scratch(sample_slice, &mut solve_scratch);
                    log_probs[k] = log_w + log_p;
                    if log_probs[k] > max_log {
                        max_log = log_probs[k];
                    }
                }
                if !max_log.is_finite() {
                    return Err(GmmError::NumericalFailure(
                        "all component log probabilities are non-finite",
                    ));
                }

                // Log-sum-exp for numerical stability
                let mut sum_exp = 0.0;
                for lp in log_probs.iter_mut() {
                    if *lp > f64::NEG_INFINITY {
                        *lp = (*lp - max_log).exp();
                        sum_exp += *lp;
                    } else {
                        *lp = 0.0;
                    }
                }

                if !sum_exp.is_finite() || sum_exp <= 1e-300 {
                    return Err(GmmError::NumericalFailure(
                        "responsibility normalization is non-finite or zero",
                    ));
                }
                total_ll += max_log + sum_exp.ln();

                // M-Step: accumulate sufficient statistics (zero-allocation)
                if sum_exp > 1e-20 {
                    let inv_sum = 1.0 / sum_exp;
                    for (k, &lp) in log_probs.iter().enumerate() {
                        let r = lp * inv_sum; // Posterior probability
                        if r > 1e-10 {
                            stats.weight_sum[k] += r;

                            // mean_sum += r * x (axpy: y = a*x + y, zero alloc)
                            stats.mean_sum[k].axpy(r, &x, 1.0);

                            // outer_sum += r * x * x^T (ger: M = a*x*y^T + M, zero alloc)
                            stats.outer_sum[k].ger(r, &x, &x, 1.0);
                        }
                    }
                }
            }

            if !total_ll.is_finite() {
                return Err(GmmError::NumericalFailure(
                    "total log likelihood is non-finite",
                ));
            }

            // Check convergence
            if (total_ll - prev_ll).abs() < tolerance {
                break;
            }
            prev_ll = total_ll;

            // Update parameters from sufficient statistics
            let total_weight: f64 = stats.weight_sum.iter().sum();
            if !total_weight.is_finite() || total_weight <= 0.0 {
                return Err(GmmError::NumericalFailure(
                    "total responsibility weight is non-finite or zero",
                ));
            }

            for k in 0..n_components {
                let nk = stats.weight_sum[k];

                // Prune dead components
                if nk < 1e-5 {
                    weights[k] = 1e-10;
                    covs[k] = DMatrix::identity(dim, dim);
                    continue;
                }

                weights[k] = nk / total_weight;

                // New mean: μ_k = Σ r_{ik} x_i / N_k
                let mu = &stats.mean_sum[k] / nk;

                // New covariance using identity: Cov = E[xx^T] - μμ^T
                // This enables single-pass computation without storing deviations
                let mut cov = &stats.outer_sum[k] / nk;
                cov.ger(-1.0, &mu, &mu, 1.0); // Subtract μμ^T in-place (zero alloc)

                // Add regularization
                for i in 0..dim {
                    cov[(i, i)] += reg;
                }

                means[k] = mu;
                covs[k] = cov;
            }
        }

        // Build final components, keeping weights aligned with survivors.
        //
        // Drop sub-threshold components first. A pruned component above
        // (weight set to 1e-10 with an identity covariance) would otherwise
        // survive `with_regularization` (an identity covariance passes
        // Cholesky) and inflate `n_components()` with a dead cluster. Filtering
        // out weight <= 1e-9 removes these before assembly; the renormalization
        // below restores the sum-to-one invariant over the genuine survivors.
        let survivors: Vec<(f64, GaussianComponent)> = weights
            .into_iter()
            .zip(means.into_iter().zip(covs))
            .filter(|(weight, _)| *weight > 1e-9)
            .map(|(weight, (mean, covariance))| {
                GaussianComponent::with_regularization(mean, covariance, reg)
                    .map(|component| (weight, component))
            })
            .collect::<Result<_, _>>()?;

        if survivors.is_empty() {
            return Err(GmmError::NumericalFailure(
                "all fitted components collapsed",
            ));
        }

        let (surviving_weights, final_components): (Vec<_>, Vec<_>) = survivors.into_iter().unzip();
        let w_sum: f64 = surviving_weights.iter().sum();
        if !w_sum.is_finite() || w_sum <= 0.0 {
            return Err(GmmError::NumericalFailure(
                "surviving component weights have an invalid sum",
            ));
        }
        let normalized_weights: Vec<f64> = surviving_weights.iter().map(|w| w / w_sum).collect();

        Self::new(normalized_weights, final_components)
    }
}

// =============================================================================
// Private Helper Types and Functions
// =============================================================================

/// Sufficient statistics for fused E/M step.
struct SufficientStats {
    weight_sum: Vec<f64>,
    mean_sum: Vec<DVector<f64>>,
    outer_sum: Vec<DMatrix<f64>>,
}

impl SufficientStats {
    fn new(k: usize, dim: usize) -> Self {
        Self {
            weight_sum: vec![0.0; k],
            mean_sum: vec![DVector::zeros(dim); k],
            outer_sum: vec![DMatrix::zeros(dim, dim); k],
        }
    }
}

/// K-means++ initialization for GMM fitting.
type GmmInitialization = (Vec<f64>, Vec<DVector<f64>>, Vec<DMatrix<f64>>);

fn kmeans_pp_init(
    flat_data: &[f64],
    dim: usize,
    n: usize,
    k: usize,
    seed: u64,
) -> Result<GmmInitialization, GmmError> {
    if dim == 0 || n == 0 || k == 0 || k > n || flat_data.len() != dim.saturating_mul(n) {
        return Err(GmmError::NumericalFailure(
            "invalid dimensions passed to k-means++ initialization",
        ));
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut means: Vec<DVector<f64>> = Vec::with_capacity(k);

    // First center: random sample
    let first_idx = rng.random_range(0..n);
    means.push(DVector::from_column_slice(
        &flat_data[first_idx * dim..(first_idx + 1) * dim],
    ));

    // Track minimum distance to any center for each point
    let mut min_dists = vec![f64::INFINITY; n];

    for _ in 1..k {
        let last_mean = means.last().ok_or(GmmError::NumericalFailure(
            "k-means++ lost its initial center",
        ))?;

        // Update minimum distances
        for (i, min_d) in min_dists.iter_mut().enumerate() {
            let s_slice = &flat_data[i * dim..(i + 1) * dim];
            let d = s_slice
                .iter()
                .zip(last_mean.iter())
                .map(|(value, mean)| (value - mean).powi(2))
                .sum();
            if d < *min_d {
                *min_d = d;
            }
        }

        // Sample next center proportional to squared distance
        let sum_dist: f64 = min_dists.iter().sum();
        if !sum_dist.is_finite() {
            return Err(GmmError::NumericalFailure(
                "k-means++ distance sum is non-finite",
            ));
        }
        if sum_dist <= 0.0 {
            // All points are already centers
            let idx = rng.random_range(0..n);
            means.push(DVector::from_column_slice(
                &flat_data[idx * dim..(idx + 1) * dim],
            ));
            continue;
        }

        let target = rng.random::<f64>() * sum_dist;
        let mut cumsum = 0.0;
        let mut next_idx = n - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= target {
                next_idx = i;
                break;
            }
        }
        means.push(DVector::from_column_slice(
            &flat_data[next_idx * dim..(next_idx + 1) * dim],
        ));
    }

    // Initialize with uniform weights and identity covariances
    let weights = vec![1.0 / k as f64; k];
    let covs = vec![DMatrix::identity(dim, dim); k];

    Ok((weights, means, covs))
}

// =============================================================================
// Strategy Wrapper
// =============================================================================

/// A GMM-based sampling strategy.
///
/// Samples from a Gaussian Mixture Model in the unit hypercube [0, 1]^n.
/// Points are clipped to the hypercube bounds (censored GMM).
///
/// # Example
///
/// ```
/// use nalgebra::DVector;
/// use opt_engine::{ContinuousSpace, ProductSpace, Strategy};
/// use opt_engine::strategies::{GaussianComponent, GmmParams, GmmStrategy};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a GMM with two components
/// let comp1 = GaussianComponent::isotropic(DVector::from_vec(vec![0.3, 0.3]), 0.01)?;
/// let comp2 = GaussianComponent::isotropic(DVector::from_vec(vec![0.7, 0.7]), 0.01)?;
/// let params = GmmParams::new(vec![0.5, 0.5], vec![comp1, comp2])?;
///
/// let space = ProductSpace {
///     a: ContinuousSpace::new(0.0, 1.0),
///     b: ContinuousSpace::new(0.0, 1.0),
/// };
/// let strategy = GmmStrategy::<ProductSpace<ContinuousSpace, ContinuousSpace>>::new(42, params);
/// let _point = strategy.suggest(&space);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct GmmStrategy<S, Obs = f64> {
    seed: u64,
    #[serde(
        serialize_with = "serialize_counter",
        deserialize_with = "deserialize_counter"
    )]
    counter: AtomicU64,
    /// GMM parameters (wrapped in RwLock for interior mutability during fitting).
    #[serde(
        serialize_with = "serialize_gmm_params",
        deserialize_with = "deserialize_gmm_params"
    )]
    params: Arc<RwLock<GmmParams>>,
    /// Caller-owned sampling buffer reused by the production suggestion path.
    ///
    /// The mutex permits concurrent `suggest` calls without sharing mutable
    /// storage unsafely. Scratch capacity is a runtime optimization and is not
    /// part of durable strategy state.
    #[serde(skip)]
    sample_scratch: Mutex<Vec<f64>>,
    /// Configuration for GMM refitting behavior.
    #[serde(default)]
    refit_config: GmmRefitConfig,
    #[serde(skip)]
    _marker: PhantomData<fn() -> (S, Obs)>,
}

fn serialize_counter<S>(val: &AtomicU64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_u64(val.load(Ordering::Relaxed))
}

fn deserialize_counter<'de, D>(deserializer: D) -> Result<AtomicU64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val = u64::deserialize(deserializer)?;
    Ok(AtomicU64::new(val))
}

fn serialize_gmm_params<S>(
    params: &Arc<RwLock<GmmParams>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let params = params.read().map_err(serde::ser::Error::custom)?;
    params.serialize(serializer)
}

fn deserialize_gmm_params<'de, D>(deserializer: D) -> Result<Arc<RwLock<GmmParams>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let params = GmmParams::deserialize(deserializer)?;
    Ok(Arc::new(RwLock::new(params)))
}

impl<S, Obs> GmmStrategy<S, Obs> {
    /// Create a new GMM strategy with the given seed and parameters.
    pub fn new(seed: u64, params: GmmParams) -> Self {
        let sample_scratch = Mutex::new(Vec::with_capacity(params.dim()));
        Self {
            seed,
            counter: AtomicU64::new(0),
            params: Arc::new(RwLock::new(params)),
            sample_scratch,
            refit_config: GmmRefitConfig::default(),
            _marker: PhantomData,
        }
    }

    /// Create a GMM strategy with an automatically chosen seed and the given parameters.
    pub fn auto_seed(params: GmmParams) -> Self {
        Self::new(rand::random(), params)
    }

    /// Create a GMM strategy with a uniform prior centered in the hypercube.
    pub fn uniform_prior(seed: u64, dim: usize, variance: f64) -> Result<Self, GmmError> {
        Ok(Self::new(seed, GmmParams::uniform_prior(dim, variance)?))
    }

    /// Update the GMM parameters.
    pub fn set_params(&self, params: GmmParams) -> Result<(), GmmError> {
        *self
            .params
            .write()
            .map_err(|_| GmmError::LockPoisoned("parameter write"))? = params;
        Ok(())
    }

    /// Get a clone of the current GMM parameters.
    pub fn params(&self) -> Result<GmmParams, GmmError> {
        Ok(self
            .params
            .read()
            .map_err(|_| GmmError::LockPoisoned("parameter read"))?
            .clone())
    }

    /// Advance the deterministic sampling/refit cursor without generating
    /// discarded samples. Used when importing history without strategy state.
    pub fn advance_to(&self, counter: u64) {
        self.counter.fetch_max(counter, Ordering::Relaxed);
    }

    /// Base seed recorded in checkpoints.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Next deterministic sampling/refit stream position.
    pub fn counter(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Fit the GMM to normalized samples.
    ///
    /// This updates the internal GMM parameters based on the provided samples.
    pub fn fit_from_samples(
        &self,
        samples: &[Vec<f64>],
        n_components: usize,
        max_iters: usize,
        tolerance: f64,
        reg: f64,
        seed: u64,
    ) -> Result<(), GmmError> {
        let fitted = GmmParams::fit(samples, n_components, max_iters, tolerance, reg, seed)?;
        self.set_params(fitted)
    }
}

impl<S, Obs> Clone for GmmStrategy<S, Obs> {
    fn clone(&self) -> Self {
        let params = self
            .params
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        Self {
            seed: self.seed,
            counter: AtomicU64::new(self.counter.load(Ordering::Relaxed)),
            sample_scratch: Mutex::new(Vec::with_capacity(params.dim())),
            params: Arc::new(RwLock::new(params)),
            refit_config: self.refit_config.clone(),
            _marker: PhantomData,
        }
    }
}

impl<S, Obs> Strategy for GmmStrategy<S, Obs>
where
    S: StandardizedSpace,
    Obs: Serialize + DeserializeOwned + Send + Sync + Clone + Debug + 'static,
{
    type Space = S;
    type Observation = Obs;

    fn suggest(&self, space: &Self::Space) -> S::Domain {
        let dim = space.dimensionality();
        let params = self
            .params
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let call_index = self.counter.fetch_add(1, Ordering::Relaxed);
        let call_seed = self
            .seed
            .wrapping_add(call_index.wrapping_mul(6364136223846793005));
        let mut rng = SmallRng::seed_from_u64(call_seed);
        let mut sample = self
            .sample_scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // If the fitted GMM dimensionality disagrees with the space (e.g. after
        // a refit on a differently-shaped space, or a recovered deserialized
        // model), the GMM sample cannot be mapped. Fall back to the cube center
        // so this hot path, which runs under the engine write lock inside axum/
        // PyO3 handlers, never panics.
        if params.dim() == dim {
            if let Err(error) = params.sample_clamped_into(&mut rng, &mut sample) {
                eprintln!(
                    "GmmStrategy::suggest: sampling failed ({error}), falling back to cube center"
                );
                sample.resize(dim, 0.5);
                sample.fill(0.5);
            }
        } else {
            eprintln!(
                "GmmStrategy::suggest: GMM dim {} != space dim {}, falling back to cube center",
                params.dim(),
                dim
            );
            sample.resize(dim, 0.5);
            sample.fill(0.5);
        }

        match space.from_unit_cube(&sample) {
            Some(domain) => domain,
            None => {
                // Mapping rejected the point; retry with a deterministic
                // in-cube center rather than panicking.
                eprintln!(
                    "GmmStrategy::suggest: from_unit_cube failed, falling back to cube center"
                );
                sample.resize(dim, 0.5);
                sample.fill(0.5);
                space
                    .from_unit_cube(&sample)
                    .expect("cube center must map within the space")
            }
        }
    }

    fn update(&mut self, _candidate: &S::Domain, _result: Obs) {
        // Nothing to update.
    }
}

// =============================================================================
// RefittableStrategy Implementation
// =============================================================================

use crate::traits::RefittableStrategy;

/// Configuration for GMM refitting.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "GmmRefitConfigSerde", into = "GmmRefitConfigSerde")]
pub struct GmmRefitConfig {
    /// Number of GMM components to fit.
    n_components: usize,
    /// Maximum EM iterations.
    max_iters: usize,
    /// Convergence tolerance.
    tolerance: f64,
    /// Regularization for covariance matrices.
    regularization: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GmmRefitConfigSerde {
    n_components: usize,
    max_iters: usize,
    tolerance: f64,
    regularization: f64,
}

impl From<GmmRefitConfig> for GmmRefitConfigSerde {
    fn from(value: GmmRefitConfig) -> Self {
        Self {
            n_components: value.n_components,
            max_iters: value.max_iters,
            tolerance: value.tolerance,
            regularization: value.regularization,
        }
    }
}

impl TryFrom<GmmRefitConfigSerde> for GmmRefitConfig {
    type Error = GmmError;

    fn try_from(value: GmmRefitConfigSerde) -> Result<Self, Self::Error> {
        Self::new(
            value.n_components,
            value.max_iters,
            value.tolerance,
            value.regularization,
        )
    }
}

impl GmmRefitConfig {
    pub fn new(
        n_components: usize,
        max_iters: usize,
        tolerance: f64,
        regularization: f64,
    ) -> Result<Self, GmmError> {
        if n_components == 0 {
            return Err(GmmError::InvalidCount {
                parameter: "n_components",
                value: n_components,
                maximum: None,
            });
        }
        if max_iters == 0 {
            return Err(GmmError::InvalidCount {
                parameter: "max_iters",
                value: max_iters,
                maximum: None,
            });
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(GmmError::InvalidPositiveValue {
                parameter: "tolerance",
                value: tolerance,
            });
        }
        if !regularization.is_finite() || regularization <= 0.0 {
            return Err(GmmError::InvalidPositiveValue {
                parameter: "regularization",
                value: regularization,
            });
        }
        Ok(Self {
            n_components,
            max_iters,
            tolerance,
            regularization,
        })
    }

    pub fn n_components(&self) -> usize {
        self.n_components
    }

    pub fn max_iters(&self) -> usize {
        self.max_iters
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    pub fn regularization(&self) -> f64 {
        self.regularization
    }
}

impl Default for GmmRefitConfig {
    fn default() -> Self {
        Self {
            n_components: 3,
            max_iters: 100,
            tolerance: 1e-6,
            regularization: 1e-4,
        }
    }
}

impl<S, Obs> GmmStrategy<S, Obs> {
    /// Get the current refit configuration.
    pub fn get_refit_config(&self) -> &GmmRefitConfig {
        &self.refit_config
    }

    /// Set the refit configuration for this strategy instance.
    pub fn set_refit_config(&mut self, config: GmmRefitConfig) {
        self.refit_config = config;
    }
}

impl<S, Obs> GmmStrategy<S, Obs>
where
    S: StandardizedSpace,
    Obs: Serialize + DeserializeOwned + Send + Sync + Clone + Debug + 'static,
{
    /// Refit from completed domain-space trials and surface validation errors.
    pub fn try_refit(&mut self, space: &S, trials: &[(S::Domain, Obs)]) -> Result<(), GmmError> {
        if trials.is_empty() {
            return Ok(());
        }

        let samples: Vec<Vec<f64>> = trials
            .iter()
            .map(|(candidate, _)| space.to_unit_cube(candidate))
            .collect();

        // Derive the fit seed from the current cursor without consuming it.
        // Failed fitting and failed installation must leave *all* durable
        // strategy state unchanged, including the next sampling/refit seed.
        let counter = self.counter.load(Ordering::Relaxed);
        let refit_seed = self
            .seed
            .wrapping_add(counter.wrapping_mul(6364136223846793005));

        const MIN_SAMPLES_PER_COMPONENT: usize = 10;
        let supported_components = (samples.len() / MIN_SAMPLES_PER_COMPONENT).max(1);
        let fitted = GmmParams::fit(
            &samples,
            self.refit_config
                .n_components
                .min(supported_components)
                .min(samples.len()),
            self.refit_config.max_iters,
            self.refit_config.tolerance,
            self.refit_config.regularization,
            refit_seed,
        )?;

        self.set_params(fitted)?;
        self.counter
            .store(counter.wrapping_add(1), Ordering::Relaxed);
        Ok(())
    }
}

impl<S, Obs> RefittableStrategy for GmmStrategy<S, Obs>
where
    S: StandardizedSpace,
    Obs: Serialize + DeserializeOwned + Send + Sync + Clone + Debug + 'static,
{
    fn refit(&mut self, space: &Self::Space, trials: &[(S::Domain, Self::Observation)]) {
        if let Err(error) = self.try_refit(space, trials) {
            eprintln!("GmmStrategy::refit rejected invalid input: {error}");
        }
    }

    fn reconcile_after_refit(&mut self, live: &Self) {
        // `self` holds the freshly fitted parameters; `live` may have advanced
        // its RNG counter via concurrent `suggest` calls while this refit ran
        // off-lock. Keep the new model but adopt the furthest-advanced counter
        // so subsequent draws do not reuse an already-issued seed.
        let merged = self
            .counter
            .load(Ordering::Relaxed)
            .max(live.counter.load(Ordering::Relaxed));
        self.counter.store(merged, Ordering::Relaxed);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::UnitSquare;
    use crate::traits::SampleSpace;

    #[test]
    fn test_gaussian_component_sampling() {
        let mean = DVector::from_vec(vec![0.5, 0.5]);
        let comp = GaussianComponent::isotropic(mean.clone(), 0.01).unwrap();

        let mut rng = rand::rng();
        let sample = comp.sample(&mut rng).unwrap();

        assert_eq!(sample.len(), 2);
        // With small variance, samples should be near the mean
        assert!((sample[0] - 0.5).abs() < 0.5);
        assert!((sample[1] - 0.5).abs() < 0.5);
    }

    #[test]
    fn test_gmm_sampling_single_component() {
        let mean = DVector::from_vec(vec![0.5, 0.5]);
        let comp = GaussianComponent::isotropic(mean, 0.01).unwrap();
        let params = GmmParams::single(comp).unwrap();

        let mut rng = rand::rng();
        for _ in 0..100 {
            let sample = params.sample_clamped(&mut rng).unwrap();
            assert!(sample[0] >= 0.0 && sample[0] <= 1.0);
            assert!(sample[1] >= 0.0 && sample[1] <= 1.0);
        }
    }

    #[test]
    fn test_gmm_sampling_multiple_components() {
        let comp1 = GaussianComponent::isotropic(DVector::from_vec(vec![0.2, 0.2]), 0.01).unwrap();
        let comp2 = GaussianComponent::isotropic(DVector::from_vec(vec![0.8, 0.8]), 0.01).unwrap();
        let params = GmmParams::new(vec![0.5, 0.5], vec![comp1, comp2]).unwrap();

        let mut rng = rand::rng();
        let mut near_first = 0;
        let mut near_second = 0;

        for _ in 0..1000 {
            let sample = params.sample_clamped(&mut rng).unwrap();
            if sample[0] < 0.5 {
                near_first += 1;
            } else {
                near_second += 1;
            }
        }

        // With equal weights, should be roughly balanced
        assert!(near_first > 300 && near_first < 700);
        assert!(near_second > 300 && near_second < 700);
    }

    #[test]
    fn test_gmm_strategy_suggest() {
        let comp = GaussianComponent::isotropic(DVector::from_vec(vec![0.5, 0.5]), 0.01).unwrap();
        let params = GmmParams::single(comp).unwrap();
        let strategy = GmmStrategy::<UnitSquare>::new(42, params);
        let space = UnitSquare;

        for _ in 0..100 {
            let point = strategy.suggest(&space);
            assert!(space.contains(&point));
        }
    }

    #[test]
    fn test_gmm_fit_single_cluster() {
        // Generate samples around (0.3, 0.7)
        let mut samples = Vec::new();
        let mut rng = rand::rng();
        for _ in 0..100 {
            let x: f64 = 0.3 + (rng.random::<f64>() - 0.5) * 0.1;
            let y: f64 = 0.7 + (rng.random::<f64>() - 0.5) * 0.1;
            samples.push(vec![x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)]);
        }

        let fitted = GmmParams::fit(&samples, 1, 100, 1e-6, 1e-4, 42).unwrap();

        assert_eq!(fitted.n_components(), 1);
        let mean = fitted.components()[0].mean();
        assert!((mean[0] - 0.3).abs() < 0.1);
        assert!((mean[1] - 0.7).abs() < 0.1);
    }

    #[test]
    fn test_gmm_fit_two_clusters() {
        // Generate samples from two clusters
        let mut samples = Vec::new();
        let mut rng = rand::rng();

        // Cluster 1 around (0.2, 0.2)
        for _ in 0..50 {
            let x: f64 = 0.2 + (rng.random::<f64>() - 0.5) * 0.1;
            let y: f64 = 0.2 + (rng.random::<f64>() - 0.5) * 0.1;
            samples.push(vec![x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)]);
        }

        // Cluster 2 around (0.8, 0.8)
        for _ in 0..50 {
            let x: f64 = 0.8 + (rng.random::<f64>() - 0.5) * 0.1;
            let y: f64 = 0.8 + (rng.random::<f64>() - 0.5) * 0.1;
            samples.push(vec![x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)]);
        }

        let fitted = GmmParams::fit(&samples, 2, 100, 1e-6, 1e-4, 42).unwrap();

        assert_eq!(fitted.n_components(), 2);

        // Check that means are near the cluster centers (order may vary)
        let m1 = fitted.components()[0].mean();
        let m2 = fitted.components()[1].mean();

        let dist_to_first = |m: &DVector<f64>| (m[0] - 0.2).powi(2) + (m[1] - 0.2).powi(2);
        let dist_to_second = |m: &DVector<f64>| (m[0] - 0.8).powi(2) + (m[1] - 0.8).powi(2);

        // One mean should be near (0.2, 0.2), the other near (0.8, 0.8)
        let (near_first, near_second) = if dist_to_first(m1) < dist_to_second(m1) {
            (m1, m2)
        } else {
            (m2, m1)
        };

        assert!(dist_to_first(near_first).sqrt() < 0.15);
        assert!(dist_to_second(near_second).sqrt() < 0.15);
    }

    #[test]
    fn test_gmm_fit_drops_collapsed_components() {
        // Guards the final-assembly filter: a component starved during EM is
        // pruned to weight 1e-10 with an identity covariance. The identity
        // covariance passes Cholesky, so without the `weight > 1e-9` filter in
        // the final assembly it would survive `with_regularization` and leak a
        // dead, near-zero-weight component into the returned mixture, inflating
        // n_components(). The filter drops it, and renormalization restores the
        // sum-to-one invariant over the genuine survivors.
        //
        // Triggering scenario (fully deterministic, since both the data RNG and
        // the fit's k-means++ seed are fixed): two tight, well-separated clusters
        // of only 4 points each (8 points total) while asking for 7 components.
        // With far more requested components than supportable mass, k-means++
        // packs several centers into each tiny cluster; during EM the redundant
        // ones are out-competed and starved below the 1e-5 prune threshold. With
        // data_seed=1 and fit_seed=0 this reliably starves and prunes 2 of the 7
        // components, so fit returns 5.
        let mut rng = SmallRng::seed_from_u64(1);
        let centers = [(0.2, 0.2), (0.8, 0.8)];
        let mut samples = Vec::new();
        for &(cx, cy) in centers.iter() {
            for _ in 0..4 {
                let x = cx + (rng.random::<f64>() - 0.5) * 0.02;
                let y = cy + (rng.random::<f64>() - 0.5) * 0.02;
                samples.push(vec![x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)]);
            }
        }

        // Ask for 7 components over a layout that supports far fewer.
        let requested = 7;
        let fitted = GmmParams::fit(&samples, requested, 300, 1e-6, 1e-4, 0).unwrap();

        // (1) At least one component collapsed and was dropped, but the mixture
        // is never empty.
        assert!(
            fitted.n_components() >= 1 && fitted.n_components() < requested,
            "expected fewer than {requested} components after collapse, got {}",
            fitted.n_components()
        );

        // (2) No pruned/dead component (weight ~1e-10) leaked into the output,
        // which is what the final-assembly filter guarantees.
        assert!(
            fitted.weights().iter().all(|&w| w > 1e-9),
            "no sub-threshold (pruned) weight may remain after filtering, got {:?}",
            fitted.weights()
        );

        // (3) Surviving weights renormalize to sum to 1.
        let sum: f64 = fitted.weights().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "surviving weights must renormalize to sum to 1, got {sum}"
        );
    }

    #[test]
    fn test_diagonal_component() {
        let mean = DVector::from_vec(vec![0.5, 0.5]);
        let variances = DVector::from_vec(vec![0.01, 0.04]); // Different variance in each dim
        let comp = GaussianComponent::diagonal(mean, variances).unwrap();

        let mut rng = rand::rng();
        let mut x_var = 0.0;
        let mut y_var = 0.0;
        let n = 1000;

        for _ in 0..n {
            let sample = comp.sample(&mut rng).unwrap();
            x_var += (sample[0] - 0.5).powi(2);
            y_var += (sample[1] - 0.5).powi(2);
        }

        x_var /= n as f64;
        y_var /= n as f64;

        // y should have roughly 4x the variance of x
        assert!((x_var / 0.01 - 1.0).abs() < 0.3);
        assert!((y_var / 0.04 - 1.0).abs() < 0.3);
    }

    #[test]
    fn test_robust_singular_covariance() {
        let mean = DVector::from_vec(vec![0.5, 0.5]);
        let covariance = DMatrix::zeros(2, 2);

        assert!(matches!(
            GaussianComponent::new(mean.clone(), covariance.clone()),
            Err(GmmError::CovarianceNotPositiveDefinite)
        ));
        assert!(GaussianComponent::with_regularization(mean, covariance, 1e-6).is_ok());
    }

    #[test]
    fn test_gaussian_constructor_returns_structured_errors() {
        let mean = DVector::from_vec(vec![0.4, 0.6]);
        assert!(matches!(
            GaussianComponent::new(mean.clone(), DMatrix::identity(1, 1)),
            Err(GmmError::ShapeMismatch {
                expected: 2,
                rows: 1,
                columns: 1,
                ..
            })
        ));

        let mut non_finite = DMatrix::identity(2, 2);
        non_finite[(0, 0)] = f64::NAN;
        assert!(matches!(
            GaussianComponent::new(mean.clone(), non_finite),
            Err(GmmError::NonFiniteValue {
                context: "covariance",
                ..
            })
        ));

        let asymmetric = DMatrix::from_row_slice(2, 2, &[1.0, 0.2, 0.1, 1.0]);
        assert!(matches!(
            GaussianComponent::new(mean.clone(), asymmetric),
            Err(GmmError::NonSymmetricCovariance { .. })
        ));

        let negative = DMatrix::from_diagonal_element(2, 2, -1.0);
        assert!(matches!(
            GaussianComponent::with_regularization(mean, negative, 1e-4),
            Err(GmmError::CovarianceNotPositiveDefinite)
        ));
    }

    #[test]
    fn test_log_pdf_in_place_solve_matches_reference() {
        let component = GaussianComponent::diagonal(
            DVector::from_vec(vec![0.4, 0.6]),
            DVector::from_vec(vec![0.04, 0.09]),
        )
        .unwrap();
        let point = [0.5, 0.3];
        let mut scratch = [0.0; 2];
        let actual = component.log_pdf_with_scratch(&point, &mut scratch);

        let difference = DVector::from_column_slice(&point) - component.mean();
        let solved = component
            .cholesky_l
            .solve_lower_triangular(&difference)
            .unwrap();
        let expected = component.log_norm_const - 0.5 * solved.norm_squared();
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn test_gmm_params_constructor_returns_structured_errors() {
        let one = || GaussianComponent::isotropic(DVector::from_vec(vec![0.5]), 0.1).unwrap();
        let two = || GaussianComponent::isotropic(DVector::from_vec(vec![0.5, 0.5]), 0.1).unwrap();

        assert!(matches!(
            GmmParams::new(vec![], vec![]),
            Err(GmmError::EmptyMixture)
        ));
        assert!(matches!(
            GmmParams::new(vec![], vec![one()]),
            Err(GmmError::LengthMismatch { .. })
        ));
        for invalid in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                GmmParams::new(vec![invalid], vec![one()]),
                Err(GmmError::InvalidWeight { index: 0, .. })
            ));
        }
        assert!(matches!(
            GmmParams::new(vec![0.4, 0.4], vec![one(), one()]),
            Err(GmmError::WeightsDoNotSumToOne { .. })
        ));
        assert!(matches!(
            GmmParams::new(vec![0.5, 0.5], vec![one(), two()]),
            Err(GmmError::ComponentDimensionMismatch {
                index: 1,
                expected: 1,
                actual: 2,
            })
        ));
    }

    #[test]
    fn test_isotropic_rejects_bad_variance() {
        for &bad in &[-1.0, 0.0, f64::NAN, f64::INFINITY] {
            let error =
                GaussianComponent::isotropic(DVector::from_vec(vec![0.5, 0.5]), bad).unwrap_err();
            assert!(matches!(error, GmmError::InvalidPositiveValue { .. }));
        }
    }

    #[test]
    fn test_diagonal_rejects_bad_variances() {
        let mean = DVector::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let variances = DVector::from_vec(vec![-1.0, 0.0, f64::NAN, f64::INFINITY]);
        assert!(matches!(
            GaussianComponent::diagonal(mean, variances),
            Err(GmmError::InvalidPositiveValue { .. })
        ));
    }

    #[test]
    fn test_gmm_configuration() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;

        let mut gmm =
            GmmStrategy::<ContinuousSpace<LinearScale>>::uniform_prior(42, 2, 1.0).unwrap();
        let config = GmmRefitConfig::new(5, 100, 1e-6, 1e-4).unwrap();
        gmm.set_refit_config(config);
        assert_eq!(gmm.get_refit_config().n_components(), 5);

        let params = gmm.params().unwrap();
        let original_n = params.n_components();
        gmm.set_params(params).unwrap();
        assert_eq!(gmm.params().unwrap().n_components(), original_n);
    }

    #[test]
    fn test_fit_from_samples_does_not_mutate_on_error() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;

        let gmm = GmmStrategy::<ContinuousSpace<LinearScale>>::uniform_prior(42, 1, 0.25).unwrap();
        let before = serde_json::to_value(gmm.params().unwrap()).unwrap();
        let error = gmm
            .fit_from_samples(&[vec![f64::NAN]], 1, 100, 1e-6, 1e-4, 9)
            .unwrap_err();
        assert!(matches!(error, GmmError::NonFiniteSample { .. }));
        let after = serde_json::to_value(gmm.params().unwrap()).unwrap();
        assert_eq!(after, before);
    }

    #[test]
    fn test_failed_refit_preserves_full_serialized_state_and_next_sample() {
        let params = GmmParams::uniform_prior(2, 0.25).unwrap();
        let mut strategy = GmmStrategy::<UnitSquare>::new(42, params);
        strategy.set_refit_config(GmmRefitConfig::new(1, 100, 1e-6, 1e-4).unwrap());

        let before = serde_json::to_string(&strategy).unwrap();
        let control: GmmStrategy<UnitSquare> = serde_json::from_str(&before).unwrap();
        let error = strategy
            .try_refit(&UnitSquare, &[((f64::NAN, 0.5), 0.0)])
            .unwrap_err();
        assert!(matches!(error, GmmError::NonFiniteSample { .. }));

        assert_eq!(serde_json::to_string(&strategy).unwrap(), before);
        assert_eq!(strategy.suggest(&UnitSquare), control.suggest(&UnitSquare));
    }

    #[test]
    fn test_gmm_determinism_same_seed() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;

        let space = ContinuousSpace::new(0.0, 1.0);
        let params = GmmParams::uniform_prior(1, 0.5).unwrap();
        let strat1 = GmmStrategy::<ContinuousSpace<LinearScale>>::new(42, params.clone());
        let strat2 = GmmStrategy::<ContinuousSpace<LinearScale>>::new(42, params);
        for i in 0..10 {
            let a = strat1.suggest(&space);
            let b = strat2.suggest(&space);
            assert_eq!(a, b, "GMM mismatch at suggestion {i}");
        }
    }

    #[test]
    fn test_gmm_different_seeds_differ() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;

        let space = ContinuousSpace::new(0.0, 1.0);
        let params = GmmParams::uniform_prior(1, 0.5).unwrap();
        let strat1 = GmmStrategy::<ContinuousSpace<LinearScale>>::new(42, params.clone());
        let strat2 = GmmStrategy::<ContinuousSpace<LinearScale>>::new(999, params);
        assert_ne!(strat1.suggest(&space), strat2.suggest(&space));
    }

    #[test]
    fn test_gmm_suggest_in_bounds_product_space() {
        use crate::scales::LinearScale;
        use crate::spaces::{ContinuousSpace, ProductSpace};

        let space = ProductSpace {
            a: ContinuousSpace::new(-1.0, 1.0),
            b: ContinuousSpace::new(0.0, 10.0),
        };
        type Sp = ProductSpace<ContinuousSpace<LinearScale>, ContinuousSpace<LinearScale>>;
        let gmm = GmmStrategy::<Sp>::uniform_prior(42, 2, 1.0).unwrap();
        for _ in 0..50 {
            let candidate = gmm.suggest(&space);
            assert!(space.contains(&candidate));
        }
    }

    #[test]
    fn test_gmm_fit_updates_params() {
        let mut samples = Vec::new();
        for i in 0..50 {
            let v = if i < 25 {
                0.3 + (i as f64) * 0.002
            } else {
                0.7 + ((i - 25) as f64) * 0.002
            };
            samples.push(vec![v]);
        }
        let fitted = GmmParams::fit(&samples, 2, 100, 1e-6, 1e-4, 42).unwrap();
        assert_eq!(fitted.n_components(), 2);
    }

    #[test]
    fn test_gmm_refit_biases_sampling() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;
        use crate::traits::RefittableStrategy;

        let space = ContinuousSpace::new(0.0, 1.0);
        let params = GmmParams::uniform_prior(1, 1.0).unwrap();
        let mut gmm = GmmStrategy::<ContinuousSpace<LinearScale>>::new(42, params);
        gmm.set_refit_config(GmmRefitConfig::new(1, 100, 1e-6, 1e-4).unwrap());

        let trials: Vec<(f64, f64)> = (0..50).map(|i| (0.18 + (i as f64) * 0.001, 0.0)).collect();
        gmm.refit(&space, &trials);

        let mut below_04 = 0;
        for _ in 0..100 {
            if gmm.suggest(&space) < 0.4 {
                below_04 += 1;
            }
        }
        assert!(
            below_04 > 60,
            "After refit on data ~0.2, expected >60% < 0.4, got {below_04}%"
        );
    }

    #[test]
    fn test_gmm_refit_empty_noop() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;
        use crate::traits::RefittableStrategy;

        let space = ContinuousSpace::new(0.0, 1.0);
        let params = GmmParams::uniform_prior(1, 0.5).unwrap();
        let mut gmm = GmmStrategy::<ContinuousSpace<LinearScale>>::new(42, params);

        let empty: Vec<(f64, f64)> = vec![];
        gmm.refit(&space, &empty);

        assert_eq!(gmm.params().unwrap().n_components(), 1);
        assert!(space.contains(&gmm.suggest(&space)));
    }

    #[test]
    fn test_gmm_refit_config_defaults() {
        let config = GmmRefitConfig::default();
        assert_eq!(config.n_components(), 3);
        assert_eq!(config.max_iters(), 100);
        assert!((config.tolerance() - 1e-6).abs() < 1e-12);
        assert!((config.regularization() - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn test_gmm_refit_config_rejects_invalid_values_and_deserialization() {
        assert!(matches!(
            GmmRefitConfig::new(0, 100, 1e-6, 1e-4),
            Err(GmmError::InvalidCount {
                parameter: "n_components",
                ..
            })
        ));
        assert!(matches!(
            GmmRefitConfig::new(1, 0, 1e-6, 1e-4),
            Err(GmmError::InvalidCount {
                parameter: "max_iters",
                ..
            })
        ));
        assert!(GmmRefitConfig::new(1, 100, f64::NAN, 1e-4).is_err());
        assert!(GmmRefitConfig::new(1, 100, 1e-6, 0.0).is_err());

        let malformed = serde_json::json!({
            "n_components": 0,
            "max_iters": 100,
            "tolerance": 1e-6,
            "regularization": 1e-4,
        });
        let error = serde_json::from_value::<GmmRefitConfig>(malformed).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("n_components must be at least 1")
        );
    }

    #[test]
    fn test_gaussian_component_dim() {
        let mean = DVector::from_vec(vec![0.1, 0.5, 0.9]);
        let comp = GaussianComponent::isotropic(mean, 0.01).unwrap();
        assert_eq!(comp.dim(), 3);
    }

    #[test]
    fn test_gmm_params_single_component() {
        let comp = GaussianComponent::isotropic(DVector::from_vec(vec![0.5, 0.5]), 0.1).unwrap();
        let params = GmmParams::single(comp).unwrap();
        assert_eq!(params.n_components(), 1);
    }

    #[test]
    fn test_reconcile_after_refit_adopts_higher_counter() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;
        use crate::traits::RefittableStrategy;

        // `live` is the engine's current strategy, whose RNG counter advanced
        // via concurrent `suggest` calls while a refit ran off-lock.
        let live = GmmStrategy::<ContinuousSpace<LinearScale>>::uniform_prior(42, 1, 0.1).unwrap();
        for _ in 0..5 {
            live.counter.fetch_add(1, Ordering::Relaxed);
        }
        assert_eq!(live.counter.load(Ordering::Relaxed), 5);

        // `fitted` is the off-lock snapshot taken before those suggests, so its
        // counter is lower. After reconciliation it must adopt the live (higher)
        // counter so subsequent draws do not reuse an already-issued seed.
        let mut fitted = live.clone();
        fitted.counter.store(2, Ordering::Relaxed);
        fitted.reconcile_after_refit(&live);

        assert_eq!(fitted.counter.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_fit_rejects_non_finite_and_out_of_cube_samples() {
        let non_finite = vec![vec![0.2, 0.3], vec![f64::NAN, 0.5]];
        assert!(matches!(
            GmmParams::fit(&non_finite, 1, 100, 1e-4, 1e-4, 42),
            Err(GmmError::NonFiniteSample {
                sample: 1,
                dimension: 0,
                ..
            })
        ));

        let out_of_cube = vec![vec![0.2, 0.3], vec![0.5, 1.01]];
        assert!(matches!(
            GmmParams::fit(&out_of_cube, 1, 100, 1e-4, 1e-4, 42),
            Err(GmmError::SampleOutOfCube {
                sample: 1,
                dimension: 1,
                ..
            })
        ));
    }

    #[test]
    fn test_fit_rejects_empty_and_ragged_samples() {
        assert!(matches!(
            GmmParams::fit(&[], 1, 100, 1e-4, 1e-4, 42),
            Err(GmmError::EmptySamples)
        ));
        assert!(matches!(
            GmmParams::fit(&[vec![]], 1, 100, 1e-4, 1e-4, 42),
            Err(GmmError::EmptyDimension { .. })
        ));
        let ragged = vec![vec![0.2, 0.3], vec![0.4]];
        assert!(matches!(
            GmmParams::fit(&ragged, 1, 100, 1e-4, 1e-4, 42),
            Err(GmmError::RaggedSample {
                sample: 1,
                expected: 2,
                actual: 1,
            })
        ));
    }

    #[test]
    fn test_fit_rejects_invalid_algorithm_parameters() {
        let samples = vec![vec![0.2], vec![0.8]];
        for n_components in [0, 3] {
            assert!(matches!(
                GmmParams::fit(&samples, n_components, 100, 1e-4, 1e-4, 42),
                Err(GmmError::InvalidCount {
                    parameter: "n_components",
                    ..
                })
            ));
        }
        assert!(matches!(
            GmmParams::fit(&samples, 1, 0, 1e-4, 1e-4, 42),
            Err(GmmError::InvalidCount {
                parameter: "max_iters",
                ..
            })
        ));
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                GmmParams::fit(&samples, 1, 100, tolerance, 1e-4, 42),
                Err(GmmError::InvalidPositiveValue {
                    parameter: "tolerance",
                    ..
                })
            ));
        }
        for regularization in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                GmmParams::fit(&samples, 1, 100, 1e-4, regularization, 42),
                Err(GmmError::InvalidPositiveValue {
                    parameter: "regularization",
                    ..
                })
            ));
        }
    }

    #[test]
    fn test_gaussian_rejects_invalid_mean() {
        assert!(matches!(
            GaussianComponent::isotropic(DVector::from_vec(vec![f64::NAN, 0.5]), 0.01),
            Err(GmmError::NonFiniteValue {
                context: "Gaussian mean",
                index: 0,
                ..
            })
        ));
        assert!(matches!(
            GaussianComponent::isotropic(DVector::from_vec(vec![0.5, 1.1]), 0.01),
            Err(GmmError::OutOfCubeValue {
                context: "Gaussian mean",
                index: 1,
                ..
            })
        ));
    }

    #[test]
    fn test_deserialize_malformed_gmm_returns_error() {
        let c2 = GaussianComponent::isotropic(DVector::from_vec(vec![0.5, 0.5]), 0.01).unwrap();
        let c1 = GaussianComponent::isotropic(DVector::from_vec(vec![0.5]), 0.01).unwrap();
        let serde_form = serde_json::json!({
            "weights": [0.2, 0.2],
            "components": [
                GaussianComponentSerde::from(c1),
                GaussianComponentSerde::from(c2),
            ],
        });
        let error = serde_json::from_value::<GmmParams>(serde_form).unwrap_err();
        assert!(error.to_string().contains("weights must sum to 1"));
    }

    #[test]
    fn test_deserialize_rejects_malformed_component() {
        let good = GaussianComponent::isotropic(DVector::from_vec(vec![0.9, 0.1]), 0.01).unwrap();
        let mut bad = GaussianComponentSerde::from(good.clone());
        bad.mean = DVector::from_vec(vec![0.3]);
        bad.covariance = DMatrix::from_element(2, 2, 0.01);

        let serde_form = serde_json::json!({
            "weights": [0.4, 0.6],
            "components": [
                bad,
                GaussianComponentSerde::from(good.clone()),
            ],
        });
        let error = serde_json::from_value::<GmmParams>(serde_form).unwrap_err();
        assert!(error.to_string().contains("covariance must be 1x1"));
    }

    #[test]
    fn test_deserialize_empty_gmm_returns_error() {
        let serde_form = serde_json::json!({
            "weights": serde_json::Value::Array(vec![]),
            "components": serde_json::Value::Array(vec![]),
        });
        let error = serde_json::from_value::<GmmParams>(serde_form).unwrap_err();
        assert!(error.to_string().contains("at least one component"));
    }

    #[test]
    fn test_deserialize_rejects_incorrect_declared_dimension() {
        let params = GmmParams::uniform_prior(1, 0.1).unwrap();
        let mut serialized = serde_json::to_value(params).unwrap();
        serialized
            .as_object_mut()
            .unwrap()
            .insert("dim".to_string(), serde_json::json!(2));

        let error = serde_json::from_value::<GmmParams>(serialized).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("declares dimension 2, but components have dimension 1")
        );
    }

    #[test]
    fn test_suggest_does_not_panic_on_dim_mismatch() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;

        // The space is 1-D but the GMM is 2-D: a dim mismatch on the hot path.
        let space = ContinuousSpace::new(0.0, 1.0);
        let params = GmmParams::uniform_prior(2, 0.1).unwrap();
        let strat = GmmStrategy::<ContinuousSpace<LinearScale>>::new(42, params);

        // Must not panic and must yield an in-bounds point.
        for _ in 0..20 {
            let p = strat.suggest(&space);
            assert!(space.contains(&p));
        }
    }

    #[test]
    fn test_gmm_checkpoint_roundtrip_post_fit_determinism() {
        // Exercise a FITTED GMM through a JSON serialize/deserialize roundtrip
        // and assert the deserialized strategy resumes sampling identically.
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;
        use crate::traits::RefittableStrategy;

        let space = ContinuousSpace::new(0.0, 1.0);

        // Start from a uniform prior, then REFIT from a deterministic two-cluster
        // sample set so the strategy holds fitted mixture params rather
        // than the pre-fit single-component uniform prior.
        let params = GmmParams::uniform_prior(1, 1.0).unwrap();
        let mut strat = GmmStrategy::<ContinuousSpace<LinearScale>>::new(42, params);
        strat.set_refit_config(GmmRefitConfig::new(2, 100, 1e-9, 1e-4).unwrap());

        let mut trials: Vec<(f64, f64)> = Vec::new();
        for i in 0..30 {
            trials.push((0.18 + (i as f64) * 0.001, 0.0));
        }
        for i in 0..30 {
            trials.push((0.78 + (i as f64) * 0.001, 0.0));
        }
        strat.refit(&space, &trials);

        // Confirm the model is actually fitted (multi-component), not the prior.
        assert!(
            strat.params().unwrap().n_components() >= 2,
            "refit should have produced fitted multi-component params"
        );

        // Advance the sampling counter past the refit so the roundtrip must also
        // preserve the counter, not just the mixture params.
        for _ in 0..3 {
            let _ = strat.suggest(&space);
        }

        // Serialize the fitted strategy to JSON and deserialize it back.
        let json = serde_json::to_string(&strat).expect("serialize fitted GMM strategy");
        let restored: GmmStrategy<ContinuousSpace<LinearScale>> =
            serde_json::from_str(&json).expect("deserialize fitted GMM strategy");

        // The deserialized strategy must produce the SAME next suggest() outputs
        // as the original: this only holds if both the fitted mixture params and
        // the atomic sampling counter survived the roundtrip.
        for i in 0..10 {
            let a = strat.suggest(&space);
            let b = restored.suggest(&space);
            assert_eq!(
                a, b,
                "post-fit checkpoint roundtrip diverged at suggestion {i}"
            );
        }
    }

    #[test]
    fn test_weighted_index_sampling() {
        // Test that weighted sampling works correctly
        let comp1 = GaussianComponent::isotropic(DVector::from_vec(vec![0.1]), 0.001).unwrap();
        let comp2 = GaussianComponent::isotropic(DVector::from_vec(vec![0.9]), 0.001).unwrap();
        let params = GmmParams::new(vec![0.9, 0.1], vec![comp1, comp2]).unwrap();

        let mut rng = rand::rng();
        let mut near_first = 0;

        for _ in 0..1000 {
            let sample = params.sample_clamped(&mut rng).unwrap();
            if sample[0] < 0.5 {
                near_first += 1;
            }
        }

        // Should be approximately 90% near first component
        assert!(near_first > 800 && near_first < 980);
    }

    #[test]
    fn test_hot_paths_reuse_cached_storage() {
        let component = GaussianComponent::diagonal(
            DVector::from_vec(vec![0.4, 0.6]),
            DVector::from_vec(vec![0.04, 0.09]),
        )
        .unwrap();
        let params = GmmParams::single(component.clone()).unwrap();
        let distribution_address = std::ptr::addr_of!(params.component_distribution);
        let mut rng = SmallRng::seed_from_u64(9);
        for _ in 0..100 {
            params.sample_unclamped(&mut rng).unwrap();
            assert_eq!(
                std::ptr::addr_of!(params.component_distribution),
                distribution_address
            );
        }

        let mut sample = Vec::with_capacity(component.dim());
        params.sample_clamped_into(&mut rng, &mut sample).unwrap();
        let sample_address = sample.as_ptr();
        for _ in 0..100 {
            params.sample_clamped_into(&mut rng, &mut sample).unwrap();
            assert_eq!(sample.as_ptr(), sample_address);
        }

        let mut scratch = vec![0.0; component.dim()];
        let scratch_address = scratch.as_ptr();
        for _ in 0..100 {
            assert!(
                component
                    .log_pdf_with_scratch(&[0.45, 0.55], &mut scratch)
                    .is_finite()
            );
            assert_eq!(scratch.as_ptr(), scratch_address);
        }
    }

    #[test]
    fn test_strategy_suggest_reuses_owned_sample_scratch() {
        let params = GmmParams::uniform_prior(2, 0.1).unwrap();
        let strategy = GmmStrategy::<UnitSquare>::new(42, params);
        let scratch_address = strategy
            .sample_scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ptr();

        for _ in 0..100 {
            assert!(UnitSquare.contains(&strategy.suggest(&UnitSquare)));
            let scratch = strategy
                .sample_scratch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert_eq!(scratch.len(), 2);
            assert_eq!(scratch.as_ptr(), scratch_address);
        }
    }

    #[test]
    #[ignore = "performance probe; run explicitly with --ignored --nocapture"]
    fn gmm_hot_path_throughput_probe() {
        use std::time::{Duration, Instant};

        let params = GmmParams::uniform_prior(4, 0.1).unwrap();
        let mut rng = SmallRng::seed_from_u64(11);
        let sampling_start = Instant::now();
        for _ in 0..100_000 {
            params.sample_clamped(&mut rng).unwrap();
        }
        let sampling_elapsed = sampling_start.elapsed();

        let samples: Vec<Vec<f64>> = (0..2_000)
            .map(|index| {
                let base = (index % 100) as f64 / 100.0;
                vec![base, 1.0 - base, base / 2.0, 0.25 + base / 2.0]
            })
            .collect();
        let fitting_start = Instant::now();
        GmmParams::fit(&samples, 3, 25, 1e-6, 1e-4, 13).unwrap();
        let fitting_elapsed = fitting_start.elapsed();

        eprintln!(
            "100k cached-distribution suggestions: {sampling_elapsed:?}; 2k-sample fit: {fitting_elapsed:?}"
        );
        let budget = Duration::from_secs(5);
        assert!(
            sampling_elapsed < budget,
            "cached sampling exceeded the {budget:?} debug-build budget"
        );
        assert!(
            fitting_elapsed < budget,
            "GMM fit exceeded the {budget:?} debug-build budget"
        );
    }
}
