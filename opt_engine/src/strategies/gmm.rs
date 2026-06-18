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

use crate::traits::{StandardizedSpace, Strategy};
use nalgebra::{DMatrix, DVector, DVectorView};
use rand::SeedableRng;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

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
#[serde(from = "GaussianComponentSerde", into = "GaussianComponentSerde")]
pub struct GaussianComponent {
    /// Mean vector (dimensionality = n).
    pub mean: DVector<f64>,
    /// Covariance matrix (n × n), symmetric positive definite.
    pub covariance: DMatrix<f64>,
    /// Cached Lower Cholesky factor (L where Σ = L L^T).
    cholesky_l: DMatrix<f64>,
    /// Cached constant term: -0.5 * (d * ln(2π) + ln(det(Σ)))
    log_norm_const: f64,
}

impl GaussianComponent {
    /// Create a new Gaussian component with robustness to near-singular covariances.
    ///
    /// If the covariance matrix is not positive definite, regularization is applied.
    /// Returns `None` only if the matrix is completely degenerate.
    pub fn new(mean: DVector<f64>, covariance: DMatrix<f64>) -> Option<Self> {
        Self::with_regularization(mean, covariance, 1e-6)
    }

    /// Create a new Gaussian component with explicit regularization.
    pub fn with_regularization(
        mean: DVector<f64>,
        mut covariance: DMatrix<f64>,
        reg: f64,
    ) -> Option<Self> {
        let dim = mean.len();
        assert_eq!(covariance.nrows(), dim);
        assert_eq!(covariance.ncols(), dim);

        // Try Cholesky; if it fails, add regularization
        let chol = match covariance.clone().cholesky() {
            Some(c) => c,
            None => {
                // Apply stronger regularization
                for i in 0..dim {
                    covariance[(i, i)] += reg * 10.0;
                }
                covariance.clone().cholesky()?
            }
        };

        let l = chol.l();

        // Log determinant = 2 * sum(log(diag(L)))
        let log_det: f64 = 2.0 * l.diagonal().iter().map(|x| x.ln()).sum::<f64>();
        let log_norm_const = -0.5 * (dim as f64 * (2.0 * std::f64::consts::PI).ln() + log_det);

        Some(Self {
            mean,
            covariance,
            cholesky_l: l,
            log_norm_const,
        })
    }

    /// Create an isotropic (spherical) Gaussian component.
    ///
    /// Non-positive or non-finite variances are clamped to a small positive
    /// value so the covariance is always valid and construction never panics.
    pub fn isotropic(mean: DVector<f64>, variance: f64) -> Self {
        let dim = mean.len();
        let var = if variance.is_finite() && variance > 0.0 {
            variance
        } else {
            1e-6
        };
        let covariance = DMatrix::identity(dim, dim) * var;
        Self::new(mean, covariance).expect("Isotropic covariance should always be valid")
    }

    /// Create a diagonal Gaussian component.
    ///
    /// Non-positive or non-finite variances are clamped to a small positive
    /// value so the covariance is always valid and construction never panics.
    pub fn diagonal(mean: DVector<f64>, variances: DVector<f64>) -> Self {
        let dim = mean.len();
        assert_eq!(variances.len(), dim);
        let safe = variances.map(|v| if v.is_finite() && v > 0.0 { v } else { 1e-6 });
        let covariance = DMatrix::from_diagonal(&safe);
        Self::new(mean, covariance).expect("Diagonal covariance should always be valid")
    }

    /// Sample from this Gaussian component.
    ///
    /// Uses the reparameterization: x = μ + L * z, where z ~ N(0, I).
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64> {
        let dim = self.mean.len();
        let z = DVector::from_fn(dim, |_, _| StandardNormal.sample(rng));
        &self.mean + &self.cholesky_l * z
    }

    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Compute log probability density at a point.
    #[inline]
    fn log_pdf(&self, x: &DVectorView<f64>) -> f64 {
        let diff = x - &self.mean;
        match self.cholesky_l.solve_lower_triangular(&diff) {
            Some(solved) => {
                let mahal_sq = solved.norm_squared();
                self.log_norm_const - 0.5 * mahal_sq
            }
            None => f64::NEG_INFINITY,
        }
    }
}

// Serde helper for GaussianComponent - only serializes mean and covariance
#[derive(Serialize, Deserialize)]
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

impl GaussianComponentSerde {
    /// Reconstruct a component from on-disk data, or `None` if the data is
    /// malformed (zero-length, mismatched covariance dims, or non-finite
    /// values). `new` regularizes merely near-singular covariances, so a
    /// `None` here means the stored data was structurally unusable.
    fn try_into_component(self) -> Option<GaussianComponent> {
        let dim = self.mean.len();
        let dims_ok = self.covariance.nrows() == dim && self.covariance.ncols() == dim;
        let finite = self.mean.iter().all(|x| x.is_finite())
            && self.covariance.iter().all(|x| x.is_finite());

        if dim > 0 && dims_ok && finite {
            GaussianComponent::new(self.mean, self.covariance)
        } else {
            None
        }
    }
}

impl From<GaussianComponentSerde> for GaussianComponent {
    fn from(s: GaussianComponentSerde) -> Self {
        let dim = s.mean.len();
        let safe_dim = dim.max(1);
        // Recover a malformed standalone component into a safe isotropic one of
        // the recovered dimensionality rather than panicking on corrupt input.
        s.try_into_component().unwrap_or_else(|| {
            GaussianComponent::isotropic(DVector::from_element(safe_dim, 0.5), 0.1)
        })
    }
}

/// Parameters for a Gaussian Mixture Model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "GmmParamsSerde")]
pub struct GmmParams {
    /// Mixture weights (must sum to 1, all positive).
    pub weights: Vec<f64>,
    /// Gaussian components.
    pub components: Vec<GaussianComponent>,
}

// Serde helper for GmmParams: deserialization validates and recovers instead of
// constructing an invalid mixture that would later panic in `new` or `suggest`.
// Components are kept in their raw serde form so individual malformed components
// can be detected and dropped rather than silently recovered into a default that
// would hide a corrupt mixture.
#[derive(Deserialize)]
struct GmmParamsSerde {
    weights: Vec<f64>,
    components: Vec<GaussianComponentSerde>,
    /// Declared dimensionality, used to pick the mixture dim and to size the
    /// fallback prior when no component survives recovery.
    #[serde(default)]
    dim: Option<usize>,
}

impl From<GmmParamsSerde> for GmmParams {
    fn from(s: GmmParamsSerde) -> Self {
        let GmmParamsSerde {
            weights,
            components,
            dim: declared_dim,
        } = s;

        // Pair each raw component with its weight, defaulting a missing or
        // non-finite/negative weight to zero so the entry drops out below.
        let n = components.len();
        let safe_weight = |i: usize| -> f64 {
            match weights.get(i) {
                Some(w) if w.is_finite() && *w >= 0.0 => *w,
                _ => 0.0,
            }
        };

        // Determine the mixture's intended dimensionality from the declared dim
        // (if present) or the first component that reconstructs successfully.
        let recovered: Vec<(f64, GaussianComponent)> = (0..n)
            .zip(components)
            .filter_map(|(i, c)| c.try_into_component().map(|comp| (safe_weight(i), comp)))
            .collect();

        let target_dim = declared_dim
            .filter(|&d| d > 0)
            .or_else(|| recovered.first().map(|(_, c)| c.dim()));

        // Keep only components matching the target dim with a positive weight,
        // preserving the mixture's actual dimensionality. A single bad component
        // is dropped, not allowed to demote the whole mixture.
        let (kept_weights, kept_components): (Vec<f64>, Vec<GaussianComponent>) = recovered
            .into_iter()
            .filter(|(w, c)| *w > 0.0 && target_dim.is_some_and(|d| c.dim() == d))
            .unzip();

        let sum: f64 = kept_weights.iter().sum();
        if !kept_components.is_empty() && sum > 0.0 {
            // Renormalize the surviving weights so the sum-to-one invariant holds.
            let normalized: Vec<f64> = kept_weights.iter().map(|w| w / sum).collect();
            return Self {
                weights: normalized,
                components: kept_components,
            };
        }

        // No valid component remained: fall back to a uniform prior of the
        // declared dimensionality (or 1-D if nothing usable was declared).
        let dim = target_dim.or(declared_dim).unwrap_or(1).max(1);
        Self::uniform_prior(dim, 0.1)
    }
}

impl GmmParams {
    /// Create a new GMM from weights and components.
    ///
    /// # Panics
    /// Panics if weights don't sum to ~1, any weight is negative,
    /// or components have mismatched dimensionality.
    pub fn new(weights: Vec<f64>, components: Vec<GaussianComponent>) -> Self {
        assert!(!weights.is_empty());
        assert_eq!(weights.len(), components.len());

        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Weights must sum to 1, got {sum}",);
        assert!(
            weights.iter().all(|&w| w >= 0.0),
            "All weights must be non-negative"
        );

        if components.len() > 1 {
            let dim = components[0].dim();
            assert!(
                components.iter().all(|c| c.dim() == dim),
                "All components must have the same dimensionality"
            );
        }

        Self {
            weights,
            components,
        }
    }

    /// Create a single-component GMM (just a multivariate normal).
    pub fn single(component: GaussianComponent) -> Self {
        Self {
            weights: vec![1.0],
            components: vec![component],
        }
    }

    /// Create a uniform GMM centered in the unit hypercube.
    pub fn uniform_prior(dim: usize, variance: f64) -> Self {
        let mean = DVector::from_element(dim, 0.5);
        Self::single(GaussianComponent::isotropic(mean, variance))
    }

    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    pub fn dim(&self) -> usize {
        self.components.first().map(|c| c.dim()).unwrap_or(0)
    }

    /// Sample from the GMM (unclamped).
    pub fn sample_unclamped<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64> {
        if self.components.is_empty() {
            return DVector::zeros(0);
        }

        // Use WeightedIndex for efficient component selection
        let dist = WeightedIndex::new(&self.weights).expect("Invalid weights");
        let idx = dist.sample(rng);
        self.components[idx].sample(rng)
    }

    /// Sample from the GMM, clamped to [0, 1]^n.
    ///
    /// Non-finite drawn values (NaN/inf) are replaced with the cube center
    /// (0.5) before clamping, since `f64::clamp` propagates NaN and would
    /// otherwise let it escape into a suggestion.
    pub fn sample_clamped<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let sample = self.sample_unclamped(rng);
        sample
            .iter()
            .map(|&x| {
                if x.is_finite() {
                    x.clamp(0.0, 1.0)
                } else {
                    0.5
                }
            })
            .collect()
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
    pub fn fit(
        samples: &[Vec<f64>],
        n_components: usize,
        max_iters: usize,
        tolerance: f64,
        reg: f64,
        seed: u64,
    ) -> Self {
        if samples.is_empty() {
            // Dimensionality is unknown with no samples; callers that need a
            // specific dim should pass at least one sample. Default to a 1-D
            // prior.
            return Self::uniform_prior(1, 0.1);
        }

        let dim = samples[0].len();
        if dim == 0 {
            return Self::uniform_prior(1, 0.1);
        }

        // Drop any sample row containing a non-finite value (NaN/inf); a single
        // such value would otherwise corrupt the fitted means and covariances.
        // Rows whose length disagrees with `dim` are also dropped.
        let finite_samples: Vec<&Vec<f64>> = samples
            .iter()
            .filter(|s| s.len() == dim && s.iter().all(|x| x.is_finite()))
            .collect();

        if finite_samples.is_empty() {
            // No usable rows remained; fall back to a uniform prior of the
            // correct dimensionality so later suggestions still match the space.
            return Self::uniform_prior(dim.max(1), 0.1);
        }

        let n_samples = finite_samples.len();

        // Flatten data for cache-friendly access (each sample is contiguous)
        let flat_data: Vec<f64> = finite_samples
            .iter()
            .flat_map(|s| s.iter().copied())
            .collect();

        // Initialize with K-means++
        let (mut weights, mut means, mut covs) =
            kmeans_pp_init(&flat_data, dim, n_samples, n_components, seed);

        let mut prev_ll = f64::NEG_INFINITY;
        let mut log_probs = vec![0.0f64; n_components]; // Reusable buffer

        for _iter in 0..max_iters {
            // Build components for E-step (computes Cholesky once per iteration)
            let components: Vec<Option<GaussianComponent>> = means
                .iter()
                .zip(covs.iter())
                .map(|(m, c)| GaussianComponent::with_regularization(m.clone(), c.clone(), reg))
                .collect();

            // Fused E-Step & M-Step: single pass over data
            let mut stats = SufficientStats::new(n_components, dim);
            let mut total_ll = 0.0;

            for sample_slice in flat_data.chunks_exact(dim) {
                let x = DVectorView::from_slice(sample_slice, dim);
                let mut max_log = f64::NEG_INFINITY;

                // E-Step: compute log responsibilities
                for (k, comp_opt) in components.iter().enumerate() {
                    if let Some(comp) = comp_opt {
                        let log_w = weights[k].max(1e-300).ln();
                        let log_p = comp.log_pdf(&x);
                        log_probs[k] = log_w + log_p;
                        if log_probs[k] > max_log {
                            max_log = log_probs[k];
                        }
                    } else {
                        log_probs[k] = f64::NEG_INFINITY;
                    }
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

                if sum_exp > 1e-300 {
                    total_ll += max_log + sum_exp.ln();
                }

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

            // Check convergence
            if (total_ll - prev_ll).abs() < tolerance {
                break;
            }
            prev_ll = total_ll;

            // Update parameters from sufficient statistics
            let total_weight: f64 = stats.weight_sum.iter().sum();

            for k in 0..n_components {
                let nk = stats.weight_sum[k];

                // Prune dead components
                if nk < 1e-5 {
                    weights[k] = 1e-10;
                    covs[k] = DMatrix::identity(dim, dim);
                    continue;
                }

                weights[k] = nk / total_weight.max(1e-10);

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
        let (surviving_weights, final_components): (Vec<f64>, Vec<GaussianComponent>) = weights
            .into_iter()
            .zip(means.into_iter().zip(covs))
            .filter(|(w, _)| *w > 1e-9)
            .filter_map(|(w, (m, c))| {
                GaussianComponent::with_regularization(m, c, reg).map(|comp| (w, comp))
            })
            .unzip();

        if final_components.is_empty() {
            return Self::uniform_prior(dim, 0.1);
        }

        let w_sum: f64 = surviving_weights.iter().sum();
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
fn kmeans_pp_init(
    flat_data: &[f64],
    dim: usize,
    n: usize,
    k: usize,
    seed: u64,
) -> (Vec<f64>, Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
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
        let last_mean = means.last().unwrap();

        // Update minimum distances
        for (i, min_d) in min_dists.iter_mut().enumerate() {
            let s_slice = &flat_data[i * dim..(i + 1) * dim];
            let x = DVectorView::from_slice(s_slice, dim);
            let d = (x - last_mean).norm_squared();
            if d < *min_d {
                *min_d = d;
            }
        }

        // Sample next center proportional to squared distance
        let sum_dist: f64 = min_dists.iter().sum();
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

    (weights, means, covs)
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
/// ```ignore
/// use nalgebra::DVector;
/// use opt_engine::strategies::{GmmStrategy, GmmParams, GaussianComponent};
///
/// // Create a GMM with two components
/// let comp1 = GaussianComponent::isotropic(DVector::from_vec(vec![0.3, 0.3]), 0.01);
/// let comp2 = GaussianComponent::isotropic(DVector::from_vec(vec![0.7, 0.7]), 0.01);
/// let params = GmmParams::new(vec![0.5, 0.5], vec![comp1, comp2]);
///
/// let strategy = GmmStrategy::<MySpace>::new(params);
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
        Self {
            seed,
            counter: AtomicU64::new(0),
            params: Arc::new(RwLock::new(params)),
            refit_config: GmmRefitConfig::default(),
            _marker: PhantomData,
        }
    }

    /// Create a GMM strategy with an automatically chosen seed and the given parameters.
    pub fn auto_seed(params: GmmParams) -> Self {
        Self::new(rand::random(), params)
    }

    /// Create a GMM strategy with a uniform prior centered in the hypercube.
    pub fn uniform_prior(seed: u64, dim: usize, variance: f64) -> Self {
        Self::new(seed, GmmParams::uniform_prior(dim, variance))
    }

    /// Update the GMM parameters.
    pub fn set_params(&self, params: GmmParams) {
        *self.params.write().unwrap() = params;
    }

    /// Get a clone of the current GMM parameters.
    pub fn params(&self) -> GmmParams {
        self.params.read().unwrap().clone()
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
    ) {
        let fitted = GmmParams::fit(samples, n_components, max_iters, tolerance, reg, seed);
        self.set_params(fitted);
    }
}

impl<S, Obs> Clone for GmmStrategy<S, Obs> {
    fn clone(&self) -> Self {
        Self {
            seed: self.seed,
            counter: AtomicU64::new(self.counter.load(Ordering::Relaxed)),
            params: Arc::new(RwLock::new(self.params.read().unwrap().clone())),
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
        let params = self.params.read().unwrap();

        let call_index = self.counter.fetch_add(1, Ordering::Relaxed);
        let call_seed = self
            .seed
            .wrapping_add(call_index.wrapping_mul(6364136223846793005));
        let mut rng = SmallRng::seed_from_u64(call_seed);

        // If the fitted GMM dimensionality disagrees with the space (e.g. after
        // a refit on a differently-shaped space, or a recovered deserialized
        // model), the GMM sample cannot be mapped. Fall back to the cube center
        // so this hot path, which runs under the engine write lock inside axum/
        // PyO3 handlers, never panics.
        let sample = if params.dim() == dim {
            params.sample_clamped(&mut rng)
        } else {
            eprintln!(
                "GmmStrategy::suggest: GMM dim {} != space dim {}, falling back to cube center",
                params.dim(),
                dim
            );
            vec![0.5; dim]
        };

        match space.from_unit_cube(&sample) {
            Some(domain) => domain,
            None => {
                // Mapping rejected the point; retry with a deterministic
                // in-cube center rather than panicking.
                eprintln!(
                    "GmmStrategy::suggest: from_unit_cube failed, falling back to cube center"
                );
                space
                    .from_unit_cube(&vec![0.5; dim])
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
pub struct GmmRefitConfig {
    /// Number of GMM components to fit.
    pub n_components: usize,
    /// Maximum EM iterations.
    pub max_iters: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Regularization for covariance matrices.
    pub regularization: f64,
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

impl<S, Obs> RefittableStrategy for GmmStrategy<S, Obs>
where
    S: StandardizedSpace,
    Obs: Serialize + DeserializeOwned + Send + Sync + Clone + Debug + 'static,
{
    fn refit(&mut self, space: &Self::Space, trials: &[(S::Domain, Self::Observation)]) {
        if trials.is_empty() {
            return;
        }

        // Convert candidates to unit cube representation
        let samples: Vec<Vec<f64>> = trials
            .iter()
            .map(|(candidate, _)| space.to_unit_cube(candidate))
            .collect();

        // Use stored config for fitting
        let config = &self.refit_config;

        // Derive a seed for this refit from the strategy's seed + counter
        let refit_seed = self.seed.wrapping_add(
            self.counter
                .fetch_add(1, Ordering::Relaxed)
                .wrapping_mul(6364136223846793005),
        );

        // Fit the GMM to the samples
        let fitted = GmmParams::fit(
            &samples,
            config.n_components.min(samples.len()),
            config.max_iters,
            config.tolerance,
            config.regularization,
            refit_seed,
        );

        self.set_params(fitted);
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
        let comp = GaussianComponent::isotropic(mean.clone(), 0.01);

        let mut rng = rand::rng();
        let sample = comp.sample(&mut rng);

        assert_eq!(sample.len(), 2);
        // With small variance, samples should be near the mean
        assert!((sample[0] - 0.5).abs() < 0.5);
        assert!((sample[1] - 0.5).abs() < 0.5);
    }

    #[test]
    fn test_gmm_sampling_single_component() {
        let mean = DVector::from_vec(vec![0.5, 0.5]);
        let comp = GaussianComponent::isotropic(mean, 0.01);
        let params = GmmParams::single(comp);

        let mut rng = rand::rng();
        for _ in 0..100 {
            let sample = params.sample_clamped(&mut rng);
            assert!(sample[0] >= 0.0 && sample[0] <= 1.0);
            assert!(sample[1] >= 0.0 && sample[1] <= 1.0);
        }
    }

    #[test]
    fn test_gmm_sampling_multiple_components() {
        let comp1 = GaussianComponent::isotropic(DVector::from_vec(vec![0.2, 0.2]), 0.01);
        let comp2 = GaussianComponent::isotropic(DVector::from_vec(vec![0.8, 0.8]), 0.01);
        let params = GmmParams::new(vec![0.5, 0.5], vec![comp1, comp2]);

        let mut rng = rand::rng();
        let mut near_first = 0;
        let mut near_second = 0;

        for _ in 0..1000 {
            let sample = params.sample_clamped(&mut rng);
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
        let comp = GaussianComponent::isotropic(DVector::from_vec(vec![0.5, 0.5]), 0.01);
        let params = GmmParams::single(comp);
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

        let fitted = GmmParams::fit(&samples, 1, 100, 1e-6, 1e-4, 42);

        assert_eq!(fitted.n_components(), 1);
        let mean = &fitted.components[0].mean;
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

        let fitted = GmmParams::fit(&samples, 2, 100, 1e-6, 1e-4, 42);

        assert_eq!(fitted.n_components(), 2);

        // Check that means are near the cluster centers (order may vary)
        let m1 = &fitted.components[0].mean;
        let m2 = &fitted.components[1].mean;

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
        let fitted = GmmParams::fit(&samples, requested, 300, 1e-6, 1e-4, 0);

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
            fitted.weights.iter().all(|&w| w > 1e-9),
            "no sub-threshold (pruned) weight may remain after filtering, got {:?}",
            fitted.weights
        );

        // (3) Surviving weights renormalize to sum to 1.
        let sum: f64 = fitted.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "surviving weights must renormalize to sum to 1, got {sum}"
        );
    }

    #[test]
    fn test_diagonal_component() {
        let mean = DVector::from_vec(vec![0.5, 0.5]);
        let variances = DVector::from_vec(vec![0.01, 0.04]); // Different variance in each dim
        let comp = GaussianComponent::diagonal(mean, variances);

        let mut rng = rand::rng();
        let mut x_var = 0.0;
        let mut y_var = 0.0;
        let n = 1000;

        for _ in 0..n {
            let sample = comp.sample(&mut rng);
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
        // Test that near-singular covariances are regularized instead of panicking
        let mean = DVector::from_vec(vec![0.5, 0.5]);
        let mut cov = DMatrix::zeros(2, 2);
        cov[(0, 0)] = 1e-12; // Nearly singular
        cov[(1, 1)] = 1e-12;

        // Should not panic, should regularize
        let comp = GaussianComponent::new(mean, cov);
        assert!(comp.is_some());
    }

    #[test]
    fn test_isotropic_clamps_bad_variance() {
        // Non-positive or non-finite variances must be clamped to a small
        // positive value so construction does not panic and the covariance
        // diagonal stays finite and strictly positive.
        for &bad in &[-1.0, 0.0, f64::NAN, f64::INFINITY] {
            let comp = GaussianComponent::isotropic(DVector::from_vec(vec![0.5, 0.5]), bad);
            let diag = comp.covariance.diagonal();
            assert!(
                diag.iter().all(|&v| v.is_finite() && v > 0.0),
                "isotropic variance {bad} produced invalid diagonal {diag:?}"
            );
        }
    }

    #[test]
    fn test_diagonal_clamps_bad_variances() {
        // Each non-positive/non-finite per-dimension variance must be clamped
        // independently, leaving every covariance diagonal entry finite and
        // strictly positive without panicking.
        let mean = DVector::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let variances = DVector::from_vec(vec![-1.0, 0.0, f64::NAN, f64::INFINITY]);
        let comp = GaussianComponent::diagonal(mean, variances);
        let diag = comp.covariance.diagonal();
        assert!(
            diag.iter().all(|&v| v.is_finite() && v > 0.0),
            "diagonal variances produced invalid diagonal {diag:?}"
        );
    }

    #[test]
    fn test_gmm_configuration() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;

        let mut gmm = GmmStrategy::<ContinuousSpace<LinearScale>>::uniform_prior(42, 2, 1.0);
        let mut config = gmm.get_refit_config().clone();
        config.n_components = 5;
        gmm.set_refit_config(config);
        assert_eq!(gmm.get_refit_config().n_components, 5);

        let params = gmm.params();
        let original_n = params.n_components();
        gmm.set_params(params);
        assert_eq!(gmm.params().n_components(), original_n);
    }

    #[test]
    fn test_gmm_determinism_same_seed() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;

        let space = ContinuousSpace::new(0.0, 1.0);
        let params = GmmParams::uniform_prior(1, 0.5);
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
        let params = GmmParams::uniform_prior(1, 0.5);
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
        let gmm = GmmStrategy::<Sp>::uniform_prior(42, 2, 1.0);
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
        let fitted = GmmParams::fit(&samples, 2, 100, 1e-6, 1e-4, 42);
        assert_eq!(fitted.n_components(), 2);
    }

    #[test]
    fn test_gmm_refit_biases_sampling() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;
        use crate::traits::RefittableStrategy;

        let space = ContinuousSpace::new(0.0, 1.0);
        let params = GmmParams::uniform_prior(1, 1.0);
        let mut gmm = GmmStrategy::<ContinuousSpace<LinearScale>>::new(42, params);
        gmm.set_refit_config(GmmRefitConfig {
            n_components: 1,
            max_iters: 100,
            tolerance: 1e-6,
            regularization: 1e-4,
        });

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
        let params = GmmParams::uniform_prior(1, 0.5);
        let mut gmm = GmmStrategy::<ContinuousSpace<LinearScale>>::new(42, params);

        let empty: Vec<(f64, f64)> = vec![];
        gmm.refit(&space, &empty);

        assert_eq!(gmm.params().n_components(), 1);
        assert!(space.contains(&gmm.suggest(&space)));
    }

    #[test]
    fn test_gmm_refit_config_defaults() {
        let config = GmmRefitConfig::default();
        assert_eq!(config.n_components, 3);
        assert_eq!(config.max_iters, 100);
        assert!((config.tolerance - 1e-6).abs() < 1e-12);
        assert!((config.regularization - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_component_dim() {
        let mean = DVector::from_vec(vec![0.1, 0.5, 0.9]);
        let comp = GaussianComponent::isotropic(mean, 0.01);
        assert_eq!(comp.dim(), 3);
    }

    #[test]
    fn test_gmm_params_single_component() {
        let comp = GaussianComponent::isotropic(DVector::from_vec(vec![0.5, 0.5]), 0.1);
        let params = GmmParams::single(comp);
        assert_eq!(params.n_components(), 1);
    }

    #[test]
    fn test_reconcile_after_refit_adopts_higher_counter() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;
        use crate::traits::RefittableStrategy;

        // `live` is the engine's current strategy, whose RNG counter advanced
        // via concurrent `suggest` calls while a refit ran off-lock.
        let live = GmmStrategy::<ContinuousSpace<LinearScale>>::uniform_prior(42, 1, 0.1);
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
    fn test_fit_with_nan_samples_produces_finite_params() {
        // A sample set polluted with NaN/inf rows must still fit a finite
        // mixture; the offending rows are dropped rather than corrupting it.
        let mut samples = vec![
            vec![f64::NAN, 0.5],
            vec![0.5, f64::INFINITY],
            vec![f64::NEG_INFINITY, f64::NAN],
        ];
        for _ in 0..50 {
            samples.push(vec![0.3, 0.7]);
        }

        let fitted = GmmParams::fit(&samples, 1, 100, 1e-4, 1e-4, 42);
        assert_eq!(fitted.dim(), 2);

        for comp in &fitted.components {
            assert!(comp.mean.iter().all(|x| x.is_finite()));
            assert!(comp.covariance.iter().all(|x| x.is_finite()));
        }
        assert!(fitted.weights.iter().all(|w| w.is_finite()));

        // Drawn samples must also be finite and in-cube.
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..100 {
            let s = fitted.sample_clamped(&mut rng);
            assert!(s.iter().all(|x| x.is_finite()));
            assert!(s.iter().all(|&x| (0.0..=1.0).contains(&x)));
        }
    }

    #[test]
    fn test_fit_all_nan_falls_back_to_correct_dim() {
        // If every row is non-finite, fall back to a uniform prior of the
        // original dimensionality (not 1-D) so later suggestions still match.
        let samples = vec![vec![f64::NAN, f64::NAN, 0.5], vec![0.1, f64::INFINITY, 0.2]];
        let fitted = GmmParams::fit(&samples, 2, 50, 1e-4, 1e-4, 1);
        assert_eq!(fitted.dim(), 3);
    }

    #[test]
    fn test_fit_nan_row_filter_matches_finite_only_fit() {
        // Exercises the fit() NaN-row filter: a fit over a NaN-polluted sample
        // set must numerically match a fit over the finite rows alone. The seed,
        // n_components, and cluster geometry are fixed so that, absent the
        // filter, the NaN rows would enter K-means++ and EM and deterministically
        // corrupt at least one component (NaN means/covariances or shifted
        // centers), breaking the equality below.
        let seed = 12345u64;
        let n_components = 2;

        // Two well-separated finite clusters in a fixed order.
        let mut finite: Vec<Vec<f64>> = Vec::new();
        for i in 0..30 {
            let t = i as f64 * 0.001;
            finite.push(vec![0.20 + t, 0.20 + t]);
        }
        for i in 0..30 {
            let t = i as f64 * 0.001;
            finite.push(vec![0.80 - t, 0.80 - t]);
        }

        // Polluted set: identical finite rows in the same relative order, with
        // NaN/inf rows interspersed. After filtering, the surviving rows are
        // byte-for-byte the `finite` set in the same order, so the fit must be
        // identical.
        let mut polluted: Vec<Vec<f64>> = Vec::new();
        polluted.push(vec![f64::NAN, 0.0]);
        for (i, row) in finite.iter().enumerate() {
            polluted.push(row.clone());
            if i % 10 == 9 {
                polluted.push(vec![0.5, f64::INFINITY]);
            }
        }
        polluted.push(vec![f64::NEG_INFINITY, f64::NAN]);

        let reference = GmmParams::fit(&finite, n_components, 100, 1e-9, 1e-4, seed);
        let recovered = GmmParams::fit(&polluted, n_components, 100, 1e-9, 1e-4, seed);

        assert_eq!(recovered.n_components(), reference.n_components());

        // Means and covariances must match the finite-only fit within tolerance.
        // (Absent the filter, the NaN rows would change K-means++ inputs and EM
        // statistics, so these would differ or become NaN.)
        for (rc, rf) in recovered.components.iter().zip(reference.components.iter()) {
            for (a, b) in rc.mean.iter().zip(rf.mean.iter()) {
                assert!(a.is_finite() && b.is_finite());
                assert!((a - b).abs() < 1e-9, "mean mismatch: {a} vs {b}");
            }
            for (a, b) in rc.covariance.iter().zip(rf.covariance.iter()) {
                assert!(a.is_finite() && b.is_finite());
                assert!((a - b).abs() < 1e-9, "covariance mismatch: {a} vs {b}");
            }
        }
        for (a, b) in recovered.weights.iter().zip(reference.weights.iter()) {
            assert!((a - b).abs() < 1e-9, "weight mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_sample_clamped_replaces_non_finite() {
        // A component whose draws are non-finite must not leak NaN/inf out of
        // sample_clamped, since f64::clamp propagates NaN.
        let comp =
            GaussianComponent::isotropic(DVector::from_vec(vec![f64::NAN, f64::INFINITY]), 0.01);
        let params = GmmParams::single(comp);
        let mut rng = SmallRng::seed_from_u64(3);
        for _ in 0..50 {
            let s = params.sample_clamped(&mut rng);
            assert!(s.iter().all(|x| x.is_finite()));
            assert!(s.iter().all(|&x| (0.0..=1.0).contains(&x)));
        }
    }

    #[test]
    fn test_deserialize_malformed_gmm_does_not_panic() {
        // Build a serialized GMM whose weights do not sum to 1 and whose two
        // components have mismatched dimensionality. Without the recovering
        // serde path this builds an invalid mixture (the bypassed `new`
        // invariants) that later panics in `suggest`. Deserialization must
        // recover into a valid mixture instead.
        let c2 = GaussianComponent::isotropic(DVector::from_vec(vec![0.5, 0.5]), 0.01);
        let c1 = GaussianComponent::isotropic(DVector::from_vec(vec![0.5]), 0.01);

        // Use the serde shape directly so we control the (invalid) field values.
        let serde_form = serde_json::json!({
            "weights": [0.2, 0.2],
            "components": [
                GaussianComponentSerde::from(c1),
                GaussianComponentSerde::from(c2),
            ],
        });

        let params: GmmParams =
            serde_json::from_value(serde_form).expect("should recover, not panic");
        let sum: f64 = params.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "recovered weights must sum to 1");
        assert!(params.dim() >= 1);

        // All components must share one dimensionality and be finite.
        let d = params.dim();
        for comp in &params.components {
            assert_eq!(comp.dim(), d);
            assert!(comp.mean.iter().all(|x| x.is_finite()));
            assert!(comp.covariance.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_deserialize_drops_bad_component_keeps_valid_mixture() {
        // A mixture with one malformed component (dimension mismatch: a 1-D mean
        // paired with a 2x2 covariance) and one valid 2-D component must recover
        // by DROPPING the bad component and keeping the valid one at its actual
        // dimensionality, not by demoting the whole mixture to a generic 1-D
        // uniform prior. The surviving component's mean is distinctive (0.9, 0.1)
        // so this also fails if recovery silently substituted a uniform_prior
        // centered at (0.5, 0.5).
        //
        // The bad component is built as structurally-valid JSON (a parseable
        // GaussianComponentSerde) so it survives serde deserialization and is
        // rejected later by try_into_component's dims_ok check. A NaN/inf
        // covariance value would instead serialize to JSON null and fail
        // GmmParamsSerde at the serde-type level, before recovery can run.
        let good = GaussianComponent::isotropic(DVector::from_vec(vec![0.9, 0.1]), 0.01);

        // mean length 1 but a 2x2 covariance: parseable, but a dimension mismatch
        // that try_into_component must reject and recovery must drop.
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

        let params: GmmParams =
            serde_json::from_value(serde_form).expect("should recover, not panic");

        // Dimensionality is preserved (2-D), the bad component is gone, and the
        // surviving component is the valid one (not a substituted prior).
        assert_eq!(params.dim(), 2, "must preserve mixture dimensionality");
        assert_eq!(params.n_components(), 1, "bad component must be dropped");
        let mean = &params.components[0].mean;
        assert!((mean[0] - 0.9).abs() < 1e-9 && (mean[1] - 0.1).abs() < 1e-9);

        // Surviving weight is renormalized to sum to 1.
        let sum: f64 = params.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "recovered weights must sum to 1");
    }

    #[test]
    fn test_deserialize_empty_gmm_recovers() {
        // An empty mixture would make `new`/`suggest` panic; recover to a prior.
        let serde_form = serde_json::json!({
            "weights": serde_json::Value::Array(vec![]),
            "components": serde_json::Value::Array(vec![]),
        });
        let params: GmmParams =
            serde_json::from_value(serde_form).expect("should recover, not panic");
        assert!(params.n_components() >= 1);
        assert!(params.dim() >= 1);
    }

    #[test]
    fn test_suggest_does_not_panic_on_dim_mismatch() {
        use crate::scales::LinearScale;
        use crate::spaces::ContinuousSpace;

        // The space is 1-D but the GMM is 2-D: a dim mismatch on the hot path.
        let space = ContinuousSpace::new(0.0, 1.0);
        let params = GmmParams::uniform_prior(2, 0.1);
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
        let params = GmmParams::uniform_prior(1, 1.0);
        let mut strat = GmmStrategy::<ContinuousSpace<LinearScale>>::new(42, params);
        strat.set_refit_config(GmmRefitConfig {
            n_components: 2,
            max_iters: 100,
            tolerance: 1e-9,
            regularization: 1e-4,
        });

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
            strat.params().n_components() >= 2,
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
        let comp1 = GaussianComponent::isotropic(DVector::from_vec(vec![0.1]), 0.001);
        let comp2 = GaussianComponent::isotropic(DVector::from_vec(vec![0.9]), 0.001);
        let params = GmmParams::new(vec![0.9, 0.1], vec![comp1, comp2]); // 90% first, 10% second

        let mut rng = rand::rng();
        let mut near_first = 0;

        for _ in 0..1000 {
            let sample = params.sample_clamped(&mut rng);
            if sample[0] < 0.5 {
                near_first += 1;
            }
        }

        // Should be approximately 90% near first component
        assert!(near_first > 800 && near_first < 980);
    }
}
