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
    pub fn isotropic(mean: DVector<f64>, variance: f64) -> Self {
        let dim = mean.len();
        let covariance = DMatrix::identity(dim, dim) * variance;
        Self::new(mean, covariance).expect("Isotropic covariance should always be valid")
    }

    /// Create a diagonal Gaussian component.
    pub fn diagonal(mean: DVector<f64>, variances: DVector<f64>) -> Self {
        let dim = mean.len();
        assert_eq!(variances.len(), dim);
        let covariance = DMatrix::from_diagonal(&variances);
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

impl From<GaussianComponentSerde> for GaussianComponent {
    fn from(s: GaussianComponentSerde) -> Self {
        // Recompute cached values on deserialization
        GaussianComponent::new(s.mean, s.covariance)
            .expect("Failed to reconstruct GaussianComponent from serialized data")
    }
}

/// Parameters for a Gaussian Mixture Model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GmmParams {
    /// Mixture weights (must sum to 1, all positive).
    pub weights: Vec<f64>,
    /// Gaussian components.
    pub components: Vec<GaussianComponent>,
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
    pub fn sample_clamped<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let sample = self.sample_unclamped(rng);
        sample.iter().map(|&x| x.clamp(0.0, 1.0)).collect()
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
            return Self::uniform_prior(1, 0.1);
        }

        let dim = samples[0].len();
        let n_samples = samples.len();

        // Flatten data for cache-friendly access (each sample is contiguous)
        let flat_data: Vec<f64> = samples.iter().flat_map(|s| s.iter().copied()).collect();

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

        // Build final components, keeping weights aligned with survivors
        let (surviving_weights, final_components): (Vec<f64>, Vec<GaussianComponent>) = weights
            .into_iter()
            .zip(means.into_iter().zip(covs))
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

        assert_eq!(
            params.dim(),
            dim,
            "GMM dimensionality ({}) must match space dimensionality ({})",
            params.dim(),
            dim
        );

        let call_index = self.counter.fetch_add(1, Ordering::Relaxed);
        let call_seed = self
            .seed
            .wrapping_add(call_index.wrapping_mul(6364136223846793005));
        let mut rng = SmallRng::seed_from_u64(call_seed);
        let sample = params.sample_clamped(&mut rng);

        space.from_unit_cube(&sample).expect("Mapping failed")
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
        // Test that we handle near-singular covariances gracefully
        let mean = DVector::from_vec(vec![0.5, 0.5]);
        let mut cov = DMatrix::zeros(2, 2);
        cov[(0, 0)] = 1e-12; // Nearly singular
        cov[(1, 1)] = 1e-12;

        // Should not panic, should regularize
        let comp = GaussianComponent::new(mean, cov);
        assert!(comp.is_some());
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
