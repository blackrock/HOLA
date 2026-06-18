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

//! Sobol' sequence strategy for quasi-random sampling.
//!
//! Sobol' sequences are low-discrepancy sequences that provide better coverage
//! of the search space compared to pseudo-random sampling. This makes them
//! particularly effective for initial exploration in optimization.
//!
//! `sobol_burley` provides a valid low-discrepancy Sobol' sequence only for the
//! first `2^16` points: `sobol_burley::sample` debug-asserts the sample index is
//! below `2^16`, and in release it masks the index, yielding unspecified or
//! duplicate points beyond that. It also supports at most 256 dimensions. This
//! strategy therefore uses Sobol' for the first `2^16` draws on spaces of at
//! most 256 dimensions, and otherwise falls back to deterministic seeded
//! pseudo-random sampling, so it never panics and never emits invalid or
//! duplicate points.

use crate::traits::{StandardizedSpace, Strategy};
use rand::SeedableRng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};

/// Maximum dimensionality supported by `sobol_burley::sample`.
///
/// The crate ships 256 dimensions of direction numbers and has an always-on
/// `assert!(dimension < 256)` that panics in release as well as debug. This
/// constant backs a defensive secondary net in `suggest`; the product layer
/// (hola) is the primary guard and rejects Sobol on spaces beyond 256
/// dimensions with a clear error.
const MAX_SOBOL_DIMS: usize = 256;

/// Maximum number of points `sobol_burley::sample` produces before its sequence
/// stops being valid.
///
/// `sobol_burley::sample` has a `debug_assert!(sample_index < 1 << 16)` that
/// panics in debug builds past `2^16`, and in release builds it masks the index
/// to the top bits, yielding unspecified or duplicate points. `suggest` switches
/// to deterministic pseudo-random sampling once the index reaches this bound.
const MAX_SOBOL_SAMPLES: u32 = 1 << 16;

/// Quasi-random sampling via Sobol' sequences with Owen scrambling.
///
/// Uses Sobol' sequences with Owen scrambling (Burley's 2020 variant) to generate
/// quasi-random points in the unit hypercube, which are then mapped to the domain.
/// This provides better space coverage than pseudo-random sampling, which is
/// desirable for the initial exploration phase of optimization.
///
/// The observation type `Obs` is generic because the strategy never uses it —
/// Sobol' is a pure sampler.
///
/// # Serialization
///
/// The sequence state (seed + index) is fully serializable, so checkpointing
/// preserves the exact position in the Sobol' sequence.
///
/// # Example
/// ```ignore
/// // 1-D continuous space with Log10 scale
/// let space = ContinuousSpace::with_scale(1e-4, 0.1, Log10Scale);
/// let strategy = SobolStrategy::<_, f64>::new(42); // explicit observation type
/// let strategy = SobolStrategy::new(42);            // defaults to f64
///
/// // Generate points
/// let point1 = strategy.suggest(&space);
/// let point2 = strategy.suggest(&space); // different point, same sequence
/// ```
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SobolStrategy<S, Obs = f64> {
    /// Current index in the Sobol' sequence (serialized as u32).
    #[serde(
        serialize_with = "serialize_atomic",
        deserialize_with = "deserialize_atomic"
    )]
    index: AtomicU32,
    /// Seed for Owen scrambling (provides randomization while preserving low-discrepancy).
    seed: u32,
    #[serde(skip)]
    _marker: PhantomData<fn() -> (S, Obs)>,
}

fn serialize_atomic<S>(val: &AtomicU32, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_u32(val.load(Ordering::Relaxed))
}

fn deserialize_atomic<'de, D>(deserializer: D) -> Result<AtomicU32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val = u32::deserialize(deserializer)?;
    Ok(AtomicU32::new(val))
}

impl<S, Obs> Clone for SobolStrategy<S, Obs> {
    fn clone(&self) -> Self {
        Self {
            index: AtomicU32::new(self.index.load(Ordering::Relaxed)),
            seed: self.seed,
            _marker: PhantomData,
        }
    }
}

impl<S, Obs> SobolStrategy<S, Obs> {
    /// Create a new Sobol' strategy with the given scrambling seed.
    ///
    /// The seed randomizes the sequence while preserving its low-discrepancy properties.
    /// Different seeds produce different (but equally valid) Sobol' sequences.
    pub fn new(seed: u32) -> Self {
        Self {
            index: AtomicU32::new(0),
            seed,
            _marker: PhantomData,
        }
    }

    /// Create a new Sobol' strategy with an automatically chosen seed.
    pub fn auto_seed() -> Self {
        Self::new(rand::random())
    }
}

impl<S, Obs> Default for SobolStrategy<S, Obs> {
    /// Deterministic default using a fixed seed of 0.
    ///
    /// `Default` must honor the reproducibility contract, so it delegates to
    /// `Self::new(0)` rather than `auto_seed()` (which draws a nondeterministic
    /// seed). Use `auto_seed()` explicitly when a random seed is wanted.
    fn default() -> Self {
        Self::new(0)
    }
}

impl<S, Obs> Strategy for SobolStrategy<S, Obs>
where
    S: StandardizedSpace,
    Obs: Serialize + DeserializeOwned + Send + Sync + Clone + Debug + 'static,
{
    type Space = S;
    type Observation = Obs;

    fn suggest(&self, space: &Self::Space) -> S::Domain {
        let dim = space.dimensionality();
        // Atomically fetch the current index and increment it
        let current_index = self.index.fetch_add(1, Ordering::Relaxed);

        // Two conditions make `sobol_burley::sample` unsafe to call:
        //   - dimensions beyond 256: the crate has an always-on
        //     `assert!(dimension < 256)` that panics in release as well as
        //     debug. The product layer (hola) is the primary guard and rejects
        //     Sobol on such spaces; this is a defensive secondary net.
        //   - sample index at or beyond `2^16`: the crate `debug_assert!`s the
        //     index is below `2^16` (panicking in debug) and masks the index in
        //     release, producing unspecified or duplicate points.
        // In either case fall back to deterministic seeded pseudo-random
        // sampling so the strategy never panics and never emits invalid points.
        let unit_vec: Vec<f64> = if dim > MAX_SOBOL_DIMS || current_index >= MAX_SOBOL_SAMPLES {
            // Silent fallback. The hola product layer is the primary guard for
            // the dimension limit and emits a single construction-time warning,
            // so a per-call warning here would be redundant and noisy during the
            // exploration phase.
            let call_seed = (self.seed as u64)
                .wrapping_add((current_index as u64).wrapping_mul(6364136223846793005));
            let mut rng = rand::rngs::SmallRng::seed_from_u64(call_seed);
            (0..dim)
                .map(|_| rand::Rng::random_range(&mut rng, 0.0..1.0))
                .collect()
        } else {
            (0..dim)
                .map(|d| sobol_burley::sample(current_index, d as u32, self.seed) as f64)
                .collect()
        };

        space.from_unit_cube(&unit_vec).unwrap_or_else(|| {
            // `from_unit_cube` only returns `None` on a length mismatch, which
            // should not happen since the vector matches `dimensionality()`.
            // Fall back to a deterministic in-cube midpoint rather than panic.
            eprintln!(
                "Warning: Sobol' unit-cube mapping failed; falling back to the cube midpoint"
            );
            space
                .from_unit_cube(&vec![0.5; dim])
                .expect("midpoint of the unit cube is always a valid mapping")
        })
    }

    fn update(&mut self, _candidate: &S::Domain, _result: Obs) {
        // Pure sampler: nothing to update.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::UnitInterval;

    #[test]
    fn test_sobol_deterministic_with_same_seed() {
        let strategy1 = SobolStrategy::<UnitInterval>::new(42);
        let strategy2 = SobolStrategy::<UnitInterval>::new(42);
        let space = UnitInterval;

        let point1 = strategy1.suggest(&space);
        let point2 = strategy2.suggest(&space);

        assert_eq!(point1, point2, "Same seed should produce same sequence");
    }

    #[test]
    fn test_sobol_different_seeds_differ() {
        let strategy1 = SobolStrategy::<UnitInterval>::new(42);
        let strategy2 = SobolStrategy::<UnitInterval>::new(123);
        let space = UnitInterval;

        let point1 = strategy1.suggest(&space);
        let point2 = strategy2.suggest(&space);

        assert_ne!(
            point1, point2,
            "Different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_sobol_advances_index() {
        let strategy = SobolStrategy::<UnitInterval>::new(42);
        let space = UnitInterval;

        let point1 = strategy.suggest(&space);
        let point2 = strategy.suggest(&space);

        assert_ne!(
            point1, point2,
            "Successive calls should produce different points"
        );
    }

    #[test]
    fn test_sobol_points_in_unit_interval() {
        let strategy = SobolStrategy::<UnitInterval>::new(42);
        let space = UnitInterval;

        for _ in 0..100 {
            let point = strategy.suggest(&space);
            assert!(
                (0.0..=1.0).contains(&point),
                "Sobol point {point} should be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_sobol_multi_dimensional() {
        use crate::scales::LinearScale;
        use crate::spaces::{ContinuousSpace, ProductSpace};
        use crate::traits::SampleSpace;

        let space = ProductSpace {
            a: ContinuousSpace::new(0.0, 1.0),
            b: ProductSpace {
                a: ContinuousSpace::new(-10.0, 10.0),
                b: ContinuousSpace::new(100.0, 200.0),
            },
        };
        assert_eq!(space.dimensionality(), 3);

        type Sp = ProductSpace<
            ContinuousSpace<LinearScale>,
            ProductSpace<ContinuousSpace<LinearScale>, ContinuousSpace<LinearScale>>,
        >;
        let strat = SobolStrategy::<Sp>::new(42);

        let mut points = Vec::new();
        for _ in 0..50 {
            let p = strat.suggest(&space);
            assert!(space.contains(&p));
            points.push(p);
        }

        // All points should be distinct
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                assert_ne!(points[i], points[j]);
            }
        }
    }

    #[test]
    fn test_sobol_space_filling() {
        use crate::scales::LinearScale;
        use crate::spaces::{ContinuousSpace, ProductSpace};

        let space = ProductSpace {
            a: ContinuousSpace::new(0.0, 1.0),
            b: ContinuousSpace::new(0.0, 1.0),
        };
        type Sp = ProductSpace<ContinuousSpace<LinearScale>, ContinuousSpace<LinearScale>>;
        let strat = SobolStrategy::<Sp>::new(42);

        let mut quadrant_counts = [0u32; 4];
        for _ in 0..100 {
            let (x, y) = strat.suggest(&space);
            let q = match (x < 0.5, y < 0.5) {
                (true, true) => 0,
                (false, true) => 1,
                (true, false) => 2,
                (false, false) => 3,
            };
            quadrant_counts[q] += 1;
        }

        for (i, &count) in quadrant_counts.iter().enumerate() {
            assert!(
                count >= 10,
                "Quadrant {i} has only {count} points (expected >= 10)"
            );
        }
    }

    #[test]
    fn test_sobol_auto_seed_differs() {
        let space = UnitInterval;
        let strat1 = SobolStrategy::<UnitInterval>::auto_seed();
        let strat2 = SobolStrategy::<UnitInterval>::auto_seed();
        assert_ne!(strat1.suggest(&space), strat2.suggest(&space));
    }

    #[test]
    fn test_sobol_default_is_deterministic() {
        // `Default` must honor the reproducibility contract: two default-
        // constructed strategies use the same fixed seed and so produce
        // identical sequences.
        let space = UnitInterval;
        let strat1 = SobolStrategy::<UnitInterval>::default();
        let strat2 = SobolStrategy::<UnitInterval>::default();
        for i in 0..20 {
            let a = strat1.suggest(&space);
            let b = strat2.suggest(&space);
            assert_eq!(a, b, "Default strategies diverged at suggestion {i}");
        }
    }

    use crate::traits::SampleSpace;

    /// A test space with arbitrary dimensionality mapping to a `Vec<f64>`.
    struct HighDimSpace {
        dim: usize,
    }

    impl SampleSpace for HighDimSpace {
        type Domain = Vec<f64>;

        fn contains(&self, point: &Self::Domain) -> bool {
            point.len() == self.dim && point.iter().all(|x| (0.0..=1.0).contains(x))
        }
    }

    impl StandardizedSpace for HighDimSpace {
        fn dimensionality(&self) -> usize {
            self.dim
        }

        fn to_unit_cube(&self, point: &Self::Domain) -> Vec<f64> {
            point.clone()
        }

        fn from_unit_cube(&self, vec: &[f64]) -> Option<Self::Domain> {
            if vec.len() == self.dim {
                Some(vec.to_vec())
            } else {
                None
            }
        }
    }

    /// A space whose `from_unit_cube` rejects the strategy's raw sample (here,
    /// any vector other than the exact midpoint) but accepts the `vec![0.5; n]`
    /// fallback. This exercises the non-panicking fallback path without making
    /// a valid domain point unconstructible.
    struct MismatchSpace;

    impl SampleSpace for MismatchSpace {
        type Domain = f64;

        fn contains(&self, _point: &Self::Domain) -> bool {
            true
        }
    }

    impl StandardizedSpace for MismatchSpace {
        fn dimensionality(&self) -> usize {
            1
        }

        fn to_unit_cube(&self, point: &Self::Domain) -> Vec<f64> {
            vec![*point]
        }

        fn from_unit_cube(&self, vec: &[f64]) -> Option<Self::Domain> {
            match vec.first() {
                Some(&x) if x == 0.5 => Some(x),
                _ => None,
            }
        }
    }

    #[test]
    fn test_sobol_high_dimensional_does_not_panic() {
        // More than the 256-dimension limit of `sobol_burley::sample`: must not
        // reach the crate's always-on assert.
        let space = HighDimSpace { dim: 300 };
        let strat = SobolStrategy::<HighDimSpace>::new(42);
        let point = strat.suggest(&space);
        assert_eq!(point.len(), 300);
        assert!(space.contains(&point), "fallback point should be in bounds");
    }

    #[test]
    fn test_sobol_mapping_mismatch_does_not_panic() {
        let space = MismatchSpace;
        let strat = SobolStrategy::<MismatchSpace>::new(42);
        // The raw Sobol' sample is rejected; `suggest` must fall back to the
        // midpoint instead of panicking on `expect`.
        let point = strat.suggest(&space);
        assert_eq!(point, 0.5, "expected the midpoint fallback value");
    }

    #[test]
    fn test_sobol_sample_count_limit_does_not_panic() {
        // Low-dimensional space so the dimension guard is not exercised; only the
        // sample-count guard can prevent a panic here.
        let space = UnitInterval;
        let strat = SobolStrategy::<UnitInterval>::new(42);

        // Position the next draw at `2^16 - 1` (still valid for Sobol').
        strat.index.store(MAX_SOBOL_SAMPLES - 1, Ordering::Relaxed);

        // Index `2^16 - 1`: last valid Sobol' draw.
        let p1 = strat.suggest(&space);
        assert!((0.0..=1.0).contains(&p1), "Sobol point {p1} out of bounds");

        // Index `2^16`: without the sample-count guard this would trip
        // `sobol_burley::sample`'s `debug_assert!(sample_index < 1 << 16)` and
        // panic in a test build. The guard routes it to the deterministic
        // pseudo-random fallback instead.
        let p2 = strat.suggest(&space);
        assert!(
            (0.0..=1.0).contains(&p2),
            "fallback point {p2} out of bounds"
        );
    }
}
