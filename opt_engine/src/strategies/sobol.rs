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

use crate::traits::{StandardizedSpace, Strategy};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};

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
    fn default() -> Self {
        Self::auto_seed()
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

        // Generate Sobol' point for each dimension
        let sobol_vec: Vec<f64> = (0..dim)
            .map(|d| sobol_burley::sample(current_index, d as u32, self.seed) as f64)
            .collect();

        space.from_unit_cube(&sobol_vec).expect("Mapping failed")
    }

    fn update(&mut self, _candidate: &S::Domain, _result: Obs) {
        // Pure sampler — nothing to update.
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
}
