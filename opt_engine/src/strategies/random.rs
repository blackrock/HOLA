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

//! Random search strategy.

use crate::traits::{StandardizedSpace, Strategy};
use rand::SeedableRng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// Uniform random sampling in the unit hypercube.
///
/// Each call to `suggest` generates an independent random point using a
/// deterministic seed derived from the strategy's base seed and an
/// auto-incrementing counter, so results are reproducible given the same
/// seed. The space must implement
/// [`StandardizedSpace`](crate::traits::StandardizedSpace) so the strategy
/// can sample in `[0, 1]^n` and map back to the domain.
///
/// # Serialization
///
/// The seed and counter are serialized for checkpointing. When restored,
/// the sequence continues from where it left off.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RandomStrategy<S, Obs = f64> {
    seed: u64,
    #[serde(
        serialize_with = "serialize_atomic",
        deserialize_with = "deserialize_atomic"
    )]
    counter: AtomicU64,
    #[serde(skip)]
    _marker: PhantomData<fn() -> (S, Obs)>,
}

fn serialize_atomic<S>(val: &AtomicU64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_u64(val.load(Ordering::Relaxed))
}

fn deserialize_atomic<'de, D>(deserializer: D) -> Result<AtomicU64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val = u64::deserialize(deserializer)?;
    Ok(AtomicU64::new(val))
}

impl<S, Obs> Clone for RandomStrategy<S, Obs> {
    fn clone(&self) -> Self {
        Self {
            seed: self.seed,
            counter: AtomicU64::new(self.counter.load(Ordering::Relaxed)),
            _marker: PhantomData,
        }
    }
}

impl<S, Obs> RandomStrategy<S, Obs> {
    /// Create a random strategy with the given seed for reproducibility.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            counter: AtomicU64::new(0),
            _marker: PhantomData,
        }
    }

    /// Create a random strategy with an automatically chosen seed.
    pub fn auto_seed() -> Self {
        Self::new(rand::random())
    }
}

impl<S, Obs> Default for RandomStrategy<S, Obs> {
    fn default() -> Self {
        Self::auto_seed()
    }
}

impl<S, Obs> Strategy for RandomStrategy<S, Obs>
where
    S: StandardizedSpace,
    Obs: Serialize + DeserializeOwned + Send + Sync + Clone + Debug + 'static,
{
    type Space = S;
    type Observation = Obs;

    fn suggest(&self, space: &Self::Space) -> S::Domain {
        let dim = space.dimensionality();
        // Derive a per-call seed from the base seed and counter
        let call_index = self.counter.fetch_add(1, Ordering::Relaxed);
        let call_seed = self
            .seed
            .wrapping_add(call_index.wrapping_mul(6364136223846793005));
        let mut rng = rand::rngs::SmallRng::seed_from_u64(call_seed);

        let random_vec: Vec<f64> = (0..dim)
            .map(|_| rand::Rng::random_range(&mut rng, 0.0..1.0))
            .collect();

        space.from_unit_cube(&random_vec).expect("Mapping failed")
    }

    fn update(&mut self, _candidate: &S::Domain, _result: Obs) {
        // Pure sampler — nothing to update.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scales::LinearScale;
    use crate::spaces::{ContinuousSpace, DiscreteSpace, ProductSpace};
    use crate::traits::SampleSpace;

    #[test]
    fn test_in_bounds() {
        let space = ContinuousSpace::new(-10.0, 10.0);
        let strat: RandomStrategy<_, f64> = RandomStrategy::auto_seed();
        for _ in 0..50 {
            let candidate = strat.suggest(&space);
            assert!(
                space.contains(&candidate),
                "Random candidate {candidate} out of bounds"
            );
        }
    }

    #[test]
    fn test_update_is_noop() {
        let space = ContinuousSpace::new(0.0, 1.0);
        let mut strat: RandomStrategy<_, f64> = RandomStrategy::auto_seed();
        strat.update(&0.5, 0.42);
        let candidate = strat.suggest(&space);
        assert!(space.contains(&candidate));
    }

    #[test]
    fn test_determinism_same_seed() {
        let space = ContinuousSpace::new(0.0, 1.0);
        let strat1 = RandomStrategy::<ContinuousSpace<LinearScale>, f64>::new(42);
        let strat2 = RandomStrategy::<ContinuousSpace<LinearScale>, f64>::new(42);
        for i in 0..20 {
            let a = strat1.suggest(&space);
            let b = strat2.suggest(&space);
            assert_eq!(a, b, "Mismatch at suggestion {i}");
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let space = ContinuousSpace::new(0.0, 1.0);
        let strat1 = RandomStrategy::<ContinuousSpace<LinearScale>, f64>::new(42);
        let strat2 = RandomStrategy::<ContinuousSpace<LinearScale>, f64>::new(123);
        assert_ne!(strat1.suggest(&space), strat2.suggest(&space));
    }

    #[test]
    fn test_multi_dimensional() {
        let space = ProductSpace {
            a: ContinuousSpace::new(-5.0, 5.0),
            b: DiscreteSpace::new(0, 10),
        };
        let strat =
            RandomStrategy::<ProductSpace<ContinuousSpace<LinearScale>, DiscreteSpace>, f64>::new(
                99,
            );
        for _ in 0..50 {
            let candidate = strat.suggest(&space);
            assert!(space.contains(&candidate));
        }
    }
}
