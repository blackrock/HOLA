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

//! Combinators for composing primitive spaces into richer parameter spaces.

use crate::traits::{SampleSpace, StandardizedSpace};
use serde::{Deserialize, Serialize};

/// Cartesian product of two spaces, producing a tuple domain `(A, B)`.
///
/// Use this to combine independent parameters into a single search space.
/// Products can be nested: `ProductSpace<ProductSpace<A, B>, C>` gives `((A, B), C)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductSpace<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> SampleSpace for ProductSpace<A, B>
where
    A: SampleSpace,
    B: SampleSpace,
{
    type Domain = (A::Domain, B::Domain);

    fn contains(&self, point: &Self::Domain) -> bool {
        self.a.contains(&point.0) && self.b.contains(&point.1)
    }

    fn clamp(&self, point: &Self::Domain) -> Self::Domain
    where
        Self::Domain: Clone,
    {
        (self.a.clamp(&point.0), self.b.clamp(&point.1))
    }
}

impl<A, B> StandardizedSpace for ProductSpace<A, B>
where
    A: StandardizedSpace,
    B: StandardizedSpace,
{
    fn dimensionality(&self) -> usize {
        self.a.dimensionality() + self.b.dimensionality()
    }

    fn to_unit_cube(&self, point: &Self::Domain) -> Vec<f64> {
        let mut vec = self.a.to_unit_cube(&point.0);
        vec.extend(self.b.to_unit_cube(&point.1));
        vec
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<Self::Domain> {
        let split = self.a.dimensionality();
        if vec.len() < split {
            return None;
        }

        let (left_vec, right_vec) = vec.split_at(split);

        let left_val = self.a.from_unit_cube(left_vec)?;
        let right_val = self.b.from_unit_cube(right_vec)?;

        Some((left_val, right_val))
    }
}

/// Domain type for `BranchingSpace`.
/// Represents a choice between value from space A OR space B.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum EitherDomain<A, B> {
    Left(A),
    Right(B),
}

/// Disjoint union of two spaces — the domain is either a value from `A` or
/// from `B`, but not both.
///
/// In the unit-hypercube representation, the first dimension is a routing
/// selector: values below 0.5 activate the left branch, values at or above
/// 0.5 activate the right branch. Inactive dimensions are zero-padded.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BranchingSpace<A, B> {
    pub left: A,
    pub right: B,
}

impl<A, B> SampleSpace for BranchingSpace<A, B>
where
    A: SampleSpace,
    B: SampleSpace,
{
    type Domain = EitherDomain<A::Domain, B::Domain>;

    fn contains(&self, point: &Self::Domain) -> bool {
        match point {
            EitherDomain::Left(val) => self.left.contains(val),
            EitherDomain::Right(val) => self.right.contains(val),
        }
    }

    fn clamp(&self, point: &Self::Domain) -> Self::Domain
    where
        Self::Domain: Clone,
    {
        match point {
            EitherDomain::Left(val) => EitherDomain::Left(self.left.clamp(val)),
            EitherDomain::Right(val) => EitherDomain::Right(self.right.clamp(val)),
        }
    }
}

impl<A, B> StandardizedSpace for BranchingSpace<A, B>
where
    A: StandardizedSpace,
    B: StandardizedSpace,
{
    /// Layout: [routing_selector, ...padded_branch_dims]
    /// Total = 1 + max(dim(A), dim(B))
    fn dimensionality(&self) -> usize {
        1 + self.left.dimensionality().max(self.right.dimensionality())
    }

    fn to_unit_cube(&self, point: &Self::Domain) -> Vec<f64> {
        let max_branch_dim = self.left.dimensionality().max(self.right.dimensionality());
        let mut vec = Vec::with_capacity(1 + max_branch_dim);

        match point {
            EitherDomain::Left(val) => {
                vec.push(0.25); // Routing: < 0.5 = Left
                let branch_vec = self.left.to_unit_cube(val);
                vec.extend(&branch_vec);
                // Zero-pad inactive dimensions
                vec.extend(std::iter::repeat_n(
                    0.0,
                    max_branch_dim.saturating_sub(branch_vec.len()),
                ));
            }
            EitherDomain::Right(val) => {
                vec.push(0.75); // Routing: >= 0.5 = Right
                let branch_vec = self.right.to_unit_cube(val);
                vec.extend(&branch_vec);
                vec.extend(std::iter::repeat_n(
                    0.0,
                    max_branch_dim.saturating_sub(branch_vec.len()),
                ));
            }
        }
        vec
    }

    fn from_unit_cube(&self, vec: &[f64]) -> Option<Self::Domain> {
        if vec.is_empty() {
            return None;
        }

        let selector = vec[0];
        let branch_dims = &vec[1..];

        if selector < 0.5 {
            let left_dim = self.left.dimensionality();
            if branch_dims.len() < left_dim {
                return None;
            }
            let val = self.left.from_unit_cube(&branch_dims[..left_dim])?;
            Some(EitherDomain::Left(val))
        } else {
            let right_dim = self.right.dimensionality();
            if branch_dims.len() < right_dim {
                return None;
            }
            let val = self.right.from_unit_cube(&branch_dims[..right_dim])?;
            Some(EitherDomain::Right(val))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::{CategoricalSpace, ContinuousSpace, DiscreteSpace};

    // ======================================================================
    // ProductSpace
    // ======================================================================

    #[test]
    fn test_product_space_contains() {
        let space = ProductSpace {
            a: ContinuousSpace::new(0.0, 1.0),
            b: DiscreteSpace::new(1, 5),
        };
        assert!(space.contains(&(0.5, 3)));
        assert!(!space.contains(&(1.5, 3)));
        assert!(!space.contains(&(0.5, 6)));
        assert!(!space.contains(&(-0.1, 0)));
    }

    #[test]
    fn test_product_space_clamp() {
        let space = ProductSpace {
            a: ContinuousSpace::new(0.0, 1.0),
            b: DiscreteSpace::new(1, 5),
        };
        let clamped = space.clamp(&(2.0, 10));
        assert_eq!(clamped.0, 1.0);
        assert_eq!(clamped.1, 5);

        let clamped2 = space.clamp(&(-1.0, -5));
        assert_eq!(clamped2.0, 0.0);
        assert_eq!(clamped2.1, 1);
    }

    #[test]
    fn test_product_space_unit_cube_roundtrip() {
        let space = ProductSpace {
            a: ContinuousSpace::new(-10.0, 10.0),
            b: ContinuousSpace::new(0.0, 100.0),
        };
        assert_eq!(space.dimensionality(), 2);

        let point = (3.0, 75.0);
        let unit = space.to_unit_cube(&point);
        assert_eq!(unit.len(), 2);
        assert!(unit.iter().all(|x| *x >= 0.0 && *x <= 1.0));

        let recon = space.from_unit_cube(&unit).unwrap();
        assert!((recon.0 - 3.0).abs() < 1e-9);
        assert!((recon.1 - 75.0).abs() < 1e-9);
    }

    #[test]
    fn test_product_space_from_unit_cube_too_short() {
        let space = ProductSpace {
            a: ContinuousSpace::new(0.0, 1.0),
            b: ContinuousSpace::new(0.0, 1.0),
        };
        assert!(space.from_unit_cube(&[0.5]).is_none());
        assert!(space.from_unit_cube(&[]).is_none());
    }

    #[test]
    fn test_product_space_continuous_discrete_categorical() {
        let space = ProductSpace {
            a: ProductSpace {
                a: ContinuousSpace::new(0.0, 1.0),
                b: DiscreteSpace::new(1, 10),
            },
            b: CategoricalSpace::from_strs(&["a", "b", "c"]),
        };
        assert_eq!(space.dimensionality(), 3);

        let point = ((0.5, 5), "b".to_string());
        assert!(space.contains(&point));

        let unit = space.to_unit_cube(&point);
        assert_eq!(unit.len(), 3);
        assert!(unit.iter().all(|x| *x >= 0.0 && *x <= 1.0));

        let restored = space.from_unit_cube(&unit).unwrap();
        assert!((restored.0.0 - 0.5).abs() < 1e-9);
        assert_eq!(restored.0.1, 5);
        assert_eq!(restored.1, "b");
    }

    #[test]
    fn test_product_space_deeply_nested() {
        let space = ProductSpace {
            a: ProductSpace {
                a: ContinuousSpace::new(0.0, 1.0),
                b: ContinuousSpace::new(0.0, 1.0),
            },
            b: ProductSpace {
                a: ContinuousSpace::new(0.0, 1.0),
                b: ContinuousSpace::new(0.0, 1.0),
            },
        };
        assert_eq!(space.dimensionality(), 4);

        let point = ((0.1, 0.2), (0.3, 0.4));
        let unit = space.to_unit_cube(&point);
        assert_eq!(unit.len(), 4);

        let restored = space.from_unit_cube(&unit).unwrap();
        assert!((restored.0.0 - 0.1).abs() < 1e-9);
        assert!((restored.0.1 - 0.2).abs() < 1e-9);
        assert!((restored.1.0 - 0.3).abs() < 1e-9);
        assert!((restored.1.1 - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_product_space_clamp_preserves_valid() {
        let space = ProductSpace {
            a: ContinuousSpace::new(0.0, 1.0),
            b: DiscreteSpace::new(1, 5),
        };
        let clamped = space.clamp(&(0.5, 3));
        assert_eq!(clamped.0, 0.5);
        assert_eq!(clamped.1, 3);
    }

    #[test]
    fn test_product_space_from_unit_cube_exact_length_required() {
        let space = ProductSpace {
            a: ContinuousSpace::new(0.0, 1.0),
            b: ContinuousSpace::new(0.0, 1.0),
        };
        assert!(space.from_unit_cube(&[0.5, 0.5]).is_some());
        assert!(space.from_unit_cube(&[0.5]).is_none());
    }

    // ======================================================================
    // BranchingSpace
    // ======================================================================

    #[test]
    fn test_branching_space_contains() {
        let space = BranchingSpace {
            left: ContinuousSpace::new(0.0, 1.0),
            right: DiscreteSpace::new(1, 10),
        };
        assert!(space.contains(&EitherDomain::Left(0.5)));
        assert!(!space.contains(&EitherDomain::Left(1.5)));
        assert!(space.contains(&EitherDomain::Right(5)));
        assert!(!space.contains(&EitherDomain::Right(11)));
    }

    #[test]
    fn test_branching_space_clamp() {
        let space = BranchingSpace {
            left: ContinuousSpace::new(0.0, 1.0),
            right: DiscreteSpace::new(1, 10),
        };
        assert_eq!(
            space.clamp(&EitherDomain::Left(2.0)),
            EitherDomain::Left(1.0)
        );
        assert_eq!(
            space.clamp(&EitherDomain::Right(20)),
            EitherDomain::Right(10)
        );
    }

    #[test]
    fn test_branching_space_unit_cube_roundtrip() {
        let space = BranchingSpace {
            left: ContinuousSpace::new(0.0, 1.0),
            right: ContinuousSpace::new(0.0, 100.0),
        };
        assert_eq!(space.dimensionality(), 2);

        // Left roundtrip
        let left_point = EitherDomain::Left(0.7);
        let unit = space.to_unit_cube(&left_point);
        assert_eq!(unit.len(), 2);
        assert!(unit[0] < 0.5);
        let recon = space.from_unit_cube(&unit).unwrap();
        match recon {
            EitherDomain::Left(v) => assert!((v - 0.7).abs() < 1e-9),
            _ => panic!("Expected Left branch"),
        }

        // Right roundtrip
        let right_point = EitherDomain::Right(42.0);
        let unit = space.to_unit_cube(&right_point);
        assert!(unit[0] >= 0.5);
        let recon = space.from_unit_cube(&unit).unwrap();
        match recon {
            EitherDomain::Right(v) => assert!((v - 42.0).abs() < 1e-6),
            _ => panic!("Expected Right branch"),
        }
    }

    #[test]
    fn test_branching_space_from_unit_cube_empty() {
        let space = BranchingSpace {
            left: ContinuousSpace::new(0.0, 1.0),
            right: ContinuousSpace::new(0.0, 1.0),
        };
        assert!(space.from_unit_cube(&[]).is_none());
    }

    #[test]
    fn test_branching_space_asymmetric_dimensionality() {
        let space = BranchingSpace {
            left: ContinuousSpace::new(0.0, 1.0),
            right: ProductSpace {
                a: ContinuousSpace::new(0.0, 1.0),
                b: ContinuousSpace::new(0.0, 1.0),
            },
        };
        assert_eq!(space.dimensionality(), 3);
    }

    #[test]
    fn test_branching_space_selector_boundary_exact_half() {
        let space = BranchingSpace {
            left: ContinuousSpace::new(0.0, 1.0),
            right: ContinuousSpace::new(0.0, 1.0),
        };

        // Selector exactly 0.5 should go to Right branch (>= 0.5)
        let result = space.from_unit_cube(&[0.5, 0.7]).unwrap();
        match result {
            EitherDomain::Right(v) => assert!((v - 0.7).abs() < 1e-9),
            EitherDomain::Left(_) => panic!("Selector=0.5 should route to Right branch"),
        }

        // Selector just below 0.5 should go to Left branch
        let result = space.from_unit_cube(&[0.49999, 0.7]).unwrap();
        match result {
            EitherDomain::Left(v) => assert!((v - 0.7).abs() < 1e-9),
            EitherDomain::Right(_) => panic!("Selector<0.5 should route to Left branch"),
        }
    }

    #[test]
    fn test_branching_space_asymmetric_roundtrip() {
        // Left has dim 1, Right has dim 3
        let space = BranchingSpace {
            left: ContinuousSpace::new(0.0, 10.0),
            right: ProductSpace {
                a: ProductSpace {
                    a: ContinuousSpace::new(0.0, 1.0),
                    b: ContinuousSpace::new(0.0, 1.0),
                },
                b: ContinuousSpace::new(0.0, 1.0),
            },
        };
        assert_eq!(space.dimensionality(), 4);

        let left_point = EitherDomain::Left(7.5);
        let unit = space.to_unit_cube(&left_point);
        assert_eq!(unit.len(), 4);
        assert!(unit[0] < 0.5);
        let restored = space.from_unit_cube(&unit).unwrap();
        match restored {
            EitherDomain::Left(v) => assert!((v - 7.5).abs() < 1e-9),
            _ => panic!("Expected Left branch"),
        }

        let right_point = EitherDomain::Right(((0.2, 0.4), 0.6));
        let unit = space.to_unit_cube(&right_point);
        assert!(unit[0] >= 0.5);
        let restored = space.from_unit_cube(&unit).unwrap();
        match restored {
            EitherDomain::Right(((a, b), c)) => {
                assert!((a - 0.2).abs() < 1e-9);
                assert!((b - 0.4).abs() < 1e-9);
                assert!((c - 0.6).abs() < 1e-9);
            }
            _ => panic!("Expected Right branch"),
        }
    }

    #[test]
    fn test_branching_space_insufficient_dims() {
        let space = BranchingSpace {
            left: ContinuousSpace::new(0.0, 1.0),
            right: ProductSpace {
                a: ContinuousSpace::new(0.0, 1.0),
                b: ContinuousSpace::new(0.0, 1.0),
            },
        };
        assert!(space.from_unit_cube(&[0.75, 0.5]).is_none());
        assert!(space.from_unit_cube(&[]).is_none());
    }
}
