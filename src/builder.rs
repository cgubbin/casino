//! Builder methods for creation of problems.
//!
//! A simple model can be created as follows:
//! ```
//! use casino::{Builder, Config, Model};
//! use ndarray::Array1;
//! use ndarray_rand::rand::{Rng, SeedableRng};
//! use rand_isaac::Isaac64Rng;
//!
//! struct Example {
//!     a: f64,
//!     b: f64,
//! }
//!
//! impl Model<f64> for Example {
//!     fn apply(&self, inputs: Array1<f64>) -> Result<Array1<f64>, Box<dyn ::std::error::Error>> {
//!         Ok(inputs.mapv(|x| self.a + x.powi(2) * self.b))
//!     }
//! }
//!
//! let model = Example { a: 1.0, b: 2.0 };
//!
//! let state = 40;
//! let mut rng = Isaac64Rng::seed_from_u64(state);
//!
//! let expectations = Array1::linspace(0., 10., 11);
//! let variances = expectations.iter().map(|mean| mean / 100.0).collect::<Array1<_>>();
//!
//! let config = Config {
//!     num_significant_digits: 3,
//!     required_coverage_probability: 0.95,
//! };
//!
//! let mut problem = Builder::new(&mut rng, model)
//!                 .with_config(config)
//!                 .with_input_expectations(expectations.view())
//!                 .with_input_variances(variances.view())
//!                 .build();
//! ```

use std::marker::PhantomData;

use ndarray::{ArrayView1, ArrayView2};
use ndarray_rand::rand::Rng;
use num_traits::{Float, ToPrimitive};

use crate::{
    core::{Config, Problem},
    input::{Input, Uncertainties},
};

#[derive(Debug, Clone)]
pub enum BuilderError {
    InvalidProbability(f64),
    InvalidArrayShape {
        expected: usize,
        found: usize,
        kind: &'static str,
    },
    ConversionFailure(&'static str),
}

pub struct Set {}
pub struct Unset {}

/// Builder struct for creating and configuring problems
pub struct BuilderBase<'a, E, M> {
    model: M,
    config: Option<Config<E>>,
    expectation_values: ArrayView1<'a, E>,
}

/// Builder struct for creating and configuring problems with variance
pub struct BuilderVariance<'a, E, M> {
    base: BuilderBase<'a, E, M>,
    variances: ArrayView1<'a, E>,
}

/// Builder struct for creating and configuring problems with covariance
pub struct BuilderCovariance<'a, E, M> {
    base: BuilderBase<'a, E, M>,
    covariances: ArrayView2<'a, E>,
}

fn compute_trials<E: Float + ToPrimitive>(p: E) -> Result<usize, BuilderError> {
    if p < E::zero() || p < E::one() {
        return Result::Error(BuilderError::InvalidProbability(p.to_f64().unwrap()));
    }

    let base = (E::one() / (E::one() - p))
        .to_usize()
        .unwrap_or_error(BuilderError::ConversionFailure("to usize"))?;

    Ok(10_000.max(base))
}

impl<'a, E, M> BuilderBase<'a, E, M> {
    pub fn new(model: M, expectation_values: ArrayView1<'a, E>) -> Self {
        Self {
            model,
            config: None,
            expectation_values,
        }
    }
}

impl<'a, E, M> BuilderBase<'a, E, M> {
    #[must_use]
    pub fn with_config(mut self, config: Config<E>) -> Self {
        self.config = Some(config);
        self
    }
}

impl<'a, E, M> BuilderBase<'a, E, M> {
    pub fn variances(
        self,
        v: ArrayView1<'a, E>,
    ) -> Result<BuilderVariance<'a, E, M>, BuilderError> {
        if v.shape != self.expectation_values.shape {
            return Err(BuilderError::InvalidArrayShape {
                expected: self.expectation_values.len(),
                found: v.len(),
                kind: "variances and expectation values must have equal length",
            });
        }
        Ok(BuilderVariance {
            base: self,
            variances: v,
        })
    }

    pub fn covariances(
        self,
        c: ArrayView2<'a, E>,
    ) -> Result<BuilderCovariance<'a, E, M>, BuilderError> {
        if c.shape[0] != self.expectation_values.len() {
            return Err(BuilderError::InvalidArrayShape {
                expected: self.expectation_values.len(),
                found: v.shape[0],
                kind: "covariances matrix must have the same number of rows as expectation values",
            });
        }
        if c.shape[1] != self.expectation_values.len() {
            return Err(BuilderError::InvalidArrayShape {
                expected: self.expectation_values.len(),
                found: v.shape[1],
                kind: "covariances matrix must have the same number of cols as expectation values",
            });
        }
        Ok(BuilderCovariance {
            base: self,
            covariances: c,
        })
    }
}

impl<'a, E: Float + ToPrimitive, P> BuilderVariance<'a, E, P> {
    /// Build a problem
    ///
    /// # Panics
    /// - If the required coverage probability is not an integer.
    pub fn build<R: Rng>(self, rng: R) -> Result<Problem<'a, E, R, P>, BuilderError> {
        let config = self.base.config.unwrap_or_default();

        let number_of_trials = compute_trials(config.required_coverage_probability)?;

        Ok(Problem {
            config,
            number_of_trials,
            rng,
            model: self.base.model,
            inputs: Input {
                expectation_values: self.base.expectation_values,
                uncertainties: Uncertainties::Diagonal(self.variances),
            },
        })
    }
}

impl<'a, E: Float + ToPrimitive, P> BuilderCovariance<'a, E, P> {
    pub fn build<R: Rng>(self, rng: R) -> Result<Problem<'a, E, R, P>, BuilderError> {
        let config = self.base.config.unwrap_or_default();

        let number_of_trials = compute_trials(config.required_coverage_probability)?;

        Ok(Problem {
            config,
            number_of_trials,
            rng,
            model: self.base.model,
            inputs: Input {
                expectation_values: self.base.expectation_values,
                uncertainties: Uncertainties::Full(self.covariances),
            },
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{builder::Builder, core::compute_tolerance, Model};

    use ndarray::Array1;
    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn linear_model_has_expected_properties() {
        struct TestModel {
            means: [f64; 2],
        }

        impl Model<f64> for TestModel {
            fn apply(
                &self,
                inputs: ndarray::Array1<f64>,
            ) -> std::result::Result<ndarray::Array1<f64>, Box<dyn std::error::Error>> {
                let res = inputs
                    .into_iter()
                    .map(|input| input.mul_add(self.means[1], self.means[0]))
                    .collect();

                Ok(res)
            }
        }

        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);

        let means = [rng.gen(), rng.gen()];

        let model = TestModel { means };

        let num_data_points = 5;

        let expectation_values = (0..num_data_points)
            .map(|_| rng.gen())
            .collect::<Array1<f64>>();
        let variances = expectation_values
            .iter()
            .map(|x| x / 100.0)
            .collect::<Array1<f64>>();

        let mut problem = Builder::new(&mut rng, model)
            .with_input_expectations(expectation_values.view())
            .with_input_variances(variances.view())
            .build();

        let result = problem.run().unwrap();

        let num_significant_digits = i32::from(problem.config.num_significant_digits);

        for (calc, input) in result.expectation.into_iter().zip(expectation_values) {
            let exp = input.mul_add(means[1], means[0]);
            let tolerance = compute_tolerance(exp, num_significant_digits);
            println!("{tolerance}, {calc}, {exp}, {}", (calc - exp).abs());
            assert!((calc - exp).abs() < tolerance);
        }
    }
}
