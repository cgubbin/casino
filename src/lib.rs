//! Casino is a library to carry out Monte-Carlo propagation of distributions.
//!
//! Users need to implement the [`Model`] trait. This has a single method which takes
//! an array of values, which are the inputs to the measurement model, and converts
//! them to an array of outputs.
//!
//! Calculations are run using using the [`crate::Problem`] type. The following creates a simple
//! measurement model in which the outputs relate to the inputs via a quadratic equation:
//!
//! When the algorithm is run, the apply method is called repeatedly until convergence is achieved
//! in the distributional properties of the output variables.
//!
#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

use ndarray::{Array1, LinalgScalar, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num_traits::{Float, FromPrimitive};

mod controller;
mod input;
mod observer;
mod operator;
mod stats;

use controller::{AdaptiveController, McController};
use input::{CompiledInput, InputModel, InputSpec, SampleError, SamplingStrategy};
use observer::{McObserver, StatsObserver};
use operator::Operator;
use stats::{RunningStats, SampleBatch, SummaryStatistics};

pub use operator::{EvalResult, OperatorError};

#[derive(thiserror::Error, Debug)]
pub enum CasinoError {
    #[error("error in application of user defined operator: {0:?}")]
    Operator(#[from] operator::OperatorError),
    #[error("error in sampler: {0:?}")]
    Sampler(#[from] input::SampleError),
}

pub struct MonteCarlo;

impl MonteCarlo {
    pub fn run<'a, E, O>(
        input: InputSpec<'a, E>,
        operator: O,
        seed: u64,
        n_per_batch: usize,
        min_samples: usize,
        max_samples: usize,
        rel_tol: E,
    ) -> Result<SummaryStatistics<E>, CasinoError>
    where
        E: Float + Scalar + From<f64> + Lapack + ScalarOperand,
        StandardNormal: Distribution<E>,
        O: Operator<E>,
    {
        let mut compiled = input.compile_gaussian(seed)?;
        let mut observer = StatsObserver::new(operator.dim_out());

        let controller = AdaptiveController::new(min_samples, max_samples, rel_tol);

        let mut engine = MonteCarloEngine {
            compiled,
            observer,
            controller,
            operator,
        };

        engine.run(n_per_batch)
    }
}

pub struct MonteCarloEngine<E, S, M, O, W, C>
where
    S: SamplingStrategy<E>,
    M: InputModel<E, Space = S::Space>,
    O: Operator<E>,
{
    compiled: CompiledInput<E, S, M>,
    operator: O,
    observer: W,
    controller: C,
}

impl<E, S, M, O, W, C> MonteCarloEngine<E, S, M, O, W, C>
where
    S: SamplingStrategy<E>,
    M: InputModel<E, Space = S::Space>,
    O: Operator<E>,
    W: McObserver<E, State = RunningStats<E>>,
    C: McController<E>,
{
    // pub fn new(compiled: CompiledInput<E, S, M>, operator: O) -> Self {
    //     Self { compiled, operator }
    // }

    pub fn run(&mut self, n_per_batch: usize) -> Result<W::Output, CasinoError>
    where
        E: Float + LinalgScalar + ScalarOperand + FromPrimitive,
    {
        let mut state = self.compiled.init_state();
        loop {
            // ------------------------------------------------------------
            // 1. sample latent space Z
            // ------------------------------------------------------------
            let z = self.compiled.sample(&mut state, n_per_batch);

            // ------------------------------------------------------------
            // 2. affine transform → physical input space
            // ------------------------------------------------------------
            let samples = self.compiled.model.apply(z);

            let batch = SampleBatch { values: samples };

            // ------------------------------------------------------------
            // 3. evaluate operator
            // ------------------------------------------------------------
            let eval = self.operator.apply_batch(batch.values.view())?;

            // ------------------------------------------------------------
            // 4. accumulate statistics (validity-aware)
            // ------------------------------------------------------------
            self.observer.update_batch(eval);

            if self.controller.should_stop(&self.observer.state()) {
                break;
            }
        }

        Ok(self.observer.finalize())
    }
}
//     pub fn run_streaming(&self, n: usize, chunk: usize) -> Result<SummaryStatistics<E>, CasinoError>
//     where
//         E: Float + LinalgScalar + ScalarOperand + FromPrimitive,
//     {
//         let mut stats = RunningStats::new(self.operator.dim_out());

//         let dim = self.compiled.model.dim();

//         let mut remaining = n;

//         let mut state = self.compiled.init_state();

//         while remaining > 0 {
//             let m = remaining.min(chunk);
//             remaining -= m;

//             let z = self.compiled.sample(&mut state, m);

//             let samples = self.compiled.model.apply(z);

//             let eval = self.operator.apply_batch(samples.view())?;

//             let (value, valid) = eval.split();
//             stats.update_batch(value.view(), valid.view());
//         }

//         Ok(SummaryStatistics {
//             mean: stats.finalize_mean(),
//             covariance: stats.covariance(),
//         })
//     }
// }
//
//
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Ix1;

    #[test]
    fn monte_carlo_is_reproducible() {
        let means = ndarray::array![0.0, 1.0];
        let marginal_scale = ndarray::array![1.0, 2.0];
        let input = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        };

        struct IdentityOp;

        impl Operator<f64> for IdentityOp {
            fn dim_in(&self) -> usize {
                2
            }
            fn dim_out(&self) -> usize {
                2
            }

            fn apply(
                &self,
                x: ndarray::ArrayView1<f64>,
            ) -> Result<EvalResult<f64, Ix1>, OperatorError> {
                EvalResult::try_from_parts(x.to_owned(), ndarray::array![true, true])
            }
        }

        let seed = 12345;

        let r1 = MonteCarlo::run(input.clone(), IdentityOp, seed, 128, 512, 512, 1e-3).unwrap();
        let r2 = MonteCarlo::run(input, IdentityOp, seed, 128, 512, 512, 1e-3).unwrap();

        approx::assert_abs_diff_eq!(r1.mean, r2.mean, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(r1.covariance, r2.covariance, epsilon = 1e-12);
    }

    #[test]
    fn gaussian_mean_is_correct() {
        let means = ndarray::array![1.0, 2.0];
        let marginal_scale = ndarray::array![0.01, 0.01];
        let input = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        };

        struct Identity;

        impl Operator<f64> for Identity {
            fn dim_in(&self) -> usize {
                2
            }
            fn dim_out(&self) -> usize {
                2
            }

            fn apply(
                &self,
                x: ndarray::ArrayView1<f64>,
            ) -> Result<EvalResult<f64, Ix1>, OperatorError> {
                EvalResult::try_from_parts(x.to_owned(), ndarray::array![true, true])
            }
        }

        let result = MonteCarlo::run(input, Identity, 42, 256, 2000, 2000, 1e-2).unwrap();

        // expectation of identity should match input mean
        approx::assert_abs_diff_eq!(result.mean[0], 1.0, epsilon = 1e-1);
        approx::assert_abs_diff_eq!(result.mean[1], 2.0, epsilon = 1e-1);
    }

    #[test]
    fn independent_inputs_have_diagonal_covariance() {
        let means = ndarray::array![0.0, 0.0, 0.0];
        let marginal_scale = ndarray::array![1.0, 2.0, 3.0];
        let input = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        };

        struct Identity;

        impl Operator<f64> for Identity {
            fn dim_in(&self) -> usize {
                3
            }
            fn dim_out(&self) -> usize {
                3
            }

            fn apply(
                &self,
                x: ndarray::ArrayView1<f64>,
            ) -> Result<EvalResult<f64, Ix1>, OperatorError> {
                EvalResult::try_from_parts(x.to_owned(), ndarray::array![true, true, true])
            }
        }

        let result = MonteCarlo::run(input, Identity, 7, 500, 100000, 200000, 1e-5).unwrap();

        dbg!(&result.covariance);
        // off-diagonals should be ~0
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(result.covariance[[i, j]].abs() < 1e-1);
                }
            }
        }
    }
}
