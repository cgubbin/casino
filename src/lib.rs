//! # Casino — Adaptive Monte Carlo Uncertainty Propagation
//!
//! Casino is a scientific Monte Carlo simulation framework for propagating uncertainty
//! through deterministic or stochastic models.
//!
//! It is designed for *numerical reliability, reproducibility, and composability* in
//! engineering and scientific computing.
//!
//! ---
//!
//! ## Core Concept
//!
//! Casino decomposes Monte Carlo simulation into three independent components:
//!
//! ### 1. Input models (`InputSpec`)
//! Define uncertain inputs as either:
//! - Independent distributions (mean + dispersion)
//! - Correlated Gaussian distributions (mean + covariance)
//!
//! ### 2. Sampling strategies (`SamplingStrategy`)
//! Control how the input space is explored:
//! - Standard Monte Carlo (Gaussian)
//! - Latin Hypercube Sampling (LHS)
//! - Low-discrepancy sequences (Sobol, planned)
//!
//! ### 3. Operators (`Operator`)
//! User-defined models that map inputs → outputs while explicitly handling
//! numerical failure via validity masks rather than panics or exceptions.
//!
//! ---
//!
//! ## Design Philosophy
//!
//! Casino is built around three principles:
//!
//! ### 1. No exceptions in Monte Carlo flow
//! Numerical or domain failures are encoded via validity masks.
//! Invalid samples remain in the dataset but are excluded from statistics.
//!
//! ### 2. Streaming statistics
//! No full sample storage is required. Mean, covariance, and uncertainty
//! estimates are computed incrementally using numerically stable algorithms.
//!
//! ### 3. Adaptive convergence
//! Simulations terminate automatically when statistical uncertainty drops below
//! a user-defined tolerance, or when sample limits are reached.
//!
//! ---
//!
//! ## Output
//!
//! The primary result of a simulation is:
//!
//! - `SummaryStatistics`
//!   - expectation (mean)
//!   - covariance
//!
//! - `StatisticalDiagnostics`
//!   - standard errors
//!   - validity rates
//!   - convergence metadata
//!
//! - `StopReason`
//!   - explains why the simulation terminated
//!
//! ---
//!
//! ## Example (simple Gaussian propagation)
//!
//! ```rust
//! use casino::*;
//! use ndarray::arr1;
//!
//! struct MyModel;
//!
//! impl Operator<f64> for MyModel {
//!     fn dim_in(&self) -> usize { 2 }
//!     fn dim_out(&self) -> usize { 1 }
//!
//!     fn apply(
//!         &self,
//!         x: ndarray::ArrayView1<'_, f64>,
//!     ) -> Result<EvalResult<f64, ndarray::Ix1>, OperatorError> {
//!         let y = x[0] * x[1];
//!
//!         EvalResult::try_from_parts(arr1(&[y]), arr1(&[true]))
//!     }
//! }
//! ```
//!
//! ---
//!
//! ## Why Casino exists
//!
//! Traditional Monte Carlo frameworks often:
//! - store all samples (memory heavy)
//! - mix sampling and model logic
//! - rely on exceptions for numerical failure
//! - lack adaptive stopping criteria
//!
//! Casino explicitly separates these concerns to enable:
//! - large-scale simulations
//! - reproducible scientific workflows
//! - safe handling of numerical instability
//! - interchangeable sampling strategies
//!
//! ---
//!
//! ## Status
//!
//! This crate is actively evolving. Current focus areas:
//! - Sobol sampling integration
//! - parallel batch execution
//! - improved convergence diagnostics
//! - importance sampling extensions

#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

use ndarray::{LinalgScalar, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num_traits::{Float, FromPrimitive};

mod controller;
mod input;
mod observer;
mod operator;
mod result;
mod stats;

use controller::{AdaptiveController, McController, StopDecision};
use input::{CompiledInput, InputModel, SamplingStrategy};
use observer::{McObserver, StatsObserver};
use stats::{FinalStatistics, RunningStats, SampleBatch};

pub use controller::StopReason;
pub use input::{InputSpec, SamplingMethod};
pub use operator::{EvalResult, Operator, OperatorError};
pub use result::MonteCarloResult;

#[derive(thiserror::Error, Debug)]
pub enum CasinoError {
    #[error("error in application of user defined operator: {0:?}")]
    Operator(#[from] operator::OperatorError),
    #[error("error in sampler: {0:?}")]
    Sampler(#[from] input::SampleError),
}

pub struct MonteCarlo;

#[derive(Clone)]
pub struct MonteCarloOptions<E> {
    pub seed: u64,
    pub batch_size: usize,
    pub min_samples: usize,
    pub max_samples: usize,
    pub rel_tol: E,
}

impl MonteCarlo {
    pub fn run<E, O>(
        input: InputSpec<'_, E>,
        operator: O,
        sampling: SamplingMethod,
        options: MonteCarloOptions<E>,
    ) -> Result<MonteCarloResult<E>, CasinoError>
    where
        E: Float + Scalar + From<f64> + Lapack + ScalarOperand,
        StandardNormal: Distribution<E>,
        O: Operator<E>,
    {
        let observer = StatsObserver::new(operator.dim_out());

        let controller =
            AdaptiveController::new(options.min_samples, options.max_samples, options.rel_tol);

        match sampling {
            SamplingMethod::Gaussian => {
                let compiled = input.compile_gaussian(options.seed)?;
                let mut engine = MonteCarloEngine {
                    compiled,
                    operator,
                    observer,
                    controller,
                };
                engine.run(options.batch_size)
            }
            SamplingMethod::LatinHypercube => {
                let compiled = input.compile_lhs(options.seed)?;
                let mut engine = MonteCarloEngine {
                    compiled,
                    operator,
                    observer,
                    controller,
                };

                engine.run(options.batch_size)
            }
        }
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
    W: McObserver<E, State = RunningStats<E>, Output = FinalStatistics<E>>,
    C: McController<E>,
{
    pub fn run(&mut self, n_per_batch: usize) -> Result<MonteCarloResult<E>, CasinoError>
    where
        E: Float + LinalgScalar + ScalarOperand + FromPrimitive,
    {
        let mut state = self.compiled.init_state();

        let mut stop_reason: Option<StopReason> = None;

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

            if let StopDecision::Stop { reason } =
                self.controller.should_stop(self.observer.state())
            {
                stop_reason.replace(reason);
                break;
            }
        }

        let total_samples = self.observer.state().count();
        let final_statistics = self.observer.finalize();

        Ok(MonteCarloResult {
            statistics: final_statistics.summary,
            diagnostics: final_statistics.diagnostics,
            total_samples,
            stop_reason: stop_reason.expect("can't break the loop without filling the option"),
        })
    }
}

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

        let options = MonteCarloOptions {
            seed: seed,
            batch_size: 128,
            min_samples: 512,
            max_samples: 512,
            rel_tol: 1e-3,
        };

        let r1 = MonteCarlo::run(
            input.clone(),
            IdentityOp,
            SamplingMethod::Gaussian,
            options.clone(),
        )
        .unwrap();
        let r2 = MonteCarlo::run(input, IdentityOp, SamplingMethod::Gaussian, options).unwrap();

        approx::assert_abs_diff_eq!(r1.statistics.mean, r2.statistics.mean, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(
            r1.statistics.covariance,
            r2.statistics.covariance,
            epsilon = 1e-12
        );
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

        let options = MonteCarloOptions {
            seed: 42,
            batch_size: 256,
            min_samples: 2000,
            max_samples: 4000,
            rel_tol: 1e-2,
        };

        let result = MonteCarlo::run(input, Identity, SamplingMethod::Gaussian, options).unwrap();

        // expectation of identity should match input mean
        approx::assert_abs_diff_eq!(result.statistics.mean[0], 1.0, epsilon = 1e-1);
        approx::assert_abs_diff_eq!(result.statistics.mean[1], 2.0, epsilon = 1e-1);
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

        let options = MonteCarloOptions {
            seed: 7,
            batch_size: 256,
            min_samples: 100_000,
            max_samples: 200_000,
            rel_tol: 1e-5,
        };

        let result = MonteCarlo::run(input, Identity, SamplingMethod::Gaussian, options).unwrap();

        // off-diagonals should be ~0
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(result.statistics.covariance[[i, j]].abs() < 1e-1);
                }
            }
        }
    }
}
