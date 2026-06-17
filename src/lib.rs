//! # Montecore
//!
//! A Rust library for uncertainty propagation using Monte Carlo simulation.
//!
//! Montecore provides:
//!
//! * Gaussian Monte Carlo sampling
//! * Latin Hypercube Sampling (LHS)
//! * Adaptive convergence control
//! * Correlated and independent input models
//! * Streaming statistics (mean, variance, covariance)
//! * Deterministic and reproducible execution
//! * Batch-oriented execution suitable for future parallelisation
//!
//! ---
//!
//! # Philosophy
//!
//! Montecore is designed for scientific and engineering uncertainty analysis.
//!
//! The library separates uncertainty propagation into three independent concepts:
//!
//! 1. **Input model** — describes uncertain inputs.
//! 2. **Sampling strategy** — describes how the input space is explored.
//! 3. **Operator** — the physical or numerical model being evaluated.
//!
//! This separation allows the same operator to be evaluated using different sampling methods without modifying user code.
//!
//! ---
//!
//! # Installation
//!
//! ```toml
//! [dependencies]
//! montecore = "0.2"
//! ```
//!
//! ---
//!
//! # Quick Start
//!
//! ```
//! use montecore::*;
//! use ndarray::{arr1, ArrayView1, Ix1};
//!
//! struct BeamDeflection;
//!
//! impl Operator<f64> for BeamDeflection {
//!     fn dim_in(&self) -> usize {
//!         2
//!     }
//!
//!     fn dim_out(&self) -> usize {
//!         1
//!     }
//!
//!     fn apply(
//!         &self,
//!         x: ArrayView1<'_, f64>,
//!     ) -> Result<EvalResult<f64, Ix1>, OperatorError> {
//!
//!         let force = x[0];
//!         let stiffness = x[1];
//!
//!         let deflection = force / stiffness;
//!
//!         EvalResult::try_from_parts(
//!             arr1(&[deflection]),
//!             arr1(&[true]),
//!         )
//!     }
//! }
//!
//! let means = arr1(&[100.0, 10.0]);
//! let stddev = arr1(&[5.0, 1.0]);
//!
//! let result = MonteCarlo::run(
//!     InputSpec::Independent {
//!         means: means.view(),
//!         marginal_scale: stddev.view(),
//!     },
//!     BeamDeflection,
//!     SamplingMethod::Gaussian,
//!     MonteCarloOptions {
//!         seed: 42,
//!         batch_size: 1024,
//!         min_samples: 10_000,
//!         max_samples: 1_000_000,
//!         rel_tol: 1e-3,
//!     },
//! );
//! ```
//!
//! The returned result contains estimated output statistics:
//!
//! ```ignore
//! result.statistics.mean
//! result.statistics.covariance
//! ```
//!
//! ---
//!
//! # Input Models
//!
//! Montecore supports both independent and correlated inputs.
//!
//! ## Independent Inputs
//!
//! ```ignore
//! InputSpec::Independent {
//!     means: means.view(),
//!     marginal_scale: marginal_scale.view(),
//! };
//! ```
//!
//! Each variable is sampled independently.
//!
//! ---
//!
//! ## Correlated Inputs
//!
//! ```ignore
//! InputSpec::Correlated {
//!     means: means.view(),
//!     covariance: covariance.view(),
//! }
//! ```
//!
//! The covariance matrix describes correlations between inputs.
//!
//! For Gaussian sampling, Montecore computes a Cholesky factorisation internally and generates correlated samples automatically.
//!
//! ---
//!
//! # Sampling Methods
//!
//! Sampling strategy determines how the input space is explored.
//!
//! ## Gaussian Sampling
//!
//! ```ignore
//! SamplingMethod::Gaussian
//! ```
//!
//! Traditional Monte Carlo sampling using independent Gaussian random variables.
//!
//! Use when:
//!
//! * Input uncertainty is naturally Gaussian
//! * Statistical independence is important
//! * Direct comparison with analytical Gaussian uncertainty propagation is desired
//!
//! ---
//!
//! ## Latin Hypercube Sampling
//!
//! ```ignore
//! SamplingMethod::LatinHypercube
//! ```
//!
//! Stratified sampling method
//!
//! Use when:
//!
//! * Model evaluations are expensive
//! * Inputs are independent
//! * Improved convergence is desired
//!
//! ---
//!
//! # Choosing a Sampling Method
//!
//! | Method          | Characteristics                         |
//! | --------------- | --------------------------------------- |
//! | Gaussian        | Random, unbiased, familiar              |
//! | Latin Hypercube | Stratified, faster convergence          |
//!
//! For most engineering uncertainty propagation problems:
//!
//! ```ignore
//! SamplingMethod::LatinHypercube
//! ```
//!
//! is a good default.
//!
//! ---
//!
//! # The Operator Trait
//!
//! User code implements the measurement model through the `Operator` trait.
//!
//! ```ignore
//! pub trait Operator<E> {
//!     fn dim_in(&self) -> usize;
//!
//!     fn dim_out(&self) -> usize;
//!
//!     fn apply(&self, inputs: ArrayView1<'_, E>) -> Result<EvalResult<E, Ix1>, OperatorError>;
//! }
//! ```
//!
//! ---
//!
//! # Validity Masks
//!
//! Montecore uses validity masks rather than exceptions to handle numerical failures.
//!
//! Each output value has a corresponding validity flag.
//!
//! ```ignore
//! EvalResult::try_from_parts(
//!     value,
//!     valid,
//! )
//! ```
//!
//! where:
//!
//! ```text
//! valid[i] == true
//! ```
//!
//! means the output is statistically valid.
//!
//! and
//!
//! ```text
//! valid[i] == false
//! ```
//!
//! means the output should be excluded from all downstream statistics.
//!
//! ---
//!
//! ## Example
//!
//! ```
//! use montecore::*;
//! use ndarray::{ArrayView1, Ix1, arr1};
//!
//! struct Reciprocal;
//!
//! impl Operator<f64> for Reciprocal {
//!     fn dim_in(&self) -> usize {
//!         1
//!     }
//!
//!     fn dim_out(&self) -> usize {
//!         1
//!     }
//!
//!     fn apply(&self, x: ArrayView1<'_, f64>) -> Result<EvalResult<f64, Ix1>, OperatorError> {
//!         if x[0] == 0.0 {
//!             return EvalResult::try_from_parts(arr1(&[0.0]), arr1(&[false]));
//!         }
//!
//!         EvalResult::try_from_parts(arr1(&[1.0 / x[0]]), arr1(&[true]))
//!     }
//! }
//! ```
//!
//! This allows Monte Carlo simulations to continue even when individual evaluations fail.
//!
//! ---
//!
//! # Adaptive Convergence
//!
//! Montecore automatically monitors convergence during simulation.
//!
//! The engine terminates when either:
//!
//! ```text
//! maximum relative uncertainty < rel_tol
//! ```
//!
//! or
//!
//! ```text
//! sample count >= max_samples
//! ```
//!
//! while also enforcing:
//!
//! ```text
//! sample count >= min_samples
//! ```
//!
//! before convergence checks begin.
//!
//! This allows simulations to stop early once sufficient statistical precision has been achieved.
//!
//! ---
//!
//! # Reproducibility
//!
//! All sampling methods are deterministic.
//!
//! ```ignore
//! MonteCarloOptions {
//!     seed: 42,
//!     ..
//! }
//! ```
//!
//! Using the same:
//!
//! * input model
//! * operator
//! * sampling method
//! * seed
//!
//! will always produce identical results.
//!
//! ---
//!
//! # Statistics
//!
//! Montecore computes statistics incrementally using streaming estimators.
//!
//! The following quantities are available:
//!
//! ```ignore
//! SummaryStatistics {
//!     expectation,
//!     covariance,
//! }
//! ```
//!
//! where:
//!
//! ```text
//! expectation
//! ```
//!
//! is the estimated mean output vector and
//!
//! ```text
//! covariance
//! ```
//!
//! is the estimated covariance matrix.
//!
//! Streaming accumulation avoids storing all Monte Carlo samples in memory.
//!
//! ---
//!
//! # Numerical Robustness
//!
//! Montecore uses:
//!
//! * Welford-style online moment accumulation
//! * Chan-style batch merging
//! * Validity-aware covariance estimation
//!
//! to provide stable statistics for large simulations.
//!
//! The library never stores the full Monte Carlo history.
//!
//! Memory usage scales with output dimension rather than sample count.
//!
//! ---

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
pub enum MontecoreError {
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
    ) -> Result<MonteCarloResult<E>, MontecoreError>
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
    pub fn run(&mut self, n_per_batch: usize) -> Result<MonteCarloResult<E>, MontecoreError>
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
