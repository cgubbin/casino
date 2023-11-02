//! Casino is a library to carry out Monte-Carlo propagation of distributions.
//!
//! Users need to implement the [`Model`] trait. This has a single method which takes
//! an array of values, which are the inputs to the measurement model, and converts
//! them to an array of outputs.
//!
//! Calculations are run using using the [`crate::Problem`] type. The following creates a simple
//! measurement model in which the outputs relate to the inputs via a quadratic equation:
//! ```
//! use casino::{Builder, Model};
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
//! let mut problem = Builder::new(&mut rng, model)
//!                 .with_input_expectations(expectations.view())
//!                 .with_input_variances(variances.view())
//!                 .build();
//! ```
//!
//! When the algorithm is run, the apply method is called repeatedly until convergence is achieved
//! in the distributional properties of the output variables.
//!
#![allow(dead_code)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

use ndarray::Array1;

mod builder;
mod core;
mod input;
mod output;
mod stats;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;

pub trait Model<E> {
    /// The measurement model, which converts inputs to outputs.
    ///
    /// # Errors
    /// - Internal error in the domain specific code.
    fn apply(
        &self,
        inputs: Array1<E>,
    ) -> ::std::result::Result<Array1<E>, Box<dyn ::std::error::Error>>;
}

pub use builder::Builder;
pub use core::Problem;
