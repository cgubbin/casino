//! Core Monte Carlo algorithm.
use crate::{
    input::{Input, Samples, Uncertainties},
    stats::{EpochOutput, MonteCarloOutput, SummaryStatistics},
    Model, Result,
};
use ndarray::{Array1, ScalarOperand};
use ndarray_linalg::{eig::EigVals, error::LinalgError, Lapack, Scalar};
use ndarray_rand::{
    rand::Rng,
    rand_distr::{Distribution, StandardNormal},
};
use num_traits::{Float, ToPrimitive};
use tracing::{event, Level};

pub struct Config<E> {
    pub(crate) num_significant_digits: u8,
    pub(crate) required_coverage_probability: E,
}

impl<E: Float> Default for Config<E> {
    fn default() -> Self {
        Self {
            num_significant_digits: 3,
            required_coverage_probability: E::from(0.95).unwrap(),
        }
    }
}

pub struct Problem<'a, E, R, F> {
    pub(crate) inputs: Input<'a, E>,
    /// Max(1 / (100 - p), 10^4)
    pub(crate) number_of_trials: usize,
    pub(crate) config: Config<E>,
    pub(crate) rng: &'a mut R,
    pub(crate) model: F,
}

impl<'a, E, R: Rng, F> Problem<'a, E, R, F>
where
    E: Float + Lapack + PartialOrd + Scalar<Real = E> + ScalarOperand + ToPrimitive,
    StandardNormal: Distribution<E>,
    F: Model<E>,
{
    #[tracing::instrument(skip_all)]
    /// The core algorithm.
    pub fn run(&mut self) -> Result<SummaryStatistics<E>> {
        self.security_checks()?;
        let mut h = 1;

        let mut collated = MonteCarloOutput::new();

        while !collated.is_converged(i32::from(self.config.num_significant_digits)) || h < 4 {
            // We need to run at least two trials to have access to summary statistics
            event!(Level::INFO, iter = h);
            let outputs = self.epoch()?;
            collated.add_vals(&outputs)?;
            h += 1;
        }

        collated.summary_statistics()
    }

    fn epoch(&mut self) -> Result<EpochOutput<E>> {
        let input_samples = self.generate_samples()?;
        let mut outputs = EpochOutput::new();

        for sample in input_samples.samples() {
            let output = self.apply(sample.to_owned())?;
            outputs.add_row(output.view())?;
        }

        Ok(outputs)
    }

    fn generate_samples(&mut self) -> ::std::result::Result<Samples<E>, LinalgError> {
        self.inputs
            .generate_samples(self.number_of_trials, &mut self.rng)
    }

    fn apply(&mut self, input: Array1<E>) -> Result<Array1<E>> {
        self.model.apply(input)
    }

    fn security_checks(&self) -> Result<()> {
        // Check the covariance matrix is positive definite
        // Required according to the spectral theorem for matrices
        if let Uncertainties::Full(covariance) = self.inputs.uncertainties {
            let covariance_eigs = covariance.eigvals()?;
            if covariance_eigs
                .into_iter()
                .any(|eigenvalue| eigenvalue.re() < E::zero())
            {
                return Err("covariance matrix provided is not positive definite".into());
            }
        }
        Ok(())
    }
}

fn base_10_decompose<E: Float + ToPrimitive>(value: E) -> (E, i32, E) {
    let sign = value.signum();
    let value = value / sign;
    let base_10_exponent = value
        .log10()
        .floor()
        .to_i32()
        .expect("failed to fit exponent in i32");

    let mantissa = value / E::from(10f64).unwrap().powi(base_10_exponent);

    (mantissa, base_10_exponent, sign)
}

pub fn compute_tolerance<E: Float + ToPrimitive>(value: E, num_significant_digits: i32) -> E {
    let (_, base_10_exponent, _) = base_10_decompose(value);
    let l = base_10_exponent - num_significant_digits + 1;
    E::from(10f64).unwrap().powi(l) / (E::one() + E::one())
}

#[cfg(test)]
mod test {
    use super::{base_10_decompose, compute_tolerance};
    use ndarray_rand::rand::{Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    #[test]
    fn base_10_decomposition_is_correct() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);

        let exponent: i32 = i32::from(rng.gen::<i8>());
        let mantissa: f64 = rng.gen_range(-10f64..10f64);

        let number = mantissa * 10f64.powi(exponent);
        let (computed_mantissa, computed_exponent, computed_sign) = base_10_decompose(number);

        approx::assert_relative_eq!(mantissa, computed_mantissa);
        approx::assert_relative_eq!(mantissa.signum(), computed_sign);
        assert_eq!(exponent, computed_exponent);
    }

    #[test]
    fn tolerance_is_computed_correctly() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);

        let exponent: i32 = i32::from(rng.gen::<i8>());
        let mantissa: f64 = rng.gen_range(-10f64..10f64);

        let num_significant_digits = 2;

        let number = mantissa * 10f64.powi(exponent);
        let computed = compute_tolerance(number, num_significant_digits);

        let expected = 0.5 * 10f64.powi(exponent - num_significant_digits + 1);

        approx::assert_relative_eq!(expected, computed);
    }
}
