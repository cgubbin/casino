//! Methods to generate random samples from their known distributions.

use ndarray::{
    iter::{Lanes, LanesMut},
    Array2, ArrayView1, ArrayView2, Dim, ScalarOperand,
};
use ndarray_linalg::{error::LinalgError, Cholesky, Lapack, Scalar, UPLO};
use ndarray_rand::{
    rand::Rng,
    rand_distr::{Distribution, StandardNormal},
    RandomExt,
};

pub enum Uncertainties<'a, E> {
    // Diagional uncertainties representing the variance of each input variable
    Diagonal(ArrayView1<'a, E>),
    // Full uncertainties, whose diagonal elements represent the input variances and whose
    // off-diagonal elements contain co-variance correlations
    Full(ArrayView2<'a, E>),
}

pub struct Input<'a, E> {
    pub(crate) expectation_values: ArrayView1<'a, E>,
    pub(crate) uncertainties: Uncertainties<'a, E>,
}

pub struct Samples<E> {
    samples: Array2<E>,
}

impl<E> Samples<E>
where
    StandardNormal: Distribution<E>,
{
    fn new<R: Rng>(num_samples: usize, num_inputs: usize, rng: &mut R) -> Self {
        Self {
            samples: Array2::random_using((num_samples, num_inputs), StandardNormal, rng),
        }
    }

    fn rows_mut(&mut self) -> LanesMut<'_, E, Dim<[usize; 1]>> {
        self.samples.rows_mut()
    }

    pub(crate) fn samples(&self) -> Lanes<'_, E, Dim<[usize; 1]>> {
        self.samples.rows()
    }
}

impl<'a, E> Input<'a, E>
where
    E: Lapack + Scalar<Real = E> + ScalarOperand,
    StandardNormal: Distribution<E>,
{
    pub(crate) fn generate_samples<R: Rng>(
        &self,
        number_of_samples: usize,
        rng: &mut R,
    ) -> Result<Samples<E>, LinalgError> {
        let samples = match self.uncertainties {
            Uncertainties::Diagonal(variance) => {
                self.generate_from_variance(variance, number_of_samples, rng)
            }
            Uncertainties::Full(covariance) => {
                self.generate_from_covariance(covariance, number_of_samples, rng)?
            }
        };
        Ok(samples)
    }

    fn num_inputs(&self) -> usize {
        self.expectation_values.len()
    }

    fn generate_from_variance<R: Rng>(
        &self,
        variance: ArrayView1<'a, E>,
        number_of_samples: usize,
        rng: &mut R,
    ) -> Samples<E> {
        let mut samples = Samples::new(number_of_samples, self.num_inputs(), rng);
        let std_dev = variance.mapv(ndarray_linalg::Scalar::sqrt);

        for mut row in samples.rows_mut() {
            row.assign(&(self.expectation_values.to_owned() + std_dev.clone() * row.to_owned()));
        }

        samples
    }

    fn generate_from_covariance<R: Rng>(
        &self,
        covariance: ArrayView2<'a, E>,
        number_of_samples: usize,
        rng: &mut R,
    ) -> Result<Samples<E>, LinalgError> {
        let mut samples = Samples::new(number_of_samples, self.num_inputs(), rng);
        let chol = covariance.cholesky(UPLO::Lower)?;

        for mut row in samples.rows_mut() {
            row.assign(&(self.expectation_values.to_owned() + chol.dot(&row)));
        }

        Ok(samples)
    }
}
