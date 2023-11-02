//! Methods for computation of summary statistics from Monte Carlo simulation results.
use crate::{core::compute_tolerance, Result};
use ndarray::{Array1, Array2, ArrayView1, Axis, ScalarOperand, ShapeError};
use ndarray_linalg::Scalar;
use num_traits::{Float, ToPrimitive};

/// The output from a single Monte Carlo epoch
pub struct EpochOutput<E> {
    output: Array2<E>,
}

/// Collated outputs from multiple Monte Carlo epochs.
pub struct MonteCarloOutput<E> {
    full_output: Array2<E>,
    expectation: Array2<E>,
    variance: Array2<E>,
}

#[derive(Debug)]
/// Resulting properties of the measurement model outputs.
pub struct SummaryStatistics<E> {
    pub(crate) expectation: Array1<E>,
    pub(crate) covariance: Array2<E>,
}

/// Calculates the outer product of two vectors.
pub fn outer_product<E: Scalar>(
    a: &Array1<E>,
    b: &Array1<E>,
) -> ::std::result::Result<Array2<E>, ndarray::ShapeError> {
    let a: Array2<E> = a.clone().into_shape((a.len(), 1))?;
    let b: Array2<E> = b.clone().into_shape((1, b.len()))?;

    Ok(ndarray::linalg::kron(&a, &b))
}

impl<E> MonteCarloOutput<E>
where
    E: Float + PartialOrd + Scalar<Real = E> + ScalarOperand + ToPrimitive,
{
    pub(crate) fn new() -> Self {
        Self {
            full_output: Array2::zeros((0, 0)),
            expectation: Array2::zeros((0, 0)),
            variance: Array2::zeros((0, 0)),
        }
    }

    pub(crate) fn add_vals(&mut self, vals: &EpochOutput<E>) -> Result<()> {
        match self.full_output.dim() {
            (0, 0) => {
                self.full_output = vals.output.clone();
            }
            _ => {
                for row in vals.output.rows() {
                    self.full_output.push_row(row.view())?;
                }
            }
        }
        let stats = vals.summary_statistics()?;
        match self.expectation.dim() {
            (0, 0) => {
                self.expectation = Array2::from_shape_vec(
                    (1, stats.expectation.len()),
                    stats.expectation.to_vec(),
                )?;
            }
            _ => self.expectation.push_row(stats.expectation.view())?,
        }
        match self.variance.dim() {
            (0, 0) => {
                self.variance = Array2::from_shape_vec(
                    (1, stats.covariance.dim().0),
                    stats.covariance.diag().to_vec(),
                )?;
            }
            _ => self.variance.push_row(stats.covariance.diag())?,
        }
        Ok(())
    }

    fn h(&self) -> usize {
        self.expectation.dim().0
    }

    pub(crate) fn is_converged(&self, num_significant_digits: i32) -> bool {
        let mean_expectation = self.mean_expectations();
        let expectation_tolerance = mean_expectation
            .iter()
            .map(|&y| compute_tolerance(y, num_significant_digits))
            .collect::<Vec<_>>();

        let mean_std_dev = self.mean_std_dev();
        let std_dev_tolerance = mean_std_dev
            .iter()
            .map(|&y| compute_tolerance(y, num_significant_digits))
            .collect::<Vec<_>>();

        dbg!(&self.std_dev_std_devs());
        dbg!(&std_dev_tolerance);

        self.std_dev_expectations()
            .into_iter()
            .zip(expectation_tolerance)
            .all(|(val, tolerance)| Scalar::abs(val) < tolerance)
            && self
                .std_dev_std_devs()
                .into_iter()
                .zip(std_dev_tolerance)
                .all(|(val, tolerance)| Scalar::abs(val) < tolerance)
    }

    pub(crate) fn std_dev_expectations(&self) -> Array1<E> {
        (&self.expectation - self.mean_expectations())
            .mapv(|x| Scalar::powi(x, 2))
            .sum_axis(Axis(0))
            .mapv(|x| x / E::from(self.h() * (self.h() - 1)).unwrap())
            .mapv(|x| Scalar::sqrt(x))
    }

    pub(crate) fn mean_expectations(&self) -> Array1<E> {
        self.expectation
            .sum_axis(Axis(0))
            .mapv(|x| x / E::from(self.h()).unwrap())
    }

    pub(crate) fn std_dev_std_devs(&self) -> Array1<E> {
        let std_dev = self.variance.mapv(|v| Scalar::sqrt(v));
        (&std_dev - self.mean_std_dev())
            .mapv(|x| Scalar::powi(x, 2))
            .sum_axis(Axis(0))
            .mapv(|x| x / E::from(self.h() * (self.h() - 1)).unwrap())
            .mapv(|x| Scalar::sqrt(x))
    }

    pub(crate) fn mean_std_dev(&self) -> Array1<E> {
        self.variance
            .mapv(|v| Scalar::sqrt(v))
            .sum_axis(Axis(0))
            .mapv(|x| x / E::from(self.h()).unwrap())
    }

    pub(crate) fn summary_statistics(&self) -> Result<SummaryStatistics<E>> {
        let expectation = self.expectation();
        let covariance = self.covariance(expectation.view())?;

        Ok(SummaryStatistics {
            expectation,
            covariance,
        })
    }

    fn num_samples(&self) -> usize {
        self.full_output.dim().0 / self.h()
    }

    fn expectation(&self) -> Array1<E> {
        self.full_output
            .sum_axis(Axis(0))
            .mapv(|sum| sum / E::from(self.full_output.dim().0).unwrap())
    }

    fn covariance(&self, expectation: ArrayView1<'_, E>) -> Result<Array2<E>> {
        let f = self.full_output.t().to_owned();
        let expectation = expectation.to_owned();
        let ones = Array1::ones(self.num_samples() * self.h());
        let outer = outer_product(&expectation, &ones)?;

        let mat = f - outer;

        let cov = mat
            .dot(&mat.t())
            .mapv(|x| x / E::from(self.full_output.dim().0 - 1).unwrap());

        Ok(cov)
    }
}

impl<E> EpochOutput<E>
where
    E: Scalar<Real = E> + ScalarOperand,
{
    pub(crate) fn new() -> Self {
        Self {
            output: Array2::zeros((0, 0)),
        }
    }

    pub(crate) fn add_row(
        &mut self,
        row: ArrayView1<'_, E>,
    ) -> ::std::result::Result<(), ShapeError> {
        match self.output.dim() {
            (0, 0) => {
                self.output = Array2::from_shape_vec((1, row.len()), row.to_vec())?;
                Ok(())
            }
            _ => self.output.push_row(row),
        }
    }

    pub(crate) fn summary_statistics(&self) -> Result<SummaryStatistics<E>> {
        let expectation = self.expectation();
        let covariance = self.covariance(expectation.view())?;

        Ok(SummaryStatistics {
            expectation,
            covariance,
        })
    }

    fn expectation(&self) -> Array1<E> {
        self.output
            .sum_axis(Axis(0))
            .mapv(|sum| sum / E::from(self.num_samples()).unwrap())
    }

    fn covariance(&self, expectation: ArrayView1<'_, E>) -> Result<Array2<E>> {
        let f = self.output.t().to_owned();
        let expectation = expectation.to_owned();
        let ones = Array1::ones(self.num_samples());
        let outer = outer_product(&expectation, &ones)?;

        let mat = f - outer;

        let cov = mat
            .dot(&mat.t())
            .mapv(|x| x / E::from(self.num_samples() - 1).unwrap());

        Ok(cov)
    }

    fn num_samples(&self) -> usize {
        self.output.dim().0
    }
}

#[cfg(test)]
mod test {
    use ndarray::{s, Array1};
    use ndarray_rand::{
        rand::{Rng, SeedableRng},
        rand_distr::{Distribution, Normal, StandardNormal},
    };
    use rand_isaac::Isaac64Rng;

    use crate::stats::EpochOutput;

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    const fn convert(x: usize) -> Result<f64, &'static str> {
        let result = x as f64;
        if result as usize != x {
            return Err("cannot convert");
        }
        Ok(result)
    }

    #[test]
    fn epoch_expectation_matches_manual_compute() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let mut epochdata: EpochOutput<f64> = EpochOutput::new();

        let values = StandardNormal;

        let row_size = rng.gen::<u8>() as usize;
        let num_samples = 1_000;

        for _ in 0..num_samples {
            let row: Array1<f64> = values.sample_iter(&mut rng).take(row_size).collect();
            epochdata.add_row(row.view()).unwrap();
        }

        let summary_statistics = epochdata.summary_statistics().unwrap();

        for (ii, computed) in summary_statistics.expectation.into_iter().enumerate() {
            let expected = epochdata.output.slice(s![.., ii]).sum()
                / convert(epochdata.num_samples()).unwrap();
            approx::assert_relative_eq!(computed, expected);
        }
    }

    #[test]
    fn epoch_expectation_roughly_matches_expected_when_all_equal() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let mut epochdata: EpochOutput<f64> = EpochOutput::new();

        let mean = rng.gen();
        let std_dev = mean / 100.0;
        let values = Normal::new(mean, std_dev).unwrap();

        let row_size = 5;
        let num_samples = 10_000;

        for _ in 0..num_samples {
            let row: Array1<f64> = values.sample_iter(&mut rng).take(row_size).collect();
            epochdata.add_row(row.view()).unwrap();
        }

        let summary_statistics = epochdata.summary_statistics().unwrap();

        for computed in summary_statistics.expectation {
            approx::assert_relative_eq!(computed, mean, max_relative = 1e-3);
        }
    }

    #[test]
    fn epoch_standard_deviation_roughly_matches_expected_when_all_equal() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        let mut epochdata: EpochOutput<f64> = EpochOutput::new();

        let mean = rng.gen();
        let std_dev = mean / 100.0;
        let values = Normal::new(mean, std_dev).unwrap();

        let row_size = 2;
        let num_samples = 100_000;

        for _ in 0..num_samples {
            let row: Array1<f64> = values.sample_iter(&mut rng).take(row_size).collect();
            epochdata.add_row(row.view()).unwrap();
        }

        let summary_statistics = epochdata.summary_statistics().unwrap();

        for computed in summary_statistics.covariance.diag().mapv(f64::sqrt) {
            approx::assert_relative_eq!(computed, std_dev, max_relative = 1e-2);
        }
    }
}
