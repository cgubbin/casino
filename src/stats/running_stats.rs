use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, LinalgScalar, ScalarOperand};
use num_traits::Float;

use super::{FinalStatistics, StatisticalDiagnostics, SummaryStatistics};
use crate::controller::{ConvergenceCriterion, ConvergenceStatus};

#[derive(Clone, Debug)]
pub struct RunningStats<E> {
    n: usize, // Global batch count
    mean: Array1<E>,
    m2: Array2<E>, // unnormalised covariance accumulator

    pair_count: Array2<usize>, // Per-dimension effective sample count
    count: Array1<usize>,      // Per-dimension valid counts
}

impl<E> RunningStats<E>
where
    E: Float + LinalgScalar + ScalarOperand + num_traits::FromPrimitive,
{
    /// Create an empty accumulator for a given dimension
    pub fn new(dim: usize) -> Self {
        Self {
            n: 0,
            mean: Array1::zeros(dim),
            m2: Array2::zeros((dim, dim)),
            count: Array1::zeros(dim),
            pair_count: Array2::zeros((dim, dim)),
        }
    }

    pub fn count(&self) -> usize {
        self.n
    }

    pub fn mean(&self) -> ArrayView1<'_, E> {
        self.mean.view()
    }

    pub fn update(&mut self, x: ArrayView1<'_, E>, valid: ArrayView1<'_, bool>)
    where
        E: Float + LinalgScalar,
    {
        self.n += 1;
        let dim = self.mean.len();

        assert_eq!(x.len(), dim);
        assert_eq!(valid.len(), dim);

        // Save old means because covariance update
        // requires the pre-update means.
        let old_mean = self.mean.clone();

        //
        // Mean update (per dimension)
        //
        for i in 0..dim {
            if !valid[i] {
                continue;
            }

            self.count[i] += 1;

            let n = E::from(self.count[i]).unwrap();

            let delta = x[i] - self.mean[i];

            self.mean[i] = self.mean[i] + delta / n;
        }

        //
        // Covariance update (pairwise)
        //
        for i in 0..dim {
            if !valid[i] {
                continue;
            }

            let delta_i = x[i] - old_mean[i];
            let _delta2_i = x[i] - self.mean[i];

            for j in 0..dim {
                if !valid[j] {
                    continue;
                }

                self.pair_count[[i, j]] += 1;

                let _delta_j = x[j] - old_mean[j];
                let delta2_j = x[j] - self.mean[j];

                // Multivariate Welford update
                self.m2[[i, j]] = self.m2[[i, j]] + delta_i * delta2_j;
            }
        }
    }

    pub fn update_batch(&mut self, x: ArrayView2<'_, E>, valid: ArrayView2<'_, bool>)
    where
        E: Float + LinalgScalar + std::fmt::Debug,
    {
        // TODO: For simplicity uses the correct row-wise implementation, should be vectorised
        for i in 0..x.nrows() {
            self.update(x.row(i), valid.row(i));
        }
    }

    // // Merge two accumulators (parallel-safe monoid operation)
    pub fn merge(&mut self, other: &Self) {
        if self.n == 0 {
            self.n = other.n;
            self.mean = other.mean.clone();
            self.m2 = other.m2.clone();
            self.pair_count = other.pair_count.clone();
            return;
        }
        if other.n == 0 {
            return;
        }

        let n1 = E::from(self.n).unwrap();
        let n2 = E::from(other.n).unwrap();
        let n = n1 + n2;

        let mean1 = self.mean.clone();
        let mean2 = other.mean.clone();

        let delta = &mean2 - &mean1;

        // combined mean
        self.mean = (&mean1 * n1 + &mean2 * n2) / n;

        // combine M2 (standard parallel Welford merge)
        let m2 = &self.m2
            + other.m2.clone()
            + (&delta
                .clone()
                .insert_axis(Axis(1))
                .dot(&delta.insert_axis(Axis(0))))
                * (E::from((n1 * n2) / n).unwrap());

        self.m2 = m2;
        self.n += other.n;

        self.pair_count = &self.pair_count + &other.pair_count;
        self.count = &self.count + &other.count;
    }

    /// Final covariance estimate
    pub fn covariance(&self) -> Array2<E> {
        if self.n < 2 {
            return Array2::zeros((self.mean.len(), self.mean.len()));
        }

        let denom = E::from(self.n - 1).unwrap();
        &self.m2 / denom
    }

    pub fn std_dev(&self) -> Array1<E> {
        self.covariance().diag().mapv(|v| v.sqrt())
    }

    /// Standard error of the mean per component
    pub fn mean_standard_error(&self) -> Array1<E> {
        let cov = self.covariance();

        let n = E::from(self.n).unwrap();

        cov.diag().mapv(|v| (v / n).sqrt())
    }

    /// Standard error of the standard deviation (delta method)
    pub fn std_standard_error(&self) -> Array1<E> {
        let std = self.std_dev();

        let n = E::from(self.n.saturating_sub(1)).unwrap();

        std.mapv(|s| s / (E::from(2.0).unwrap() * n).sqrt())
    }

    fn valid_fraction(&self) -> Array1<E> {
        self.count
            .mapv(|each| E::from(each).unwrap() / E::from(self.n).unwrap())
    }

    /// Final mean
    pub fn finalize_mean(&self) -> Array1<E> {
        self.mean.clone()
    }

    pub fn finalize(&self) -> FinalStatistics<E> {
        let summary = SummaryStatistics {
            mean: self.finalize_mean(),
            covariance: self.covariance(),
        };
        let diagnostics = StatisticalDiagnostics {
            mean_standard_error: self.mean_standard_error(),
            std_standard_error: self.std_standard_error(),
            valid_fraction: self.valid_fraction(),
        };

        FinalStatistics {
            summary,
            diagnostics,
        }
    }

    pub fn check_convergence(&self, criterion: &ConvergenceCriterion<E>) -> ConvergenceStatus<E> {
        let se = self.mean_standard_error();
        let mean = &self.mean;

        let mut converged = true;

        match criterion {
            ConvergenceCriterion::AbsoluteMeanError { tol } => {
                for i in 0..mean.len() {
                    if se[i] > *tol {
                        converged = false;
                        break;
                    }
                }
            }

            ConvergenceCriterion::RelativeMeanError { rel_tol } => {
                for i in 0..mean.len() {
                    let scale = mean[i].abs().max(E::from(1e-12).unwrap());
                    if se[i] / scale > *rel_tol {
                        converged = false;
                        break;
                    }
                }
            }
        }

        ConvergenceStatus {
            converged,
            max_se: se,
        }
    }
}

#[cfg(test)]
mod running_stats_tests {
    use super::*;
    use ndarray_rand::{
        rand::{rngs::StdRng, SeedableRng},
        rand_distr::{Distribution, Normal},
    };

    #[test]
    fn mean_matches_direct_computation() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Normal::new(2.0, 0.5).unwrap();

        let dim = 3;
        let mut stats = RunningStats::<f64>::new(dim);
        let valid = ndarray::Array1::from(vec![true; dim]);

        let mut samples = Vec::new();

        for _ in 0..10_000 {
            let row: Vec<f64> = (0..dim).map(|_| dist.sample(&mut rng)).collect();
            let arr = ndarray::Array1::from(row.clone());
            stats.update(arr.view(), valid.view());
            samples.push(arr);
        }

        let empirical: Vec<f64> = (0..dim)
            .map(|j| samples.iter().map(|s| s[j]).sum::<f64>() / samples.len() as f64)
            .collect();

        let mean = stats.finalize_mean();

        for j in 0..dim {
            assert!((mean[j] - empirical[j]).abs() < 1e-2);
        }
    }

    #[test]
    fn covariance_is_positive_semi_definite() {
        let mut rng = StdRng::seed_from_u64(7);
        let dist = Normal::new(0.0, 1.0).unwrap();

        let dim = 4;
        let mut stats = RunningStats::<f64>::new(dim);
        let valid = ndarray::Array1::from(vec![true; dim]);

        for _ in 0..5_000 {
            let row: Vec<f64> = (0..dim).map(|_| dist.sample(&mut rng)).collect();
            let arr = ndarray::Array1::from(row);
            stats.update(arr.view(), valid.view());
        }

        let cov = stats.covariance();

        // PSD check via x^T C x ≥ 0
        let x = ndarray::Array1::from(vec![1.0, -1.0, 0.5, -0.2]);
        let quad = x.t().dot(&cov.dot(&x));

        assert!(quad >= -1e-10);
    }

    #[test]
    fn running_stats_merge_is_consistent() {
        let mut a = RunningStats::<f64>::new(2);
        let mut b = RunningStats::<f64>::new(2);

        let x1 = ndarray::Array1::from(vec![1.0, 2.0]);
        let x2 = ndarray::Array1::from(vec![3.0, 4.0]);

        let valid = ndarray::Array1::from(vec![true, true]);

        a.update(x1.view(), valid.view());
        b.update(x2.view(), valid.view());

        a.merge(&b);

        let mean = a.finalize_mean();

        assert!(mean[0] > 0.0);
    }

    #[test]
    fn running_stats_matches_numpy_like_mean() {
        let mut rs = RunningStats::<f64>::new(2);

        let data = vec![
            ndarray::arr1(&[1.0, 2.0]),
            ndarray::arr1(&[3.0, 4.0]),
            ndarray::arr1(&[5.0, 6.0]),
        ];

        for x in &data {
            rs.update(x.view(), ndarray::Array1::from_elem(2, true).view());
        }

        let mean = rs.mean();

        assert!((mean[0] - 3.0).abs() < 1e-12);
        assert!((mean[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn running_stats_respects_validity_mask() {
        let mut rs = RunningStats::<f64>::new(2);

        let x = ndarray::arr1(&[1.0, 2.0]);
        let valid = ndarray::arr1(&[true, false]);

        rs.update(x.view(), valid.view());

        let mean = rs.mean();

        // only first dimension should contribute
        assert_eq!(mean[0], 1.0);
    }

    #[test]
    fn running_stats_merge_is_associative() {
        let mut a = RunningStats::<f64>::new(2);
        let mut b = RunningStats::<f64>::new(2);

        for i in 0..10 {
            let x = ndarray::arr1(&[i as f64, (i * 2) as f64]);
            let valid = ndarray::Array1::from_elem(2, true);

            if i < 5 {
                a.update(x.view(), valid.view());
            } else {
                b.update(x.view(), valid.view());
            }
        }

        let mut c1 = a.clone();
        c1.merge(&b);

        let mut c2 = b.clone();
        c2.merge(&a);

        assert_eq!(c1.mean(), c2.mean());
    }
}
