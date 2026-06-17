use approx::assert_abs_diff_eq;

use casino::{EvalResult, InputSpec, MonteCarlo, Operator, OperatorError};

use ndarray::{Array1, Array2, ArrayView1};

//
// ============================
// Analytical reference model
// ============================
//
//
pub trait AnalyticalModel<E> {
    fn true_mean(&self) -> ndarray::Array1<E>;
    fn true_covariance(&self) -> ndarray::Array2<E>;
}

struct LinearProblem {
    a: Array2<f64>,
    mu: Array1<f64>,
    sigma: Array2<f64>,
}

impl AnalyticalModel<f64> for LinearProblem {
    fn true_mean(&self) -> Array1<f64> {
        self.a.dot(&self.mu)
    }

    fn true_covariance(&self) -> Array2<f64> {
        self.a.dot(&self.sigma).dot(&self.a.t())
    }
}

//
// ============================
// Operator under test
// ============================
//

#[derive(Clone)]
struct LinearOperator {
    a: Array2<f64>,
}

impl Operator<f64> for LinearOperator {
    fn dim_in(&self) -> usize {
        self.a.ncols()
    }

    fn dim_out(&self) -> usize {
        self.a.nrows()
    }

    fn apply(
        &self,
        x: ArrayView1<'_, f64>,
    ) -> Result<EvalResult<f64, ndarray::Ix1>, OperatorError> {
        dbg!(&x);
        let y = self.a.dot(&x.to_owned());

        EvalResult::try_from_parts(y.clone(), ndarray::Array1::from_elem(y.len(), true))
    }
}

//
// ============================
// Test: covariance correctness
// ============================
//

#[test]
fn gaussian_linear_covariance_matches_analytic() {
    use casino::*;
    use ndarray::{arr1, arr2};

    let problem = LinearProblem {
        a: arr2(&[[1.0, 2.0], [0.0, 1.0]]),
        mu: arr1(&[1.0, 2.0]),
        // sigma: arr2(&[[1.0, 0.3], [0.3, 2.0]]),
        sigma: arr2(&[[0.001, 0.0003], [0.0003, 0.002]]),
    };

    let input = InputSpec::Correlated {
        means: problem.mu.view(),
        covariance: problem.sigma.view(),
    };

    let result = MonteCarlo::run(
        input,
        LinearOperator {
            a: problem.a.clone(),
        },
        SamplingMethod::Gaussian,
        MonteCarloOptions {
            seed: 42,
            batch_size: 512,
            min_samples: 20_000,
            max_samples: 200_000,
            rel_tol: 1e-3,
        },
    )
    .expect("Monte Carlo run failed");

    dbg!(&result);

    let expected_mean = problem.true_mean();
    let expected_cov = problem.true_covariance();
    dbg!(&expected_mean, &expected_cov);

    // ---- mean check ----
    for i in 0..expected_mean.len() {
        assert_abs_diff_eq!(result.statistics.mean[i], expected_mean[i], epsilon = 0.05);
    }

    // ---- covariance check ----
    for i in 0..expected_cov.nrows() {
        for j in 0..expected_cov.ncols() {
            assert_abs_diff_eq!(
                result.statistics.covariance[[i, j]],
                expected_cov[[i, j]],
                epsilon = 0.1
            );
        }
    }
}
