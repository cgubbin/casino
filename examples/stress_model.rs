use casino::*;
use ndarray::arr2;

struct StressModel;

// σ = σx + σy - ρ sqrt(σx σy)
impl Operator<f64> for StressModel {
    fn dim_in(&self) -> usize {
        2
    }
    fn dim_out(&self) -> usize {
        1
    }

    fn apply(
        &self,
        x: ndarray::ArrayView1<'_, f64>,
    ) -> Result<EvalResult<f64, ndarray::Ix1>, OperatorError> {
        let sx = x[0];
        let sy = x[1];

        let stress = sx + sy;

        EvalResult::try_from_parts(ndarray::arr1(&[stress]), ndarray::arr1(&[true]))
    }
}

fn main() -> Result<(), CasinoError> {
    let covariance = arr2(&[[1.0, 0.8], [0.8, 1.0]]);
    let means = ndarray::arr1(&[10.0, 10.0]);

    let input = InputSpec::Correlated {
        means: means.view(),
        covariance: covariance.view(),
    };

    let sampling_method = SamplingMethod::LatinHypercube;

    let options = MonteCarloOptions {
        seed: 42,
        batch_size: 512,
        min_samples: 5_000,
        max_samples: 500_000,
        rel_tol: 1e-3,
    };

    let result = MonteCarlo::run(input, StressModel, sampling_method, options)?;

    println!("{:?}", result.statistics);
    Ok(())
}
