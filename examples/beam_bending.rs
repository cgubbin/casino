use montecore::*;
use ndarray::{arr1, ArrayView1};

struct BeamBending;

// y = F / (E I)
impl Operator<f64> for BeamBending {
    fn dim_in(&self) -> usize {
        2
    }
    fn dim_out(&self) -> usize {
        1
    }

    fn apply(
        &self,
        x: ArrayView1<'_, f64>,
    ) -> Result<EvalResult<f64, ndarray::Ix1>, OperatorError> {
        let force = x[0];
        let stiffness = x[1];

        if stiffness <= 0.0 {
            return EvalResult::try_from_parts(arr1(&[0.0]), arr1(&[false]));
        }

        let deflection = force / stiffness;

        EvalResult::try_from_parts(arr1(&[deflection]), arr1(&[true]))
    }
}

fn main() -> Result<(), MontecoreError> {
    let means = arr1(&[100.0, 10.0]);
    let marginal_scale = arr1(&[10.0, 2.0]);

    let input = InputSpec::Independent {
        means: means.view(),
        marginal_scale: marginal_scale.view(),
    };

    let sampling_method = SamplingMethod::LatinHypercube;

    let options = MonteCarloOptions {
        seed: 42,
        batch_size: 512,
        min_samples: 10_000,
        max_samples: 1_000_000,
        rel_tol: 1e-3,
    };

    let result = MonteCarlo::run(input, BeamBending, sampling_method, options)?;

    println!("mean deflection = {:?}", result.statistics.mean);
    println!("covariance = {:?}", result.statistics.covariance);

    Ok(())
}
