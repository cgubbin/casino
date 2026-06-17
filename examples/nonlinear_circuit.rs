use montecore::*;
use ndarray::{arr1, ArrayView1};

struct DiodeCircuit;

// I = I0 (exp(V/Vt) - 1)
impl Operator<f64> for DiodeCircuit {
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
        let v = x[0];
        let vt = x[1];

        if vt <= 0.0 || v > 100.0 {
            return EvalResult::try_from_parts(arr1(&[0.0]), arr1(&[false]));
        }

        let i0 = 1e-9;
        let current = i0 * ((v / vt).exp() - 1.0);

        EvalResult::try_from_parts(arr1(&[current]), arr1(&[true]))
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

    let result = MonteCarlo::run(input, DiodeCircuit, sampling_method, options)?;

    println!("mean deflection = {:?}", result.statistics.mean);
    println!("covariance = {:?}", result.statistics.covariance);

    Ok(())
}
