use std::marker::PhantomData;

use ndarray::{ArrayView1, ArrayView2};
use ndarray_rand::rand::Rng;
use num_traits::{Float, ToPrimitive};

use crate::{
    core::{Config, Problem},
    input::{Input, Uncertainties},
};

pub struct Set {}
pub struct Unset {}

pub struct Builder<'a, E, R, M, Ex, Va, Co> {
    config: Option<Config<E>>,
    rng: &'a mut R,
    model: M,
    expectation_values: Option<ArrayView1<'a, E>>,
    variances: Option<ArrayView1<'a, E>>,
    covariances: Option<ArrayView2<'a, E>>,
    typestate: PhantomData<(Ex, Va, Co)>,
}

impl<'a, E, R, M> Builder<'a, E, R, M, Unset, Unset, Unset> {
    pub fn new(rng: &'a mut R, model: M) -> Self {
        Self {
            rng,
            model,
            config: None,
            expectation_values: None,
            variances: None,
            covariances: None,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, R, M, Ex, Va, Co> Builder<'a, E, R, M, Ex, Va, Co> {
    fn with_config(mut self, config: Config<E>) -> Self {
        self.config = Some(config);
        self
    }
}

impl<'a, E, R, M, Va, Co> Builder<'a, E, R, M, Unset, Va, Co> {
    pub fn with_input_expectations(
        self,
        expectation_values: ArrayView1<'a, E>,
    ) -> Builder<'_, E, R, M, Set, Va, Co> {
        Builder {
            rng: self.rng,
            model: self.model,
            config: self.config,
            expectation_values: Some(expectation_values),
            variances: self.variances,
            covariances: self.covariances,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, R, M, Ex> Builder<'a, E, R, M, Ex, Unset, Unset> {
    pub fn with_input_variances(
        self,
        variances: ArrayView1<'a, E>,
    ) -> Builder<'_, E, R, M, Ex, Set, Unset> {
        Builder {
            rng: self.rng,
            model: self.model,
            config: self.config,
            expectation_values: self.expectation_values,
            variances: Some(variances),
            covariances: self.covariances,
            typestate: PhantomData,
        }
    }
}

impl<'a, E, R, M, Ex> Builder<'a, E, R, M, Ex, Unset, Unset> {
    fn with_input_covariances(
        self,
        covariances: ArrayView2<'a, E>,
    ) -> Builder<'_, E, R, M, Ex, Unset, Set> {
        Builder {
            rng: self.rng,
            model: self.model,
            config: self.config,
            expectation_values: self.expectation_values,
            variances: self.variances,
            covariances: Some(covariances),
            typestate: PhantomData,
        }
    }
}

impl<'a, E: Float + ToPrimitive, R: Rng, P> Builder<'a, E, R, P, Set, Set, Unset> {
    pub fn build(self) -> Problem<'a, E, R, P> {
        let config = self.config.unwrap_or_default();
        let number_of_trials =
            10_000.max(1 / (1 - 100 * config.required_coverage_probability.to_usize().unwrap()));
        Problem {
            config,
            number_of_trials,
            rng: self.rng,
            model: self.model,
            inputs: Input {
                expectation_values: self.expectation_values.unwrap(),
                uncertainties: Uncertainties::Diagonal(self.variances.unwrap()),
            },
        }
    }
}

impl<'a, E: Float + ToPrimitive, R: Rng, P> Builder<'a, E, R, P, Set, Unset, Set> {
    fn build(self) -> Problem<'a, E, R, P> {
        let config = self.config.unwrap_or_default();
        let number_of_trials =
            10_000.max(1 / (1 - 100 * config.required_coverage_probability.to_usize().unwrap()));
        Problem {
            config,
            number_of_trials,
            rng: self.rng,
            model: self.model,
            inputs: Input {
                expectation_values: self.expectation_values.unwrap(),
                uncertainties: Uncertainties::Full(self.covariances.unwrap()),
            },
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{builder::Builder, core::compute_tolerance, Model};

    use ndarray::Array1;
    use ndarray_rand::{
        rand::{Rng, SeedableRng},
        rand_distr::{Distribution, Normal},
    };
    use rand_isaac::Isaac64Rng;

    #[test]
    fn linear_model_has_expected_properties() {
        let state = 40;
        let mut rng = Isaac64Rng::seed_from_u64(state);
        struct TestModel {
            means: [f64; 2],
            var: [f64; 2],
        }

        impl Model<f64, Isaac64Rng> for TestModel {
            fn apply(
                &self,
                inputs: ndarray::Array1<f64>,
                rng: &mut Isaac64Rng,
            ) -> std::result::Result<ndarray::Array1<f64>, Box<dyn std::error::Error>> {
                let dists = self
                    .means
                    .iter()
                    .zip(self.var)
                    .map(|(&mean, var)| Normal::new(mean, var.sqrt()))
                    .collect::<Result<Vec<_>, _>>()?;

                let res = inputs
                    .into_iter()
                    .map(|input| dists[0].sample(rng) + input * dists[1].sample(rng))
                    .collect();

                Ok(res)
            }
        }

        let means = [rng.gen(), rng.gen()];
        let var = [means[0] / 100.0, means[1] / 100.0];

        let model = TestModel { means, var };

        let num_data_points = 5;

        let expectation_values = (0..num_data_points)
            .map(|_| rng.gen())
            .collect::<Array1<f64>>();
        let variances = expectation_values
            .iter()
            .map(|x| x / 100.0)
            .collect::<Array1<f64>>();

        let mut problem = Builder::new(&mut rng, model)
            .with_input_expectations(expectation_values.view())
            .with_input_variances(variances.view())
            .build();

        let result = problem.run().unwrap();

        let num_significant_digits = problem.config.num_significant_digits as i32;

        for (calc, input) in result.expectation.into_iter().zip(expectation_values) {
            let exp = means[0] + input * means[1];
            let tolerance = compute_tolerance(exp, num_significant_digits);
            println!("{tolerance}, {calc}, {exp}, {}", (calc - exp).abs());
            assert!((calc - exp).abs() < tolerance);
        }
    }
}
