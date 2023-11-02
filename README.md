Casino is a library to carry out Monte-Carlo propagation of distributions.

Users need to implement the [`Model`] trait. This has a single method which takes
an array of values, which are the inputs to the measurement model, and converts
them to an array of outputs.

Calculations are run using using the [`Problem`] type. The following creates a simple
measurement model in which the outputs relate to the inputs via a quadratic equation:
```rust
use casino::{Builder, Model};
use ndarray::Array1;
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
//!
struct Example {
    a: f64,
    b: f64,
}
//!
impl Model<f64> for Example {
    fn apply(&self, inputs: Array1<f64>) -> Result<Array1<f64>, Box<dyn ::std::error::Error>> {
        Ok(inputs.mapv(|x| self.a + x.powi(2) * self.b))
    }
}
//!
let model = Example { a: 1.0, b: 2.0 };
//!
let state = 40;
let mut rng = Isaac64Rng::seed_from_u64(state);
//!
let expectations = Array1::linspace(0., 10., 11);
let variances = expectations.iter().map(|mean| mean / 100.0).collect::<Array1<_>>();
//!
let mut problem = Builder::new(&mut rng, model)
                .with_input_expectations(expectations.view())
                .with_input_variances(variances.view())
                .build();
```

When the algorithm is run, the apply method is called repeatedly until convergence is achieved
in the distributional properties of the output variables. This follows ISO Standard Uncertainty of measurement Supplement 1 (98-3).
