use ndarray::{Array2, ArrayView1, ArrayView2, LinalgScalar};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num_traits::Float;

use crate::input::{
    AffineModel, GaussianAffine, GaussianSampler, InputModel, LatinHypercubeSampler, LhsAffine,
    SampleError, SamplingStrategy,
};

pub type SampleMatrix<E> = Array2<E>;

/// Input to the Monte Carlo simulation
///
/// Input can be provided as a set of means and `marginal_scales`, which can be interpreted according to
/// the chosen sampling model:
/// - Gaussian -> Standard Deviation
/// - Latin Hypercube -> Spread
///
/// Correlated inputs are assumed to be described by a covariance matrix, and only support Gaussian
/// sampling
#[derive(Clone)]
pub enum InputSpec<'a, E> {
    Independent {
        means: ArrayView1<'a, E>,
        marginal_scale: ArrayView1<'a, E>,
    },
    Correlated {
        means: ArrayView1<'a, E>,
        covariance: ArrayView2<'a, E>,
    },
}

/// Compiled input for sampling
pub struct CompiledInput<E, S, M>
where
    S: SamplingStrategy<E>,
    M: InputModel<E, Space = S::Space>,
{
    pub(crate) model: M,
    pub(crate) seed: u64,
    pub(crate) sampler: S,
    float: std::marker::PhantomData<E>,
}

impl<E> InputSpec<'_, E> {
    pub fn compile_gaussian(
        self,
        seed: u64,
    ) -> Result<CompiledInput<E, GaussianSampler, GaussianAffine<E>>, SampleError>
    where
        E: Float + Scalar + Lapack,
        StandardNormal: Distribution<E>,
    {
        let sampler = GaussianSampler;

        let model = match self {
            Self::Independent {
                means,
                marginal_scale,
            } => AffineModel::diagonal(means, marginal_scale),

            Self::Correlated { means, covariance } => AffineModel::gaussian(means, covariance),
        }?;

        Ok(CompiledInput {
            model: GaussianAffine { inner: model },
            sampler,
            seed,
            float: std::marker::PhantomData,
        })
    }

    pub fn compile_lhs(
        self,
        seed: u64,
    ) -> Result<CompiledInput<E, LatinHypercubeSampler, LhsAffine<E>>, SampleError>
    where
        E: Float + Scalar + From<f64>,
        StandardNormal: Distribution<E>,
    {
        let sampler = LatinHypercubeSampler;

        let model = match self {
            Self::Independent {
                means,
                marginal_scale,
            } => AffineModel::diagonal(means, marginal_scale),

            Self::Correlated { .. } => {
                return Err(SampleError::InvalidPattern(
                    "called an lhs sampling routine with correlated data",
                ));
            }
        }?;

        Ok(CompiledInput {
            model: LhsAffine { inner: model },
            sampler,
            seed,
            float: std::marker::PhantomData,
        })
    }
}

impl<E, S, M> CompiledInput<E, S, M>
where
    S: SamplingStrategy<E>,
    M: InputModel<E, Space = S::Space>,
    E: Float + LinalgScalar,
{
    pub fn init_state(&self) -> S::State {
        S::init(self.seed)
    }

    pub fn sample(&self, state: &mut S::State, n: usize) -> SampleMatrix<E> {
        S::sample(state, n, self.model.dim())
    }

    pub fn sample_stateless(&self, n: usize) -> SampleMatrix<E> {
        let mut state = self.init_state();
        self.sample(&mut state, n)
    }
}

#[cfg(test)]
mod test {
    use super::InputSpec;
    use crate::input::model::InputModel;
    #[test]
    fn gaussian_sample_has_expected_shape() {
        use ndarray::array;

        let means = array![0.0, 0.0, 0.0];
        let marginal_scale = array![1.0, 1.0, 1.0];

        let compiled = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        }
        .compile_gaussian(42)
        .unwrap();

        let z = compiled.sample_stateless(500);
        let samples = compiled.model.apply(z);

        assert_eq!(samples.shape(), &[500, 3]);
    }

    #[test]
    fn gaussian_independent_recovers_moments() {
        use ndarray::array;

        let means = array![5.0];
        let stddev = array![2.0];

        let input = InputSpec::Independent {
            means: means.view(),
            marginal_scale: stddev.view(),
        };

        let compiled = input.compile_gaussian(42).unwrap();

        let z = compiled.sample_stateless(100_000);
        let samples = compiled.model.apply(z);

        let x = samples.column(0);

        let mean = x.iter().copied().sum::<f64>() / x.len() as f64;

        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0);

        assert!((mean - 5.0).abs() < 0.02);
        assert!((variance - 4.0).abs() < 0.05);
    }

    #[test]
    fn gaussian_correlated_recovers_covariance() {
        use ndarray::array;

        let means = array![0.0, 0.0];

        let covariance = array![[1.0, 0.8], [0.8, 1.0]];

        let input = InputSpec::Correlated {
            means: means.view(),
            covariance: covariance.view(),
        };

        let compiled = input.compile_gaussian(42).unwrap();

        let z = compiled.sample_stateless(200_000);
        let samples = compiled.model.apply(z);

        let x = samples.column(0);
        let y = samples.column(1);

        let mx = x.iter().copied().sum::<f64>() / x.len() as f64;

        let my = y.iter().copied().sum::<f64>() / y.len() as f64;

        let cov = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - mx) * (b - my))
            .sum::<f64>()
            / (x.len() as f64 - 1.0);

        assert!((cov - 0.8).abs() < 0.02);
    }

    #[test]
    fn gaussian_same_seed_is_reproducible() {
        use ndarray::array;

        let means = array![0.0];
        let scale = array![1.0];

        let a = InputSpec::Independent {
            means: means.view(),
            marginal_scale: scale.view(),
        }
        .compile_gaussian(123)
        .unwrap();

        let b = InputSpec::Independent {
            means: means.view(),
            marginal_scale: scale.view(),
        }
        .compile_gaussian(123)
        .unwrap();

        let za = a.sample_stateless(10_000);
        let zb = b.sample_stateless(10_000);

        let sa = a.model.apply(za);
        let sb = b.model.apply(zb);

        assert_eq!(sa, sb);
    }

    #[test]
    fn gaussian_different_seed_changes_output() {
        use ndarray::array;

        let means = array![0.0];
        let scale = array![1.0];

        let a = InputSpec::Independent {
            means: means.view(),
            marginal_scale: scale.view(),
        }
        .compile_gaussian(1)
        .unwrap();

        let b = InputSpec::Independent {
            means: means.view(),
            marginal_scale: scale.view(),
        }
        .compile_gaussian(2)
        .unwrap();

        let za = a.sample_stateless(100);
        let zb = b.sample_stateless(100);

        let sa = a.model.apply(za);
        let sb = b.model.apply(zb);

        assert_ne!(sa, sb);
    }

    #[test]
    fn compile_rejects_non_positive_definite_covariance() {
        use ndarray::array;

        let means = array![0.0, 0.0];

        let covariance = array![[1.0, 2.0], [2.0, 1.0]];

        let input = InputSpec::Correlated {
            means: means.view(),
            covariance: covariance.view(),
        };

        assert!(input.compile_gaussian(42).is_err());
    }

    #[test]
    fn lhs_sample_has_expected_shape() {
        use ndarray::array;

        let means = array![0.0, 0.0, 0.0];
        let marginal_scale = array![1.0, 1.0, 1.0];

        let compiled = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        }
        .compile_lhs(42)
        .unwrap();

        let z = compiled.sample_stateless(500);
        let samples = compiled.model.apply(z);

        assert_eq!(samples.shape(), &[500, 3]);
    }

    #[test]
    fn lhs_same_seed_is_reproducible() {
        use ndarray::array;

        let means = array![0.0];
        let marginal_scale = array![1.0];

        let a = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        }
        .compile_lhs(123)
        .unwrap();

        let b = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        }
        .compile_lhs(123)
        .unwrap();

        assert_eq!(a.sample_stateless(1000), b.sample_stateless(1000));
    }

    #[test]
    fn lhs_uses_each_stratum_once() {
        use ndarray::array;

        let means = array![0.0];
        let scale = array![1.0];

        let compiled = InputSpec::Independent {
            means: means.view(),
            marginal_scale: scale.view(),
        }
        .compile_lhs(42)
        .unwrap();

        let n = 100;

        let samples = compiled.sample_stateless(n);

        let mut vals = samples.column(0).to_vec();

        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (i, x) in vals.iter().enumerate() {
            let lower = i as f64 / n as f64;
            let upper = (i + 1) as f64 / n as f64;

            assert!(*x >= lower);
            assert!(*x < upper);
        }
    }

    #[test]
    fn lhs_stratifies_every_dimension() {
        use ndarray::array;

        let means = array![0.0, 0.0, 0.0];

        let scale = array![1.0, 1.0, 1.0];

        let compiled = InputSpec::Independent {
            means: means.view(),
            marginal_scale: scale.view(),
        }
        .compile_lhs(42)
        .unwrap();

        let n = 100;

        let samples = compiled.sample_stateless(n);

        for col in samples.columns() {
            let mut vals = col.to_vec();

            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for (i, x) in vals.iter().enumerate() {
                let lower = i as f64 / n as f64;
                let upper = (i + 1) as f64 / n as f64;

                assert!(*x >= lower);
                assert!(*x < upper);
            }
        }
    }

    #[test]
    fn rejects_mismatched_mean_scaling_lengths() {
        use ndarray::array;

        let means = array![1.0, 2.0];
        let marginal_scale = array![1.0];

        let input = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        };

        assert!(input.compile_gaussian(42).is_err());
    }

    #[test]
    fn rejects_mismatched_mean_covariance_dimensions() {
        use ndarray::array;

        let means = array![1.0, 2.0];

        let covariance = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let input = InputSpec::Correlated {
            means: means.view(),
            covariance: covariance.view(),
        };

        assert!(input.compile_gaussian(42).is_err());
    }
    #[test]
    fn zero_scale_produces_constant_samples() {
        use ndarray::array;

        let means = array![5.0];
        let marginal_scale = array![0.0];

        let compiled = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        }
        .compile_gaussian(42)
        .unwrap();

        let z = compiled.sample_stateless(10_000);

        let samples = compiled.model.apply(z);

        assert!(samples.iter().all(|x| (*x - 5f64).abs() < 1e-12));
    }

    #[test]
    fn diagonal_covariance_matches_independent_statistics() {
        use ndarray::array;

        let means = array![1.0, 2.0, 3.0];

        let marginal_scale = array![1.0, 2.0, 3.0];

        let covariance = array![[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]];

        let independent = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        }
        .compile_gaussian(42)
        .unwrap();

        let correlated = InputSpec::Correlated {
            means: means.view(),
            covariance: covariance.view(),
        }
        .compile_gaussian(42)
        .unwrap();

        let za = independent.sample_stateless(200_000);
        let zb = correlated.sample_stateless(200_000);

        let a = independent.model.apply(za);
        let b = correlated.model.apply(zb);

        for j in 0..3 {
            let mean_a = a.column(j).iter().copied().sum::<f64>() / a.nrows() as f64;

            let mean_b = b.column(j).iter().copied().sum::<f64>() / b.nrows() as f64;

            assert!((mean_a - mean_b).abs() < 0.02);
        }
    }

    #[test]
    fn sampling_is_pure() {
        use ndarray::array;

        let means = array![0.0, 0.0];
        let marginal_scale = array![1.0, 1.0];

        let compiled = InputSpec::Independent {
            means: means.view(),
            marginal_scale: marginal_scale.view(),
        }
        .compile_gaussian(42)
        .unwrap();

        let za = compiled.sample_stateless(10_000);
        let zb = compiled.sample_stateless(10_000);

        let a = compiled.model.apply(za);
        let b = compiled.model.apply(zb);

        assert_eq!(a, b);
    }
}
