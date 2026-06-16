use ndarray::Array2;
use ndarray_rand::{
    rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng},
    rand_distr::{Distribution, StandardNormal},
    RandomExt,
};
use num_traits::Float;

pub trait ReferenceSpace {}

pub struct GaussianSpace;
pub struct UnitCubeSpace;

impl ReferenceSpace for GaussianSpace {}
impl ReferenceSpace for UnitCubeSpace {}

pub enum SamplingMethod {
    Gaussian,
    LatinHypercube,
}

pub trait LinearSamplingStrategy<E>: SamplingStrategy<E> {}

pub trait DesignSamplingStrategy<E>: SamplingStrategy<E> {}

pub trait SamplingStrategy<E> {
    type Space: ReferenceSpace;
    type State;

    fn init(seed: u64) -> Self::State;

    fn sample(state: &mut Self::State, n: usize, dim: usize) -> Array2<E>;
}

pub struct GaussianSampler;

pub struct GaussianState {
    rng: ndarray_rand::rand::rngs::StdRng,
}

impl<E> SamplingStrategy<E> for GaussianSampler
where
    StandardNormal: Distribution<E>,
{
    type Space = GaussianSpace;
    type State = GaussianState;

    fn init(seed: u64) -> Self::State {
        GaussianState {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn sample(state: &mut Self::State, n: usize, dim: usize) -> Array2<E> {
        Array2::random_using((n, dim), StandardNormal, &mut state.rng)
    }
}

pub struct LatinHypercubeSampler;

pub struct LhsState {
    rng: ndarray_rand::rand::rngs::StdRng,
}

impl<E> SamplingStrategy<E> for LatinHypercubeSampler
where
    E: Float + From<f64>,
{
    type Space = UnitCubeSpace;
    type State = LhsState;

    fn init(seed: u64) -> Self::State {
        LhsState {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn sample(state: &mut Self::State, n: usize, dim: usize) -> Array2<E> {
        let mut out = Array2::<E>::zeros((n, dim));

        // Precompute base intervals: (i + u) / n
        let inv_n = 1.0 / (n as f64);

        // For each dimension independently
        for j in 0..dim {
            // 1. create stratified points in [0,1]
            let mut column: Vec<f64> = (0..n)
                .map(|i| {
                    let u: f64 = state.rng.random();
                    (i as f64 + u) * inv_n
                })
                .collect();

            // 2. permute them (key LHS step)
            column.shuffle(&mut state.rng);

            // 3. write into output
            for i in 0..n {
                out[[i, j]] = <E as From<f64>>::from(column[i]);
            }
        }

        out
    }
}

// pub struct Sobol;

// impl<E> SamplingStrategy<E> for Sobol
// where
//     E: Float + From<f64>,
// {
//     fn sample(&self, seed: u64, n: usize, dim: usize) -> Array2<E> {
//         // placeholder: use sobol crate or deterministic generator
//         unimplemented!()
//     }
// }
