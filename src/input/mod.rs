mod input;
mod model;
mod sampling;

use model::{AffineModel, GaussianAffine, LhsAffine};
use sampling::{
    GaussianSampler, GaussianSpace, LatinHypercubeSampler, ReferenceSpace, UnitCubeSpace,
};

pub(crate) use input::{CompiledInput, InputSpec};
pub(crate) use model::InputModel;
pub(crate) use sampling::SamplingStrategy;

#[derive(thiserror::Error, Debug)]
pub enum SampleError {
    #[error("invalid pattern: {0:?}")]
    InvalidPattern(&'static str),
    #[error("error in cholesky routine")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),
    #[error("expected a vector of {expected:?} elements, found {found:?}")]
    VectorShapeError { expected: usize, found: usize },
    #[error("expected a matrix of shape {expected:?}, found {found:?}")]
    MatrixShapeError {
        expected: (usize, usize),
        found: (usize, usize),
    },
}
