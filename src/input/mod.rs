mod input;
mod model;
mod sampling;

use model::{AffineModel, GaussianAffine, LhsAffine};
use sampling::{
    GaussianSampler, GaussianSpace, LatinHypercubeSampler, ReferenceSpace, UnitCubeSpace,
};

pub use input::CompiledInput;
pub use model::InputModel;
pub use sampling::SamplingStrategy;

pub use input::InputSpec;
pub use sampling::SamplingMethod;

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
