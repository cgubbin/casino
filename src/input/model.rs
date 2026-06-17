use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, LinalgScalar};
use ndarray_linalg::{Cholesky, Lapack, Scalar, UPLO};
use num_traits::Float;

use super::{GaussianSpace, ReferenceSpace, SampleError, UnitCubeSpace};

pub trait InputModel<E> {
    type Space: ReferenceSpace;

    fn dim(&self) -> usize;
    fn apply(&self, z: Array2<E>) -> Array2<E>;
}

#[derive(Clone)]
pub struct AffineModel<E> {
    mean: Array1<E>,
    transform: Array2<E>,
}

impl<E> AffineModel<E> {
    pub(super) fn gaussian(
        mean: ArrayView1<'_, E>,
        covariance: ArrayView2<'_, E>,
    ) -> Result<Self, SampleError>
    where
        E: Scalar + Lapack,
    {
        let dim = mean.len();
        if covariance.shape() != [dim, dim] {
            return Err(SampleError::MatrixShapeError {
                found: (covariance.shape()[0], covariance.shape()[1]),
                expected: (dim, dim),
            });
        }

        let chol = covariance.cholesky(UPLO::Lower)?;

        Ok(Self {
            mean: mean.to_owned(),
            transform: chol.t().to_owned(),
        })
    }

    pub(super) fn diagonal(
        mean: ArrayView1<'_, E>,
        scale: ArrayView1<'_, E>,
    ) -> Result<Self, SampleError>
    where
        E: Float,
    {
        if scale.len() != mean.len() {
            return Err(SampleError::VectorShapeError {
                found: scale.len(),
                expected: mean.len(),
            });
        }

        Ok(Self {
            mean: mean.to_owned(),
            transform: Array2::from_diag(&scale),
        })
    }
}

impl<E> InputModel<E> for AffineModel<E>
where
    E: Float + LinalgScalar + std::fmt::Debug,
{
    type Space = UnitCubeSpace; // default assumption

    fn dim(&self) -> usize {
        self.mean.len()
    }

    fn apply(&self, z: Array2<E>) -> Array2<E> {
        let mu = self.mean.view().insert_axis(Axis(0));
        
        z.dot(&self.transform) + mu
    }
}

pub struct GaussianAffine<E> {
    pub(super) inner: AffineModel<E>,
}

impl<E> InputModel<E> for GaussianAffine<E>
where
    E: Float + LinalgScalar + std::fmt::Debug,
{
    type Space = GaussianSpace;

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn apply(&self, z: Array2<E>) -> Array2<E> {
        self.inner.apply(z)
    }
}

pub struct LhsAffine<E> {
    pub(super) inner: AffineModel<E>,
}

impl<E> InputModel<E> for LhsAffine<E>
where
    E: Float + LinalgScalar + std::fmt::Debug,
{
    type Space = UnitCubeSpace;

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn apply(&self, z: Array2<E>) -> Array2<E> {
        self.inner.apply(z)
    }
}
