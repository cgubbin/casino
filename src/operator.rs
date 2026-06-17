/// The [`Operator`] trait is implemented by callers. The operator transforms a set of input values
/// to a set of output values:
///
/// ```
/// use casino::*;
/// use ndarray::{arr1, ArrayView1, Ix1};
///
/// struct Identity;
///
/// impl Operator<f64> for Identity {
///     fn dim_in(&self) -> usize { 2 }
///     fn dim_out(&self) -> usize { 2 }
///
///     fn apply(
///         &self,
///         x: ArrayView1<'_, f64>,
///     ) -> Result<EvalResult<f64, Ix1>, OperatorError> {
///
///         EvalResult::try_from_parts(
///             x.to_owned(),
///             arr1(&[true, true]),
///         )
///     }
/// }
/// ```
///
/// Fallible operators should return `false` for failed evaluations rather than panicking:
/// ```
/// use casino::*;
/// use ndarray::{arr1, ArrayView1, Ix1};
///
/// struct Reciprocal;
///
/// impl Operator<f64> for Reciprocal {
///     fn dim_in(&self) -> usize { 1 }
///     fn dim_out(&self) -> usize { 1 }
///
///     fn apply(
///         &self,
///         x: ArrayView1<'_, f64>,
///     ) -> Result<EvalResult<f64, Ix1>, OperatorError> {
///
///         if x[0] == 0.0 {
///             return EvalResult::try_from_parts(
///                 arr1(&[0.0]),
///                 arr1(&[false]),
///             );
///         }
///
///         EvalResult::try_from_parts(
///             arr1(&[1.0 / x[0]]),
///             arr1(&[true]),
///         )
///     }
/// }
/// ```
use ndarray::{Array, Array2, ArrayView1, ArrayView2, Dimension, Ix1, Ix2, Zip};
use num_traits::{Float, Zero};

#[derive(thiserror::Error, Debug)]
pub enum OperatorError {
    #[error("shape error in array stacking: {0:?}")]
    ShapeError(#[from] ndarray::ShapeError),
    #[error("shape mismatch in EvalResult creation. Expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        found: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error(
        "shape mismatch in Operator evaluation. Expected dim_out {expected:?}, found {found:?}"
    )]
    InconsistentOutputDim { found: usize, expected: usize },
    #[error("shape mismatch in Operator input. Expected dim_in {expected:?}, found {found:?}")]
    InconsistentInputDim { found: usize, expected: usize },
}

/// Result of operator evaluation on a single Monte Carlo sample
///
/// # Contract
/// - `value.len()` MUST equal `valid.len()`
/// - Each entry in `valid` has 1:1 correspondance with entries in `value`
/// - `valid[i] == false` means `value[i]` must be ignored in downstream reduction
#[derive(Clone)]
pub struct EvalResult<E, D: Dimension> {
    /// Output values from the operator
    value: Array<E, D>,

    /// Validity mask for each output element
    ///
    /// A `true` indicates the corresponding element of `value` is valid and useable in statistical
    /// reduction
    ///
    /// A `false` indicates the corresponding element of `value` is invalid and cannot be used in
    /// statistical reduction
    valid: Array<bool, D>,
}

impl<E, D: Dimension> EvalResult<E, D> {
    pub fn try_from_parts(
        value: Array<E, D>,
        valid: Array<bool, D>,
    ) -> Result<Self, OperatorError> {
        if value.shape() != valid.shape() {
            return Err(OperatorError::ShapeMismatch {
                expected: value.shape().to_vec(),
                found: valid.shape().to_vec(),
            });
        }
        Ok(Self { value, valid })
    }

    pub fn split(self) -> (Array<E, D>, Array<bool, D>) {
        (self.value, self.valid)
    }

    pub fn invalidate_where_nan(&mut self)
    where
        E: 'static + Float,
    {
        let nan_mask = self.value.is_nan();

        // self.valid is true if a value is valid, but the mask is true if it is an nan
        // so we flip the mask
        self.valid = !nan_mask & &self.valid;
    }

    pub fn invalidate_where_infinite(&mut self)
    where
        E: 'static + Float,
    {
        let inf_mask = self.value.is_infinite();

        // self.valid is true if a value is valid, but the mask is true if it is infinite
        // so we flip the mask
        self.valid = !inf_mask & &self.valid;
    }

    pub fn invalidate(&mut self)
    where
        E: 'static + Float,
    {
        self.invalidate_where_nan();
        self.invalidate_where_infinite();
    }

    pub fn all_valid(&self) -> bool {
        Zip::from(self.valid.view()).all(|&v| v)
    }
}

struct WeightedEvalResult<E, D> {
    value: Array<E, D>,
    weight: Array<E, D>,
}

/// Trait representing a deterministic or stochastic transformation
/// applied to Monte Carlo input samples.
///
/// This trait is the core evaluation primitive of the simulation pipeline.
///
/// Each invocation of `apply` maps a single input vector to:
/// - a numeric output vector (`value`)
/// - a validity mask (`valid`)
///
/// # Key design principle
/// This system does NOT use exception-based or result-based failure
/// handling in the Monte Carlo core path.
///
/// Instead:
/// - numerical or domain failures MUST be encoded via `valid = false`
/// - invalid values are excluded from all downstream statistical reduction
///
/// This ensures:
/// - uninterrupted Monte Carlo execution
/// - statistical representation of failure modes
/// - reproducibility of sampling behaviour
///
/// # Validity semantics
/// - `valid[i] == true` → `value[i]` is valid and included in analysis
/// - `valid[i] == false` → `value[i]` is excluded from all reduction steps
///
/// # Implementation contract
/// Implementations MUST guarantee:
/// - `value.shape == valid.shape`
/// - no panics during evaluation
/// - deterministic behaviour for identical inputs
///
/// # Batch execution
/// `apply_batch` provides a reference implementation that:
/// - evaluates rows sequentially
/// - stacks outputs into matrices
/// - preserves validity structure
/// - equivalent to calling apply_checked in row order
///
/// Implementations MAY override `apply_batch` for performance.
///
/// # Performance note
/// Default implementation is not vectorised and is intended for correctness,
/// not throughput. This may be upgraded in future for SIMD or parallel execution
///
///There are two categories of failure:
///
/// 1. Structural failure (hard error)
///    - incorrect input shape
///    - incorrect output shape
///    - violated interface contract
///    → handled via Result::Err
///
/// 2. Numerical or domain failure (soft failure)
///    - NaN results
///    - divergence
///    - invalid physics domain
///    → handled via valid = FALSE
///
/// TODO: Currently the dim_in, dim_out handling is quite frustrating. This could be encoded in the
/// type system `Operator<E, const IN: usize, const OUT: usize>`
///
/// TODO: Batch execution loop is blocking, and the fill logic should be upgraded if
/// parallelisation is required in future
pub trait Operator<E> {
    fn dim_in(&self) -> usize;
    fn dim_out(&self) -> usize;

    fn apply_batch(&self, inputs: ArrayView2<'_, E>) -> Result<EvalResult<E, Ix2>, OperatorError>
    where
        E: Clone + Zero,
    {
        if inputs.shape()[1] != self.dim_in() {
            return Err(OperatorError::InconsistentInputDim {
                found: inputs.shape()[1],
                expected: self.dim_in(),
            });
        }
        let n = inputs.shape()[0];

        let mut values = Array2::zeros((n, self.dim_out()));
        let mut valid = Array2::from_elem((n, self.dim_out()), false);

        for (ii, row) in inputs.rows().into_iter().enumerate() {
            let res = self.apply_checked(row)?;

            values.row_mut(ii).assign(&res.value);
            valid.row_mut(ii).assign(&res.valid);
        }

        Ok(EvalResult::try_from_parts(values, valid)?)
    }

    fn apply_checked(
        &self,
        inputs: ArrayView1<'_, E>,
    ) -> Result<EvalResult<E, Ix1>, OperatorError> {
        if inputs.len() != self.dim_in() {
            return Err(OperatorError::InconsistentInputDim {
                found: inputs.len(),
                expected: self.dim_in(),
            });
        }

        let res = self.apply(inputs)?;

        if res.valid.len() != self.dim_out() {
            return Err(OperatorError::InconsistentOutputDim {
                found: res.valid.len(),
                expected: self.dim_out(),
            });
        }
        Ok(res)
    }

    /// Implementations of `apply` MUST assume:
    /// - input length == dim_in (already validated by apply_checked)
    ///
    /// They MUST guarantee:
    /// - returned EvalResult has:
    ///     - value.len() == dim_out
    ///     - valid.len() == dim_out
    ///
    /// This should be enforced by constructing the `EvalResult` through the `try_from_parts`
    /// interface. It should not be possible for the caller to construct these in any other way.
    ///
    /// TODO: If we parallelise later, this is the atomic function and needs to be thread safe
    fn apply(&self, inputs: ArrayView1<'_, E>) -> Result<EvalResult<E, Ix1>, OperatorError>;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn eval_result_rejects_mismatched_shapes() {
        let value = ndarray::Array2::<f64>::zeros((3, 2));
        let valid = ndarray::Array2::<bool>::from_elem((3, 3), true);

        assert!(EvalResult::try_from_parts(value, valid).is_err());
    }

    #[test]
    fn eval_result_invalidates_nans() {
        let value =
            ndarray::Array2::<f64>::from_shape_vec((2, 2), vec![1., f64::NAN, 3., 4.]).unwrap();
        let valid = ndarray::Array2::<bool>::from_elem((2, 2), true);

        let res = EvalResult::try_from_parts(value, valid);
        assert!(res.is_ok());

        let mut res = res.unwrap();

        res.invalidate_where_nan();

        let (v, m) = res.split();

        let expected =
            ndarray::Array2::<bool>::from_shape_vec((2, 2), vec![true, false, true, true]).unwrap();

        assert_eq!(expected, m);
    }

    #[test]
    fn eval_result_invalidates_infinite() {
        let value = ndarray::Array2::<f64>::from_shape_vec((2, 2), vec![1., f64::INFINITY, 3., 4.])
            .unwrap();
        let valid = ndarray::Array2::<bool>::from_elem((2, 2), true);

        let res = EvalResult::try_from_parts(value, valid);
        assert!(res.is_ok());

        let mut res = res.unwrap();

        res.invalidate_where_infinite();

        let (v, m) = res.split();

        let expected =
            ndarray::Array2::<bool>::from_shape_vec((2, 2), vec![true, false, true, true]).unwrap();

        assert_eq!(expected, m);
    }

    #[test]
    fn all_valid_returns_true_when_all_valid() {
        let value = ndarray::Array2::<f64>::from_shape_vec((2, 2), vec![1., 2., 3., 4.]).unwrap();
        let valid = ndarray::Array2::<bool>::from_elem((2, 2), true);

        let res = EvalResult::try_from_parts(value, valid);
        assert!(res.is_ok());

        let mut res = res.unwrap();

        assert!(res.all_valid());
    }

    #[test]
    fn all_valid_returns_false_when_not_all_valid() {
        let value = ndarray::Array2::<f64>::from_shape_vec((2, 2), vec![1., 2., 3., 4.]).unwrap();
        let valid =
            ndarray::Array2::<bool>::from_shape_vec((2, 2), vec![true, false, true, true]).unwrap();

        let res = EvalResult::try_from_parts(value, valid);
        assert!(res.is_ok());

        let mut res = res.unwrap();

        assert!(!res.all_valid());
    }

    #[test]
    fn eval_result_preserves_shapes() {
        let value = ndarray::Array2::<f64>::zeros((5, 2));
        let valid = ndarray::Array2::<bool>::from_elem((5, 2), true);

        let res = EvalResult::try_from_parts(value.clone(), valid.clone()).unwrap();
        let (v, m) = res.split();

        assert_eq!(v.shape(), value.shape());
        assert_eq!(m.shape(), valid.shape());
    }

    #[test]
    fn eval_result_roundtrip_split() {
        let value = Array2::<f64>::ones((4, 2));
        let valid = ndarray::Array::from_elem((4, 2), true);

        let res = EvalResult::try_from_parts(value.clone(), valid.clone()).unwrap();
        let (v1, v2) = res.split();

        assert_eq!(v1, value);
        assert_eq!(v2, valid);
    }

    struct IdentityOp;

    impl<E: Clone + num_traits::Zero> Operator<E> for IdentityOp {
        fn dim_in(&self) -> usize {
            2
        }
        fn dim_out(&self) -> usize {
            2
        }

        fn apply(
            &self,
            x: ndarray::ArrayView1<E>,
        ) -> Result<EvalResult<E, Ix1>, super::OperatorError> {
            let valid = ndarray::Array1::from(vec![true; 2]);
            Ok(EvalResult::try_from_parts(x.to_owned(), valid).unwrap())
        }
    }

    #[test]
    fn batch_matches_rowwise() {
        let op = IdentityOp;

        let inputs =
            ndarray::Array2::<f64>::from_shape_vec((3, 2), vec![1., 2., 3., 4., 5., 6.]).unwrap();

        let batch = op.apply_batch(inputs.view()).unwrap();

        assert_eq!(batch.value.shape(), &[3, 2]);
    }
}
