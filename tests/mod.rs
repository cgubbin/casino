mod linear;

use ndarray::{Array1, Array2};

pub trait AnalyticalProblem<E> {
    fn true_mean(&self) -> ndarray::Array1<E>;
    fn true_covariance(&self) -> ndarray::Array2<E>;
}
