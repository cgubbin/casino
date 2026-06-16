use ndarray::{Array1, Array2};

pub struct SummaryStatistics<E> {
    pub mean: Array1<E>,
    pub covariance: Array2<E>,
}
