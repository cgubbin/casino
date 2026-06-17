use ndarray::{Array1, Array2};

#[derive(Debug)]
pub struct FinalStatistics<E> {
    pub summary: SummaryStatistics<E>,
    pub diagnostics: StatisticalDiagnostics<E>,
}

#[derive(Debug)]
pub struct SummaryStatistics<E> {
    pub mean: Array1<E>,
    pub covariance: Array2<E>,
}

#[derive(Debug)]
pub struct StatisticalDiagnostics<E> {
    pub mean_standard_error: Array1<E>,
    pub std_standard_error: Array1<E>,
    pub valid_fraction: Array1<E>,
}
