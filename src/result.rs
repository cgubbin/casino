use crate::controller::StopReason;
use crate::stats::{StatisticalDiagnostics, SummaryStatistics};
use ndarray::Array1;

#[derive(Debug)]
pub struct MonteCarloResult<E> {
    pub statistics: SummaryStatistics<E>,
    pub diagnostics: StatisticalDiagnostics<E>,
    pub stop_reason: StopReason,
    pub total_samples: usize,
}
