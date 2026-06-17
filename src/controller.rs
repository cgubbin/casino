use ndarray::{Array1, LinalgScalar, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::stats::{RunningStats, SummaryStatistics};

pub enum StopDecision {
    Continue,
    Stop { reason: StopReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    RelativeToleranceReached,
    MaxSamplesReached,
}

pub enum ConvergenceCriterion<E> {
    AbsoluteMeanError { tol: E },
    RelativeMeanError { rel_tol: E },
}

pub struct ConvergenceStatus<E> {
    pub converged: bool,
    pub max_se: Array1<E>,
}

pub struct AdaptiveController<E> {
    min_samples: usize,
    max_samples: usize,
    criterion: ConvergenceCriterion<E>,
    confidence: E, // z-score (e.g. 1.96)
}

impl<E> AdaptiveController<E>
where
    E: Float + LinalgScalar,
{
    pub fn new(min_samples: usize, max_samples: usize, rel_tol: E) -> Self {
        Self {
            min_samples,
            max_samples,
            criterion: ConvergenceCriterion::RelativeMeanError { rel_tol },
            confidence: E::from(1.96).unwrap(),
        }
    }
}

pub(crate) trait McController<E> {
    fn should_stop(&self, stats: &RunningStats<E>) -> StopDecision;
}

impl<E> McController<E> for AdaptiveController<E>
where
    E: Float + LinalgScalar + ScalarOperand + FromPrimitive,
{
    fn should_stop(&self, stats: &RunningStats<E>) -> StopDecision {
        let status = stats.check_convergence(&self.criterion);

        if status.converged && stats.count() >= self.min_samples {
            return StopDecision::Stop {
                reason: StopReason::RelativeToleranceReached,
            };
        }

        if stats.count() >= self.max_samples {
            return StopDecision::Stop {
                reason: StopReason::MaxSamplesReached,
            };
        }

        StopDecision::Continue
    }
}
