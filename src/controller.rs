use ndarray::{LinalgScalar, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::stats::{RunningStats, SummaryStatistics};

pub struct AdaptiveController<E> {
    min_samples: usize,
    max_samples: usize,
    rel_tol: E,
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
            rel_tol,
            confidence: E::from(1.96).unwrap(),
        }
    }
}

pub(crate) trait McController<E> {
    fn should_stop(&self, stats: &RunningStats<E>) -> bool;
}

impl<E> McController<E> for AdaptiveController<E>
where
    E: Float + LinalgScalar + ScalarOperand + FromPrimitive,
{
    fn should_stop(&self, stats: &RunningStats<E>) -> bool {
        if stats.count() < self.min_samples {
            return false;
        }

        let SummaryStatistics { mean, covariance } = stats.finalize();

        let n = E::from(stats.count()).unwrap();

        let stderr = covariance.diag().mapv(|v| (v / n).sqrt());

        let rel_err = (&stderr / &mean.mapv(|x| x.abs() + E::from(1e-12).unwrap()));

        rel_err.iter().all(|e| *e < self.rel_tol) || stats.count() >= self.max_samples
    }
}
