use ndarray::{Ix2, LinalgScalar, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::controller::StopReason;
use crate::operator::EvalResult;
use crate::stats::{FinalStatistics, RunningStats};

pub trait McObserver<E> {
    type State;
    type Output;

    fn update_batch(&mut self, batch: EvalResult<E, Ix2>);

    fn merge(&mut self, other: &Self::State);

    fn state(&self) -> &Self::State;

    fn finalize(&self) -> Self::Output;
}

pub struct StatsObserver<E> {
    stats: RunningStats<E>,
}

impl<E> StatsObserver<E>
where
    E: Float + LinalgScalar + ScalarOperand + FromPrimitive,
{
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            stats: RunningStats::new(dim),
        }
    }
}

impl<E> McObserver<E> for StatsObserver<E>
where
    E: Float + LinalgScalar + ScalarOperand + FromPrimitive + std::fmt::Debug,
{
    type State = RunningStats<E>;
    type Output = FinalStatistics<E>;

    fn update_batch(&mut self, batch: EvalResult<E, Ix2>) {
        let (xs, valid) = batch.split();
        self.stats.update_batch(xs.view(), valid.view());
    }

    fn merge(&mut self, other: &Self::State) {
        self.stats.merge(other);
    }

    fn state(&self) -> &Self::State {
        &self.stats
    }

    fn finalize(&self) -> Self::Output {
        self.stats.finalize()
    }
}

#[cfg(test)]
mod observer_tests {
    use super::*;

    #[test]
    fn observer_accumulates_batches() {
        let mut obs = StatsObserver {
            stats: RunningStats::<f64>::new(2),
        };

        let batch = EvalResult::try_from_parts(
            ndarray::Array2::from_shape_vec((2, 2), vec![1., 2., 3., 4.]).unwrap(),
            ndarray::Array2::from_elem((2, 2), true),
        )
        .unwrap();

        obs.update_batch(batch.clone());
        obs.update_batch(batch);

        let state = obs.state();
        let mean = state.finalize_mean();

        assert!(mean[0] > 0.0);
    }
}
