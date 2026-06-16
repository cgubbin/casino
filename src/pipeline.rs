use ndarray::{LinalgScalar, ScalarOperand};
use num_traits::Float;

use crate::input::{CompiledInput, InputModel, SamplingStrategy};
use crate::operator::Operator;
use crate::stats::{RunningStats, SampleBatch, SummaryStatistics};

pub fn run_pipeline<E, S, M, O>(
    compiled: &CompiledInput<E, S, M>,
    operator: &O,
    n: usize,
) -> SummaryStatistics<E>
where
    E: Float + LinalgScalar + ScalarOperand + num_traits::FromPrimitive,
    S: SamplingStrategy<E>,
    M: InputModel<E, Space = S::Space>,
    O: Operator<E>,
{
    let mut state = compiled.init_state();
    let z = compiled.sample(&mut state, n);

    let samples = compiled.model.apply(z);

    let batch = SampleBatch { values: samples };

    let mut stats = RunningStats::new(operator.dim_out());

    stats.ingest_batch(batch, operator);

    SummaryStatistics {
        mean: stats.finalize_mean(),
        covariance: stats.covariance(),
    }
}
