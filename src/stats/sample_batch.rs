use ndarray::Array2;

pub struct SampleBatch<E> {
    pub values: Array2<E>, // (n_samples, dim)
}
