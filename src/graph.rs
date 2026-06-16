pub enum Node<'a, E> {
    Input(InputNode<'a, E>),
    Sampler(SamplerNode<'a, E>),
    Model(ModelNode),
    Reduction(ReductionNode),
}

pub struct Pipeline<'a, E, M> {
    nodes: Vec<Node<'a, E>>,
    model: M,
}

impl<'a, E: Float, R: Rng, M> Pipeline<'a, E, M> {
    pub fn run(&self, rng: &mut R, trials: usize) -> Result<Output<E>, Error> {
        let samples = self.sample_inputs(rng, trials)?;
        let outputs = self.model.apply_batch(samples)?;
        Ok(self.reduce(outputs))
    }
}
