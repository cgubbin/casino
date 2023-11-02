use ndarray::Array1;

struct Output<E> {
    results: Vec<Array1<E>>,
}

impl<E> Output<E> {
    fn add_output(&mut self, latest: Array1<E>) {
        self.results.push(latest);
    }
}
