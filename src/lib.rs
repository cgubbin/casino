use ndarray::Array1;
use ndarray_rand::rand::Rng;

mod builder;
mod core;
mod input;
mod output;
mod stats;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;

pub trait Model<E> {
    fn apply(&self, inputs: Array1<E>) -> ::std::result::Result<Array1<E>, Box<dyn ::std::error::Error>>;
}

pub use builder::Builder;
