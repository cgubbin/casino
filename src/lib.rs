mod core;
mod input;
mod output;
mod stats;

pub type Result<T> = ::std::result::Result<T, Box<dyn ::std::error::Error>>;
