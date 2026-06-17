# montecore

## Montecore

A Rust library for uncertainty propagation using Monte Carlo simulation.

Montecore provides:

* Gaussian Monte Carlo sampling
* Latin Hypercube Sampling (LHS)
* Adaptive convergence control
* Correlated and independent input models
* Streaming statistics (mean, variance, covariance)
* Deterministic and reproducible execution
* Batch-oriented execution suitable for future parallelisation

---

## Philosophy

Montecore is designed for scientific and engineering uncertainty analysis.

The library separates uncertainty propagation into three independent concepts:

1. **Input model** — describes uncertain inputs.
2. **Sampling strategy** — describes how the input space is explored.
3. **Operator** — the physical or numerical model being evaluated.

This separation allows the same operator to be evaluated using different sampling methods without modifying user code.

---

## Installation

```toml
[dependencies]
montecore = "0.2"
```

---

## Quick Start

```rust
use montecore::*;
use ndarray::{arr1, ArrayView1, Ix1};

struct BeamDeflection;

impl Operator<f64> for BeamDeflection {
    fn dim_in(&self) -> usize {
        2
    }

    fn dim_out(&self) -> usize {
        1
    }

    fn apply(
        &self,
        x: ArrayView1<'_, f64>,
    ) -> Result<EvalResult<f64, Ix1>, OperatorError> {

        let force = x[0];
        let stiffness = x[1];

        let deflection = force / stiffness;

        EvalResult::try_from_parts(
            arr1(&[deflection]),
            arr1(&[true]),
        )
    }
}

let means = arr1(&[100.0, 10.0]);
let stddev = arr1(&[5.0, 1.0]);

let result = MonteCarlo::run(
    InputSpec::Independent {
        means: means.view(),
        marginal_scale: stddev.view(),
    },
    BeamDeflection,
    SamplingMethod::Gaussian,
    MonteCarloOptions {
        seed: 42,
        batch_size: 1024,
        min_samples: 10_000,
        max_samples: 1_000_000,
        rel_tol: 1e-3,
    },
);
```

The returned result contains estimated output statistics:

```rust
result.statistics.mean
result.statistics.covariance
```

---

## Input Models

Montecore supports both independent and correlated inputs.

### Independent Inputs

```rust
InputSpec::Independent {
    means: means.view(),
    marginal_scale: marginal_scale.view(),
};
```

Each variable is sampled independently.

---

### Correlated Inputs

```rust
InputSpec::Correlated {
    means: means.view(),
    covariance: covariance.view(),
}
```

The covariance matrix describes correlations between inputs.

For Gaussian sampling, Montecore computes a Cholesky factorisation internally and generates correlated samples automatically.

---

## Sampling Methods

Sampling strategy determines how the input space is explored.

### Gaussian Sampling

```rust
SamplingMethod::Gaussian
```

Traditional Monte Carlo sampling using independent Gaussian random variables.

Use when:

* Input uncertainty is naturally Gaussian
* Statistical independence is important
* Direct comparison with analytical Gaussian uncertainty propagation is desired

---

### Latin Hypercube Sampling

```rust
SamplingMethod::LatinHypercube
```

Stratified sampling method

Use when:

* Model evaluations are expensive
* Inputs are independent
* Improved convergence is desired

---

## Choosing a Sampling Method

| Method          | Characteristics                         |
| --------------- | --------------------------------------- |
| Gaussian        | Random, unbiased, familiar              |
| Latin Hypercube | Stratified, faster convergence          |

For most engineering uncertainty propagation problems:

```rust
SamplingMethod::LatinHypercube
```

is a good default.

---

## The Operator Trait

User code implements the measurement model through the `Operator` trait.

```rust
pub trait Operator<E> {
    fn dim_in(&self) -> usize;

    fn dim_out(&self) -> usize;

    fn apply(&self, inputs: ArrayView1<'_, E>) -> Result<EvalResult<E, Ix1>, OperatorError>;
}
```

---

## Validity Masks

Montecore uses validity masks rather than exceptions to handle numerical failures.

Each output value has a corresponding validity flag.

```rust
EvalResult::try_from_parts(
    value,
    valid,
)
```

where:

```
valid[i] == true
```

means the output is statistically valid.

and

```
valid[i] == false
```

means the output should be excluded from all downstream statistics.

---

### Example

```rust
use montecore::*;
use ndarray::{ArrayView1, Ix1, arr1};

struct Reciprocal;

impl Operator<f64> for Reciprocal {
    fn dim_in(&self) -> usize {
        1
    }

    fn dim_out(&self) -> usize {
        1
    }

    fn apply(&self, x: ArrayView1<'_, f64>) -> Result<EvalResult<f64, Ix1>, OperatorError> {
        if x[0] == 0.0 {
            return EvalResult::try_from_parts(arr1(&[0.0]), arr1(&[false]));
        }

        EvalResult::try_from_parts(arr1(&[1.0 / x[0]]), arr1(&[true]))
    }
}
```

This allows Monte Carlo simulations to continue even when individual evaluations fail.

---

## Adaptive Convergence

Montecore automatically monitors convergence during simulation.

The engine terminates when either:

```
maximum relative uncertainty < rel_tol
```

or

```
sample count >= max_samples
```

while also enforcing:

```
sample count >= min_samples
```

before convergence checks begin.

This allows simulations to stop early once sufficient statistical precision has been achieved.

---

## Reproducibility

All sampling methods are deterministic.

```rust
MonteCarloOptions {
    seed: 42,
    ..
}
```

Using the same:

* input model
* operator
* sampling method
* seed

will always produce identical results.

---

## Statistics

Montecore computes statistics incrementally using streaming estimators.

The following quantities are available:

```rust
SummaryStatistics {
    expectation,
    covariance,
}
```

where:

```
expectation
```

is the estimated mean output vector and

```
covariance
```

is the estimated covariance matrix.

Streaming accumulation avoids storing all Monte Carlo samples in memory.

---

## Numerical Robustness

Montecore uses:

* Welford-style online moment accumulation
* Chan-style batch merging
* Validity-aware covariance estimation

to provide stable statistics for large simulations.

The library never stores the full Monte Carlo history.

Memory usage scales with output dimension rather than sample count.

---

License: MIT
