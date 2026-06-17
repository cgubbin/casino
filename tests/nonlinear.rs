// use crate::Analyticalproblem;

// #[test]
// fn lhs_reduces_variance_vs_gaussian() {
//     use casino::*;
//     use ndarray::arr1;

//     fn run(strategy: SamplingMethod) -> f64 {
//         let means = arr1(&[1.0, 2.0]);
//         let marginal_scale = arr1(&[0.5, 0.5]);
//         let input = InputSpec::Independent {
//             means: means.view(),
//             marginal_scale: marginal_scale.view(),
//         };

//         let result = MonteCarlo::run(
//             input,
//             QuadraticModel,
//             strategy,
//             MonteCarloOptions {
//                 seed: 42,
//                 batch_size: 512,
//                 min_samples: 20_000,
//                 max_samples: 200_000,
//                 rel_tol: 1e-3,
//             },
//         )
//         .unwrap();

//         result.statistics.covariance[[0, 0]]
//     }

//     let var_gaussian = run(SamplingMethod::Gaussian);
//     let var_lhs = run(SamplingMethod::LatinHypercube);

//     assert!(var_lhs < var_gaussian);
// }
