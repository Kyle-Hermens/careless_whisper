use crate::model;
use tch::{Device, Tensor};

#[test]
fn test_low_dim_sinusoids_1_4() {
    let output = model::sinusoids(1.0, 4);

    let expected: Tensor = ndarray::arr2(&[[0.0, 0.0, 1.0, 1.0]])
        .try_into()
        .expect("Failed to convert 2d array to Tensor");

    assert_eq!(expected, output);
}

#[test]
fn test_low_dim_sinusoids_2_6() {
    let output = model::sinusoids(2.0, 6);

    let expected: Tensor = ndarray::arr2(&[
        [
            0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        ],
        [
            8.4147e-01, 9.9998e-03, 1.0000e-04, 5.4030e-01, 9.9995e-01, 1.0000e+00,
        ],
    ])
    .try_into()
    .expect("Failed to convert 2d array to Tensor");
    println!("expected: {expected}");
    println!("output: {output}");

    assert!(
        tensors_roughly_equal(&expected, &output),
        "Expected {expected} was not equal to output {output}"
    );
}

fn tensors_roughly_equal(x: &Tensor, y: &Tensor) -> bool {
    x.allclose(&y, 1e-05, 1e-08, true)
}

#[test]
fn do_tensor_clones_work() {
    let expected: Tensor = ndarray::arr2(&[
        [
            0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        ],
        [
            8.4147e-01, 9.9998e-03, 1.0000e-04, 5.4030e-01, 9.9995e-01, 1.0000e+00,
        ],
    ])
    .try_into()
    .expect("Failed to convert 2d array to Tensor");
    // let alternate: Tensor = ndarray::arr2(&[
    //     [
    //         0.5000e+00, 0.1000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
    //     ],
    //     [
    //         8.4147e-01, 9.9998e-03, 1.0000e-04, 5.4030e-01, 9.9995e-01, 1.0000e+00,
    //     ],
    // ])
    // .try_into()
    // .expect("Failed to convert 2d array to Tensor");
    //
    // let output: Tensor = expected.clone(&alternate);
    let output = expected.copy();

    println!("expected: {expected}");
    println!("output: {output}");

    assert!(
        tensors_roughly_equal(&expected, &output),
        "Expected {expected} was not equal to output {output}"
    );
}
