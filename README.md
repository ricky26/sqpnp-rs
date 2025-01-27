# sqpnp-rs

[![crate](https://img.shields.io/crates/v/sqpnp.svg)](https://crates.io/crates/sqpnp)

`sqpnp-rs` is a pure Rust implementation of the SQPnP perspective-n-point
algorithm, based on the [C++ implementation](https://github.com/terzakig/sqpnp).

## Usage

```rust
let p3d = [
    // 3D object points
    vec3(1.0, 1.0, 1.0),
];
let p2d = [
    // projected points
    vec2(-0.5, -0.5),
];

let mut solver = Solver::<DefaultParameters>::new();
if (solver.solve(&p3d, &p2d, None)) {
    let solution = solver.best_solution().unwrap();
    let r = solution.rotation_matrix();
    let t = solution.translation();

    // ...
}
```

## See Also

There is another pure-Rust implementation of SQPnP, here:
[powei-lin/sqpnp_simple](https://github.com/powei-lin/sqpnp_simple).

## License

sqpnp-rs is permissively licensed under either the [MIT License](LICENSE-MIT) or
the [Apache 2.0 License](LICENSE-APACHE).
