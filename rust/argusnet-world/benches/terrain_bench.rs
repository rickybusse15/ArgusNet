use argusnet_world::{GridTerrain, TerrainQuery};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn terrain() -> GridTerrain {
    let cols = 513;
    let rows = 513;
    let cell = 4.0;
    let mut heights = Vec::with_capacity(cols * rows);
    for y in 0..rows {
        for x in 0..cols {
            let xf = x as f64 * cell;
            let yf = y as f64 * cell;
            heights.push((xf / 80.0).sin() * 12.0 + (yf / 110.0).cos() * 8.0);
        }
    }
    GridTerrain::new(heights, cols, rows, cell, -1024.0, -1024.0, 0.0)
}

fn bench_height(c: &mut Criterion) {
    let t = terrain();
    c.bench_function("grid_height_at", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for i in 0..4096 {
                let x = -900.0 + (i % 256) as f64 * 7.0;
                let y = -900.0 + (i / 256) as f64 * 7.0;
                acc += t.height_at(black_box(x), black_box(y));
            }
            acc
        })
    });
}

fn bench_gradient_and_los(c: &mut Criterion) {
    let t = terrain();
    c.bench_function("grid_gradient_at", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for i in 0..2048 {
                let g = t.gradient_at(
                    -800.0 + (i % 128) as f64 * 8.0,
                    -750.0 + (i / 128) as f64 * 8.0,
                    2.0,
                );
                acc += g[0] + g[1];
            }
            acc
        })
    });
    c.bench_function("grid_los_raycast", |b| {
        b.iter(|| {
            t.los_raycast(
                black_box([-900.0, -900.0, 120.0]),
                black_box([900.0, 900.0, 120.0]),
                1.0,
            )
        })
    });
}

fn bench_tiled_world(c: &mut Criterion) {
    use argusnet_world::{obstacles::Bounds2d, tiled::TiledTerrain, visibility};
    let bounds = Bounds2d::new(0.0, 512.0, 0.0, 512.0);
    let grid: Vec<Vec<f64>> = (0..129)
        .map(|y| {
            (0..129)
                .map(|x| 20.0 + (x as f64 / 8.0).sin() * (y as f64 / 8.0).cos() * 10.0)
                .collect()
        })
        .collect();
    let world = TiledTerrain::from_grid(bounds, &grid, 4.0, 32, 0.0, &[4.0, 8.0]).unwrap();
    c.bench_function("tiled_height_4096", |b| {
        b.iter(|| {
            (0..4096)
                .map(|i| world.height_at(black_box((i % 128) as f64 * 4.0), (i / 128) as f64 * 4.0))
                .sum::<f64>()
        })
    });
    c.bench_function("tiled_dda_los", |b| {
        b.iter(|| {
            visibility::terrain_intersection(
                &world,
                black_box([500.0, 500.0, 22.0]),
                [1.0, 1.0, 22.0],
                1.0,
            )
        })
    });
}
criterion_group!(
    benches,
    bench_height,
    bench_gradient_and_los,
    bench_tiled_world
);
criterion_main!(benches);
