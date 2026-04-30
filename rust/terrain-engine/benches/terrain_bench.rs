use criterion::{black_box, criterion_group, criterion_main, Criterion};
use terrain_engine::{GridTerrain, TerrainQuery};

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

criterion_group!(benches, bench_height, bench_gradient_and_los);
criterion_main!(benches);
