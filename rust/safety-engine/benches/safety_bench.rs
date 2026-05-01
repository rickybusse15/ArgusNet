use criterion::{black_box, criterion_group, criterion_main, Criterion};
use safety_engine::limits::DronePhysicalLimits;
use safety_engine::validator::{validate_constraints, DroneCommandedState, DroneObservedState};
use terrain_engine::{FlatTerrain, TerrainBounds};

fn terrain() -> FlatTerrain {
    FlatTerrain::new(
        0.0,
        TerrainBounds::new(-1000.0, 1000.0, -1000.0, 1000.0, 0.0, 0.0),
    )
}

fn nominal_state() -> DroneCommandedState {
    DroneCommandedState {
        position_m: [100.0, 100.0, 80.0],
        velocity_mps: [20.0, 0.0, 0.0],
        acceleration_mps2: [0.5, 0.0, 0.0],
        gimbal_pitch_rad: 0.3,
        gimbal_yaw_offset_rad: 0.0,
    }
}

fn bench_constraints(c: &mut Criterion) {
    let limits = DronePhysicalLimits::interceptor_default();
    let terrain = terrain();
    let peers = vec![DroneObservedState {
        drone_id: "peer-0".to_string(),
        position_m: [200.0, 200.0, 85.0],
    }];
    c.bench_function("validate_constraints_nominal", |b| {
        b.iter(|| {
            validate_constraints(
                black_box(&limits),
                black_box(&nominal_state()),
                black_box(&terrain),
                black_box(&peers),
                black_box(0.65),
                black_box(-60.0),
            )
        })
    });
}

criterion_group!(benches, bench_constraints);
criterion_main!(benches);
