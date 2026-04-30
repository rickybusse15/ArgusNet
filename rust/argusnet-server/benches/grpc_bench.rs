use argusnet_server::{load_tracker_config, ServeArgs};
use criterion::{criterion_group, criterion_main, Criterion};

fn args() -> ServeArgs {
    ServeArgs {
        listen: "127.0.0.1:50051".to_string(),
        config: None,
        min_observations: None,
        max_stale_steps: None,
        min_confidence: None,
        max_bearing_std_rad: None,
        max_timestamp_skew_s: None,
        min_intersection_angle_deg: None,
        data_association_mode: None,
        cv_process_accel_std: None,
        ct_process_accel_std: None,
        ct_turn_rate_std: None,
        innovation_window: None,
        innovation_scale_factor: None,
        innovation_max_scale: None,
        adaptive_measurement_noise: None,
        chi_squared_gate_threshold: None,
        cluster_distance_threshold_m: None,
        near_parallel_rejection_angle_deg: None,
        confirmation_m: None,
        confirmation_n: None,
        max_coast_frames: None,
        max_coast_seconds: None,
        min_quality_score: None,
    }
}

fn bench_config_load(c: &mut Criterion) {
    let args = args();
    c.bench_function("server_config_load_default", |b| {
        b.iter(|| load_tracker_config(&args).unwrap())
    });
}

criterion_group!(benches, bench_config_load);
criterion_main!(benches);
