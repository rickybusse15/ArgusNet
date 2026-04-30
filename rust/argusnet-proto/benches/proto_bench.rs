use argusnet_proto::pb::{BearingObservation, IngestFrameRequest, NodeState, TruthState, Vector3};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use prost::Message;

fn vec3(x: f64, y: f64, z: f64) -> Option<Vector3> {
    Some(Vector3 {
        x_m: x,
        y_m: y,
        z_m: z,
    })
}

fn request(observation_count: usize) -> IngestFrameRequest {
    let node_states = (0..8)
        .map(|i| NodeState {
            node_id: format!("node-{i}"),
            position: vec3(i as f64 * 10.0, 0.0, 20.0),
            velocity: vec3(0.0, 0.0, 0.0),
            is_mobile: true,
            timestamp_s: 1.0,
            health: 1.0,
            sensor_type: "optical".to_string(),
            fov_half_angle_deg: 180.0,
            max_range_m: 2000.0,
        })
        .collect();
    let observations = (0..observation_count)
        .map(|i| BearingObservation {
            node_id: format!("node-{}", i % 8),
            target_id: format!("target-{}", i % 4),
            origin: vec3(i as f64, 0.0, 20.0),
            direction: vec3(1.0, 0.0, -0.1),
            bearing_std_rad: 0.02,
            timestamp_s: 1.0,
            confidence: 0.95,
        })
        .collect();
    let truths = (0..4)
        .map(|i| TruthState {
            target_id: format!("target-{i}"),
            position: vec3(100.0 + i as f64, 50.0, 10.0),
            velocity: vec3(1.0, 0.0, 0.0),
            timestamp_s: 1.0,
        })
        .collect();
    IngestFrameRequest {
        timestamp_s: 1.0,
        node_states,
        observations,
        truths,
    }
}

fn bench_proto_roundtrip(c: &mut Criterion) {
    for count in [16_usize, 64, 256] {
        let req = request(count);
        c.bench_function(&format!("ingest_frame_request_roundtrip_{count}obs"), |b| {
            b.iter(|| {
                let mut bytes = Vec::new();
                black_box(&req).encode(&mut bytes).unwrap();
                IngestFrameRequest::decode(black_box(bytes.as_slice())).unwrap()
            })
        });
    }
}

criterion_group!(benches, bench_proto_roundtrip);
criterion_main!(benches);
