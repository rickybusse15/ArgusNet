use argusnet_core::{
    fuse_bearing_cluster, AssociationMode, BearingObservation, NodeState, TrackerConfig,
    TrackingEngine, TruthState,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::Vector3;

fn node(node_id: usize, x: f64, y: f64) -> NodeState {
    NodeState {
        node_id: format!("node-{node_id}"),
        position: Vector3::new(x, y, 25.0),
        velocity: Vector3::zeros(),
        is_mobile: true,
        timestamp_s: 0.0,
        health: 1.0,
    }
}

fn observation(
    node: &NodeState,
    target_id: &str,
    target: Vector3<f64>,
    timestamp_s: f64,
) -> BearingObservation {
    let direction = (target - node.position).normalize();
    BearingObservation {
        node_id: node.node_id.clone(),
        target_id: target_id.to_string(),
        origin: node.position,
        direction,
        bearing_std_rad: 0.02,
        timestamp_s,
        confidence: 0.95,
    }
}

fn frame_inputs(
    node_count: usize,
    observation_count: usize,
    timestamp_s: f64,
) -> (Vec<NodeState>, Vec<BearingObservation>, Vec<TruthState>) {
    let target = Vector3::new(120.0, 80.0, 10.0);
    let nodes: Vec<_> = (0..node_count)
        .map(|i| {
            node(
                i,
                (i as f64 * 37.0) - 180.0,
                ((i % 7) as f64 * 41.0) - 120.0,
            )
        })
        .collect();
    let observations = (0..observation_count)
        .map(|i| observation(&nodes[i % nodes.len()], "target-0", target, timestamp_s))
        .collect();
    let truths = vec![TruthState {
        target_id: "target-0".to_string(),
        position: target,
        velocity: Vector3::new(1.0, 0.0, 0.0),
        timestamp_s,
    }];
    (nodes, observations, truths)
}

fn bench_fusion(c: &mut Criterion) {
    let (_, observations, _) = frame_inputs(16, 16, 0.0);
    c.bench_function("fuse_bearing_cluster_16", |b| {
        b.iter(|| fuse_bearing_cluster(black_box(&observations)).unwrap())
    });
}

fn bench_ingest(c: &mut Criterion) {
    for (nodes, observations) in [(8, 4), (32, 16), (128, 64)] {
        c.bench_function(&format!("ingest_frame_{nodes}n_{observations}o"), |b| {
            b.iter_batched(
                || {
                    let mut config = TrackerConfig::default();
                    config.data_association_mode = AssociationMode::Labeled;
                    let engine = TrackingEngine::new(config).unwrap();
                    (engine, frame_inputs(nodes, observations, 1.0))
                },
                |(mut engine, (node_states, obs, truths))| {
                    engine.ingest_frame(black_box(1.0), node_states, obs, truths)
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(benches, bench_fusion, bench_ingest);
criterion_main!(benches);
