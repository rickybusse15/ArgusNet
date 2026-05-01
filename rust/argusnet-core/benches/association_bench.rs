use argusnet_core::{
    AssociationMode, BearingObservation, NodeState, TrackerConfig, TrackingEngine, TruthState,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::Vector3;

fn nodes(count: usize) -> Vec<NodeState> {
    (0..count)
        .map(|i| NodeState {
            node_id: format!("node-{i}"),
            position: Vector3::new(
                (i as f64 * 45.0) - 300.0,
                ((i % 9) as f64 * 35.0) - 150.0,
                35.0,
            ),
            velocity: Vector3::zeros(),
            is_mobile: true,
            timestamp_s: 0.0,
            health: 1.0,
        })
        .collect()
}

fn observations(
    nodes: &[NodeState],
    targets: &[TruthState],
    timestamp_s: f64,
) -> Vec<BearingObservation> {
    let mut out = Vec::with_capacity(nodes.len() * targets.len());
    for target in targets {
        for node in nodes.iter().take(8) {
            out.push(BearingObservation {
                node_id: node.node_id.clone(),
                target_id: target.target_id.clone(),
                origin: node.position,
                direction: (target.position - node.position).normalize(),
                bearing_std_rad: 0.025,
                timestamp_s,
                confidence: 0.9,
            });
        }
    }
    out
}

fn truths(count: usize, timestamp_s: f64) -> Vec<TruthState> {
    (0..count)
        .map(|i| TruthState {
            target_id: format!("target-{i}"),
            position: Vector3::new(80.0 + i as f64 * 45.0, 35.0 - i as f64 * 20.0, 12.0),
            velocity: Vector3::new(1.0, 0.0, 0.0),
            timestamp_s,
        })
        .collect()
}

fn seeded_engine(mode: AssociationMode) -> TrackingEngine {
    let mut config = TrackerConfig::default();
    config.data_association_mode = mode;
    config.min_observations = 2;
    TrackingEngine::new(config).unwrap()
}

fn bench_mode(c: &mut Criterion, mode: AssociationMode, name: &str) {
    c.bench_function(name, |b| {
        b.iter_batched(
            || {
                let mut engine = seeded_engine(mode.clone());
                let ns = nodes(32);
                let ts0 = truths(4, 0.0);
                let obs0 = observations(&ns, &ts0, 0.0);
                engine.ingest_frame(0.0, ns.clone(), obs0, ts0);
                let ts1 = truths(4, 0.25);
                let obs1 = observations(&ns, &ts1, 0.25);
                (engine, ns, obs1, ts1)
            },
            |(mut engine, ns, obs, ts)| engine.ingest_frame(black_box(0.25), ns, obs, ts),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_association_modes(c: &mut Criterion) {
    bench_mode(c, AssociationMode::GNN, "association_gnn");
    bench_mode(c, AssociationMode::JPDA, "association_jpda");
}

criterion_group!(benches, bench_association_modes);
criterion_main!(benches);
