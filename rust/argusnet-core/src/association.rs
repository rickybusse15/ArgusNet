//! Data association helpers for unlabeled multi-target tracking.

use nalgebra::{Matrix3, Vector3};
use smallvec::SmallVec;
use std::collections::HashMap;

use crate::{fuse_bearing_cluster, BearingObservation, ManagedTrack};

/// Result of associating a cluster of observations to a track.
#[derive(Clone, Debug)]
pub struct Assignment {
    pub cluster_index: usize,
    pub track_id: TrackAssignment,
    pub position: Vector3<f64>,
    pub measurement_std_m: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrackAssignment {
    Existing(String),
    NewTrack,
}

#[derive(Clone, Debug)]
pub struct ClusterEstimate {
    pub cluster_index: usize,
    pub position: Vector3<f64>,
    pub measurement_std_m: f64,
}

#[derive(Clone, Debug)]
pub struct JPDATrackUpdate {
    pub track_id: String,
    pub position: Vector3<f64>,
    pub measurement_std_m: f64,
    pub association_probability: f64,
}

#[derive(Clone, Debug, Default)]
pub struct JPDAResult {
    pub track_updates: Vec<JPDATrackUpdate>,
    pub new_tracks: Vec<ClusterEstimate>,
}

/// Global Nearest Neighbor associator using Mahalanobis-distance costs.
pub struct GNNAssociator {
    pub gate_threshold: f64,
}

impl Default for GNNAssociator {
    fn default() -> Self {
        Self {
            gate_threshold: 16.0,
        }
    }
}

impl GNNAssociator {
    pub(crate) fn associate(
        &self,
        tracks: &HashMap<String, ManagedTrack>,
        clusters: &[Vec<BearingObservation>],
        timestamp_s: f64,
    ) -> Vec<Assignment> {
        let cluster_estimates = cluster_estimates(clusters);
        if cluster_estimates.is_empty() {
            return Vec::new();
        }

        if tracks.is_empty() {
            return cluster_estimates
                .into_iter()
                .map(|estimate| Assignment {
                    cluster_index: estimate.cluster_index,
                    track_id: TrackAssignment::NewTrack,
                    position: estimate.position,
                    measurement_std_m: estimate.measurement_std_m,
                })
                .collect();
        }

        let track_ids = sorted_track_ids(tracks);
        let n_clusters = cluster_estimates.len();
        let n_tracks = track_ids.len();
        let dim = n_clusters.max(n_tracks);
        let big_cost = self.gate_threshold * 10.0;
        let mut cost_matrix = vec![vec![big_cost; dim]; dim];

        for (cluster_row, estimate) in cluster_estimates.iter().enumerate() {
            for (track_col, track_id) in track_ids.iter().enumerate() {
                if let Some(track) = tracks.get(track_id) {
                    if let Some(cost) = mahalanobis_cost(
                        track,
                        &estimate.position,
                        estimate.measurement_std_m,
                        timestamp_s,
                    ) {
                        if cost <= self.gate_threshold {
                            cost_matrix[cluster_row][track_col] = cost;
                        }
                    }
                }
            }
        }

        let assignment_result = hungarian_assignment(&cost_matrix);
        cluster_estimates
            .into_iter()
            .enumerate()
            .map(|(cluster_row, estimate)| {
                let assigned_col = assignment_result[cluster_row];
                let track_id = if assigned_col < n_tracks
                    && cost_matrix[cluster_row][assigned_col] < big_cost
                {
                    TrackAssignment::Existing(track_ids[assigned_col].clone())
                } else {
                    TrackAssignment::NewTrack
                };
                Assignment {
                    cluster_index: estimate.cluster_index,
                    track_id,
                    position: estimate.position,
                    measurement_std_m: estimate.measurement_std_m,
                }
            })
            .collect()
    }
}

/// Approximate JPDA associator for ambiguous multi-target frames.
pub struct JPDAAssociator {
    pub gate_threshold: f64,
    pub new_track_probability_threshold: f64,
    pub update_probability_threshold: f64,
}

impl JPDAAssociator {
    pub(crate) fn associate(
        &self,
        tracks: &HashMap<String, ManagedTrack>,
        clusters: &[Vec<BearingObservation>],
        timestamp_s: f64,
    ) -> JPDAResult {
        let cluster_estimates = cluster_estimates(clusters);
        if cluster_estimates.is_empty() {
            return JPDAResult::default();
        }

        if tracks.is_empty() {
            return JPDAResult {
                track_updates: Vec::new(),
                new_tracks: cluster_estimates,
            };
        }

        let track_ids = sorted_track_ids(tracks);
        let mut likelihoods = vec![vec![0.0; cluster_estimates.len()]; track_ids.len()];
        let mut row_denominators = vec![1.0; track_ids.len()];

        for (track_index, track_id) in track_ids.iter().enumerate() {
            let Some(track) = tracks.get(track_id) else {
                continue;
            };
            for (cluster_index, estimate) in cluster_estimates.iter().enumerate() {
                if let Some(cost) = mahalanobis_cost(
                    track,
                    &estimate.position,
                    estimate.measurement_std_m,
                    timestamp_s,
                ) {
                    if cost <= self.gate_threshold {
                        let likelihood = (-0.5 * cost).exp();
                        likelihoods[track_index][cluster_index] = likelihood;
                        row_denominators[track_index] += likelihood;
                    }
                }
            }
        }

        let mut track_updates = Vec::new();
        for (track_index, track_id) in track_ids.iter().enumerate() {
            let association_probability = 1.0 - (1.0 / row_denominators[track_index]);
            if association_probability < self.update_probability_threshold {
                continue;
            }

            let mut weighted_position = Vector3::zeros();
            let mut weighted_std_sq = 0.0;
            let mut total_weight = 0.0;
            for (cluster_index, estimate) in cluster_estimates.iter().enumerate() {
                let weight = likelihoods[track_index][cluster_index] / row_denominators[track_index];
                if weight <= 0.0 {
                    continue;
                }
                weighted_position += estimate.position * weight;
                weighted_std_sq += estimate.measurement_std_m.powi(2) * weight;
                total_weight += weight;
            }
            if total_weight <= 0.0 {
                continue;
            }
            track_updates.push(JPDATrackUpdate {
                track_id: track_id.clone(),
                position: weighted_position / total_weight,
                measurement_std_m: (weighted_std_sq / total_weight).sqrt().max(1.0),
                association_probability,
            });
        }

        let mut new_tracks = Vec::new();
        for (cluster_index, estimate) in cluster_estimates.into_iter().enumerate() {
            let max_probability = track_ids
                .iter()
                .enumerate()
                .map(|(track_index, _)| likelihoods[track_index][cluster_index] / row_denominators[track_index])
                .fold(0.0_f64, f64::max);
            if max_probability < self.new_track_probability_threshold {
                new_tracks.push(estimate);
            }
        }

        JPDAResult {
            track_updates,
            new_tracks,
        }
    }
}

fn sorted_track_ids(tracks: &HashMap<String, ManagedTrack>) -> SmallVec<[String; 8]> {
    let mut track_ids: SmallVec<[String; 8]> = tracks.keys().cloned().collect();
    track_ids.sort();
    track_ids
}

fn cluster_estimates(clusters: &[Vec<BearingObservation>]) -> Vec<ClusterEstimate> {
    let mut result = Vec::with_capacity(clusters.len());
    result.extend(clusters.iter().enumerate().filter_map(|(cluster_index, cluster)| {
        fuse_bearing_cluster(cluster).ok().map(|estimate| ClusterEstimate {
            cluster_index,
            position: estimate.position,
            measurement_std_m: estimate.measurement_std_m,
        })
    }));
    result
}

#[inline]
fn mahalanobis_cost(
    track: &ManagedTrack,
    position: &Vector3<f64>,
    measurement_std_m: f64,
    timestamp_s: f64,
) -> Option<f64> {
    let (predicted_position, predicted_covariance) = track.predicted_measurement(timestamp_s);
    let innovation = position - predicted_position;
    let innovation_covariance =
        predicted_covariance + (Matrix3::identity() * measurement_std_m.powi(2));
    innovation_covariance.cholesky().map(|decomp| {
        let solved = decomp.solve(&innovation);
        innovation.dot(&solved)
    })
}

/// Cluster observations by ray proximity or target label.
pub fn cluster_observations(
    observations: &[BearingObservation],
    distance_threshold_m: f64,
) -> Vec<Vec<BearingObservation>> {
    if observations.is_empty() {
        return Vec::new();
    }

    let has_labels = observations
        .iter()
        .all(|observation| !observation.target_id.is_empty() && observation.target_id != "unknown");
    if has_labels {
        let mut groups: HashMap<String, Vec<BearingObservation>> =
            HashMap::with_capacity(observations.len());
        for observation in observations {
            groups
                .entry(observation.target_id.clone())
                .or_default()
                .push(observation.clone());
        }
        let mut keys: Vec<String> = Vec::with_capacity(groups.len());
        keys.extend(groups.keys().cloned());
        keys.sort();
        return keys
            .into_iter()
            .map(|key| groups.remove(&key).unwrap_or_default())
            .collect();
    }

    let mut clusters: Vec<Vec<BearingObservation>> = Vec::new();
    for observation in observations {
        let mut best_cluster = None;
        let mut best_distance = f64::MAX;

        for (cluster_index, cluster) in clusters.iter().enumerate() {
            for existing in cluster {
                let distance = ray_closest_approach_distance(
                    &existing.origin,
                    &existing.direction,
                    &observation.origin,
                    &observation.direction,
                );
                if distance < best_distance {
                    best_distance = distance;
                    best_cluster = Some(cluster_index);
                }
            }
        }

        if best_distance < distance_threshold_m {
            if let Some(cluster_index) = best_cluster {
                clusters[cluster_index].push(observation.clone());
                continue;
            }
        }

        clusters.push(vec![observation.clone()]);
    }

    clusters
}

fn ray_closest_approach_distance(
    origin_a: &Vector3<f64>,
    dir_a: &Vector3<f64>,
    origin_b: &Vector3<f64>,
    dir_b: &Vector3<f64>,
) -> f64 {
    let w0 = origin_a - origin_b;
    let a = dir_a.dot(dir_a);
    let b = dir_a.dot(dir_b);
    let c = dir_b.dot(dir_b);
    let d = dir_a.dot(&w0);
    let e = dir_b.dot(&w0);

    let denom = a * c - b * b;
    if denom.abs() < 1.0e-12 {
        return w0.cross(dir_a).norm() / dir_a.norm().max(1.0e-12);
    }

    let t = ((b * e) - (c * d)) / denom;
    let s = ((a * e) - (b * d)) / denom;
    let closest_a = origin_a + dir_a * t.max(0.0);
    let closest_b = origin_b + dir_b * s.max(0.0);
    (closest_a - closest_b).norm()
}

/// Solve the assignment problem on a square cost matrix.
fn hungarian_assignment(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }

    let m = cost[0].len();
    let dim = n.max(m);
    let big = cost
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(0.0_f64, f64::max)
        + 1.0;

    let mut c = vec![vec![0.0; dim]; dim];
    for row in 0..dim {
        for col in 0..dim {
            c[row][col] = if row < n && col < m { cost[row][col] } else { big };
        }
    }

    for row in 0..dim {
        let row_min = c[row].iter().copied().fold(f64::INFINITY, f64::min);
        for col in 0..dim {
            c[row][col] -= row_min;
        }
    }
    for col in 0..dim {
        let col_min = (0..dim).map(|row| c[row][col]).fold(f64::INFINITY, f64::min);
        for row in 0..dim {
            c[row][col] -= col_min;
        }
    }

    let mut row_match = vec![usize::MAX; dim];
    let mut col_match = vec![usize::MAX; dim];

    for row in 0..dim {
        let mut dist = vec![f64::INFINITY; dim];
        let mut prev = vec![usize::MAX; dim];
        let mut visited = vec![false; dim];

        for col in 0..dim {
            dist[col] = c[row][col];
            prev[col] = row;
        }

        let mut terminal_col;
        loop {
            terminal_col = usize::MAX;
            let mut min_dist = f64::INFINITY;
            for col in 0..dim {
                if !visited[col] && dist[col] < min_dist {
                    min_dist = dist[col];
                    terminal_col = col;
                }
            }

            if terminal_col == usize::MAX {
                break;
            }
            visited[terminal_col] = true;

            if col_match[terminal_col] == usize::MAX {
                break;
            }

            let matched_row = col_match[terminal_col];
            for col in 0..dim {
                if !visited[col] {
                    let new_dist = min_dist + c[matched_row][col] - c[matched_row][terminal_col];
                    if new_dist < dist[col] {
                        dist[col] = new_dist;
                        prev[col] = matched_row;
                    }
                }
            }
        }

        if terminal_col == usize::MAX {
            continue;
        }

        loop {
            let matched_row = prev[terminal_col];
            col_match[terminal_col] = matched_row;
            let old_col = row_match[matched_row];
            row_match[matched_row] = terminal_col;
            if matched_row == row {
                break;
            }
            terminal_col = old_col;
        }
    }

    row_match.truncate(n);
    row_match
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrackerConfig;

    fn vec3(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    fn make_observation(
        node_id: &str,
        target_id: &str,
        origin: Vector3<f64>,
        target: Vector3<f64>,
    ) -> BearingObservation {
        let direction = (target - origin).normalize();
        BearingObservation {
            node_id: node_id.to_string(),
            target_id: target_id.to_string(),
            origin,
            direction,
            bearing_std_rad: 0.002,
            timestamp_s: 0.0,
            confidence: 1.0,
        }
    }

    #[test]
    fn hungarian_identity_assignment() {
        let cost = vec![
            vec![1.0, 100.0, 100.0],
            vec![100.0, 2.0, 100.0],
            vec![100.0, 100.0, 3.0],
        ];
        let result = hungarian_assignment(&cost);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn gnn_no_tracks_creates_new_assignments() {
        let associator = GNNAssociator::default();
        let tracks = HashMap::new();
        let target = vec3(50.0, 15.0, 10.0);
        let clusters = vec![vec![
            make_observation("a", "unknown", vec3(0.0, 0.0, 0.0), target),
            make_observation("b", "unknown", vec3(100.0, 0.0, 0.0), target),
        ]];

        let assignments = associator.associate(&tracks, &clusters, 0.0);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].track_id, TrackAssignment::NewTrack);
    }

    #[test]
    fn gnn_assigns_to_existing_track_with_mahalanobis_gate() {
        let associator = GNNAssociator::default();
        let target = vec3(50.0, 15.0, 10.0);
        let mut tracks = HashMap::new();
        tracks.insert(
            "track-0".to_string(),
            ManagedTrack::new(
                "track-0".to_string(),
                0.0,
                vec3(49.0, 14.0, 10.0),
                5.0,
                &TrackerConfig::default(),
            ),
        );

        let clusters = vec![vec![
            make_observation("a", "unknown", vec3(0.0, 0.0, 0.0), target),
            make_observation("b", "unknown", vec3(100.0, 0.0, 0.0), target),
        ]];

        let assignments = associator.associate(&tracks, &clusters, 0.0);
        assert_eq!(assignments.len(), 1);
        assert_eq!(
            assignments[0].track_id,
            TrackAssignment::Existing("track-0".to_string())
        );
    }

    #[test]
    fn gnn_gate_rejects_distant_track() {
        let associator = GNNAssociator { gate_threshold: 5.0 };
        let target = vec3(50.0, 15.0, 10.0);
        let mut tracks = HashMap::new();
        tracks.insert(
            "track-far".to_string(),
            ManagedTrack::new(
                "track-far".to_string(),
                0.0,
                vec3(500.0, 500.0, 500.0),
                5.0,
                &TrackerConfig::default(),
            ),
        );

        let clusters = vec![vec![
            make_observation("a", "unknown", vec3(0.0, 0.0, 0.0), target),
            make_observation("b", "unknown", vec3(100.0, 0.0, 0.0), target),
        ]];

        let assignments = associator.associate(&tracks, &clusters, 0.0);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].track_id, TrackAssignment::NewTrack);
    }

    #[test]
    fn jpda_generates_soft_update_and_suppresses_new_track_when_claimed() {
        let target = vec3(50.0, 15.0, 10.0);
        let mut tracks = HashMap::new();
        tracks.insert(
            "track-0".to_string(),
            ManagedTrack::new(
                "track-0".to_string(),
                0.0,
                vec3(48.0, 13.0, 10.0),
                5.0,
                &TrackerConfig::default(),
            ),
        );
        tracks.insert(
            "track-1".to_string(),
            ManagedTrack::new(
                "track-1".to_string(),
                0.0,
                vec3(52.0, 17.0, 10.0),
                5.0,
                &TrackerConfig::default(),
            ),
        );

        let clusters = vec![vec![
            make_observation("a", "unknown", vec3(0.0, 0.0, 0.0), target),
            make_observation("b", "unknown", vec3(100.0, 0.0, 0.0), target),
        ]];

        let result = JPDAAssociator {
            gate_threshold: 16.0,
            new_track_probability_threshold: 0.45,
            update_probability_threshold: 0.10,
        }
        .associate(&tracks, &clusters, 0.0);

        assert!(!result.track_updates.is_empty());
        assert!(result.new_tracks.is_empty());
    }

    #[test]
    fn cluster_labeled_observations_by_target() {
        let target_a = vec3(50.0, 15.0, 10.0);
        let target_b = vec3(-50.0, 30.0, 5.0);
        let observations = vec![
            make_observation("a", "asset-a", vec3(0.0, 0.0, 0.0), target_a),
            make_observation("b", "asset-b", vec3(100.0, 0.0, 0.0), target_b),
            make_observation("c", "asset-a", vec3(100.0, 0.0, 0.0), target_a),
        ];
        let clusters = cluster_observations(&observations, 50.0);
        assert_eq!(clusters.len(), 2);
    }
}
