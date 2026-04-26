use anyhow::{anyhow, Result};
use chrono::Utc;
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use tonic::{transport::Server, Request, Response, Status};
use argusnet_core::{PlatformFrame, TrackerConfig, TrackingEngine};
use argusnet_proto::pb::world_model_service_server::{WorldModelService, WorldModelServiceServer};
use argusnet_proto::pb::{
    GetConfigRequest, GetConfigResponse, HealthRequest, HealthResponse, IngestFrameRequest,
    IngestFrameResponse, LatestFrameRequest, LatestFrameResponse, MissionStatusRequest,
    MissionStatusResponse, ResetRequest, ResetResponse,
};

#[derive(Debug, Parser)]
#[command(
    name = "argusnetd",
    about = "ArgusNet Rust tracking daemon."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Serve(ServeArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1:50051")]
    pub listen: String,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub min_observations: Option<u32>,
    #[arg(long)]
    pub max_stale_steps: Option<u32>,
    #[arg(long)]
    pub min_confidence: Option<f64>,
    #[arg(long)]
    pub max_bearing_std_rad: Option<f64>,
    #[arg(long)]
    pub max_timestamp_skew_s: Option<f64>,
    #[arg(long)]
    pub min_intersection_angle_deg: Option<f64>,
    #[arg(long)]
    pub data_association_mode: Option<String>,
    #[arg(long)]
    pub cv_process_accel_std: Option<f64>,
    #[arg(long)]
    pub ct_process_accel_std: Option<f64>,
    #[arg(long)]
    pub ct_turn_rate_std: Option<f64>,
    #[arg(long)]
    pub innovation_window: Option<u32>,
    #[arg(long)]
    pub innovation_scale_factor: Option<f64>,
    #[arg(long)]
    pub innovation_max_scale: Option<f64>,
    #[arg(long)]
    pub adaptive_measurement_noise: Option<bool>,
    #[arg(long)]
    pub chi_squared_gate_threshold: Option<f64>,
    #[arg(long)]
    pub cluster_distance_threshold_m: Option<f64>,
    #[arg(long)]
    pub near_parallel_rejection_angle_deg: Option<f64>,
    #[arg(long)]
    pub confirmation_m: Option<u32>,
    #[arg(long)]
    pub confirmation_n: Option<u32>,
    #[arg(long)]
    pub max_coast_frames: Option<u32>,
    #[arg(long)]
    pub max_coast_seconds: Option<f64>,
    #[arg(long)]
    pub min_quality_score: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct PartialTrackerConfig {
    min_observations: Option<u32>,
    max_stale_steps: Option<u32>,
    retain_history: Option<bool>,
    min_confidence: Option<f64>,
    max_bearing_std_rad: Option<f64>,
    max_timestamp_skew_s: Option<f64>,
    min_intersection_angle_deg: Option<f64>,
    data_association_mode: Option<String>,
    cv_process_accel_std: Option<f64>,
    ct_process_accel_std: Option<f64>,
    ct_turn_rate_std: Option<f64>,
    innovation_window: Option<u32>,
    innovation_scale_factor: Option<f64>,
    innovation_max_scale: Option<f64>,
    adaptive_measurement_noise: Option<bool>,
    chi_squared_gate_threshold: Option<f64>,
    cluster_distance_threshold_m: Option<f64>,
    near_parallel_rejection_angle_deg: Option<f64>,
    confirmation_m: Option<u32>,
    confirmation_n: Option<u32>,
    max_coast_frames: Option<u32>,
    max_coast_seconds: Option<f64>,
    min_quality_score: Option<f64>,
}

impl PartialTrackerConfig {
    fn apply(self, mut config: TrackerConfig) -> TrackerConfig {
        if let Some(value) = self.min_observations {
            config.min_observations = value;
        }
        if let Some(value) = self.max_stale_steps {
            config.max_stale_steps = value;
        }
        if let Some(value) = self.retain_history {
            config.retain_history = value;
        }
        if let Some(value) = self.min_confidence {
            config.min_confidence = value;
        }
        if let Some(value) = self.max_bearing_std_rad {
            config.max_bearing_std_rad = value;
        }
        if let Some(value) = self.max_timestamp_skew_s {
            config.max_timestamp_skew_s = value;
        }
        if let Some(value) = self.min_intersection_angle_deg {
            config.min_intersection_angle_deg = value;
        }
        if let Some(value) = self.data_association_mode {
            config.data_association_mode = match value.to_lowercase().as_str() {
                "gnn" => argusnet_core::AssociationMode::GNN,
                "jpda" => argusnet_core::AssociationMode::JPDA,
                _ => argusnet_core::AssociationMode::Labeled,
            };
        }
        if let Some(value) = self.cv_process_accel_std {
            config.cv_process_accel_std = value;
        }
        if let Some(value) = self.ct_process_accel_std {
            config.ct_process_accel_std = value;
        }
        if let Some(value) = self.ct_turn_rate_std {
            config.ct_turn_rate_std = value;
        }
        if let Some(value) = self.innovation_window {
            config.innovation_window = value;
        }
        if let Some(value) = self.innovation_scale_factor {
            config.innovation_scale_factor = value;
        }
        if let Some(value) = self.innovation_max_scale {
            config.innovation_max_scale = value;
        }
        if let Some(value) = self.adaptive_measurement_noise {
            config.adaptive_measurement_noise = value;
        }
        if let Some(value) = self.chi_squared_gate_threshold {
            config.chi_squared_gate_threshold = value;
        }
        if let Some(value) = self.cluster_distance_threshold_m {
            config.cluster_distance_threshold_m = value;
        }
        if let Some(value) = self.near_parallel_rejection_angle_deg {
            config.near_parallel_rejection_angle_deg = value;
        }
        if let Some(value) = self.confirmation_m {
            config.confirmation_m = value;
        }
        if let Some(value) = self.confirmation_n {
            config.confirmation_n = value;
        }
        if let Some(value) = self.max_coast_frames {
            config.max_coast_frames = value;
        }
        if let Some(value) = self.max_coast_seconds {
            config.max_coast_seconds = value;
        }
        if let Some(value) = self.min_quality_score {
            config.min_quality_score = value;
        }
        config
    }
}

pub fn load_tracker_config(args: &ServeArgs) -> Result<TrackerConfig> {
    let mut config = TrackerConfig::default();

    if let Some(path) = &args.config {
        let parsed: PartialTrackerConfig = serde_yaml::from_str(&std::fs::read_to_string(path)?)?;
        config = parsed.apply(config);
    }

    config = PartialTrackerConfig {
        min_observations: args.min_observations,
        max_stale_steps: args.max_stale_steps,
        retain_history: None,
        min_confidence: args.min_confidence,
        max_bearing_std_rad: args.max_bearing_std_rad,
        max_timestamp_skew_s: args.max_timestamp_skew_s,
        min_intersection_angle_deg: args.min_intersection_angle_deg,
        data_association_mode: args.data_association_mode.clone(),
        cv_process_accel_std: args.cv_process_accel_std,
        ct_process_accel_std: args.ct_process_accel_std,
        ct_turn_rate_std: args.ct_turn_rate_std,
        innovation_window: args.innovation_window,
        innovation_scale_factor: args.innovation_scale_factor,
        innovation_max_scale: args.innovation_max_scale,
        adaptive_measurement_noise: args.adaptive_measurement_noise,
        chi_squared_gate_threshold: args.chi_squared_gate_threshold,
        cluster_distance_threshold_m: args.cluster_distance_threshold_m,
        near_parallel_rejection_angle_deg: args.near_parallel_rejection_angle_deg,
        confirmation_m: args.confirmation_m,
        confirmation_n: args.confirmation_n,
        max_coast_frames: args.max_coast_frames,
        max_coast_seconds: args.max_coast_seconds,
        min_quality_score: args.min_quality_score,
    }
    .apply(config);

    config.validate().map_err(|error| anyhow!(error))?;
    Ok(config)
}

#[derive(Debug)]
enum EngineCommand {
    Ingest {
        request: IngestFrameRequest,
        response: oneshot::Sender<Result<IngestFrameResponse, Status>>,
    },
    Latest {
        response: oneshot::Sender<Result<LatestFrameResponse, Status>>,
    },
    Reset {
        response: oneshot::Sender<Result<ResetResponse, Status>>,
    },
    GetConfig {
        response: oneshot::Sender<Result<GetConfigResponse, Status>>,
    },
    Health {
        response: oneshot::Sender<Result<HealthResponse, Status>>,
    },
    MissionStatus {
        scenario_name: String,
        response: oneshot::Sender<Result<MissionStatusResponse, Status>>,
    },
}

#[derive(Clone, Debug)]
struct EngineHandle {
    sender: mpsc::Sender<EngineCommand>,
}

impl EngineHandle {
    async fn ingest(&self, request: IngestFrameRequest) -> Result<IngestFrameResponse, Status> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(EngineCommand::Ingest {
                request,
                response: tx,
            })
            .await
            .map_err(|_| Status::unavailable("tracking engine is unavailable"))?;
        rx.await
            .map_err(|_| Status::unavailable("tracking engine stopped"))?
    }

    async fn latest(&self) -> Result<LatestFrameResponse, Status> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(EngineCommand::Latest { response: tx })
            .await
            .map_err(|_| Status::unavailable("tracking engine is unavailable"))?;
        rx.await
            .map_err(|_| Status::unavailable("tracking engine stopped"))?
    }

    async fn reset(&self) -> Result<ResetResponse, Status> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(EngineCommand::Reset { response: tx })
            .await
            .map_err(|_| Status::unavailable("tracking engine is unavailable"))?;
        rx.await
            .map_err(|_| Status::unavailable("tracking engine stopped"))?
    }

    async fn get_config(&self) -> Result<GetConfigResponse, Status> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(EngineCommand::GetConfig { response: tx })
            .await
            .map_err(|_| Status::unavailable("tracking engine is unavailable"))?;
        rx.await
            .map_err(|_| Status::unavailable("tracking engine stopped"))?
    }

    async fn health(&self) -> Result<HealthResponse, Status> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(EngineCommand::Health { response: tx })
            .await
            .map_err(|_| Status::unavailable("tracking engine is unavailable"))?;
        rx.await
            .map_err(|_| Status::unavailable("tracking engine stopped"))?
    }

    async fn mission_status(&self, scenario_name: String) -> Result<MissionStatusResponse, Status> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(EngineCommand::MissionStatus {
                scenario_name,
                response: tx,
            })
            .await
            .map_err(|_| Status::unavailable("tracking engine is unavailable"))?;
        rx.await
            .map_err(|_| Status::unavailable("tracking engine stopped"))?
    }
}

fn invalid_argument(message: impl Into<String>) -> Status {
    Status::invalid_argument(message.into())
}

fn convert_request(
    request: IngestFrameRequest,
) -> Result<
    (
        f64,
        Vec<argusnet_core::NodeState>,
        Vec<argusnet_core::BearingObservation>,
        Vec<argusnet_core::TruthState>,
    ),
    Status,
> {
    let timestamp_s = request.timestamp_s;
    let node_states = request
        .node_states
        .into_iter()
        .map(argusnet_proto::node_state_from_pb)
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid_argument)?;
    let observations = request
        .observations
        .into_iter()
        .map(argusnet_proto::observation_from_pb)
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid_argument)?;
    let truths = request
        .truths
        .into_iter()
        .map(argusnet_proto::truth_from_pb)
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid_argument)?;
    Ok((timestamp_s, node_states, observations, truths))
}

fn frame_response(frame: PlatformFrame) -> IngestFrameResponse {
    IngestFrameResponse {
        frame: Some(argusnet_proto::frame_to_pb(frame)),
    }
}

async fn spawn_engine(config: TrackerConfig) -> Result<EngineHandle> {
    let mut engine = TrackingEngine::new(config.clone()).map_err(|error| anyhow!(error))?;
    let started_at_utc = Utc::now().to_rfc3339();
    let (sender, mut receiver) = mpsc::channel::<EngineCommand>(64);

    tokio::spawn(async move {
        let mut processed_frame_count = 0_u64;
        let mut total_ingest_latency_s = 0.0_f64;
        while let Some(command) = receiver.recv().await {
            match command {
                EngineCommand::Ingest { request, response } => {
                    let t0 = Instant::now();
                    let reply = match convert_request(request) {
                        Ok((timestamp_s, node_states, observations, truths)) => {
                            let frame =
                                engine.ingest_frame(timestamp_s, node_states, observations, truths);
                            processed_frame_count += 1;
                            total_ingest_latency_s += t0.elapsed().as_secs_f64();
                            Ok(frame_response(frame))
                        }
                        Err(error) => Err(error),
                    };
                    let _ = response.send(reply);
                }
                EngineCommand::Latest { response } => {
                    let reply = Ok(LatestFrameResponse {
                        frame: engine
                            .latest_frame()
                            .cloned()
                            .map(argusnet_proto::frame_to_pb),
                    });
                    let _ = response.send(reply);
                }
                EngineCommand::Reset { response } => {
                    engine.reset();
                    let _ = response.send(Ok(ResetResponse {}));
                }
                EngineCommand::GetConfig { response } => {
                    let reply = Ok(GetConfigResponse {
                        config: Some(argusnet_proto::tracker_config_to_pb(engine.config())),
                    });
                    let _ = response.send(reply);
                }
                EngineCommand::Health { response } => {
                    let latest_ts = engine.latest_frame().map_or(0.0, |f| f.timestamp_s);
                    let health_snapshots = engine.node_health_snapshot(latest_ts);
                    let active_node_count = health_snapshots.len() as u32;
                    let stale_node_count = engine.stale_node_count(latest_ts);
                    let reply = Ok(HealthResponse {
                        status: "SERVING".to_string(),
                        started_at_utc: started_at_utc.clone(),
                        processed_frame_count,
                        node_health: health_snapshots
                            .iter()
                            .map(argusnet_proto::node_health_to_pb)
                            .collect(),
                        mean_frame_rate_hz: engine.mean_frame_rate_hz(),
                        mean_ingest_latency_s: if processed_frame_count > 0 {
                            total_ingest_latency_s / processed_frame_count as f64
                        } else {
                            0.0
                        },
                        active_node_count,
                        stale_node_count,
                    });
                    let _ = response.send(reply);
                }
                EngineCommand::MissionStatus { scenario_name, response } => {
                    let latest = engine.latest_frame();
                    let elapsed_s = latest.map_or(0.0, |f| f.timestamp_s);
                    let active_track_count = latest
                        .map_or(0, |f| f.metrics.active_track_count);
                    let active_track_ids: Vec<String> = latest
                        .map(|f| f.tracks.iter().map(|t| t.track_id.clone()).collect())
                        .unwrap_or_default();
                    let reply = Ok(MissionStatusResponse {
                        scenario_name,
                        elapsed_s,
                        active_track_count,
                        active_zone_count: 0,  // zones pushed from Python, not stored server-side
                        zones: vec![],
                        active_track_ids,
                    });
                    let _ = response.send(reply);
                }
            }
        }
    });

    Ok(EngineHandle { sender })
}

#[derive(Clone)]
struct GrpcTrackerService {
    engine: EngineHandle,
}

#[tonic::async_trait]
impl WorldModelService for GrpcTrackerService {
    async fn ingest_frame(
        &self,
        request: Request<IngestFrameRequest>,
    ) -> Result<Response<IngestFrameResponse>, Status> {
        let response = self.engine.ingest(request.into_inner()).await?;
        Ok(Response::new(response))
    }

    type TrackStreamStream =
        Pin<Box<dyn Stream<Item = Result<IngestFrameResponse, Status>> + Send + 'static>>;

    async fn track_stream(
        &self,
        request: Request<tonic::Streaming<IngestFrameRequest>>,
    ) -> Result<Response<Self::TrackStreamStream>, Status> {
        let mut inbound = request.into_inner();
        let engine = self.engine.clone();
        let (sender, receiver) = mpsc::channel::<Result<IngestFrameResponse, Status>>(16);

        tokio::spawn(async move {
            loop {
                match inbound.message().await {
                    Ok(Some(message)) => {
                        let reply = engine.ingest(message).await;
                        if sender.send(reply).await.is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        let _ = sender.send(Err(Status::internal(error.to_string()))).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(
            Box::pin(ReceiverStream::new(receiver)) as Self::TrackStreamStream
        ))
    }

    async fn latest_frame(
        &self,
        _request: Request<LatestFrameRequest>,
    ) -> Result<Response<LatestFrameResponse>, Status> {
        Ok(Response::new(self.engine.latest().await?))
    }

    async fn reset(
        &self,
        _request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> {
        Ok(Response::new(self.engine.reset().await?))
    }

    async fn get_config(
        &self,
        _request: Request<GetConfigRequest>,
    ) -> Result<Response<GetConfigResponse>, Status> {
        Ok(Response::new(self.engine.get_config().await?))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(self.engine.health().await?))
    }

    async fn mission_status(
        &self,
        request: Request<MissionStatusRequest>,
    ) -> Result<Response<MissionStatusResponse>, Status> {
        let scenario_name = request.into_inner().scenario_name;
        Ok(Response::new(self.engine.mission_status(scenario_name).await?))
    }
}

pub async fn serve(args: ServeArgs) -> Result<()> {
    let config = load_tracker_config(&args)?;
    let engine = spawn_engine(config).await?;
    let service = GrpcTrackerService { engine };
    let address: SocketAddr = args
        .listen
        .parse()
        .map_err(|error| anyhow!("invalid listen address {}: {error}", args.listen))?;

    Server::builder()
        .add_service(WorldModelServiceServer::new(service))
        .serve(address)
        .await?;
    Ok(())
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Serve(args) => serve(args).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn yaml_config_is_loaded_and_cli_overrides_it() {
        let path = std::env::temp_dir().join("argusnetd-config-test.yaml");
        fs::write(
            &path,
            "min_observations: 3\nmax_stale_steps: 4\nmin_confidence: 0.4\n",
        )
        .unwrap();
        let args = ServeArgs {
            listen: "127.0.0.1:50051".to_string(),
            config: Some(path.clone()),
            min_observations: None,
            max_stale_steps: Some(9),
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
        };

        let config = load_tracker_config(&args).unwrap();
        assert_eq!(3, config.min_observations);
        assert_eq!(9, config.max_stale_steps);
        assert_eq!(0.4, config.min_confidence);

        let _ = fs::remove_file(path);
    }
}
