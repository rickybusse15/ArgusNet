//! Live gRPC frame ingestion (feature = "live-stream").
//!
//! A dedicated OS thread owns a small tokio runtime, subscribes to the
//! daemon's `WatchFramesV2` fan-out, converts protobuf frames into the viewer's
//! replay representation, and hands them to the Bevy side over a std mpsc
//! channel. The Bevy system drains the channel each tick, so the render loop
//! never blocks on the network. Connection loss triggers reconnect with
//! backoff.

use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use argusnet_proto::pb::world_model_service_client::WorldModelServiceClient;
use argusnet_proto::pb::WatchFramesV2Request;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Identity};

use crate::replay::{frame_from_pb, ReplayFrame, ReplayState};

const RECONNECT_BACKOFF: Duration = Duration::from_secs(2);
const LIVE_CHANNEL_CAPACITY: usize = 256;
const LIVE_HISTORY_CAPACITY: usize = 10_000;

/// TLS material for the live gRPC subscription.
///
/// The viewer is the third gRPC client in the system, after the Python client in
/// `argusnet.adapters.argusnet_grpc` and the daemon's own peers. The other two
/// refuse to carry off-loopback traffic in the clear -- `argusnet-server` will
/// not even bind a non-loopback address without `--tls-cert`/`--tls-key`, and
/// `argusnet.security.transport` raises `TransportSecurityError` rather than
/// warn. The viewer previously hardcoded `http://`, so `--live remote:50051`
/// streamed the world model in plaintext. It now applies the same rule.
#[derive(Debug, Clone, Default)]
pub struct LiveTlsConfig {
    /// CA (PEM) used to verify the daemon's certificate. Presence enables TLS.
    pub ca_cert: Option<PathBuf>,
    /// Client certificate (PEM) for mTLS, paired with `client_key`.
    pub client_cert: Option<PathBuf>,
    /// Client private key (PEM), matching `client_cert`.
    pub client_key: Option<PathBuf>,
    /// Overrides the name verified against the server certificate.
    pub domain_name: Option<String>,
}

impl LiveTlsConfig {
    /// Whether any TLS material was supplied.
    pub fn is_enabled(&self) -> bool {
        self.ca_cert.is_some() || self.client_cert.is_some() || self.client_key.is_some()
    }
}

/// Host portion of an endpoint, with any scheme and port stripped.
fn endpoint_host(endpoint: &str) -> &str {
    let without_scheme = endpoint
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(endpoint);
    let authority = without_scheme
        .split_once('/')
        .map(|(auth, _)| auth)
        .unwrap_or(without_scheme);
    // IPv6 literals are bracketed: [::1]:50051
    if let Some(rest) = authority.strip_prefix('[') {
        return rest.split_once(']').map(|(host, _)| host).unwrap_or(rest);
    }
    authority
        .split_once(':')
        .map(|(host, _)| host)
        .unwrap_or(authority)
}

/// True for endpoints the daemon will serve without TLS.
///
/// Mirrors `is_loopback` in `argusnet-server` and `is_loopback_host` in
/// `argusnet.security.transport`.
pub fn is_loopback_endpoint(endpoint: &str) -> bool {
    let host = endpoint_host(endpoint);
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => ip.is_loopback(),
        Ok(IpAddr::V6(ip)) => ip.is_loopback(),
        Err(_) => false,
    }
}

/// Decide the URL to dial, refusing plaintext to anything but loopback.
pub fn resolve_endpoint_url(endpoint: &str, tls: &LiveTlsConfig) -> Result<String> {
    let host = endpoint_host(endpoint);
    if host.is_empty() {
        bail!("--live endpoint {endpoint:?} has no host");
    }
    let authority = endpoint
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(endpoint);

    if tls.is_enabled() {
        if tls.client_cert.is_some() != tls.client_key.is_some() {
            bail!("--live-tls-cert and --live-tls-key must both be set to enable mTLS");
        }
        return Ok(format!("https://{authority}"));
    }

    if endpoint.starts_with("https://") {
        bail!(
            "--live endpoint {endpoint:?} requests TLS but no --live-tls-ca was given; \
             supply the CA that signed the daemon certificate"
        );
    }

    if !is_loopback_endpoint(endpoint) {
        bail!(
            "refusing to stream from non-loopback endpoint {endpoint:?} without TLS. \
             Pass --live-tls-ca (and --live-tls-cert/--live-tls-key for mTLS), or connect \
             over loopback. This matches argusnet-server, which will not bind a \
             non-loopback address without --tls-cert/--tls-key."
        );
    }

    Ok(format!("http://{authority}"))
}

fn build_tls_config(tls: &LiveTlsConfig) -> Result<Option<ClientTlsConfig>> {
    if !tls.is_enabled() {
        return Ok(None);
    }
    let mut config = ClientTlsConfig::new();
    if let Some(ca_path) = &tls.ca_cert {
        let ca = std::fs::read(ca_path)
            .with_context(|| format!("failed to read --live-tls-ca {}", ca_path.display()))?;
        config = config.ca_certificate(Certificate::from_pem(ca));
    }
    match (&tls.client_cert, &tls.client_key) {
        (Some(cert_path), Some(key_path)) => {
            let cert = std::fs::read(cert_path).with_context(|| {
                format!("failed to read --live-tls-cert {}", cert_path.display())
            })?;
            let key = std::fs::read(key_path)
                .with_context(|| format!("failed to read --live-tls-key {}", key_path.display()))?;
            config = config.identity(Identity::from_pem(cert, key));
        }
        (None, None) => {}
        _ => bail!("--live-tls-cert and --live-tls-key must both be set to enable mTLS"),
    }
    if let Some(domain) = &tls.domain_name {
        config = config.domain_name(domain.clone());
    }
    Ok(Some(config))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveConnectionStatus {
    Connecting,
    Connected,
    Reconnecting,
}

impl LiveConnectionStatus {
    pub fn label(self) -> &'static str {
        match self {
            LiveConnectionStatus::Connecting => "connecting",
            LiveConnectionStatus::Connected => "connected",
            LiveConnectionStatus::Reconnecting => "reconnecting",
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            LiveConnectionStatus::Connecting => 0,
            LiveConnectionStatus::Connected => 1,
            LiveConnectionStatus::Reconnecting => 2,
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            1 => LiveConnectionStatus::Connected,
            2 => LiveConnectionStatus::Reconnecting,
            _ => LiveConnectionStatus::Connecting,
        }
    }
}

/// Bevy resource holding the live subscription state.
#[derive(Resource)]
pub struct LiveStream {
    pub endpoint: String,
    pub received_frame_count: u64,
    pub dropped_frame_count: u64,
    pub latest_sequence: u64,
    receiver: Mutex<Receiver<(String, ReplayFrame)>>,
    /// The daemon session the held frames came from.
    session: Option<String>,
    status: Arc<AtomicU8>,
    total_received: Arc<AtomicU64>,
    total_dropped: Arc<AtomicU64>,
    latest_sequence_shared: Arc<AtomicU64>,
}

impl LiveStream {
    pub fn status(&self) -> LiveConnectionStatus {
        LiveConnectionStatus::from_u8(self.status.load(Ordering::Relaxed))
    }
}

/// Spawn the background client thread and return the Bevy-side resource.
///
/// Fails before any connection attempt when the endpoint would be dialled in the
/// clear off loopback, or when the supplied TLS material is incomplete.
pub fn connect(endpoint: String, tls: LiveTlsConfig) -> Result<LiveStream> {
    let url = resolve_endpoint_url(&endpoint, &tls)?;
    let tls_config = build_tls_config(&tls)?;
    let (sender, receiver) =
        std::sync::mpsc::sync_channel::<(String, ReplayFrame)>(LIVE_CHANNEL_CAPACITY);
    let status = Arc::new(AtomicU8::new(LiveConnectionStatus::Connecting.as_u8()));
    let total_received = Arc::new(AtomicU64::new(0));
    let total_dropped = Arc::new(AtomicU64::new(0));
    let latest_sequence = Arc::new(AtomicU64::new(0));

    let thread_url = url;
    let thread_status = Arc::clone(&status);
    let thread_total = Arc::clone(&total_received);
    let thread_dropped = Arc::clone(&total_dropped);
    let thread_sequence = Arc::clone(&latest_sequence);
    std::thread::Builder::new()
        .name("argusnet-live-stream".into())
        .spawn(move || {
            stream_worker(
                thread_url,
                tls_config,
                sender,
                thread_status,
                thread_total,
                thread_dropped,
                thread_sequence,
            )
        })
        .expect("spawn live-stream thread");

    Ok(LiveStream {
        endpoint,
        received_frame_count: 0,
        dropped_frame_count: 0,
        latest_sequence: 0,
        receiver: Mutex::new(receiver),
        session: None,
        status,
        total_received,
        total_dropped,
        latest_sequence_shared: latest_sequence,
    })
}

/// Dial the daemon, applying TLS when configured.
async fn connect_client(
    url: &str,
    tls_config: Option<&ClientTlsConfig>,
) -> Result<WorldModelServiceClient<Channel>> {
    let mut endpoint = Channel::from_shared(url.to_string())
        .map_err(|error| anyhow!("invalid live endpoint {url:?}: {error}"))?;
    if let Some(tls) = tls_config {
        endpoint = endpoint.tls_config(tls.clone())?;
    }
    let channel = endpoint.connect().await?;
    Ok(WorldModelServiceClient::new(channel))
}

fn stream_worker(
    url: String,
    tls_config: Option<ClientTlsConfig>,
    sender: SyncSender<(String, ReplayFrame)>,
    status: Arc<AtomicU8>,
    total_received: Arc<AtomicU64>,
    total_dropped: Arc<AtomicU64>,
    latest_sequence: Arc<AtomicU64>,
) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build live-stream tokio runtime");
    runtime.block_on(async move {
        loop {
            if let Ok(mut client) = connect_client(&url, tls_config.as_ref()).await {
                if let Ok(response) = client
                    .watch_frames_v2(WatchFramesV2Request {
                        include_truth: false,
                        max_rate_hz: 0.0,
                        resume_after_sequence: 0,
                    })
                    .await
                {
                    status.store(LiveConnectionStatus::Connected.as_u8(), Ordering::Relaxed);
                    let mut stream = response.into_inner();
                    while let Ok(Some(message)) = stream.message().await {
                        let Some(frame) = message.frame else {
                            continue;
                        };
                        latest_sequence.store(message.sequence, Ordering::Relaxed);
                        total_dropped.fetch_add(message.dropped_since_last, Ordering::Relaxed);
                        total_received.fetch_add(1, Ordering::Relaxed);
                        match sender.try_send((message.session_id, frame_from_pb(frame))) {
                            Ok(()) | Err(TrySendError::Full(_)) => {}
                            Err(TrySendError::Disconnected(_)) => return,
                        }
                    }
                }
            }
            status.store(
                LiveConnectionStatus::Reconnecting.as_u8(),
                Ordering::Relaxed,
            );
            tokio::time::sleep(RECONNECT_BACKOFF).await;
        }
    });
}

/// Drain pending live frames into the replay document each tick.
///
/// When the user is at (or past) the previous tail -- i.e. not reviewing
/// history -- the playhead follows the newest frame.
pub fn ingest_live_frames_system(live: ResMut<LiveStream>, mut replay_state: ResMut<ReplayState>) {
    let live = live.into_inner();
    let appended = {
        let receiver = live.receiver.lock().expect("live receiver poisoned");
        ingest(
            &mut replay_state,
            &mut live.session,
            std::iter::from_fn(|| receiver.try_recv().ok()),
        )
    };
    if appended > 0 {
        live.received_frame_count = live.total_received.load(Ordering::Relaxed);
        live.dropped_frame_count = live.total_dropped.load(Ordering::Relaxed);
        live.latest_sequence = live.latest_sequence_shared.load(Ordering::Relaxed);
    }
}

/// Append `frames` to the replay, as `(session, frame)` pairs.
///
/// A frame from a different daemon session than the held history is a
/// different run (P2.0): the history is dropped and a new presentation epoch
/// begins, so nothing keyed to the old run (entities, selection,
/// interpolation) survives into it. A reconnect to the *same* session keeps
/// the epoch. Returns how many frames were appended.
pub fn ingest(
    replay_state: &mut ReplayState,
    session: &mut Option<String>,
    frames: impl Iterator<Item = (String, ReplayFrame)>,
) -> u64 {
    let mut appended = 0_u64;
    for (frame_session, frame) in frames {
        if session.as_deref() != Some(frame_session.as_str()) {
            if session.is_some() {
                replay_state.begin_epoch(None);
            }
            *session = Some(frame_session);
        }
        replay_state.push_frame(frame);
        let excess = replay_state
            .frame_count()
            .saturating_sub(LIVE_HISTORY_CAPACITY);
        if excess > 0 {
            replay_state.drain_front(excess);
        }
        appended += 1;
    }
    if appended > 0 {
        replay_state.live = true;
        let frame_count = replay_state.frame_count();
        let was_at_tail = frame_count as u64 <= appended
            || replay_state.frame_index + (appended as usize) + 1 >= frame_count;
        if was_at_tail {
            replay_state.step_to(frame_count.saturating_sub(1));
        }
    }
    appended
}

pub fn live_status_ui_system(mut contexts: EguiContexts, live: Res<LiveStream>) {
    let color = match live.status() {
        LiveConnectionStatus::Connected => egui::Color32::from_rgb(70, 210, 120),
        LiveConnectionStatus::Connecting => egui::Color32::YELLOW,
        LiveConnectionStatus::Reconnecting => egui::Color32::from_rgb(255, 150, 60),
    };
    // No egui context yet means no frame to draw the status badge into.
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Area::new(egui::Id::new("argusnet_live_status"))
        .anchor(egui::Align2::LEFT_TOP, [12.0, 58.0])
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(color, "● LIVE");
                    ui.label(live.status().label());
                    ui.separator();
                    ui.label(format!("seq {}", live.latest_sequence));
                    ui.label(format!("received {}", live.received_frame_count));
                    if live.dropped_frame_count > 0 {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            format!("dropped {}", live.dropped_frame_count),
                        );
                    }
                });
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(t: f32) -> ReplayFrame {
        serde_json::from_value(serde_json::json!({
            "timestamp_s": t, "tracks": [], "truths": [],
            "nodes": [{"node_id": "drone-a", "position": [t, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0], "is_mobile": true}]
        }))
        .expect("frame")
    }

    fn batch(session: &str, times: std::ops::Range<u32>) -> Vec<(String, ReplayFrame)> {
        times
            .map(|t| (session.to_string(), frame(t as f32)))
            .collect()
    }

    /// Same session, reconnected: the same run, the same epoch. A different
    /// session is a different run: its history replaces the old, in a new
    /// epoch.
    #[test]
    fn only_a_new_session_begins_a_new_epoch() {
        let mut replay = ReplayState::default();
        let mut session = None;
        assert_eq!(
            ingest(
                &mut replay,
                &mut session,
                batch("argusnetd-1", 0..5).into_iter()
            ),
            5
        );
        let epoch = replay.epoch();
        ingest(
            &mut replay,
            &mut session,
            batch("argusnetd-1", 5..8).into_iter(),
        );
        assert_eq!(replay.epoch(), epoch);
        assert_eq!(replay.frame_count(), 8);

        ingest(
            &mut replay,
            &mut session,
            batch("argusnetd-2", 0..3).into_iter(),
        );
        assert_eq!(replay.epoch(), epoch + 1);
        assert_eq!(replay.frame_count(), 3, "the old run's history is gone");
        assert_eq!(replay.frame_index, 2, "following the tail");
        assert_eq!(session.as_deref(), Some("argusnetd-2"));
    }

    /// Trimming history to capacity keeps the epoch, and keeps the playhead
    /// on the frame it was on.
    #[test]
    fn trimming_history_keeps_the_epoch_and_the_frame() {
        let mut replay = ReplayState::default();
        let mut session = None;
        let cap = LIVE_HISTORY_CAPACITY as u32;
        ingest(&mut replay, &mut session, batch("s", 0..cap).into_iter());
        replay.step_to(10);
        let epoch = replay.epoch();
        ingest(
            &mut replay,
            &mut session,
            batch("s", cap..cap + 4).into_iter(),
        );
        assert_eq!(replay.epoch(), epoch);
        assert_eq!(replay.frame_count(), LIVE_HISTORY_CAPACITY);
        assert_eq!(replay.current_frame().unwrap().timestamp_s, 10.0);
        // Live history is read-only review, and says what it dropped (P2.4a).
        assert!(replay.live);
        assert_eq!(replay.trimmed_before_s(), Some(4.0));
        assert_eq!(replay.time_s(), 10.0);
    }

    fn tls_with_ca() -> LiveTlsConfig {
        LiveTlsConfig {
            ca_cert: Some(PathBuf::from("/tmp/ca.pem")),
            ..Default::default()
        }
    }

    #[test]
    fn loopback_endpoints_are_recognised_in_every_spelling() {
        for endpoint in [
            "127.0.0.1:50051",
            "http://127.0.0.1:50051",
            "localhost:50051",
            "LocalHost:50051",
            "[::1]:50051",
            "127.5.6.7:50051",
        ] {
            assert!(
                is_loopback_endpoint(endpoint),
                "{endpoint} should be loopback"
            );
        }
    }

    #[test]
    fn non_loopback_endpoints_are_not_mistaken_for_loopback() {
        for endpoint in [
            "10.0.0.5:50051",
            "https://tracker.example.com:50051",
            "example.com:50051",
            "[2001:db8::1]:50051",
            // A host that merely starts with the loopback digits is not loopback.
            "127.0.0.1.example.com:50051",
        ] {
            assert!(
                !is_loopback_endpoint(endpoint),
                "{endpoint} should not be loopback"
            );
        }
    }

    #[test]
    fn loopback_without_tls_dials_plaintext() {
        let url = resolve_endpoint_url("127.0.0.1:50051", &LiveTlsConfig::default()).unwrap();
        assert_eq!(url, "http://127.0.0.1:50051");
    }

    /// The defect this guards: the viewer hardcoded `http://{endpoint}`, so a
    /// remote daemon was streamed in the clear while the Python client and the
    /// server both refuse to do that.
    #[test]
    fn non_loopback_without_tls_is_refused() {
        let error = resolve_endpoint_url("10.0.0.5:50051", &LiveTlsConfig::default())
            .expect_err("plaintext to a remote host must be refused");
        let message = format!("{error:#}");
        assert!(message.contains("non-loopback"), "got: {message}");
        assert!(message.contains("--live-tls-ca"), "got: {message}");
    }

    #[test]
    fn tls_material_upgrades_the_scheme() {
        let url = resolve_endpoint_url("10.0.0.5:50051", &tls_with_ca()).unwrap();
        assert_eq!(url, "https://10.0.0.5:50051");
    }

    #[test]
    fn explicit_scheme_is_not_duplicated() {
        let url = resolve_endpoint_url("http://10.0.0.5:50051", &tls_with_ca()).unwrap();
        assert_eq!(url, "https://10.0.0.5:50051");
    }

    #[test]
    fn https_without_a_ca_is_refused_rather_than_silently_downgraded() {
        let error = resolve_endpoint_url("https://tracker.example.com", &LiveTlsConfig::default())
            .expect_err("https without TLS material must be refused");
        assert!(format!("{error:#}").contains("--live-tls-ca"));
    }

    #[test]
    fn half_configured_mtls_is_refused() {
        let tls = LiveTlsConfig {
            ca_cert: Some(PathBuf::from("/tmp/ca.pem")),
            client_cert: Some(PathBuf::from("/tmp/client.pem")),
            client_key: None,
            domain_name: None,
        };
        let error = resolve_endpoint_url("10.0.0.5:50051", &tls)
            .expect_err("a client cert without its key must be refused");
        assert!(format!("{error:#}").contains("--live-tls-key"));
    }
}
