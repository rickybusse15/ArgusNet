//! Live gRPC frame ingestion (feature = "live-stream").
//!
//! A dedicated OS thread owns a small tokio runtime, subscribes to the
//! daemon's `WatchFrames` fan-out, converts protobuf frames into the viewer's
//! replay representation, and hands them to the Bevy side over a std mpsc
//! channel. The Bevy system drains the channel each tick, so the render loop
//! never blocks on the network. Connection loss triggers reconnect with
//! backoff.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use argusnet_proto::pb::world_model_service_client::WorldModelServiceClient;
use argusnet_proto::pb::WatchFramesV2Request;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::replay::{frame_from_pb, ReplayDocument, ReplayFrame, ReplayState};

const RECONNECT_BACKOFF: Duration = Duration::from_secs(2);
const LIVE_CHANNEL_CAPACITY: usize = 256;
const LIVE_HISTORY_CAPACITY: usize = 10_000;

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
    receiver: Mutex<Receiver<ReplayFrame>>,
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
pub fn connect(endpoint: String) -> LiveStream {
    let (sender, receiver) = std::sync::mpsc::sync_channel::<ReplayFrame>(LIVE_CHANNEL_CAPACITY);
    let status = Arc::new(AtomicU8::new(LiveConnectionStatus::Connecting.as_u8()));
    let total_received = Arc::new(AtomicU64::new(0));
    let total_dropped = Arc::new(AtomicU64::new(0));
    let latest_sequence = Arc::new(AtomicU64::new(0));

    let thread_endpoint = endpoint.clone();
    let thread_status = Arc::clone(&status);
    let thread_total = Arc::clone(&total_received);
    let thread_dropped = Arc::clone(&total_dropped);
    let thread_sequence = Arc::clone(&latest_sequence);
    std::thread::Builder::new()
        .name("argusnet-live-stream".into())
        .spawn(move || {
            stream_worker(
                thread_endpoint,
                sender,
                thread_status,
                thread_total,
                thread_dropped,
                thread_sequence,
            )
        })
        .expect("spawn live-stream thread");

    LiveStream {
        endpoint,
        received_frame_count: 0,
        dropped_frame_count: 0,
        latest_sequence: 0,
        receiver: Mutex::new(receiver),
        status,
        total_received,
        total_dropped,
        latest_sequence_shared: latest_sequence,
    }
}

fn stream_worker(
    endpoint: String,
    sender: SyncSender<ReplayFrame>,
    status: Arc<AtomicU8>,
    total_received: Arc<AtomicU64>,
    total_dropped: Arc<AtomicU64>,
    latest_sequence: Arc<AtomicU64>,
) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build live-stream tokio runtime");
    let url = if endpoint.starts_with("http") {
        endpoint
    } else {
        format!("http://{endpoint}")
    };

    runtime.block_on(async move {
        loop {
            match WorldModelServiceClient::connect(url.clone()).await {
                Ok(mut client) => match client
                    .watch_frames_v2(WatchFramesV2Request {
                        include_truth: false,
                        max_rate_hz: 0.0,
                        resume_after_sequence: 0,
                    })
                    .await
                {
                    Ok(response) => {
                        status.store(LiveConnectionStatus::Connected.as_u8(), Ordering::Relaxed);
                        let mut stream = response.into_inner();
                        loop {
                            match stream.message().await {
                                Ok(Some(message)) => {
                                    let Some(frame) = message.frame else {
                                        continue;
                                    };
                                    latest_sequence.store(message.sequence, Ordering::Relaxed);
                                    total_dropped
                                        .fetch_add(message.dropped_since_last, Ordering::Relaxed);
                                    total_received.fetch_add(1, Ordering::Relaxed);
                                    match sender.try_send(frame_from_pb(frame)) {
                                        Ok(()) | Err(TrySendError::Full(_)) => {}
                                        Err(TrySendError::Disconnected(_)) => return,
                                    }
                                }
                                Ok(None) | Err(_) => break, // stream ended or errored
                            }
                        }
                    }
                    Err(_) => {}
                },
                Err(_) => {}
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
/// When the user is at (or past) the previous tail — i.e. not reviewing
/// history — the playhead follows the newest frame.
pub fn ingest_live_frames_system(live: ResMut<LiveStream>, mut replay_state: ResMut<ReplayState>) {
    let mut appended = 0_u64;
    {
        let receiver = live.receiver.lock().expect("live receiver poisoned");
        loop {
            match receiver.try_recv() {
                Ok(frame) => {
                    let document = replay_state.document.get_or_insert_with(|| ReplayDocument {
                        meta: None,
                        summary: None,
                        frames: Vec::new(),
                    });
                    document.frames.push(frame);
                    if document.frames.len() > LIVE_HISTORY_CAPACITY {
                        let excess = document.frames.len() - LIVE_HISTORY_CAPACITY;
                        document.frames.drain(..excess);
                        replay_state.frame_index = replay_state.frame_index.saturating_sub(excess);
                    }
                    appended += 1;
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
    }
    if appended > 0 {
        let live = live.into_inner();
        live.received_frame_count = live.total_received.load(Ordering::Relaxed);
        live.dropped_frame_count = live.total_dropped.load(Ordering::Relaxed);
        live.latest_sequence = live.latest_sequence_shared.load(Ordering::Relaxed);
        let frame_count = replay_state.frame_count();
        let was_at_tail = frame_count as u64 <= appended
            || replay_state.frame_index + (appended as usize) + 1 >= frame_count;
        if was_at_tail {
            replay_state.frame_index = frame_count.saturating_sub(1);
        }
    }
}

pub fn live_status_ui_system(mut contexts: EguiContexts, live: Res<LiveStream>) {
    let color = match live.status() {
        LiveConnectionStatus::Connected => egui::Color32::from_rgb(70, 210, 120),
        LiveConnectionStatus::Connecting => egui::Color32::YELLOW,
        LiveConnectionStatus::Reconnecting => egui::Color32::from_rgb(255, 150, 60),
    };
    egui::Area::new(egui::Id::new("argusnet_live_status"))
        .anchor(egui::Align2::LEFT_TOP, [12.0, 58.0])
        .show(contexts.ctx_mut(), |ui| {
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
