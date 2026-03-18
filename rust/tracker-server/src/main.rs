use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracker_server::run(tracker_server::Cli::parse()).await
}
