use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    argusnet_server::run(argusnet_server::Cli::parse()).await
}
