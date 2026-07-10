fn main() {
    let protoc_path = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    std::env::set_var("PROTOC", protoc_path);
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("repo root");
    let proto_file = repo_root.join("proto/argusnet/v1/world_model.proto");
    let proto_root = repo_root.join("proto");
    // The proto lives outside this package dir, so cargo won't notice edits
    // to it without an explicit rerun trigger.
    println!("cargo:rerun-if-changed={}", proto_file.display());
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&[proto_file], &[proto_root])
        .expect("compile tracker proto");
}
