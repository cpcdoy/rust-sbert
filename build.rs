use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Just for bin targets (that is force-build for Docker cache optimisation)
    if env::var("OUT_DIR").is_err() {
        env::set_var("OUT_DIR", concat!(env!("CARGO_MANIFEST_DIR"), "/target/"));
    }

    tonic_build::compile_protos("proto/service.proto")?;
    Ok(())
}