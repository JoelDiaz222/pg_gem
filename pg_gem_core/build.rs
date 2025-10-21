fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "remote")]
    tonic_prost_build::compile_protos("../server/proto/tei.proto")?;
    Ok(())
}
