pub mod fastembed;
pub mod grpc;

use anyhow::Result;
use linkme::distributed_slice;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedMethod {
    #[cfg(feature = "fastembed")]
    FastEmbed = 0,
    #[cfg(feature = "grpc")]
    Grpc = 1,
}

pub trait Embedder: Send + Sync {
    fn method(&self) -> EmbedMethod;
    fn embed(&self, model: &str, text_slices: Vec<&str>) -> Result<(Vec<f32>, usize, usize)>;
    fn is_model_allowed(&self, model: &str) -> bool;
}

#[distributed_slice]
pub static EMBEDDERS: [&'static dyn Embedder] = [..];

pub struct EmbedderRegistry {}

impl EmbedderRegistry {
    pub fn get_embedder_by_method_id(method: i32) -> Option<&'static dyn Embedder> {
        EMBEDDERS
            .iter()
            .find(|e| e.method() as i32 == method)
            .copied()
    }
}
