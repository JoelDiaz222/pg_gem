#![cfg(feature = "fastembed")]
use crate::embedders::{EmbedMethod, Embedder, EMBEDDERS};
use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::{cell::RefCell, collections::HashMap, path::PathBuf, str::FromStr};

#[unsafe(no_mangle)]
pub static EMBED_METHOD_FASTEMBED: i32 = EmbedMethod::FastEmbed as i32;

thread_local! {
    static FASTEMBED_MODELS: RefCell<HashMap<String, TextEmbedding>> = RefCell::new(HashMap::new());
}

struct FastEmbedder;

impl Embedder for FastEmbedder {
    fn method(&self) -> EmbedMethod {
        EmbedMethod::FastEmbed
    }

    fn embed(&self, model: &str, text_slices: Vec<&str>) -> Result<(Vec<f32>, usize, usize)> {
        FASTEMBED_MODELS.with(|cell| {
            let mut models = cell.borrow_mut();

            let model_instance = models.entry(model.to_string()).or_insert_with(|| {
                let embedding_model =
                    EmbeddingModel::from_str(model).expect("Failed to parse model");
                TextEmbedding::try_new(
                    InitOptions::new(embedding_model)
                        .with_cache_dir(PathBuf::from("./fastembed_models")),
                )
                .expect("Failed to initialize model")
            });

            model_instance.embed_flat(text_slices, None)
        })
    }
}

#[linkme::distributed_slice(EMBEDDERS)]
static FASTEMBED: &dyn Embedder = &FastEmbedder;
