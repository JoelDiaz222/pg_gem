use anyhow::{anyhow, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::{cell::RefCell, collections::HashMap, path::PathBuf, str::FromStr};

thread_local! {
    static FASTEMBED_MODELS: RefCell<HashMap<String, TextEmbedding>> = RefCell::new(HashMap::new());
}

pub fn generate_embeddings_fastembed(
    model: &str,
    text_slices: Vec<&str>,
) -> Result<(Vec<f32>, usize, usize)> {
    Ok(FASTEMBED_MODELS.with(|cell| {
        let mut models = cell.borrow_mut();

        let model_instance = match models.get_mut(model) {
            Some(instance) => instance,
            None => {
                let embedding_model = EmbeddingModel::from_str(model)
                    .map_err(|e| anyhow!("Failed to parse model '{}': {}", model, e))?;

                let model_instance = TextEmbedding::try_new(
                    InitOptions::new(embedding_model)
                        .with_cache_dir(PathBuf::from("./fastembed_models")),
                )?;

                models.insert(model.to_string(), model_instance);
                models.get_mut(model).expect("model just inserted")
            }
        };

        model_instance.embed_flat(text_slices, None)
    })?)
}
