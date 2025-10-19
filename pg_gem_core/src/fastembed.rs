use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::{cell::RefCell, path::PathBuf};

thread_local! {
    static FASTEMBED_MODEL: RefCell<Option<TextEmbedding>> = RefCell::new(None);
}

pub fn generate_embeddings_fastembed(
    text_slices: Vec<&str>,
) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    Ok(FASTEMBED_MODEL.with(|cell| {
        let mut model_ref = cell.borrow_mut();

        if model_ref.is_none() {
            let model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                    .with_cache_dir(PathBuf::from("./fastembed_model")),
            )?;
            *model_ref = Some(model);
        }

        let model = model_ref.as_mut().unwrap();
        model.embed_flat(text_slices, None)
    })?)
}
