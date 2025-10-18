use std::os::raw::{c_char, c_float, c_int};
use std::ffi::CStr;
use std::slice;
use tei::v1::{embed_client::EmbedClient, EmbedBatchRequest};

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

#[repr(C)]
pub struct EmbeddingBatch {
    pub data: *mut *mut c_float,    // array of float pointers (each vector)
    pub n_vectors: usize,           // number of vectors
    pub dim: usize,                 // number of dimensions per vector
}

/// Generate embeddings using a gRPC service
#[unsafe(no_mangle)]
pub extern "C" fn generate_embeddings_from_texts(
    inputs: *const *const c_char,
    n_inputs: usize,
    out_batch: *mut EmbeddingBatch,
) -> c_int {
    if inputs.is_null() || out_batch.is_null() {
        return -1;
    }

    let text_slices: Vec<String> = unsafe {
        slice::from_raw_parts(inputs, n_inputs)
            .iter()
            .filter_map(|&ptr| {
                if ptr.is_null() {
                    None
                } else {
                    CStr::from_ptr(ptr).to_str().ok().map(|s| s.to_string())
                }
            })
            .collect()
    };

    if text_slices.is_empty() {
        return -2;
    }

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return -3,
    };

    let mut client = match rt.block_on(async {
        EmbedClient::connect("http://localhost:50051").await
    }) {
        Ok(client) => client,
        Err(_) => return -4,
    };

    // Prepare the request
    let request = tonic::Request::new(EmbedBatchRequest {
        inputs: text_slices,
        truncate: true,
        normalize: true,
        truncation_direction: 0,
        prompt_name: None,
        dimensions: None,
    });

    // Get embeddings from TEI service
    let embeddings_response = match rt.block_on(async {
        client.embed_batch(request).await
    }) {
        Ok(response) => response.into_inner(),
        Err(_) => return -5,
    };

    let embeddings = embeddings_response.embeddings;

    if embeddings.is_empty() {
        return -6;
    }

    // Get dimension from first embedding
    let dim = embeddings[0].values.len();

    // Convert embeddings to FFI-compatible format
    let mut data_ptrs: Vec<*mut c_float> = Vec::with_capacity(embeddings.len());

    for embedding in embeddings {
        let mut values: Vec<f32> = embedding.values;
        let ptr = values.as_mut_ptr();
        std::mem::forget(values);
        data_ptrs.push(ptr);
    }

    let batch = EmbeddingBatch {
        data: data_ptrs.as_mut_ptr(),
        n_vectors: data_ptrs.len(),
        dim,
    };

    std::mem::forget(data_ptrs);

    unsafe {
        *out_batch = batch;
    }

    0
}

// Helper function to free the embedding batch
#[unsafe(no_mangle)]
pub extern "C" fn free_embedding_batch(batch: *mut EmbeddingBatch) {
    if batch.is_null() {
        return;
    }

    unsafe {
        let batch_ref = &*batch;

        // Free each vector
        let data_slice = slice::from_raw_parts(batch_ref.data, batch_ref.n_vectors);
        for &ptr in data_slice {
            if !ptr.is_null() {
                drop(Vec::from_raw_parts(ptr, batch_ref.dim, batch_ref.dim));
            }
        }

        // Free the array of pointers
        drop(Vec::from_raw_parts(
            batch_ref.data,
            batch_ref.n_vectors,
            batch_ref.n_vectors,
        ));
    }
}
