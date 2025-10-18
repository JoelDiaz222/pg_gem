use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int};
use std::slice;
use std::sync::LazyLock;
use tei::v1::{embed_client::EmbedClient, EmbedBatchRequest};
use tokio::runtime::Runtime;
use tonic::transport::Channel;

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

#[repr(C)]
pub struct EmbeddingBatch {
    pub data: *mut c_float,
    pub n_vectors: usize,
    pub dim: usize,
}

static RUNTIME: LazyLock<Runtime> =
    LazyLock::new(|| Runtime::new().expect("Failed to build Tokio runtime"));

thread_local! {
    static CLIENT: std::cell::RefCell<Option<EmbedClient<Channel>>> =
        std::cell::RefCell::new(None);
}

fn get_client() -> Result<EmbedClient<Channel>, Box<dyn std::error::Error>> {
    CLIENT.with(|cell| {
        let mut client_opt = cell.borrow_mut();
        if client_opt.is_none() {
            let channel = RUNTIME.block_on(async {
                Channel::from_static("http://127.0.0.1:50051")
                    .connect()
                    .await
            })?;
            *client_opt = Some(EmbedClient::new(channel));
        }
        Ok(client_opt.as_ref().unwrap().clone())
    })
}

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
                if !ptr.is_null() {
                    CStr::from_ptr(ptr).to_str().ok().map(str::to_owned)
                } else {
                    None
                }
            })
            .collect()
    };

    if text_slices.is_empty() {
        return -2;
    }

    let mut client = match get_client() {
        Ok(c) => c,
        Err(_) => return -3,
    };

    let response = match RUNTIME.block_on(async {
        client
            .embed_batch(tonic::Request::new(EmbedBatchRequest {
                inputs: text_slices,
                truncate: true,
                normalize: true,
                truncation_direction: 0,
                prompt_name: None,
                dimensions: None,
            }))
            .await
    }) {
        Ok(resp) => resp.into_inner(),
        Err(_) => return -5,
    };

    if response.embeddings.is_empty() {
        return -6;
    }

    let n_vectors = response.embeddings.len();
    let dim = response.embeddings[0].values.len();
    let total = n_vectors * dim;

    let mut flat = Vec::with_capacity(total);
    for e in response.embeddings {
        flat.extend_from_slice(&e.values);
    }

    let ptr = flat.as_mut_ptr();
    std::mem::forget(flat);

    unsafe {
        *out_batch = EmbeddingBatch {
            data: ptr,
            n_vectors,
            dim,
        };
    }

    0
}

#[unsafe(no_mangle)]
pub extern "C" fn free_embedding_batch(batch: *mut EmbeddingBatch) {
    if batch.is_null() {
        return;
    }

    unsafe {
        let batch_ref = &*batch;
        if !batch_ref.data.is_null() && batch_ref.n_vectors > 0 && batch_ref.dim > 0 {
            let total = batch_ref.n_vectors * batch_ref.dim;
            drop(Vec::from_raw_parts(batch_ref.data, total, total));
        }
    }
}
