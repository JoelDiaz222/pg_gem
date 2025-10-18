use std::os::raw::{c_char, c_float, c_int};
use std::slice;
use std::sync::LazyLock;
use std::time::Duration;
use tei::v1::{embed_client::EmbedClient, EmbedBatchRequest};
use tokio::runtime::Runtime;
use tonic::transport::{Channel, Endpoint};

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

#[repr(C)]
pub struct StringSlice {
    pub ptr: *const c_char,
    pub len: usize,
}

#[repr(C)]
pub struct EmbeddingBatch {
    pub data: *mut c_float,
    pub n_vectors: usize,
    pub dim: usize,
}

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime")
});

static ENDPOINT: LazyLock<Endpoint> = LazyLock::new(|| {
    Channel::from_static("http://127.0.0.1:50051")
        .http2_keep_alive_interval(Duration::from_secs(75))
        .keep_alive_timeout(Duration::from_secs(20))
        .connect_timeout(Duration::from_secs(5))
        .tcp_nodelay(true)
        .http2_adaptive_window(true)
});

thread_local! {
    static CLIENT: std::cell::RefCell<Option<EmbedClient<Channel>>> = std::cell::RefCell::new(None);
}

fn get_client() -> Result<EmbedClient<Channel>, Box<dyn std::error::Error>> {
    CLIENT.with(|cell| {
        let mut client_opt = cell.borrow_mut();
        if client_opt.is_none() {
            let channel = RUNTIME.block_on(ENDPOINT.connect())?;
            *client_opt = Some(EmbedClient::new(channel));
        }
        Ok(client_opt.as_ref().unwrap().clone())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn generate_embeddings_from_texts(
    inputs: *const StringSlice,
    n_inputs: usize,
    out_batch: *mut EmbeddingBatch,
) -> c_int {
    if inputs.is_null() || out_batch.is_null() {
        return -1;
    }

    let text_slices: Result<Vec<&str>, _> = unsafe {
        slice::from_raw_parts(inputs, n_inputs)
            .iter()
            .map(|slice| {
                if slice.ptr.is_null() || slice.len == 0 {
                    Err(())
                } else {
                    let bytes = slice::from_raw_parts(slice.ptr as *const u8, slice.len);
                    std::str::from_utf8(bytes).map_err(|_| ())
                }
            })
            .collect()
    };

    let text_slices = match text_slices {
        Ok(v) if !v.is_empty() => v,
        _ => return -2,
    };

    let mut client = match get_client() {
        Ok(c) => c,
        Err(_) => return -3,
    };

    let response = match RUNTIME.block_on(async {
        let request = EmbedBatchRequest {
            inputs: text_slices.iter().map(|&s| s.to_string()).collect(),
            truncate: true,
            normalize: true,
            truncation_direction: 0,
            prompt_name: None,
            dimensions: None,
        };

        client.embed_batch(tonic::Request::new(request)).await
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

    let mut flat: Vec<c_float> = Vec::with_capacity(total);
    unsafe {
        let ptr = flat.as_mut_ptr();
        let mut offset = 0;
        for e in response.embeddings {
            std::ptr::copy_nonoverlapping(e.values.as_ptr(), ptr.add(offset), dim);
            offset += dim;
        }
        flat.set_len(total);
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
