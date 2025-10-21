mod fastembed;
mod remote;

use crate::fastembed::generate_embeddings_fastembed;
use crate::remote::generate_embeddings_remote;
use anyhow::Result;
use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int};
use std::slice;

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

const ERR_INVALID_POINTERS: c_int = -1;
const ERR_EMPTY_INPUT: c_int = -2;
const ERR_INVALID_UTF8: c_int = -3;
const ERR_INVALID_METHOD: c_int = -4;
const ERR_EMBEDDING_FAILED: c_int = -5;

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

#[repr(C)]
pub enum EmbedMethod {
    FastEmbed = 0,
    Remote = 1,
}

#[unsafe(no_mangle)]
pub extern "C" fn generate_embeddings_from_texts(
    method: c_int,
    model: *const c_char,
    inputs: *const StringSlice,
    n_inputs: usize,
    out_batch: *mut EmbeddingBatch,
) -> c_int {
    if inputs.is_null() || out_batch.is_null() || model.is_null() {
        return ERR_INVALID_POINTERS;
    }

    let model_str = unsafe {
        match CStr::from_ptr(model).to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_UTF8,
        }
    };

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
        Ok(_) => return ERR_EMPTY_INPUT,
        Err(_) => return ERR_INVALID_UTF8,
    };

    let result: Result<(Vec<f32>, usize, usize)> = match method {
        0 => generate_embeddings_fastembed(model_str, text_slices),
        1 => generate_embeddings_remote(model_str, text_slices),
        _ => return ERR_INVALID_METHOD,
    };

    let (mut flat, n_vectors, dim) = match result {
        Ok((flat, n_vectors, dim)) if n_vectors > 0 && !flat.is_empty() => (flat, n_vectors, dim),
        _ => return ERR_EMBEDDING_FAILED,
    };

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
        let b = &mut *batch;
        if !b.data.is_null() && b.n_vectors > 0 && b.dim > 0 {
            let total = b.n_vectors * b.dim;
            drop(Vec::from_raw_parts(b.data, total, total));
            b.data = std::ptr::null_mut();
            b.n_vectors = 0;
            b.dim = 0;
        }
    }
}
