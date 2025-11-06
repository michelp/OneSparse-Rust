// FFI Layer: GrB_Vector C API

use crate::ffi::error::*;
use crate::ffi::matrix::GrB_Index;
use crate::ffi::types::GrB_Type;
use std::panic::catch_unwind;

/// Opaque GrB_Vector handle
#[repr(C)]
pub struct GrB_Vector_opaque {
    _private: [u8; 0],
}

/// GrB_Vector pointer
pub type GrB_Vector = *mut GrB_Vector_opaque;

/// Create a new vector
#[no_mangle]
pub unsafe extern "C" fn GrB_Vector_new(
    vector_out: *mut GrB_Vector,
    _type_: GrB_Type,
    _size: GrB_Index,
) -> GrB_Info {
    let result = catch_unwind(|| {
        if vector_out.is_null() {
            return GrB_NULL_POINTER;
        }
        GrB_NOT_IMPLEMENTED
    });

    result.unwrap_or(GrB_PANIC)
}

/// Free a vector
#[no_mangle]
pub unsafe extern "C" fn GrB_Vector_free(vector: *mut GrB_Vector) -> GrB_Info {
    let result = catch_unwind(|| {
        if vector.is_null() {
            return GrB_NULL_POINTER;
        }
        *vector = std::ptr::null_mut();
        GrB_SUCCESS
    });

    result.unwrap_or(GrB_PANIC)
}

/// Get vector size
#[no_mangle]
pub unsafe extern "C" fn GrB_Vector_size(size: *mut GrB_Index, vector: GrB_Vector) -> GrB_Info {
    let result = catch_unwind(|| {
        if size.is_null() || vector.is_null() {
            return GrB_NULL_POINTER;
        }
        GrB_NOT_IMPLEMENTED
    });

    result.unwrap_or(GrB_PANIC)
}

/// Get number of stored values
#[no_mangle]
pub unsafe extern "C" fn GrB_Vector_nvals(nvals: *mut GrB_Index, vector: GrB_Vector) -> GrB_Info {
    let result = catch_unwind(|| {
        if nvals.is_null() || vector.is_null() {
            return GrB_NULL_POINTER;
        }
        GrB_NOT_IMPLEMENTED
    });

    result.unwrap_or(GrB_PANIC)
}
