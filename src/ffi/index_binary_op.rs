// FFI Layer: GrB_IndexBinaryOp C API

use crate::ffi::error::*;
use std::panic::catch_unwind;

/// Opaque GrB_IndexBinaryOp handle
#[repr(C)]
pub struct GrB_IndexBinaryOp_opaque {
    _private: [u8; 0],
}

/// GrB_IndexBinaryOp pointer
pub type GrB_IndexBinaryOp = *mut GrB_IndexBinaryOp_opaque;

/// Free an index binary operator
#[no_mangle]
pub unsafe extern "C" fn GrB_IndexBinaryOp_free(op: *mut GrB_IndexBinaryOp) -> GrB_Info {
    let result = catch_unwind(|| {
        if op.is_null() {
            return GrB_NULL_POINTER;
        }
        *op = std::ptr::null_mut();
        GrB_SUCCESS
    });

    result.unwrap_or(GrB_PANIC)
}
