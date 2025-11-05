// FFI Layer: GrB_IndexUnaryOp C API

use crate::ffi::error::*;
use std::panic::catch_unwind;

/// Opaque GrB_IndexUnaryOp handle
#[repr(C)]
pub struct GrB_IndexUnaryOp_opaque {
    _private: [u8; 0],
}

/// GrB_IndexUnaryOp pointer
pub type GrB_IndexUnaryOp = *mut GrB_IndexUnaryOp_opaque;

/// Free an index unary operator
#[no_mangle]
pub unsafe extern "C" fn GrB_IndexUnaryOp_free(op: *mut GrB_IndexUnaryOp) -> GrB_Info {
    let result = catch_unwind(|| {
        if op.is_null() {
            return GrB_NULL_POINTER;
        }
        *op = std::ptr::null_mut();
        GrB_SUCCESS
    });

    result.unwrap_or(GrB_PANIC)
}
