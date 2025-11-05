// FFI Layer: GrB_UnaryOp C API

use crate::ffi::error::*;
use std::panic::catch_unwind;

/// Opaque GrB_UnaryOp handle
#[repr(C)]
pub struct GrB_UnaryOp_opaque {
    _private: [u8; 0],
}

/// GrB_UnaryOp pointer
pub type GrB_UnaryOp = *mut GrB_UnaryOp_opaque;

/// Free a unary operator
#[no_mangle]
pub unsafe extern "C" fn GrB_UnaryOp_free(op: *mut GrB_UnaryOp) -> GrB_Info {
    let result = catch_unwind(|| {
        if op.is_null() {
            return GrB_NULL_POINTER;
        }
        *op = std::ptr::null_mut();
        GrB_SUCCESS
    });

    result.unwrap_or(GrB_PANIC)
}
