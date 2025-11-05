// FFI Layer: GrB_BinaryOp C API

use crate::ffi::error::*;
use std::panic::catch_unwind;

/// Opaque GrB_BinaryOp handle
#[repr(C)]
pub struct GrB_BinaryOp_opaque {
    _private: [u8; 0],
}

/// GrB_BinaryOp pointer
pub type GrB_BinaryOp = *mut GrB_BinaryOp_opaque;

/// Free a binary operator
#[no_mangle]
pub unsafe extern "C" fn GrB_BinaryOp_free(op: *mut GrB_BinaryOp) -> GrB_Info {
    let result = catch_unwind(|| {
        if op.is_null() {
            return GrB_NULL_POINTER;
        }
        *op = std::ptr::null_mut();
        GrB_SUCCESS
    });

    result.unwrap_or(GrB_PANIC)
}
