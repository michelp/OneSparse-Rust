// FFI Layer: GrB_Semiring C API

use crate::ffi::error::*;
use std::panic::catch_unwind;

/// Opaque GrB_Semiring handle
#[repr(C)]
pub struct GrB_Semiring_opaque {
    _private: [u8; 0],
}

/// GrB_Semiring pointer
pub type GrB_Semiring = *mut GrB_Semiring_opaque;

/// Free a semiring
#[no_mangle]
pub unsafe extern "C" fn GrB_Semiring_free(semiring: *mut GrB_Semiring) -> GrB_Info {
    let result = catch_unwind(|| {
        if semiring.is_null() {
            return GrB_NULL_POINTER;
        }
        *semiring = std::ptr::null_mut();
        GrB_SUCCESS
    });

    result.unwrap_or(GrB_PANIC)
}
