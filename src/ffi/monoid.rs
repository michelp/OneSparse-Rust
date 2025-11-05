// FFI Layer: GrB_Monoid C API

use crate::ffi::error::*;
use std::panic::catch_unwind;

/// Opaque GrB_Monoid handle
#[repr(C)]
pub struct GrB_Monoid_opaque {
    _private: [u8; 0],
}

/// GrB_Monoid pointer
pub type GrB_Monoid = *mut GrB_Monoid_opaque;

/// Free a monoid
#[no_mangle]
pub unsafe extern "C" fn GrB_Monoid_free(monoid: *mut GrB_Monoid) -> GrB_Info {
    let result = catch_unwind(|| {
        if monoid.is_null() {
            return GrB_NULL_POINTER;
        }
        *monoid = std::ptr::null_mut();
        GrB_SUCCESS
    });

    result.unwrap_or(GrB_PANIC)
}
