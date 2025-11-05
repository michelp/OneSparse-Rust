// FFI Layer: GrB_Scalar C API

use crate::ffi::error::*;
use crate::ffi::types::GrB_Type;
use std::panic::catch_unwind;

/// Opaque GrB_Scalar handle
#[repr(C)]
pub struct GrB_Scalar_opaque {
    _private: [u8; 0],
}

/// GrB_Scalar pointer
pub type GrB_Scalar = *mut GrB_Scalar_opaque;

/// Create a new scalar
#[no_mangle]
pub unsafe extern "C" fn GrB_Scalar_new(
    scalar_out: *mut GrB_Scalar,
    _type_: GrB_Type,
) -> GrB_Info {
    let result = catch_unwind(|| {
        if scalar_out.is_null() {
            return GrB_NULL_POINTER;
        }
        GrB_NOT_IMPLEMENTED
    });

    result.unwrap_or(GrB_PANIC)
}

/// Free a scalar
#[no_mangle]
pub unsafe extern "C" fn GrB_Scalar_free(scalar: *mut GrB_Scalar) -> GrB_Info {
    let result = catch_unwind(|| {
        if scalar.is_null() {
            return GrB_NULL_POINTER;
        }
        *scalar = std::ptr::null_mut();
        GrB_SUCCESS
    });

    result.unwrap_or(GrB_PANIC)
}
