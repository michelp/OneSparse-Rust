// FFI Layer: GrB_Matrix C API

use crate::ffi::error::*;
use crate::ffi::types::GrB_Type;
use std::panic::catch_unwind;

/// Opaque GrB_Matrix handle
#[repr(C)]
pub struct GrB_Matrix_opaque {
    _private: [u8; 0],
}

/// GrB_Matrix pointer
pub type GrB_Matrix = *mut GrB_Matrix_opaque;

/// GrB_Index type (unsigned 64-bit)
pub type GrB_Index = u64;

/// Create a new matrix
#[no_mangle]
pub unsafe extern "C" fn GrB_Matrix_new(
    matrix_out: *mut GrB_Matrix,
    _type_: GrB_Type,
    _nrows: GrB_Index,
    _ncols: GrB_Index,
) -> GrB_Info {
    let result = catch_unwind(|| {
        if matrix_out.is_null() {
            return GrB_NULL_POINTER;
        }

        // TODO: Create matrix based on type_
        // For now, return not implemented
        GrB_NOT_IMPLEMENTED
    });

    result.unwrap_or(GrB_PANIC)
}

/// Free a matrix
#[no_mangle]
pub unsafe extern "C" fn GrB_Matrix_free(matrix: *mut GrB_Matrix) -> GrB_Info {
    let result = catch_unwind(|| {
        if matrix.is_null() {
            return GrB_NULL_POINTER;
        }

        // TODO: Free matrix
        // For now, just set to null
        *matrix = std::ptr::null_mut();
        GrB_SUCCESS
    });

    result.unwrap_or(GrB_PANIC)
}

/// Get number of rows
#[no_mangle]
pub unsafe extern "C" fn GrB_Matrix_nrows(nrows: *mut GrB_Index, matrix: GrB_Matrix) -> GrB_Info {
    let result = catch_unwind(|| {
        if nrows.is_null() || matrix.is_null() {
            return GrB_NULL_POINTER;
        }

        // TODO: Get nrows from matrix
        GrB_NOT_IMPLEMENTED
    });

    result.unwrap_or(GrB_PANIC)
}

/// Get number of columns
#[no_mangle]
pub unsafe extern "C" fn GrB_Matrix_ncols(ncols: *mut GrB_Index, matrix: GrB_Matrix) -> GrB_Info {
    let result = catch_unwind(|| {
        if ncols.is_null() || matrix.is_null() {
            return GrB_NULL_POINTER;
        }

        // TODO: Get ncols from matrix
        GrB_NOT_IMPLEMENTED
    });

    result.unwrap_or(GrB_PANIC)
}

/// Get number of stored values
#[no_mangle]
pub unsafe extern "C" fn GrB_Matrix_nvals(nvals: *mut GrB_Index, matrix: GrB_Matrix) -> GrB_Info {
    let result = catch_unwind(|| {
        if nvals.is_null() || matrix.is_null() {
            return GrB_NULL_POINTER;
        }

        // TODO: Get nvals from matrix
        GrB_NOT_IMPLEMENTED
    });

    result.unwrap_or(GrB_PANIC)
}
