// FFI Layer: GraphBLAS C API Error Codes
// GrB_Info return type for all C API functions

/// GraphBLAS return status codes
pub type GrB_Info = i32;

// Success codes (>= 0)
pub const GrB_SUCCESS: GrB_Info = 0;
pub const GrB_NO_VALUE: GrB_Info = 1; // Requested value not present (sparse structure)

// Standard GraphBLAS C API error codes (< 0)
pub const GrB_UNINITIALIZED_OBJECT: GrB_Info = -1;
pub const GrB_NULL_POINTER: GrB_Info = -2;
pub const GrB_INVALID_VALUE: GrB_Info = -3;
pub const GrB_INVALID_INDEX: GrB_Info = -4;
pub const GrB_DOMAIN_MISMATCH: GrB_Info = -5;
pub const GrB_DIMENSION_MISMATCH: GrB_Info = -6;
pub const GrB_OUTPUT_NOT_EMPTY: GrB_Info = -7;
pub const GrB_NOT_IMPLEMENTED: GrB_Info = -8;

// Additional error codes
pub const GrB_PANIC: GrB_Info = -101; // Rust panic caught at FFI boundary
pub const GrB_OUT_OF_MEMORY: GrB_Info = -102;
pub const GrB_INSUFFICIENT_SPACE: GrB_Info = -103;
pub const GrB_INVALID_OBJECT: GrB_Info = -104;
pub const GrB_INDEX_OUT_OF_BOUNDS: GrB_Info = -105;
pub const GrB_EMPTY_OBJECT: GrB_Info = -106;

/// Helper to get error message for a GrB_Info code
pub fn grb_info_to_string(info: GrB_Info) -> &'static str {
    match info {
        GrB_SUCCESS => "GrB_SUCCESS: operation completed successfully",
        GrB_NO_VALUE => "GrB_NO_VALUE: requested value not present in sparse structure",
        GrB_UNINITIALIZED_OBJECT => "GrB_UNINITIALIZED_OBJECT: object has not been initialized",
        GrB_NULL_POINTER => "GrB_NULL_POINTER: input pointer is NULL",
        GrB_INVALID_VALUE => "GrB_INVALID_VALUE: invalid parameter value",
        GrB_INVALID_INDEX => "GrB_INVALID_INDEX: index is not valid",
        GrB_DOMAIN_MISMATCH => "GrB_DOMAIN_MISMATCH: type mismatch between operands",
        GrB_DIMENSION_MISMATCH => "GrB_DIMENSION_MISMATCH: dimension mismatch between operands",
        GrB_OUTPUT_NOT_EMPTY => "GrB_OUTPUT_NOT_EMPTY: output object must be empty",
        GrB_NOT_IMPLEMENTED => "GrB_NOT_IMPLEMENTED: method not implemented",
        GrB_PANIC => "GrB_PANIC: internal panic caught at FFI boundary",
        GrB_OUT_OF_MEMORY => "GrB_OUT_OF_MEMORY: memory allocation failed",
        GrB_INSUFFICIENT_SPACE => "GrB_INSUFFICIENT_SPACE: insufficient space in output",
        GrB_INVALID_OBJECT => "GrB_INVALID_OBJECT: object is invalid or corrupted",
        GrB_INDEX_OUT_OF_BOUNDS => "GrB_INDEX_OUT_OF_BOUNDS: index exceeds valid bounds",
        GrB_EMPTY_OBJECT => "GrB_EMPTY_OBJECT: object is empty",
        _ => "Unknown GrB_Info code",
    }
}
