// FFI Layer: GrB_Type C API
//
// Opaque type handles for the GraphBLAS type system.
// These are exposed to C code as opaque pointers.

use crate::core::error::GraphBlasError;
use crate::core::handles::HandleRegistry;
use crate::ffi::error::*;
use crate::types::TypeDescriptor;
use std::panic::catch_unwind;

/// Opaque GrB_Type handle
#[repr(C)]
pub struct GrB_Type_opaque {
    _private: [u8; 0],
}

/// GrB_Type pointer (opaque handle to type descriptor)
pub type GrB_Type = *mut GrB_Type_opaque;

// Predefined built-in type constants
// These are implemented as constant handles that map to built-in types

/// Boolean type
pub const GrB_BOOL: GrB_Type = 1 as GrB_Type;

/// Signed 8-bit integer
pub const GrB_INT8: GrB_Type = 2 as GrB_Type;

/// Signed 16-bit integer
pub const GrB_INT16: GrB_Type = 3 as GrB_Type;

/// Signed 32-bit integer
pub const GrB_INT32: GrB_Type = 4 as GrB_Type;

/// Signed 64-bit integer
pub const GrB_INT64: GrB_Type = 5 as GrB_Type;

/// Unsigned 8-bit integer
pub const GrB_UINT8: GrB_Type = 6 as GrB_Type;

/// Unsigned 16-bit integer
pub const GrB_UINT16: GrB_Type = 7 as GrB_Type;

/// Unsigned 32-bit integer
pub const GrB_UINT32: GrB_Type = 8 as GrB_Type;

/// Unsigned 64-bit integer
pub const GrB_UINT64: GrB_Type = 9 as GrB_Type;

/// 32-bit floating point
pub const GrB_FP32: GrB_Type = 10 as GrB_Type;

/// 64-bit floating point
pub const GrB_FP64: GrB_Type = 11 as GrB_Type;

// Handle registry for user-defined types
lazy_static::lazy_static! {
    static ref TYPE_REGISTRY: HandleRegistry<TypeDescriptor> = HandleRegistry::new();
}

/// Create a new user-defined type
///
/// # Safety
/// This function must be called from C with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn GrB_Type_new(
    type_: *mut GrB_Type,
    sizeof_ctype: usize,
) -> GrB_Info {
    let result = catch_unwind(|| {
        // Validate output pointer
        if type_.is_null() {
            return GrB_NULL_POINTER;
        }

        // Create a user-defined type descriptor
        // Note: We can't know the Rust type at runtime, so we use a placeholder
        let descriptor = TypeDescriptor::new_user_defined::<u8>(
            format!("user_type_{}", sizeof_ctype)
        );

        // Register and get handle
        let handle = TYPE_REGISTRY.insert(descriptor);

        // Write handle as opaque pointer
        *type_ = handle as GrB_Type;

        GrB_SUCCESS
    });

    result.unwrap_or(GrB_PANIC)
}

/// Free a type object
///
/// # Safety
/// This function must be called from C with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn GrB_Type_free(type_: *mut GrB_Type) -> GrB_Info {
    let result = catch_unwind(|| {
        // Validate pointer
        if type_.is_null() {
            return GrB_NULL_POINTER;
        }

        let handle = *type_ as usize;

        // Don't free built-in types (handles 1-11)
        if handle > 0 && handle <= 11 {
            *type_ = std::ptr::null_mut();
            return GrB_SUCCESS;
        }

        // Remove from registry
        match TYPE_REGISTRY.remove(handle) {
            Ok(_) => {
                *type_ = std::ptr::null_mut();
                GrB_SUCCESS
            }
            Err(e) => e.to_grb_info(),
        }
    });

    result.unwrap_or(GrB_PANIC)
}

/// Helper function to validate a type handle
pub fn validate_type_handle(type_: GrB_Type) -> Result<(), GraphBlasError> {
    let handle = type_ as usize;

    // Check for null
    if handle == 0 {
        return Err(GraphBlasError::NullPointer);
    }

    // Built-in types (1-11) are always valid
    if handle >= 1 && handle <= 11 {
        return Ok(());
    }

    // Check if user-defined type exists
    if TYPE_REGISTRY.contains(handle) {
        Ok(())
    } else {
        Err(GraphBlasError::UninitializedObject)
    }
}

/// Get the size of a type
pub fn get_type_size(type_: GrB_Type) -> Result<usize, GraphBlasError> {
    let handle = type_ as usize;

    // Built-in types
    match handle {
        1 => Ok(std::mem::size_of::<bool>()),
        2 => Ok(std::mem::size_of::<i8>()),
        3 => Ok(std::mem::size_of::<i16>()),
        4 => Ok(std::mem::size_of::<i32>()),
        5 => Ok(std::mem::size_of::<i64>()),
        6 => Ok(std::mem::size_of::<u8>()),
        7 => Ok(std::mem::size_of::<u16>()),
        8 => Ok(std::mem::size_of::<u32>()),
        9 => Ok(std::mem::size_of::<u64>()),
        10 => Ok(std::mem::size_of::<f32>()),
        11 => Ok(std::mem::size_of::<f64>()),
        _ => {
            // User-defined type
            let desc = TYPE_REGISTRY.get(handle)?;
            Ok(desc.size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_type_sizes() {
        assert_eq!(get_type_size(GrB_BOOL).unwrap(), std::mem::size_of::<bool>());
        assert_eq!(get_type_size(GrB_INT32).unwrap(), 4);
        assert_eq!(get_type_size(GrB_INT64).unwrap(), 8);
        assert_eq!(get_type_size(GrB_FP32).unwrap(), 4);
        assert_eq!(get_type_size(GrB_FP64).unwrap(), 8);
    }

    #[test]
    fn test_validate_builtin_types() {
        assert!(validate_type_handle(GrB_BOOL).is_ok());
        assert!(validate_type_handle(GrB_INT32).is_ok());
        assert!(validate_type_handle(GrB_FP64).is_ok());
    }

    #[test]
    fn test_null_type() {
        assert!(validate_type_handle(std::ptr::null_mut()).is_err());
        assert_eq!(
            get_type_size(std::ptr::null_mut()).unwrap_err(),
            GraphBlasError::NullPointer
        );
    }
}
