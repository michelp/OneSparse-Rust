// Type System: Bridge between runtime type descriptors and compile-time generic types
//
// GraphBLAS uses runtime type descriptors (GrB_Type) while Rust has compile-time generics.
// This module bridges the two worlds using:
// 1. TypeCode - Runtime enum representation
// 2. GraphBLASType trait - Compile-time type information
// 3. TypeDescriptor - Runtime type metadata
// 4. Type registry - Mapping between codes and descriptors

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    /// Global registry for user-defined types
    static ref TYPE_REGISTRY: Mutex<HashMap<u64, TypeDescriptor>> = Mutex::new(HashMap::new());

    /// Counter for user-defined type codes
    static ref NEXT_USER_TYPE_CODE: Mutex<u64> = Mutex::new(1000);
}

/// Runtime type code enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeCode {
    /// Boolean type
    Bool,
    /// Signed 8-bit integer
    Int8,
    /// Signed 16-bit integer
    Int16,
    /// Signed 32-bit integer
    Int32,
    /// Signed 64-bit integer
    Int64,
    /// Unsigned 8-bit integer
    Uint8,
    /// Unsigned 16-bit integer
    Uint16,
    /// Unsigned 32-bit integer
    Uint32,
    /// Unsigned 64-bit integer
    Uint64,
    /// 32-bit floating point
    Fp32,
    /// 64-bit floating point
    Fp64,
    /// User-defined type with unique code
    UserDefined(u64),
}

impl TypeCode {
    /// Get the size in bytes for this type code
    pub fn size(&self) -> usize {
        match self {
            TypeCode::Bool => std::mem::size_of::<bool>(),
            TypeCode::Int8 => std::mem::size_of::<i8>(),
            TypeCode::Int16 => std::mem::size_of::<i16>(),
            TypeCode::Int32 => std::mem::size_of::<i32>(),
            TypeCode::Int64 => std::mem::size_of::<i64>(),
            TypeCode::Uint8 => std::mem::size_of::<u8>(),
            TypeCode::Uint16 => std::mem::size_of::<u16>(),
            TypeCode::Uint32 => std::mem::size_of::<u32>(),
            TypeCode::Uint64 => std::mem::size_of::<u64>(),
            TypeCode::Fp32 => std::mem::size_of::<f32>(),
            TypeCode::Fp64 => std::mem::size_of::<f64>(),
            TypeCode::UserDefined(code) => {
                // Look up size in registry
                let registry = TYPE_REGISTRY.lock().unwrap();
                registry.get(code).map(|desc| desc.size).unwrap_or(0)
            }
        }
    }

    /// Get the alignment in bytes for this type code
    pub fn alignment(&self) -> usize {
        match self {
            TypeCode::Bool => std::mem::align_of::<bool>(),
            TypeCode::Int8 => std::mem::align_of::<i8>(),
            TypeCode::Int16 => std::mem::align_of::<i16>(),
            TypeCode::Int32 => std::mem::align_of::<i32>(),
            TypeCode::Int64 => std::mem::align_of::<i64>(),
            TypeCode::Uint8 => std::mem::align_of::<u8>(),
            TypeCode::Uint16 => std::mem::align_of::<u16>(),
            TypeCode::Uint32 => std::mem::align_of::<u32>(),
            TypeCode::Uint64 => std::mem::align_of::<u64>(),
            TypeCode::Fp32 => std::mem::align_of::<f32>(),
            TypeCode::Fp64 => std::mem::align_of::<f64>(),
            TypeCode::UserDefined(code) => {
                let registry = TYPE_REGISTRY.lock().unwrap();
                registry.get(code).map(|desc| desc.alignment).unwrap_or(1)
            }
        }
    }

    /// Get human-readable name for this type code
    pub fn name(&self) -> String {
        match self {
            TypeCode::Bool => "bool".to_string(),
            TypeCode::Int8 => "int8".to_string(),
            TypeCode::Int16 => "int16".to_string(),
            TypeCode::Int32 => "int32".to_string(),
            TypeCode::Int64 => "int64".to_string(),
            TypeCode::Uint8 => "uint8".to_string(),
            TypeCode::Uint16 => "uint16".to_string(),
            TypeCode::Uint32 => "uint32".to_string(),
            TypeCode::Uint64 => "uint64".to_string(),
            TypeCode::Fp32 => "float32".to_string(),
            TypeCode::Fp64 => "float64".to_string(),
            TypeCode::UserDefined(code) => {
                let registry = TYPE_REGISTRY.lock().unwrap();
                registry
                    .get(code)
                    .map(|desc| desc.name.clone())
                    .unwrap_or_else(|| format!("user_type_{}", code))
            }
        }
    }
}

/// Runtime type descriptor
#[derive(Debug, Clone)]
pub struct TypeDescriptor {
    /// Type code
    pub code: TypeCode,
    /// Size in bytes
    pub size: usize,
    /// Alignment in bytes
    pub alignment: usize,
    /// Human-readable name
    pub name: String,
    /// Rust TypeId for runtime type checking
    pub type_id: TypeId,
}

impl TypeDescriptor {
    /// Create a new type descriptor
    pub fn new<T: GraphBLASType>() -> Self {
        Self {
            code: T::TYPE_CODE,
            size: std::mem::size_of::<T>(),
            alignment: std::mem::align_of::<T>(),
            name: T::type_name(),
            type_id: TypeId::of::<T>(),
        }
    }

    /// Create a user-defined type descriptor
    pub fn new_user_defined<T: 'static>(name: String) -> Self {
        let mut next_code = NEXT_USER_TYPE_CODE.lock().unwrap();
        let code = *next_code;
        *next_code += 1;

        let descriptor = Self {
            code: TypeCode::UserDefined(code),
            size: std::mem::size_of::<T>(),
            alignment: std::mem::align_of::<T>(),
            name,
            type_id: TypeId::of::<T>(),
        };

        // Register the type
        let mut registry = TYPE_REGISTRY.lock().unwrap();
        registry.insert(code, descriptor.clone());

        descriptor
    }
}

/// Trait for types that can be used with GraphBLAS
///
/// This trait provides compile-time type information that bridges
/// to the runtime type system used by the C API.
pub trait GraphBLASType: Copy + 'static {
    /// The runtime type code for this type
    const TYPE_CODE: TypeCode;

    /// Get human-readable type name
    fn type_name() -> String {
        Self::TYPE_CODE.name()
    }

    /// Get type descriptor
    fn descriptor() -> TypeDescriptor {
        TypeDescriptor::new::<Self>()
    }
}

// Implementations for built-in types

impl GraphBLASType for bool {
    const TYPE_CODE: TypeCode = TypeCode::Bool;
}

impl GraphBLASType for i8 {
    const TYPE_CODE: TypeCode = TypeCode::Int8;
}

impl GraphBLASType for i16 {
    const TYPE_CODE: TypeCode = TypeCode::Int16;
}

impl GraphBLASType for i32 {
    const TYPE_CODE: TypeCode = TypeCode::Int32;
}

impl GraphBLASType for i64 {
    const TYPE_CODE: TypeCode = TypeCode::Int64;
}

impl GraphBLASType for u8 {
    const TYPE_CODE: TypeCode = TypeCode::Uint8;
}

impl GraphBLASType for u16 {
    const TYPE_CODE: TypeCode = TypeCode::Uint16;
}

impl GraphBLASType for u32 {
    const TYPE_CODE: TypeCode = TypeCode::Uint32;
}

impl GraphBLASType for u64 {
    const TYPE_CODE: TypeCode = TypeCode::Uint64;
}

impl GraphBLASType for f32 {
    const TYPE_CODE: TypeCode = TypeCode::Fp32;
}

impl GraphBLASType for f64 {
    const TYPE_CODE: TypeCode = TypeCode::Fp64;
}

/// Helper function to check if two types match at runtime
pub fn types_match(code1: TypeCode, code2: TypeCode) -> bool {
    code1 == code2
}

/// Get type descriptor for a given type code
pub fn get_type_descriptor(code: TypeCode) -> Option<TypeDescriptor> {
    match code {
        TypeCode::Bool => Some(TypeDescriptor::new::<bool>()),
        TypeCode::Int8 => Some(TypeDescriptor::new::<i8>()),
        TypeCode::Int16 => Some(TypeDescriptor::new::<i16>()),
        TypeCode::Int32 => Some(TypeDescriptor::new::<i32>()),
        TypeCode::Int64 => Some(TypeDescriptor::new::<i64>()),
        TypeCode::Uint8 => Some(TypeDescriptor::new::<u8>()),
        TypeCode::Uint16 => Some(TypeDescriptor::new::<u16>()),
        TypeCode::Uint32 => Some(TypeDescriptor::new::<u32>()),
        TypeCode::Uint64 => Some(TypeDescriptor::new::<u64>()),
        TypeCode::Fp32 => Some(TypeDescriptor::new::<f32>()),
        TypeCode::Fp64 => Some(TypeDescriptor::new::<f64>()),
        TypeCode::UserDefined(code) => {
            let registry = TYPE_REGISTRY.lock().unwrap();
            registry.get(&code).cloned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_codes() {
        assert_eq!(TypeCode::Bool.size(), std::mem::size_of::<bool>());
        assert_eq!(TypeCode::Int32.size(), std::mem::size_of::<i32>());
        assert_eq!(TypeCode::Fp64.size(), std::mem::size_of::<f64>());
    }

    #[test]
    fn test_type_trait() {
        assert_eq!(i32::TYPE_CODE, TypeCode::Int32);
        assert_eq!(f64::TYPE_CODE, TypeCode::Fp64);
        assert_eq!(bool::TYPE_CODE, TypeCode::Bool);
    }

    #[test]
    fn test_type_names() {
        assert_eq!(i32::type_name(), "int32");
        assert_eq!(f64::type_name(), "float64");
        assert_eq!(bool::type_name(), "bool");
    }

    #[test]
    fn test_type_descriptor() {
        let desc = i32::descriptor();
        assert_eq!(desc.code, TypeCode::Int32);
        assert_eq!(desc.size, 4);
        assert_eq!(desc.name, "int32");
    }
}
