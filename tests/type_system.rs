// Integration tests for the type system

use rustsparse::types::{GraphBLASType, TypeCode, TypeDescriptor};

#[test]
fn test_builtin_type_codes() {
    assert_eq!(bool::TYPE_CODE, TypeCode::Bool);
    assert_eq!(i8::TYPE_CODE, TypeCode::Int8);
    assert_eq!(i16::TYPE_CODE, TypeCode::Int16);
    assert_eq!(i32::TYPE_CODE, TypeCode::Int32);
    assert_eq!(i64::TYPE_CODE, TypeCode::Int64);
    assert_eq!(u8::TYPE_CODE, TypeCode::Uint8);
    assert_eq!(u16::TYPE_CODE, TypeCode::Uint16);
    assert_eq!(u32::TYPE_CODE, TypeCode::Uint32);
    assert_eq!(u64::TYPE_CODE, TypeCode::Uint64);
    assert_eq!(f32::TYPE_CODE, TypeCode::Fp32);
    assert_eq!(f64::TYPE_CODE, TypeCode::Fp64);
}

#[test]
fn test_type_sizes() {
    assert_eq!(TypeCode::Bool.size(), std::mem::size_of::<bool>());
    assert_eq!(TypeCode::Int8.size(), 1);
    assert_eq!(TypeCode::Int16.size(), 2);
    assert_eq!(TypeCode::Int32.size(), 4);
    assert_eq!(TypeCode::Int64.size(), 8);
    assert_eq!(TypeCode::Uint8.size(), 1);
    assert_eq!(TypeCode::Uint16.size(), 2);
    assert_eq!(TypeCode::Uint32.size(), 4);
    assert_eq!(TypeCode::Uint64.size(), 8);
    assert_eq!(TypeCode::Fp32.size(), 4);
    assert_eq!(TypeCode::Fp64.size(), 8);
}

#[test]
fn test_type_names() {
    assert_eq!(TypeCode::Bool.name(), "bool");
    assert_eq!(TypeCode::Int32.name(), "int32");
    assert_eq!(TypeCode::Int64.name(), "int64");
    assert_eq!(TypeCode::Fp32.name(), "float32");
    assert_eq!(TypeCode::Fp64.name(), "float64");
}

#[test]
fn test_type_descriptors() {
    let desc_i32 = i32::descriptor();
    assert_eq!(desc_i32.code, TypeCode::Int32);
    assert_eq!(desc_i32.size, 4);
    assert_eq!(desc_i32.name, "int32");

    let desc_f64 = f64::descriptor();
    assert_eq!(desc_f64.code, TypeCode::Fp64);
    assert_eq!(desc_f64.size, 8);
    assert_eq!(desc_f64.name, "float64");

    let desc_bool = bool::descriptor();
    assert_eq!(desc_bool.code, TypeCode::Bool);
    assert_eq!(desc_bool.size, std::mem::size_of::<bool>());
    assert_eq!(desc_bool.name, "bool");
}

#[test]
fn test_user_defined_type() {
    #[derive(Copy, Clone)]
    struct MyType {
        x: f64,
        y: f64,
    }

    let desc = TypeDescriptor::new_user_defined::<MyType>("MyType".to_string());

    assert_eq!(desc.size, std::mem::size_of::<MyType>());
    assert_eq!(desc.name, "MyType");

    // User-defined types should have UserDefined variant
    match desc.code {
        TypeCode::UserDefined(_) => {}, // Expected
        _ => panic!("Expected UserDefined type code"),
    }
}

#[test]
fn test_multiple_user_defined_types() {
    #[derive(Copy, Clone)]
    struct Type1 { x: i32 }

    #[derive(Copy, Clone)]
    struct Type2 { y: f64 }

    let desc1 = TypeDescriptor::new_user_defined::<Type1>("Type1".to_string());
    let desc2 = TypeDescriptor::new_user_defined::<Type2>("Type2".to_string());

    // Each should have a unique code
    assert_ne!(desc1.code, desc2.code);

    // But both should be UserDefined
    match (desc1.code, desc2.code) {
        (TypeCode::UserDefined(_), TypeCode::UserDefined(_)) => {}, // Expected
        _ => panic!("Expected UserDefined type codes"),
    }
}
