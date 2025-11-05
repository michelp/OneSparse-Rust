// Descriptor Support
//
// Descriptors modify the behavior of GraphBLAS operations.
// Common descriptor fields:
// - Output replace: clear output before writing
// - Mask complement: invert the mask
// - Input transpose: transpose inputs (A, B)
// - Structure only: use structure, ignore values

use crate::core::error::{GraphBlasError, Result};

/// Descriptor for modifying operation behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Descriptor {
    /// Clear output matrix/vector before operation
    pub output_replace: bool,

    /// Use complement of mask (select elements where mask is false/zero)
    pub mask_complement: bool,

    /// Use only structure of mask (ignore values)
    pub mask_structure: bool,

    /// Transpose first input (A in C=A*B)
    pub transpose_first: bool,

    /// Transpose second input (B in C=A*B)
    pub transpose_second: bool,
}

impl Descriptor {
    /// Create a new descriptor with default settings
    pub fn new() -> Self {
        Self {
            output_replace: false,
            mask_complement: false,
            mask_structure: false,
            transpose_first: false,
            transpose_second: false,
        }
    }

    /// Create descriptor with output replace enabled
    pub fn with_output_replace() -> Self {
        Self {
            output_replace: true,
            ..Self::new()
        }
    }

    /// Create descriptor with mask complement enabled
    pub fn with_mask_complement() -> Self {
        Self {
            mask_complement: true,
            ..Self::new()
        }
    }

    /// Create descriptor with mask structure enabled
    pub fn with_mask_structure() -> Self {
        Self {
            mask_structure: true,
            ..Self::new()
        }
    }

    /// Create descriptor with first input transpose enabled
    pub fn with_transpose_first() -> Self {
        Self {
            transpose_first: true,
            ..Self::new()
        }
    }

    /// Create descriptor with second input transpose enabled
    pub fn with_transpose_second() -> Self {
        Self {
            transpose_second: true,
            ..Self::new()
        }
    }

    /// Set output replace
    pub fn set_output_replace(&mut self, value: bool) -> &mut Self {
        self.output_replace = value;
        self
    }

    /// Set mask complement
    pub fn set_mask_complement(&mut self, value: bool) -> &mut Self {
        self.mask_complement = value;
        self
    }

    /// Set mask structure
    pub fn set_mask_structure(&mut self, value: bool) -> &mut Self {
        self.mask_structure = value;
        self
    }

    /// Set transpose first
    pub fn set_transpose_first(&mut self, value: bool) -> &mut Self {
        self.transpose_first = value;
        self
    }

    /// Set transpose second
    pub fn set_transpose_second(&mut self, value: bool) -> &mut Self {
        self.transpose_second = value;
        self
    }

    /// Validate descriptor for specific operation
    pub fn validate_for_matmul(&self) -> Result<()> {
        // All fields are valid for matmul
        Ok(())
    }

    /// Validate descriptor for unary operations
    pub fn validate_for_unary(&self) -> Result<()> {
        // Transpose flags don't make sense for unary operations
        if self.transpose_first || self.transpose_second {
            return Err(GraphBlasError::InvalidValue);
        }
        Ok(())
    }

    /// Validate descriptor for element-wise operations
    pub fn validate_for_ewise(&self) -> Result<()> {
        // Transpose flags don't make sense for element-wise
        if self.transpose_first || self.transpose_second {
            return Err(GraphBlasError::InvalidValue);
        }
        Ok(())
    }
}

impl Default for Descriptor {
    fn default() -> Self {
        Self::new()
    }
}

/// Descriptor fields (for C API compatibility)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorField {
    /// Output replace mode
    OutputReplace,
    /// Mask complement mode
    MaskComplement,
    /// Mask structure only mode
    MaskStructure,
    /// Transpose first input
    TransposeFirst,
    /// Transpose second input
    TransposeSecond,
}

/// Descriptor values (for C API compatibility)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorValue {
    /// Default behavior
    Default,
    /// Enable the feature
    Enable,
    /// Transpose the input
    Transpose,
    /// Use structure only
    Structure,
    /// Use complement
    Complement,
    /// Replace output
    Replace,
}

impl Descriptor {
    /// Set a descriptor field to a value (C API style)
    pub fn set_field(&mut self, field: DescriptorField, value: DescriptorValue) -> Result<()> {
        match (field, value) {
            (DescriptorField::OutputReplace, DescriptorValue::Replace) => {
                self.output_replace = true;
                Ok(())
            }
            (DescriptorField::OutputReplace, DescriptorValue::Default) => {
                self.output_replace = false;
                Ok(())
            }
            (DescriptorField::MaskComplement, DescriptorValue::Complement) => {
                self.mask_complement = true;
                Ok(())
            }
            (DescriptorField::MaskComplement, DescriptorValue::Default) => {
                self.mask_complement = false;
                Ok(())
            }
            (DescriptorField::MaskStructure, DescriptorValue::Structure) => {
                self.mask_structure = true;
                Ok(())
            }
            (DescriptorField::MaskStructure, DescriptorValue::Default) => {
                self.mask_structure = false;
                Ok(())
            }
            (DescriptorField::TransposeFirst, DescriptorValue::Transpose) => {
                self.transpose_first = true;
                Ok(())
            }
            (DescriptorField::TransposeFirst, DescriptorValue::Default) => {
                self.transpose_first = false;
                Ok(())
            }
            (DescriptorField::TransposeSecond, DescriptorValue::Transpose) => {
                self.transpose_second = true;
                Ok(())
            }
            (DescriptorField::TransposeSecond, DescriptorValue::Default) => {
                self.transpose_second = false;
                Ok(())
            }
            _ => Err(GraphBlasError::InvalidValue),
        }
    }

    /// Get a descriptor field value
    pub fn get_field(&self, field: DescriptorField) -> DescriptorValue {
        match field {
            DescriptorField::OutputReplace => {
                if self.output_replace {
                    DescriptorValue::Replace
                } else {
                    DescriptorValue::Default
                }
            }
            DescriptorField::MaskComplement => {
                if self.mask_complement {
                    DescriptorValue::Complement
                } else {
                    DescriptorValue::Default
                }
            }
            DescriptorField::MaskStructure => {
                if self.mask_structure {
                    DescriptorValue::Structure
                } else {
                    DescriptorValue::Default
                }
            }
            DescriptorField::TransposeFirst => {
                if self.transpose_first {
                    DescriptorValue::Transpose
                } else {
                    DescriptorValue::Default
                }
            }
            DescriptorField::TransposeSecond => {
                if self.transpose_second {
                    DescriptorValue::Transpose
                } else {
                    DescriptorValue::Default
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptor_creation() {
        let desc = Descriptor::new();
        assert!(!desc.output_replace);
        assert!(!desc.mask_complement);
        assert!(!desc.mask_structure);
        assert!(!desc.transpose_first);
        assert!(!desc.transpose_second);
    }

    #[test]
    fn test_descriptor_with_output_replace() {
        let desc = Descriptor::with_output_replace();
        assert!(desc.output_replace);
        assert!(!desc.mask_complement);
    }

    #[test]
    fn test_descriptor_with_transpose() {
        let desc = Descriptor::with_transpose_first();
        assert!(desc.transpose_first);
        assert!(!desc.transpose_second);

        let desc = Descriptor::with_transpose_second();
        assert!(!desc.transpose_first);
        assert!(desc.transpose_second);
    }

    #[test]
    fn test_descriptor_chaining() {
        let mut desc = Descriptor::new();
        desc.set_output_replace(true)
            .set_transpose_first(true)
            .set_mask_complement(true);

        assert!(desc.output_replace);
        assert!(desc.transpose_first);
        assert!(desc.mask_complement);
    }

    #[test]
    fn test_descriptor_validation() {
        let desc = Descriptor::with_transpose_first();
        assert!(desc.validate_for_matmul().is_ok());
        assert!(desc.validate_for_unary().is_err()); // Transpose not valid for unary
        assert!(desc.validate_for_ewise().is_err()); // Transpose not valid for ewise
    }

    #[test]
    fn test_descriptor_field_set_get() {
        let mut desc = Descriptor::new();

        desc.set_field(DescriptorField::OutputReplace, DescriptorValue::Replace)
            .unwrap();
        assert_eq!(
            desc.get_field(DescriptorField::OutputReplace),
            DescriptorValue::Replace
        );

        desc.set_field(DescriptorField::TransposeFirst, DescriptorValue::Transpose)
            .unwrap();
        assert_eq!(
            desc.get_field(DescriptorField::TransposeFirst),
            DescriptorValue::Transpose
        );

        desc.set_field(DescriptorField::TransposeFirst, DescriptorValue::Default)
            .unwrap();
        assert_eq!(
            desc.get_field(DescriptorField::TransposeFirst),
            DescriptorValue::Default
        );
    }

    #[test]
    fn test_descriptor_invalid_field_value() {
        let mut desc = Descriptor::new();

        // Invalid combination: OutputReplace with Transpose value
        let result = desc.set_field(DescriptorField::OutputReplace, DescriptorValue::Transpose);
        assert!(result.is_err());
    }
}
