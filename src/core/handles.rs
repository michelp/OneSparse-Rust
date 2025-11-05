// Handle Management System: Safe bridge between C opaque pointers and Rust objects
//
// This module provides a handle-based system for managing Rust objects accessed through
// C API opaque pointers. Instead of dereferencing opaque pointers (which would break
// link compatibility), we use integer handles that index into a registry of Arc<T> objects.

use crate::core::error::{GraphBlasError, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Generic handle registry for a specific object type
pub struct HandleRegistry<T> {
    handles: Mutex<HashMap<usize, Arc<T>>>,
    next_handle: Mutex<usize>,
}

impl<T> HandleRegistry<T> {
    /// Create a new handle registry
    pub fn new() -> Self {
        Self {
            handles: Mutex::new(HashMap::new()),
            next_handle: Mutex::new(1), // Start at 1 (NULL is 0)
        }
    }

    /// Register a new object and return its handle
    ///
    /// The returned handle can be safely cast to an opaque C pointer.
    /// The object is wrapped in Arc for shared ownership and thread safety.
    pub fn insert(&self, object: T) -> usize {
        let arc = Arc::new(object);
        let mut next = self.next_handle.lock().unwrap();
        let handle = *next;
        *next += 1;

        let mut handles = self.handles.lock().unwrap();
        handles.insert(handle, arc);

        handle
    }

    /// Get a reference to an object by its handle
    ///
    /// Returns an Arc clone, incrementing the reference count.
    /// Returns error if handle is invalid.
    pub fn get(&self, handle: usize) -> Result<Arc<T>> {
        if handle == 0 {
            return Err(GraphBlasError::NullPointer);
        }

        let handles = self.handles.lock().unwrap();
        handles
            .get(&handle)
            .cloned()
            .ok_or(GraphBlasError::UninitializedObject)
    }

    /// Remove an object by its handle
    ///
    /// Returns the Arc to the object if found.
    /// The object will be dropped when all Arc references are released.
    pub fn remove(&self, handle: usize) -> Result<Arc<T>> {
        if handle == 0 {
            return Err(GraphBlasError::NullPointer);
        }

        let mut handles = self.handles.lock().unwrap();
        handles
            .remove(&handle)
            .ok_or(GraphBlasError::UninitializedObject)
    }

    /// Check if a handle is valid
    pub fn contains(&self, handle: usize) -> bool {
        if handle == 0 {
            return false;
        }
        let handles = self.handles.lock().unwrap();
        handles.contains_key(&handle)
    }

    /// Get the number of registered handles
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        let handles = self.handles.lock().unwrap();
        handles.len()
    }

    /// Check if the registry is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        let handles = self.handles.lock().unwrap();
        handles.is_empty()
    }
}

impl<T> Default for HandleRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro to create a global handle registry for a specific type
#[macro_export]
macro_rules! define_handle_registry {
    ($registry_name:ident, $type:ty) => {
        lazy_static::lazy_static! {
            static ref $registry_name: $crate::core::handles::HandleRegistry<$type> =
                $crate::core::handles::HandleRegistry::new();
        }
    };
}

/// Helper macro to convert C pointer to handle
#[macro_export]
macro_rules! ptr_to_handle {
    ($ptr:expr) => {
        $ptr as usize
    };
}

/// Helper macro to convert handle to C pointer
#[macro_export]
macro_rules! handle_to_ptr {
    ($handle:expr, $ptr_type:ty) => {
        $handle as $ptr_type
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_registry() {
        let registry = HandleRegistry::<i32>::new();

        // Insert an object
        let handle = registry.insert(42);
        assert_ne!(handle, 0);

        // Get the object
        let obj = registry.get(handle).unwrap();
        assert_eq!(*obj, 42);

        // Remove the object
        let removed = registry.remove(handle).unwrap();
        assert_eq!(*removed, 42);

        // Getting removed handle should fail
        assert!(registry.get(handle).is_err());
    }

    #[test]
    fn test_null_handle() {
        let registry = HandleRegistry::<i32>::new();

        assert!(registry.get(0).is_err());
        assert!(registry.remove(0).is_err());
        assert!(!registry.contains(0));
    }

    #[test]
    fn test_invalid_handle() {
        let registry = HandleRegistry::<i32>::new();

        assert!(registry.get(999).is_err());
        assert!(!registry.contains(999));
    }

    #[test]
    fn test_multiple_references() {
        let registry = HandleRegistry::<Vec<i32>>::new();

        let handle = registry.insert(vec![1, 2, 3]);

        // Get multiple references
        let ref1 = registry.get(handle).unwrap();
        let ref2 = registry.get(handle).unwrap();

        assert_eq!(*ref1, vec![1, 2, 3]);
        assert_eq!(*ref2, vec![1, 2, 3]);

        // Both references should point to the same object
        assert_eq!(Arc::strong_count(&ref1), 3); // registry + ref1 + ref2
    }
}
