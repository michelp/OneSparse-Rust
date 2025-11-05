// Kernel Cache: Cache compiled kernels to avoid recompilation
//
// Cache key is based on: (IR structure hash, element types, dimensions)

use crate::compiler::backend::CompiledFunction;
use crate::ir::{IRGraph, NodeId};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Cache key for compiled kernels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Hash of the IR graph structure
    ir_hash: [u8; 32],
    /// Dimension information (if concrete)
    dimensions: Vec<(NodeId, DimensionInfo)>,
}

/// Dimension information for a node
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimensionInfo {
    Scalar,
    Vector(Option<usize>),  // None if symbolic
    Matrix(Option<usize>, Option<usize>),  // None if symbolic
}

impl CacheKey {
    /// Create a cache key from an IR graph
    pub fn from_graph(graph: &IRGraph) -> Self {
        // Compute hash of graph structure
        let ir_hash = Self::hash_graph(graph);

        // Extract dimension information
        let mut dimensions = Vec::new();
        for (node_id, node) in graph.nodes() {
            let dim_info = match &node.output_shape {
                crate::ir::Shape::Scalar => DimensionInfo::Scalar,
                crate::ir::Shape::Vector(d) => {
                    let size = if let crate::ir::Dim::Concrete(n) = d {
                        Some(*n)
                    } else {
                        None
                    };
                    DimensionInfo::Vector(size)
                }
                crate::ir::Shape::Matrix(m, n) => {
                    let nrows = if let crate::ir::Dim::Concrete(r) = m {
                        Some(*r)
                    } else {
                        None
                    };
                    let ncols = if let crate::ir::Dim::Concrete(c) = n {
                        Some(*c)
                    } else {
                        None
                    };
                    DimensionInfo::Matrix(nrows, ncols)
                }
            };
            dimensions.push((*node_id, dim_info));
        }

        // Sort by node ID for consistent hashing
        dimensions.sort_by_key(|(id, _)| *id);

        Self {
            ir_hash,
            dimensions,
        }
    }

    fn hash_graph(graph: &IRGraph) -> [u8; 32] {
        let mut hasher = Sha256::new();

        // Hash nodes in topological order for consistency
        if let Ok(topo_order) = graph.topological_order() {
            for node_id in topo_order {
                if let Some(node) = graph.get_node(node_id) {
                    // Hash node ID
                    hasher.update(&node.id.to_le_bytes());

                    // Hash operation type (discriminant)
                    let op_disc = std::mem::discriminant(&node.op);
                    hasher.update(&format!("{:?}", op_disc).as_bytes());

                    // Hash operation-specific parameters (CRITICAL for correctness)
                    // Different semirings, binary ops, etc. must produce different hashes
                    match &node.op {
                        crate::ir::Operation::MatVec { semiring } |
                        crate::ir::Operation::VecMat { semiring } |
                        crate::ir::Operation::MatMul { semiring, .. } => {
                            // Hash semiring operations
                            hasher.update(&format!("{:?}", semiring.add_op.binary_op).as_bytes());
                            hasher.update(&format!("{:?}", semiring.mul_op).as_bytes());
                            hasher.update(&format!("{:?}", semiring.add_op.identity).as_bytes());
                        }
                        crate::ir::Operation::EWiseAdd { binary_op } |
                        crate::ir::Operation::EWiseMult { binary_op } => {
                            hasher.update(&format!("{:?}", binary_op).as_bytes());
                        }
                        crate::ir::Operation::Apply { unary_op } => {
                            hasher.update(&format!("{:?}", unary_op).as_bytes());
                        }
                        crate::ir::Operation::ApplyBinaryLeft { binary_op, scalar } |
                        crate::ir::Operation::ApplyBinaryRight { binary_op, scalar } => {
                            hasher.update(&format!("{:?}", binary_op).as_bytes());
                            hasher.update(&format!("{:?}", scalar).as_bytes());
                        }
                        crate::ir::Operation::Select { predicate } => {
                            hasher.update(&format!("{:?}", predicate).as_bytes());
                        }
                        crate::ir::Operation::ReduceMatrix { monoid, axis } => {
                            hasher.update(&format!("{:?}", monoid).as_bytes());
                            hasher.update(&format!("{:?}", axis).as_bytes());
                        }
                        crate::ir::Operation::ReduceVector { monoid } => {
                            hasher.update(&format!("{:?}", monoid).as_bytes());
                        }
                        crate::ir::Operation::ConvertFormat { from, to } => {
                            hasher.update(&format!("{:?}{:?}", from, to).as_bytes());
                        }
                        crate::ir::Operation::Input { format, .. } => {
                            hasher.update(&format!("{:?}", format).as_bytes());
                        }
                        // Operations without parameters
                        crate::ir::Operation::Output |
                        crate::ir::Operation::Transpose |
                        crate::ir::Operation::Extract {} |
                        crate::ir::Operation::Assign {} => {}
                    }

                    // Hash input dependencies
                    for input in &node.inputs {
                        hasher.update(&input.to_le_bytes());
                    }

                    // Hash output type
                    hasher.update(&format!("{:?}", node.output_type).as_bytes());
                }
            }
        }

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

/// Compiled kernel with metadata
#[derive(Clone)]
pub struct CachedKernel {
    /// The compiled function
    pub function: Arc<CompiledFunction>,
    /// Number of times this kernel has been used
    pub use_count: usize,
    /// Approximate size in bytes (for LRU eviction)
    pub size_bytes: usize,
}

/// Kernel cache with LRU eviction
pub struct KernelCache {
    /// Map from cache key to compiled kernel
    cache: Mutex<HashMap<CacheKey, CachedKernel>>,
    /// Maximum cache size in bytes
    max_size_bytes: usize,
    /// Current cache size in bytes
    current_size: Mutex<usize>,
}

impl KernelCache {
    /// Create a new kernel cache
    ///
    /// # Arguments
    /// * `max_size_mb` - Maximum cache size in megabytes
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            max_size_bytes: max_size_mb * 1024 * 1024,
            current_size: Mutex::new(0),
        }
    }

    /// Try to get a compiled kernel from the cache
    pub fn get(&self, key: &CacheKey) -> Option<Arc<CompiledFunction>> {
        let mut cache = self.cache.lock().unwrap();

        if let Some(kernel) = cache.get_mut(key) {
            // Update use count for LRU
            kernel.use_count += 1;
            Some(kernel.function.clone())
        } else {
            None
        }
    }

    /// Insert a compiled kernel into the cache
    pub fn insert(
        &self,
        key: CacheKey,
        function: CompiledFunction,
        size_bytes: usize,
    ) {
        let mut cache = self.cache.lock().unwrap();
        let mut current_size = self.current_size.lock().unwrap();

        // Evict if necessary
        while *current_size + size_bytes > self.max_size_bytes && !cache.is_empty() {
            self.evict_lru(&mut cache, &mut current_size);
        }

        // Insert new entry
        let kernel = CachedKernel {
            function: Arc::new(function),
            use_count: 1,
            size_bytes,
        };

        cache.insert(key, kernel);
        *current_size += size_bytes;
    }

    fn evict_lru(
        &self,
        cache: &mut HashMap<CacheKey, CachedKernel>,
        current_size: &mut usize,
    ) {
        // Find entry with lowest use count
        if let Some((key_to_remove, size_to_remove)) = cache
            .iter()
            .min_by_key(|(_, kernel)| kernel.use_count)
            .map(|(k, kernel)| (k.clone(), kernel.size_bytes))
        {
            cache.remove(&key_to_remove);
            *current_size = current_size.saturating_sub(size_to_remove);
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        let mut current_size = self.current_size.lock().unwrap();
        cache.clear();
        *current_size = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.lock().unwrap();
        let current_size = *self.current_size.lock().unwrap();

        CacheStats {
            num_entries: cache.len(),
            size_bytes: current_size,
            max_size_bytes: self.max_size_bytes,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub num_entries: usize,
    pub size_bytes: usize,
    pub max_size_bytes: usize,
}

impl Default for KernelCache {
    fn default() -> Self {
        // Default to 100 MB cache
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{GraphBuilder, Shape, ScalarType, semirings};

    #[test]
    fn test_cache_key_generation() {
        let mut builder = GraphBuilder::new();

        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder
            .input_matrix("B", ScalarType::Float64, Shape::matrix(20, 30))
            .unwrap();

        let semiring = semirings::plus_times(ScalarType::Float64);
        builder.matmul(a, b, semiring).unwrap();

        let graph = builder.build();
        let key = CacheKey::from_graph(&graph);

        // Hash should be deterministic
        let key2 = CacheKey::from_graph(&graph);
        assert_eq!(key, key2);
    }

    #[test]
    fn test_different_graphs_different_keys() {
        let mut builder1 = GraphBuilder::new();
        let a = builder1
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let b = builder1
            .input_matrix("B", ScalarType::Float64, Shape::matrix(20, 30))
            .unwrap();
        let semiring = semirings::plus_times(ScalarType::Float64);
        builder1.matmul(a, b, semiring.clone()).unwrap();
        let graph1 = builder1.build();

        let mut builder2 = GraphBuilder::new();
        let c = builder2
            .input_matrix("C", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        let d = builder2
            .input_matrix("D", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        builder2.ewise_add(c, d, crate::ir::BinaryOpKind::Add).unwrap();
        let graph2 = builder2.build();

        let key1 = CacheKey::from_graph(&graph1);
        let key2 = CacheKey::from_graph(&graph2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_operations() {
        let cache = KernelCache::new(1); // 1 MB cache

        let mut builder = GraphBuilder::new();
        let a = builder
            .input_matrix("A", ScalarType::Float64, Shape::matrix(10, 20))
            .unwrap();
        builder.transpose(a).unwrap();
        let graph = builder.build();

        let key = CacheKey::from_graph(&graph);

        // Cache miss initially
        assert!(cache.get(&key).is_none());

        // Insert into cache
        let function = CompiledFunction::new_stub();
        cache.insert(key.clone(), function, 1000);

        // Cache hit
        assert!(cache.get(&key).is_some());

        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.num_entries, 1);
        assert_eq!(stats.size_bytes, 1000);
    }
}
