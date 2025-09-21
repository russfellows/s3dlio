// src/memory.rs
//
// High-performance memory management for zero-copy S3 operations
// Optimized for AI/ML workloads with aligned buffers and buffer pools

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

/// Aligned memory buffer suitable for O_DIRECT I/O and zero-copy operations
/// 
/// This buffer is allocated with specific alignment requirements (typically 4096 bytes)
/// to satisfy O_DIRECT constraints when file I/O is needed, while also being optimal
/// for memory-only operations.
#[derive(Debug, Clone)]
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Create a new aligned buffer with the specified size and alignment
    /// 
    /// # Arguments
    /// * `len` - Buffer size in bytes
    /// * `align` - Alignment requirement (typically 4096 for O_DIRECT)
    pub fn new(len: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(len, align)
            .expect("Invalid layout for aligned buffer");
        let ptr = unsafe { 
            let raw_ptr = alloc(layout);
            NonNull::new(raw_ptr).expect("Failed to allocate aligned buffer")
        };
        Self { ptr, len, layout }
    }

    /// Get a mutable slice view of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get an immutable slice view of the buffer
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get the buffer length in bytes
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the buffer alignment
    pub fn align(&self) -> usize {
        self.layout.align()
    }

    /// Get raw pointer (for advanced use cases)
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer (for advanced use cases)
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// High-performance buffer pool for aligned buffers
/// 
/// Maintains a pool of pre-allocated aligned buffers to eliminate allocation
/// overhead during high-throughput S3 operations. Buffers are reused across
/// multiple operations to minimize GC pressure and maintain consistent performance.
#[derive(Debug)]
pub struct BufferPool {
    tx: mpsc::Sender<AlignedBuf>,
    rx: tokio::sync::Mutex<mpsc::Receiver<AlignedBuf>>,
    permits: Arc<Semaphore>,
    buf_len: usize,
    align: usize,
}

impl BufferPool {
    /// Create a new buffer pool with the specified capacity and buffer parameters
    /// 
    /// # Arguments
    /// * `capacity` - Number of buffers to maintain in the pool
    /// * `buf_len` - Size of each buffer in bytes
    /// * `align` - Alignment requirement (use 4096 for O_DIRECT compatibility)
    pub fn new(capacity: usize, buf_len: usize, align: usize) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(capacity);
        let permits = Arc::new(Semaphore::new(capacity));
        
        let pool = Arc::new(Self {
            tx,
            rx: tokio::sync::Mutex::new(rx),
            permits,
            buf_len,
            align,
        });

        // Pre-allocate all buffers
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            for _ in 0..capacity {
                let buf = AlignedBuf::new(buf_len, align);
                if pool_clone.tx.send(buf).await.is_err() {
                    break; // Pool was dropped
                }
            }
        });

        pool
    }

    /// Take a buffer from the pool (blocks if none available)
    pub async fn take(&self) -> AlignedBuf {
        // Acquire permit first to ensure we don't exceed capacity
        let _permit = self.permits.acquire().await.expect("Pool semaphore closed");
        
        // Try to get a buffer from the pool
        let mut rx = self.rx.lock().await;
        if let Some(buf) = rx.try_recv().ok() {
            buf
        } else {
            // Pool is temporarily empty, allocate a new one
            AlignedBuf::new(self.buf_len, self.align)
        }
    }

    /// Return a buffer to the pool for reuse
    pub async fn give(&self, buf: AlignedBuf) {
        // Only return buffers with the correct size and alignment
        if buf.len() == self.buf_len && buf.align() == self.align {
            let _ = self.tx.send(buf).await; // Ignore error if pool is full/closed
        }
        // If buffer doesn't match pool parameters, just drop it
    }

    /// Get pool statistics
    pub fn capacity(&self) -> usize {
        self.permits.available_permits()
    }
}

/// Configuration for buffer pool creation
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Number of buffers to maintain in the pool
    pub capacity: usize,
    /// Size of each buffer in bytes (should match part size for S3 operations)
    pub buffer_size: usize,
    /// Alignment requirement (4096 for O_DIRECT compatibility)
    pub alignment: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            capacity: 64,                    // 64 buffers
            buffer_size: 16 * 1024 * 1024,  // 16 MiB per buffer
            alignment: 4096,                 // 4 KiB alignment for O_DIRECT
        }
    }
}

impl BufferPoolConfig {
    /// Create a buffer pool from this configuration
    pub fn build(self) -> Arc<BufferPool> {
        BufferPool::new(self.capacity, self.buffer_size, self.alignment)
    }

    /// Set the pool capacity (number of buffers)
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Set the buffer size in bytes
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the alignment requirement
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buf_creation() {
        let mut buf = AlignedBuf::new(8192, 4096);
        assert_eq!(buf.len(), 8192);
        assert_eq!(buf.align(), 4096);
        assert_eq!(buf.as_ptr() as usize % 4096, 0); // Check alignment
        
        // Ensure we can write to the buffer
        let slice = buf.as_mut_slice();
        slice[0] = 42;
        assert_eq!(slice[0], 42);
    }

    #[tokio::test]
    async fn buffer_pool_basic() {
        let pool = BufferPool::new(2, 4096, 4096);
        
        // Take buffer
        let mut buf1 = pool.take().await;
        assert_eq!(buf1.len(), 4096);
        
        // Modify buffer
        buf1.as_mut_slice()[0] = 123;
        
        // Return buffer
        pool.give(buf1).await;
        
        // Take again - should get a buffer (possibly reused)
        let buf2 = pool.take().await;
        assert_eq!(buf2.len(), 4096);
    }

    #[test]
    fn buffer_pool_config() {
        let config = BufferPoolConfig::default()
            .with_capacity(32)
            .with_buffer_size(8 * 1024 * 1024)
            .with_alignment(4096);
            
        assert_eq!(config.capacity, 32);
        assert_eq!(config.buffer_size, 8 * 1024 * 1024);
        assert_eq!(config.alignment, 4096);
    }
}