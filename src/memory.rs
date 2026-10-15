// src/memory.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};
use tokio::sync::{mpsc, Semaphore};

// Import global runtime handle so BufferPool pre-allocation uses the same
// dedicated s3dlio runtime as all other async operations (S3, file://, direct://).
use crate::s3_client::global_rt_handle;

/// Aligned memory buffer suitable for O_DIRECT I/O and zero-copy operations
///
/// This buffer is allocated with specific alignment requirements (typically 4096 bytes)
/// to satisfy O_DIRECT constraints when file I/O is needed, while also being optimal
/// for memory-only operations.
///
/// Deliberately NOT `Clone`: a derived clone would copy the pointer and free it twice.
///
/// # Initialisation
/// [`AlignedBuf::new`] does not touch the memory (so a 1 GiB buffer costs nothing until it is
/// used, and every page is first touched by the thread that fills it). Raw access
/// ([`as_ptr`](Self::as_ptr), [`as_mut_ptr`](Self::as_mut_ptr)) therefore sees uninitialised
/// bytes until you write them. The *slice* accessors ([`as_slice`](Self::as_slice),
/// [`as_mut_slice`](Self::as_mut_slice)) never expose uninitialised memory: the first call
/// zero-fills the buffer, once, unless the owner declared it fully written with the unsafe
/// [`assume_init`](Self::assume_init). [`AlignedBuf::new_zeroed`] zero-fills eagerly.
#[derive(Debug)]
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
    /// Set once every byte is initialised (zero-filled or declared written).
    init: OnceLock<()>,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Create a new aligned buffer with the specified size and alignment.
    ///
    /// The memory is NOT initialised. Returns an error (never panics) when `align` is not
    /// a non-zero power of two, the size overflows, or the allocation fails, so callers
    /// can fall back to a smaller buffer or to buffered I/O.
    ///
    /// # Arguments
    /// * `len` - Buffer size in bytes (0 is allowed: an empty buffer that owns no memory)
    /// * `align` - Alignment requirement (typically 4096 for O_DIRECT)
    pub fn new(len: usize, align: usize) -> Result<Self, AlignedAllocError> {
        let err = AlignedAllocError { len, align };
        let layout = Layout::from_size_align(len, align).map_err(|_| err)?;
        if len == 0 {
            // `alloc` with a zero-size layout is undefined behaviour: own nothing.
            let ptr = NonNull::new(align as *mut u8).ok_or(err)?;
            return Ok(Self {
                ptr,
                len,
                layout,
                init: OnceLock::from(()),
            });
        }
        // SAFETY: `layout` has a non-zero size.
        let raw_ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw_ptr).ok_or(err)?;
        Ok(Self {
            ptr,
            len,
            layout,
            init: OnceLock::new(),
        })
    }

    /// Like [`new`](Self::new) but the memory is zero-filled (touching every page).
    pub fn new_zeroed(len: usize, align: usize) -> Result<Self, AlignedAllocError> {
        let err = AlignedAllocError { len, align };
        let layout = Layout::from_size_align(len, align).map_err(|_| err)?;
        if len == 0 {
            return Self::new(0, align);
        }
        // SAFETY: `layout` has a non-zero size.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(err)?;
        Ok(Self {
            ptr,
            len,
            layout,
            init: OnceLock::from(()),
        })
    }

    /// Declare every byte of the buffer initialised (the owner wrote all of it through the raw
    /// pointer), so the slice accessors need not zero-fill it first.
    ///
    /// # Safety
    /// All `len()` bytes must have been written.
    pub unsafe fn assume_init(&self) {
        let _ = self.init.set(());
    }

    fn ensure_init(&self) {
        self.init.get_or_init(|| {
            // SAFETY: `ptr` is a live allocation of `len` bytes owned by `self`.
            unsafe { std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.len) };
        });
    }

    /// Get a mutable slice view of the buffer (zero-fills it first if it was never written).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.ensure_init();
        // SAFETY: initialised above, `len` bytes owned by `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get an immutable slice view of the buffer (zero-fills it first if it was never written).
    pub fn as_slice(&self) -> &[u8] {
        self.ensure_init();
        // SAFETY: initialised above, `len` bytes owned by `self`.
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
        if self.layout.size() != 0 {
            // SAFETY: allocated in `new` with exactly this layout.
            unsafe {
                dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

/// An aligned allocation could not be made (bad alignment, size overflow or out of memory).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignedAllocError {
    /// Requested size in bytes.
    pub len: usize,
    /// Requested alignment in bytes.
    pub align: usize,
}

impl std::fmt::Display for AlignedAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cannot allocate {} bytes aligned to {}",
            self.len, self.align
        )
    }
}

impl std::error::Error for AlignedAllocError {}

/// A [`BufferPool`] could not be created or could not hand out a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferPoolError {
    /// `capacity` was 0: such a pool could never hand out a buffer.
    ZeroCapacity,
    /// The pool was shut down while waiting for a buffer.
    Closed,
    /// A new buffer could not be allocated.
    Alloc(AlignedAllocError),
}

impl std::fmt::Display for BufferPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "buffer pool capacity must be at least 1"),
            Self::Closed => write!(f, "buffer pool is closed"),
            Self::Alloc(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for BufferPoolError {}

impl From<AlignedAllocError> for BufferPoolError {
    fn from(e: AlignedAllocError) -> Self {
        Self::Alloc(e)
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
    ///
    /// # Errors
    /// [`BufferPoolError::ZeroCapacity`] when `capacity` is 0.
    pub fn new(
        capacity: usize,
        buf_len: usize,
        align: usize,
    ) -> Result<Arc<Self>, BufferPoolError> {
        if capacity == 0 {
            return Err(BufferPoolError::ZeroCapacity);
        }
        let (tx, rx) = mpsc::channel(capacity);
        let permits = Arc::new(Semaphore::new(capacity));

        let pool = Arc::new(Self {
            tx,
            rx: tokio::sync::Mutex::new(rx),
            permits,
            buf_len,
            align,
        });

        // Pre-allocate all buffers onto the global s3dlio Tokio runtime.
        //
        // Using global_rt_handle().spawn() (instead of tokio::spawn()) makes
        // file:// and direct:// behave exactly like S3: all async work goes
        // through the same dedicated background runtime regardless of URI scheme.
        // This is safe to call from any thread context — the global runtime is
        // always available once first accessed (lazy init via OnceLock).
        let pool_clone = pool.clone();
        global_rt_handle().spawn(async move {
            for _ in 0..capacity {
                let Ok(buf) = AlignedBuf::new(buf_len, align) else {
                    break; // out of memory: `take()` reports it to the caller that needs a buffer
                };
                if pool_clone.tx.send(buf).await.is_err() {
                    break; // Pool was dropped
                }
            }
        });

        Ok(pool)
    }

    /// Take a buffer from the pool (blocks if none available).
    ///
    /// Errors (instead of aborting) when the pool is closed, or empty with a new buffer that
    /// cannot be allocated.
    pub async fn take(&self) -> Result<AlignedBuf, BufferPoolError> {
        // Acquire permit first to ensure we don't exceed capacity
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| BufferPoolError::Closed)?;

        // Try to get a buffer from the pool
        let mut rx = self.rx.lock().await;
        if let Ok(buf) = rx.try_recv() {
            Ok(buf)
        } else {
            // Pool is temporarily empty, allocate a new one
            Ok(AlignedBuf::new(self.buf_len, self.align)?)
        }
    }

    /// Return a buffer to the pool for reuse
    ///
    /// Uses `try_send` (non-blocking): if the pool channel is already at capacity
    /// (e.g. because all pre-allocated buffers are present), the returned buffer is
    /// simply dropped rather than blocking the caller forever.
    pub fn give(&self, buf: AlignedBuf) {
        // Only return buffers with the correct size and alignment
        if buf.len() == self.buf_len && buf.align() == self.align {
            let _ = self.tx.try_send(buf); // Non-blocking: drop if pool is full
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
            capacity: 64,                  // 64 buffers
            buffer_size: 16 * 1024 * 1024, // 16 MiB per buffer
            alignment: 4096,               // 4 KiB alignment for O_DIRECT
        }
    }
}

impl BufferPoolConfig {
    /// Create a buffer pool from this configuration
    pub fn build(self) -> Result<Arc<BufferPool>, BufferPoolError> {
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
        let mut buf = AlignedBuf::new(8192, 4096).unwrap();
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
        // Timeout guard: this test must complete in under 5 seconds.
        // A hang here means give() is blocking on a full channel (deadlock) —
        // that is a real production bug, not just a test issue.
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let pool = BufferPool::new(2, 4096, 4096).unwrap();

            // Take buffer
            let mut buf1 = pool.take().await.unwrap();
            assert_eq!(buf1.len(), 4096);

            // Modify buffer
            buf1.as_mut_slice()[0] = 123;

            // Return buffer (non-blocking try_send — must not deadlock even if pool is full)
            pool.give(buf1);

            // Take again - should get a buffer (possibly reused)
            let buf2 = pool.take().await.unwrap();
            assert_eq!(buf2.len(), 4096);
        })
        .await;
        assert!(
            result.is_ok(),
            "buffer_pool_basic timed out — give() deadlocked on full channel"
        );
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

#[cfg(test)]
mod aligned_alloc_tests {
    use super::*;

    /// Round 2, item 3: allocation problems are errors, never panics or aborts.
    #[test]
    fn impossible_allocations_are_errors_not_panics() {
        assert!(AlignedBuf::new(usize::MAX, 4096).is_err(), "size overflow");
        assert!(
            AlignedBuf::new(1 << 20, 3).is_err(),
            "alignment not a power of two"
        );
        assert!(AlignedBuf::new(1 << 20, 0).is_err(), "zero alignment");
        let e = AlignedBuf::new(usize::MAX / 2, 4096).unwrap_err();
        assert_eq!(e.align, 4096);
        assert!(e.to_string().contains("4096"));
    }

    #[test]
    fn zero_length_buffer_owns_nothing_and_drops_cleanly() {
        let b = AlignedBuf::new(0, 4096).unwrap();
        assert!(b.is_empty());
        assert!(b.as_slice().is_empty());
    }
}

#[cfg(test)]
mod aligned_init_tests {
    use super::*;

    /// Round 3: slice accessors never expose uninitialised memory.
    #[test]
    fn slices_over_a_fresh_buffer_are_zero() {
        let mut b = AlignedBuf::new(1 << 20, 4096).unwrap();
        // Poison the raw memory first, as a recycled allocation might look.
        unsafe { std::ptr::write_bytes(b.as_mut_ptr(), 0xEE, b.len()) };
        // (assume_init was NOT called, so the first slice access zero-fills.)
        assert!(b.as_slice().iter().all(|&x| x == 0));
        b.as_mut_slice()[7] = 9;
        assert_eq!(b.as_slice()[7], 9, "zero-fill must happen only once");
    }

    #[test]
    fn assume_init_keeps_what_the_owner_wrote() {
        let mut b = AlignedBuf::new(8192, 4096).unwrap();
        unsafe {
            std::ptr::write_bytes(b.as_mut_ptr(), 0x5A, 8192);
            b.assume_init();
        }
        assert!(b.as_slice().iter().all(|&x| x == 0x5A));
    }

    #[test]
    fn new_zeroed_is_zero_and_aligned() {
        let b = AlignedBuf::new_zeroed(1 << 20, 4096).unwrap();
        assert_eq!(b.as_ptr() as usize % 4096, 0);
        assert!(b.as_slice().iter().all(|&x| x == 0));
        assert!(AlignedBuf::new_zeroed(usize::MAX, 4096).is_err());
        assert!(AlignedBuf::new_zeroed(0, 4096).unwrap().is_empty());
    }

    #[tokio::test]
    async fn zero_capacity_pool_is_an_error_not_a_panic() {
        assert_eq!(
            BufferPool::new(0, 4096, 4096).unwrap_err(),
            BufferPoolError::ZeroCapacity
        );
        assert!(BufferPoolConfig::default()
            .with_capacity(0)
            .build()
            .is_err());
    }

    #[tokio::test]
    async fn closed_pool_take_is_an_error_not_a_panic() {
        let pool = BufferPool::new(1, 4096, 4096).unwrap();
        pool.permits.close();
        assert_eq!(pool.take().await.unwrap_err(), BufferPoolError::Closed);
    }
}
