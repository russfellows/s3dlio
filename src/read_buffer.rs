// src/read_buffer.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! The destination buffer for a streamed GET (GCS HTTP, Azure): sized once from the length
//! the server announced (GCS: the stored object size in the response; Azure: the blob length from
//! the response's `Content-Range`/`Content-Length`, or the size hint or requested range), so
//! the buffer is never reallocated and re-copied as the data arrives.
//!
//! Issue #252: the GCS HTTP path started a 1 MiB `BytesMut` and let it double, so an object over
//! 1 MiB was copied again at every growth step (about one extra full copy of the object, plus
//! allocator churn); the Azure path used the SDK's `collect()`, which also grows chunk by chunk.
//! This is **one copy inside s3dlio after the SDK delivers the data**: it does not remove
//! copies the SDK itself makes (the Azure SDK's partitioned downloads, used for blobs above the
//! partition size, buffer each partition and hand it up). Growth is still allowed when the server
//! sends more than it announced (it must not lose data), it is just no longer the normal case.

use bytes::Bytes;

/// An in-memory accumulator for one object read.
pub(crate) struct ReadBuffer {
    data: Vec<u8>,
    reallocations: usize,
}

impl ReadBuffer {
    /// A buffer with room for `expected_len` bytes, the length the server announced
    /// (`Content-Length`, the object size, or the requested range). An announced length that
    /// cannot be allocated is not an error: the buffer starts empty and grows as data arrives.
    pub(crate) fn with_expected_len(expected_len: u64) -> Self {
        let mut data = Vec::new();
        if let Ok(len) = usize::try_from(expected_len) {
            // Best effort: on failure (an absurd announced length) fall back to growth.
            let _ = data.try_reserve_exact(len);
        }
        Self {
            data,
            reallocations: 0,
        }
    }

    /// Append one received chunk.
    pub(crate) fn push(&mut self, chunk: &[u8]) {
        let before = self.data.capacity();
        self.data.extend_from_slice(chunk);
        if self.data.capacity() != before {
            self.reallocations += 1;
        }
    }

    /// Bytes received so far.
    #[cfg(any(feature = "backend-gcs", test))]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether nothing has been received.
    #[cfg(feature = "backend-gcs")]
    pub(crate) fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The bytes received so far.
    #[cfg(feature = "backend-gcs")]
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// How many times the buffer had to grow after it was created.
    #[cfg(test)]
    pub(crate) fn reallocations(&self) -> usize {
        self.reallocations
    }

    /// The received bytes, without copying them again.
    pub(crate) fn into_bytes(self) -> Bytes {
        Bytes::from(self.data)
    }
}

/// A zero-filled `Vec<u8>` of exactly `len` bytes, or `None` when the allocation fails.
///
/// A destination buffer that the network (or an SDK) will fill in place, sized from a length the
/// SERVER announced: an absurd announced length must be an error, never an allocator abort
/// (`vec![0; n]` aborts). The memory comes from `alloc_zeroed`, so the operating system supplies
/// zero pages lazily and no byte is written before the data arrives.
#[cfg(feature = "backend-azure")]
pub(crate) fn try_zeroed_vec(len: usize) -> Option<Vec<u8>> {
    if len == 0 {
        return Some(Vec::new());
    }
    let layout = std::alloc::Layout::array::<u8>(len).ok()?;
    // SAFETY: `layout` has a non-zero size (len > 0). The pointer is checked for null below.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `ptr` was just allocated by the global allocator with the layout of exactly `len`
    // `u8` (alignment 1), is initialised (zeroed) for all `len` bytes, and is owned by nobody
    // else; capacity equals the allocation size, which is what `Vec<u8>` deallocates with.
    Some(unsafe { Vec::from_raw_parts(ptr, len, len) })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1024 * 1024;

    /// Feed `total` bytes in `chunk`-sized pieces, as an HTTP body arrives.
    fn fill(buffer: &mut ReadBuffer, total: usize, chunk: usize) {
        let piece = vec![0xA5u8; chunk];
        let mut sent = 0;
        while sent < total {
            let n = chunk.min(total - sent);
            buffer.push(&piece[..n]);
            sent += n;
        }
    }

    // Issue #252: an 8 MiB object announced by the server is received into one allocation.
    #[test]
    fn a_known_length_is_received_without_reallocation() {
        let total = 8 * MIB;
        let mut buffer = ReadBuffer::with_expected_len(total as u64);
        let start = buffer.data.as_ptr();
        fill(&mut buffer, total, 64 * 1024);

        assert_eq!(
            buffer.reallocations(),
            0,
            "RED: an 8 MiB read grew its buffer {} time(s) (each growth copies everything \
             received so far)",
            buffer.reallocations()
        );
        assert_eq!(buffer.len(), total);
        assert_eq!(buffer.data.as_ptr(), start);
        let bytes = buffer.into_bytes();
        assert_eq!(bytes.as_ptr(), start, "into_bytes must not copy");
        assert!(bytes.iter().all(|&b| b == 0xA5));
    }

    // More data than announced is kept, not lost or truncated.
    #[test]
    fn a_longer_than_announced_body_is_still_received_whole() {
        let mut buffer = ReadBuffer::with_expected_len(3 * MIB as u64);
        fill(&mut buffer, 5 * MIB, 128 * 1024);
        assert_eq!(buffer.len(), 5 * MIB);
        assert_eq!(buffer.into_bytes().len(), 5 * MIB);
    }

    // A shorter body (the object changed, or a range ran past the end) is returned as received.
    #[test]
    fn a_shorter_than_announced_body_is_returned_as_received() {
        let mut buffer = ReadBuffer::with_expected_len(6 * MIB as u64);
        fill(&mut buffer, MIB + 17, 4096);
        assert_eq!(buffer.reallocations(), 0);
        assert_eq!(buffer.into_bytes().len(), MIB + 17);
    }

    // An announced length that cannot be allocated must not abort the process.
    #[test]
    fn an_impossible_announced_length_falls_back_to_growth() {
        let mut buffer = ReadBuffer::with_expected_len(u64::MAX);
        fill(&mut buffer, 3 * 4096, 4096);
        assert_eq!(buffer.into_bytes().len(), 3 * 4096);
    }
}
