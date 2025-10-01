// src/page_cache.rs
//
// Copyright, 2025. Signal65 / Futurum Group.
//
//! Page cache control via posix_fadvise() on Linux/Unix systems
//!
//! Provides hints to the kernel about file access patterns to optimize
//! page cache behavior for different workloads.

use crate::data_loader::options::PageCacheMode;
use std::os::unix::io::AsRawFd;

/// File size threshold for automatic mode selection
/// Files larger than this will use Sequential, smaller files use Random
const AUTO_SEQUENTIAL_THRESHOLD: u64 = 64 * 1024 * 1024; // 64 MB

/// Apply posix_fadvise hint to a file descriptor
///
/// # Arguments
/// * `fd` - File descriptor (must implement AsRawFd)
/// * `mode` - Page cache mode to apply
/// * `file_size` - Total file size in bytes (used for Auto mode)
///
/// # Platform Support
/// - Linux/Unix: Uses posix_fadvise() system call
/// - Windows: No-op (returns Ok)
/// - Other platforms: No-op (returns Ok)
pub fn apply_page_cache_hint<F: AsRawFd>(fd: &F, mode: PageCacheMode, file_size: u64) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use libc::{posix_fadvise, POSIX_FADV_NORMAL, POSIX_FADV_SEQUENTIAL, POSIX_FADV_RANDOM, POSIX_FADV_DONTNEED};
        
        let raw_fd = fd.as_raw_fd();
        
        let advice = match mode {
            PageCacheMode::Normal => POSIX_FADV_NORMAL,
            PageCacheMode::Sequential => POSIX_FADV_SEQUENTIAL,
            PageCacheMode::Random => POSIX_FADV_RANDOM,
            PageCacheMode::DontNeed => POSIX_FADV_DONTNEED,
            PageCacheMode::Auto => {
                // Automatically choose based on file size
                if file_size >= AUTO_SEQUENTIAL_THRESHOLD {
                    POSIX_FADV_SEQUENTIAL
                } else {
                    POSIX_FADV_RANDOM
                }
            }
        };
        
        // Apply the hint to the entire file (offset=0, len=0 means whole file)
        let result = unsafe { posix_fadvise(raw_fd, 0, 0, advice) };
        
        if result != 0 {
            return Err(std::io::Error::from_raw_os_error(result));
        }
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        // On non-Linux platforms, this is a no-op
        // We could add macOS/BSD support with fcntl(F_RDAHEAD) if needed
        let _ = (fd, mode, file_size); // Silence unused warnings
    }
    
    Ok(())
}

/// Drop page cache for a specific file region (useful after reading data once)
///
/// This is particularly useful for streaming workloads where data won't be re-read.
#[cfg(target_os = "linux")]
pub fn drop_cache_region<F: AsRawFd>(fd: &F, offset: i64, length: i64) -> std::io::Result<()> {
    use libc::{posix_fadvise, POSIX_FADV_DONTNEED};
    
    let raw_fd = fd.as_raw_fd();
    let result = unsafe { posix_fadvise(raw_fd, offset, length, POSIX_FADV_DONTNEED) };
    
    if result != 0 {
        return Err(std::io::Error::from_raw_os_error(result));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_apply_page_cache_hint_normal() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test data").unwrap();
        
        let result = apply_page_cache_hint(file.as_file(), PageCacheMode::Normal, 9);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_page_cache_hint_sequential() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test data").unwrap();
        
        let result = apply_page_cache_hint(file.as_file(), PageCacheMode::Sequential, 9);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_page_cache_hint_random() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test data").unwrap();
        
        let result = apply_page_cache_hint(file.as_file(), PageCacheMode::Random, 9);
        assert!(result.is_ok());
    }

    #[test]
    fn test_auto_mode_small_file() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"small file").unwrap();
        
        // Small file should get RANDOM hint
        let result = apply_page_cache_hint(file.as_file(), PageCacheMode::Auto, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_auto_mode_large_file() {
        let file = NamedTempFile::new().unwrap();
        
        // Large file should get SEQUENTIAL hint
        let result = apply_page_cache_hint(file.as_file(), PageCacheMode::Auto, 100 * 1024 * 1024);
        assert!(result.is_ok());
    }
}
