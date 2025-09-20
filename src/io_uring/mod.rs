// src/io_uring/mod.rs
//
// io_uring backend for high-performance Linux I/O

#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub mod backend;

#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub use backend::{
    IoUringFileStore, IoUringConfig
};

#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub fn create_io_uring_file_store(config: IoUringConfig) -> anyhow::Result<IoUringFileStore> {
    IoUringFileStore::new(config)
}

// Fallback for non-Linux or when feature is disabled
#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
pub mod fallback {
    use anyhow::Result;
    use crate::file_store::FileSystemObjectStore;
    
    pub type IoUringFileStore = FileSystemObjectStore;
    
    #[derive(Debug, Clone, Default)]
    pub struct IoUringConfig;
    
    pub fn create_io_uring_file_store(_config: IoUringConfig) -> Result<IoUringFileStore> {
        log::info!("io_uring not available, falling back to standard file I/O");
        Ok(FileSystemObjectStore::new())
    }
}

#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
pub use fallback::*;