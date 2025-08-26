// src/file_store.rs
//
// FileSystemObjectStore implementation for POSIX file I/O
// This provides the same ObjectStore interface for local filesystem operations

use anyhow::{bail, Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use crc32fast::Hasher;

use crate::object_store::{ObjectStore, ObjectMetadata, ObjectWriter};

/// FileSystem adapter that implements ObjectStore for local POSIX file operations.
/// 
/// URI Mapping:
/// - `file:///absolute/path/to/file` -> `/absolute/path/to/file`
/// - `file://./relative/path/to/file` -> `./relative/path/to/file`
/// - `file://../relative/path/to/file` -> `../relative/path/to/file`
///
/// Container Operations:
/// - create_container: creates directories
/// - delete_container: removes empty directories
pub struct FileSystemObjectStore;

/// Streaming writer for filesystem operations
pub struct FileSystemWriter {
    file: Option<fs::File>,
    path: PathBuf,
    bytes_written: u64,
    finalized: bool,
    hasher: Hasher,
}

impl FileSystemWriter {
    async fn new(path: PathBuf) -> Result<Self> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        // Create and open the file for writing
        let file = fs::File::create(&path).await?;
        
        Ok(Self {
            file: Some(file),
            path,
            bytes_written: 0,
            finalized: false,
            hasher: Hasher::new(),
        })
    }
}

#[async_trait]
impl ObjectWriter for FileSystemWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        
        if let Some(ref mut file) = self.file {
            file.write_all(chunk).await?;
            self.hasher.update(chunk);
            self.bytes_written += chunk.len() as u64;
            Ok(())
        } else {
            bail!("Writer has been finalized or cancelled");
        }
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        
        if let Some(mut file) = self.file.take() {
            file.flush().await?;
            file.sync_all().await?;
        }
        
        self.finalized = true;
        Ok(())
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }
    
    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        if let Some(_file) = self.file.take() {
            // File will be closed when dropped
        }
        
        // Remove the partial file
        if self.path.exists() {
            let _ = fs::remove_file(&self.path).await; // Ignore errors
        }
        
        Ok(())
    }
}

impl FileSystemObjectStore {
    pub fn new() -> Self { Self }
    
    /// Convert a URI to a filesystem path
    fn uri_to_path(uri: &str) -> Result<PathBuf> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expects file:// URI, got: {}", uri);
        }
        
        // Strip file:// prefix
        let path = &uri[7..];
        Ok(PathBuf::from(path))
    }
    
    /// Convert a filesystem path back to a URI for list operations
    fn path_to_uri(path: &Path) -> String {
        if path.is_absolute() {
            format!("file://{}", path.display())
        } else {
            path.display().to_string()
        }
    }
    
    /// Create ObjectMetadata from filesystem metadata
    async fn metadata_from_path(path: &Path) -> Result<ObjectMetadata> {
        let metadata = fs::metadata(path).await?;
        
        // Create ObjectMetadata compatible with S3ObjectStat
        let last_modified = metadata.modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
            .map(|duration| {
                // Convert to RFC 3339 format like S3 uses
                let secs = duration.as_secs();
                let datetime = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs);
                format!("{:?}", datetime) // Simple formatting for now
            });
            
        let file_hash = format!("file-{}-{}", 
            metadata.len(),
            metadata.modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        
        Ok(ObjectMetadata {
            size: metadata.len(),
            last_modified,
            e_tag: Some(file_hash),
            content_type: None, // Could infer from file extension
            content_language: None,
            content_encoding: None,
            cache_control: None,
            content_disposition: None,
            expires: None,
            storage_class: Some("STANDARD".to_string()), // Default for files
            server_side_encryption: None,
            ssekms_key_id: None,
            sse_customer_algorithm: None,
            version_id: None,
            replication_status: None,
            metadata: HashMap::new(),
        })
    }
    
    /// Recursively collect files in a directory
    async fn collect_files_recursive(dir: &Path, prefix: &str, results: &mut Vec<String>) -> Result<()> {
        let mut entries = fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            let file_name = entry_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
                
            if entry_path.is_dir() {
                let new_prefix = if prefix.is_empty() {
                    file_name.to_string()
                } else {
                    format!("{}/{}", prefix, file_name)
                };
                Box::pin(Self::collect_files_recursive(&entry_path, &new_prefix, results)).await?;
            } else {
                let file_uri = if prefix.is_empty() {
                    Self::path_to_uri(&entry_path)
                } else {
                    Self::path_to_uri(&entry_path)
                };
                results.push(file_uri);
            }
        }
        
        Ok(())
    }
}

#[async_trait]
impl ObjectStore for FileSystemObjectStore {
    async fn get(&self, uri: &str) -> Result<Vec<u8>> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        if !path.exists() {
            bail!("File not found: {}", path.display());
        }
        
        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }
        
        let data = fs::read(&path).await?;
        Ok(data)
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        if !path.exists() {
            bail!("File not found: {}", path.display());
        }
        
        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }
        
        let mut file = fs::File::open(&path).await?;
        file.seek(std::io::SeekFrom::Start(offset)).await?;
        
        let read_length = length.unwrap_or(u64::MAX);
        let mut buffer = Vec::new();
        
        if read_length == u64::MAX {
            // Read to end of file
            file.read_to_end(&mut buffer).await?;
        } else {
            // Read specific length
            buffer.resize(read_length as usize, 0);
            let bytes_read = file.read(&mut buffer).await?;
            buffer.truncate(bytes_read);
        }
        
        Ok(buffer)
    }

    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        fs::write(&path, data).await?;
        Ok(())
    }

    async fn put_multipart(&self, uri: &str, data: &[u8], _part_size: Option<usize>) -> Result<()> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        // For filesystem, multipart is the same as regular put
        // In a more sophisticated implementation, we could write in chunks
        self.put(uri, data).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        if !uri_prefix.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let base_path = Self::uri_to_path(uri_prefix)?;
        let mut results = Vec::new();
        
        if !base_path.exists() {
            return Ok(results); // Empty list for non-existent paths
        }
        
        if base_path.is_file() {
            // If the prefix points to a file, return just that file
            results.push(Self::path_to_uri(&base_path));
            return Ok(results);
        }
        
        if base_path.is_dir() {
            if recursive {
                Self::collect_files_recursive(&base_path, "", &mut results).await?;
            } else {
                // Non-recursive: only direct children
                let mut entries = fs::read_dir(&base_path).await?;
                while let Some(entry) = entries.next_entry().await? {
                    let entry_path = entry.path();
                    if entry_path.is_file() {
                        results.push(Self::path_to_uri(&entry_path));
                    }
                }
            }
        }
        
        Ok(results)
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        if !path.exists() {
            bail!("File not found: {}", path.display());
        }
        
        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }
        
        Self::metadata_from_path(&path).await
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        if !path.exists() {
            // Already deleted, consider it success
            return Ok(());
        }
        
        if path.is_file() {
            fs::remove_file(&path).await?;
        } else if path.is_dir() {
            fs::remove_dir_all(&path).await?;
        }
        
        Ok(())
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        if !uri_prefix.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let base_path = Self::uri_to_path(uri_prefix)?;
        
        if !base_path.exists() {
            return Ok(()); // Nothing to delete
        }
        
        if base_path.is_file() {
            fs::remove_file(&base_path).await?;
        } else if base_path.is_dir() {
            // Collect all files under the prefix and delete them
            let files = self.list(uri_prefix, true).await?;
            for file_uri in files {
                self.delete(&file_uri).await?;
            }
            
            // Try to remove the directory if it's empty
            if let Err(_) = fs::remove_dir(&base_path).await {
                // Directory might not be empty due to subdirectories
                // For now, we'll leave non-empty directories
            }
        }
        
        Ok(())
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        let path = PathBuf::from(name);
        fs::create_dir_all(&path).await?;
        Ok(())
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        let path = PathBuf::from(name);
        
        if !path.exists() {
            return Ok(()); // Already deleted
        }
        
        if path.is_dir() {
            fs::remove_dir(&path).await?; // Only removes empty directories
        } else {
            bail!("Path is not a directory: {}", path.display());
        }
        
        Ok(())
    }

    async fn rename(&self, src_uri: &str, dst_uri: &str) -> Result<()> {
        if !src_uri.starts_with("file://") { 
            bail!("FileSystemObjectStore expected file:// URI for source"); 
        }
        if !dst_uri.starts_with("file://") { 
            bail!("FileSystemObjectStore expected file:// URI for destination"); 
        }
        
        let src_path = Self::uri_to_path(src_uri)?;
        let dst_path = Self::uri_to_path(dst_uri)?;
        
        if !src_path.exists() {
            bail!("Source file not found: {}", src_path.display());
        }
        
        // Create parent directories for destination if they don't exist
        if let Some(parent) = dst_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        // Use tokio::fs::rename for atomic filesystem rename operation
        fs::rename(&src_path, &dst_path).await?;
        Ok(())
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        if !uri.starts_with("file://") { 
            bail!("FileSystemObjectStore expected file:// URI"); 
        }
        
        let path = Self::uri_to_path(uri)?;
        let writer = FileSystemWriter::new(path).await?;
        Ok(Box::new(writer))
    }
}
