// src/data_loader/directio_bytes.rs
//
// Direct I/O bytes dataset implementation that provides the same Dataset interface
// for local file operations using O_DIRECT as S3BytesDataset does for S3 operations.

use crate::data_loader::{Dataset, DatasetError};
use crate::data_loader::options::{LoaderOptions, ReaderMode};
use crate::object_store::store_for_uri;
use async_trait::async_trait;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct DirectIOBytesDataset {
    files: Vec<String>, // Store as URIs with direct:// scheme
    reader_mode: ReaderMode,
    #[allow(dead_code)]
    part_size: usize,
    #[allow(dead_code)]
    max_inflight_parts: usize,
}

impl DirectIOBytesDataset {
    /// Create dataset from a direct:// URI with default options
    pub fn from_uri(uri: &str) -> Result<Self, DatasetError> {
        Self::from_uri_with_opts(uri, &LoaderOptions::default())
    }

    /// Create dataset from a direct:// URI with custom options
    pub fn from_uri_with_opts(uri: &str, opts: &LoaderOptions) -> Result<Self, DatasetError> {
        let path = Self::parse_direct_uri(uri)?;
        let files = Self::collect_files(&path)?;

        Ok(Self {
            files,
            reader_mode: opts.reader_mode,
            part_size: opts.part_size,
            max_inflight_parts: opts.max_inflight_parts,
        })
    }

    /// Parse direct:// URI to local path
    fn parse_direct_uri(uri: &str) -> Result<PathBuf, DatasetError> {
        if !uri.starts_with("direct://") {
            return Err(DatasetError::from(format!("Expected direct:// scheme, got: {}", uri)));
        }
        
        // Remove "direct://" prefix (9 characters)
        let path_str = &uri[9..];
        
        if path_str.is_empty() {
            return Err(DatasetError::from("Empty path in direct:// URI"));
        }
        
        Ok(PathBuf::from(path_str))
    }

    /// Collect all files in a directory or return single file as a vector
    fn collect_files(path: &Path) -> Result<Vec<String>, DatasetError> {
        if path.is_file() {
            // Single file - convert back to direct:// URI
            let uri = format!("direct://{}", path.to_string_lossy());
            Ok(vec![uri])
        } else if path.is_dir() {
            // Directory - collect all files recursively and convert to direct:// URIs
            let mut files = Vec::new();
            Self::collect_files_recursive(path, &mut files)?;
            files.sort(); // Ensure deterministic ordering
            Ok(files.into_iter()
                .map(|p| format!("direct://{}", p.to_string_lossy()))
                .collect())
        } else {
            Err(DatasetError::from(format!("Path does not exist or is not accessible: {}", path.display())))
        }
    }

    /// Recursively collect files from a directory
    fn collect_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), DatasetError> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| DatasetError::from(format!("Failed to read directory {}: {}", dir.display(), e)))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| DatasetError::from(format!("Failed to read directory entry: {}", e)))?;
            let path = entry.path();

            if path.is_file() {
                files.push(path);
            } else if path.is_dir() {
                Self::collect_files_recursive(&path, files)?;
            }
        }

        Ok(())
    }

    /// Get the file URI for a given index
    fn get_file_uri(&self, index: usize) -> Result<&String, DatasetError> {
        self.files.get(index)
            .ok_or_else(|| DatasetError::from(format!("Index {} out of bounds (dataset has {} files)", index, self.files.len())))
    }

    /// Read entire file contents using Direct I/O object store
    async fn read_file(&self, uri: &str) -> Result<Vec<u8>, DatasetError> {
        let store = store_for_uri(uri)
            .map_err(|e| DatasetError::from(format!("Failed to create Direct I/O store: {}", e)))?;
        
        // Convert Bytes to Vec<u8> for Dataset API compatibility
        store.get(uri).await
            .map(|bytes| bytes.to_vec())
            .map_err(|e| DatasetError::from(format!("Failed to read file with Direct I/O {}: {}", uri, e)))
    }
}

#[async_trait]
impl Dataset for DirectIOBytesDataset {
    type Item = Vec<u8>;

    fn len(&self) -> Option<usize> {
        Some(self.files.len())
    }

    async fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        let uri = self.get_file_uri(index)?;
        
        match self.reader_mode {
            ReaderMode::Sequential => {
                // Read entire file using Direct I/O
                self.read_file(uri).await
            }
            ReaderMode::Range => {
                // For range mode, read in chunks but return entire file for simplicity
                // This could be optimized to use range reads if needed
                self.read_file(uri).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_single_file_direct_io() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, b"test content").unwrap();

        let uri = format!("direct://{}", file_path.to_string_lossy());
        let dataset = DirectIOBytesDataset::from_uri(&uri).unwrap();

        assert_eq!(dataset.len(), Some(1));
        
        let content = dataset.get(0).await.unwrap();
        assert_eq!(content, b"test content");
    }

    #[tokio::test]
    async fn test_directory_direct_io() {
        let temp_dir = TempDir::new().unwrap();
        
        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");
        fs::write(&file1, b"content 1").unwrap();
        fs::write(&file2, b"content 2").unwrap();

        let uri = format!("direct://{}", temp_dir.path().to_string_lossy());
        let dataset = DirectIOBytesDataset::from_uri(&uri).unwrap();

        assert_eq!(dataset.len(), Some(2));
        
        // Read both files (order may vary due to filesystem)
        let content1 = dataset.get(0).await.unwrap();
        let content2 = dataset.get(1).await.unwrap();
        
        let contents = vec![content1, content2];
        assert!(contents.contains(&b"content 1".to_vec()));
        assert!(contents.contains(&b"content 2".to_vec()));
    }
}