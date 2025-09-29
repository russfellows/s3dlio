// src/data_loader/fs_bytes.rs
//
// FileSystem bytes dataset implementation that provides the same Dataset interface
// for local file operations as S3BytesDataset does for S3 operations.

use crate::data_loader::{Dataset, DatasetError};
use crate::data_loader::options::{LoaderOptions, ReaderMode};
use async_trait::async_trait;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct FileSystemBytesDataset {
    files: Vec<PathBuf>,
    reader_mode: ReaderMode,
    // Note: part_size and max_inflight_parts are stored for future use
    // when we implement streaming/chunked reading
    #[allow(dead_code)]
    part_size: usize,
    #[allow(dead_code)]
    max_inflight_parts: usize,
}

impl FileSystemBytesDataset {
    /// Create dataset from a file:// URI with default options
    pub fn from_uri(uri: &str) -> Result<Self, DatasetError> {
        Self::from_uri_with_opts(uri, &LoaderOptions::default())
    }

    /// Create dataset from a file:// URI with custom options
    pub fn from_uri_with_opts(uri: &str, opts: &LoaderOptions) -> Result<Self, DatasetError> {
        let path = Self::parse_file_uri(uri)?;
        let files = Self::collect_files(&path)?;

        Ok(Self {
            files,
            reader_mode: opts.reader_mode,
            part_size: opts.part_size,
            max_inflight_parts: opts.max_inflight_parts,
        })
    }

    /// Parse file:// URI to local path
    fn parse_file_uri(uri: &str) -> Result<PathBuf, DatasetError> {
        if !uri.starts_with("file://") {
            return Err(DatasetError::from(format!("Expected file:// scheme, got: {}", uri)));
        }
        
        // Remove "file://" prefix (7 characters)
        let path_str = &uri[7..];
        
        // Handle empty path
        if path_str.is_empty() {
            return Err(DatasetError::from("Empty path in file URI"));
        }
        
        let path = PathBuf::from(path_str);
        Ok(path)
    }

    /// Collect all files from the given path
    /// If path is a file, returns it as a single-item vector
    /// If path is a directory, recursively collects all files
    fn collect_files(path: &Path) -> Result<Vec<PathBuf>, DatasetError> {
        if !path.exists() {
            return Err(DatasetError::from(format!("Path does not exist: {}", path.display())));
        }

        if path.is_file() {
            return Ok(vec![path.to_owned()]);
        }

        if path.is_dir() {
            let mut files = Vec::new();
            Self::collect_files_recursive(path, &mut files)?;
            files.sort(); // Ensure consistent ordering
            return Ok(files);
        }

        Err(DatasetError::from(format!("Path is neither file nor directory: {}", path.display())))
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

    /// Get the file path for a given index
    fn get_file_path(&self, index: usize) -> Result<&PathBuf, DatasetError> {
        self.files.get(index)
            .ok_or_else(|| DatasetError::from(format!("Index {} out of bounds (dataset has {} files)", index, self.files.len())))
    }

    /// Read entire file contents
    async fn read_file(&self, path: &Path) -> Result<Vec<u8>, DatasetError> {
        tokio::fs::read(path).await
            .map_err(|e| DatasetError::from(format!("Failed to read file {}: {}", path.display(), e)))
    }


}

#[async_trait]
impl Dataset for FileSystemBytesDataset {
    type Item = Vec<u8>;

    fn len(&self) -> Option<usize> {
        Some(self.files.len())
    }

    async fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        let path = self.get_file_path(index)?;
        
        match self.reader_mode {
            ReaderMode::Sequential => {
                // Read entire file
                self.read_file(path).await
            }
            ReaderMode::Range => {
                // For range mode, read in chunks but return entire file for simplicity
                // This could be optimized to stream chunks if needed
                self.read_file(path).await
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
    async fn test_single_file() {
        // Use temporary directory for test
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        
        // Create test file
        fs::write(&file_path, b"Hello, World!").unwrap();

        let uri = format!("file://{}", file_path.display());
        let dataset = FileSystemBytesDataset::from_uri(&uri).unwrap();

        assert_eq!(dataset.len(), Some(1));
        let data = dataset.get(0).await.unwrap();
        assert_eq!(data, b"Hello, World!");
        
        // TempDir automatically cleans up when dropped
    }

    #[tokio::test]
    async fn test_directory() {
        // Use temporary directory for test
        let temp_dir = TempDir::new().unwrap();
        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");
        
        // Create test files
        fs::write(&file1, b"File 1 content").unwrap();
        fs::write(&file2, b"File 2 content").unwrap();

        let uri = format!("file://{}", temp_dir.path().display());
        let dataset = FileSystemBytesDataset::from_uri(&uri).unwrap();

        assert_eq!(dataset.len(), Some(2));
        
        // Files should be sorted
        let data1 = dataset.get(0).await.unwrap();
        let data2 = dataset.get(1).await.unwrap();
        
        // Note: exact order depends on filesystem but should be consistent
        assert!(data1 == b"File 1 content" || data1 == b"File 2 content");
        assert!(data2 == b"File 1 content" || data2 == b"File 2 content");
        assert_ne!(data1, data2);
        
        // TempDir automatically cleans up when dropped
    }
}