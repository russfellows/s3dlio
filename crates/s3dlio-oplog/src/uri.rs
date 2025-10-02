//! URI translation utilities for backend retargeting
//!
//! Provides simple 1:1 URI translation to replay operations from one backend to another.

use anyhow::Result;

/// Translate URI from original endpoint to target backend
///
/// Simple 1:1 remapping that preserves relative path structure:
/// - Original: `file:///tmp/data/obj_001.dat` (endpoint: `file:///tmp/`)
/// - Target: `s3://bucket/test/`
/// - Result: `s3://bucket/test/data/obj_001.dat`
///
/// # Arguments
/// * `file` - Full object path from original operation
/// * `endpoint` - Original endpoint/prefix to strip
/// * `target` - New target URI prefix
///
/// # Examples
/// ```
/// use s3dlio_oplog::uri::translate_uri;
///
/// let result = translate_uri(
///     "/bucket/data/file.bin",
///     "/bucket/",
///     "s3://newbucket"
/// ).unwrap();
/// assert_eq!(result, "s3://newbucket/data/file.bin");
/// ```
pub fn translate_uri(file: &str, endpoint: &str, target: &str) -> Result<String> {
    // Remove endpoint prefix from file path
    let relative = file.strip_prefix(endpoint).unwrap_or(file);
    let clean = relative.trim_start_matches('/');

    // Construct new URI with target
    let target_clean = target.trim_end_matches('/');
    Ok(format!("{}/{}", target_clean, clean))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_uri_basic() {
        let result = translate_uri("/bucket/data/file.bin", "/bucket/", "s3://newbucket").unwrap();
        assert_eq!(result, "s3://newbucket/data/file.bin");
    }

    #[test]
    fn test_translate_uri_with_trailing_slashes() {
        let result = translate_uri("/bucket/data/file.bin", "/bucket/", "s3://newbucket/").unwrap();
        assert_eq!(result, "s3://newbucket/data/file.bin");
    }

    #[test]
    fn test_translate_uri_no_prefix_match() {
        // When file doesn't start with endpoint, use file as-is
        let result = translate_uri("/other/path/file.bin", "/bucket/", "s3://newbucket").unwrap();
        assert_eq!(result, "s3://newbucket/other/path/file.bin");
    }

    #[test]
    fn test_translate_uri_different_backends() {
        // File to S3
        let result = translate_uri("file:///tmp/data/obj.dat", "file:///tmp/", "s3://bucket/prefix").unwrap();
        assert_eq!(result, "s3://bucket/prefix/data/obj.dat");

        // S3 to Azure
        let result = translate_uri("s3://source/data/obj.dat", "s3://source/", "az://account/container").unwrap();
        assert_eq!(result, "az://account/container/data/obj.dat");

        // Azure to Direct I/O
        let result = translate_uri("az://account/container/data/obj.dat", "az://account/container/", "direct:///nvme/cache").unwrap();
        assert_eq!(result, "direct:///nvme/cache/data/obj.dat");
    }
}
