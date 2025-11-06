// src/uri_utils.rs
//! URI manipulation utilities for multi-endpoint support
//!
//! This module provides functions for expanding URI templates and parsing
//! multi-endpoint specifications. It enables consistent multi-endpoint
//! configuration across all backends (S3, Azure, GCS, file://, direct://).

use anyhow::{Context, Result, bail};
use regex::Regex;
use std::path::Path;
use std::fs;

/// Expand a URI template containing {start...end} range patterns
///
/// # Supported Patterns
///
/// - Numerical ranges: `s3://bucket-{1...5}/` → `["s3://bucket-1/", ..., "s3://bucket-5/"]`
/// - IP ranges: `s3://10.0.0.{1...10}:9000/bucket/`
/// - File systems: `file:///mnt/storage{1...4}/data/`
/// - Direct I/O: `direct:///dev/nvme{0...3}n1`
/// - Multiple ranges: `s3://host-{1...2}.{10...12}/bucket/` (cartesian product)
///
/// # Examples
///
/// ```
/// use s3dlio::uri_utils::expand_uri_template;
///
/// let uris = expand_uri_template("s3://bucket-{1...3}/").unwrap();
/// assert_eq!(uris, vec!["s3://bucket-1/", "s3://bucket-2/", "s3://bucket-3/"]);
///
/// let ips = expand_uri_template("s3://10.0.0.{1...3}:9000/bucket/").unwrap();
/// assert_eq!(ips[0], "s3://10.0.0.1:9000/bucket/");
/// ```
pub fn expand_uri_template(template: &str) -> Result<Vec<String>> {
    // Check if template contains range patterns
    if !template.contains('{') || !template.contains('}') {
        return Ok(vec![template.to_string()]);
    }
    
    // Regex to match {start...end} patterns
    let range_re = Regex::new(r"\{(\d+)\.\.\.(\d+)\}").unwrap();
    
    // Find all range patterns
    let mut ranges = Vec::new();
    for cap in range_re.captures_iter(template) {
        let start: i32 = cap[1].parse()
            .context("Invalid range start number")?;
        let end: i32 = cap[2].parse()
            .context("Invalid range end number")?;
        
        if start > end {
            bail!("Invalid range: start ({}) > end ({})", start, end);
        }
        
        ranges.push((start, end, cap.get(0).unwrap().as_str().to_string()));
    }
    
    if ranges.is_empty() {
        return Ok(vec![template.to_string()]);
    }
    
    // Expand all ranges (cartesian product for multiple ranges)
    let mut results = vec![template.to_string()];
    
    for (start, end, pattern) in ranges {
        let mut new_results = Vec::new();
        
        for current in results {
            for i in start..=end {
                let expanded = current.replace(&pattern, &i.to_string());
                new_results.push(expanded);
            }
        }
        
        results = new_results;
    }
    
    // Validate all expanded URIs have valid scheme
    let first_scheme = infer_scheme_from_uri(&results[0])?;
    for uri in &results[1..] {
        let scheme = infer_scheme_from_uri(uri)?;
        if scheme != first_scheme {
            bail!("URI expansion resulted in mixed schemes: {} and {}", 
                  first_scheme, scheme);
        }
    }
    
    Ok(results)
}

/// Parse a comma-separated list of URIs
///
/// # Examples
///
/// ```
/// use s3dlio::uri_utils::parse_uri_list;
///
/// let uris = parse_uri_list("s3://host1/bucket/, s3://host2/bucket/").unwrap();
/// assert_eq!(uris.len(), 2);
/// assert_eq!(uris[0], "s3://host1/bucket/");
/// ```
pub fn parse_uri_list(uri_str: &str) -> Result<Vec<String>> {
    let uris: Vec<String> = uri_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    
    if uris.is_empty() {
        bail!("Empty URI list");
    }
    
    // Validate all URIs have the same scheme
    let first_scheme = infer_scheme_from_uri(&uris[0])?;
    for uri in &uris[1..] {
        let scheme = infer_scheme_from_uri(uri)?;
        if scheme != first_scheme {
            bail!("URI list contains mixed schemes: {} and {}", 
                  first_scheme, scheme);
        }
    }
    
    Ok(uris)
}

/// Load URIs from a newline-separated file
///
/// Lines starting with '#' are treated as comments and ignored.
/// Empty lines are skipped.
///
/// # Examples
///
/// Given a file `endpoints.txt`:
/// ```text
/// # S3 endpoints
/// s3://10.0.0.1:9000/bucket/
/// s3://10.0.0.2:9000/bucket/
/// s3://10.0.0.3:9000/bucket/
/// ```
///
/// ```no_run
/// use s3dlio::uri_utils::load_uris_from_file;
/// use std::path::Path;
///
/// let uris = load_uris_from_file(Path::new("endpoints.txt")).unwrap();
/// assert_eq!(uris.len(), 3);
/// ```
pub fn load_uris_from_file(path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read URI file: {}", path.display()))?;
    
    let uris: Vec<String> = content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.to_string())
        .collect();
    
    if uris.is_empty() {
        bail!("No URIs found in file: {}", path.display());
    }
    
    // Validate all URIs have the same scheme
    let first_scheme = infer_scheme_from_uri(&uris[0])?;
    for uri in &uris[1..] {
        let scheme = infer_scheme_from_uri(uri)?;
        if scheme != first_scheme {
            bail!("URI file contains mixed schemes: {} and {}", 
                  first_scheme, scheme);
        }
    }
    
    Ok(uris)
}

/// Infer the scheme from a URI string
pub fn infer_scheme_from_uri(uri: &str) -> Result<String> {
    if let Some(pos) = uri.find("://") {
        Ok(uri[..pos].to_lowercase())
    } else {
        bail!("Invalid URI format (missing scheme): {}", uri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_expand_simple_range() {
        let result = expand_uri_template("s3://bucket-{1...3}/").unwrap();
        assert_eq!(result, vec![
            "s3://bucket-1/",
            "s3://bucket-2/",
            "s3://bucket-3/"
        ]);
    }
    
    #[test]
    fn test_expand_ip_range() {
        let result = expand_uri_template("s3://10.0.0.{1...3}:9000/bucket/").unwrap();
        assert_eq!(result, vec![
            "s3://10.0.0.1:9000/bucket/",
            "s3://10.0.0.2:9000/bucket/",
            "s3://10.0.0.3:9000/bucket/"
        ]);
    }
    
    #[test]
    fn test_expand_file_path() {
        let result = expand_uri_template("file:///mnt/storage{1...3}/data/").unwrap();
        assert_eq!(result, vec![
            "file:///mnt/storage1/data/",
            "file:///mnt/storage2/data/",
            "file:///mnt/storage3/data/"
        ]);
    }
    
    #[test]
    fn test_expand_direct_io() {
        let result = expand_uri_template("direct:///dev/nvme{0...2}n1").unwrap();
        assert_eq!(result, vec![
            "direct:///dev/nvme0n1",
            "direct:///dev/nvme1n1",
            "direct:///dev/nvme2n1"
        ]);
    }
    
    #[test]
    fn test_expand_multiple_ranges() {
        let result = expand_uri_template("s3://host-{1...2}.subnet-{10...11}/bucket/").unwrap();
        assert_eq!(result.len(), 4); // 2 * 2 = 4 combinations
        assert!(result.contains(&"s3://host-1.subnet-10/bucket/".to_string()));
        assert!(result.contains(&"s3://host-1.subnet-11/bucket/".to_string()));
        assert!(result.contains(&"s3://host-2.subnet-10/bucket/".to_string()));
        assert!(result.contains(&"s3://host-2.subnet-11/bucket/".to_string()));
    }
    
    #[test]
    fn test_expand_zero_padding() {
        let result = expand_uri_template("s3://bucket-{01...03}/").unwrap();
        // Note: Current implementation doesn't preserve zero-padding
        // This is acceptable for v1
        assert_eq!(result.len(), 3);
    }
    
    #[test]
    fn test_expand_invalid_range() {
        let result = expand_uri_template("s3://bucket-{5...3}/");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid range"));
    }
    
    #[test]
    fn test_expand_no_range() {
        let result = expand_uri_template("s3://bucket/path/").unwrap();
        assert_eq!(result, vec!["s3://bucket/path/"]);
    }
    
    #[test]
    fn test_parse_comma_separated() {
        let result = parse_uri_list("s3://host1/bucket/, s3://host2/bucket/, s3://host3/bucket/").unwrap();
        assert_eq!(result, vec![
            "s3://host1/bucket/",
            "s3://host2/bucket/",
            "s3://host3/bucket/"
        ]);
    }
    
    #[test]
    fn test_parse_comma_separated_with_extra_spaces() {
        let result = parse_uri_list("  s3://host1/  ,  s3://host2/  ").unwrap();
        assert_eq!(result, vec![
            "s3://host1/",
            "s3://host2/"
        ]);
    }
    
    #[test]
    fn test_parse_mixed_schemes_error() {
        let result = parse_uri_list("s3://bucket1/, file:///path/");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mixed schemes"));
    }
    
    #[test]
    fn test_load_from_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# Test endpoints").unwrap();
        writeln!(file, "s3://10.0.0.1:9000/bucket/").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "s3://10.0.0.2:9000/bucket/").unwrap();
        writeln!(file, "# Another comment").unwrap();
        writeln!(file, "s3://10.0.0.3:9000/bucket/").unwrap();
        file.flush().unwrap();
        
        let result = load_uris_from_file(file.path()).unwrap();
        assert_eq!(result, vec![
            "s3://10.0.0.1:9000/bucket/",
            "s3://10.0.0.2:9000/bucket/",
            "s3://10.0.0.3:9000/bucket/"
        ]);
    }
    
    #[test]
    fn test_load_from_file_empty() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# Only comments").unwrap();
        writeln!(file, "").unwrap();
        file.flush().unwrap();
        
        let result = load_uris_from_file(file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No URIs found"));
    }
    
    #[test]
    fn test_load_from_file_mixed_schemes() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "s3://bucket1/").unwrap();
        writeln!(file, "file:///path/").unwrap();
        file.flush().unwrap();
        
        let result = load_uris_from_file(file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mixed schemes"));
    }
    
    #[test]
    fn test_infer_scheme() {
        assert_eq!(infer_scheme_from_uri("s3://bucket/").unwrap(), "s3");
        assert_eq!(infer_scheme_from_uri("S3://bucket/").unwrap(), "s3");
        assert_eq!(infer_scheme_from_uri("file:///path/").unwrap(), "file");
        assert_eq!(infer_scheme_from_uri("direct:///dev/nvme0n1").unwrap(), "direct");
        assert_eq!(infer_scheme_from_uri("az://container/").unwrap(), "az");
        assert_eq!(infer_scheme_from_uri("gs://bucket/").unwrap(), "gs");
    }
    
    #[test]
    fn test_infer_scheme_invalid() {
        let result = infer_scheme_from_uri("invalid-uri");
        assert!(result.is_err());
    }
}
