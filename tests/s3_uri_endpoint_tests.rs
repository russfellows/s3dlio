// Integration tests for S3 URI parsing with endpoint support
// Tests the public API as external users would interact with it

use s3dlio::{parse_s3_uri_full, parse_s3_uri, S3UriComponents};

#[test]
fn test_s3_uri_components_type() {
    // Test that we can explicitly type the return value
    let components: S3UriComponents = parse_s3_uri_full("s3://192.168.1.1:9000/bucket/key").unwrap();
    
    assert_eq!(components.endpoint, Some("192.168.1.1:9000".to_string()));
    assert_eq!(components.bucket, "bucket");
    assert_eq!(components.key, "key");
    
    // Test public fields are accessible
    let endpoint_copy = components.endpoint.clone();
    let bucket_copy = components.bucket.clone();
    let key_copy = components.key.clone();
    
    assert_eq!(endpoint_copy, Some("192.168.1.1:9000".to_string()));
    assert_eq!(bucket_copy, "bucket");
    assert_eq!(key_copy, "key");
}

#[test]
fn test_warp_style_multi_endpoint_uris() {
    // Warp-style URIs with different IP addresses and ports
    let uris = vec![
        "s3://192.168.100.1:9001/bucket/test.dat",
        "s3://192.168.100.1:9002/bucket/test.dat",
        "s3://192.168.100.2:9001/bucket/test.dat",
        "s3://192.168.100.2:9002/bucket/test.dat",
    ];
    
    let expected_endpoints = vec![
        "192.168.100.1:9001",
        "192.168.100.1:9002",
        "192.168.100.2:9001",
        "192.168.100.2:9002",
    ];
    
    for (uri, expected_endpoint) in uris.iter().zip(expected_endpoints.iter()) {
        let result = parse_s3_uri_full(uri).unwrap();
        assert_eq!(result.endpoint.as_ref().unwrap(), expected_endpoint);
        assert_eq!(result.bucket, "bucket");
        assert_eq!(result.key, "test.dat");
    }
}

#[test]
fn test_minio_cluster_endpoints() {
    // Multiple MinIO servers in a cluster
    let uris = vec![
        "s3://minio1.example.com:9000/mybucket/data/file1.bin",
        "s3://minio2.example.com:9000/mybucket/data/file2.bin",
        "s3://minio3.example.com:9000/mybucket/data/file3.bin",
    ];
    
    for (idx, uri) in uris.iter().enumerate() {
        let result = parse_s3_uri_full(uri).unwrap();
        assert!(result.endpoint.is_some());
        assert_eq!(result.bucket, "mybucket");
        assert_eq!(result.key, format!("data/file{}.bin", idx + 1));
    }
}

#[test]
fn test_mixed_standard_and_custom_endpoints() {
    // Standard AWS URI (no endpoint)
    let aws_uri = "s3://aws-bucket/path/to/file.txt";
    let aws_result = parse_s3_uri_full(aws_uri).unwrap();
    assert_eq!(aws_result.endpoint, None);
    assert_eq!(aws_result.bucket, "aws-bucket");
    
    // Custom endpoint URI
    let custom_uri = "s3://192.168.1.100:9000/custom-bucket/path/to/file.txt";
    let custom_result = parse_s3_uri_full(custom_uri).unwrap();
    assert_eq!(custom_result.endpoint, Some("192.168.1.100:9000".to_string()));
    assert_eq!(custom_result.bucket, "custom-bucket");
}

#[test]
fn test_backwards_compatibility() {
    // Old parse_s3_uri function should still work for standard URIs
    let (bucket, key) = parse_s3_uri("s3://mybucket/mykey").unwrap();
    assert_eq!(bucket, "mybucket");
    assert_eq!(key, "mykey");
    
    // Also works with endpoint URIs (just ignores endpoint)
    let (bucket2, key2) = parse_s3_uri("s3://192.168.1.1:9000/bucket2/key2").unwrap();
    assert_eq!(bucket2, "bucket2");
    assert_eq!(key2, "key2");
}

#[test]
fn test_localhost_development() {
    // Common localhost development scenarios
    let uris = vec![
        "s3://localhost:9000/testbucket/file.txt",
        "s3://127.0.0.1:9000/testbucket/file.txt",
        "s3://localhost:9001/testbucket/file.txt",
    ];
    
    for uri in uris {
        let result = parse_s3_uri_full(uri).unwrap();
        assert!(result.endpoint.is_some());
        assert_eq!(result.bucket, "testbucket");
        assert_eq!(result.key, "file.txt");
    }
}

#[test]
fn test_ipv4_endpoint_detection() {
    // Various IPv4 formats
    let test_cases = vec![
        ("s3://10.0.0.1:9000/bucket/key", true),
        ("s3://192.168.1.1:9000/bucket/key", true),
        ("s3://172.16.0.1:9000/bucket/key", true),
        ("s3://1.2.3.4:9000/bucket/key", true),
    ];
    
    for (uri, should_have_endpoint) in test_cases {
        let result = parse_s3_uri_full(uri).unwrap();
        assert_eq!(result.endpoint.is_some(), should_have_endpoint, "Failed for URI: {}", uri);
        assert_eq!(result.bucket, "bucket");
        assert_eq!(result.key, "key");
    }
}

#[test]
fn test_fqdn_endpoint_detection() {
    // Fully qualified domain names
    let test_cases = vec![
        "s3://storage.example.com:9000/bucket/key",
        "s3://s3-compatible.mydomain.org:9000/bucket/key",
        "s3://minio.internal.network:9000/bucket/key",
    ];
    
    for uri in test_cases {
        let result = parse_s3_uri_full(uri).unwrap();
        assert!(result.endpoint.is_some(), "Failed to detect endpoint for: {}", uri);
        assert_eq!(result.bucket, "bucket");
        assert_eq!(result.key, "key");
    }
}

#[test]
fn test_nested_paths() {
    // Deep nested paths with custom endpoints
    let uri = "s3://192.168.1.1:9000/mybucket/path/to/deeply/nested/file.dat";
    let result = parse_s3_uri_full(uri).unwrap();
    
    assert_eq!(result.endpoint, Some("192.168.1.1:9000".to_string()));
    assert_eq!(result.bucket, "mybucket");
    assert_eq!(result.key, "path/to/deeply/nested/file.dat");
}

#[test]
fn test_special_characters_in_key() {
    // Keys with special characters
    let uri = "s3://192.168.1.1:9000/bucket/file-name_with.special+chars.txt";
    let result = parse_s3_uri_full(uri).unwrap();
    
    assert_eq!(result.endpoint, Some("192.168.1.1:9000".to_string()));
    assert_eq!(result.bucket, "bucket");
    assert_eq!(result.key, "file-name_with.special+chars.txt");
}

#[test]
fn test_prefix_only_uris() {
    // URIs that represent prefixes (no trailing filename)
    let uri1 = "s3://192.168.1.1:9000/bucket/prefix/";
    let result1 = parse_s3_uri_full(uri1).unwrap();
    assert_eq!(result1.endpoint, Some("192.168.1.1:9000".to_string()));
    assert_eq!(result1.bucket, "bucket");
    assert_eq!(result1.key, "prefix/");
    
    // Standard format prefix
    let uri2 = "s3://bucket/prefix/";
    let result2 = parse_s3_uri_full(uri2).unwrap();
    assert_eq!(result2.endpoint, None);
    assert_eq!(result2.bucket, "bucket");
    assert_eq!(result2.key, "prefix/");
}

#[test]
fn test_error_cases() {
    // Missing s3:// prefix
    assert!(parse_s3_uri_full("bucket/key").is_err());
    
    // Missing slash
    assert!(parse_s3_uri_full("s3://bucket").is_err());
    
    // Empty bucket with endpoint
    assert!(parse_s3_uri_full("s3://192.168.1.1:9000//key").is_err());
}

#[test]
fn test_component_struct_clone() {
    // Test that S3UriComponents can be cloned
    let result = parse_s3_uri_full("s3://192.168.1.1:9000/bucket/key").unwrap();
    let cloned = result.clone();
    
    assert_eq!(result.endpoint, cloned.endpoint);
    assert_eq!(result.bucket, cloned.bucket);
    assert_eq!(result.key, cloned.key);
}

#[test]
fn test_component_struct_debug() {
    // Test that S3UriComponents implements Debug
    let result = parse_s3_uri_full("s3://192.168.1.1:9000/bucket/key").unwrap();
    let debug_str = format!("{:?}", result);
    
    assert!(debug_str.contains("endpoint"));
    assert!(debug_str.contains("bucket"));
    assert!(debug_str.contains("key"));
}
