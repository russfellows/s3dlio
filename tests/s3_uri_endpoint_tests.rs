// Integration tests for S3 URI parsing with endpoint support
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use s3dlio::{parse_s3_uri, parse_s3_uri_full, S3UriComponents};

#[test]
fn test_s3_uri_components_type() {
    // Test that we can explicitly type the return value
    let components: S3UriComponents =
        parse_s3_uri_full("s3://192.168.1.1:9000/bucket/key").unwrap();

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
    let uris = [
        "s3://192.168.100.1:9001/bucket/test.dat",
        "s3://192.168.100.1:9002/bucket/test.dat",
        "s3://192.168.100.2:9001/bucket/test.dat",
        "s3://192.168.100.2:9002/bucket/test.dat",
    ];

    let expected_endpoints = [
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
    let uris = [
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
    assert_eq!(
        custom_result.endpoint,
        Some("192.168.1.100:9000".to_string())
    );
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
        assert_eq!(
            result.endpoint.is_some(),
            should_have_endpoint,
            "Failed for URI: {}",
            uri
        );
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
        assert!(
            result.endpoint.is_some(),
            "Failed to detect endpoint for: {}",
            uri
        );
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

// RED-then-GREEN regression tests for s3dlio issue #154 bug 4.1 (B8).
//
// Bug: the endpoint-detection heuristic fired on ANY name with >= 2 dots,
// a digit prefix, or a "minio"/"ceph"/"localhost" substring prefix --
// misrouting legitimate S3 bucket names that happen to use dots (a
// common naming convention) or start with a digit (explicitly legal per
// S3 bucket naming rules) or merely contain those cluster-software names
// as a prefix of an otherwise-unrelated bucket name.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// Sec 4, bug B8): narrow the heuristic to fire only on strings that
// grammatically look like hostnames -- contain a colon (port), are a
// dotted-quad IPv4 address, or end in a known endpoint-hint TLD/suffix
// (configurable via S3DLIO_S3_ENDPOINT_HINT_TLDS). "Bucket with dots" and
// "bucket starting with a digit" no longer misroute; genuine
// endpoint-in-URI usage (with a port, or a recognizable hostname suffix)
// still works.

#[test]
fn test_dotted_bucket_names_no_longer_misroute_as_endpoints() {
    // audit f24 / B8: >= 2 dots alone used to force endpoint routing.
    let result = parse_s3_uri_full("s3://mycompany.data.backups/mykey.dat").unwrap();
    assert_eq!(
        result.endpoint, None,
        "dotted bucket name must not be misrouted as an endpoint"
    );
    assert_eq!(result.bucket, "mycompany.data.backups");
    assert_eq!(result.key, "mykey.dat");
}

#[test]
fn test_digit_prefixed_bucket_names_no_longer_misroute_as_endpoints() {
    // S3 bucket naming rules explicitly permit a leading digit.
    let result = parse_s3_uri_full("s3://123-logs/2024/foo.dat").unwrap();
    assert_eq!(result.endpoint, None);
    assert_eq!(result.bucket, "123-logs");
    assert_eq!(result.key, "2024/foo.dat");
}

#[test]
fn test_minio_prefixed_bucket_name_no_longer_misroutes_as_endpoint() {
    let result = parse_s3_uri_full("s3://minio-data/foo").unwrap();
    assert_eq!(result.endpoint, None);
    assert_eq!(result.bucket, "minio-data");
    assert_eq!(result.key, "foo");
}

#[test]
fn test_ceph_prefixed_bucket_name_no_longer_misroutes_as_endpoint() {
    let result = parse_s3_uri_full("s3://cephcluster1/bar").unwrap();
    assert_eq!(result.endpoint, None);
    assert_eq!(result.bucket, "cephcluster1");
    assert_eq!(result.key, "bar");
}

#[test]
fn test_localhost_prefixed_bucket_name_no_longer_misroutes_as_endpoint() {
    let result = parse_s3_uri_full("s3://localhost-shard/x").unwrap();
    assert_eq!(result.endpoint, None);
    assert_eq!(result.bucket, "localhost-shard");
    assert_eq!(result.key, "x");
}

#[test]
fn test_hostname_like_dotted_names_still_route_as_endpoints() {
    // Genuine hostnames (recognizable TLD-like suffix) without a port
    // must still be detected as endpoints -- the narrowing must not
    // regress the endpoint-in-URI workflow for its typical form.
    let result = parse_s3_uri_full("s3://minio.example.com/bucket/key").unwrap();
    assert_eq!(result.endpoint, Some("minio.example.com".to_string()));
    assert_eq!(result.bucket, "bucket");
    assert_eq!(result.key, "key");

    let result2 = parse_s3_uri_full("s3://s3.company.internal/bucket/key").unwrap();
    assert_eq!(result2.endpoint, Some("s3.company.internal".to_string()));
    assert_eq!(result2.bucket, "bucket");
    assert_eq!(result2.key, "key");
}

#[test]
fn test_endpoint_hint_tld_env_var_extends_the_default_list() {
    // A bucket-shaped-but-dotted name with a suffix NOT in the default
    // list stays a bucket by default...
    let uri = "s3://s3.mycorp.storagegrid/bucket/key";
    let result = parse_s3_uri_full(uri).unwrap();
    assert_eq!(
        result.endpoint, None,
        "unrecognized suffix must not be treated as an endpoint hint by default"
    );

    // ...but adding it via the env var makes the same URI route as an endpoint.
    std::env::set_var("S3DLIO_S3_ENDPOINT_HINT_TLDS", "storagegrid");
    let result2 = parse_s3_uri_full(uri).unwrap();
    std::env::remove_var("S3DLIO_S3_ENDPOINT_HINT_TLDS");
    assert_eq!(
        result2.endpoint,
        Some("s3.mycorp.storagegrid".to_string()),
        "S3DLIO_S3_ENDPOINT_HINT_TLDS must extend the default suffix list"
    );
    assert_eq!(result2.bucket, "bucket");
}
