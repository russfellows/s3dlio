// tests/common/mod.rs
//
// Common test utilities and configuration for GCS tests.

use std::env;

/// Test configuration for GCS tests
pub struct TestConfig {
    /// The GCS bucket to use for testing
    pub bucket: String,
    /// The prefix for all test objects
    pub test_prefix: String,
}

/// Get test configuration from environment
pub fn get_test_config() -> TestConfig {
    let bucket = env::var("GCS_TEST_BUCKET")
        .unwrap_or_else(|_| panic!(
            "GCS_TEST_BUCKET environment variable not set. \
             Please set it to your test bucket name."
        ));
    
    TestConfig {
        bucket,
        test_prefix: "s3dlio-test".to_string(),
    }
}

/// Print test header with formatting
pub fn print_test_header(test_name: &str, backend_name: &str) {
    println!("\n{}", "=".repeat(60));
    println!("TEST: {}", test_name);
    println!("Backend: {}", backend_name);
    println!("{}", "=".repeat(60));
}

/// Print test result
pub fn print_test_result(success: bool) {
    if success {
        println!("\n✅ TEST PASSED\n");
    } else {
        println!("\n❌ TEST FAILED\n");
    }
}
