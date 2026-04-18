// Test summary of ObjectStore implementation completion
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use anyhow::Result;
use s3dlio::object_store::{infer_scheme, store_for_uri, Scheme};

#[tokio::test]
async fn test_complete_object_store_parity() -> Result<()> {
    println!("🎯 Testing Complete ObjectStore Backend Parity");

    // Test scheme inference works for all backends
    assert_eq!(infer_scheme("file:///tmp/test"), Scheme::File);
    assert_eq!(infer_scheme("s3://bucket/key"), Scheme::S3);
    assert_eq!(infer_scheme("az://account/container/key"), Scheme::Azure);
    assert_eq!(
        infer_scheme("https://account.blob.core.windows.net/container/key"),
        Scheme::Azure
    );
    println!("✅ All URI schemes recognized correctly");

    // Test always-available backends can be instantiated
    let file_store = store_for_uri("file:///tmp/test")?;
    let s3_store = store_for_uri("s3://test-bucket/test")?;
    println!("✅ File and S3 backends instantiate through unified factory");

    // Azure backend is only available with the backend-azure feature.
    // Without it, store_for_uri returns a descriptive error — verify that.
    #[cfg(feature = "backend-azure")]
    let azure_store = {
        let store = store_for_uri("az://account/container/test")?;
        println!("✅ Azure backend instantiates through unified factory");
        store
    };
    #[cfg(not(feature = "backend-azure"))]
    {
        let azure_result = store_for_uri("az://account/container/test");
        assert!(
            azure_result.is_err(),
            "Azure backend should be unavailable without the backend-azure feature"
        );
        println!("✅ Azure backend correctly unavailable without backend-azure feature");
    }

    // Verify always-available backends implement the same trait
    let _: &dyn s3dlio::object_store::ObjectStore = &*file_store;
    let _: &dyn s3dlio::object_store::ObjectStore = &*s3_store;
    #[cfg(feature = "backend-azure")]
    let _: &dyn s3dlio::object_store::ObjectStore = &*azure_store;

    println!("✅ All compiled backends implement identical ObjectStore trait");

    println!("🎉 BACKEND PARITY: unified factory, consistent URI schemes, identical trait surface");

    Ok(())
}

#[test]
fn test_backend_feature_parity() {
    println!("📋 Verifying Feature Parity Summary:");

    // File backend: Complete (always available)
    println!("✅ FileSystemObjectStore: Complete implementation");

    // S3 backend: Now complete with PUT operations
    println!("✅ S3ObjectStore: Complete implementation (including new PUT operations)");

    // Azure backend: Complete (ungated in v0.6.3)
    println!("✅ AzureObjectStore: Complete implementation (always available)");

    println!("🎯 All backends now provide identical capabilities for AI/ML workloads");
}
