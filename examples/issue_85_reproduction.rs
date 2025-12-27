// examples/issue_85_reproduction.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Reproduction of Issue #85: FileSystemConfig Type Mismatch
//!
//! This example demonstrates the bug that existed in v0.9.31 and earlier:
//! - The public API exported `crate::file_store::FileSystemConfig`
//! - But `store_for_uri_with_config()` expected `crate::file_store_direct::FileSystemConfig`
//! - Result: Code using the documented public API would not compile
//!
//! **Before Fix (v0.9.31)**: This would fail with:
//! ```
//! error[E0308]: mismatched types
//! expected `s3dlio::file_store_direct::FileSystemConfig`
//! found `s3dlio::file_store::FileSystemConfig`
//! ```
//!
//! **After Fix (v0.9.32)**: This compiles and runs successfully!

use anyhow::Result;
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

// This is what users would naturally write based on the public API documentation
use s3dlio::api::{FileSystemConfig, PageCacheMode, StorageConfig, store_for_uri_with_config};

#[tokio::main]
async fn main() -> Result<()> {
    println!("Issue #85 Reproduction Test");
    println!("============================\n");
    
    // Create a test file
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("test.txt");
    let mut file = File::create(&test_file)?;
    file.write_all(b"Hello from Issue #85 test!")?;
    drop(file);
    
    println!("✓ Created test file: {}", test_file.display());
    
    // Create FileSystemConfig with page_cache_mode - the feature users want to use
    let config = FileSystemConfig {
        enable_range_engine: false,
        range_engine: Default::default(),
        page_cache_mode: Some(PageCacheMode::Sequential),
    };
    
    println!("✓ Created FileSystemConfig with PageCacheMode::Sequential");
    
    // Before Fix (v0.9.31): This line would cause a compile error!
    // After Fix (v0.9.32): This works!
    let uri = format!("file://{}", test_file.display());
    let store = store_for_uri_with_config(&uri, Some(StorageConfig::File(config)))?;
    
    println!("✓ Successfully created store with config (this would have FAILED in v0.9.31)");
    
    // Verify it actually works
    let data = store.get(&uri).await?;
    assert_eq!(&data[..], b"Hello from Issue #85 test!");
    
    println!("✓ Successfully read data: {:?}", String::from_utf8_lossy(&data));
    
    println!("\n✅ SUCCESS! Issue #85 is FIXED!");
    println!("\nIn v0.9.31 and earlier, this example would not even compile.");
    println!("The compiler would complain about mismatched FileSystemConfig types.");
    println!("Now it compiles and runs perfectly!\n");
    
    Ok(())
}
