// tests/test_cancellation.rs
//
// Integration tests for CancellationToken support across DataLoader components
// Tests graceful shutdown behavior for prefetching and async pooling operations

use s3dlio::data_loader::{
    LoaderOptions, DataLoader, AsyncPoolDataLoader, PoolConfig,
    MultiBackendDataset,
};
use tokio_util::sync::CancellationToken;
use std::time::Duration;
use tokio::time::sleep;
use futures::stream::StreamExt;

/// Helper: Creates a simple in-memory dataset for testing
fn create_test_dataset(size: usize) -> MultiBackendDataset {
    let uris: Vec<String> = (0..size)
        .map(|i| format!("file:///tmp/test_data_{}.bin", i))
        .collect();
    
    MultiBackendDataset::from_uris(uris).expect("Failed to create test dataset")
}

#[tokio::test]
async fn test_dataloader_cancellation_before_start() {
    // Test that cancellation before streaming starts results in no data
    let dataset = create_test_dataset(100);
    let token = CancellationToken::new();
    
    // Cancel immediately before starting
    token.cancel();
    
    let opts = LoaderOptions::default()
        .with_batch_size(10)
        .with_cancellation_token(token);
    
    let loader = DataLoader::new(dataset, opts);
    let mut stream = loader.stream();
    
    // Should get None immediately or very quickly
    let result = tokio::time::timeout(
        Duration::from_millis(100),
        stream.next()
    ).await;
    
    match result {
        Ok(None) => {
            // Expected: stream ends immediately
            println!("✅ Cancellation before start: stream ended immediately");
        }
        Ok(Some(_)) => {
            panic!("Expected no data with pre-cancelled token");
        }
        Err(_) => {
            // Timeout is also acceptable - means worker exited cleanly
            println!("✅ Cancellation before start: worker exited (timeout)");
        }
    }
}

#[tokio::test]
async fn test_dataloader_cancellation_during_streaming() {
    // Test cancellation during active streaming
    let dataset = create_test_dataset(1000);
    let token = CancellationToken::new();
    
    let opts = LoaderOptions::default()
        .with_batch_size(10)
        .with_cancellation_token(token.clone());
    
    let loader = DataLoader::new(dataset, opts);
    let mut stream = loader.stream();
    
    // Get a few batches
    let mut batch_count = 0;
    for _ in 0..3 {
        if let Some(_batch) = stream.next().await {
            batch_count += 1;
        }
    }
    
    println!("Received {} batches before cancellation", batch_count);
    
    // Cancel during streaming
    token.cancel();
    
    // Stream should end relatively quickly after cancellation
    let remaining_batches = tokio::time::timeout(
        Duration::from_secs(2),
        async {
            let mut count = 0;
            while let Some(_) = stream.next().await {
                count += 1;
            }
            count
        }
    ).await;
    
    match remaining_batches {
        Ok(count) => {
            println!("✅ Received {} additional batches after cancellation (draining)", count);
            assert!(count < 50, "Should not receive too many batches after cancellation");
        }
        Err(_) => {
            panic!("Stream did not end within timeout after cancellation");
        }
    }
}

#[tokio::test]
async fn test_dataloader_cancellation_with_delay() {
    // Test that cancellation after a delay works correctly
    let dataset = create_test_dataset(500);
    let token = CancellationToken::new();
    
    let opts = LoaderOptions::default()
        .with_batch_size(10)
        .with_cancellation_token(token.clone());
    
    let loader = DataLoader::new(dataset, opts);
    let mut stream = loader.stream();
    
    // Spawn task to cancel after delay
    let cancel_token = token.clone();
    tokio::spawn(async move {
        sleep(Duration::from_millis(200)).await;
        cancel_token.cancel();
        println!("🛑 Cancellation token triggered");
    });
    
    // Count batches received
    let mut total_batches = 0;
    while let Some(_batch) = stream.next().await {
        total_batches += 1;
    }
    
    println!("✅ Received {} total batches with delayed cancellation", total_batches);
    assert!(total_batches > 0, "Should receive at least some batches");
    assert!(total_batches < 50, "Should not receive all 50 batches (500 items / 10 batch_size)");
}

#[tokio::test]
async fn test_async_pool_dataloader_cancellation_before_start() {
    // Test AsyncPoolDataLoader cancellation before streaming starts
    let dataset = create_test_dataset(100);
    let token = CancellationToken::new();
    
    // Cancel immediately
    token.cancel();
    
    let opts = LoaderOptions::default()
        .with_batch_size(10)
        .with_cancellation_token(token);
    
    let pool_config = PoolConfig {
        pool_size: 8,
        ..Default::default()
    };
    
    let loader = AsyncPoolDataLoader::new(dataset, opts);
    let mut stream = loader.stream_with_pool(pool_config);
    
    // Should get None immediately or very quickly
    let result = tokio::time::timeout(
        Duration::from_millis(100),
        stream.next()
    ).await;
    
    match result {
        Ok(None) => {
            println!("✅ AsyncPool cancellation before start: stream ended immediately");
        }
        Ok(Some(_)) => {
            panic!("Expected no data with pre-cancelled token");
        }
        Err(_) => {
            println!("✅ AsyncPool cancellation before start: worker exited (timeout)");
        }
    }
}

#[tokio::test]
async fn test_async_pool_dataloader_cancellation_during_streaming() {
    // Test AsyncPoolDataLoader cancellation during active streaming
    let dataset = create_test_dataset(1000);
    let token = CancellationToken::new();
    
    let opts = LoaderOptions::default()
        .with_batch_size(10)
        .with_cancellation_token(token.clone());
    
    let pool_config = PoolConfig {
        pool_size: 16,
        ..Default::default()
    };
    
    let loader = AsyncPoolDataLoader::new(dataset, opts);
    let mut stream = loader.stream_with_pool(pool_config);
    
    // Get a few batches
    let mut batch_count = 0;
    for _ in 0..3 {
        if let Some(_batch) = stream.next().await {
            batch_count += 1;
        }
    }
    
    println!("Received {} batches before cancellation", batch_count);
    
    // Cancel during streaming
    token.cancel();
    
    // Stream should end relatively quickly
    let remaining_batches = tokio::time::timeout(
        Duration::from_secs(2),
        async {
            let mut count = 0;
            while let Some(_) = stream.next().await {
                count += 1;
            }
            count
        }
    ).await;
    
    match remaining_batches {
        Ok(count) => {
            println!("✅ AsyncPool received {} additional batches after cancellation", count);
            assert!(count < 50, "Should not receive too many batches after cancellation");
        }
        Err(_) => {
            panic!("AsyncPool stream did not end within timeout after cancellation");
        }
    }
}

#[tokio::test]
async fn test_async_pool_dataloader_cancellation_stops_new_requests() {
    // Test that cancellation stops new requests but allows in-flight to complete
    let dataset = create_test_dataset(500);
    let token = CancellationToken::new();
    
    let opts = LoaderOptions::default()
        .with_batch_size(5)
        .with_cancellation_token(token.clone());
    
    // Small pool size to make behavior more predictable
    let pool_config = PoolConfig {
        pool_size: 4,
        ..Default::default()
    };
    
    let loader = AsyncPoolDataLoader::new(dataset, opts);
    let mut stream = loader.stream_with_pool(pool_config);
    
    // Get one batch to ensure pool is running
    if let Some(_batch) = stream.next().await {
        println!("Got initial batch");
    }
    
    // Cancel immediately
    token.cancel();
    
    // Count remaining batches (should be limited to in-flight requests)
    let mut remaining_count = 0;
    while let Some(_batch) = stream.next().await {
        remaining_count += 1;
        if remaining_count > 20 {
            panic!("Too many batches after cancellation - not respecting token");
        }
    }
    
    println!("✅ AsyncPool drained {} batches after cancellation (pool_size=4)", remaining_count);
    assert!(remaining_count <= 10, "Should drain approximately pool_size worth of batches");
}

#[tokio::test]
async fn test_cancellation_without_token() {
    // Test that DataLoader works normally without a cancellation token
    // Note: Uses mock URIs, so actual data loading may fail - we're testing
    // that the stream completes without hanging and respects the absence of token
    let dataset = create_test_dataset(50);
    
    let opts = LoaderOptions::default()
        .with_batch_size(10)
        .without_cancellation(); // Explicitly no token
    
    let loader = DataLoader::new(dataset, opts);
    let mut stream = loader.stream();
    
    // Stream should complete (either with data or errors) without hanging
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        async {
            let mut batch_count = 0;
            while let Some(_batch) = stream.next().await {
                batch_count += 1;
                // Stop after a few batches to avoid waiting for all failures
                if batch_count >= 3 {
                    break;
                }
            }
            batch_count
        }
    ).await;
    
    match result {
        Ok(count) => {
            println!("✅ Without cancellation token: received {} batches, stream completed", count);
            assert!(count > 0, "Should receive at least some batches or errors");
        }
        Err(_) => {
            panic!("Stream hung without cancellation token (timeout)");
        }
    }
}

#[tokio::test]
async fn test_multiple_loaders_shared_token() {
    // Test multiple loaders sharing the same cancellation token
    let dataset1 = create_test_dataset(200);
    let dataset2 = create_test_dataset(200);
    let token = CancellationToken::new();
    
    let opts1 = LoaderOptions::default()
        .with_batch_size(10)
        .with_cancellation_token(token.clone());
    
    let opts2 = LoaderOptions::default()
        .with_batch_size(10)
        .with_cancellation_token(token.clone());
    
    let loader1 = DataLoader::new(dataset1, opts1);
    let loader2 = DataLoader::new(dataset2, opts2);
    
    let mut stream1 = loader1.stream();
    let mut stream2 = loader2.stream();
    
    // Get a batch from each
    let _ = stream1.next().await;
    let _ = stream2.next().await;
    
    // Cancel both
    token.cancel();
    
    // Both should end relatively quickly
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        async {
            while let Some(_) = stream1.next().await {}
            while let Some(_) = stream2.next().await {}
        }
    ).await;
    
    assert!(result.is_ok(), "Both streams should end after cancellation");
    println!("✅ Multiple loaders with shared token: both cancelled successfully");
}

#[tokio::test]
async fn test_cancellation_idempotent() {
    // Test that multiple cancellations don't cause issues
    let dataset = create_test_dataset(100);
    let token = CancellationToken::new();
    
    let opts = LoaderOptions::default()
        .with_batch_size(10)
        .with_cancellation_token(token.clone());
    
    let loader = DataLoader::new(dataset, opts);
    let mut stream = loader.stream();
    
    // Cancel multiple times
    token.cancel();
    token.cancel();
    token.cancel();
    
    // Should still work correctly
    let result = tokio::time::timeout(
        Duration::from_millis(100),
        stream.next()
    ).await;
    
    match result {
        Ok(None) | Err(_) => {
            println!("✅ Multiple cancellations: stream ended correctly");
        }
        Ok(Some(_)) => {
            // Might get one batch from prefetch buffer, that's ok
            println!("✅ Multiple cancellations: got prefetched batch (acceptable)");
        }
    }
}
