// debug_async_pool.rs - Simple debug program to test async pool batching

use s3dlio::data_loader::{AsyncPoolDataLoader, MultiBackendDataset, LoaderOptions, PoolConfig};
use s3dlio::object_store::store_for_uri;
use std::time::Duration;
use tempfile::TempDir;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Debug: Testing async pool batching logic");
    
    // Create a simple test case
    let temp_dir = TempDir::new()?;
    let mut uris = Vec::new();
    
    // Create exactly 5 files
    for i in 0..5 {
        let file_path = temp_dir.path().join(format!("test_{}.dat", i));
        let uri = format!("file://{}", file_path.display());
        
        let data = format!("Data for file {}", i).into_bytes();
        let store = store_for_uri(&uri)?;
        store.put(&uri, &data).await?;
        
        uris.push(uri);
        println!("Created file {}: {}", i, file_path.display());
    }
    
    // Test with batch_size=2, drop_last=false
    // Should get: batch1=[file0,file1], batch2=[file2,file3], batch3=[file4]
    let dataset = MultiBackendDataset::from_uris(uris)?;
    println!("Dataset has {} items", dataset.len());
    
    let options = LoaderOptions {
        batch_size: 2,
        drop_last: false,
        ..Default::default()
    };
    
    let pool_config = PoolConfig {
        pool_size: 4,
        readahead_batches: 2,
        batch_timeout: Duration::from_millis(1000),
        max_inflight: 8,
    };
    
    let dataloader = AsyncPoolDataLoader::new(dataset, options);
    let mut stream = dataloader.stream_with_pool(pool_config);
    
    let mut batch_count = 0;
    let mut total_items = 0;
    
    println!("Starting to collect batches...");
    while let Some(batch_result) = stream.next().await {
        match batch_result {
            Ok(batch) => {
                batch_count += 1;
                total_items += batch.len();
                println!("Batch {}: {} items", batch_count, batch.len());
                for (i, data) in batch.iter().enumerate() {
                    println!("  Item {}: {} bytes", i, data.len());
                }
            }
            Err(e) => {
                println!("Error: {}", e);
                break;
            }
        }
    }
    
    println!("Summary:");
    println!("  Total batches: {}", batch_count);
    println!("  Total items: {}", total_items);
    println!("  Expected: 3 batches, 5 items");
    
    if batch_count == 3 && total_items == 5 {
        println!("✅ SUCCESS!");
    } else {
        println!("❌ FAILED!");
    }
    
    Ok(())
}
