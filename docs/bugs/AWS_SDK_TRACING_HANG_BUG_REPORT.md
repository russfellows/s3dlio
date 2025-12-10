# Bug Report: Tracing debug!() macros cause hangs inside tokio::spawn with AWS SDK S3 operations

**Filed as:** [aws-sdk-rust#1388](https://github.com/awslabs/aws-sdk-rust/issues/1388)

## Summary

When using `tracing::debug!()` (or other tracing macros) **inside a `tokio::spawn` async task** that also performs AWS SDK S3 operations, the application hangs indefinitely. The same tracing macros work correctly when placed **outside** the spawned task.

## Environment

- **OS**: Linux (Ubuntu 24.04)
- **Rust Version**: rustc 1.87.0 (stable)
- **AWS SDK Versions**:
  - `aws-sdk-s3` = "1.116.0"
  - `aws-config` = "1.8.11"
  - `aws-smithy-http-client` = "1.1.4"
- **Tracing Versions**:
  - `tracing` = "0.1"
  - `tracing-subscriber` = "0.3" (with `env-filter` feature)
- **Tokio Version**: "1.45" (full features)
- **HTTP Stack**:
  - `hyper` = "1.8.1"
  - `hyper-util` = "0.1.19"
  - `hyper-rustls` = "0.27.7"

## Minimal Reproduction

### Cargo.toml

```toml
[package]
name = "aws-tracing-hang-repro"
version = "0.1.0"
edition = "2021"

[dependencies]
aws-config = "1.8.11"
aws-sdk-s3 = "1.116.0"
tokio = { version = "1.45", features = ["full"] }
tokio-stream = "0.1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
anyhow = "1.0"
```

### main.rs - HANGS

```rust
use aws_sdk_s3::Client;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use futures::StreamExt;
use tracing::debug;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing at debug level
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    // Set up credentials (can be dummy for S3-compatible endpoints)
    std::env::set_var("AWS_ACCESS_KEY_ID", "test");
    std::env::set_var("AWS_SECRET_ACCESS_KEY", "test");
    std::env::set_var("AWS_ENDPOINT_URL", "http://127.0.0.1:9000"); // MinIO/LocalStack/etc
    
    let (tx, rx) = mpsc::channel::<Result<String, anyhow::Error>>(1000);
    
    // Spawn task to list objects
    tokio::spawn(async move {
        // Build AWS config and S3 client
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .load()
            .await;
        
        let s3_config = aws_sdk_s3::config::Builder::from(&config)
            .force_path_style(true)
            .build();
        let client = Client::from_conf(s3_config);
        
        // ============================================================
        // THIS debug!() CALL CAUSES THE HANG
        // Comment it out and the code works fine
        // ============================================================
        debug!("About to list objects in bucket");
        
        // Make S3 API call
        let resp = client
            .list_objects_v2()
            .bucket("test-bucket")
            .send()
            .await;
            
        match resp {
            Ok(r) => {
                for obj in r.contents() {
                    if let Some(key) = obj.key() {
                        let _ = tx.send(Ok(key.to_string())).await;
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(anyhow::anyhow!("S3 error: {}", e))).await;
            }
        }
    });
    
    // Consume results from stream
    let mut stream = ReceiverStream::new(rx);
    let mut count = 0;
    while let Some(result) = stream.next().await {
        match result {
            Ok(key) => {
                println!("Object: {}", key);
                count += 1;
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    
    println!("Total objects: {}", count);
    Ok(())
}
```

### main.rs - WORKS (debug outside spawn)

```rust
use aws_sdk_s3::Client;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use futures::StreamExt;
use tracing::debug;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing at debug level
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    std::env::set_var("AWS_ACCESS_KEY_ID", "test");
    std::env::set_var("AWS_SECRET_ACCESS_KEY", "test");
    std::env::set_var("AWS_ENDPOINT_URL", "http://127.0.0.1:9000");
    
    // ============================================================
    // THIS debug!() CALL WORKS FINE - it's OUTSIDE tokio::spawn
    // ============================================================
    debug!("Starting list operation");
    
    let (tx, rx) = mpsc::channel::<Result<String, anyhow::Error>>(1000);
    
    tokio::spawn(async move {
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .load()
            .await;
        
        let s3_config = aws_sdk_s3::config::Builder::from(&config)
            .force_path_style(true)
            .build();
        let client = Client::from_conf(s3_config);
        
        // No debug!() here - removed to avoid hang
        
        let resp = client
            .list_objects_v2()
            .bucket("test-bucket")
            .send()
            .await;
            
        match resp {
            Ok(r) => {
                for obj in r.contents() {
                    if let Some(key) = obj.key() {
                        let _ = tx.send(Ok(key.to_string())).await;
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(anyhow::anyhow!("S3 error: {}", e))).await;
            }
        }
    });
    
    let mut stream = ReceiverStream::new(rx);
    let mut count = 0;
    while let Some(result) = stream.next().await {
        match result {
            Ok(key) => {
                println!("Object: {}", key);
                count += 1;
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    
    println!("Total objects: {}", count);
    Ok(())
}
```

## Observed Behavior

### HANGS (tracing inside spawn):
```bash
$ RUST_LOG=debug cargo run
# Application hangs indefinitely after printing initial tracing output
# No CPU usage - appears to be blocked/deadlocked
# Must be killed with Ctrl+C or timeout
```

### WORKS (tracing outside spawn):
```bash
$ RUST_LOG=debug cargo run
2025-12-04T05:40:54.248296Z DEBUG Starting list operation
Total objects: 0
# Completes successfully
```

## Additional Observations

### 1. Filter-based workaround works

If we filter tracing to only enable debug for our own crate (not AWS SDK internals), the hang does not occur:

```rust
tracing_subscriber::fmt()
    .with_env_filter("warn,my_crate=debug")  // AWS SDK stays at warn
    .init();
```

This suggests the issue may be related to AWS SDK's internal tracing calls interacting with the subscriber.

### 2. Hang location varies

Through debugging with `eprintln!()` statements (which don't hang), we observed:
- The `debug!()` call itself appears to complete
- The hang occurs on the **next** async operation (e.g., `client.list_objects_v2().send().await`)
- Sometimes the hang occurs during `aws_config::load().await`

### 3. Consistent reproduction

- Tested against MinIO and LocalStack (S3-compatible gateways)
- Tested with both default AWS SDK HTTP client and custom hyper configurations
- **Tested with SDK 1.104.0 and 1.116.0 - bug exists in both (not a recent regression)**
- Issue reproduces 100% of the time when debug!() is inside spawn

### 4. No issue without tokio::spawn

If we run the same code **without** `tokio::spawn` (directly in main async block), the `debug!()` calls work fine:

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    
    // This works fine - not in a spawned task
    debug!("Building client");
    
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .load()
        .await;
    
    debug!("Config loaded");  // This also works
    
    let client = Client::new(&config);
    
    debug!("About to list");  // This also works
    
    let resp = client.list_objects_v2().bucket("test").send().await?;
    
    debug!("Listed {} objects", resp.contents().len());  // This works too
    
    Ok(())
}
```

## Hypothesis

The issue appears to be a **deadlock or blocking condition** that occurs when:

1. A tracing subscriber is initialized at debug level
2. Code runs inside a `tokio::spawn` task
3. AWS SDK S3 operations are performed in the same task
4. The AWS SDK (or its dependencies) also emits tracing events

Possible causes:
- **Subscriber lock contention**: The tracing subscriber may hold a lock while formatting output, and the AWS SDK may try to acquire the same lock from a different context
- **Async executor blocking**: The subscriber's I/O (writing to stdout/stderr) may block in a way that's incompatible with the spawned task's executor context
- **Hidden synchronous code**: There may be synchronous tracing calls inside the SDK that block when combined with async operations

## Impact

This bug prevents users from:
- Adding debug logging inside async S3 operations
- Using structured logging for observability in production
- Debugging issues in streaming/paginated S3 operations

## Current Workaround

1. **Move tracing calls outside `tokio::spawn`**:
   ```rust
   debug!("Starting operation");  // OK - outside spawn
   tokio::spawn(async move {
       // No debug!() calls in here
       client.list_objects_v2().send().await
   });
   ```

2. **Filter to exclude AWS SDK from debug logging**:
   ```rust
   tracing_subscriber::fmt()
       .with_env_filter("warn,my_crate=debug")
       .init();
   ```

3. **Use `eprintln!()` for debugging instead of tracing macros**

## Related Issues

- This may be related to general tracing/async interactions
- Similar patterns might affect other AWS SDK operations (DynamoDB, SQS, etc.)

## Request

Please investigate:
1. Whether AWS SDK has internal tracing calls that might cause lock contention
2. Whether there's a known incompatibility between tracing subscribers and spawned async tasks
3. Whether this is actually a tracing-subscriber bug that should be reported there instead

## Test Environment Setup

To reproduce, you need an S3-compatible endpoint. Options:
1. **MinIO**: `docker run -p 9000:9000 minio/minio server /data`
2. **LocalStack**: `docker run -p 4566:4566 localstack/localstack`
3. **Real AWS S3**: Set proper credentials and remove `force_path_style`

---

**Reported by**: s3dlio project (https://github.com/russfellows/s3dlio)  
**Date**: December 3, 2025  
**Discovered during**: Integration testing with multi-protocol gateway
