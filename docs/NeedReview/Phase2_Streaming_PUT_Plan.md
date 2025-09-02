# Phase 2: Streaming Multipart PUT Optimization

## Overview
Transform PUT operations from buffered to streaming, eliminate unnecessary copies, and optimize multipart uploads for maximum throughput and minimum memory usage.

## Current Issues Analysis

1. **Full Object Buffering**: `MultipartUploadSink` accumulates entire object in `Vec<u8>`
2. **Multiple Copies**: `to_vec()` calls create unnecessary data copies
3. **Upload Delay**: Parts only uploaded on `finish()`, not as data arrives
4. **Memory Pressure**: Large objects consume RAM equal to object size

## Changes

### 1. Zero-Copy Multipart Foundation (`src/multipart.rs`)

**Replace existing buffer management with BytesMut/Bytes streaming:**

```rust
use bytes::{Bytes, BytesMut};

/// Enhanced multipart configuration with streaming settings
#[derive(Clone, Debug)]
pub struct MultipartUploadConfig {
    pub part_size: usize,
    pub max_in_flight: usize,
    pub abort_on_drop: bool,
    pub content_type: Option<String>,
    // NEW: streaming-specific settings
    pub buffer_capacity: usize,  // Initial BytesMut capacity
    pub auto_flush: bool,        // Flush parts as they reach part_size
}

impl Default for MultipartUploadConfig {
    fn default() -> Self {
        Self {
            part_size: std::env::var("S3DLIO_PUT_PART_SIZE")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(8 * 1024 * 1024), // 8MB default
            max_in_flight: std::env::var("S3DLIO_PUT_INFLIGHT")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(64), // High concurrency default
            abort_on_drop: true,
            content_type: None,
            buffer_capacity: 16 * 1024 * 1024, // 16MB initial capacity
            auto_flush: true,
        }
    }
}

/// Streaming multipart upload sink with zero-copy optimization
pub struct StreamingMultipartSink {
    client: Client,
    bucket: String,
    key: String,
    upload_id: String,
    cfg: MultipartUploadConfig,

    // NEW: Streaming buffer management
    staging_buffer: BytesMut,
    next_part_number: i32,
    total_bytes: u64,
    started_at: SystemTime,

    // Concurrency control (unchanged)
    sem: Arc<Semaphore>,
    tasks: Vec<JoinHandle<Result<(i32, String)>>>,
    completed: Arc<Mutex<Vec<(i32, String)>>>,
    finished: bool,
}

impl StreamingMultipartSink {
    pub fn new(
        client: Client,
        bucket: String,
        key: String,
        upload_id: String,
        cfg: MultipartUploadConfig,
    ) -> Self {
        let sem = Arc::new(Semaphore::new(cfg.max_in_flight));
        Self {
            client,
            bucket,
            key,
            upload_id,
            staging_buffer: BytesMut::with_capacity(cfg.buffer_capacity),
            next_part_number: 1,
            total_bytes: 0,
            started_at: SystemTime::now(),
            sem,
            tasks: Vec::new(),
            completed: Arc::new(Mutex::new(Vec::new())),
            cfg,
            finished: false,
        }
    }

    /// Write data chunk - streams immediately when buffer fills
    pub async fn write_chunk(&mut self, data: &[u8]) -> Result<()> {
        if self.finished {
            bail!("Cannot write to finished upload");
        }

        self.staging_buffer.extend_from_slice(data);
        self.total_bytes += data.len() as u64;

        // Auto-flush when buffer reaches part size
        if self.cfg.auto_flush && self.staging_buffer.len() >= self.cfg.part_size {
            self.flush_part().await?;
        }

        Ok(())
    }

    /// Zero-copy write for owned Bytes (optimal path)
    pub async fn write_owned_bytes(&mut self, data: Bytes) -> Result<()> {
        if self.finished {
            bail!("Cannot write to finished upload");
        }

        let data_len = data.len();
        
        // Optimal case: perfect part size alignment with empty buffer
        if self.staging_buffer.is_empty() && data_len == self.cfg.part_size {
            // Direct upload - zero copy!
            self.upload_part_direct(data).await?;
        } else if self.staging_buffer.len() + data_len >= self.cfg.part_size {
            // Need to flush current buffer first
            if !self.staging_buffer.is_empty() {
                let needed = self.cfg.part_size - self.staging_buffer.len();
                self.staging_buffer.extend_from_slice(&data[..needed]);
                self.flush_part().await?;
                
                // Handle remaining data
                if data_len > needed {
                    self.staging_buffer.extend_from_slice(&data[needed..]);
                }
            } else {
                self.staging_buffer.extend_from_slice(&data);
            }
            
            // Flush if we've accumulated enough
            if self.cfg.auto_flush && self.staging_buffer.len() >= self.cfg.part_size {
                self.flush_part().await?;
            }
        } else {
            // Just accumulate
            self.staging_buffer.extend_from_slice(&data);
        }

        self.total_bytes += data_len as u64;
        Ok(())
    }

    /// Flush current staging buffer as a part (zero-copy split)
    async fn flush_part(&mut self) -> Result<()> {
        if self.staging_buffer.is_empty() {
            return Ok(());
        }

        let part_size = std::cmp::min(self.staging_buffer.len(), self.cfg.part_size);
        let part_data = self.staging_buffer.split_to(part_size).freeze();
        
        self.upload_part_direct(part_data).await
    }

    /// Upload a part directly from Bytes (zero-copy)
    async fn upload_part_direct(&mut self, part_data: Bytes) -> Result<()> {
        let part_number = self.next_part_number;
        self.next_part_number += 1;

        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let key = self.key.clone();
        let upload_id = self.upload_id.clone();
        let completed = self.completed.clone();
        let permit = self.sem.clone().acquire_owned().await?;

        let task = crate::s3_client::run_on_global_rt(async move {
            let _permit = permit; // Hold permit until upload completes

            // Convert Bytes to ByteStream for AWS SDK
            let body = aws_sdk_s3::primitives::ByteStream::from(part_data);
            
            let response = client
                .upload_part()
                .bucket(&bucket)
                .key(&key)
                .upload_id(&upload_id)
                .part_number(part_number)
                .body(body)
                .send()
                .await?;

            let etag = response
                .e_tag()
                .ok_or_else(|| anyhow!("Missing ETag in upload_part response"))?
                .to_string();

            // Record completion
            completed.lock().unwrap().push((part_number, etag.clone()));
            
            Ok((part_number, etag))
        })?;

        self.tasks.push(task);
        Ok(())
    }

    /// Finish upload and complete multipart
    pub async fn finish(mut self) -> Result<MultipartCompleteInfo> {
        if self.finished {
            bail!("Upload already finished");
        }

        // Flush any remaining data
        if !self.staging_buffer.is_empty() {
            self.flush_part().await?;
        }

        // Wait for all parts to complete
        let mut part_results = Vec::new();
        for task in self.tasks {
            let (part_num, etag) = task.await??;
            part_results.push((part_num, etag));
        }

        // Sort parts by number for completion
        part_results.sort_by_key(|(num, _)| *num);

        // Complete multipart upload
        let completed_parts: Vec<_> = part_results
            .into_iter()
            .map(|(num, etag)| {
                aws_sdk_s3::types::CompletedPart::builder()
                    .part_number(num)
                    .e_tag(etag)
                    .build()
            })
            .collect();

        let complete_request = self.client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(&self.upload_id)
            .multipart_upload(
                aws_sdk_s3::types::CompletedMultipartUpload::builder()
                    .set_parts(Some(completed_parts))
                    .build(),
            )
            .send()
            .await?;

        self.finished = true;

        Ok(MultipartCompleteInfo {
            e_tag: complete_request.e_tag().map(String::from),
            total_bytes: self.total_bytes,
            parts: (self.next_part_number - 1) as usize,
            started_at: self.started_at,
            completed_at: SystemTime::now(),
        })
    }
}
```

### 2. Streaming ObjectWriter Trait (`src/object_store.rs`)

**Create a universal streaming writer interface:**

```rust
/// Universal streaming writer for all backends
#[async_trait]
pub trait ObjectWriter: Send {
    /// Write a chunk of data (copies once into internal buffer)
    async fn write_chunk(&mut self, data: &[u8]) -> Result<()>;
    
    /// Write owned bytes (zero-copy when possible)
    async fn write_owned(&mut self, data: Bytes) -> Result<()> {
        // Default implementation converts to slice
        self.write_chunk(&data).await
    }
    
    /// Get total bytes written so far
    fn bytes_written(&self) -> u64;
    
    /// Finish writing and return metadata
    async fn finish(self: Box<Self>) -> Result<ObjectMetadata>;
    
    /// Cancel the write operation
    async fn cancel(self: Box<Self>) -> Result<()> {
        // Default: just drop, subclasses can override for cleanup
        Ok(())
    }
}

/// Configuration for streaming writers
#[derive(Clone, Debug)]
pub struct WriterOptions {
    pub part_size: usize,
    pub max_concurrency: usize,
    pub enable_compression: bool,
    pub enable_checksums: bool,
    pub content_type: Option<String>,
}

impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            part_size: 8 * 1024 * 1024, // 8MB
            max_concurrency: 64,
            enable_compression: true,
            enable_checksums: true,
            content_type: None,
        }
    }
}

// Add to ObjectStore trait
#[async_trait]
pub trait ObjectStore {
    // ... existing methods ...
    
    /// Create a streaming writer for efficient uploads
    async fn create_writer(&self, uri: &str, options: Option<WriterOptions>) -> Result<Box<dyn ObjectWriter>>;
}
```

### 3. S3 Streaming Writer Implementation (`src/object_store.rs`)

**Implement streaming writer for S3:**

```rust
/// High-performance S3 streaming writer
pub struct S3StreamingWriter {
    sink: StreamingMultipartSink,
    bytes_written: u64,
    uri: String,
}

#[async_trait]
impl ObjectWriter for S3StreamingWriter {
    async fn write_chunk(&mut self, data: &[u8]) -> Result<()> {
        self.sink.write_chunk(data).await?;
        self.bytes_written += data.len() as u64;
        Ok(())
    }
    
    async fn write_owned(&mut self, data: Bytes) -> Result<()> {
        let len = data.len() as u64;
        self.sink.write_owned_bytes(data).await?;
        self.bytes_written += len;
        Ok(())
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    async fn finish(self: Box<Self>) -> Result<ObjectMetadata> {
        let info = self.sink.finish().await?;
        
        // Return metadata (stat the uploaded object for size confirmation)
        stat_object_uri_async(&self.uri).await
    }
}

// S3ObjectStore implementation
impl ObjectStore for S3ObjectStore {
    async fn create_writer(&self, uri: &str, options: Option<WriterOptions>) -> Result<Box<dyn ObjectWriter>> {
        let opts = options.unwrap_or_default();
        let (bucket, key) = parse_s3_uri(uri)?;
        
        let client = get_s3_client().await?;
        
        // Start multipart upload
        let response = client
            .create_multipart_upload()
            .bucket(&bucket)
            .key(&key)
            .set_content_type(opts.content_type)
            .send()
            .await?;
            
        let upload_id = response
            .upload_id()
            .ok_or_else(|| anyhow!("No upload ID returned"))?
            .to_string();
        
        let config = MultipartUploadConfig {
            part_size: opts.part_size,
            max_in_flight: opts.max_concurrency,
            auto_flush: true,
            ..Default::default()
        };
        
        let sink = StreamingMultipartSink::new(client, bucket, key, upload_id, config);
        
        Ok(Box::new(S3StreamingWriter {
            sink,
            bytes_written: 0,
            uri: uri.to_string(),
        }))
    }
}
```

### 4. FileSystem Streaming Writer (`src/file_store.rs`)

**Add streaming support for filesystem backend:**

```rust
/// Streaming filesystem writer with atomic completion
pub struct FileSystemStreamingWriter {
    temp_path: PathBuf,
    final_path: PathBuf,
    file: Option<tokio::fs::File>,
    bytes_written: u64,
    finished: bool,
}

#[async_trait]
impl ObjectWriter for FileSystemStreamingWriter {
    async fn write_chunk(&mut self, data: &[u8]) -> Result<()> {
        if let Some(ref mut file) = self.file {
            file.write_all(data).await?;
            self.bytes_written += data.len() as u64;
            Ok(())
        } else {
            bail!("Writer is closed")
        }
    }
    
    async fn write_owned(&mut self, data: Bytes) -> Result<()> {
        // Convert to slice for filesystem (no zero-copy benefit here)
        self.write_chunk(&data).await
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    async fn finish(mut self: Box<Self>) -> Result<ObjectMetadata> {
        if let Some(file) = self.file.take() {
            file.sync_all().await?;
            drop(file);
            
            // Atomic rename from temp to final location
            tokio::fs::rename(&self.temp_path, &self.final_path).await?;
            
            // Return metadata
            let metadata = tokio::fs::metadata(&self.final_path).await?;
            Ok(ObjectMetadata {
                size: metadata.len(),
                e_tag: None,
                last_modified: metadata.modified().ok(),
            })
        } else {
            bail!("Writer already finished")
        }
    }
}

// FileSystemObjectStore implementation
impl ObjectStore for FileSystemObjectStore {
    async fn create_writer(&self, uri: &str, _options: Option<WriterOptions>) -> Result<Box<dyn ObjectWriter>> {
        let path = Path::new(&uri[7..]); // Remove "file://" prefix
        let temp_path = path.with_extension(format!("{}.tmp", path.extension().unwrap_or_default().to_string_lossy()));
        
        // Ensure parent directory exists
        if let Some(parent) = temp_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        
        let file = tokio::fs::File::create(&temp_path).await?;
        
        Ok(Box::new(FileSystemStreamingWriter {
            temp_path,
            final_path: path.to_path_buf(),
            file: Some(file),
            bytes_written: 0,
            finished: false,
        }))
    }
}
```

### 5. CLI Integration (`src/bin/cli.rs`)

**Add streaming PUT command:**

```rust
// Replace buffered PUT with streaming PUT
async fn put_object_streaming(store: &dyn ObjectStore, uri: &str, file_path: &str) -> Result<()> {
    let mut writer = store.create_writer(uri, None).await?;
    let mut file = tokio::fs::File::open(file_path).await?;
    
    // Stream in chunks
    let mut buffer = vec![0u8; 8 * 1024 * 1024]; // 8MB chunks
    
    loop {
        match file.read(&mut buffer).await? {
            0 => break, // EOF
            n => {
                writer.write_chunk(&buffer[..n]).await?;
            }
        }
    }
    
    let metadata = writer.finish().await?;
    println!("Uploaded {} bytes, ETag: {:?}", metadata.size, metadata.e_tag);
    Ok(())
}
```

## Environment Variables

```bash
# Multipart upload tuning
export S3DLIO_PUT_PART_SIZE=8388608     # 8MB parts (good balance)
export S3DLIO_PUT_INFLIGHT=64           # 64 concurrent uploads
export S3DLIO_PUT_BUFFER_CAPACITY=16777216  # 16MB initial buffer
```

## Expected Results

- **Memory Usage**: Constant ~16MB instead of full object size
- **Latency**: First byte uploads immediately, not on finish()
- **Throughput**: 2-3x improvement due to streaming + high concurrency
- **CPU**: Reduced due to fewer memory copies

## Compatibility

- **Backwards Compatible**: Existing `put()` methods unchanged
- **Opt-in**: New streaming API via `create_writer()`
- **Fallback**: All backends implement the interface (FS with temp files)

## Testing

```bash
# Test streaming upload of large file
time ./target/release/s3-cli put-streaming s3://bucket/large-file.bin /path/to/large-file.bin

# Compare with buffered approach
time ./target/release/s3-cli put s3://bucket/large-file-buffered.bin /path/to/large-file.bin
```
