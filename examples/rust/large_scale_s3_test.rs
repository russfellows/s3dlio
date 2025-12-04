// examples/large_scale_s3_test.rs
//
// Large-scale S3 performance test with comprehensive profiling
// - Creates 50+ GB of test data (1MB to 24MB objects)
// - Tests complete S3 lifecycle: bucket creation, upload, download, cleanup
// - Comprehensive profiling and performance analysis

use anyhow::{Result, Context};
use dotenvy;
use s3dlio::profiling::*;
use s3dlio::profile_span;
use s3dlio::data_gen::generate_controlled_data;
use s3dlio::s3_utils::{self, put_object_uri_async, get_object_uri_optimized_async, stat_object_uri_async};
use std::env;
use std::time::Instant;
use rand::Rng;

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment configuration
    dotenvy::dotenv().ok();
    
    // Initialize comprehensive profiling
    init_profiling()?;
    
    println!("🚀 Large-Scale S3 Performance Test with Profiling");
    println!("=================================================");
    
    // Test configuration
    let test_config = TestConfig::from_env()?;
    test_config.print_summary();
    
    // Run the complete test suite
    let mut test_runner = S3TestRunner::new(test_config).await?;
    
    test_runner.run_complete_test().await?;
    
    println!("\n✅ Large-scale S3 test completed successfully!");
    println!("📊 Check logs and profiling output for performance insights");
    
    Ok(())
}

#[derive(Debug, Clone)]
struct TestConfig {
    test_bucket_prefix: String,
    target_data_size_gb: u64,
    min_object_size_mb: u64,
    max_object_size_mb: u64,
    concurrent_uploads: usize,
    concurrent_downloads: usize,
    dedup_factor: usize,
    compress_factor: usize,
}

impl TestConfig {
    fn from_env() -> Result<Self> {
        Ok(Self {
            test_bucket_prefix: env::var("S3DLIO_TEST_BUCKET_PREFIX")
                .unwrap_or_else(|_| "s3dlio-perf-test".to_string()),
            target_data_size_gb: env::var("S3DLIO_TEST_SIZE_GB")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(50),
            min_object_size_mb: env::var("S3DLIO_MIN_OBJECT_MB")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
            max_object_size_mb: env::var("S3DLIO_MAX_OBJECT_MB")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(24),
            concurrent_uploads: env::var("S3DLIO_UPLOAD_CONCURRENCY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(16),
            concurrent_downloads: env::var("S3DLIO_DOWNLOAD_CONCURRENCY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(32),
            dedup_factor: env::var("S3DLIO_DEDUP_FACTOR")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            compress_factor: env::var("S3DLIO_COMPRESS_FACTOR")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
        })
    }
    
    fn print_summary(&self) {
        println!("📋 Test Configuration:");
        println!("   Target data size: {} GB", self.target_data_size_gb);
        println!("   Object size range: {} - {} MB", self.min_object_size_mb, self.max_object_size_mb);
        println!("   Upload concurrency: {}", self.concurrent_uploads);
        println!("   Download concurrency: {}", self.concurrent_downloads);
        println!("   Test bucket prefix: {}", self.test_bucket_prefix);
        println!();
    }
}

struct S3TestRunner {
    config: TestConfig,
    test_bucket: String,
    test_objects: Vec<TestObject>,
    performance_stats: PerformanceStats,
}

#[derive(Debug, Clone)]
struct TestObject {
    key: String,
    size_bytes: usize,
    uri: String,
}

#[derive(Debug, Default)]
struct PerformanceStats {
    bucket_create_time: std::time::Duration,
    total_upload_time: std::time::Duration,
    total_download_time: std::time::Duration,
    cleanup_time: std::time::Duration,
    total_bytes_uploaded: u64,
    total_bytes_downloaded: u64,
    object_count: usize,
}

impl S3TestRunner {
    async fn new(config: TestConfig) -> Result<Self> {
        let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
        let test_bucket = format!("{}-{}", config.test_bucket_prefix, timestamp);
        
        Ok(Self {
            config,
            test_bucket,
            test_objects: Vec::new(),
            performance_stats: PerformanceStats::default(),
        })
    }
    
    async fn run_complete_test(&mut self) -> Result<()> {
        // Step 1: List existing buckets
        self.list_existing_buckets().await?;
        
        // Step 2: Create test bucket
        self.create_test_bucket().await?;
        
        // Step 3: Generate test plan
        self.generate_test_plan().await?;
        
        // Step 4: Upload all test data
        self.upload_test_data().await?;
        
        // Step 5: Verify bucket contents
        self.verify_bucket_contents().await?;
        
        // Step 6: Download all test data
        self.download_test_data().await?;
        
        // Step 7: Final verification
        self.list_bucket_contents().await?;
        
        // Step 8: Cleanup
        self.cleanup_test_data().await?;
        
        // Step 9: Performance summary
        self.print_performance_summary().await?;
        
        Ok(())
    }
    
    async fn list_existing_buckets(&self) -> Result<()> {
        let _span = profile_span!("list_existing_buckets");
        println!("📋 Listing existing buckets...");
        
        // Note: We don't have a direct list_buckets function, so we'll skip this for now
        // In a real implementation, you'd add this to s3_utils.rs
        println!("   ℹ️  Bucket listing skipped (not implemented in s3_utils)");
        
        Ok(())
    }
    
    async fn create_test_bucket(&mut self) -> Result<()> {
        let _span = profile_span!("create_test_bucket", bucket = %self.test_bucket);
        println!("🪣 Creating test bucket: {}", self.test_bucket);
        
        let start = Instant::now();
        
        // Create bucket using s3_utils function
        s3_utils::create_bucket(&self.test_bucket)
            .context("Failed to create test bucket")?;
        
        self.performance_stats.bucket_create_time = start.elapsed();
        
        println!("   ✅ Bucket created in {:?}", self.performance_stats.bucket_create_time);
        Ok(())
    }
    
    async fn generate_test_plan(&mut self) -> Result<()> {
        let _span = profile_span!("generate_test_plan");
        println!("📝 Generating test plan...");
        
        let target_bytes = self.config.target_data_size_gb * 1024 * 1024 * 1024;
        let min_size_bytes = self.config.min_object_size_mb * 1024 * 1024;
        let max_size_bytes = self.config.max_object_size_mb * 1024 * 1024;
        
        let mut total_bytes = 0u64;
        let mut object_counter = 0;
        let mut rng = rand::rng();
        
        while total_bytes < target_bytes {
            let size_bytes = rng.random_range(min_size_bytes as usize..=max_size_bytes as usize);
            let key = format!("test-object-{:06}-{}MB.dat", object_counter, size_bytes / (1024 * 1024));
            let uri = format!("s3://{}/{}", self.test_bucket, key);
            
            self.test_objects.push(TestObject {
                key,
                size_bytes,
                uri,
            });
            
            total_bytes += size_bytes as u64;
            object_counter += 1;
        }
        
        self.performance_stats.object_count = self.test_objects.len();
        
        println!("   📊 Generated plan for {} objects", self.test_objects.len());
        println!("   📊 Total planned data: {:.2} GB", total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
        println!("   📊 Average object size: {:.2} MB", (total_bytes as f64 / self.test_objects.len() as f64) / (1024.0 * 1024.0));
        
        Ok(())
    }
    
    async fn upload_test_data(&mut self) -> Result<()> {
        let _span = profile_span!("upload_test_data", object_count = self.test_objects.len());
        println!("⬆️  Uploading {} test objects...", self.test_objects.len());
        
        let start = Instant::now();
        
        #[cfg(feature = "profiling")]
        let upload_guard = pprof::ProfilerGuardBuilder::default()
            .frequency(1000)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to start upload profiler: {}", e))?;
        
        // Create a semaphore to limit concurrent uploads
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(self.config.concurrent_uploads));
        let mut upload_tasks = Vec::new();
        
        let mut total_bytes = 0u64;
        
        for (_i, test_obj) in self.test_objects.iter().enumerate() {
            let semaphore = semaphore.clone();
            let uri = test_obj.uri.clone();
            let size = test_obj.size_bytes;
            let dedup = self.config.dedup_factor;
            let compress = self.config.compress_factor;
            
            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                let _span = profile_span!("upload_single_object", size_mb = size / (1024 * 1024), index = i);
                
                // Generate controlled data
                let data = generate_controlled_data(size, dedup, compress);
                
                // Upload to S3
                put_object_uri_async(&uri, &data).await
                    .with_context(|| format!("Failed to upload {}", uri))?;
                
                Ok::<usize, anyhow::Error>(size)
            });
            
            upload_tasks.push(task);
            total_bytes += size as u64;
        }
        
        // Wait for all uploads to complete
        let mut completed_bytes = 0u64;
        for (i, task) in upload_tasks.into_iter().enumerate() {
            match task.await? {
                Ok(size) => {
                    completed_bytes += size as u64;
                    if i % 10 == 0 || i == self.test_objects.len() - 1 {
                        let progress = (completed_bytes as f64 / total_bytes as f64) * 100.0;
                        println!("   📈 Upload progress: {:.1}% ({} objects)", progress, i + 1);
                    }
                }
                Err(e) => {
                    println!("   ❌ Upload failed for object {}: {}", i, e);
                    return Err(e);
                }
            }
        }
        
        self.performance_stats.total_upload_time = start.elapsed();
        self.performance_stats.total_bytes_uploaded = total_bytes;
        
        let throughput_mbps = (total_bytes as f64 / (1024.0 * 1024.0)) / self.performance_stats.total_upload_time.as_secs_f64();
        
        println!("   ✅ Upload completed: {:.2} GB in {:?}", 
                 total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                 self.performance_stats.total_upload_time);
        println!("   📊 Upload throughput: {:.2} MB/s", throughput_mbps);
        
        // Save upload flamegraph
        #[cfg(feature = "profiling")]
        {
            match upload_guard.report().build() {
                Ok(report) => {
                    let file = std::fs::File::create("profiles/large_scale_upload_profile.svg")?;
                    report.flamegraph(file)?;
                    println!("   🔥 Upload flamegraph saved: profiles/large_scale_upload_profile.svg");
                }
                Err(e) => {
                    eprintln!("   ⚠️  Failed to generate upload flamegraph: {}", e);
                }
            }
        }
        
        Ok(())
    }
    
    async fn verify_bucket_contents(&self) -> Result<()> {
        let _span = profile_span!("verify_bucket_contents");
        println!("🔍 Verifying bucket contents...");
        
        // Note: We would need to implement list_objects_detailed to get sizes
        // For now, we'll do a basic verification by statting a few objects
        let sample_size = std::cmp::min(5, self.test_objects.len());
        
        for i in 0..sample_size {
            let obj = &self.test_objects[i];
            match stat_object_uri_async(&obj.uri).await {
                Ok(stat) => {
                    println!("   ✅ {} -> {} bytes", obj.key, stat.size);
                    if stat.size != obj.size_bytes as u64 {
                        anyhow::bail!("Size mismatch for {}: expected {}, got {}", 
                                      obj.key, obj.size_bytes, stat.size);
                    }
                }
                Err(e) => {
                    anyhow::bail!("Failed to stat {}: {}", obj.key, e);
                }
            }
        }
        
        println!("   ✅ Verified {} sample objects", sample_size);
        Ok(())
    }
    
    async fn download_test_data(&mut self) -> Result<()> {
        let _span = profile_span!("download_test_data", object_count = self.test_objects.len());
        println!("⬇️  Downloading {} test objects...", self.test_objects.len());
        
        let start = Instant::now();
        
        #[cfg(feature = "profiling")]
        let download_guard = pprof::ProfilerGuardBuilder::default()
            .frequency(1000)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to start download profiler: {}", e))?;
        
        // Create a semaphore to limit concurrent downloads
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(self.config.concurrent_downloads));
        let mut download_tasks = Vec::new();
        
        for (_i, test_obj) in self.test_objects.iter().enumerate() {
            let semaphore = semaphore.clone();
            let uri = test_obj.uri.clone();
            let expected_size = test_obj.size_bytes;
            
            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                #[allow(unused_variables)] // Used in profiling span when feature enabled
                let _span = profile_span!("download_single_object", 
                    size_mb = expected_size / (1024 * 1024), 
                    index = _i
                );
                
                // Use optimized download (will automatically use concurrent range for large objects)
                let data = get_object_uri_optimized_async(&uri).await
                    .with_context(|| format!("Failed to download {}", uri))?;
                
                if data.len() != expected_size {
                    anyhow::bail!("Size mismatch for {}: expected {}, got {}", 
                                  uri, expected_size, data.len());
                }
                
                Ok::<usize, anyhow::Error>(data.len())
            });
            
            download_tasks.push(task);
        }
        
        // Wait for all downloads to complete
        let mut completed_bytes = 0u64;
        for (i, task) in download_tasks.into_iter().enumerate() {
            match task.await? {
                Ok(size) => {
                    completed_bytes += size as u64;
                    if i % 10 == 0 || i == self.test_objects.len() - 1 {
                        let progress = (completed_bytes as f64 / self.performance_stats.total_bytes_uploaded as f64) * 100.0;
                        println!("   📈 Download progress: {:.1}% ({} objects)", progress, i + 1);
                    }
                }
                Err(e) => {
                    println!("   ❌ Download failed for object {}: {}", i, e);
                    return Err(e);
                }
            }
        }
        
        self.performance_stats.total_download_time = start.elapsed();
        self.performance_stats.total_bytes_downloaded = completed_bytes;
        
        let throughput_mbps = (completed_bytes as f64 / (1024.0 * 1024.0)) / self.performance_stats.total_download_time.as_secs_f64();
        
        println!("   ✅ Download completed: {:.2} GB in {:?}", 
                 completed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                 self.performance_stats.total_download_time);
        println!("   📊 Download throughput: {:.2} MB/s", throughput_mbps);
        
        // Save download flamegraph
        #[cfg(feature = "profiling")]
        {
            match download_guard.report().build() {
                Ok(report) => {
                    let file = std::fs::File::create("profiles/large_scale_download_profile.svg")?;
                    report.flamegraph(file)?;
                    println!("   🔥 Download flamegraph saved: profiles/large_scale_download_profile.svg");
                }
                Err(e) => {
                    eprintln!("   ⚠️  Failed to generate download flamegraph: {}", e);
                }
            }
        }
        
        Ok(())
    }
    
    async fn list_bucket_contents(&self) -> Result<()> {
        let _span = profile_span!("list_bucket_contents");
        println!("📋 Listing bucket contents for final verification...");
        
        // Use the list_objects function
        let objects = s3_utils::list_objects(&self.test_bucket, "", true)
            .context("Failed to list bucket contents")?;
        let object_count = objects.len();
        
        println!("   📊 Found {} objects in bucket", object_count);
        
        if object_count != self.test_objects.len() {
            anyhow::bail!("Object count mismatch: expected {}, found {}", 
                          self.test_objects.len(), object_count);
        }
        
        println!("   ✅ Object count verification passed");
        Ok(())
    }
    
    async fn cleanup_test_data(&mut self) -> Result<()> {
        let _span = profile_span!("cleanup_test_data");
        println!("🧹 Cleaning up test data...");
        
        let start = Instant::now();
        
        // Delete all objects in batches
        let batch_size = 50; // AWS limit for batch delete
        let mut deleted_count = 0;
        
        for chunk in self.test_objects.chunks(batch_size) {
            let keys: Vec<String> = chunk.iter().map(|obj| obj.key.clone()).collect();
            
            s3_utils::delete_objects(&self.test_bucket, &keys)
                .context("Failed to delete batch of objects")?;
            
            deleted_count += chunk.len();
            println!("   🗑️  Deleted {} objects ({}/{})", 
                     chunk.len(), deleted_count, self.test_objects.len());
        }
        
        // Verify bucket is empty
        let remaining_objects = s3_utils::list_objects(&self.test_bucket, "", true)
            .context("Failed to verify bucket is empty")?;

        if !remaining_objects.is_empty() {
            anyhow::bail!("Bucket not empty after cleanup: {} objects remaining", remaining_objects.len());
        }        // Delete the bucket
        s3_utils::delete_bucket(&self.test_bucket)
            .context("Failed to delete test bucket")?;
        
        self.performance_stats.cleanup_time = start.elapsed();
        
        println!("   ✅ Cleanup completed in {:?}", self.performance_stats.cleanup_time);
        println!("   ✅ Test bucket deleted: {}", self.test_bucket);
        
        Ok(())
    }
    
    async fn print_performance_summary(&self) -> Result<()> {
        println!("\n📊 Performance Summary");
        println!("=====================");
        
        let total_gb = self.performance_stats.total_bytes_uploaded as f64 / (1024.0 * 1024.0 * 1024.0);
        let upload_throughput = (self.performance_stats.total_bytes_uploaded as f64 / (1024.0 * 1024.0)) / self.performance_stats.total_upload_time.as_secs_f64();
        let download_throughput = (self.performance_stats.total_bytes_downloaded as f64 / (1024.0 * 1024.0)) / self.performance_stats.total_download_time.as_secs_f64();
        
        println!("📋 Test Scale:");
        println!("   Total data: {:.2} GB", total_gb);
        println!("   Object count: {}", self.performance_stats.object_count);
        println!("   Size range: {} - {} MB", self.config.min_object_size_mb, self.config.max_object_size_mb);
        println!();
        
        println!("⚡ Performance Metrics:");
        println!("   Bucket creation: {:?}", self.performance_stats.bucket_create_time);
        println!("   Upload time: {:?}", self.performance_stats.total_upload_time);
        println!("   Upload throughput: {:.2} MB/s", upload_throughput);
        println!("   Download time: {:?}", self.performance_stats.total_download_time);
        println!("   Download throughput: {:.2} MB/s", download_throughput);
        println!("   Cleanup time: {:?}", self.performance_stats.cleanup_time);
        println!();
        
        println!("🔧 Configuration Used:");
        println!("   Upload concurrency: {}", self.config.concurrent_uploads);
        println!("   Download concurrency: {}", self.config.concurrent_downloads);
        println!("   HTTP max connections: {}", env::var("S3DLIO_HTTP_MAX_CONNECTIONS").unwrap_or("200".to_string()));
        println!("   Runtime threads: {}", env::var("S3DLIO_RUNTIME_THREADS").unwrap_or("auto".to_string()));
        println!("   Range threshold: {} MB", env::var("S3DLIO_RANGE_THRESHOLD_MB").unwrap_or("4".to_string()));
        println!();
        
        println!("📈 Flamegraphs Generated:");
        println!("   Upload profile: profiles/large_scale_upload_profile.svg");
        println!("   Download profile: profiles/large_scale_download_profile.svg");
        
        Ok(())
    }
}
