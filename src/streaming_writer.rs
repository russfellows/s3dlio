use anyhow::Result;
use async_trait::async_trait;
use crc32fast::Hasher;

use crate::data_gen::{DataGenerator, ObjectGen}; 
use crate::object_store::{ObjectWriter, CompressionConfig, ObjectStore, WriterOptions};

/// Streaming data writer that generates synthetic data directly during upload.
/// 
/// This writer integrates the streaming DataGenerator with ObjectWriter to provide
/// zero-copy data generation directly into upload pipelines. Instead of pre-generating
/// large datasets, this allows generating data on-demand during S3/Azure uploads.
pub struct StreamingDataWriter {
    /// Underlying ObjectWriter for the target storage backend
    writer: Box<dyn ObjectWriter>,
    /// Data generator state for current object
    object_gen: Option<ObjectGen>,
    /// Total bytes of synthetic data to generate
    target_size: usize,
    /// Bytes already generated and written
    bytes_generated: u64,
    /// Whether generation is complete
    finalized: bool,
    /// Checksum of generated data (before compression)
    hasher: Hasher,
}

impl StreamingDataWriter {
    /// Create a new StreamingDataWriter for the given URI and parameters.
    /// 
    /// # Parameters
    /// - `uri`: Target URI for upload (e.g., "s3://bucket/key")
    /// - `size`: Total size of synthetic data to generate in bytes
    /// - `dedup`: Deduplication factor (0 treated as 1)
    /// - `compress`: Compression factor for controllable compressibility 
    /// - `store`: ObjectStore for the target backend
    /// - `options`: Writer options (compression, part size, etc.)
    /// 
    /// # Returns
    /// StreamingDataWriter ready for chunk-by-chunk generation and upload
    pub async fn new(
        uri: &str,
        size: usize,
        dedup: usize,
        compress: usize,
        store: &dyn ObjectStore,
        options: WriterOptions,
    ) -> Result<Self> {
        let writer = store.create_writer(uri, options).await?;
        let generator = DataGenerator::new();
        let object_gen = generator.begin_object(size, dedup, compress);
        
        Ok(Self {
            writer,
            object_gen: Some(object_gen),
            target_size: size,
            bytes_generated: 0,
            finalized: false,
            hasher: Hasher::new(),
        })
    }
    
    /// Generate and write the next chunk of synthetic data.
    /// 
    /// This is the core streaming method that generates data in chunks and 
    /// immediately streams it to the underlying ObjectWriter for upload.
    /// 
    /// # Parameters
    /// - `chunk_size`: Size of chunk to generate (may be less at end of object)
    /// 
    /// # Returns
    /// Number of bytes actually generated and written
    pub async fn generate_chunk(&mut self, chunk_size: usize) -> Result<usize> {
        if self.finalized {
            anyhow::bail!("Cannot generate chunks after finalization");
        }
        
        let object_gen = self.object_gen.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Object generation already complete"))?;
            
        if object_gen.is_complete() {
            return Ok(0); // No more data to generate
        }
        
        // Generate the requested chunk
        let chunk = match object_gen.fill_chunk(chunk_size) {
            Some(data) => data,
            None => return Ok(0), // No more data available
        };
        let actual_size = chunk.len();
        
        // Update checksum with generated data (before compression)
        self.hasher.update(&chunk);
        self.bytes_generated += actual_size as u64;
        
        // Stream directly to underlying writer
        self.writer.write_chunk(&chunk).await?;
        
        Ok(actual_size)
    }
    
    /// Generate and write all remaining data.
    /// 
    /// Convenience method that generates all remaining synthetic data
    /// and streams it to the underlying writer.
    pub async fn generate_remaining(&mut self) -> Result<()> {
        while !self.is_complete() {
            self.generate_chunk(64 * 1024).await?; // 64KB chunks
        }
        Ok(())
    }
    
    /// Check if all target data has been generated.
    pub fn is_complete(&self) -> bool {
        self.bytes_generated >= self.target_size as u64 ||
        self.object_gen.as_ref().map_or(true, |obj_gen| obj_gen.is_complete())
    }
    
    /// Get the total bytes of synthetic data generated so far.
    pub fn bytes_generated(&self) -> u64 {
        self.bytes_generated
    }
    
    /// Get the target size of synthetic data to generate.
    pub fn target_size(&self) -> usize {
        self.target_size
    }
}

#[async_trait]
impl ObjectWriter for StreamingDataWriter {
    /// Write a chunk of pre-existing data.
    /// 
    /// NOTE: This bypasses the synthetic data generation and writes the provided
    /// chunk directly. Use generate_chunk() for synthetic data generation.
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            anyhow::bail!("Cannot write to finalized writer");
        }
        
        // Update checksum and counters
        self.hasher.update(chunk);
        self.bytes_generated += chunk.len() as u64;
        
        // Pass through to underlying writer
        self.writer.write_chunk(chunk).await
    }
    
    async fn write_owned_bytes(&mut self, data: Vec<u8>) -> Result<()> {
        // Update checksum and counters before passing ownership
        self.hasher.update(&data);
        self.bytes_generated += data.len() as u64;
        
        self.writer.write_owned_bytes(data).await
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;
        
        // Finalize the underlying writer
        self.writer.finalize().await
    }
    
    fn bytes_written(&self) -> u64 {
        self.writer.bytes_written()
    }
    
    fn compressed_bytes(&self) -> u64 {
        self.writer.compressed_bytes()
    }
    
    fn checksum(&self) -> Option<String> {
        // Return checksum of generated data (before compression)
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }
    
    fn compression(&self) -> CompressionConfig {
        self.writer.compression()
    }
    
    fn compression_ratio(&self) -> f64 {
        self.writer.compression_ratio()
    }
    
    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.writer.cancel().await
    }
}