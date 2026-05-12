// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Fast Parquet file generation using dgen_data streaming random bytes.
//!
//! One `RollingPool` is instantiated per call and reused across all row groups
//! and columns — never re-seeded, never re-created mid-generation.
//!
//! For column slices smaller than the 1 MiB backing block, `RollingPool`
//! returns zero-copy `Bytes` windows from the current block and refills
//! automatically (calling the Xoshiro256++ generator) only when the block is
//! exhausted.  This keeps throughput in the 5-15 GB/s range.

use anyhow::{Context, Result};
use bytes::Bytes;
use dgen_data::RollingPool;
use std::sync::{Arc, Mutex};

use crate::object_store::store_for_uri;

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};

/// Generate a complete Parquet file in memory as raw bytes.
///
/// Schema is `num_cols` Float32 columns named `col0`…`colN-1`.
/// All column data is incompressible random bytes sourced from a single
/// `RollingPool` — the pool is created once and reused for every column of
/// every row group.
///
/// The returned `Bytes` is a valid, self-contained Parquet file:
/// - UNCOMPRESSED pages
/// - No row-group or column statistics
/// - No dictionary encoding
///
/// # Example (Python)
/// ```python
/// import s3dlio
/// bv = s3dlio.generate_parquet_bytes(200, 8192, 123)
/// with open("/mnt/test/train/img_00_of_64.parquet", "wb") as f:
///     f.write(memoryview(bv))
/// ```
pub fn generate_parquet_bytes(
    num_cols: usize,
    rows_per_rg: usize,
    num_row_groups: usize,
) -> Result<Bytes> {
    // ── Schema ────────────────────────────────────────────────────────────────
    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("col{i}"), DataType::Float32, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    // ── Writer properties ─────────────────────────────────────────────────────
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_statistics_enabled(EnabledStatistics::None)
        .set_dictionary_enabled(false)
        .build();

    // ── Output buffer ─────────────────────────────────────────────────────────
    // Float32 = 4 bytes/value; add 1 MiB headroom for Parquet metadata/headers.
    let col_bytes = rows_per_rg * std::mem::size_of::<f32>();
    let estimated = col_bytes * num_cols * num_row_groups + 1024 * 1024;
    let mut buf: Vec<u8> = Vec::with_capacity(estimated);

    let mut writer = ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props))
        .context("ArrowWriter init")?;

    // ── ONE pool, reused for every column of every row group ──────────────────
    // Never re-seeded; the Xoshiro256++ cursor advances continuously.
    let mut pool = RollingPool::new(1, 1);

    for _ in 0..num_row_groups {
        let arrays: Vec<Arc<dyn Array>> = (0..num_cols)
            .map(|_| {
                // Zero-copy: pool hands back a bytes::Bytes window into the
                // current 1 MiB backing block.  Arrow's Buffer understands
                // bytes::Bytes natively — no allocation, no memcpy.
                let raw: Bytes = pool.next_slice(col_bytes);
                let buf: Buffer = raw.into();
                // Reinterpret the raw bytes as f32 values — all bit patterns
                // are valid storage; no CPU transformation required.
                let scalar_buf = arrow_buffer::ScalarBuffer::<f32>::new(buf, 0, rows_per_rg);
                Arc::new(Float32Array::new(scalar_buf, None)) as Arc<dyn Array>
            })
            .collect();

        let batch =
            RecordBatch::try_new(Arc::clone(&schema), arrays).context("RecordBatch::try_new")?;
        writer.write(&batch).context("ArrowWriter::write")?;
        // Force a row-group boundary after each batch.  Without flush() the
        // ArrowWriter buffers all batches into a single row group.
        writer.flush().context("ArrowWriter::flush")?;
    }

    writer.close().context("ArrowWriter::close")?;

    Ok(Bytes::from(buf))
}

/// Generate a complete Parquet file entirely in Rust and write it to any
/// supported URI in a single operation.
///
/// This is the high-level companion to [`generate_parquet_bytes`].  The entire
/// pipeline — data generation, Parquet serialization, and store write — runs
/// inside a single async task with the GIL released for the full duration.
/// No data ever crosses the Python/Rust boundary mid-file.
///
/// Supports any URI scheme recognised by `store_for_uri`:
/// `file://`, `direct://`, `s3://`, `az://`, `gs://`.
///
/// # Example (Python)
/// ```python
/// import s3dlio
/// # Generate and write one file — single GIL release, everything in Rust
/// s3dlio.generate_and_write_parquet("file:///mnt/test/train/img_00_of_64.parquet", 200, 8192, 123)
/// ```
pub async fn generate_and_write_parquet(
    uri: &str,
    num_cols: usize,
    rows_per_rg: usize,
    num_row_groups: usize,
) -> Result<()> {
    let bytes = generate_parquet_bytes(num_cols, rows_per_rg, num_row_groups)?;
    let store = store_for_uri(uri)?;
    store.put_multipart(uri, bytes, None).await.context("store put_multipart")?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema-flexible generation: named columns with per-column Float32 count
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Parquet file with an arbitrary named schema.
///
/// `columns` is a slice of `(name, num_float32_per_row)` pairs.
/// - `num_float32_per_row == 1` → plain `Float32` scalar column
/// - `num_float32_per_row > 1`  → `FixedSizeList<Float32>[N]` column (embedding)
///
/// All column data comes from the same `RollingPool` (Xoshiro256++ backed),
/// so no allocations occur mid-generation and no Python is involved.
///
/// # Example
/// ```python
/// import s3dlio
/// # Flux schema: t5_encodings[524328], clip_encodings[409], ...
/// bv = s3dlio.generate_parquet_bytes_schema(
///     [("t5_encodings", 524328), ("clip_encodings", 409),
///      ("mean", 8232), ("logvar", 8232), ("timestamp", 7)],
///     rows_per_rg=48, num_row_groups=6)
/// ```
pub fn generate_parquet_bytes_schema(
    columns: &[(String, usize)],
    rows_per_rg: usize,
    num_row_groups: usize,
) -> Result<Bytes> {
    // ── Schema ────────────────────────────────────────────────────────────────
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, col_size)| {
            if *col_size == 1 {
                Field::new(name, DataType::Float32, false)
            } else {
                Field::new(
                    name,
                    DataType::FixedSizeList(
                        Arc::new(Field::new("element", DataType::Float32, false)),
                        *col_size as i32,
                    ),
                    false,
                )
            }
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));

    // ── Writer properties ─────────────────────────────────────────────────────
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_statistics_enabled(EnabledStatistics::None)
        .set_dictionary_enabled(false)
        .build();

    // ── Output buffer ─────────────────────────────────────────────────────────
    let total_floats_per_rg: usize = columns.iter().map(|(_, s)| s * rows_per_rg).sum();
    let estimated = total_floats_per_rg * std::mem::size_of::<f32>() * num_row_groups + 1024 * 1024;
    let mut buf: Vec<u8> = Vec::with_capacity(estimated);

    let mut writer = ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props))
        .context("ArrowWriter init")?;

    // ── ONE pool, reused for every column of every row group ──────────────────
    let mut pool = RollingPool::new(1, 1);

    for _ in 0..num_row_groups {
        let arrays: Vec<Arc<dyn Array>> = columns
            .iter()
            .map(|(_, col_size)| -> Arc<dyn Array> {
                let n_floats = rows_per_rg * col_size;
                let raw: Bytes = pool.next_slice(n_floats * std::mem::size_of::<f32>());
                let arr_buf: Buffer = raw.into();

                if *col_size == 1 {
                    // Plain scalar Float32 column
                    let scalar_buf =
                        arrow_buffer::ScalarBuffer::<f32>::new(arr_buf, 0, rows_per_rg);
                    Arc::new(Float32Array::new(scalar_buf, None))
                } else {
                    // FixedSizeList<Float32>[col_size] embedding column
                    let scalar_buf =
                        arrow_buffer::ScalarBuffer::<f32>::new(arr_buf, 0, n_floats);
                    let values = Arc::new(Float32Array::new(scalar_buf, None));
                    let child_field =
                        Arc::new(Field::new("element", DataType::Float32, false));
                    Arc::new(FixedSizeListArray::new(
                        child_field,
                        *col_size as i32,
                        values,
                        None,
                    ))
                }
            })
            .collect();

        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
            .context("RecordBatch::try_new")?;
        writer.write(&batch).context("ArrowWriter::write")?;
        writer.flush().context("ArrowWriter::flush")?;
    }

    writer.close().context("ArrowWriter::close")?;
    Ok(Bytes::from(buf))
}

/// Generate a Parquet file with a named schema and write it to any URI.
///
/// Schema-flexible counterpart of [`generate_and_write_parquet`].
/// Entire pipeline (generation + serialization + write) runs off the GIL.
pub async fn generate_and_write_parquet_schema(
    uri: &str,
    columns: Vec<(String, usize)>,
    rows_per_rg: usize,
    num_row_groups: usize,
) -> Result<()> {
    let bytes = generate_parquet_bytes_schema(&columns, rows_per_rg, num_row_groups)?;
    let store = store_for_uri(uri)?;
    store.put_multipart(uri, bytes, None).await.context("store put_multipart")?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming producer/consumer generation: row groups pipelined into multipart
// ─────────────────────────────────────────────────────────────────────────────

/// `Write` impl backed by a shared `Vec<u8>`.
///
/// The `ArrowWriter` holds this and calls `write()` for every serialised byte.
/// After each `ArrowWriter::flush()` (= one completed row group), the caller
/// takes the accumulated bytes via `std::mem::take` on the locked Vec and
/// forwards them to the upload sink.  `flush()` is a no-op: data is already
/// in the Vec as soon as `write()` returns.
struct SharedBufWriter {
    inner: Arc<Mutex<Vec<u8>>>,
}

impl std::io::Write for SharedBufWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.inner
            .lock()
            .expect("SharedBufWriter: mutex poisoned")
            .extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(()) // data lands in the Vec on every write() call — nothing to do
    }
}

/// Generate a Parquet file with a named schema and stream it to any S3 URI,
/// pipelining data generation with multipart upload.
///
/// ## How it works
///
/// A bounded Tokio channel (capacity 2) connects two concurrent actors:
///
/// **Producer** (`spawn_blocking` thread — never blocks the async executor):
///   - Builds schema and `ArrowWriter` backed by a `SharedBufWriter`.
///   - Generates one row group at a time using `RollingPool` (Xoshiro256++).
///   - After each `ArrowWriter::flush()` the completed row-group bytes are
///     drained from the shared buffer and sent to the channel.
///   - After all row groups, `ArrowWriter::close()` writes the Parquet footer;
///     those bytes are sent as the final message.
///
/// **Consumer** (the calling async task):
///   - Opens a `MultipartUploadSink` immediately (before generation starts).
///   - Feeds each received chunk into the sink with `write()`.
///   - The sink's internal coordinator task uploads parts concurrently on the
///     Tokio runtime while the producer is generating the next row group.
///
/// ## Memory
///
/// At most two row-group-sized buffers are live simultaneously (one in the
/// channel, one being uploaded).  For Flux (48 rows × ~2.06 MiB/row = ~99 MiB
/// per row group), peak RSS ≈ 200 MiB instead of 595 MiB (full file).
///
/// ## Thread budget
///
/// One `spawn_blocking` slot is consumed per concurrent call.  Tokio's
/// blocking pool is bounded by `TOKIO_BLOCKING_THREADS` (default 512) and in
/// practice by the number of MPI ranks × files in flight.
pub async fn generate_and_write_parquet_schema_streaming(
    uri: &str,
    columns: Vec<(String, usize)>,
    rows_per_rg: usize,
    num_row_groups: usize,
) -> Result<()> {
    use crate::multipart::{MultipartUploadConfig, MultipartUploadSink};
    use crate::s3_utils::parse_s3_uri;
    use tokio::sync::mpsc;

    // Channel capacity 2: producer can be one row group ahead of consumer.
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(2);

    let columns_owned = columns;

    // ── Producer: sync generation in a Tokio blocking thread ─────────────────
    let producer = tokio::task::spawn_blocking(move || -> Result<()> {
        // Build schema (identical to generate_parquet_bytes_schema).
        let fields: Vec<Field> = columns_owned
            .iter()
            .map(|(name, col_size)| {
                if *col_size == 1 {
                    Field::new(name, DataType::Float32, false)
                } else {
                    Field::new(
                        name,
                        DataType::FixedSizeList(
                            Arc::new(Field::new("element", DataType::Float32, false)),
                            *col_size as i32,
                        ),
                        false,
                    )
                }
            })
            .collect();
        let schema = Arc::new(Schema::new(fields));

        let props = WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_statistics_enabled(EnabledStatistics::None)
            .set_dictionary_enabled(false)
            .build();

        // Shared buffer: ArrowWriter writes here; we drain it per row group.
        let shared: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let writer_buf = SharedBufWriter { inner: Arc::clone(&shared) };
        let mut writer =
            ArrowWriter::try_new(writer_buf, Arc::clone(&schema), Some(props))
                .context("ArrowWriter init")?;

        // ONE pool reused across all row groups and columns — no re-seeding.
        let mut pool = RollingPool::new(1, 1);

        for _ in 0..num_row_groups {
            let arrays: Vec<Arc<dyn Array>> = columns_owned
                .iter()
                .map(|(_, col_size)| -> Arc<dyn Array> {
                    let n_floats = rows_per_rg * col_size;
                    let raw: Bytes =
                        pool.next_slice(n_floats * std::mem::size_of::<f32>());
                    let arr_buf: Buffer = raw.into();
                    if *col_size == 1 {
                        let scalar_buf =
                            arrow_buffer::ScalarBuffer::<f32>::new(arr_buf, 0, rows_per_rg);
                        Arc::new(Float32Array::new(scalar_buf, None))
                    } else {
                        let scalar_buf =
                            arrow_buffer::ScalarBuffer::<f32>::new(arr_buf, 0, n_floats);
                        let values = Arc::new(Float32Array::new(scalar_buf, None));
                        let child_field =
                            Arc::new(Field::new("element", DataType::Float32, false));
                        Arc::new(FixedSizeListArray::new(
                            child_field,
                            *col_size as i32,
                            values,
                            None,
                        ))
                    }
                })
                .collect();

            let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
                .context("RecordBatch::try_new")?;
            writer.write(&batch).context("ArrowWriter::write")?;
            // flush() closes the current row group and writes all column bytes
            // to SharedBufWriter.  After this call, shared contains exactly the
            // serialised bytes for this row group.
            writer.flush().context("ArrowWriter::flush")?;

            let chunk = std::mem::take(&mut *shared.lock().unwrap());
            if !chunk.is_empty() {
                tx.blocking_send(chunk)
                    .map_err(|_| anyhow::anyhow!("upload consumer dropped before all row groups"))?;
            }
        }

        // close() writes the Parquet footer + file metadata.
        writer.close().context("ArrowWriter::close")?;
        let footer = std::mem::take(&mut *shared.lock().unwrap());
        if !footer.is_empty() {
            tx.blocking_send(footer)
                .map_err(|_| anyhow::anyhow!("upload consumer dropped before footer"))?;
        }
        // Dropping tx signals the consumer that all data has been sent.
        Ok(())
    });

    // ── Consumer: async upload of received chunks ─────────────────────────────
    // Open the multipart upload before the first chunk arrives so that
    // CreateMultipartUpload is in flight while the producer generates RG 0.
    let (bucket, key) = parse_s3_uri(uri)?;
    let mut sink =
        MultipartUploadSink::new_async(&bucket, &key, MultipartUploadConfig::default())
            .await
            .context("CreateMultipartUpload failed")?;

    while let Some(chunk) = rx.recv().await {
        // Vec<u8> → Bytes::from() is zero-copy inside write().
        sink.write(chunk).await.context("multipart write chunk failed")?;
    }

    // Propagate any generation error from the producer.
    producer
        .await
        .context("producer task panicked")?
        .context("row group generation failed")?;

    // Flush tail + CompleteMultipartUpload.
    sink.finish().await.context("CompleteMultipartUpload failed")?;

    Ok(())
}
