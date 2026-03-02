// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::Result;
use crate::model::{Object, ReadObjectRequest};
use crate::model_ext::WriteObjectRequest;
use crate::read_object::ReadObjectResponse;
use crate::storage::client::StorageInner;
use crate::storage::perform_upload::PerformUpload;
use crate::storage::read_object::Reader;
use crate::storage::request_options::RequestOptions;
use crate::storage::streaming_source::{Seek, StreamingSource};
use crate::{
    model_ext::OpenObjectRequest, object_descriptor::ObjectDescriptor,
    storage::bidi::connector::Connector, storage::bidi::transport::ObjectDescriptorTransport,
};
use std::sync::Arc;
use gaxi::gcs_constants::{DEFAULT_GRPC_WRITE_CHUNK_SIZE, ENV_GRPC_WRITE_CHUNK_SIZE, MAX_GRPC_WRITE_CHUNK_SIZE};

/// Returns the effective gRPC write chunk size.
///
/// Priority:
///   1. `S3DLIO_GRPC_WRITE_CHUNK_SIZE` env var (bytes) — silently clamped to
///      [`MAX_GRPC_WRITE_CHUNK_SIZE`] if the provided value exceeds the server limit.
///   2. [`DEFAULT_GRPC_WRITE_CHUNK_SIZE`] (= [`MAX_GRPC_WRITE_CHUNK_SIZE`] = 4 MiB).
///
/// Both constants are defined in
/// `vendor/google-cloud-gax-internal/src/gcs_constants.rs` — the single
/// source of truth for all GCS/gRPC tuning values.
fn grpc_write_chunk_size() -> usize {
    let requested = std::env::var(ENV_GRPC_WRITE_CHUNK_SIZE)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_GRPC_WRITE_CHUNK_SIZE);
    requested.min(MAX_GRPC_WRITE_CHUNK_SIZE)
}

/// An implementation of [`stub::Storage`][crate::storage::stub::Storage] that
/// interacts with the Cloud Storage service.
///
/// This is the default implementation of a
/// [`client::Storage<T>`][crate::storage::client::Storage].
///
/// ## Example
///
/// ```
/// # async fn sample() -> anyhow::Result<()> {
/// use google_cloud_storage::client::Storage;
/// use google_cloud_storage::stub::DefaultStorage;
/// let client: Storage<DefaultStorage> = Storage::builder().build().await?;
/// # Ok(()) }
/// ```
#[derive(Clone, Debug)]
pub struct Storage {
    inner: Arc<StorageInner>,
}

impl Storage {
    pub(crate) fn new(inner: Arc<StorageInner>) -> Arc<Self> {
        Arc::new(Self { inner })
    }
}

impl super::stub::Storage for Storage {
    /// Implements [crate::client::Storage::read_object].
    async fn read_object(
        &self,
        req: ReadObjectRequest,
        options: RequestOptions,
    ) -> Result<ReadObjectResponse> {
        let reader = Reader {
            inner: self.inner.clone(),
            request: req,
            options,
        };
        reader.response().await
    }

    /// Implements [crate::client::Storage::write_object].
    async fn write_object_buffered<P>(
        &self,
        payload: P,
        req: WriteObjectRequest,
        options: RequestOptions,
    ) -> Result<Object>
    where
        P: StreamingSource + Send + Sync + 'static,
    {
        PerformUpload::new(payload, self.inner.clone(), req.spec, req.params, options)
            .send()
            .await
    }

    /// Implements [crate::client::Storage::write_object].
    async fn write_object_unbuffered<P>(
        &self,
        payload: P,
        req: WriteObjectRequest,
        options: RequestOptions,
    ) -> Result<Object>
    where
        P: StreamingSource + Seek + Send + Sync + 'static,
    {
        PerformUpload::new(payload, self.inner.clone(), req.spec, req.params, options)
            .send_unbuffered()
            .await
    }

    /// Implements gRPC BidiWriteObject for RAPID/zonal buckets.
    ///
    /// This creates a BidiWriteObject gRPC stream, sends the data in chunks
    /// (default 2 MiB), with a WriteObjectSpec on the first message and
    /// finish_write=true + object-level checksums on the last message.
    async fn write_object_grpc(
        &self,
        data: bytes::Bytes,
        req: WriteObjectRequest,
        options: RequestOptions,
    ) -> Result<Object> {
        use crate::google::storage::v2::{
            bidi_write_object_request, bidi_write_object_response, BidiWriteObjectRequest,
            BidiWriteObjectResponse, ChecksummedData, ObjectChecksums,
            WriteObjectSpec as ProtoWriteObjectSpec,
        };
        use crate::storage::info::X_GOOG_API_CLIENT_HEADER;
        use crate::Error;
        use gaxi::grpc::tonic::{Extensions, GrpcMethod, Streaming};
        use gaxi::prost::ToProto;

        // Convert model Object to proto Object for the WriteObjectSpec
        let resource = req
            .spec
            .resource
            .as_ref()
            .expect("resource field must be set");
        let bucket_name = resource.bucket.clone();
        let proto_resource = req
            .spec
            .resource
            .clone()
            .map(|r| r.to_proto())
            .transpose()
            .map_err(|e| Error::io(format!("failed to convert Object to proto: {e}")))?;

        // Build the proto WriteObjectSpec
        let proto_spec = ProtoWriteObjectSpec {
            resource: proto_resource,
            predefined_acl: req.spec.predefined_acl.clone(),
            if_generation_match: req.spec.if_generation_match,
            if_generation_not_match: req.spec.if_generation_not_match,
            if_metageneration_match: req.spec.if_metageneration_match,
            if_metageneration_not_match: req.spec.if_metageneration_not_match,
            object_size: Some(data.len() as i64),
            appendable: req.spec.appendable,
        };

        // Calculate whole-object CRC32C
        let object_crc32c = crc32c::crc32c(&data);

        // Determine chunk size
        let chunk_size = grpc_write_chunk_size();
        let total_len = data.len();
        let num_chunks_est = if total_len == 0 { 1 } else { (total_len + chunk_size - 1) / chunk_size };
        tracing::trace!(
            "BidiWriteObject: total_size={} bytes, chunk_size={} bytes ({:.1} MiB), estimated_chunks={}, appendable={:?}, object_crc32c={:#010x}",
            total_len, chunk_size, chunk_size as f64 / (1024.0 * 1024.0), num_chunks_est, req.spec.appendable, object_crc32c
        );

        // Build x-goog-request-params routing header
        let x_goog_request_params = format!("bucket={bucket_name}");

        // Create the gRPC extensions
        let extensions = {
            let mut e = Extensions::new();
            e.insert(GrpcMethod::new(
                "google.storage.v2.Storage",
                "BidiWriteObject",
            ));
            e
        };
        let path = http::uri::PathAndQuery::from_static(
            "/google.storage.v2.Storage/BidiWriteObject",
        );

        // Use a small bounded channel so the producer and the gRPC network
        // sender run concurrently.  The previous approach used an unbounded
        // channel (sized to hold ALL chunks), which serialised CRC computation
        // and gRPC stream setup: all N chunks were queued before the first byte
        // was sent on the wire.  With a capacity of 8 chunks the producer
        // stays at most 8 chunks ahead of the network consumer, keeping both
        // the CPU (CRC) and the network (gRPC) busy simultaneously.
        // For a single-chunk object the channel never blocks, so there is no
        // overhead when total_len <= chunk_size.
        const PRODUCER_CHANNEL_CAPACITY: usize = 8;
        let (tx, rx) =
            tokio::sync::mpsc::channel::<BidiWriteObjectRequest>(PRODUCER_CHANNEL_CAPACITY);

        // Spawn the producer as a separate task so it runs concurrently with
        // the gRPC sender below.  `data` is `bytes::Bytes` (Arc-backed), so
        // the clone is effectively free.
        let producer_task = {
            let data = data.clone();
            let proto_spec = proto_spec.clone();
            tokio::spawn(async move {
                let mut offset: usize = 0;
                let mut msg_index: usize = 0;
                while offset < total_len || (total_len == 0 && msg_index == 0) {
                    let end = std::cmp::min(offset + chunk_size, total_len);
                    // `data.slice()` is zero-copy: it increments the Arc refcount.
                    let chunk = data.slice(offset..end);
                    let chunk_crc = crc32c::crc32c(&chunk);
                    let is_first = msg_index == 0;
                    let is_last = end >= total_len;

                    let request = BidiWriteObjectRequest {
                        write_offset: offset as i64,
                        object_checksums: if is_last {
                            Some(ObjectChecksums {
                                crc32c: Some(object_crc32c),
                                md5_hash: bytes::Bytes::new(),
                            })
                        } else {
                            None
                        },
                        state_lookup: false,
                        flush: is_last,
                        finish_write: is_last,
                        common_object_request_params: None,
                        first_message: if is_first {
                            Some(
                                bidi_write_object_request::FirstMessage::WriteObjectSpec(
                                    proto_spec.clone(),
                                ),
                            )
                        } else {
                            None
                        },
                        data: Some(bidi_write_object_request::Data::ChecksummedData(
                            ChecksummedData {
                                content: chunk,
                                crc32c: Some(chunk_crc),
                            },
                        )),
                    };

                    tracing::trace!(
                        "BidiWriteObject: chunk {}/{} offset={} len={} crc32c={:#010x} first={} last={}",
                        msg_index + 1, num_chunks_est, offset, end - offset, chunk_crc, is_first, is_last
                    );
                    if tx.send(request).await.is_err() {
                        // Stream was dropped (e.g. gRPC error); abort producer.
                        break;
                    }
                    offset = end;
                    msg_index += 1;
                }
                tracing::trace!("BidiWriteObject producer: sent {} chunk(s)", msg_index);
                // Dropping `tx` signals end-of-stream to the ReceiverStream.
            })
        };

        // Start the bidi stream immediately — concurrently with the producer.
        let request_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let response: std::result::Result<
            gaxi::grpc::tonic::Result<gaxi::grpc::tonic::Response<Streaming<BidiWriteObjectResponse>>>,
            crate::Error,
        > = self
            .inner
            .grpc
            .bidi_stream_with_status(
                extensions,
                path,
                request_stream,
                options.gax(),
                &X_GOOG_API_CLIENT_HEADER,
                &x_goog_request_params,
            )
            .await;

        // `request_stream` (which held `rx`) was consumed by `bidi_stream_with_status`.
        // With `rx` dropped the producer task's next `tx.send()` returns `Err` and
        // the loop exits.  Abort + await ensures no task leak even on error paths.
        producer_task.abort();
        let _ = producer_task.await; // JoinError on abort is expected — ignore

        let tonic_result = response?;
        let tonic_response = tonic_result.map_err(|status| {
            gaxi::grpc::from_status::to_gax_error(status)
        })?;

        let (_, mut stream, _) = tonic_response.into_parts();

        // Read BidiWriteObjectResponse messages until we get one with a Resource
        loop {
            match stream.message().await {
                Ok(Some(msg)) => {
                    if let Some(bidi_write_object_response::WriteStatus::Resource(
                        proto_obj,
                    )) = msg.write_status
                    {
                        // Convert proto Object back to model Object
                        use gaxi::prost::FromProto;
                        let model_obj: crate::model::Object =
                            proto_obj.cnv().map_err(|e| {
                                Error::io(format!(
                                    "failed to convert proto Object to model: {e}"
                                ))
                            })?;
                        return Ok(model_obj);
                    }
                    // PersistedSize status — continue reading
                }
                Ok(None) => {
                    return Err(Error::io(
                        "BidiWriteObject stream ended without returning Object",
                    ));
                }
                Err(status) => {
                    return Err(gaxi::grpc::from_status::to_gax_error(status));
                }
            }
        }
    }

    async fn open_object(
        &self,
        request: OpenObjectRequest,
        options: RequestOptions,
    ) -> Result<(ObjectDescriptor, Vec<ReadObjectResponse>)> {
        let (spec, ranges) = request.into_parts();
        let connector = Connector::new(spec, options, self.inner.grpc.clone());
        let (transport, readers) = ObjectDescriptorTransport::new(connector, ranges).await?;

        Ok((ObjectDescriptor::new(transport), readers))
    }
}
