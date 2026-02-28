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

/// Default gRPC write chunk size: 2 MiB.
/// This aligns with GCS server expectations for BidiWriteObject.
/// Can be overridden via the `S3DLIO_GRPC_WRITE_CHUNK_SIZE` environment variable
/// (value in bytes).
pub const DEFAULT_GRPC_WRITE_CHUNK_SIZE: usize = 2 * 1024 * 1024;

/// Returns the configured gRPC write chunk size.
/// Checks `S3DLIO_GRPC_WRITE_CHUNK_SIZE` env var, falls back to DEFAULT_GRPC_WRITE_CHUNK_SIZE.
fn grpc_write_chunk_size() -> usize {
    std::env::var("S3DLIO_GRPC_WRITE_CHUNK_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_GRPC_WRITE_CHUNK_SIZE)
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

        // Create the channel for sending requests
        // Buffer enough for all chunks + 1 to avoid blocking
        let num_chunks = (total_len + chunk_size - 1) / chunk_size;
        let channel_size = std::cmp::max(num_chunks + 1, 2);
        let (tx, rx) =
            tokio::sync::mpsc::channel::<BidiWriteObjectRequest>(channel_size);

        // Build and send all messages
        let mut offset: usize = 0;
        let mut msg_index: usize = 0;

        while offset < total_len || (total_len == 0 && msg_index == 0) {
            let end = std::cmp::min(offset + chunk_size, total_len);
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

            tx.send(request).await.map_err(Error::io)?;
            offset = end;
            msg_index += 1;
        }
        // Drop sender so the server sees end-of-stream after all messages
        drop(tx);

        // Start the bidi stream
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
