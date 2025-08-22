#!/bin/bash
# one-time: set env vars in your shell
export AZURE_BLOB_ACCOUNT="egiazurestore1"
export AZURE_BLOB_CONTAINER="s3dlio"

# Self-contained multi-blob roundtrip:
cargo test --release --features azure --test azure_blob_multi -- --nocapture

# Sequence part 1 (upload and show):
cargo test --release --features azure --test azure_blob_sequence -- --nocapture sequence_1_upload_and_list

# Sequence part 2 (cleanup and show):
cargo test --release --features azure --test azure_blob_sequence -- --nocapture sequence_2_cleanup_and_verify

