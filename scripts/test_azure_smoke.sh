# !/bin/bash
# one-time: set env vars in your shell
export AZURE_BLOB_ACCOUNT="egiazurestore1"
export AZURE_BLOB_CONTAINER="s3dlio"
# export AZURE_BLOB_ACCOUNT_URL="https://mystorageacct.blob.core.windows.net" # optional

# run only these integration tests
cargo test --release --features azure --test azure_blob_smoke -- --nocapture

