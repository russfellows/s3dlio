# !/bin/bash
# one-time: set env vars in your shell
export AZURE_BLOB_ACCOUNT="${AZURE_BLOB_ACCOUNT:?set AZURE_BLOB_ACCOUNT to your Azure storage account name}"
export AZURE_BLOB_CONTAINER="${S3DLIO_TEST_AZURE_CONTAINER:?set S3DLIO_TEST_AZURE_CONTAINER to an Azure container you own (no default)}"
# export AZURE_BLOB_ACCOUNT_URL="https://mystorageacct.blob.core.windows.net" # optional

# run only these integration tests
cargo test --release --features azure --test azure_blob_smoke -- --nocapture

