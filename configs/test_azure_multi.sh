# Self-contained multi-blob roundtrip:
cargo test --release --features azure --test azure_blob_multi -- --nocapture

# Sequence part 1 (upload and show):
cargo test --release --features azure --test azure_blob_sequence -- --nocapture sequence_1_upload_and_list

# Sequence part 2 (cleanup and show):
cargo test --release --features azure --test azure_blob_sequence -- --nocapture sequence_2_cleanup_and_verify

