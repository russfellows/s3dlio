#!/bin/bash
uv pip uninstall dlio-s3-rust
#uv pip install --force-reinstall ./target/wheels/dlio_s3_rust-*-cp313-cp313-manylinux_2_39_x86_64.whl 
uv pip install --force-reinstall ./target/wheels/dlio_s3_rust-*-cp312-cp312-manylinux_2_39_x86_64.whl 

