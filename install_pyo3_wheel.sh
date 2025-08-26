#!/bin/bash
uv pip uninstall s3dlio 
uv pip install --force-reinstall ./target/wheels/s3dlio-*7*-cp312-cp312-manylinux_2_39_x86_64.whl

