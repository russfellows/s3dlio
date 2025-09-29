#!/bin/bash
uv pip uninstall s3dlio 
uv pip install --force-reinstall ./target/wheels/s3dlio-*.whl

