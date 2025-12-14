"""
dlio-s3dlio-storage: s3dlio storage backend for DLIO Benchmark

This package provides a high-performance storage backend for the 
Argonne DLIO Benchmark using s3dlio's multi-protocol capabilities.
"""

from .s3dlio_storage import S3DLIOStorage

__version__ = "0.1.0"
__all__ = ["S3DLIOStorage"]
