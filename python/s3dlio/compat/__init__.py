"""
s3dlio compatibility modules for third-party libraries.

This package provides API-compatible wrappers for popular storage libraries,
allowing users to easily switch between implementations for testing and comparison.
"""

from .s3torchconnector import S3IterableDataset, S3MapDataset, S3Checkpoint

__all__ = ["S3IterableDataset", "S3MapDataset", "S3Checkpoint"]
