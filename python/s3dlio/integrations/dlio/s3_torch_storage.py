"""
Drop-in replacement for DLIO's s3_torch_storage.py using s3dlio

This file replaces dlio_benchmark/storage/s3_torch_storage.py
to use s3dlio instead of s3torchconnector.

Installation:
    1. pip install s3dlio  (or install from wheel)
    2. Copy this file to: dlio_benchmark/storage/s3_torch_storage.py
    3. Run DLIO benchmark as normal

Licensed under Apache 2.0

Compatible with DLIO Benchmark v1.0+ (after PR #307)
"""
import os
from urllib.parse import urlparse

import s3dlio

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.storage.s3_storage import S3Storage
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)


class S3PyTorchConnectorStorage(S3Storage):
    """
    Storage backend using s3dlio for high-performance multi-protocol I/O.
    
    This is a drop-in replacement for the original S3PyTorchConnectorStorage
    that uses s3torchconnector. It provides the same interface but uses
    s3dlio's multi-backend support (S3, Azure, GCS, file://, direct://).
    
    Environment Variables (for S3):
        AWS_ACCESS_KEY_ID: S3 access key
        AWS_SECRET_ACCESS_KEY: S3 secret key
        AWS_REGION: S3 region (default: us-east-1)
        AWS_ENDPOINT_URL: Custom endpoint for MinIO, Ceph, etc.
    
    Environment Variables (for Azure):
        AZURE_STORAGE_ACCOUNT_NAME: Azure account name
        AZURE_STORAGE_ACCOUNT_KEY: Azure account key
        AZURE_STORAGE_ENDPOINT: Custom endpoint (optional)
    
    Environment Variables (for GCS):
        GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
        GCS_ENDPOINT_URL: Custom endpoint (optional)
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)
        
        # Get storage options from config (same as original s3_torch_storage.py)
        storage_options = getattr(self._args, "storage_options", {}) or {}
        
        # Set environment variables from config if provided
        # This maintains compatibility with DLIO's YAML config format
        if storage_options.get("access_key_id"):
            os.environ.setdefault("AWS_ACCESS_KEY_ID", storage_options["access_key_id"])
        if storage_options.get("secret_access_key"):
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", storage_options["secret_access_key"])
        if storage_options.get("region"):
            os.environ.setdefault("AWS_REGION", storage_options["region"])
        if storage_options.get("endpoint_url"):
            os.environ.setdefault("AWS_ENDPOINT_URL", storage_options["endpoint_url"])

    @dlp.log
    def get_uri(self, id):
        """Return the id as-is (full URI expected)."""
        return id

    @dlp.log
    def create_namespace(self, exist_ok=False):
        """Namespace creation - buckets typically pre-exist."""
        return True

    @dlp.log
    def get_namespace(self):
        return self.get_node(self.namespace.name)

    @dlp.log
    def create_node(self, id, exist_ok=False):
        """Create directory node using s3dlio.mkdir for all backends."""
        uri = self.get_uri(id)
        try:
            s3dlio.mkdir(uri)
            return True
        except Exception as e:
            if not exist_ok:
                raise
            return True

    @dlp.log
    def get_node(self, id=""):
        """Get node type (FILE, DIRECTORY, or None)."""
        uri = self.get_uri(id)
        
        # Check if it's a file
        if hasattr(s3dlio, 'exists'):
            if s3dlio.exists(uri):
                return MetadataType.FILE
        else:
            # Fallback for older s3dlio versions
            try:
                metadata = s3dlio.stat(uri)
                if metadata and 'size' in metadata:
                    return MetadataType.FILE
            except Exception:
                pass
        
        # Check if it's a "directory" by listing children
        try:
            check_uri = uri if uri.endswith('/') else uri + '/'
            children = s3dlio.list(check_uri)
            if children:
                return MetadataType.DIRECTORY
        except Exception:
            pass
        
        return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        """
        List objects under a path. Returns relative filenames.
        
        This matches the original s3_torch_storage.py behavior where
        walk_node returns just the filenames, not full URIs.
        """
        # Parse s3://bucket/prefix path
        parsed = urlparse(id)
        if parsed.scheme not in ('s3', 'gs', 'az', 'azure', 'file', 'direct'):
            # Assume S3 if no scheme
            id = f"s3://{id}"
            parsed = urlparse(id)
        
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
        
        # Ensure prefix ends with / for directory listing
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        try:
            # Build full URI for listing
            full_uri = f"{parsed.scheme}://{bucket}/{prefix}"
            
            # s3dlio.list returns full URIs
            full_uris = s3dlio.list(full_uri)
            
            # Convert to relative paths (just filenames)
            # This matches the original s3_torch_storage.py behavior
            paths = []
            for uri in full_uris:
                # Extract key from full URI
                uri_parsed = urlparse(uri)
                key = uri_parsed.path.lstrip('/')
                
                # Strip the prefix to get relative path
                if key.startswith(prefix):
                    relative = key[len(prefix):]
                else:
                    relative = os.path.basename(key)
                
                if relative:  # Skip empty strings
                    paths.append(relative)
            
            return paths
            
        except Exception as e:
            print(f"Error listing {id}: {e}")
            return []

    @dlp.log
    def delete_node(self, id):
        """Delete an object."""
        uri = self.get_uri(id)
        try:
            s3dlio.delete(uri)
            return True
        except Exception:
            return False

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        """
        Write data to storage.
        
        Args:
            id: Full URI (s3://bucket/key)
            data: bytes or BytesIO object
            offset: Not supported (full object write only)
            length: Not supported (full object write only)
        """
        # Handle BytesIO objects (from numpy.save, etc.)
        if hasattr(data, 'getvalue'):
            content = data.getvalue()
        elif hasattr(data, 'read'):
            # Seek to beginning if possible
            if hasattr(data, 'seek'):
                data.seek(0)
            content = data.read()
        else:
            content = data
        
        try:
            s3dlio.put_bytes(id, content)
            return None
        except Exception as e:
            print(f"Error writing to {id}: {e}")
            raise

    @dlp.log
    def get_data(self, id, data=None, offset=None, length=None):
        """
        Read data from storage.
        
        Args:
            id: Full URI (s3://bucket/key)
            data: Ignored (buffer not needed with s3dlio)
            offset: Start byte offset (optional)
            length: Number of bytes to read (optional)
        """
        try:
            if offset is not None and length is not None:
                return s3dlio.get_range(id, offset=offset, length=length)
            else:
                return s3dlio.get(id)
        except Exception as e:
            print(f"Error reading from {id}: {e}")
            raise

    @dlp.log
    def isfile(self, id):
        """Check if path is a file (object exists)."""
        uri = self.get_uri(id)
        if hasattr(s3dlio, 'exists'):
            return s3dlio.exists(uri)
        # Fallback for older s3dlio versions
        try:
            metadata = s3dlio.stat(uri)
            return metadata is not None and 'size' in metadata
        except Exception:
            return False

    def get_basename(self, id):
        """Get filename from path."""
        return os.path.basename(id)
