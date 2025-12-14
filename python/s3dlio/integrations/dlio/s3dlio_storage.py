"""
s3dlio storage backend for DLIO Benchmark

This provides a new storage_type 's3dlio' for DLIO, using the s3dlio library
for high-performance multi-protocol I/O (S3, Azure, GCS, file://, direct://).

Installation:
    1. pip install s3dlio
    2. Copy this file to: dlio_benchmark/storage/s3dlio_storage.py
    3. Apply the patch to register the new storage type (see README)
    4. Use storage_type: s3dlio in your DLIO config

Licensed under Apache 2.0
Compatible with DLIO Benchmark v1.0+
"""
import os
from urllib.parse import urlparse

import s3dlio

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)


class S3dlioStorage(DataStorage):
    """
    Storage backend using s3dlio for high-performance multi-protocol I/O.
    
    Unlike S3PyTorchConnectorStorage which only supports S3, this backend
    supports multiple storage protocols via s3dlio:
    
    - s3://   - Amazon S3, MinIO, Ceph, S3-compatible stores
    - az://   - Azure Blob Storage
    - gs://   - Google Cloud Storage
    - file:// - Local filesystem (POSIX)
    - direct:// - Direct I/O filesystem (O_DIRECT)
    
    Configuration (DLIO YAML):
        storage:
          storage_type: s3dlio
          storage_root: s3://bucket/prefix  # or az://, gs://, file://
    
    Environment Variables (for S3):
        AWS_ACCESS_KEY_ID: S3 access key
        AWS_SECRET_ACCESS_KEY: S3 secret key  
        AWS_REGION: S3 region (default: us-east-1)
        AWS_ENDPOINT_URL: Custom endpoint for MinIO, Ceph, etc.
    
    Environment Variables (for Azure):
        AZURE_STORAGE_ACCOUNT_NAME: Azure account name
        AZURE_STORAGE_ACCOUNT_KEY: Azure account key
    
    Environment Variables (for GCS):
        GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)
        self.prefix = namespace
        
        # Detect backend from URI scheme
        parsed = urlparse(namespace)
        self.scheme = parsed.scheme or 's3'
        self.bucket = parsed.netloc
        self.base_path = parsed.path.lstrip('/')
        
        # Get storage options from config if available
        storage_options = getattr(self._args, "storage_options", {}) or {}
        
        # Set environment variables from config if provided
        if storage_options.get("access_key_id"):
            os.environ.setdefault("AWS_ACCESS_KEY_ID", storage_options["access_key_id"])
        if storage_options.get("secret_access_key"):
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", storage_options["secret_access_key"])
        if storage_options.get("region"):
            os.environ.setdefault("AWS_REGION", storage_options["region"])
        if storage_options.get("endpoint_url"):
            os.environ.setdefault("AWS_ENDPOINT_URL", storage_options["endpoint_url"])

    def _make_uri(self, path: str) -> str:
        """Convert a relative path to a full URI."""
        if path.startswith(('s3://', 'az://', 'gs://', 'file://', 'direct://')):
            return path
        # Combine with prefix
        prefix = self.prefix.rstrip('/')
        path = path.lstrip('/')
        if path:
            return f"{prefix}/{path}"
        return prefix

    @dlp.log
    def get_uri(self, id):
        """Return the id as a full URI."""
        return self._make_uri(id)

    @dlp.log
    def create_namespace(self, exist_ok=False):
        """Namespace creation - buckets/containers typically pre-exist."""
        return True

    @dlp.log
    def get_namespace(self):
        return self.get_node(self.namespace.name)

    @dlp.log
    def create_node(self, id, exist_ok=False):
        """Create directory node using s3dlio.mkdir."""
        uri = self._make_uri(id)
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
        uri = self._make_uri(id)
        
        # Check if it's a file
        if s3dlio.exists(uri):
            return MetadataType.FILE
        
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
        """
        uri = self._make_uri(id)
        
        # Ensure ends with / for directory listing
        if not uri.endswith('/'):
            uri += '/'
        
        try:
            # s3dlio.list returns full URIs
            full_uris = s3dlio.list(uri)
            
            # Convert to relative paths (just filenames)
            paths = []
            prefix = uri
            for full_uri in full_uris:
                # Extract relative path
                if full_uri.startswith(prefix):
                    relative = full_uri[len(prefix):]
                else:
                    relative = os.path.basename(urlparse(full_uri).path)
                
                if relative:
                    paths.append(relative)
            
            return paths
            
        except Exception as e:
            print(f"[s3dlio] Error listing {uri}: {e}")
            return []

    @dlp.log
    def delete_node(self, id):
        """Delete an object."""
        uri = self._make_uri(id)
        try:
            s3dlio.delete(uri)
            return True
        except Exception:
            return False

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        """
        Write data to storage using s3dlio.put_bytes.
        
        Args:
            id: Path or full URI
            data: bytes or BytesIO object
            offset: Not supported (full object write only)
            length: Not supported (full object write only)
        """
        uri = self._make_uri(id)
        
        # Handle BytesIO objects (from numpy.save, etc.)
        if hasattr(data, 'getvalue'):
            content = data.getvalue()
        elif hasattr(data, 'read'):
            if hasattr(data, 'seek'):
                data.seek(0)
            content = data.read()
        else:
            content = data
        
        try:
            s3dlio.put_bytes(uri, content)
            return None
        except Exception as e:
            print(f"[s3dlio] Error writing to {uri}: {e}")
            raise

    @dlp.log
    def get_data(self, id, data=None, offset=None, length=None):
        """
        Read data from storage using s3dlio.get or s3dlio.get_range.
        
        Args:
            id: Path or full URI
            data: Ignored (buffer not needed with s3dlio)
            offset: Start byte offset (optional)
            length: Number of bytes to read (optional)
        """
        uri = self._make_uri(id)
        
        try:
            if offset is not None and length is not None:
                return bytes(s3dlio.get_range(uri, offset=offset, length=length))
            else:
                return bytes(s3dlio.get(uri))
        except Exception as e:
            print(f"[s3dlio] Error reading from {uri}: {e}")
            raise

    @dlp.log
    def isfile(self, id):
        """Check if path is a file (object exists)."""
        uri = self._make_uri(id)
        return s3dlio.exists(uri)

    def get_basename(self, id):
        """Get filename from path."""
        return os.path.basename(id)
