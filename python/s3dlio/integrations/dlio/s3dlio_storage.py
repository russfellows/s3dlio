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

import logging
import os
from urllib.parse import urlparse

import s3dlio

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
from dlio_benchmark.utils.utility import Profile

from . import _multipart_config

dlp = Profile(MODULE_STORAGE)
_logger = logging.getLogger(__name__)


def _rank_from_env():
    """Return the current process's rank from the first available of
    OMPI_COMM_WORLD_RANK / SLURM_PROCID / PMI_RANK -- the same env var
    set already read by _select_endpoint_via_mpi -- or None if none are
    set."""
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    elif "PMI_RANK" in os.environ:
        return int(os.environ["PMI_RANK"])
    return None


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

          # Optional: Multiple endpoints for load balancing
          endpoint_uris:
            - http://endpoint1:9000
            - http://endpoint2:9000
            - http://endpoint3:9000
          load_balance_strategy: round_robin  # or random

          # Optional: MPI-based endpoint distribution (overrides load_balance_strategy)
          use_mpi_endpoint_distribution: true  # Uses MPI rank to select endpoint

          storage_options:
            access_key_id: your-key
            secret_access_key: your-secret
            region: us-east-1

    Multi-Endpoint Support:
        Two approaches available:

        1. s3dlio Native Load Balancing:
           - Set endpoint_uris list + load_balance_strategy
           - Strategies: round_robin (default), random
           - Each process selects endpoint based on PID

        2. MPI-Based Distribution (Recommended for HPC):
           - Set endpoint_uris + use_mpi_endpoint_distribution: true
           - Uses OMPI_COMM_WORLD_RANK to assign endpoints deterministically
           - Falls back to SLURM_PROCID, PMI_RANK if OpenMPI not available
           - Example: 4 endpoints, 16 ranks → 4 ranks per endpoint
           - Optimal for NUMA-aware, node-aware endpoint assignment

    Environment Variables (for S3):
        AWS_ACCESS_KEY_ID: S3 access key
        AWS_SECRET_ACCESS_KEY: S3 secret key
        AWS_REGION: S3 region (default: us-east-1)
        AWS_ENDPOINT_URL: Custom endpoint (set by multi-endpoint logic or config)

    Environment Variables (for Azure):
        AZURE_STORAGE_ACCOUNT_NAME: Azure account name
        AZURE_STORAGE_ACCOUNT_KEY: Azure account key

    Environment Variables (for GCS):
        GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON

    MPI Environment Variables (for endpoint distribution):
        OMPI_COMM_WORLD_RANK: OpenMPI process rank
        OMPI_COMM_WORLD_SIZE: OpenMPI total processes
        SLURM_PROCID: SLURM process ID (fallback)
        PMI_RANK: MPICH process rank (fallback)
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)
        self.prefix = namespace

        # Detect backend from URI scheme
        parsed = urlparse(namespace)
        self.scheme = parsed.scheme or "s3"
        self.bucket = parsed.netloc
        self.base_path = parsed.path.lstrip("/")

        # Get storage options from config if available
        storage_options = getattr(self._args, "storage_options", {}) or {}

        # Multi-endpoint support
        endpoint_uris = getattr(self._args, "endpoint_uris", None)
        load_balance_strategy = getattr(
            self._args, "load_balance_strategy", "round_robin"
        )
        use_mpi_distribution = getattr(
            self._args, "use_mpi_endpoint_distribution", False
        )

        # Handle multi-endpoint configuration
        selected_endpoint = None
        if endpoint_uris and len(endpoint_uris) > 0:
            if use_mpi_distribution:
                # MPI-based endpoint selection
                selected_endpoint = self._select_endpoint_via_mpi(endpoint_uris)
                print(f"[s3dlio] MPI-based endpoint selection: {selected_endpoint}")
            else:
                # s3dlio native multi-endpoint (via env vars for now)
                # Future: use s3dlio.MultiEndpointStore when available
                selected_endpoint = self._select_endpoint_via_strategy(
                    endpoint_uris, load_balance_strategy
                )
                print(
                    f"[s3dlio] Selected endpoint ({load_balance_strategy}): {selected_endpoint}"
                )
        elif storage_options.get("endpoint_url"):
            selected_endpoint = storage_options["endpoint_url"]

        # Set environment variables from config
        if storage_options.get("access_key_id"):
            os.environ.setdefault("AWS_ACCESS_KEY_ID", storage_options["access_key_id"])
        if storage_options.get("secret_access_key"):
            os.environ.setdefault(
                "AWS_SECRET_ACCESS_KEY", storage_options["secret_access_key"]
            )
        if storage_options.get("region"):
            os.environ.setdefault("AWS_REGION", storage_options["region"])

        # Set selected endpoint
        if selected_endpoint:
            os.environ["AWS_ENDPOINT_URL"] = selected_endpoint

    def _select_endpoint_via_mpi(self, endpoint_uris):
        """
        Select endpoint based on MPI rank for deterministic distribution.

        Uses OMPI_COMM_WORLD_RANK to assign endpoints:
        - Distributes ranks evenly across endpoints
        - Falls back to SLURM_PROCID if OpenMPI not available
        - Falls back to round-robin index 0 if no MPI environment

        Example: 4 endpoints, 16 ranks → each endpoint serves 4 ranks
          Ranks 0-3   → endpoint[0]
          Ranks 4-7   → endpoint[1]
          Ranks 8-11  → endpoint[2]
          Ranks 12-15 → endpoint[3]
        """
        rank = None

        # Try OpenMPI environment variables
        if "OMPI_COMM_WORLD_RANK" in os.environ:
            rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        # Try SLURM (alternative MPI launcher)
        elif "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
        # Try MPICH
        elif "PMI_RANK" in os.environ:
            rank = int(os.environ["PMI_RANK"])

        if rank is not None:
            # Round-robin assignment based on rank
            endpoint_index = rank % len(endpoint_uris)
            return endpoint_uris[endpoint_index]
        else:
            # No MPI environment - use first endpoint
            print(
                "[s3dlio] Warning: MPI distribution requested but no MPI rank found, using endpoint[0]"
            )
            return endpoint_uris[0]

    def _select_endpoint_via_strategy(self, endpoint_uris, strategy):
        """
        Select endpoint using specified load balancing strategy.

        Strategies:
          - round_robin: rank-based deterministic assignment when an
            MPI/SLURM/PMI rank env var is available (rank % N -- the
            same rank vars _select_endpoint_via_mpi reads), falling back
            to pid % N (logged as best-effort) only when no rank var is
            set. Audit #153 bug 3.10 (B9): previously this always used
            pid % N regardless of rank, which is not round-robin across
            ranks at all -- every process on a distributed run maps to
            whatever endpoint its OS-assigned PID happens to hash to.
            NOTE: because this now matches _select_endpoint_via_mpi's
            rank-based logic whenever a rank var is present,
            use_mpi_endpoint_distribution=true and
            load_balance_strategy=round_robin collapse to the same
            endpoint assignment in that case -- the flag becomes
            vestigial but harmless.
          - least_connections: not implemented yet (needs connection
            tracking); falls back to round_robin (logged).
          - random: random.choice(), seeded from rank-if-available or
            pid-otherwise so repeated runs aren't identically skewed
            across processes.

        Note: For production multi-endpoint with least_connections,
        use s3dlio.MultiEndpointStore when available.
        """
        import random

        rank = _rank_from_env()

        if strategy == "round_robin":
            if rank is not None:
                index = rank % len(endpoint_uris)
            else:
                _logger.warning(
                    "[s3dlio] round_robin endpoint selection requested but no "
                    "MPI/SLURM/PMI rank env var found; falling back to "
                    "pid %% N (best-effort -- not a true round-robin across ranks)"
                )
                index = os.getpid() % len(endpoint_uris)
            return endpoint_uris[index]
        elif strategy == "random":
            random.seed(rank if rank is not None else os.getpid())
            return random.choice(endpoint_uris)
        elif strategy == "least_connections":
            # TODO: Implement connection tracking
            # For now, fall back to round_robin
            _logger.warning(
                "[s3dlio] least_connections not fully implemented, using round_robin"
            )
            return self._select_endpoint_via_strategy(endpoint_uris, "round_robin")
        else:
            # Default: round_robin
            return self._select_endpoint_via_strategy(endpoint_uris, "round_robin")

    def _make_uri(self, path: str) -> str:
        """Convert a relative path to a full URI."""
        if path.startswith(("s3://", "az://", "gs://", "file://", "direct://")):
            return path
        # Combine with prefix
        prefix = self.prefix.rstrip("/")
        path = path.lstrip("/")
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
        """Create directory node using s3dlio.mkdir.

        Audit #153 bug 3.4 (C2): previously ANY exception from
        s3dlio.mkdir was swallowed into `return True` whenever
        exist_ok=True -- an auth failure, a network error, or (common
        for cloud backends, where mkdir is frequently unimplemented) a
        "not implemented" error was all silently treated as "already
        exists, success". Only a genuine already-exists signal
        (FileExistsError) is now treated as success; everything else
        propagates regardless of exist_ok.
        """
        uri = self._make_uri(id)
        try:
            s3dlio.mkdir(uri)
            return True
        except FileExistsError:
            if exist_ok:
                return True
            raise

    @dlp.log
    def get_node(self, id=""):
        """Get node type (FILE, DIRECTORY, or None)."""
        uri = self._make_uri(id)

        # Check if it's a file
        if s3dlio.exists(uri):
            return MetadataType.FILE

        # Check if it's a "directory" by listing children
        try:
            check_uri = uri if uri.endswith("/") else uri + "/"
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
        if not uri.endswith("/"):
            uri += "/"

        try:
            # s3dlio.list returns full URIs
            full_uris = s3dlio.list(uri)

            # Convert to relative paths (just filenames)
            paths = []
            prefix = uri
            for full_uri in full_uris:
                # Extract relative path
                if full_uri.startswith(prefix):
                    relative = full_uri[len(prefix) :]
                else:
                    relative = os.path.basename(urlparse(full_uri).path)

                if relative:
                    paths.append(relative)

            return paths

        except Exception:
            # Audit #153 bug 3.3 (C1): previously swallowed and returned
            # [] -- indistinguishable from "legitimately empty". Log for
            # operator visibility and propagate; a caller iterating a
            # dataset must see the real error, not silently train on
            # zero samples.
            _logger.exception("[s3dlio] Error listing %s", uri)
            raise

    @dlp.log
    def delete_node(self, id):
        """Delete an object.

        Audit #153 bug 3.5 (C3): previously ANY exception from
        s3dlio.delete was swallowed into `return False` -- indistinguishable
        from "already gone" vs. a real failure (auth, network, etc). Only
        a genuine not-found signal (FileNotFoundError) is now treated as
        success (the goal -- "this object doesn't exist" -- is already
        met); everything else propagates.
        """
        uri = self._make_uri(id)
        try:
            s3dlio.delete(uri)
            return True
        except FileNotFoundError:
            return True

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        """
        Write data to storage using s3dlio.

        Objects below the S3DLIO_MULTIPART_THRESHOLD_MB threshold (default
        32 MiB) use a single PUT for lowest overhead.  Objects at or above
        use MultipartUploadWriter for maximum throughput and to avoid the
        S3 5 GiB single-PUT limit.  See
        s3dlio.integrations.dlio._multipart_config for the full env var
        contract (S3DLIO_MULTIPART_THRESHOLD_MB, S3DLIO_MULTIPART_PART_SIZE_MB,
        S3DLIO_MULTIPART_MAX_IN_FLIGHT, S3DLIO_DISABLE_MULTIPART).

        Args:
            id: Path or full URI
            data: bytes or BytesIO object
            offset: Not supported (full object write only)
            length: Not supported (full object write only)
        """
        uri = self._make_uri(id)

        # Handle BytesIO objects (from numpy.save, etc.)
        if hasattr(data, "getvalue"):
            content = data.getvalue()
        elif hasattr(data, "read"):
            if hasattr(data, "seek"):
                data.seek(0)
            content = data.read()
        else:
            content = data

        try:
            size = len(content)
            threshold = _multipart_config.multipart_threshold_bytes()
            if _multipart_config.multipart_disabled() or size < threshold:
                # Single PUT — no three-phase overhead, matches library default behaviour.
                s3dlio.put_bytes(uri, content)
            else:
                # Multipart upload — higher throughput for large objects.
                part_size = _multipart_config.multipart_part_size_bytes()
                writer = s3dlio.MultipartUploadWriter.from_uri(
                    uri,
                    part_size=part_size,
                    max_in_flight=_multipart_config.multipart_max_in_flight(),
                    abort_on_drop=True,
                )
                try:
                    offset_pos = 0
                    while offset_pos < size:
                        n = min(part_size, size - offset_pos)
                        writer.write(content[offset_pos : offset_pos + n])
                        offset_pos += n
                    writer.close()
                except Exception:
                    # Audit #153 bug 3.7 (D6): previously relied on
                    # abort_on_drop=True firing when `writer` goes out of
                    # scope (implicit GC/refcount-timing-dependent
                    # cleanup on the Rust side) instead of aborting
                    # deterministically and immediately here. A failure
                    # partway through a large multi-part write left the
                    # in-progress upload's cleanup timing unspecified.
                    writer.abort()
                    raise
            return None
        except Exception as e:
            print(f"[s3dlio] Error writing to {uri}: {e}")
            raise

    @dlp.log
    def get_data(self, id, data=None, offset=None, length=None):
        """
        Read data from storage using s3dlio.get or s3dlio.get_range.

        Returns BytesView (implements buffer protocol) for ZERO-COPY performance.
        BytesView is compatible with PyTorch (torch.frombuffer), NumPy (np.frombuffer),
        and file writes without creating memory copies.

        Args:
            id: Path or full URI
            data: Ignored (buffer not needed with s3dlio)
            offset: Start byte offset (optional)
            length: Number of bytes to read (optional)

        Returns:
            BytesView: Zero-copy view into Rust-allocated memory (buffer protocol)
        """
        uri = self._make_uri(id)

        # Locked contract (audit #153 f4, docs/implementation-plans/
        # v0.9.109-audit-fix-plan.md bug B5): offset and length are each
        # independently optional per this method's own docstring, but the
        # old `if offset is not None and length is not None` guard only
        # took the get_range() path when BOTH were given. offset-only or
        # length-only silently fell through to a full-object get(),
        # returning the wrong bytes with no error. s3dlio.get_range()'s
        # own contract already treats length=None as "read to end of
        # object", so passing it straight through here is correct.
        try:
            if offset is not None:
                # offset given; length may be None (read to end) or set.
                return s3dlio.get_range(uri, offset=offset, length=length)
            elif length is not None:
                # length-only: read the first `length` bytes from the start.
                return s3dlio.get_range(uri, offset=0, length=length)
            else:
                # Return BytesView directly - zero-copy!
                return s3dlio.get(uri)
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
