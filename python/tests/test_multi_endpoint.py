#!/usr/bin/env python3
"""
Multi-endpoint store tests for Python bindings.

Tests the multi-endpoint functionality including:
- Round-robin and least-connections load balancing
- Zero-copy buffer protocol
- Statistics tracking
- Error handling
"""

import s3dlio
import pytest


class TestMultiEndpointCreation:
    """Tests for creating multi-endpoint stores"""

    def test_create_from_uris(self, tmp_path):
        """Test creating store from URI list"""
        # Create test directories
        dirs = [tmp_path / f"endpoint{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()

        # Create multi-endpoint store with file:// URIs
        uris = [f"file://{d}" for d in dirs]
        store = s3dlio.create_multi_endpoint_store(uris=uris, strategy="round_robin")

        assert store is not None

    def test_create_from_template(self, tmp_path):
        """Test creating store from URI template with range expansion"""
        # Create test directories
        for i in range(1, 4):
            (tmp_path / f"endpoint{i}").mkdir()

        # Use template expansion: {1...3} -> endpoint1, endpoint2, endpoint3
        template = f"file://{tmp_path}/endpoint{{1...3}}"
        store = s3dlio.create_multi_endpoint_store_from_template(
            uri_template=template, strategy="round_robin"
        )

        assert store is not None

    def test_create_from_file(self, tmp_path):
        """Test creating store from configuration file"""
        # Create test directories
        dirs = [tmp_path / f"endpoint{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()

        # Create config file with URIs
        config_file = tmp_path / "endpoints.txt"
        with open(config_file, "w") as f:
            for d in dirs:
                f.write(f"file://{d}\n")

        store = s3dlio.create_multi_endpoint_store_from_file(
            file_path=str(config_file), strategy="least_connections"
        )

        assert store is not None

    def test_invalid_strategy(self, tmp_path):
        """Test that invalid strategy raises error"""
        dir1 = tmp_path / "endpoint1"
        dir1.mkdir()

        with pytest.raises(Exception):
            s3dlio.create_multi_endpoint_store(
                uris=[f"file://{dir1}"], strategy="invalid-strategy"
            )


class TestMultiEndpointOperations:
    """Tests for multi-endpoint CRUD operations.

    `MultiEndpointStore` round-robin routing rewrites *any* fully-qualified
    URI to whichever endpoint the load-balancing strategy selects next --
    by design, for the real use case of several endpoints that are true
    replicas of the same data (see `rewrite_uri_for_endpoint` /
    `select_endpoint` in src/multi_endpoint.rs, and that file's own test
    suite, e.g. `test_round_robin_load_balancing` /
    `test_put_get_operations`, which both write identical data to every
    endpoint directory before exercising round-robin reads for exactly
    this reason). These tests replicate writes/deletes across all 3
    endpoint directories with plain file I/O to model that same
    assumption, instead of asserting read-your-writes against a single
    endpoint the way the original (pre-existing, buggy) fixture did --
    see docs/BUGS_FOUND_DURING_FFI_HARDENING_2026-07-10.md Bug Group B.
    """

    @pytest.fixture
    def multi_store(self, tmp_path):
        """Create a multi-endpoint store for testing"""
        # Create 3 endpoint directories
        dirs = [tmp_path / f"endpoint{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()

        uris = [f"file://{d}" for d in dirs]
        return s3dlio.create_multi_endpoint_store(uris=uris, strategy="round_robin")

    @staticmethod
    def _replicate(tmp_path, relative_path, data):
        """Write `data` directly to every endpoint{0,1,2} directory, so a
        round-robin read/list lands on a real copy regardless of which
        endpoint it's routed to -- mirrors the Rust test suite's own
        replicated-endpoint fixture pattern (see class docstring)."""
        for i in range(3):
            path = tmp_path / f"endpoint{i}" / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    @staticmethod
    def _delete_replicas(tmp_path, relative_path):
        """Remove `relative_path` from every endpoint{0,1,2} directory
        directly. A single `store.delete(uri)` call only removes the copy
        on whichever endpoint round-robin selects for that one call --
        it does not fan out to the other replicas. Deleting all 3 copies
        here is what actually models "the object is gone everywhere";
        relying on one delete() call alone would leave stale copies
        readable on the other 2 endpoints, which is a real, separate
        design question (Reading 2 in Bug Group B) about whether
        delete()/get() of an explicit URI should be scoped to one
        endpoint rather than routed -- not something this test fixture
        should paper over by itself."""
        for i in range(3):
            path = tmp_path / f"endpoint{i}" / relative_path
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_put_and_get(self, multi_store, tmp_path):
        """Test basic put and get operations."""
        test_data = b"Hello from multi-endpoint store!"
        uri = f"file://{tmp_path}/endpoint0/test.txt"

        # Put data (async operation)
        await multi_store.put(uri, test_data)
        # Round-robin's next call may land on a different endpoint than the
        # put did -- replicate so the read finds real data everywhere.
        self._replicate(tmp_path, "test.txt", test_data)

        # Get data back (async operation)
        result = await multi_store.get(uri)
        assert bytes(result) == test_data

    @pytest.mark.asyncio
    async def test_get_range(self, multi_store, tmp_path):
        """Test range get operations."""
        test_data = b"0123456789" * 10  # 100 bytes
        uri = f"file://{tmp_path}/endpoint0/range_test.txt"

        await multi_store.put(uri, test_data)
        self._replicate(tmp_path, "range_test.txt", test_data)

        # Get a range (offset=10, length=10)
        result = await multi_store.get_range(uri, 10, 10)
        assert len(result) == 10
        assert bytes(result) == test_data[10:20]

    @pytest.mark.asyncio
    async def test_list_objects(self, multi_store, tmp_path):
        """Test listing objects"""
        # Create test files, replicated across all 3 endpoints -- each of
        # the 5 put() calls below round-robins to a different endpoint on
        # its own, so without replication the files scatter 1-2 per
        # endpoint instead of landing together where list() looks.
        endpoint_dir = tmp_path / "endpoint0"
        for i in range(5):
            uri = f"file://{endpoint_dir}/file{i}.txt"
            data = f"data{i}".encode()
            await multi_store.put(uri, data)
            self._replicate(tmp_path, f"file{i}.txt", data)

        # List objects (async operation)
        prefix = f"file://{endpoint_dir}/"
        objects = await multi_store.list(prefix, recursive=False)

        assert isinstance(objects, list)
        assert len(objects) >= 5

    @pytest.mark.asyncio
    async def test_delete_object(self, multi_store, tmp_path):
        """Test deleting objects"""
        test_data = b"temporary data"
        uri = f"file://{tmp_path}/endpoint0/delete_me.txt"

        # Put and verify (async operations)
        await multi_store.put(uri, test_data)
        self._replicate(tmp_path, "delete_me.txt", test_data)
        result = await multi_store.get(uri)
        assert bytes(result) == test_data

        # Delete (async operation). A single delete() call only removes the
        # copy on whichever endpoint round-robin selects for this call --
        # explicitly remove the other replicas too so the "verify deleted"
        # check below actually proves the object is gone everywhere, not
        # just luckily absent from whichever endpoint the next get() lands
        # on. See _delete_replicas' docstring for why this isn't a fixture
        # shortcut but a real property of round-robin over replicas.
        await multi_store.delete(uri)
        self._delete_replicas(tmp_path, "delete_me.txt")

        # Verify deleted (should raise error)
        with pytest.raises(Exception):
            await multi_store.get(uri)


class TestZeroCopyBehavior:
    """Tests to verify zero-copy data access"""

    @pytest.fixture
    def multi_store(self, tmp_path):
        """Create a multi-endpoint store for testing"""
        dir1 = tmp_path / "endpoint1"
        dir1.mkdir()

        return s3dlio.create_multi_endpoint_store(
            uris=[f"file://{dir1}"], strategy="round_robin"
        )

    @pytest.mark.asyncio
    async def test_memoryview_access(self, multi_store, tmp_path):
        """Test that returned data supports memoryview (zero-copy)"""
        # Create test data (1 MB)
        test_data = b"x" * (1024 * 1024)
        uri = f"file://{tmp_path}/endpoint1/large.bin"

        # Put data (async operation)
        await multi_store.put(uri, test_data)

        # Get data as BytesView (async operation)
        result = await multi_store.get(uri)

        # Should support buffer protocol (zero-copy access)
        mv = result.memoryview()
        assert len(mv) == len(test_data)
        assert bytes(mv) == test_data

    @pytest.mark.asyncio
    async def test_numpy_integration(self, multi_store, tmp_path):
        """Test zero-copy integration with numpy (if available)"""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        # Create binary data
        original_array = np.arange(1000, dtype=np.float64)
        test_data = original_array.tobytes()
        uri = f"file://{tmp_path}/endpoint1/array.bin"

        # Store data (async operation)
        await multi_store.put(uri, test_data)

        # Retrieve and create numpy array from memoryview (zero-copy, async operation)
        result = await multi_store.get(uri)
        mv = result.memoryview()
        retrieved_array = np.frombuffer(mv, dtype=np.float64)

        assert np.array_equal(retrieved_array, original_array)

    @pytest.mark.asyncio
    async def test_large_object_performance(self, multi_store, tmp_path):
        """Test that large objects don't cause excessive memory copies"""
        # Create 10 MB test data
        size_mb = 10
        test_data = b"A" * (size_mb * 1024 * 1024)
        uri = f"file://{tmp_path}/endpoint1/large_perf.bin"

        # Put data (async operation)
        await multi_store.put(uri, test_data)

        # Get data - should be zero-copy via BytesView (async operation)
        result = await multi_store.get(uri)

        # Verify size without copying to bytes
        mv = result.memoryview()
        assert len(mv) == len(test_data)

        # Accessing via memoryview should not copy
        assert mv[0] == ord("A")
        assert mv[-1] == ord("A")


class TestLoadBalancing:
    """Tests for load balancing behavior"""

    @pytest.mark.asyncio
    async def test_round_robin_distribution(self, tmp_path):
        """Test that round-robin distributes requests evenly"""
        # Create 3 endpoints
        dirs = [tmp_path / f"endpoint{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()

        uris = [f"file://{d}" for d in dirs]
        store = s3dlio.create_multi_endpoint_store(uris=uris, strategy="round_robin")

        # Perform multiple operations (async)
        for i in range(9):
            uri = f"file://{dirs[i % 3]}/file{i}.txt"
            await store.put(uri, f"data{i}".encode())

        # Get statistics
        stats = store.get_endpoint_stats()
        assert isinstance(stats, list)
        assert len(stats) == 3

        # Each endpoint should have some requests
        for stat in stats:
            assert "total_requests" in stat
            assert stat["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_least_connections_strategy(self, tmp_path):
        """Test least-connections strategy"""
        # Create 2 endpoints
        dirs = [tmp_path / f"endpoint{i}" for i in range(2)]
        for d in dirs:
            d.mkdir()

        uris = [f"file://{d}" for d in dirs]
        store = s3dlio.create_multi_endpoint_store(
            uris=uris, strategy="least_connections"
        )

        # Perform operations (async)
        for i in range(4):
            uri = f"file://{dirs[0]}/file{i}.txt"
            await store.put(uri, f"data{i}".encode())

        stats = store.get_endpoint_stats()
        assert len(stats) == 2

    @pytest.mark.asyncio
    async def test_get_total_stats(self, tmp_path):
        """Test retrieving total statistics across all endpoints"""
        dir1 = tmp_path / "endpoint1"
        dir1.mkdir()

        store = s3dlio.create_multi_endpoint_store(
            uris=[f"file://{dir1}"], strategy="round_robin"
        )

        # Perform operations (async)
        uri = f"file://{dir1}/stats_test.txt"
        test_data = b"statistics test"
        await store.put(uri, test_data)
        await store.get(uri)

        # Get total stats
        total_stats = store.get_total_stats()
        assert isinstance(total_stats, dict)
        assert "total_requests" in total_stats
        assert total_stats["total_requests"] >= 2  # At least put + get


class TestErrorHandling:
    """Tests for error handling"""

    def test_empty_uri_list(self):
        """Test that empty URI list raises error"""
        with pytest.raises(Exception):
            s3dlio.create_multi_endpoint_store(uris=[], strategy="round_robin")

    def test_invalid_uri_scheme(self, tmp_path):
        """Test that invalid URI scheme raises error"""
        dir1 = tmp_path / "endpoint1"
        dir1.mkdir()

        # Try to create with mixed schemes (should fail validation)
        with pytest.raises(Exception):
            s3dlio.create_multi_endpoint_store(
                uris=[
                    f"file://{dir1}",
                    "s3://bucket/key",  # Mixed schemes not allowed
                ],
                strategy="round_robin",
            )

    @pytest.mark.asyncio
    async def test_get_nonexistent_object(self, tmp_path):
        """Test that getting non-existent object raises error"""
        dir1 = tmp_path / "endpoint1"
        dir1.mkdir()

        store = s3dlio.create_multi_endpoint_store(
            uris=[f"file://{dir1}"], strategy="round_robin"
        )

        with pytest.raises(Exception):
            await store.get(f"file://{dir1}/does_not_exist.txt")

    def test_invalid_file_path(self):
        """Test that invalid config file path raises error"""
        with pytest.raises(Exception):
            s3dlio.create_multi_endpoint_store_from_file(
                file_path="/nonexistent/path/config.txt", strategy="round_robin"
            )


class TestMultiEndpointExplicitPinning:
    """Tests for issue #162: explicit per-endpoint pinning and fan-out
    replication, for callers whose endpoints are independent/sharded
    (not true replicas) -- e.g. the DLIO_local_changes use case, which
    previously had to reimplement rank-based endpoint pinning externally
    because MultiEndpointStore had no way to target one specific endpoint.
    """

    @pytest.fixture
    def multi_store(self, tmp_path):
        dirs = [tmp_path / f"endpoint{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()
        uris = [f"file://{d}" for d in dirs]
        return s3dlio.create_multi_endpoint_store(uris=uris, strategy="round_robin")

    @pytest.mark.asyncio
    async def test_pinned_access_survives_round_robin_state(
        self, multi_store, tmp_path
    ):
        """Write DISTINCT (non-replicated) data to each endpoint by explicit
        index, perturb round-robin state with unrelated calls, then confirm
        get_from_endpoint(i, ...) always returns endpoint i's own data --
        this is the sharding scenario the ordinary round-robin get()/put()
        cannot support (see docs/BUGS_FOUND_DURING_FFI_HARDENING_2026-07-10.md
        Bug Group B and GitHub issue #162)."""
        shard_data = [f"shard-{i}-payload".encode() for i in range(3)]
        for i, data in enumerate(shard_data):
            await multi_store.put_to_endpoint(i, "shard.bin", data)

        # Perturb round-robin state the way concurrent replicated traffic would.
        for _ in range(5):
            try:
                await multi_store.get(f"file://{tmp_path}/endpoint0/shard.bin")
            except Exception:
                pass

        for i, expected in enumerate(shard_data):
            got = await multi_store.get_from_endpoint(i, "shard.bin")
            assert bytes(got) == expected, (
                f"get_from_endpoint({i}, ...) must return endpoint {i}'s own "
                f"data regardless of round-robin state"
            )

    @pytest.mark.asyncio
    async def test_delete_from_endpoint_targets_only_that_endpoint(
        self, multi_store, tmp_path
    ):
        """delete_from_endpoint(i, ...) must remove the object from endpoint i
        only -- the other endpoints' independent copies (if any) are untouched.
        This directly closes the gap test_delete_object's fixture had to work
        around manually (see Bug Group B)."""
        for i in range(3):
            await multi_store.put_to_endpoint(
                i, "shared_name.bin", f"data-{i}".encode()
            )

        await multi_store.delete_from_endpoint(1, "shared_name.bin")

        # Endpoint 1's copy is gone...
        with pytest.raises(Exception):
            await multi_store.get_from_endpoint(1, "shared_name.bin")
        # ...but endpoints 0 and 2 are untouched.
        assert (
            bytes(await multi_store.get_from_endpoint(0, "shared_name.bin"))
            == b"data-0"
        )
        assert (
            bytes(await multi_store.get_from_endpoint(2, "shared_name.bin"))
            == b"data-2"
        )

    @pytest.mark.asyncio
    async def test_pinned_index_out_of_range_raises(self, multi_store):
        with pytest.raises(Exception):
            await multi_store.get_from_endpoint(99, "x.bin")
        with pytest.raises(Exception):
            await multi_store.put_to_endpoint(99, "x.bin", b"x")
        with pytest.raises(Exception):
            await multi_store.delete_from_endpoint(99, "x.bin")

    @pytest.mark.asyncio
    async def test_put_all_endpoints_replicates_everywhere(self, multi_store, tmp_path):
        """put_all_endpoints must write identical data to every configured
        endpoint in one call -- the fan-out write primitive that was
        previously missing (only list_all_endpoints existed)."""
        data = b"replicate me to all 3 endpoints"
        await multi_store.put_all_endpoints("replicated.bin", data)

        for i in range(3):
            on_disk = (tmp_path / f"endpoint{i}" / "replicated.bin").read_bytes()
            assert on_disk == data

    @pytest.mark.asyncio
    async def test_list_all_endpoints_now_reachable_from_python(
        self, multi_store, tmp_path
    ):
        """list_all_endpoints existed in Rust but was never exposed to Python
        before this fix -- confirm it's now callable and returns the merged,
        deduplicated view across every endpoint."""
        await multi_store.put_all_endpoints("a.bin", b"a")
        await multi_store.put_all_endpoints("b.bin", b"b")

        objects = await multi_store.list_all_endpoints(
            f"file://{tmp_path}/endpoint0/", recursive=False
        )
        assert isinstance(objects, list)
        assert len(objects) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
