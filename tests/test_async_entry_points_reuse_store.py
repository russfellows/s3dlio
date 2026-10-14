"""s3dlio issue #204: the async Python entry points must reuse the cached store and its S3 client.

``stat_async``, ``exists_async``, ``put_bytes_async`` and ``put_many_async`` used to build
a new store, and with it a new per-endpoint S3 client with its own connection pool, on
EVERY call. Against one host-form server (``s3://127.0.0.1:PORT/bucket/key``) a sequence
of awaited calls therefore opened one TCP connection per call. Routed through the shared
store cache (``get_or_create_store``), the same sequence reuses one pooled connection.

``slow_s3_server.py`` (own process, delay 0, persistent HTTP/1.1 connections) counts the
connections it accepts. ``mkdir_async`` makes no request on S3 (S3 has no ``mkdir``), so
its routing is pinned by the Rust unit test ``issue204_async_entry_points_use_the_store_cache``.

Run against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_async_entry_points_reuse_store.py``.
"""

import json
import os
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CALLS = 12
# One pooled connection is expected; a little slack keeps the test off scheduler timing.
MAX_CONNECTIONS = 3


@pytest.fixture
def server():
    proc = subprocess.Popen(
        [sys.executable, "-I", str(HERE / "slow_s3_server.py"), "0"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline().strip()
        assert line.startswith("PORT "), f"mock server did not start: {line!r}"
        yield int(line.split()[1])
    finally:
        proc.terminate()  # SIGTERM only
        proc.wait(timeout=30)


def server_stats(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/__stats", timeout=30) as resp:
        return json.load(resp)


def run_awaited_calls(port, call):
    """Await `call` CALLS times, one after another, in one fresh interpreter."""
    code = textwrap.dedent(
        f"""
        import asyncio, s3dlio
        uri = "s3://127.0.0.1:{port}/bucket/key"

        async def main():
            for i in range({CALLS}):
                {call}

        asyncio.run(main())
        """
    )
    env = dict(os.environ)
    env.update(
        AWS_ACCESS_KEY_ID="test-key",
        AWS_SECRET_ACCESS_KEY="test-secret",
        AWS_REGION="us-east-1",
    )
    env.pop("AWS_ENDPOINT_URL", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


def assert_reuses_connection(port, method):
    stats = server_stats(port)
    served = [r for r in stats["requests"] if r[0] == method]
    assert len(served) == CALLS, (
        f"expected {CALLS} {method} requests, saw {stats['requests']}"
    )
    assert stats["connections"] <= MAX_CONNECTIONS, (
        f"{CALLS} sequential awaited calls opened {stats['connections']} connections: each call "
        f"built a new store and S3 client instead of reusing the cached one (issue #204)"
    )


def test_stat_async_reuses_one_client(server):
    run_awaited_calls(server, "await s3dlio.stat_async(uri)")
    assert_reuses_connection(server, "HEAD")


def test_exists_async_reuses_one_client(server):
    run_awaited_calls(server, "assert await s3dlio.exists_async(uri)")
    assert_reuses_connection(server, "HEAD")


def test_put_bytes_async_reuses_one_client(server):
    run_awaited_calls(server, "await s3dlio.put_bytes_async(uri, b'0123456789')")
    assert_reuses_connection(server, "PUT")


def test_put_many_async_reuses_one_client(server):
    # Each call carries one item (the awaits are sequential), so only store reuse can keep the count down.
    run_awaited_calls(
        server, "await s3dlio.put_many_async([(uri, b'abc')], max_in_flight=1)"
    )
    assert_reuses_connection(server, "PUT")


def test_a_bad_server_uri_errors_without_creating_a_store(server):
    """#171/#178: an invalid server (portless IP) is an error, never served from the cache."""
    code = textwrap.dedent(
        f"""
        import asyncio, s3dlio
        good = "s3://127.0.0.1:{server}/bucket/key"

        async def main():
            await s3dlio.stat_async(good)          # populate the cache for the real server
            for bad in ("s3://127.0.0.1/bucket/key", "s3://10.0.0.1/bucket/key"):
                try:
                    await s3dlio.stat_async(bad)
                except Exception as exc:
                    print("ERR", bad, type(exc).__name__)
                else:
                    raise SystemExit("bad URI " + bad + " was served")

        asyncio.run(main())
        """
    )
    env = dict(os.environ)
    env.update(
        AWS_ACCESS_KEY_ID="test-key",
        AWS_SECRET_ACCESS_KEY="test-secret",
        AWS_REGION="us-east-1",
    )
    env.pop("AWS_ENDPOINT_URL", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert proc.stdout.count("ERR") == 2, proc.stdout
    # Only the one good HEAD ever reached the server.
    assert [r[0] for r in server_stats(server)["requests"]] == ["HEAD"]
