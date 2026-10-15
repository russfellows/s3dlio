"""Issue #211: ``max_in_flight=0`` must be rejected immediately.

A concurrency limit of 0 builds a semaphore with no permits, so the first
acquire never completes and the call hangs forever.  Every Python entry point
that takes ``max_in_flight`` must instead raise ``ValueError`` at once, naming
the parameter and the valid range (an integer >= 1) -- never substitute another
value, never hang.

Each case runs in its own child process so that, on an unfixed build, the hang
is reported as a timeout instead of freezing the test run.  A hung child is
stopped with SIGTERM only (never SIGKILL).
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest
import s3dlio

# Bounded time for "immediately": import + one rejected call is well under 1 s.
REJECT_TIMEOUT_S = 20

# A dead local endpoint with dummy credentials: the S3-only entry points must
# reject 0 before any request is made, so nothing is ever contacted.
S3_ENV = {
    "AWS_ENDPOINT_URL": "http://127.0.0.1:9",
    "AWS_ACCESS_KEY_ID": "test",
    "AWS_SECRET_ACCESS_KEY": "test",
    "AWS_REGION": "us-east-1",
}

# name -> child-process body.  ``pre`` is a file:// prefix of a scratch dir
# holding object-0..2.bin; ``src`` is a local file; ``dst`` an empty dir.
SYNC_CASES = {
    "put": 's3dlio.put(pre, 2, "p-{}.bin", max_in_flight=0, size=16)',
    "get_many_file": "s3dlio.get_many(uris, max_in_flight=0)",
    "put_many": 's3dlio.put_many([(pre + "m.bin", b"x")], max_in_flight=0)',
    "download": "s3dlio.download(pre, dst, max_in_flight=0)",
    "upload": 's3dlio.upload([src], "file://" + dst + "/", max_in_flight=0)',
    "get_many_stats": "s3dlio.get_many_stats(s3uris, max_in_flight=0)",
    "get_many_s3": "s3dlio.get_many(s3uris, max_in_flight=0)",
    "mp_get": 's3dlio.mp_get("s3://b/p/", procs=1, jobs=0, num=1)',
    "mp_get_procs": 's3dlio.mp_get("s3://b/p/", procs=0, jobs=64, num=1)',
    "multipart_writer": 's3dlio.MultipartUploadWriter("b", "k", max_in_flight=0)',
    "multipart_writer_from_uri": 's3dlio.MultipartUploadWriter.from_uri("s3://b/k", max_in_flight=0)',
}

# Async entry points: the argument error must surface when the call is made
# (before any future exists), and also be awaited-safe.
ASYNC_CASES = {
    "put_async": 'await s3dlio.put_async(pre, 2, "p-{}.bin", max_in_flight=0, size=16)',
    "put_many_async": 'await s3dlio.put_many_async([(pre + "m.bin", b"x")], max_in_flight=0)',
    "get_many_async": "await s3dlio.get_many_async(s3uris, max_in_flight=0)",
    "get_many_stats_async": "await s3dlio.get_many_stats_async(s3uris, max_in_flight=0)",
}

CHILD = textwrap.dedent(
    """
    import asyncio, os, sys, tempfile
    import s3dlio

    base = tempfile.mkdtemp(prefix="s211-")
    pre = "file://" + base + "/"
    for i in range(3):
        with open(os.path.join(base, f"object-{{i}}.bin"), "wb") as f:
            f.write(b"data")
    uris = [pre + f"object-{{i}}.bin" for i in range(3)]
    s3uris = ["s3://bucket/object-0", "s3://bucket/object-1"]
    src = os.path.join(base, "object-0.bin")
    dst = tempfile.mkdtemp(prefix="s211-dst-")

    async def main():
        {body_async}

    def run():
        {body_sync}

    try:
        if {is_async}:
            asyncio.run(main())
        else:
            run()
    except ValueError as e:
        print("ValueError:", e)
        sys.exit(0)
    except BaseException as e:
        print(type(e).__name__ + ":", e)
        sys.exit(3)
    print("NO-ERROR")
    sys.exit(4)
    """
)


def _run_child(body: str, is_async: bool) -> subprocess.CompletedProcess:
    code = CHILD.format(
        body_async=body if is_async else "pass",
        body_sync="pass" if is_async else body,
        is_async=is_async,
    )
    env = dict(os.environ)
    env.update(S3_ENV)
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        out, _ = proc.communicate(timeout=REJECT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.terminate()  # SIGTERM only
        try:
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            pytest.fail("child ignored SIGTERM after hanging")
        pytest.fail(
            f"max_in_flight=0 HUNG (no error within {REJECT_TIMEOUT_S}s): {body}"
        )
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, None)


def _assert_rejected(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, f"expected ValueError, got:\n{result.stdout}"
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("ValueError:"), line
    assert any(n in line for n in ("max_in_flight", "jobs", "procs")), line
    assert ">= 1" in line, f"message must state the valid range: {line}"


@pytest.mark.parametrize("name", sorted(SYNC_CASES))
def test_zero_is_rejected_immediately_sync(name):
    _assert_rejected(_run_child(SYNC_CASES[name], is_async=False))


@pytest.mark.parametrize("name", sorted(ASYNC_CASES))
def test_zero_is_rejected_immediately_async(name):
    _assert_rejected(_run_child(ASYNC_CASES[name], is_async=True))


def test_error_names_parameter_and_range():
    result = _run_child(
        's3dlio.put_many([(pre + "x.bin", b"x")], max_in_flight=0)', is_async=False
    )
    _assert_rejected(result)
    assert "got 0" in result.stdout


def test_valid_values_still_work():
    """max_in_flight=1 (the smallest valid value) and a larger value both work."""
    for mif in (1, 4):
        with tempfile.TemporaryDirectory(prefix="s211-ok-") as d:
            pre = "file://" + d + "/"
            s3dlio.put(pre, 3, "o-{}.bin", max_in_flight=mif, size=16)
            uris = [pre + f"o-{i}.bin" for i in range(3)]
            got = s3dlio.get_many(uris, max_in_flight=mif)
            assert [u for u, _ in got] == uris
            assert all(len(bytes(b)) == 16 for _, b in got)

            s3dlio.put_many(
                [(pre + f"m-{i}.bin", b"abc") for i in range(3)], max_in_flight=mif
            )
            with tempfile.TemporaryDirectory(prefix="s211-dl-") as dl:
                s3dlio.download(pre, dl, max_in_flight=mif)
                assert sorted(os.listdir(dl)) == sorted(os.listdir(d))
            with tempfile.TemporaryDirectory(prefix="s211-up-") as up:
                s3dlio.upload([d + "/o-0.bin"], "file://" + up + "/", max_in_flight=mif)
                assert os.listdir(up) == ["o-0.bin"]


def test_valid_values_still_work_async():
    import asyncio

    async def go():
        with tempfile.TemporaryDirectory(prefix="s211-aok-") as d:
            pre = "file://" + d + "/"
            await s3dlio.put_many_async(
                [(pre + f"a-{i}.bin", b"abc") for i in range(3)], max_in_flight=1
            )
            await s3dlio.put_async(pre, 2, "p-{}.bin", max_in_flight=2, size=16)
            assert len(os.listdir(d)) == 5

    asyncio.run(go())


def test_put_with_zero_objects_is_not_an_error():
    """num=0 derives an internal job count of 0; that is not the caller's 0."""
    with tempfile.TemporaryDirectory(prefix="s211-n0-") as d:
        s3dlio.put("file://" + d + "/", 0, "z-{}", max_in_flight=4, size=16)
        assert os.listdir(d) == []
