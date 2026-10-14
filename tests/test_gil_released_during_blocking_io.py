"""s3dlio issue #201: blocking Python bindings must not hold the GIL during network I/O.

While ``list``, ``stat``, a writer's ``write_chunk`` / ``write_owned_bytes`` or
``parquet_rg_row_offsets`` waits for a slow server, a second Python thread must keep
running. Each scenario runs in a fresh interpreter (the S3 client is process-global
and latches ``AWS_ENDPOINT_URL``). A ticker thread increments a counter every 5 ms;
the call under test is made on the main thread against ``slow_s3_server.py`` (its own
process, so it can answer while the call is in flight). The ticker advances by about
``elapsed / 5 ms`` when the GIL is released and by about nothing when it is held.

The ``elapsed`` assertion proves the server really was slow, so a call that returned
early cannot pass for the wrong reason.

Run against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_gil_released_during_blocking_io.py``.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DELAY = 1.0  # seconds the server sleeps before answering each request
TICK = 0.005  # seconds between ticker increments
# Required ticker progress, as a fraction of what a free GIL would give.
MIN_FRACTION = 0.25

SCENARIO = """
    import json, sys, threading, time
    import s3dlio

    ticks = [0]
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks[0] += 1
            time.sleep({tick})

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    time.sleep(0.2)  # the ticker is running before the call starts

    {setup}

    error = None
    c0, t0 = ticks[0], time.monotonic()
    try:
        {call}
    except Exception as exc:  # the call may legitimately fail; only its timing matters
        error = repr(exc)
    elapsed, progress = time.monotonic() - t0, ticks[0] - c0
    stop.set()
    print(json.dumps({{"elapsed": elapsed, "ticks": progress, "error": error}}))
"""


@pytest.fixture
def server():
    proc = subprocess.Popen(
        [sys.executable, "-I", str(HERE / "slow_s3_server.py"), str(DELAY)],
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


def run_call(port, call, setup="pass"):
    code = textwrap.dedent(SCENARIO).format(
        tick=TICK, setup=textwrap.dedent(setup).strip(), call=call
    )
    env = dict(os.environ)
    env.update(
        AWS_ENDPOINT_URL=f"http://127.0.0.1:{port}",
        AWS_ACCESS_KEY_ID="test-key",
        AWS_SECRET_ACCESS_KEY="test-secret",
        AWS_REGION="us-east-1",
    )
    for name in [k for k in env if k.startswith("S3DLIO_OPLOG_")]:
        del env[name]
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        out, err = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.terminate()  # SIGTERM only; a hang is reported, not killed
        out, err = proc.communicate(timeout=30)
        pytest.fail(
            f"scenario hung for 120 s and was terminated; stdout={out!r} stderr={err!r}"
        )
    assert proc.returncode == 0, f"stdout={out!r} stderr={err!r}"
    return json.loads(out.strip().splitlines()[-1])


def assert_ticker_ran(result, what):
    assert result["elapsed"] >= 0.8 * DELAY, (
        f"{what}: the call returned after {result['elapsed']:.3f} s, so it never waited for the "
        f"slow server and proves nothing: {result}"
    )
    free_gil_ticks = result["elapsed"] / TICK
    assert result["ticks"] >= MIN_FRACTION * free_gil_ticks, (
        f"{what}: a second Python thread advanced only {result['ticks']} ticks in "
        f"{result['elapsed']:.3f} s (a free GIL gives about {free_gil_ticks:.0f}): "
        f"the binding held the GIL during network I/O (issue #201). {result}"
    )


def test_list_releases_the_gil(server):
    result = run_call(server, "s3dlio.list('s3://bucket/prefix/')")
    assert_ticker_ran(result, "list")


def test_stat_releases_the_gil(server):
    result = run_call(server, "s3dlio.stat('s3://bucket/key')")
    assert_ticker_ran(result, "stat")


def test_parquet_rg_row_offsets_releases_the_gil(server):
    # The footer fetch is slow; the server's 4-byte body is not a Parquet file, so the
    # call ends in an error after the wait. Only the timing is under test.
    result = run_call(
        server, "s3dlio.parquet_rg_row_offsets('s3://bucket/data.parquet')"
    )
    assert_ticker_ran(result, "parquet_rg_row_offsets")


# The writers. An S3 writer only buffers in memory until finalize, so it has no network
# wait to observe; a filesystem writer does block, and a FIFO makes that wait exact: the
# write cannot finish until the reader (a separate process that opens the FIFO, sleeps
# DELAY seconds, then drains it) starts reading.
READER = (
    "import sys, time\n"
    "f = open(sys.argv[1], 'rb')\n"
    f"time.sleep({DELAY})\n"
    "while f.read(1 << 22):\n"
    "    pass\n"
)

WRITER_SETUP = """
    writer = s3dlio.create_filesystem_writer('file://{fifo}', s3dlio.PyWriterOptions())
    chunk = bytes(16 * 1024 * 1024)
"""


@pytest.fixture
def fifo(tmp_path):
    path = tmp_path / "slow.fifo"
    os.mkfifo(path)
    reader = subprocess.Popen([sys.executable, "-I", "-c", READER, str(path)])
    try:
        yield path
    finally:
        reader.terminate()  # SIGTERM only
        reader.wait(timeout=30)


def writer_call(server, fifo, call):
    return run_call(server, call, setup=WRITER_SETUP.format(fifo=fifo))


def test_writer_write_owned_bytes_releases_the_gil(server, fifo):
    assert_ticker_ran(
        writer_call(server, fifo, "writer.write_owned_bytes(chunk)"),
        "write_owned_bytes",
    )


def test_writer_write_chunk_releases_the_gil(server, fifo):
    assert_ticker_ran(
        writer_call(server, fifo, "writer.write_chunk(chunk)"), "write_chunk"
    )


def test_control_get_already_releases_the_gil(server):
    """Harness control: `get` has always detached, so this passes before and after the fix."""
    result = run_call(server, "s3dlio.get('s3://bucket/key')")
    assert_ticker_ran(result, "get (control)")
