"""PyObjectWriter: buffer safety and concurrent use after the GIL release of issue #201.

* ``write_chunk`` of a strided view (``memoryview(b)[::2]``, ``[::-1]``) must write exactly
  the bytes the view shows. It used to build a slice from the buffer pointer and the byte
  length, which writes the wrong bytes for a stride and reads adjacent heap memory for a
  negative stride.
* ``write_chunk`` BORROWS the buffer (zero copy) and releases the GIL for the whole write
  (since 0.9.128), for a mutable exporter too: other Python threads keep running, the buffer is
  exported (``BufferError`` on a resize from another thread) and the caller must not modify it
  until the call returns. ``write_owned_bytes(copy=True)`` is the opt-in snapshot: one copy
  taken with the GIL held, so a thread that overwrites the buffer afterwards cannot tear the
  write.
* While a write is in flight, the read-only accessors (``bytes_written``, ``checksum``,
  ``compressed_bytes``) keep working and a second write waits for the first instead of
  raising "Already mutably borrowed".

Each scenario runs in a fresh interpreter. The slow writes go to a FIFO whose reader (a
separate process) starts draining after a delay.

Run against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_writer_buffers_and_concurrency.py``.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

DELAY = 1.0

READER = (
    "import sys, time\n"
    "f = open(sys.argv[1], 'rb')\n"
    "out = open(sys.argv[2], 'wb')\n"
    f"time.sleep({DELAY})\n"
    "while True:\n"
    "    block = f.read(1 << 22)\n"
    "    if not block:\n"
    "        break\n"
    "    out.write(block)\n"
    "out.close()\n"
)


def run_scenario(body, *args):
    code = "import json, sys, threading, time, s3dlio\n" + textwrap.dedent(body)
    proc = subprocess.Popen(
        [sys.executable, "-c", code, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.terminate()  # SIGTERM only; a hang is reported, not killed
        out, err = proc.communicate(timeout=30)
        pytest.fail(f"scenario hung and was terminated; stdout={out!r} stderr={err!r}")
    assert proc.returncode == 0, f"stdout={out!r} stderr={err!r}"
    return json.loads(out.strip().splitlines()[-1])


@pytest.fixture
def slow_fifo(tmp_path):
    """A FIFO drained after DELAY seconds into `drained`; yields (fifo, drained, reader).

    The test waits for `reader` once its scenario has exited and closed the writer end."""
    fifo = tmp_path / "slow.fifo"
    drained = tmp_path / "drained.bin"
    os.mkfifo(fifo)
    reader = subprocess.Popen(
        [sys.executable, "-I", "-c", READER, str(fifo), str(drained)]
    )
    try:
        yield fifo, drained, reader
    finally:
        if reader.poll() is None:
            reader.terminate()  # SIGTERM only
        reader.wait(timeout=30)


def test_write_chunk_of_strided_views_writes_what_the_view_shows(tmp_path):
    result = run_scenario(
        """
        base = bytearray(range(40))
        views = {
            "step2": memoryview(base)[::2],
            "reversed": memoryview(base)[::-1],
            "step3_offset": memoryview(base)[1::3],
            "contiguous": memoryview(base)[5:25],
        }
        out = {}
        for name, view in views.items():
            for method in ("write_chunk", "write_owned_bytes"):
                path = sys.argv[1] + "/" + name + "-" + method
                writer = s3dlio.create_filesystem_writer("file://" + path, s3dlio.PyWriterOptions())
                getattr(writer, method)(view)
                writer.finalize()
                with open(path, "rb") as f:
                    out[name + "/" + method] = f.read() == bytes(view)
        print(json.dumps(out))
        """,
        str(tmp_path),
    )
    assert result == {
        f"{name}/{method}": True
        for name in ("step2", "reversed", "step3_offset", "contiguous")
        for method in ("write_chunk", "write_owned_bytes")
    }


def test_write_chunk_releases_the_gil_and_locks_a_mutable_buffer(slow_fifo):
    """The documented behaviour since 0.9.128: a blocked ``write_chunk`` of a bytearray does not
    hold the GIL (another thread runs meanwhile), the bytearray is exported (a resize from that
    thread raises BufferError) and nothing is copied: the file receives the buffer's bytes."""
    fifo, drained, reader = slow_fifo
    result = run_scenario(
        """
        size = 16 * 1024 * 1024
        data = bytearray(b"A" * size)
        writer = s3dlio.create_filesystem_writer("file://" + sys.argv[1], s3dlio.PyWriterOptions())
        go = threading.Event()
        box = {}

        def other():
            go.wait()
            time.sleep(0.3)  # write_chunk is now blocked on the FIFO (it drains after DELAY)
            try:
                data.extend(b"x")
                box["locked"] = False
            except BufferError:
                box["locked"] = True
            box["ran_at"] = time.monotonic()

        thread = threading.Thread(target=other)
        thread.start()
        go.set()
        t0 = time.monotonic()
        writer.write_chunk(data)
        elapsed = time.monotonic() - t0
        thread.join()
        print(json.dumps({
            "elapsed": elapsed,
            "locked": box["locked"],
            "other_thread_ran_during_the_write": box["ran_at"] - t0 < elapsed,
        }))
        """,
        str(fifo),
    )
    assert result["elapsed"] >= 0.8 * DELAY, (
        f"the write never waited for the reader: {result}"
    )
    assert result["other_thread_ran_during_the_write"], (
        f"write_chunk held the GIL for a mutable buffer: {result}"
    )
    assert result["locked"], (
        f"the bytearray was not exported during the write: {result}"
    )
    reader.wait(timeout=60)
    content = drained.read_bytes()
    assert len(content) >= 8 * 1024 * 1024, f"only {len(content)} bytes were drained"
    assert content == b"A" * len(content), "the file did not receive the buffer's bytes"


def test_write_owned_bytes_copy_true_cannot_be_torn_by_a_concurrent_writer(slow_fifo):
    """The opt-in snapshot: a thread that overwrites the bytearray cannot tear the write."""
    fifo, drained, reader = slow_fifo
    result = run_scenario(
        """
        size = 16 * 1024 * 1024
        data = bytearray(b"A" * size)
        writer = s3dlio.create_filesystem_writer("file://" + sys.argv[1], s3dlio.PyWriterOptions())
        go = threading.Event()

        def mutate():
            go.wait()
            data[:] = b"Z" * size

        thread = threading.Thread(target=mutate)
        thread.start()
        go.set()
        t0 = time.monotonic()
        writer.write_owned_bytes(data, copy=True)
        elapsed = time.monotonic() - t0
        thread.join()
        print(json.dumps({"elapsed": elapsed}))
        """,
        str(fifo),
    )
    assert result["elapsed"] >= 0.8 * DELAY, (
        f"the write never waited for the reader: {result}"
    )
    reader.wait(timeout=60)
    content = drained.read_bytes()
    # (finalize is not called: it fsyncs, which a FIFO refuses; the unflushed tail is not compared.)
    assert len(content) >= 8 * 1024 * 1024, f"only {len(content)} bytes were drained"
    assert content == b"A" * len(content), (
        "a mutable buffer was torn by a concurrent writer thread: "
        f"{content.count(b'Z')} bytes of {len(content)} are 'Z'"
    )


def test_accessors_and_a_second_write_work_during_an_inflight_write(slow_fifo):
    fifo = slow_fifo[0]
    reader = slow_fifo[2]
    result = run_scenario(
        """
        writer = s3dlio.create_filesystem_writer("file://" + sys.argv[1], s3dlio.PyWriterOptions())
        chunk = bytes(16 * 1024 * 1024)
        errors, seen = [], []
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                try:
                    seen.append(writer.bytes_written())
                    writer.checksum()
                    writer.compressed_bytes()
                except Exception as exc:
                    errors.append("poll: " + repr(exc))
                time.sleep(0.01)

        def second_write():
            time.sleep(0.3)
            try:
                writer.write_chunk(b"tail")
            except Exception as exc:
                errors.append("second write: " + repr(exc))

        threads = [threading.Thread(target=poll), threading.Thread(target=second_write)]
        for t in threads:
            t.start()
        t0 = time.monotonic()
        writer.write_chunk(chunk)
        elapsed = time.monotonic() - t0
        threads[1].join()
        stop.set()
        threads[0].join()
        print(json.dumps({
            "elapsed": elapsed,
            "errors": errors[:5],
            "polls": len(seen),
            "monotonic": all(a <= b for a, b in zip(seen, seen[1:])),
            "final": writer.bytes_written(),
        }))
        """,
        str(fifo),
    )
    assert result["elapsed"] >= 0.8 * DELAY, (
        f"the write never waited for the reader: {result}"
    )
    assert result["errors"] == [], result
    assert result["polls"] >= 20, result
    assert result["monotonic"], result
    assert result["final"] == 16 * 1024 * 1024 + 4, result
    reader.wait(timeout=60)


def test_a_write_waiting_for_the_writer_lock_writes_the_buffer_as_it_is_when_it_starts(
    slow_fifo,
):
    """A mutable buffer changed while its write waits for the writer lock.

    Thread A keeps the writer busy (slow FIFO). Thread B calls write_chunk(bytearray) and
    waits for the lock; thread C overwrites the bytearray during that wait. B's bytes are
    those of the buffer when its write actually starts (all 'Z'), never a mixture.
    (Behavioural guard only: the structural Rust test
    ``write_chunk_slice_after_lock_tests`` pins that no slice over the buffer exists
    during the wait, which no output can show.)
    """
    fifo, drained, reader = slow_fifo
    result = run_scenario(
        """
        writer = s3dlio.create_filesystem_writer("file://" + sys.argv[1], s3dlio.PyWriterOptions())
        size = 8 * 1024 * 1024
        data = bytearray(b"A" * size)
        errors = []

        def first():
            try:
                writer.write_chunk(bytes(16 * 1024 * 1024))
            except Exception as exc:
                errors.append("first: " + repr(exc))

        def second():
            try:
                writer.write_chunk(data)
            except Exception as exc:
                errors.append("second: " + repr(exc))

        def mutate():
            time.sleep(0.5)  # B is waiting for the lock A holds until the reader drains
            data[:] = b"Z" * size

        a = threading.Thread(target=first)
        a.start()
        time.sleep(0.2)  # A holds the lock
        b = threading.Thread(target=second)
        c = threading.Thread(target=mutate)
        b.start()
        c.start()
        for t in (a, b, c):
            t.join()
        print(json.dumps({"errors": errors}))
        """,
        str(fifo),
    )
    assert result["errors"] == [], result
    reader.wait(timeout=60)
    content = drained.read_bytes()
    tail = content.lstrip(b"\x00")  # A's 16 MiB of zeros, then B's bytes
    assert len(tail) >= 1024 * 1024, (
        f"too little of the second write was drained: {len(tail)}"
    )
    assert tail == b"Z" * len(tail), (
        f"the second write was torn: {tail.count(b'A')} 'A' and {tail.count(b'Z')} 'Z' bytes"
    )
