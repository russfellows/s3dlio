"""Op-log lifecycle tests for s3dlio issues #168 and #170.

The op-log is process-global, so each scenario runs in a FRESH interpreter
(``sys.executable -c ...``) and cannot be affected by test ordering. Scenarios
are run with Popen; on a timeout the child gets SIGTERM (never SIGKILL) and the
test fails, reporting the hang.
"""

import shutil
import subprocess
import sys
import textwrap

import pytest

HEADER = (
    "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror"
    "\tstart\tfirst_byte\tend\tduration_ns"
)
# Columns that legitimately differ between two runs of the same workload.
VOLATILE = {"thread", "start", "first_byte", "end", "duration_ns"}
COLUMNS = HEADER.split("\t")


def run_scenario(body: str, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    code = "import sys, os, s3dlio\n" + textwrap.dedent(body)
    proc = subprocess.Popen(
        [sys.executable, "-c", code, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()  # SIGTERM only; a hang is reported, not killed
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            pytest.fail(f"scenario hung for {timeout}s and ignored SIGTERM (pid {proc.pid} left running)")
        pytest.fail(f"scenario hung for {timeout}s and was terminated; stdout={out!r} stderr={err!r}")
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def assert_ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


def assert_complete_zstd_stream(path) -> None:
    """`zstd -t` fails on a truncated stream, so this proves finalize finished it."""
    zstd = shutil.which("zstd")
    assert zstd, "the zstd command-line tool is required to verify op-log files"
    proc = subprocess.run([zstd, "-t", str(path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"{path} is not a complete zstd stream: {proc.stderr}"


def read_rows(path) -> list:
    """Decompress an op-log; check the exact header and 13 columns; return the rows."""
    assert_complete_zstd_stream(path)
    proc = subprocess.run(
        [shutil.which("zstd"), "-dc", str(path)], capture_output=True, check=True
    )
    lines = proc.stdout.decode("utf-8").split("\n")
    assert lines[-1] == "", "op-log must end with a newline"
    assert lines[0] == HEADER, f"unexpected header: {lines[0]!r}"
    rows = []
    for i, line in enumerate(lines[1:-1]):
        cols = line.split("\t")
        assert len(cols) == 13, f"row {i} has {len(cols)} columns: {line!r}"
        assert cols[0] == str(i), f"row {i} has idx {cols[0]}"
        rows.append(dict(zip(COLUMNS, cols)))
    return rows


def stable(rows: list) -> list:
    return [{k: v for k, v in r.items() if k not in VOLATILE} for r in rows]


WORKLOAD = """
    oplog, work = sys.argv[1:3]
    s3dlio.init_op_log(oplog)
    assert s3dlio.is_op_log_active() is True
    a, b = "file://" + work + "/a.bin", "file://" + work + "/b.bin"
    s3dlio.put_bytes(a, b"aaaa")
    s3dlio.put_bytes(b, b"bbbbbb")
    assert bytes(s3dlio.get(a)) == b"aaaa"
    assert bytes(s3dlio.get(b)) == b"bbbbbb"
    s3dlio.finalize_op_log()
    assert s3dlio.is_op_log_active() is False
"""


def test_reinit_after_finalize_raises_runtime_error_and_creates_no_file(tmp_path):
    first = tmp_path / "first.tsv.zst"
    second = tmp_path / "second.tsv.zst"
    existing = tmp_path / "existing.tsv.zst"
    existing.write_bytes(b"known bytes")
    proc = run_scenario(
        """
        first, second, existing = sys.argv[1:4]
        s3dlio.init_op_log(first)
        s3dlio.finalize_op_log()
        size = os.path.getsize(first)
        for path in (second, existing, first):
            try:
                s3dlio.init_op_log(path)
            except RuntimeError as e:
                assert "finalized" in str(e), str(e)
            else:
                raise SystemExit("init after finalize did not raise RuntimeError")
        assert os.path.getsize(first) == size, "first log was truncated"
        """,
        str(first),
        str(second),
        str(existing),
    )
    assert_ok(proc)
    assert not second.exists()
    assert existing.read_bytes() == b"known bytes"
    assert_complete_zstd_stream(first)


def test_is_op_log_active_false_after_finalize(tmp_path):
    path = tmp_path / "ops.tsv.zst"
    proc = run_scenario(
        """
        assert s3dlio.is_op_log_active() is False
        s3dlio.init_op_log(sys.argv[1])
        assert s3dlio.is_op_log_active() is True
        s3dlio.finalize_op_log()
        assert s3dlio.is_op_log_active() is False
        """,
        str(path),
    )
    assert_ok(proc)
    assert read_rows(path) == []


def test_finalize_twice_and_before_init_is_ok(tmp_path):
    path = tmp_path / "ops.tsv.zst"
    proc = run_scenario(
        """
        s3dlio.finalize_op_log()          # before init: no-op
        s3dlio.init_op_log(sys.argv[1])   # still allowed
        s3dlio.finalize_op_log()
        s3dlio.finalize_op_log()          # repeated: no-op
        s3dlio.finalize_op_log()
        assert s3dlio.is_op_log_active() is False
        """,
        str(path),
    )
    assert_ok(proc)
    assert read_rows(path) == []


def test_failed_init_can_be_retried(tmp_path):
    good = tmp_path / "ops.tsv.zst"
    bad = tmp_path / "no-such-dir" / "ops.tsv.zst"
    proc = run_scenario(
        """
        good, bad = sys.argv[1:3]
        try:
            s3dlio.init_op_log(bad)
        except (RuntimeError, OSError) as e:
            assert bad in str(e), "error must name the path: " + str(e)
        else:
            raise SystemExit("bad path was accepted")
        assert s3dlio.is_op_log_active() is False
        s3dlio.init_op_log(good)
        assert s3dlio.is_op_log_active() is True
        s3dlio.finalize_op_log()
        """,
        str(good),
        str(bad),
    )
    assert_ok(proc)
    assert read_rows(good) == []
    assert not bad.exists()


def test_plain_path_and_file_uri_give_the_same_rows(tmp_path):
    """The same workload logs the same rows whichever destination form is used (#168)."""
    work = tmp_path / "work"
    work.mkdir()
    plain = tmp_path / "plain.tsv.zst"
    uri = tmp_path / "uri.tsv.zst"
    assert_ok(run_scenario(WORKLOAD, str(plain), str(work)))
    assert_ok(run_scenario(WORKLOAD, f"file://{uri}", str(work)))

    plain_rows, uri_rows = read_rows(plain), read_rows(uri)
    assert [r["op"] for r in plain_rows] == ["PUT", "PUT", "GET", "GET"]
    assert stable(plain_rows) == stable(uri_rows)


@pytest.mark.parametrize("form", ["plain", "file_uri"])
def test_oplog_never_records_its_own_io(tmp_path, form):
    """Op-log inside the workload directory: one row per workload call, none naming the op-log."""
    work = tmp_path / "work"
    work.mkdir()
    oplog = work / "ops.tsv.zst"
    destination = str(oplog) if form == "plain" else f"file://{oplog}"
    assert_ok(run_scenario(WORKLOAD, destination, str(work)))

    rows = read_rows(oplog)
    assert len(rows) == 4, "exactly one row per workload call (2 puts, 2 gets)"
    for row in rows:
        assert "ops.tsv.zst" not in row["file"], row


def test_remote_scheme_is_rejected_and_creates_no_file(tmp_path):
    proc = run_scenario(
        """
        try:
            s3dlio.init_op_log("s3://some-bucket/oplog.tsv.zst")
        except RuntimeError as e:
            msg = str(e)
            assert "unsupported scheme" in msg and "s3://" in msg, msg
        else:
            raise SystemExit("s3:// op-log path was accepted")
        assert s3dlio.is_op_log_active() is False
        # A later valid init still works: the rejection left it Uninitialized.
        s3dlio.init_op_log(sys.argv[1])
        assert s3dlio.is_op_log_active() is True
        s3dlio.finalize_op_log()
        """,
        str(tmp_path / "ok.tsv.zst"),
    )
    assert_ok(proc)
    # Nothing named after the rejected URI was created, here or in the cwd of the child.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ok.tsv.zst"]
    assert not (tmp_path / "some-bucket").exists()
    read_rows(tmp_path / "ok.tsv.zst")
