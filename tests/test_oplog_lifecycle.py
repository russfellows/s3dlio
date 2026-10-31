"""Op-log lifecycle tests for s3dlio issues #168 and #170.

The op-log is process-global, so each scenario runs in a FRESH interpreter
(``sys.executable -c ...``) and cannot be affected by test ordering. Scenarios
are run with Popen; on a timeout the child gets SIGTERM (never SIGKILL) and the
test fails, reporting the hang.
"""

import os
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


def zstd_cli() -> str:
    """Path of the zstd command-line tool, or skip the test with the reason.

    The tool is what proves a finished op-log is a COMPLETE zstd stream. CI
    installs it (apt-get install zstd), so a skip there would be a CI defect.
    """
    path = shutil.which("zstd")
    if path is None:
        pytest.skip(
            "the zstd command-line tool is not installed; it is needed to verify "
            "op-log files (install it, e.g. `apt-get install zstd`)"
        )
    return path


def child_env(extra=None) -> dict:
    """This process's environment minus every S3DLIO_OPLOG_* variable, plus `extra`."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("S3DLIO_OPLOG_")}
    env.update(extra or {})
    return env


def run_scenario(
    body: str, *args: str, timeout: int = 120, cwd=None, env=None
) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter whose working directory is `cwd` (or the
    caller's if None), so tests can see what a relative path would create. The child
    gets a copy of this process's environment WITHOUT any S3DLIO_OPLOG_* variable
    (the caller's queue size / lossless / buffer settings must not change what a
    scenario measures); `env` then adds the variables a scenario wants."""
    code = "import sys, os, s3dlio\n" + textwrap.dedent(body)
    proc = subprocess.Popen(
        [sys.executable, "-c", code, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=None if cwd is None else str(cwd),
        env=child_env(env),
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
    zstd = zstd_cli()
    proc = subprocess.run([zstd, "-t", str(path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"{path} is not a complete zstd stream: {proc.stderr}"


def read_rows(path) -> list:
    """Decompress an op-log; check the exact header and 13 columns; return the rows."""
    assert_complete_zstd_stream(path)
    proc = subprocess.run(
        [zstd_cli(), "-dc", str(path)], capture_output=True, check=True
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


def test_reinit_after_finalize_starts_a_new_session_issue_294(tmp_path):
    """Keith's repro (#294): init, ops, finalize, init with another file, ops, finalize."""
    first = tmp_path / "first.tsv.zst"
    second = tmp_path / "second.tsv.zst"
    work = tmp_path / "work"
    work.mkdir()
    proc = run_scenario(
        """
        first, second, work = sys.argv[1:4]
        a, b = "file://" + work + "/a.bin", "file://" + work + "/b.bin"
        s3dlio.init_op_log(first)
        s3dlio.put_bytes(a, b"aaaa")
        assert bytes(s3dlio.get(a)) == b"aaaa"
        s3dlio.finalize_op_log()
        assert s3dlio.is_op_log_active() is False
        assert s3dlio.op_log_stats()["accepted"] == 2

        s3dlio.init_op_log(second)          # used to raise RuntimeError
        assert s3dlio.is_op_log_active() is True
        assert s3dlio.op_log_stats()["accepted"] == 0   # the new session's stats
        s3dlio.put_bytes(b, b"bbbbbb")
        assert bytes(s3dlio.get(b)) == b"bbbbbb"
        s3dlio.stat(b)
        s3dlio.finalize_op_log()
        assert s3dlio.is_op_log_active() is False
        stats = s3dlio.op_log_stats()
        assert stats["accepted"] == 3 and stats["written"] == 3, stats
        """,
        str(first),
        str(second),
        str(work),
    )
    assert_ok(proc)
    rows1 = read_rows(first)
    rows2 = read_rows(second)
    assert [r["op"] for r in rows1] == ["PUT", "GET"], rows1
    assert [r["op"] for r in rows2] == ["PUT", "GET", "HEAD"], rows2
    assert all(r["file"].endswith("a.bin") for r in rows1), rows1
    assert all(r["file"].endswith("b.bin") for r in rows2), rows2


def test_reinit_after_finalize_with_the_same_path_replaces_the_log(tmp_path):
    log = tmp_path / "ops.tsv.zst"
    work = tmp_path / "work"
    work.mkdir()
    proc = run_scenario(
        """
        log, work = sys.argv[1:3]
        a = "file://" + work + "/a.bin"
        s3dlio.init_op_log(log)
        s3dlio.put_bytes(a, b"aaaa")
        s3dlio.get(a)
        s3dlio.finalize_op_log()
        s3dlio.init_op_log(log)   # same path: create-and-truncate, like a first init
        s3dlio.get(a)
        s3dlio.finalize_op_log()
        """,
        str(log),
        str(work),
    )
    assert_ok(proc)
    assert [r["op"] for r in read_rows(log)] == ["GET"]


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
    child_cwd = tmp_path / "child-cwd"
    child_cwd.mkdir()
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
        cwd=child_cwd,
    )
    assert_ok(proc)
    # Nothing named after the rejected URI was created, here or in the child's cwd.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["child-cwd", "ok.tsv.zst"]
    assert sorted(child_cwd.iterdir()) == [], "the child's cwd must still be empty"
    read_rows(tmp_path / "ok.tsv.zst")


@pytest.mark.parametrize("scheme", ["FILE", "File", "direct", "DIRECT"])
def test_other_spellings_and_schemes_are_unsupported(tmp_path, scheme):
    """Only the lowercase `file://` prefix is the file scheme (s3dlio's file-store
    convention); every other scheme is rejected, shown as written, and creates nothing."""
    child_cwd = tmp_path / "child-cwd"
    child_cwd.mkdir()
    target = tmp_path / "target.tsv.zst"
    proc = run_scenario(
        """
        scheme, target = sys.argv[1:3]
        try:
            s3dlio.init_op_log(scheme + "://" + target)
        except RuntimeError as e:
            msg = str(e)
            assert "unsupported scheme '" + scheme + "://'" in msg, msg
        else:
            raise SystemExit(scheme + ":// op-log path was accepted")
        assert s3dlio.is_op_log_active() is False
        """,
        scheme,
        str(target),
        cwd=child_cwd,
    )
    assert_ok(proc)
    assert not target.exists()
    assert sorted(child_cwd.iterdir()) == [], "the child's cwd must still be empty"


@pytest.mark.parametrize("first_component", ["localhost", "relative", "pct%20dir"])
def test_file_uri_remainder_is_a_literal_relative_path(tmp_path, first_component):
    """s3dlio's file:// convention, not RFC 8089: `file://localhost/x` is the RELATIVE
    path `localhost/x` (under the child's cwd), and `%20` is not decoded."""
    child_cwd = tmp_path / "child-cwd"
    (child_cwd / first_component).mkdir(parents=True)
    proc = run_scenario(
        """
        s3dlio.init_op_log("file://" + sys.argv[1] + "/ops.tsv.zst")
        assert s3dlio.is_op_log_active() is True
        s3dlio.finalize_op_log()
        """,
        first_component,
        cwd=child_cwd,
    )
    assert_ok(proc)
    assert read_rows(child_cwd / first_component / "ops.tsv.zst") == []
    # It went nowhere else.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["child-cwd"]
    assert sorted(p.name for p in child_cwd.iterdir()) == [first_component]


def test_file_uri_with_nothing_after_it_is_rejected(tmp_path):
    proc = run_scenario(
        """
        try:
            s3dlio.init_op_log("file://")
        except RuntimeError as e:
            assert "empty" in str(e), str(e)
        else:
            raise SystemExit("file:// alone was accepted")
        assert s3dlio.is_op_log_active() is False
        """,
        cwd=tmp_path,
    )
    assert_ok(proc)
    assert sorted(tmp_path.iterdir()) == []


STATS_KEYS = {
    "accepted",
    "dropped_full",
    "rejected_closed",
    "writer_failed",
    "lossless",
    "written",
    "writer_error",
    "rejected_unformattable",
    "rejected_unsafe_field",
    "normalized_error_text",
}


def test_op_log_stats_none_before_init_then_counts_match_rows(tmp_path):
    """op_log_stats() is None before init, a dict of counts while active and after
    finalize; for a small file:// workload accepted == the data rows in the file (#170)."""
    work = tmp_path / "work"
    work.mkdir()
    oplog = tmp_path / "ops.tsv.zst"
    proc = run_scenario(
        """
        import json
        oplog, work = sys.argv[1:3]
        assert s3dlio.op_log_stats() is None
        s3dlio.init_op_log(oplog)
        during = s3dlio.op_log_stats()
        assert isinstance(during, dict)
        a, b = "file://" + work + "/a.bin", "file://" + work + "/b.bin"
        s3dlio.put_bytes(a, b"aaaa")
        s3dlio.put_bytes(b, b"bbbbbb")
        assert bytes(s3dlio.get(a)) == b"aaaa"
        assert bytes(s3dlio.get(b)) == b"bbbbbb"
        s3dlio.finalize_op_log()
        after = s3dlio.op_log_stats()
        print(json.dumps({"during": during, "after": after}))
        """,
        str(oplog),
        str(work),
    )
    assert_ok(proc)
    import json

    reported = json.loads(proc.stdout.strip().splitlines()[-1])
    during, after = reported["during"], reported["after"]
    assert set(during) == STATS_KEYS and set(after) == STATS_KEYS
    assert (during["accepted"], during["dropped_full"]) == (0, 0)
    assert after["lossless"] is False
    assert after["dropped_full"] == after["rejected_closed"] == after["writer_failed"] == 0
    rows = read_rows(oplog)
    assert len(rows) == 4
    assert after["accepted"] == len(rows)
    # Complete: accepted == written, nothing dropped or refused, no writer error.
    assert after["written"] == after["accepted"]
    assert after["writer_error"] is None
    # #190: rows whose timestamp cannot be formatted are counted; none here.
    assert during["rejected_unformattable"] == after["rejected_unformattable"] == 0
    # #186: records with a TAB/LF/CR in a value are rejected, error texts reworded; none here.
    for key in ("rejected_unsafe_field", "normalized_error_text"):
        assert during[key] == after[key] == 0, key
    assert "INCOMPLETE" not in proc.stderr, "no warning when nothing was lost"


def test_a_tab_or_newline_in_a_file_name_rejects_the_record_and_is_counted(tmp_path):
    """#186: file:// names with a TAB or LF no longer break the row. The record is
    absent, counted in rejected_unsafe_field (so the log is incomplete and finalize
    says so), and the records before and after it are intact 13-column rows."""
    work = tmp_path / "work"
    work.mkdir()
    oplog = tmp_path / "ops.tsv.zst"
    proc = run_scenario(
        """
        import json
        oplog, work = sys.argv[1:3]
        s3dlio.init_op_log(oplog)
        s3dlio.put_bytes("file://" + work + "/before.bin", b"x" * 10)
        s3dlio.put_bytes("file://" + work + "/a\\tb.bin", b"x" * 10)
        s3dlio.put_bytes("file://" + work + "/c\\nd.bin", b"y" * 10)
        try:
            s3dlio.get("file://" + work + "/missing\\tname.bin")
        except Exception:
            pass
        s3dlio.put_bytes("file://" + work + "/after.bin", b"z" * 10)
        s3dlio.finalize_op_log()
        print(json.dumps(s3dlio.op_log_stats()))
        """,
        str(oplog),
        str(work),
    )
    assert_ok(proc)
    import json

    stats = json.loads(proc.stdout.strip().splitlines()[-1])
    assert set(stats) == STATS_KEYS
    assert stats["rejected_unsafe_field"] == 3, stats
    assert stats["normalized_error_text"] == 0, stats
    assert stats["rejected_unformattable"] == 0, stats
    assert stats["accepted"] == stats["written"] + 3, stats
    # Every row in the file has 13 columns; the rejected ones left gaps in idx.
    zstd_cli()
    assert_complete_zstd_stream(oplog)
    text = subprocess.run(
        [zstd_cli(), "-dc", str(oplog)], capture_output=True, check=True
    ).stdout.decode("utf-8")
    lines = text.split("\n")
    assert lines[-1] == "" and lines[0] == HEADER
    rows = [dict(zip(COLUMNS, line.split("\t"))) for line in lines[1:-1]]
    assert all(len(line.split("\t")) == 13 for line in lines[1:-1]), text
    assert [r["file"].rsplit("/", 1)[-1] for r in rows] == ["before.bin", "after.bin"], rows
    assert [r["idx"] for r in rows] == ["0", "4"], rows
    assert "INCOMPLETE" in proc.stderr and "3 entries were rejected because" in proc.stderr
    assert "TAB, LF or CR" in proc.stderr


FLOOD = """
    import json, threading
    oplog, work = sys.argv[1:3]
    s3dlio.init_op_log(oplog)
    uri = "file://" + work + "/a.bin"
    s3dlio.put_bytes(uri, b"aaaa")
    def worker():
        for _ in range(WORKER_CALLS):
            s3dlio.get(uri)
    threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s3dlio.finalize_op_log()
    print(json.dumps(s3dlio.op_log_stats()))
"""
WORKERS = 8
WORKER_CALLS = 2500
FLOOD_CALLS = 1 + WORKERS * WORKER_CALLS  # 1 put + the gets


def run_flood(tmp_path, env):
    import json

    work = tmp_path / "work"
    work.mkdir()
    oplog = tmp_path / "ops.tsv.zst"
    body = FLOOD.replace("WORKER_CALLS", str(WORKER_CALLS)).replace("WORKERS", str(WORKERS))
    proc = run_scenario(body, str(oplog), str(work), env=env)
    assert_ok(proc)
    return json.loads(proc.stdout.strip().splitlines()[-1]), read_rows(oplog), proc.stderr


def test_default_mode_counts_every_call_and_warns_exactly_when_entries_were_dropped(tmp_path):
    """One-slot queue, eight threads, 20001 calls, default (lossy) mode.

    Whether any call is dropped depends on how the writer thread is scheduled on
    a shared host, so this test asserts only what must hold for EVERY schedule:
    each call is counted exactly once (accepted + dropped_full == calls),
    accepted == the rows in the file, and finalize prints one warning if and only
    if something was dropped (with the drop count, the remedies and the other
    non-zero counts). It therefore cannot flake: no schedule violates it. On a
    host where the writer happens to keep up, the "drops happened" branch is not
    exercised; that case is pinned deterministically elsewhere (a stalled sink in
    the Rust unit tests of s3_logger::oplog_stats_tests, a real disk-limit error
    in tests/oplog_lifecycle_finalize_warning.rs and in the last test below)."""
    env = {"S3DLIO_OPLOG_BUF": "1", "S3DLIO_OPLOG_LOSSLESS": "0"}
    stats, rows, stderr = run_flood(tmp_path, env)
    assert stats["lossless"] is False
    assert stats["accepted"] + stats["dropped_full"] == FLOOD_CALLS, stats
    assert stats["rejected_closed"] == stats["writer_failed"] == 0
    assert stats["accepted"] == stats["written"] == len(rows), stats
    assert stats["writer_error"] is None
    warnings = stderr.count("op-log is INCOMPLETE")
    assert warnings == (1 if stats["dropped_full"] > 0 else 0), (stats, stderr)
    if stats["dropped_full"] > 0:
        assert f"{stats['dropped_full']} entries were dropped because the queue was full" in stderr
        assert "S3DLIO_OPLOG_LOSSLESS=1" in stderr and "S3DLIO_OPLOG_BUF" in stderr


def test_lossless_mode_reports_no_drops_and_no_warning(tmp_path):
    env = {"S3DLIO_OPLOG_BUF": "1", "S3DLIO_OPLOG_LOSSLESS": "1"}
    stats, rows, stderr = run_flood(tmp_path, env)
    assert stats["lossless"] is True
    assert stats["accepted"] == stats["written"] == len(rows) == FLOOD_CALLS, stats
    assert stats["dropped_full"] == stats["rejected_closed"] == stats["writer_failed"] == 0
    assert stats["writer_error"] is None
    assert "INCOMPLETE" not in stderr, stderr


def test_write_error_at_the_end_of_the_stream_is_reported_deterministically(tmp_path):
    """A REAL I/O error, produced deterministically: after the header is on disk the
    child caps the file at its current size (RLIMIT_FSIZE, SIGXFSZ ignored so the
    kernel returns EFBIG instead of killing the process). Four tiny rows stay in the
    compressor, so finalize's write of the stream end fails: the rows were all
    handed to the compressor (written == accepted) but the log is not complete, and
    both op_log_stats() and the one stderr warning say so."""
    work = tmp_path / "work"
    work.mkdir()
    oplog = tmp_path / "ops.tsv.zst"
    proc = run_scenario(
        """
        import json, resource, signal
        oplog, work = sys.argv[1:3]
        s3dlio.init_op_log(oplog)
        size = os.path.getsize(oplog)
        assert size > 0, "the header must already be on disk"
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
        a = "file://" + work + "/a.bin"
        s3dlio.put_bytes(a, b"aaaa")
        for _ in range(3):
            s3dlio.get(a)
        s3dlio.finalize_op_log()
        print(json.dumps(s3dlio.op_log_stats()))
        """,
        str(oplog),
        str(work),
    )
    assert_ok(proc)
    import json

    stats = json.loads(proc.stdout.strip().splitlines()[-1])
    assert stats["accepted"] == stats["written"] == 4, stats
    assert stats["dropped_full"] == stats["rejected_closed"] == stats["writer_failed"] == 0
    assert isinstance(stats["writer_error"], str) and "File too large" in stats["writer_error"], stats
    assert proc.stderr.count("op-log is INCOMPLETE") == 1, proc.stderr
    assert "the writer failed:" in proc.stderr and "File too large" in proc.stderr
    assert "S3DLIO_OPLOG_LOSSLESS=1" not in proc.stderr, "no queue advice: nothing was dropped"
