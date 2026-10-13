"""Op-log lifecycle tests for s3dlio issue #170.

The op-log is process-global, so each scenario runs in a FRESH interpreter
(``sys.executable -c ...``) and cannot be affected by test ordering.
"""

import subprocess
import sys
import textwrap


def run_scenario(body: str, *args: str) -> subprocess.CompletedProcess:
    code = "import sys, os, s3dlio\n" + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def assert_ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


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
    assert path.exists()


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
    assert path.exists()


def test_failed_init_can_be_retried(tmp_path):
    good = tmp_path / "ops.tsv.zst"
    bad = tmp_path / "no-such-dir" / "ops.tsv.zst"
    proc = run_scenario(
        """
        good, bad = sys.argv[1:3]
        try:
            s3dlio.init_op_log(bad)
        except Exception:
            pass
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
    assert good.exists()
