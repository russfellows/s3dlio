"""Python side of s3dlio issue #243 (the ``S3DLIO_LOAD_DOTENV`` switch).

``import s3dlio`` is the library entry point: the policy is applied there, before any
thread or pool exists and before anything reads a setting, so a ``.env`` that is switched
on supplies values to settings that are latched at the first I/O call (for example the
runtime's worker-thread count). Without the switch the file is ignored, and the error for
missing credentials says so.

Every scenario runs in a FRESH interpreter (the switch is read once per process) in a
directory that holds a ``.env``.

Run it against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_python_dotenv_243.py``.
"""

import json
import os
import sys
import textwrap

import pytest

from test_oplog_lifecycle import run_scenario

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="counts threads through /proc"
)

DOTENV = """\
AWS_ACCESS_KEY_ID=dotenv-key
AWS_SECRET_ACCESS_KEY=dotenv-secret
AWS_REGION=us-east-1
AWS_ENDPOINT_URL=http://127.0.0.1:1
S3DLIO_243_MARKER=from-dotenv
S3DLIO_243_PRESET=from-dotenv
S3DLIO_RT_THREADS=6
"""

CHILD = """
    import glob, json

    # Python's os.environ is a snapshot taken at start-up; s3dlio sets the C environment,
    # so read it the way the Rust code does.
    import ctypes

    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p

    def c_getenv(name):
        value = libc.getenv(name.encode())
        return None if value is None else value.decode()

    # Right after the import, before any I/O: has the policy been applied?
    out = {"marker": c_getenv("S3DLIO_243_MARKER"),
           "preset": c_getenv("S3DLIO_243_PRESET")}
    try:
        s3dlio.stat("s3://bkt/x")
    except Exception as e:
        out["error"] = str(e)
    names = []
    for path in glob.glob("/proc/self/task/*/comm"):
        try:
            names.append(open(path).read().strip())
        except OSError:
            pass
    # The global runtime's workers: their count is S3DLIO_RT_THREADS, read when the
    # runtime is built at the first call.
    out["rt_workers"] = sum(1 for n in names if n.startswith("s3dlio-rt-worke"))
    print("RESULT " + json.dumps(out))
"""


def run(tmp_path, switch=None):
    (tmp_path / ".env").write_text(DOTENV)
    env = {
        "S3DLIO_243_PRESET": "real-env",
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    if switch is not None:
        env["S3DLIO_LOAD_DOTENV"] = switch
    for var in (
        "S3DLIO_RT_THREADS",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "S3DLIO_LOAD_DOTENV",
    ):
        if var in os.environ and var not in env:
            pytest.skip(f"{var} is set in the caller's environment")
    proc = run_scenario(textwrap.dedent(CHILD), cwd=tmp_path, env=env)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"child failed: rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return json.loads(lines[-1][len("RESULT ") :])


def test_switch_on_loads_the_file_at_import_before_any_io(tmp_path):
    out = run(tmp_path, "1")
    assert out["marker"] == "from-dotenv", out
    assert out["preset"] == "real-env", f"the real environment wins: {out}"


def test_a_setting_latched_at_the_first_call_sees_the_file(tmp_path):
    # S3DLIO_RT_THREADS is read when the global runtime is built, at the first call: the
    # file must already be loaded then, not only when the S3 client is built after it.
    out = run(tmp_path, "1")
    assert out["rt_workers"] == 6, out


def test_without_the_switch_the_file_is_ignored_and_the_error_says_so(tmp_path):
    out = run(tmp_path)
    assert out["marker"] is None, out
    assert out["rt_workers"] != 6, out
    assert "Missing AWS_ACCESS_KEY_ID" in out["error"], out
    assert "S3DLIO_LOAD_DOTENV=1" in out["error"], out


def test_explicit_off_gives_the_plain_error(tmp_path):
    out = run(tmp_path, "0")
    assert out["marker"] is None, out
    assert "Missing AWS_ACCESS_KEY_ID" in out["error"], out
    assert "S3DLIO_LOAD_DOTENV" not in out["error"], out


def test_a_named_file_is_loaded_at_import(tmp_path):
    named = tmp_path / "named.env"
    named.write_text("S3DLIO_243_MARKER=from-named\nS3DLIO_RT_THREADS=5\n")
    out = run(tmp_path, str(named))
    assert out["marker"] == "from-named", out
    assert out["rt_workers"] == 5, out
