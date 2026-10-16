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

# The global runtime's worker count is S3DLIO_RT_THREADS when set, else one per CPU (the
# per-process budget). Tokio names workers and blocking-pool threads alike, so the test
# counts "runtime threads" = workers + the few blocking threads started lazily, and picks a
# worker count that is (a) not clamped: s3dlio raises a value below budget/4 up to the
# budget (`clamped_env_rt_threads`), so RT_WORKERS is at least budget/4, and (b) far enough
# below the default (the budget) that workers + BLOCKING_ALLOWANCE cannot reach it.
CPUS = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else (os.cpu_count() or 1)
)
RT_WORKERS = max(2, -(-CPUS // 4))  # ceil(CPUS / 4), never below the clamp threshold
BLOCKING_ALLOWANCE = 2
_MPI_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE")

DOTENV = f"""\
AWS_ACCESS_KEY_ID=dotenv-key
AWS_SECRET_ACCESS_KEY=dotenv-secret
AWS_REGION=us-east-1
AWS_ENDPOINT_URL=http://127.0.0.1:1
S3DLIO_243_MARKER=from-dotenv
S3DLIO_243_PRESET=from-dotenv
S3DLIO_RT_THREADS={RT_WORKERS}
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
    # Runtime threads (workers plus lazily started blocking threads share one name): the
    # workers number S3DLIO_RT_THREADS, read when the
    # runtime is built at the first call.
    out["rt_workers"] = sum(1 for n in names if n.startswith("s3dlio-rt-worke"))
    print("RESULT " + json.dumps(out))
"""


def need_thread_gap():
    """Skip when the host cannot tell RT_WORKERS from the default worker count."""
    if any(v in os.environ for v in _MPI_VARS):
        pytest.skip("an MPI world-size variable changes the per-process thread budget")
    if CPUS < RT_WORKERS + BLOCKING_ALLOWANCE + 3:
        pytest.skip(
            f"{CPUS} CPUs leave no gap between {RT_WORKERS} workers "
            f"(+{BLOCKING_ALLOWANCE} blocking threads) and the default of one per CPU"
        )


def is_configured_count(runtime_threads):
    """workers + (lazily started) blocking threads, and clearly not the per-CPU default."""
    return RT_WORKERS <= runtime_threads <= RT_WORKERS + BLOCKING_ALLOWANCE


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
    need_thread_gap()
    out = run(tmp_path, "1")
    assert is_configured_count(out["rt_workers"]), (out, RT_WORKERS, CPUS)


def test_without_the_switch_the_file_is_ignored_and_the_error_says_so(tmp_path):
    out = run(tmp_path)
    assert out["marker"] is None, out
    if CPUS >= RT_WORKERS + BLOCKING_ALLOWANCE + 3 and not any(
        v in os.environ for v in _MPI_VARS
    ):
        # Without the file the runtime keeps its default size, not RT_WORKERS.
        assert not is_configured_count(out["rt_workers"]), (out, RT_WORKERS, CPUS)
    assert "Missing AWS_ACCESS_KEY_ID" in out["error"], out
    assert "S3DLIO_LOAD_DOTENV=1" in out["error"], out


def test_explicit_off_gives_the_plain_error(tmp_path):
    out = run(tmp_path, "0")
    assert out["marker"] is None, out
    assert "Missing AWS_ACCESS_KEY_ID" in out["error"], out
    assert "S3DLIO_LOAD_DOTENV" not in out["error"], out


def test_a_named_file_is_loaded_at_import(tmp_path):
    named = tmp_path / "named.env"
    named.write_text(f"S3DLIO_243_MARKER=from-named\nS3DLIO_RT_THREADS={RT_WORKERS}\n")
    out = run(tmp_path, str(named))
    assert out["marker"] == "from-named", out
    need_thread_gap()
    assert is_configured_count(out["rt_workers"]), (out, RT_WORKERS, CPUS)


FAKE_TORCH_INIT = """
import ctypes, os
_libc = ctypes.CDLL(None)
_libc.getenv.restype = ctypes.c_char_p
_value = _libc.getenv(b"S3DLIO_243_MARKER")
os.environ["FAKE_TORCH_SAW"] = "<unset>" if _value is None else _value.decode()
"""

FAKE_TORCH_DATA = """
class Dataset:
    pass


class IterableDataset:
    pass


def get_worker_info():
    return None
"""


def test_the_file_is_loaded_before_s3dlios_own_torch_import(tmp_path):
    # s3dlio/torch.py used to import torch before the native module, so a torch-enabled
    # install loaded the file only after torch's import. A stand-in torch reports what it
    # could see of the environment at the moment s3dlio imports it.
    fakes = tmp_path / "fakes"
    (fakes / "torch" / "utils" / "data").mkdir(parents=True)
    (fakes / "torch" / "__init__.py").write_text(FAKE_TORCH_INIT)
    (fakes / "torch" / "utils" / "__init__.py").write_text("")
    (fakes / "torch" / "utils" / "data" / "__init__.py").write_text(FAKE_TORCH_DATA)
    (tmp_path / ".env").write_text(DOTENV)
    env = {"S3DLIO_LOAD_DOTENV": "1", "PYTHONPATH": str(fakes)}
    proc = run_scenario(
        'print("SAW " + os.environ.get("FAKE_TORCH_SAW", "<torch never imported>"))',
        cwd=tmp_path,
        env=env,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "SAW from-dotenv" in proc.stdout, proc.stdout
