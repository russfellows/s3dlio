"""If the native module fails to import, the caller sees THAT error (issue #243 review).

``s3dlio/__init__.py`` used to import the optional torch wrapper inside ``try/except
ImportError`` before the native module, and the wrapper imports the native module itself.
A failure of the native module was therefore swallowed there, and the second import further
down hit PyO3's "can only be initialized once per process" error instead, hiding the real
cause.

The failure is simulated: a copy of the installed package's Python files, with a stand-in
``_pymod.py`` that raises "simulated native failure" the first time it is imported and a
PyO3-style "initialized once" error on any later attempt, and a stand-in ``torch`` so the
torch wrapper is imported too.

Run it against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_python_import_failure_243.py``.
"""

import importlib.util
import os
import shutil
import subprocess
import sys

FAKE_PYMOD = """
import builtins

if getattr(builtins, "_fake_pymod_imported", False):
    raise ImportError(
        "PyO3 modules compiled for CPython 3.8 or older may only be initialized once per "
        "interpreter process"
    )
builtins._fake_pymod_imported = True
raise ImportError("simulated native failure")
"""

FAKE_TORCH_DATA = """
class Dataset:
    pass


class IterableDataset:
    pass


def get_worker_info():
    return None
"""


def build_simulation(tmp_path):
    spec = importlib.util.find_spec("s3dlio")
    assert spec and spec.submodule_search_locations, "s3dlio is not installed"
    installed = spec.submodule_search_locations[0]
    root = tmp_path / "sim"
    pkg = root / "s3dlio"
    shutil.copytree(
        installed,
        pkg,
        ignore=shutil.ignore_patterns("_pymod*", "__pycache__", "*.so", "*.pyd"),
    )
    (pkg / "_pymod.py").write_text(FAKE_PYMOD)
    torch = root / "torch"
    (torch / "utils" / "data").mkdir(parents=True)
    (torch / "__init__.py").write_text("")
    (torch / "utils" / "__init__.py").write_text("")
    (torch / "utils" / "data" / "__init__.py").write_text(FAKE_TORCH_DATA)
    return root


def test_the_original_native_error_reaches_the_caller(tmp_path):
    root = build_simulation(tmp_path)
    code = (
        "try:\n"
        "    import s3dlio\n"
        "    print('IMPORTED')\n"
        "except BaseException as e:\n"
        "    print('ERR ' + type(e).__name__ + ': ' + str(e))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(root)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=120,
        check=False,
    )
    out = proc.stdout + proc.stderr
    assert "IMPORTED" not in out, out
    assert "simulated native failure" in out, out
    assert "initialized once" not in out, out
