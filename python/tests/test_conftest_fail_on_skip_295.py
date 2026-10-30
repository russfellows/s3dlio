# python/tests/test_conftest_fail_on_skip_295.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
"""The skip-to-failure switch of ``conftest.py`` (CI relies on it), tested in a child pytest."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")

SAMPLES = {
    "test_runtime_skip.py": "import pytest\n\ndef test_a():\n    pytest.skip('runtime skip')\n",
    "test_collect_skip.py": (
        "import pytest\n"
        "pytest.importorskip('no_such_module_for_s3dlio_295')\n\n"
        "def test_b():\n    pass\n"
    ),
    "test_xfail.py": "import pytest\n\n@pytest.mark.xfail(reason='known')\ndef test_c():\n    assert False\n",
    "test_marked_skip.py": "import pytest\n\n@pytest.mark.skip(reason='marked')\ndef test_d():\n    pass\n",
    "test_ok.py": "def test_e():\n    pass\n",
}


def _run(tmp_path, sample, env_value):
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / sample).write_text(SAMPLES[sample])
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("S3DLIO_FAIL", "S3DLIO_REQUIRE", "PYTEST"))
    }
    if env_value is not None:
        env["S3DLIO_FAIL_ON_SKIP"] = env_value
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", sample],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    "sample", ["test_runtime_skip.py", "test_collect_skip.py", "test_marked_skip.py"]
)
def test_skips_fail_when_requested(tmp_path, sample):
    r = _run(tmp_path, sample, "1")
    assert r.returncode != 0, r.stdout + r.stderr
    assert "S3DLIO_FAIL_ON_SKIP" in r.stdout, r.stdout


@pytest.mark.parametrize(
    "sample", ["test_runtime_skip.py", "test_collect_skip.py", "test_marked_skip.py"]
)
def test_skips_stay_skips_by_default(tmp_path, sample):
    r = _run(tmp_path, sample, None)
    assert r.returncode in (0, 5), r.stdout + r.stderr


def test_xfail_is_not_a_failure_when_requested(tmp_path):
    r = _run(tmp_path, "test_xfail.py", "1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 xfailed" in r.stdout, r.stdout


def test_passing_test_passes_when_requested(tmp_path):
    r = _run(tmp_path, "test_ok.py", "1")
    assert r.returncode == 0, r.stdout + r.stderr
