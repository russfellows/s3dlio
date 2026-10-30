# python/tests/conftest.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
"""Make skips fail on demand.

With ``S3DLIO_REQUIRE_ODIRECT_TESTS=1`` (or ``S3DLIO_FAIL_ON_SKIP=1``) a test that would be SKIPPED
is reported as FAILED instead.  CI sets it for the caller-buffer ``get_into`` / typed ``head``
tests (``test_get_into_file_direct_295.py``, ``test_caller_buffer_get_into.py``): a scratch
directory that cannot do O_DIRECT, a missing numpy or pytest-asyncio, or a wheel without the
native S3 path must break the build, not quietly turn the suite into a no-op.  Unset (the
default for a developer machine), skips behave as usual.
"""

import os

import pytest


def _fail_on_skip():
    for name in ("S3DLIO_REQUIRE_ODIRECT_TESTS", "S3DLIO_FAIL_ON_SKIP"):
        if os.environ.get(name, "").lower() in ("1", "true", "yes"):
            return True
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and _fail_on_skip():
        reason = (
            report.longrepr[2]
            if isinstance(report.longrepr, tuple)
            else str(report.longrepr)
        )
        report.outcome = "failed"
        report.longrepr = (
            "a skipped test is a failure when S3DLIO_REQUIRE_ODIRECT_TESTS / "
            f"S3DLIO_FAIL_ON_SKIP is set: {reason}"
        )
