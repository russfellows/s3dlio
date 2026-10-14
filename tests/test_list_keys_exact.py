"""s3dlio issue #198: the Python list helpers use keys exactly as written (the #172 rule).

Since #172 every S3 operation uses the key exactly as the URI spells it (``s3://b//k`` is
the key ``/k``). ``s3dlio.list_keys`` still normalised: it ``lstrip("/")``-ed the listed
path and the relative key, so an object whose key begins with ``/`` or contains ``//``
came back under a different name than the object has, and re-joining the returned key
to the listed prefix addressed another object.

``slow_s3_server.py`` (own process, no delay) lists, for any prefix ``P``, the keys
``P+"bar"``, ``P+"other/deep.dat"``, ``P+"/lead"`` and ``P+"x//y"``: the last two have an
empty segment right after the prefix and inside the relative key. Each case asserts

* the prefix the library SENT, exactly (the ListObjectsV2 ``prefix`` parameter);
* ``list_full_uris``: the URIs returned carry the real keys;
* ``list_keys``: the relative key is the real key minus the listed prefix, nothing more,
  so ``prefix_uri + key`` is the URI of the object listed.

Run against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_list_keys_exact.py``.
"""

import json
import os
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

HERE = Path(__file__).resolve().parent
RELATIVE_KEYS = ["bar", "other/deep.dat", "/lead", "x//y"]

# (listed URI, the key prefix the request must carry)
CASES = [
    ("s3://bucket/p/", "p/"),
    ("s3://bucket//p/", "/p/"),
    ("s3://bucket/a//b/", "a//b/"),
    ("s3://bucket/", ""),
    ("s3://bucket//", "/"),
]


@pytest.fixture
def server():
    proc = subprocess.Popen(
        [sys.executable, "-I", str(HERE / "slow_s3_server.py"), "0"],
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


def list_prefixes_sent(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/__stats", timeout=30) as resp:
        requests = json.load(resp)["requests"]
    return [
        parse_qs(urlsplit(target).query, keep_blank_values=True).get("prefix", [""])[0]
        for method, target in requests
        if method == "GET" and "list-type" in target
    ]


def run_listing(port, uri):
    code = textwrap.dedent(
        """
        import json, sys, s3dlio
        uri = sys.argv[1]
        print(json.dumps({"keys": s3dlio.list_keys(uri), "uris": s3dlio.list_full_uris(uri)}))
        """
    )
    env = dict(os.environ)
    env.update(
        AWS_ENDPOINT_URL=f"http://127.0.0.1:{port}",
        AWS_ACCESS_KEY_ID="test-key",
        AWS_SECRET_ACCESS_KEY="test-secret",
        AWS_REGION="us-east-1",
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, uri],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("uri,key_prefix", CASES)
def test_listing_uses_the_prefix_and_returns_the_keys_exactly(server, uri, key_prefix):
    result = run_listing(server, uri)

    # Every listing (list_keys and list_full_uris each list once) sent the exact prefix.
    assert list_prefixes_sent(server) == [key_prefix, key_prefix]

    # The listing is sorted by URI. The URIs carry the real keys: bucket + "/" + the key the
    # server holds. The relative key is the real key minus the listed prefix, so re-joining
    # addresses the same object.
    expected = sorted(("s3://bucket/" + key_prefix + rel, rel) for rel in RELATIVE_KEYS)
    assert result["uris"] == [full for full, _ in expected]
    assert result["keys"] == [rel for _, rel in expected]
    assert [uri + key for key in result["keys"]] == result["uris"]
