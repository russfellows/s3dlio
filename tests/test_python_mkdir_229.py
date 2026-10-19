"""Python ``s3dlio.mkdir()`` and the DLIO adapters' ``create_node``: s3dlio issue #229.

``mkdir`` used to work only on ``file://``; on ``direct://`` and ``s3://`` it raised
"mkdir not implemented for this backend", and the DLIO adapters' ``create_node`` (which
propagates that error) failed on every backend but ``file://``. The contract now:

* ``direct://`` is the ``file://`` behaviour: the directory (and its parents) is created.
* ``s3://`` (and ``az://``, ``gs://``) has no directories: ``mkdir`` returns ``None``,
  sends NO request and writes no marker object. The recording mock below keeps real
  state and logs every request, so each test asserts both "no request" and "every seeded
  key is still there".

Every scenario runs in a FRESH interpreter (the S3 client is process-global). Run it
against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_python_mkdir_229.py``.
Against a wheel without the fix the tests fail: that is the point of them.
"""

import json
import textwrap

import pytest

from test_oplog_lifecycle import run_scenario
from test_python_delete_contract_212 import MockS3, env_for

SEEDED = ["dir/a", "dir/b/c", "dir2/x", "dirx", "di"]

# The same stand-in for dlio_benchmark that python/tests uses (it is not a dependency).
FAKE_DLIO = """
    import sys, types
    def install():
        m = {n: types.ModuleType(n) for n in (
            "dlio_benchmark", "dlio_benchmark.common", "dlio_benchmark.common.constants",
            "dlio_benchmark.common.enumerations", "dlio_benchmark.storage",
            "dlio_benchmark.storage.storage_handler", "dlio_benchmark.storage.s3_storage",
            "dlio_benchmark.utils", "dlio_benchmark.utils.utility")}
        m["dlio_benchmark.common.constants"].MODULE_STORAGE = "storage"
        class NamespaceType: FLAT = "flat"
        class MetadataType: FILE = "file"; DIRECTORY = "directory"
        m["dlio_benchmark.common.enumerations"].NamespaceType = NamespaceType
        m["dlio_benchmark.common.enumerations"].MetadataType = MetadataType
        class Namespace:
            def __init__(self, name, ns_type): self.name, self.ns_type = name, ns_type
        class DataStorage:
            def __init__(self, framework=None):
                self.framework = framework
                self._args = types.SimpleNamespace()
        m["dlio_benchmark.storage.storage_handler"].DataStorage = DataStorage
        m["dlio_benchmark.storage.storage_handler"].Namespace = Namespace
        class S3Storage(DataStorage): pass
        m["dlio_benchmark.storage.s3_storage"].S3Storage = S3Storage
        class _Null:
            def __call__(self, fn): return fn
        class Profile:
            def __init__(self, module): self.log = _Null(); self.log_init = _Null()
        m["dlio_benchmark.utils.utility"].Profile = Profile
        sys.modules.update(m)
    install()
"""

CHILD_MKDIR = FAKE_DLIO + """
    import asyncio, json
    out = {}
    uri = sys.argv[1]
    out["mkdir"] = repr(s3dlio.mkdir(uri))
    out["mkdir_again"] = repr(s3dlio.mkdir(uri))
    async def go():
        return await s3dlio.mkdir_async(uri)
    out["mkdir_async"] = repr(asyncio.run(go()))

    from s3dlio.integrations.dlio import s3dlio_storage, s3_torch_storage
    st = object.__new__(s3dlio_storage.S3dlioStorage)
    st.prefix = sys.argv[2]
    out["create_node"] = st.create_node(sys.argv[3])
    tt = object.__new__(s3_torch_storage.S3PyTorchConnectorStorage)
    out["create_node_torch"] = tt.create_node(sys.argv[4])
    print("RESULT " + json.dumps(out))
"""


@pytest.fixture
def mock():
    m = MockS3(SEEDED)
    yield m
    m.close()


def run_child(*args, env=None):
    proc = run_scenario(textwrap.dedent(CHILD_MKDIR), *args, env=env)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"child failed: rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return json.loads(lines[-1][len("RESULT "):])


def test_direct_mkdir_and_create_node_create_real_directories(tmp_path):
    root = tmp_path / "root"
    out = run_child(
        f"direct://{root}/a/b/c",
        f"direct://{root}",
        "via_adapter/x",
        f"direct://{root}/via_torch/y",
    )
    assert out["mkdir"] == "None" and out["mkdir_again"] == "None"
    assert out["mkdir_async"] == "None"
    assert out["create_node"] is True and out["create_node_torch"] is True
    assert (root / "a/b/c").is_dir()
    assert (root / "via_adapter/x").is_dir()
    assert (root / "via_torch/y").is_dir()


def test_s3_mkdir_and_create_node_succeed_and_send_zero_requests(mock):
    out = run_child(
        "s3://bkt/dir",
        "s3://bkt",
        "dir/",
        "s3://bkt/dir/",
        env=env_for(mock),
    )
    assert out["mkdir"] == "None" and out["mkdir_again"] == "None"
    assert out["mkdir_async"] == "None"
    assert out["create_node"] is True and out["create_node_torch"] is True
    assert mock.render() == "", f"mkdir sent requests:\n    {mock.render()}"
    assert sorted(mock.keys) == sorted(SEEDED), "mkdir changed the stored keys"
