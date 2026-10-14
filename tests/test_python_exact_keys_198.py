"""Python adapters keep the object key exactly: s3dlio issue #198.

The rule (the core's since #172): a URI's key is used exactly as written. Of
``s3://b//k`` only the ONE ``/`` that separates the bucket from the key is
structural, so the object is ``/k``; ``/x`` and ``x``, or ``a//b`` and ``a/b``,
are different objects. The adapters above the core (the DLIO storage classes
and the s3torchconnector compat layer) used to ``lstrip('/')`` keys, so for a
key that begins with ``/`` they addressed or reported a different object.

Every test drives the real ``s3dlio`` extension module in a FRESH interpreter
(the S3 client is process-global) against a recording in-process mock S3
server (the one from ``test_python_delete_contract_212``) and asserts the exact
keys the adapters put on the wire. The DLIO classes import ``dlio_benchmark``,
which is not a dependency of this repo, so the child interpreter fakes just
enough of its import surface.

Run against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_python_exact_keys_198.py``.
"""

import json
import textwrap

import pytest

from test_oplog_lifecycle import run_scenario
from test_python_delete_contract_212 import MockS3, env_for

# Keys that are legal S3 keys and name three different objects.
KEYS = ["/x", "a//b", "x"]


@pytest.fixture
def mock_factory():
    mocks = []

    def make(keys=()):
        m = MockS3(keys)
        mocks.append(m)
        return m

    yield make
    for m in mocks:
        m.close()


def keys_requested(mock, method):
    """The object key of every `method` request that names one, in order."""
    return [
        r["path"].lstrip("/").partition("/")[2]
        for r in mock.requests(method)
        if "list-type" not in r["query"]
    ]


def keys_on_the_wire(mock, method):
    """The DISTINCT object keys `method` requests named (the core may repeat a HEAD)."""
    return sorted(set(keys_requested(mock, method)))


def list_prefixes(mock):
    return [r["query"].get("prefix", "") for r in mock.lists()]


def run_child(mock, body, *args):
    proc = run_scenario(textwrap.dedent(body), *args, env=env_for(mock))
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"child failed: rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return json.loads(lines[-1][len("RESULT ") :])


# ----------------------------------------------------------------------------
# A fake `dlio_benchmark`, installed in the child before the DLIO classes import
# ----------------------------------------------------------------------------

FAKE_DLIO = """
    import json, sys, types

    def install_fake_dlio_benchmark():
        def mod(name):
            m = types.ModuleType(name)
            sys.modules[name] = m
            return m

        mod("dlio_benchmark")
        mod("dlio_benchmark.common")
        mod("dlio_benchmark.common.constants").MODULE_STORAGE = "storage"
        enums = mod("dlio_benchmark.common.enumerations")
        enums.NamespaceType = types.SimpleNamespace(FLAT="flat")
        enums.MetadataType = types.SimpleNamespace(FILE="file", DIRECTORY="directory")
        mod("dlio_benchmark.storage")
        handler = mod("dlio_benchmark.storage.storage_handler")

        class Namespace:
            def __init__(self, name, ns_type):
                self.name = name

        class DataStorage:
            def __init__(self, framework=None):
                self._args = types.SimpleNamespace()

        handler.Namespace = Namespace
        handler.DataStorage = DataStorage
        mod("dlio_benchmark.storage.s3_storage").S3Storage = DataStorage
        mod("dlio_benchmark.utils")

        class Passthrough:
            def __call__(self, fn):
                return fn

        class Profile:
            def __init__(self, module):
                self.log = Passthrough()
                self.log_init = Passthrough()

        mod("dlio_benchmark.utils.utility").Profile = Profile

    install_fake_dlio_benchmark()
"""

# The compat module imports torch.utils.data at import time; torch is an optional extra
# of s3dlio and not needed here, so a minimal stand-in is installed when it is absent.
FAKE_TORCH = """
    import sys, types

    try:
        import torch.utils.data
    except ImportError:
        torch = types.ModuleType("torch")
        utils = types.ModuleType("torch.utils")
        data = types.ModuleType("torch.utils.data")
        data.Dataset = type("Dataset", (), {})
        data.IterableDataset = type("IterableDataset", (), {})
        torch.utils = utils
        utils.data = data
        sys.modules.update(
            {"torch": torch, "torch.utils": utils, "torch.utils.data": data}
        )
"""


def install_fake_torch_in_process():
    exec(textwrap.dedent(FAKE_TORCH), {})


# Run a list of [method, args...] calls on `S3dlioStorage(namespace)`.
DLIO_STORAGE_CHILD = (
    FAKE_DLIO
    + """
    from s3dlio.integrations.dlio.s3dlio_storage import S3dlioStorage

    spec = json.loads(sys.argv[1])
    storage = S3dlioStorage(spec["namespace"])
    out = []
    for name, *args in spec["calls"]:
        try:
            if name == "put_data":
                args[1] = args[1].encode()
            value = getattr(storage, name)(*args)
            if name == "get_data":
                value = len(bytes(value))
            out.append({"call": name, "value": value})
        except BaseException as e:
            out.append({"call": name, "exc": type(e).__name__, "msg": str(e)})
    print("RESULT " + json.dumps(out, default=str))
"""
)

# `S3PyTorchConnectorStorage.walk_node(uri)`.
DLIO_TORCH_CHILD = (
    FAKE_DLIO
    + """
    from s3dlio.integrations.dlio.s3_torch_storage import S3PyTorchConnectorStorage

    storage = S3PyTorchConnectorStorage("s3://bkt")
    print("RESULT " + json.dumps(storage.walk_node(sys.argv[1])))
"""
)


def storage_calls(mock, namespace, *calls):
    return run_child(
        mock,
        DLIO_STORAGE_CHILD,
        json.dumps({"namespace": namespace, "calls": list(calls)}),
    )


# ----------------------------------------------------------------------------
# 1. DLIO s3dlio_storage: put/get/delete/exists address the exact key
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize(
    "namespace,base",
    [
        ("s3://bkt", ""),
        ("s3://bkt/", ""),
        ("s3://bkt/base", "base/"),
        ("s3://bkt/base/", "base/"),
    ],
)
def test_dlio_storage_addresses_the_exact_key(mock_factory, namespace, base, key):
    """An id is joined to the namespace by ONE separator; the id itself is untouched."""
    mock = mock_factory([base + key])
    out = storage_calls(
        mock,
        namespace,
        ["put_data", key, "payload"],
        ["get_data", key],
        ["isfile", key],
        ["delete_node", key],
    )
    assert [o.get("exc") for o in out] == [None] * 4, out
    expected = base + key
    assert keys_on_the_wire(mock, "PUT") == [expected], mock.render()
    assert keys_on_the_wire(mock, "GET") == [expected], mock.render()
    assert keys_on_the_wire(mock, "HEAD") == [expected], mock.render()
    assert mock.sent_delete_keys() == [expected], mock.render()


def test_dlio_storage_keys_with_and_without_slash_are_different_objects(mock_factory):
    """Writing `/x` and `x` makes two objects, not one overwritten twice."""
    mock = mock_factory()
    storage_calls(
        mock,
        "s3://bkt/base",
        ["put_data", "/x", "slash"],
        ["put_data", "x", "plain"],
        ["put_data", "a//b", "double"],
        ["put_data", "a/b", "single"],
    )
    assert mock.keys == {"base//x", "base/x", "base/a//b", "base/a/b"}, sorted(
        mock.keys
    )


def test_dlio_storage_full_uris_pass_through(mock_factory):
    mock = mock_factory()
    storage_calls(mock, "s3://bkt/base", ["put_data", "s3://bkt//x", "d"])
    assert mock.keys == {"/x"}, sorted(mock.keys)


def test_dlio_storage_namespace_with_trailing_empty_segment_is_exact(mock_factory):
    """`s3://bkt/p//` is the base `p//`: one trailing `/` is the separator, the rest is the key."""
    mock = mock_factory()
    storage_calls(mock, "s3://bkt/p//", ["put_data", "x", "d"])
    assert mock.keys == {"p//x"}, sorted(mock.keys)


# ----------------------------------------------------------------------------
# 2. DLIO s3dlio_storage: walk_node lists relative names that re-address the object
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "namespace,prefix,seeded,names",
    [
        ("s3://bkt/base", "base/", ["base/x", "base/y", "base2/z"], ["x", "y"]),
        ("s3://bkt/base/", "base/", ["base/x", "base/y", "base2/z"], ["x", "y"]),
        # the namespace itself ends in an empty segment: the base is `base//`
        ("s3://bkt/base//", "base//", ["base//x", "base//y", "base/z"], ["x", "y"]),
        ("s3://bkt//", "/", ["/x", "/y", "z"], ["x", "y"]),
    ],
)
def test_dlio_storage_walk_node_lists_the_exact_prefix_and_names_round_trip(
    mock_factory, namespace, prefix, seeded, names
):
    mock = mock_factory(seeded)
    out = storage_calls(mock, namespace, ["walk_node", ""])
    assert not out[0].get("exc"), out
    assert sorted(out[0]["value"]) == names, (out, mock.render())
    assert list_prefixes(mock) == [prefix], mock.render()

    # Re-joining each listed name must address exactly the object that was listed.
    mock2 = mock_factory(seeded)
    heads = storage_calls(mock2, namespace, *[["isfile", n] for n in names])
    assert [h.get("exc") for h in heads] == [None] * len(names), heads
    assert keys_on_the_wire(mock2, "HEAD") == sorted(prefix + n for n in names), (
        mock2.render()
    )


# ----------------------------------------------------------------------------
# 3. DLIO s3_torch_storage.walk_node: the listing prefix is the exact key prefix
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri,prefix,seeded,expected",
    [
        ("s3://bkt//d/", "/d/", ["/d/x", "/d/y", "d/other"], ["x", "y"]),
        ("s3://bkt/d//e/", "d//e/", ["d//e/x", "d/e/y"], ["x"]),
        ("s3://bkt/d", "d/", ["d/x", "dd/y"], ["x"]),
        ("s3://bkt/d/", "d/", ["d/x", "/d/z"], ["x"]),
        ("s3://bkt//", "/", ["/x", "x"], ["x"]),
    ],
)
def test_dlio_torch_walk_node_lists_the_exact_prefix(
    mock_factory, uri, prefix, seeded, expected
):
    mock = mock_factory(seeded)
    names = run_child(mock, DLIO_TORCH_CHILD, uri)
    assert list_prefixes(mock) == [prefix], mock.render()
    assert sorted(names) == sorted(expected), (names, mock.render())


# ----------------------------------------------------------------------------
# 4. s3torchconnector compat: S3Item.key and the S3Client helpers
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri,bucket,key",
    [
        ("s3://bkt//k", "bkt", "/k"),
        ("s3://bkt/a//b", "bkt", "a//b"),
        ("s3://bkt/x", "bkt", "x"),
        ("s3://bkt/dir/x", "bkt", "dir/x"),
        ("s3://bkt///k", "bkt", "//k"),
        ("gs://bkt//k", "bkt", "/k"),
        ("gs://bkt/a//b", "bkt", "a//b"),
        ("az://acct/cont//k", "acct/cont", "/k"),
        ("az://acct/cont/a//b", "acct/cont", "a//b"),
        ("az://acct/cont/x", "acct/cont", "x"),
    ],
)
def test_compat_s3item_reports_the_exact_key(uri, bucket, key):
    install_fake_torch_in_process()
    from s3dlio.compat.s3torchconnector import S3Item

    item = S3Item(uri, b"")
    assert (item.bucket, item.key) == (bucket, key)


def test_compat_s3item_matches_the_upstream_parse_s3_uri_contract():
    """AWS's s3torchconnector `parse_s3_uri` (awslabs/s3-connector-for-pytorch,
    s3torchconnector/src/s3torchconnector/_s3dataset_common.py) does
    ``uri[len("s3://"):].split("/", maxsplit=1)``: bucket and key, nothing stripped, so
    `s3://b//k` is bucket `b`, key `/k`. Mirror it exactly."""
    install_fake_torch_in_process()
    from s3dlio.compat.s3torchconnector import S3Item

    def upstream(uri):
        rest = uri[len("s3://") :]
        parts = rest.split("/", maxsplit=1)
        return (parts[0], "") if len(parts) == 1 else (parts[0], parts[1])

    for uri in [
        "s3://b//k",
        "s3://b/a//b",
        "s3://b/x",
        "s3://b/",
        "s3://b",
        "s3://b///",
    ]:
        item = S3Item(uri, b"")
        assert (item.bucket, item.key) == upstream(uri), uri


COMPAT_CHILD = (
    FAKE_TORCH
    + """
    import json
    from s3dlio.compat.s3torchconnector import S3Client

    bucket, op, key = sys.argv[1:4]
    client = S3Client()
    if op == "put":
        w = client.put_object(bucket, key)
        w.write(b"data")
        w.close()
    elif op == "get":
        client.get_object(bucket, key).read()
    elif op == "list":
        for result in client.list_objects(bucket, key):
            pass
    print("RESULT {}")
"""
)


@pytest.mark.parametrize("key", KEYS)
def test_compat_client_put_and_get_address_the_exact_key(mock_factory, key):
    mock = mock_factory([key])
    run_child(mock, COMPAT_CHILD, "bkt", "put", key)
    run_child(mock, COMPAT_CHILD, "bkt", "get", key)
    assert keys_on_the_wire(mock, "PUT") == [key], mock.render()
    assert keys_on_the_wire(mock, "GET") == [key], mock.render()


@pytest.mark.parametrize(
    "bucket,prefix,wanted",
    [
        ("bkt", "/d/", "/d/"),
        ("bkt", "a//b/", "a//b/"),
        ("bkt", "d/", "d/"),
        ("bkt", "", ""),
        ("bkt", "s3://bkt//d", "/d/"),
        ("bkt", "s3://bkt//", "/"),
        ("bkt", "s3://bkt/a//b/", "a//b/"),
        ("bkt", "s3://bkt/d", "d/"),
    ],
)
def test_compat_list_objects_lists_the_exact_prefix(
    mock_factory, bucket, prefix, wanted
):
    mock = mock_factory(["/d/x", "a//b/y", "d/z"])
    run_child(mock, COMPAT_CHILD, bucket, "list", prefix)
    assert list_prefixes(mock) == [wanted], mock.render()
