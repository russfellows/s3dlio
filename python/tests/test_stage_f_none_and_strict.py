"""Stage F regressions against the INSTALLED wheel: #271, #272 and #277.

* #271: the async APIs that used to resolve to the Rust unit value (a Python ``()``) resolve
  to ``None``: ``put_bytes_async`` (with and without ``borrow=True``), ``put_many_async`` and
  the ``MultiEndpointStore`` methods ``put``, ``delete``, ``put_to_endpoint``,
  ``delete_from_endpoint`` and ``put_all_endpoints``.  ``put_many`` (synchronous) returns
  ``None`` too.
* #272: ``put_many`` and ``put_many_async`` use the same strict payload conversion as
  ``put_bytes``: bytes, buffer-protocol objects (bytearray, memoryview, numpy arrays, ...) and
  objects with ``__bytes__()`` are accepted; an int, a bool and an arbitrary iterable (a list of
  ints) are a ``TypeError`` raised before any request is sent.
* #277: ``get_into_async`` refuses a pointer-format destination (a ctypes array of
  ``c_void_p`` has the item format ``P``) with wording about a read destination, not an upload.

What this module needs
----------------------
No network and no environment variable.  Everything runs against ``file://`` URIs under
pytest's ``tmp_path`` (``MultiEndpointStore`` accepts ``file://`` endpoints).  The #277 check
runs in a fresh child interpreter (``get_into_async`` is not supported on ``file://``, so it
needs an ``s3://`` URI, and the child gets a dead local endpoint and fake credentials so
that no real service can be reached; the refusal happens before any I/O).  Skips: numpy
missing (the numpy cases), pytest-asyncio missing (the async tests), a wheel without
``get_into`` (the #277 test).

Run:

    uv run --no-project --isolated --with <wheel> --with pytest --with pytest-asyncio \
        --with numpy python -m pytest python/tests/test_stage_f_none_and_strict.py -v
"""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest
import s3dlio

needs_asyncio_plugin = pytest.mark.skipif(
    importlib.util.find_spec("pytest_asyncio") is None,
    reason="pytest-asyncio is not installed",
)

CHILD_TIMEOUT = 90


class DunderBytes:
    """An object that is neither a buffer nor a sequence, only convertible with bytes()."""

    def __bytes__(self):
        return b"xyz"


def file_uri(directory, name):
    return f"file://{directory}/{name}"


def bad_payloads():
    """Payloads that must be refused with a TypeError, by name."""
    return {
        "list_of_ints": [1, 2, 3],
        "int": 42,
        "bool": True,
        "tuple_of_ints": (1, 2, 3),
        "range": range(4),
        "generator": (i for i in range(3)),
    }


def good_payloads():
    """(payload, expected bytes) pairs that must be accepted, by name."""
    cases = {
        "bytes": (b"abc", b"abc"),
        "bytearray": (bytearray(b"abc"), b"abc"),
        "memoryview": (memoryview(b"abc"), b"abc"),
        "strided_memoryview": (memoryview(b"abcdef")[::2], b"ace"),
        "dunder_bytes": (DunderBytes(), b"xyz"),
    }
    try:
        import numpy as np
    except ImportError:
        return cases
    cases["numpy_uint8"] = (np.frombuffer(b"abc", dtype=np.uint8).copy(), b"abc")
    cases["numpy_uint16"] = (
        np.array([1, 2, 515], dtype="<u2"),
        np.array([1, 2, 515], dtype="<u2").tobytes(),
    )
    return cases


GOOD = good_payloads()
BAD = bad_payloads()


# ------------------------------ #271: None, not () ------------------------------


@needs_asyncio_plugin
@pytest.mark.asyncio
@pytest.mark.parametrize("borrow", [False, True])
async def test_put_bytes_async_resolves_to_none(tmp_path, borrow):
    data = bytearray(b"payload")
    result = await s3dlio.put_bytes_async(
        file_uri(tmp_path, "one.bin"), data, borrow=borrow
    )
    assert result is None, f"put_bytes_async(borrow={borrow}) resolved to {result!r}"
    assert (tmp_path / "one.bin").read_bytes() == b"payload"


@needs_asyncio_plugin
@pytest.mark.asyncio
async def test_put_many_async_resolves_to_none(tmp_path):
    items = [
        (file_uri(tmp_path, "a.bin"), b"aa"),
        (file_uri(tmp_path, "b.bin"), b"bb"),
    ]
    result = await s3dlio.put_many_async(items)
    assert result is None, f"put_many_async resolved to {result!r}"
    assert (tmp_path / "a.bin").read_bytes() == b"aa"
    assert (tmp_path / "b.bin").read_bytes() == b"bb"


def test_put_many_returns_none(tmp_path):
    items = [(file_uri(tmp_path, "s.bin"), b"ss")]
    result = s3dlio.put_many(items)
    assert result is None, f"put_many returned {result!r}"
    assert (tmp_path / "s.bin").read_bytes() == b"ss"


@needs_asyncio_plugin
@pytest.mark.asyncio
async def test_multi_endpoint_store_mutators_resolve_to_none(tmp_path):
    ep0 = tmp_path / "ep0"
    ep1 = tmp_path / "ep1"
    ep0.mkdir()
    ep1.mkdir()
    store = s3dlio.create_multi_endpoint_store(
        [f"file://{ep0}/", f"file://{ep1}/"], "round_robin"
    )

    result = await store.put(file_uri(ep0, "balanced"), b"x")
    assert result is None, f"put resolved to {result!r}"

    result = await store.put_to_endpoint(1, file_uri(ep1, "direct"), b"direct")
    assert result is None, f"put_to_endpoint resolved to {result!r}"
    assert (ep1 / "direct").read_bytes() == b"direct"

    result = await store.put_all_endpoints(file_uri(ep0, "everywhere"), b"all")
    assert result is None, f"put_all_endpoints resolved to {result!r}"
    assert (ep0 / "everywhere").read_bytes() == b"all"
    assert (ep1 / "everywhere").read_bytes() == b"all"

    result = await store.delete_from_endpoint(1, file_uri(ep1, "direct"))
    assert result is None, f"delete_from_endpoint resolved to {result!r}"
    assert not (ep1 / "direct").exists()

    result = await store.delete(file_uri(ep0, "everywhere"))
    assert result is None, f"delete resolved to {result!r}"


# ------------- #281: strict payloads for the MultiEndpointStore writes -------------

WRITE_METHODS = ["put", "put_to_endpoint", "put_all_endpoints"]


def two_file_endpoints(tmp_path):
    ep0 = tmp_path / "ep0"
    ep1 = tmp_path / "ep1"
    ep0.mkdir()
    ep1.mkdir()
    store = s3dlio.create_multi_endpoint_store(
        [f"file://{ep0}/", f"file://{ep1}/"], "round_robin"
    )
    return store, ep0, ep1


def start_write(store, method, ep0, ep1, name, payload):
    if method == "put_to_endpoint":
        return store.put_to_endpoint(1, file_uri(ep1, name), payload)
    return getattr(store, method)(file_uri(ep0, name), payload)


@needs_asyncio_plugin
@pytest.mark.asyncio
@pytest.mark.parametrize("method", WRITE_METHODS)
@pytest.mark.parametrize("kind", sorted(BAD))
async def test_multi_endpoint_writes_refuse_non_buffer_payloads(tmp_path, method, kind):
    store, ep0, ep1 = two_file_endpoints(tmp_path)
    with pytest.raises(TypeError):
        await start_write(store, method, ep0, ep1, "bad.bin", BAD[kind])
    assert not (ep0 / "bad.bin").exists()
    assert not (ep1 / "bad.bin").exists()


@needs_asyncio_plugin
@pytest.mark.asyncio
@pytest.mark.parametrize("method", WRITE_METHODS)
@pytest.mark.parametrize("kind", sorted(GOOD))
async def test_multi_endpoint_writes_accept_bytes_buffers_and_dunder_bytes(
    tmp_path, method, kind
):
    payload, expected = GOOD[kind]
    store, ep0, ep1 = two_file_endpoints(tmp_path)
    result = await start_write(store, method, ep0, ep1, "good.bin", payload)
    assert result is None
    written = [
        (ep / "good.bin").read_bytes()
        for ep in (ep0, ep1)
        if (ep / "good.bin").exists()
    ]
    assert written, "nothing was written"
    assert all(content == expected for content in written), written
    if method == "put_all_endpoints":
        assert len(written) == 2


@needs_asyncio_plugin
@pytest.mark.asyncio
async def test_multi_endpoint_put_owns_its_copy_of_a_mutable_buffer(tmp_path):
    store, ep0, ep1 = two_file_endpoints(tmp_path)
    buf = bytearray(b"before")
    awaitable = store.put_all_endpoints(file_uri(ep0, "own.bin"), buf)
    buf[:] = b"AFTER!"
    buf.extend(b"-resized")  # no BufferError: no export is held past the call
    await awaitable
    assert (ep0 / "own.bin").read_bytes() == b"before"
    assert (ep1 / "own.bin").read_bytes() == b"before"


# ------------------------ #272: strict batch payloads ------------------------


@pytest.mark.parametrize("kind", sorted(BAD))
def test_put_many_refuses_non_buffer_payloads_before_any_request(tmp_path, kind):
    items = [
        (file_uri(tmp_path, "valid.bin"), b"ok"),
        (file_uri(tmp_path, "invalid.bin"), BAD[kind]),
    ]
    with pytest.raises(TypeError):
        s3dlio.put_many(items)
    assert not (tmp_path / "invalid.bin").exists()
    assert not (tmp_path / "valid.bin").exists(), "a request was sent for a valid item"


@needs_asyncio_plugin
@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(BAD))
async def test_put_many_async_refuses_non_buffer_payloads_before_any_request(
    tmp_path, kind
):
    items = [
        (file_uri(tmp_path, "valid.bin"), b"ok"),
        (file_uri(tmp_path, "invalid.bin"), BAD[kind]),
    ]
    with pytest.raises(TypeError):
        await s3dlio.put_many_async(items)
    assert not (tmp_path / "invalid.bin").exists()
    assert not (tmp_path / "valid.bin").exists(), "a request was sent for a valid item"


@pytest.mark.parametrize("kind", sorted(GOOD))
def test_put_many_accepts_bytes_buffers_and_dunder_bytes(tmp_path, kind):
    payload, expected = GOOD[kind]
    assert s3dlio.put_many([(file_uri(tmp_path, "g.bin"), payload)]) is None
    assert (tmp_path / "g.bin").read_bytes() == expected


@needs_asyncio_plugin
@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(GOOD))
async def test_put_many_async_accepts_bytes_buffers_and_dunder_bytes(tmp_path, kind):
    payload, expected = GOOD[kind]
    result = await s3dlio.put_many_async([(file_uri(tmp_path, "g.bin"), payload)])
    assert result is None
    assert (tmp_path / "g.bin").read_bytes() == expected


@needs_asyncio_plugin
@pytest.mark.asyncio
async def test_put_many_async_owns_its_copy_of_a_mutable_buffer(tmp_path):
    """The batch may not read the caller's buffer after the call returned its awaitable."""
    buf = bytearray(b"before")
    awaitable = s3dlio.put_many_async([(file_uri(tmp_path, "own.bin"), buf)])
    buf[:] = b"AFTER!"
    buf.extend(b"-resized")  # no BufferError: no export is held past the call
    await awaitable
    assert (tmp_path / "own.bin").read_bytes() == b"before"


# ------------------- #277: a read destination is not an upload -------------------

GET_INTO_CHILD = textwrap.dedent(
    """
    import ctypes
    import sys

    import s3dlio

    dst = (ctypes.c_void_p * 4)()
    try:
        s3dlio.get_into_async("s3://stagef-bucket/key", dst, expected_len=4)
    except TypeError as exc:
        sys.stdout.write(str(exc))
        sys.exit(0)
    sys.stderr.write("the pointer destination was not refused with a TypeError")
    sys.exit(3)
    """
)


@pytest.mark.skipif(
    not (
        hasattr(s3dlio, "object_capabilities")
        and s3dlio.object_capabilities("s3://stagef-bucket/key")["get_into"]
    ),
    reason="this wheel has no native S3 get_into",
)
def test_get_into_pointer_destination_refusal_does_not_say_upload():
    env = dict(os.environ)
    env.update(
        AWS_ENDPOINT_URL="http://127.0.0.1:9",
        AWS_ACCESS_KEY_ID="stagef",
        AWS_SECRET_ACCESS_KEY="stagef",
        AWS_REGION="us-east-1",
    )
    done = subprocess.run(
        [sys.executable, "-c", GET_INTO_CHILD],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT,
    )
    assert done.returncode == 0, f"child failed: {done.stdout!r} {done.stderr!r}"
    message = done.stdout
    assert "upload" not in message, f"a read destination refusal says upload: {message}"
    assert "read destination" in message, message
    assert "P" in message.split("format")[-1], message


def test_put_bytes_pointer_buffer_refusal_still_says_upload():
    import ctypes

    with pytest.raises(TypeError, match="cannot upload a buffer"):
        s3dlio.put_bytes("file:///nonexistent/never-written", (ctypes.c_void_p * 4)())


# ------------- #282: every pointer-bearing buffer format is refused -------------


def pointer_arrays():
    """Real ctypes buffers whose items are pointers; CPython reports the format codes
    ``&<i`` (POINTER), ``<z`` (c_char_p), ``<Z`` (c_wchar_p), ``<O``, ``<P``, ``X{}`` and
    ``T{...}`` for them."""
    import ctypes

    class WithPointer(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int), ("p", ctypes.POINTER(ctypes.c_int))]

    class WithText(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int), ("s", ctypes.c_char_p)]

    return {
        "POINTER": (ctypes.POINTER(ctypes.c_int) * 4)(),
        "c_char_p": (ctypes.c_char_p * 4)(),
        "c_wchar_p": (ctypes.c_wchar_p * 4)(),
        "py_object": (ctypes.py_object * 2)(),
        "c_void_p": (ctypes.c_void_p * 4)(),
        "function_pointer": (ctypes.CFUNCTYPE(None) * 2)(),
        "struct_with_pointer": (WithPointer * 2)(),
        "struct_with_char_p": (WithText * 2)(),
    }


def plain_ctypes_arrays():
    """Real ctypes buffers without pointers: they must stay valid payloads."""
    import ctypes

    class Plain(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int), ("x", ctypes.c_double)]

    return {
        "c_ubyte": (ctypes.c_ubyte * 8)(),
        "c_short": (ctypes.c_short * 8)(),
        "c_double": (ctypes.c_double * 8)(),
        "c_bool": (ctypes.c_bool * 8)(),
        "c_wchar": (ctypes.c_wchar * 4)(),
        "plain_struct": (Plain * 2)(),
    }


POINTERS = pointer_arrays()
PLAIN = plain_ctypes_arrays()


@pytest.mark.parametrize("kind", sorted(POINTERS))
def test_put_bytes_refuses_every_pointer_bearing_buffer(tmp_path, kind):
    with pytest.raises(TypeError, match="cannot upload a buffer"):
        s3dlio.put_bytes(file_uri(tmp_path, "p.bin"), POINTERS[kind])
    assert not (tmp_path / "p.bin").exists()


@needs_asyncio_plugin
@pytest.mark.asyncio
@pytest.mark.parametrize("borrow", [False, True])
@pytest.mark.parametrize("kind", sorted(POINTERS))
async def test_put_bytes_async_refuses_every_pointer_bearing_buffer(
    tmp_path, kind, borrow
):
    with pytest.raises(TypeError, match="cannot upload a buffer"):
        await s3dlio.put_bytes_async(
            file_uri(tmp_path, "p.bin"), POINTERS[kind], borrow=borrow
        )
    assert not (tmp_path / "p.bin").exists()


@pytest.mark.parametrize("kind", sorted(POINTERS))
def test_put_many_refuses_every_pointer_bearing_buffer(tmp_path, kind):
    with pytest.raises(TypeError, match="cannot upload a buffer"):
        s3dlio.put_many([(file_uri(tmp_path, "p.bin"), POINTERS[kind])])
    assert not (tmp_path / "p.bin").exists()


@needs_asyncio_plugin
@pytest.mark.asyncio
@pytest.mark.parametrize("method", WRITE_METHODS)
@pytest.mark.parametrize("kind", sorted(POINTERS))
async def test_multi_endpoint_writes_refuse_every_pointer_bearing_buffer(
    tmp_path, method, kind
):
    store, ep0, ep1 = two_file_endpoints(tmp_path)
    with pytest.raises(TypeError, match="cannot upload a buffer"):
        await start_write(store, method, ep0, ep1, "p.bin", POINTERS[kind])
    assert not (ep0 / "p.bin").exists()
    assert not (ep1 / "p.bin").exists()


@pytest.mark.parametrize("kind", sorted(PLAIN))
def test_pointer_free_ctypes_buffers_are_still_accepted(tmp_path, kind):
    s3dlio.put_bytes(file_uri(tmp_path, "ok.bin"), PLAIN[kind])
    assert (tmp_path / "ok.bin").stat().st_size == len(bytes(PLAIN[kind]))


def test_complex_numpy_arrays_are_not_mistaken_for_pointers(tmp_path):
    np = pytest.importorskip("numpy")
    for dtype in ("complex64", "complex128"):
        data = np.arange(4, dtype=dtype)
        s3dlio.put_bytes(file_uri(tmp_path, f"{dtype}.bin"), data)
        assert (tmp_path / f"{dtype}.bin").read_bytes() == data.tobytes()


# ------------------- batch errors name the failing item -------------------


def test_put_many_error_names_the_item_and_keeps_the_message(tmp_path):
    ok = file_uri(tmp_path, "ok.bin")
    bad = file_uri(tmp_path, "bad.bin")
    with pytest.raises(TypeError) as info:
        s3dlio.put_many([(ok, b"ok"), (bad, 42)])
    message = str(info.value)
    assert "item 1" in message and bad in message, message
    assert "Cannot convert data to bytes" in message, message


def test_put_many_error_redacts_query_fragment_and_userinfo_from_the_uri(tmp_path):
    secret = "SECRETSIG123"
    plain = "https://acct.blob.core.windows.net/c/bad.bin"
    bad = f"{plain}?sv=2022&sig={secret}#frag"
    with pytest.raises(TypeError) as info:
        s3dlio.put_many([(bad, 42)])
    message = str(info.value)
    assert secret not in message and "sig=" not in message and "frag" not in message, (
        message
    )
    assert "item 0" in message and plain in message, message

    with_userinfo = "s3://AKIAUSER:PASSWORD9@127.0.0.1:1/bucket/key"
    with pytest.raises(TypeError) as info:
        s3dlio.put_many([(with_userinfo, [1, 2, 3])])
    message = str(info.value)
    assert "AKIAUSER" not in message and "PASSWORD9" not in message, message
    assert "127.0.0.1:1/bucket/key" in message, message


def test_put_many_bad_payload_type_error_keeps_the_cause(tmp_path):
    bad = file_uri(tmp_path, "bad.bin")
    with pytest.raises(TypeError) as info:
        s3dlio.put_many([(bad, 42)])
    cause = info.value.__cause__
    assert isinstance(cause, TypeError), repr(cause)
    assert cause is not info.value
    assert "item" not in str(cause), "the cause must be the original, unprefixed error"
    assert "Cannot convert data to bytes" in str(cause), str(cause)


@needs_asyncio_plugin
@pytest.mark.asyncio
async def test_put_many_async_error_names_the_item_and_keeps_the_message(tmp_path):
    ok = file_uri(tmp_path, "ok.bin")
    bad = file_uri(tmp_path, "bad.bin")
    with pytest.raises(TypeError) as info:
        await s3dlio.put_many_async([(ok, b"ok"), (bad, [1, 2, 3])])
    message = str(info.value)
    assert "item 1" in message and bad in message, message
    assert "Cannot convert data to bytes" in message, message


@needs_asyncio_plugin
@pytest.mark.asyncio
async def test_multi_endpoint_write_error_names_the_uri(tmp_path):
    store, ep0, _ = two_file_endpoints(tmp_path)
    uri = file_uri(ep0, "named.bin")
    with pytest.raises(TypeError) as info:
        await store.put(uri, 42)
    assert uri in str(info.value), str(info.value)
    assert "Cannot convert data to bytes" in str(info.value), str(info.value)


def test_put_many_error_keeps_a_question_mark_in_a_non_http_key():
    # `?` is an ordinary key character in s3:// (and gs://, az://, file://): the key is named
    # intact. Only http(s) URIs lose their query (where SAS tokens and signatures live).
    for uri in ("s3://b/what?.csv", "gs://b/a?b#c", "az://acct/c/o?x=1"):
        with pytest.raises(TypeError) as info:
            s3dlio.put_many([(uri, 42)])
        assert uri in str(info.value), str(info.value)


def test_put_many_error_drops_the_query_of_an_https_sas_uri():
    uri = "https://acct.blob.core.windows.net/c/k"
    sas = "sv=2022-11-02&sig=TOPSECRETSIG%3D&se=2030-01-01"
    with pytest.raises(TypeError) as info:
        s3dlio.put_many([(f"{uri}?{sas}", 42)])
    message = str(info.value)
    assert uri in message, message
    assert "TOPSECRETSIG" not in message and "sig=" not in message, message


@pytest.mark.parametrize(
    ("uri", "rendered"),
    [
        ("https://user:p?w@host.example/c/k?sig=TOPSECRETSIG", "https://"),
        ("https://user:p#w@host.example/c/k?sig=TOPSECRETSIG", "https://"),
        (
            "https://user:p/w@host.example/c/k?sig=TOPSECRETSIG",
            "https://host.example/c/k",
        ),
    ],
)
def test_put_many_error_hides_a_password_holding_a_delimiter(uri, rendered):
    """Redaction rule (src/redact.rs module docs; docs/PYTHON_API_GUIDE.md).

    Userinfo runs to the last ``@`` before the first ``/``. When that removed span holds ``?``
    or ``#`` it is indistinguishable from an ``@`` inside a query (``https://h?e=a@b&sig=S``),
    so the rest of the URI is treated as its query and, for http(s), removed: the URI renders
    as just ``https://`` (over-redaction, by design). One ``/`` in an http(s) password is
    recognised by the one-slash rule, so the host and path survive. No secret appears in any
    case.
    """
    with pytest.raises(TypeError) as info:
        s3dlio.put_many([(uri, 42)])
    message = str(info.value)
    assert f"item 0 ({rendered}):" in message, message
    if rendered == "https://":
        assert "host.example" not in message, message
    for secret in ("user", "p?w", "p#w", "p/w", "TOPSECRETSIG", "sig="):
        assert secret not in message, f"{secret!r} leaked: {message}"


# --------- #285: arguments documented as optional can be left out ---------


def _called_without_an_argument_error(call):
    """Run ``call``; a missing or unknown argument is the one failure that is not allowed.

    Any other outcome is fine here: a result, or an error from the backend (these calls only
    have to get past argument parsing).
    """
    try:
        return call()
    except TypeError as error:
        text = str(error)
        assert "required positional argument" not in text, text
        assert "unexpected keyword argument" not in text, text
    except (OSError, RuntimeError, ValueError):
        return None  # a backend error: the arguments were accepted
    return None


def _writer_cases(tmp_path):
    # create_s3_writer / create_azure_writer get a wrong-scheme URI: nothing leaves the process.
    return [
        ("create_s3_writer", "file:///not-an-s3-uri"),
        ("create_azure_writer", "file:///not-an-azure-uri"),
        ("create_filesystem_writer", file_uri(tmp_path, "writer.bin")),
        ("create_direct_filesystem_writer", file_uri(tmp_path, "direct.bin")),
    ]


@pytest.mark.parametrize(
    "name",
    [
        "create_s3_writer",
        "create_azure_writer",
        "create_filesystem_writer",
        "create_direct_filesystem_writer",
    ],
)
def test_writer_factories_take_options_as_an_optional_argument(tmp_path, name):
    uri = dict(_writer_cases(tmp_path))[name]
    function = getattr(s3dlio, name)
    _called_without_an_argument_error(lambda: function(uri))
    _called_without_an_argument_error(lambda: function(uri, options=None))
    if hasattr(s3dlio, "PyWriterOptions"):
        options = s3dlio.PyWriterOptions()
        _called_without_an_argument_error(lambda: function(uri, options=options))


def test_filesystem_writer_without_options_writes_a_file(tmp_path):
    writer = s3dlio.create_filesystem_writer(file_uri(tmp_path, "plain.bin"))
    assert callable(getattr(writer, "write_chunk", None)), "writer has no write_chunk"
    assert callable(getattr(writer, "finalize", None)), "writer has no finalize"
    writer.write_chunk(b"abc")
    writer.finalize()
    assert (tmp_path / "plain.bin").read_bytes() == b"abc"


def test_multi_endpoint_factories_take_strategy_as_an_optional_argument(tmp_path):
    ep0, ep1 = tmp_path / "ep0", tmp_path / "ep1"
    ep0.mkdir()
    ep1.mkdir()
    uris = [f"file://{ep0}/", f"file://{ep1}/"]
    endpoints = tmp_path / "endpoints.txt"
    endpoints.write_text("\n".join(uris) + "\n")
    template = f"file://{tmp_path}/t{{1...2}}/"

    stores = [
        s3dlio.create_multi_endpoint_store(uris),
        s3dlio.create_multi_endpoint_store(uris, strategy=None),
        s3dlio.create_multi_endpoint_store(uris, strategy="least_connections"),
        s3dlio.create_multi_endpoint_store_from_template(template),
        s3dlio.create_multi_endpoint_store_from_template(template, strategy=None),
        s3dlio.create_multi_endpoint_store_from_file(str(endpoints)),
        s3dlio.create_multi_endpoint_store_from_file(str(endpoints), strategy=None),
        s3dlio.create_multi_endpoint_store_from_file(
            str(endpoints), strategy="round_robin"
        ),
    ]
    assert all(store is not None for store in stores)


def test_multi_endpoint_store_from_env_takes_strategy_as_an_optional_argument(
    tmp_path, monkeypatch
):
    ep0, ep1 = tmp_path / "ep0", tmp_path / "ep1"
    ep0.mkdir()
    ep1.mkdir()
    monkeypatch.setenv("S3_ENDPOINT_URIS", f"file://{ep0}/,file://{ep1}/")
    assert s3dlio.create_multi_endpoint_store_from_env() is not None
    assert s3dlio.create_multi_endpoint_store_from_env(strategy=None) is not None
    assert (
        s3dlio.create_multi_endpoint_store_from_env(strategy="least_connections")
        is not None
    )


def test_writer_options_with_compression_takes_level_as_an_optional_argument():
    options = s3dlio.PyWriterOptions()
    options.with_compression("none")
    options.with_compression("zstd")
    options.with_compression("zstd", level=None)
    options.with_compression("zstd", level=5)
    options.with_compression("zstd", 7)


def _compressed_size(tmp_path, name, configure):
    # Compressible text with enough variety that different zstd levels give different sizes.
    state = 12345
    words = []
    for _ in range(60000):
        state = (state * 1103515245 + 12345) % (1 << 31)
        words.append(f"w{state % 997}")
    payload = " ".join(words).encode()
    options = s3dlio.PyWriterOptions()
    configure(options)
    writer = s3dlio.create_filesystem_writer(file_uri(tmp_path, name), options)
    writer.write_chunk(payload)
    writer.finalize()
    return (tmp_path / f"{name}.zst").stat().st_size  # the writer appends ".zst"


def test_with_compression_zstd_defaults_to_level_three(tmp_path):
    default = _compressed_size(tmp_path, "d.bin", lambda o: o.with_compression("zstd"))
    explicit = _compressed_size(
        tmp_path, "l3.bin", lambda o: o.with_compression("zstd", level=3)
    )
    best = _compressed_size(
        tmp_path, "l19.bin", lambda o: o.with_compression("zstd", level=19)
    )
    assert default == explicit, (default, explicit)
    assert best != default, "level 19 and the default level compress identically"


def test_multi_endpoint_get_range_takes_length_as_an_optional_argument(tmp_path):
    import asyncio

    store, ep0, ep1 = two_file_endpoints(tmp_path)
    for endpoint in (ep0, ep1):  # the store picks one of them for each read
        (endpoint / "range.bin").write_bytes(b"0123456789")
    uri = file_uri(ep0, "range.bin")

    async def read(**kwargs):
        view = await store.get_range(uri, 2, **kwargs)
        return bytes(view)

    assert asyncio.run(read()) == b"23456789"
    assert asyncio.run(read(length=None)) == b"23456789"
    assert asyncio.run(read(length=3)) == b"234"


def test_gcs_set_rapid_mode_takes_force_as_an_optional_argument():
    # None means auto-detect. The mode is process-global in a GCS build: put it back.
    previous = s3dlio.gcs_get_rapid_mode()
    try:
        s3dlio.gcs_set_rapid_mode()
        s3dlio.gcs_set_rapid_mode(force=None)
        s3dlio.gcs_set_rapid_mode(None)
    finally:
        s3dlio.gcs_set_rapid_mode(force=previous)
    assert s3dlio.gcs_get_rapid_mode() == previous


# ----- stub keywords: Python sees the same parameter names in every build -----


def _assert_keyword_is_accepted(function, name):
    """``function(name=..., <unknown>=...)`` must complain about the unknown keyword only.

    This never runs the function: argument parsing fails first, so no backend is touched
    (``gcs_query_rapid_bucket`` would query a real bucket in a GCS build).
    """
    with pytest.raises(TypeError) as info:
        function(**{name: "x", "no_such_keyword_xyz": 1})
    message = str(info.value)
    assert "no_such_keyword_xyz" in message, message
    assert repr(name) not in message, f"{name!r} was not accepted: {message}"


def test_gcs_set_channel_count_takes_n_by_keyword():
    _assert_keyword_is_accepted(s3dlio.gcs_set_channel_count, "n")
    previous = s3dlio.gcs_get_channel_count()  # 0 means "never set"; process-global
    try:
        s3dlio.gcs_set_channel_count(n=3)
        assert s3dlio.gcs_get_channel_count() in (0, 3)  # 0: a build without GCS
    finally:
        s3dlio.gcs_set_channel_count(n=previous)
    assert s3dlio.gcs_get_channel_count() == previous


def test_gcs_query_rapid_bucket_takes_bucket_or_uri_by_keyword():
    _assert_keyword_is_accepted(s3dlio.gcs_query_rapid_bucket, "bucket_or_uri")


def test_create_azure_writer_takes_uri_and_options_by_keyword():
    _assert_keyword_is_accepted(s3dlio.create_azure_writer, "uri")
    with pytest.raises(TypeError) as info:
        s3dlio.create_azure_writer(uri="x", options=None, no_such_keyword_xyz=1)
    assert "no_such_keyword_xyz" in str(info.value), str(info.value)


def test_pydataset_from_uris_takes_opts_by_keyword(tmp_path):
    for name in ("a.bin", "b.bin"):
        (tmp_path / name).write_bytes(b"x")
    uris = [file_uri(tmp_path, "a.bin"), file_uri(tmp_path, "b.bin")]
    assert len(s3dlio.PyDataset.from_uris(uris)) == 2
    assert len(s3dlio.PyDataset.from_uris(uris, opts=None)) == 2
    assert len(s3dlio.PyDataset.from_uris(uris, opts={})) == 2
    assert len(s3dlio.PyDataset.from_uris(uris, {})) == 2
    assert len(s3dlio.PyDataset.from_uris(uris=uris, opts=None)) == 2
    with pytest.raises(TypeError, match="_opts"):
        s3dlio.PyDataset.from_uris(uris, _opts=None)


# ------------- error text: scheme preceded by other text, malformed slashes -------------


@pytest.mark.parametrize(
    "template",
    [
        " {uri}",
        '"{uri}"',
        "url={uri}",
        "x {uri}",
        "{uri}",
    ],
)
def test_put_many_error_redacts_a_uri_preceded_by_other_text(template):
    for uri in (
        "https://u:SECRETPW@h.example/c/k?sig=TOPSECRETSIG",
        "https:/u:SECRETPW@h.example/c/k?sig=TOPSECRETSIG",
        "https:///u:SECRETPW@h.example/c/k?sig=TOPSECRETSIG",
    ):
        with pytest.raises(TypeError) as info:
            s3dlio.put_many([(template.format(uri=uri), 42)])
        message = str(info.value)
        assert "h.example/c/k" in message, message
        for secret in ("SECRETPW", "TOPSECRETSIG", "sig="):
            assert secret not in message, f"{secret!r} leaked: {message}"


@pytest.mark.parametrize(
    "uri",
    [
        "az://acct/c/o?sv=2022&sig=TOPSECRETSIG",
        "s3://b/k?X-Amz-Signature=TOPSECRETSIG",
        "gs://b/o?token=TOPSECRETSIG",
    ],
)
def test_put_many_error_cuts_a_credential_query_of_a_non_http_uri(uri):
    with pytest.raises(TypeError) as info:
        s3dlio.put_many([(uri, 42)])
    message = str(info.value)
    assert uri.split("?")[0] in message, message
    assert "TOPSECRETSIG" not in message, message


@pytest.mark.parametrize("uri", ["photo@2x.png", "a@b", "file://localhost:abc/a@b"])
def test_put_many_error_leaves_non_credential_text_alone(uri):
    with pytest.raises(TypeError) as info:
        s3dlio.put_many([(uri, 42)])
    assert uri in str(info.value), str(info.value)
