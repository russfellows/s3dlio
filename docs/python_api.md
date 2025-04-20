# dlio_s3_rust Python Library Reference

This document outlines the Python interface provided by the `dlio_s3_rust` package, exposing high‑performance S3 operations via both synchronous and asynchronous functions.

---

## Installation

After building or installing the wheel:

```bash
pip install dlio_s3_rust-<version>-py3-none-any.whl
```

Then, in Python:

```python
import dlio_s3_rust as s3
```

---

## Constants

- `s3.DEFAULT_OBJECT_SIZE` (int): Default size in bytes for generated objects (20 MiB).

---

## Synchronous API

All sync functions release the Python GIL during I/O, so other Python threads can run concurrently.

### `list(uri: str) -> List[str]`

List all object keys under the given S3 URI prefix (supports `*` globs only in async variants). Returns a list of key strings (without `s3://bucket/`).

```python
keys = s3.list("s3://my-bucket/path/")
```

### `get(uri: str) -> bytes`

Download a single object into memory. If `uri` contains no wildcard, returns raw bytes.

```python
data = s3.get("s3://my-bucket/path/object_0_of_9.dat")
```

### `get_many(uris: List[str], max_in_flight: int = 64) -> List[Tuple[str, bytes]]`

Download multiple objects concurrently. Returns a list of `(uri, bytes)` tuples in the **same order** as input.

```python
pairs = s3.get_many([
    "s3://my-bucket/obj0",
    "s3://my-bucket/obj1"
], max_in_flight=16)
```

### `put(
    prefix: str,
    num: int = 1,
    template: str = "object_{}_of_{}.dat",
    max_in_flight: int = 32,
    size: Optional[int] = None,
    should_create_bucket: bool = False,
    object_type: str = "RAW"
) -> None`

Generate and upload `num` objects of `size` bytes under `prefix`.

- **prefix**: S3 URI prefix (e.g. `s3://bucket/prefix`).
- **template**: Python format string with two `{}` slots: first → index, second → total count.
- **object_type**: one of `"NPZ"`, `"TFRECORD"`, `"HDF5"`, or `"RAW"` (case-insensitive).

```python
s3.put(
    "s3://my-bucket/test/",
    num=5,
    template="object_{}_of_{}.raw",
    max_in_flight=8,
    size=10_000_000,
    should_create_bucket=True,
    object_type="npz"
)
```

### `delete(uri: str) -> None`

Delete a single object or all objects under a prefix. If `uri` ends with `/`, deletes all keys under that prefix.

```python
s3.delete("s3://my-bucket/test/object_3_of_9.dat")
s3.delete("s3://my-bucket/test/")
```

### `read_npz(uri: str, array_name: Optional[str] = None) -> numpy.ndarray`

Convenience helper: download an `.npz` file and load a named array via `ndarray-npy`. If `array_name` is omitted, the first array in the archive is returned.

```python
arr = s3.read_npz("s3://my-bucket/data/test.npz")  # returns numpy.ndarray
```

---

## Asynchronous API

Async functions integrate with Python’s `asyncio` event loop via `pyo3-asyncio` + Tokio. They return awaitable futures and do **not** block the Python thread.

### `put_async(
    prefix: str,
    num: int = 1,
    template: str = "object_{}_of_{}.dat",
    max_in_flight: int = 32,
    size: Optional[int] = None,
    should_create_bucket: bool = False,
    object_type: str = "RAW"
) -> asyncio.Future[None]`

Async version of `put`. Usage:

```python
await s3.put_async(
    "s3://my-bucket/async-test/",
    num=3,
    object_type="hdf5"
)
```

### `get_many_async(uris: List[str], max_in_flight: int = 64) -> asyncio.Future[List[Tuple[str, bytes]]]`

Async version of `get_many`.

```python
results = await s3.get_many_async(
    ["s3://my-bucket/obj0", "s3://my-bucket/obj1"],
    max_in_flight=16
)
```

---

## Function Comparison

| Level             | Rust function                                                  | Python API                                                          | Mode   |
|-------------------|----------------------------------------------------------------|---------------------------------------------------------------------|--------|
| Basic GET         | `get_object_uri(uri: &str) -> Vec<u8>`                         | `get(uri) -> bytes`                                                 | sync   |
| Bulk GET          | `get_objects_parallel(uris, max_in_flight) -> Vec<(String,Vec<u8>)>` | `get_many(uris, max_in_flight) -> List[(str, bytes)]`             | sync   |
| GET Stats (sync)  | same as Bulk GET                                               | `get_many_stats(uris, max_in_flight) -> (count, total_bytes)`       | sync   |
| Bulk GET (async)  | same as Bulk GET                                               | `get_many_async(uris, max_in_flight) -> Future[List[(str, bytes)]]` | async  |
| GET Stats (async) | same as Bulk GET                                               | `get_many_stats_async(uris, max_in_flight) -> Future[(count, total_bytes)]` | async  |
| Basic PUT         | `put_object_with_random_data(uri, size) -> ()`                 | `put(prefix, num, template, ..., object_type) -> None`             | sync   |
| Bulk PUT          | `put_objects_parallel(uris, data, max_in_flight) -> ()`       | *not exposed directly*                                              | sync   |
| Typed Bulk PUT    | `put_objects_with_random_data_and_type(uris, size, max_in_flight, ObjectType) -> ()` | `put(..., object_type)`, `put_async(...)`                          | both   |
| LIST              | `list_objects(bucket, prefix) -> Vec<String>`                  | `list(uri) -> List[str]`                                            | sync   |
| DELETE            | `delete_objects(bucket, keys) -> ()`                           | `delete(uri) -> None`                                               | sync   |
| ENUM              | `enum ObjectType { Npz, TfRecord, Hdf5, Raw }`                  | `object_type` parameter                                             | N/A    |

## Notes & Differences

- **Sync vs Async**: Sync calls release the GIL during blocking I/O (`py.allow_threads`), whereas Async calls return `asyncio` futures that never block the Python thread.
- **Memory copying**: All `get` apis currently copy data into Python-managed bytes. Future zero‑copy APIs may avoid this overhead for pure‑benchmark runs.
- **Naming**: Sync functions live at top‐level (`put`, `get`, `get_many`, etc.). Async variants are named `put_async` and `get_many_async`.
- **Data types**: `object_type` strings are case‐insensitive; unrecognized values default to `RAW`.
- **Error handling**: All operations raise `RuntimeError` on failure.

---

> **Download**: save this file as `PYTHON_API_DOCUMENTATION.md` and add it to your repo for easy reference.

