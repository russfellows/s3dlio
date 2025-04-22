# Overview of S3 Integration into DLIO

Below is a concise game‑plan you can follow to drop your Rust + PyO3 S3 reader into  **DLIO‑Benchmark**  with the least churn to the existing code base. Everything can be done in three self‑contained layers, so you never have to fork DLIO or touch the C‑side of your bindings.

----------

## 1. Ship the Rust code as a normal Python wheel

1.  **Expose a thin, format‑agnostic Python API**
    
Rust
    
```
// src/lib.rs
#[pyfunction]
fn read_npz(path: &str) -> PyResult<Py<PyArrayDyn<u8>>> { … }

#[pyfunction]
fn read_hdf5(path: &str) -> PyResult<Py<PyAny>> { … }

#[pyfunction]
fn read_tfrecord(path: &str, index: usize) -> PyResult<Py<PyBytes>> { … }
``` 
    
   All functions should release the GIL while they touch S3 (`#[pyo3(text_signature = …)]`  and  `Python::allow_threads`)._
    
2.  Build once with  **maturin**  and publish to a private index or copy the wheel into your venv:
    
bash
    
`maturin build --release
    pip install target/wheels/dlio_s3_rust‑*.whl` 
    

----------

## 2. Give DLIO a  **real**  S3 storage back‑end

`dlio_benchmark/storage/s3_storage.py`  is currently a stub: every method delegates to  `DataStorage`, which immediately returns  `None`  – that’s why the benchmark crashes as soon as it tries to list a bucket. Replace it with a minimal, boto3‑based implementation and keep your Rust code only for the heavy data path:

python

```
# dlio_benchmark/plugins/s3_rust_storage.py
import boto3, os, re
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType

class S3RustStorage(DataStorage):
    def __init__(self, bucket, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(bucket, NamespaceType.FLAT)
        self._s3 = boto3.client("s3")

    # ---- metadata helpers used by DLIO ----
    def get_uri(self, key):                    # s3://bucket/obj
        return f"s3://{os.path.join(self.namespace.name, key)}"

    def walk_node(self, prefix, use_pattern=False):
        paginator = self._s3.get_paginator("list_objects_v2")
        prefix = prefix.lstrip("/")
        keys = []
        for page in paginator.paginate(Bucket=self.namespace.name,
                                       Prefix=prefix):
            keys += [c["Key"] for c in page.get("Contents", [])]
        if use_pattern:          # crude glob for “*/*.npz” etc.
            regex = re.compile(os.path.basename(prefix).replace("*", ".*"))
            keys = [k for k in keys if regex.match(os.path.basename(k))]
        return keys

    def get_node(self, key=""):
        return MetadataType.S3_OBJECT if key else MetadataType.DIRECTORY

    # create/delete/put/get can be stubs for read‑only benchmarking
    def create_namespace(self, exist_ok=False): return True
    def create_node(self, id, exist_ok=False):  return True
    def delete_node(self, id):                  return True
```
Hook it up by monkey‑patching  **before**  `DLIOBenchmark`  is instantiated:

python

```
# s3_rust_plugin/__init__.py
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.common.enumerations import StorageType
from .s3_rust_storage import S3RustStorage
StorageFactory.get_storage = (
    lambda self, t, ns, fw=None:
        S3RustStorage(ns) if t == StorageType.S3 else
        StorageFactory.get_storage.__wrapped__(self, t, ns, fw)
)
``` 

Put this folder on  `PYTHONPATH`  and you’re done – DLIO will now produce lists of  `s3://bucket/path/file.npz`  keys.

----------

## 3. Drop‑in readers that call the Rust bindings

DLIO already lets you provide a custom reader class through the YAML key  `reader.reader_classname` 

[dlio-benchmark.readthedocs](https://dlio-benchmark.readthedocs.io/en/latest/custom_reader.html?utm_source=chatgpt.com) , so you don’t have to touch  `reader_factory.py`.

Example for NPZ (HDF5/TFRecord are analogous):

python

```
# s3_rust_plugin/npz_s3_reader.py
from dlio_benchmark.reader.npz_reader import NPZReader
import dlio_s3_rust                         # ← your wheel

class NPZS3Reader(NPZReader):
    def open(self, filename):
        if filename.startswith("s3://"):
            # dlio_s3_rust.read_npz returns a NumPy array directly
            return dlio_s3_rust.read_npz(filename)
        return super().open(filename)       # local file path
``` 

Add to your workload config:

yaml

```
workflow:
  storage_type: s3
  storage_root: my‑bucket
  format: npz
reader:
  reader_classname: s3_rust_plugin.npz_s3_reader.NPZS3Reader
``` 

That’s literally all you need; the rest of DLIO (index maps, batching, stats, etc.) keeps working unchanged.

----------

### Why this layout?

-   **No fork**  – everything lives in a plugin folder.
    
-   **Clear separation of concerns**
    
    -   boto3 handles light‑weight metadata ops.
        
    -   Rust handles heavy data transfers and parsing with zero‑copy into NumPy.
        
-   **Works with every existing DLIO workload**  – just flip  `storage_type: s3`  and point  `reader_classname`  to your plugin.
    

----------

## 4. Extra credit / performance tweaks

| Tweak  | Where  | Notes |
|---|---|---|
| Release GIL in Rust  | PyO3 `allow_threads`  | Lets Python threads overlap I/O  |
| Multipart / range requests  | Rust Code  | Use `GetObject` with `range` header to read slice needed for sample  |
| Async prefetch  | Python Code  | Wrap `read_*` calls in `concurrent.futures.ThreadPoolExecutor`  |
| Vectorised listing | Storage | Cache the key list in a local file to avoid paging the bucket every epoch. |

----------

### Quick sanity check

bash


```
export PYTHONPATH=$PWD/s3_rust_plugin:$PYTHONPATH dlio_benchmark \
    ++workload.workflow.generate_data=False \
    ++workload.workflow.storage_type=s3 \
    ++workload.workflow.storage_root=my-bucket \
    ++workload.workflow.format=npz \
    ++reader.reader_classname=s3_rust_plugin.npz_s3_reader.NPZS3Reader \
    ++workload.workflow.num_files_train=100
``` 

If everything is wired correctly you should see DLIO listing the keys in  _my‑bucket_  and your Rust code printing S3 GETs.

----------

That should get you fully functional S3 reading in DLIO with only ~200 lines of new Python and zero modifications to the benchmark itself. Happy benchmarking!

![Favicon](https://www.google.com/s2/favicons?domain=https://dlio-benchmark.readthedocs.io&sz=32)

Sources

I've laid out a “no‑fork” route to bolt your Rust S3 reader onto  **DLIO‑Benchmark**  in three clean layers:

Layer

What you do

Why it works

**1. Ship Rust as a wheel**

Build a PyO3 / maturin wheel (`pip install dlio_s3_rust‑*.whl`) that exposes <br/>`read_npz(path) -> np.ndarray` / `read_hdf5` / `read_tfrecord`  and releases the GIL.

DLIO can import it like any other Python package.

**2. Replace the stub S3 storage**

Add  `s3_rust_plugin/s3_rust_storage.py`  that lists objects with  **boto3**  and registers itself by monkey‑patching  `StorageFactory`.

DLIO’s initial file discovery (`storage.walk_node`) finally returns real keys.

**3. Provide custom readers**

Sub‑class the built‑in readers, override  `open()`  to call the Rust binding when the filename starts with  `s3://`, and point the YAML key  `reader.reader_classname`  to your class.

All batching / index logic stays unchanged; only the per‑file read path is swapped out.

Example snippet for NPZ:

python


```
# s3_rust_plugin/npz_s3_reader.py
from dlio_benchmark.reader.npz_reader import NPZReader
import dlio_s3_rust  # ← your wheel

class NPZS3Reader(NPZReader):
    def open(self, filename):
        return (dlio_s3_rust.read_npz(filename)
                if filename.startswith("s3://")
                else super().open(filename))
``` 

and in your workload config:

yaml


```
workflow:
  storage_type: s3          # tells DLIO to give you s3:// keys
  storage_root: my‑bucket
  format: npz
reader:
  reader_classname: s3_rust_plugin.npz_s3_reader.NPZS3Reader
``` 

That’s it—no changes to DLIO’s core, and you keep Python for light metadata while Rust handles the heavy, parallel S3 I/O. If you later want to speed up listing or add range‑based reads, you only touch your plugin.

Happy benchmarking!


# o1

### You said:

Just to be clear, I would prefer NOT to use the boto3 libraries, as they do not perform well. Instead, we should use the aws-s3-sdk, which has much better performance. Note, there are Python and Rust versions of that library. Also, if it will improve performance, I can create new Rust interfaces to integrate here, I just need to know what they are. The ultimate goal of this is to produce the highest storage performance possible. Can you think about my additional directions and update and/ or revise your suggestions if required.

### ChatGPT said:

Reasoned about S3 integration with aws-s3-sdk for 16 seconds

Below is an updated outline for integrating your Rust-based S3 reading—**without**  using boto3—while still plugging nicely into DLIO‑Benchmark’s Python layers. The key idea is to let your Rust code (backed by the AWS S3 Rust SDK) handle all heavy-lifting—both listing objects and reading their contents—so that Python remains a thin wrapper. This keeps overhead to a minimum and helps you squeeze out maximum throughput.

----------

## 1. Extend your Rust/PyO3 crate to handle S3 listing & object reads

Instead of calling Boto3 for listing (which can be slow and CPU-heavy), you can implement all S3 ops in Rust. PyO3 allows you to expose them to Python. For instance:

rust


```
// src/lib.rs
use pyo3::prelude::*;
use aws_sdk_s3::{Client, Config, Region};

#[pyfunction]
#[pyo3(text_signature = "(bucket, prefix, region)")]
fn list_objects(py: Python, bucket: &str, prefix: &str, region: &str) -> PyResult<Vec<String>> {
    // Release the GIL for blocking I/O
    py.allow_threads(|| {
        // 1. Create S3 client (cache if you expect repeated calls)
        let region_provider = Region::new(region.to_owned());
        let shared_config = Config::builder().region(region_provider).build();
        let client = Client::from_conf(shared_config);

        // 2. Call list_objects_v2 repeatedly until fully paginated
        let mut keys = Vec::new();
        let mut continuation: Option<String> = None;
        loop {
            let mut req = client
                .list_objects_v2()
                .bucket(bucket)
                .prefix(prefix);
            if let Some(token) = &continuation {
                req = req.continuation_token(token);
            }
            let resp = tokio_block_on(req.send())?; // If you’re using a sync wrapper or let pyo3 handle async

            if let Some(contents) = resp.contents() {
                for obj in contents {
                    if let Some(k) = &obj.key {
                        keys.push(k.clone());
                    }
                }
            }

            if resp.next_continuation_token().is_some() {
                continuation = resp.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(keys)
    })
}

#[pyfunction]
#[pyo3(text_signature = "(s3_uri, region)")]
fn read_npz(py: Python, s3_uri: &str, region: &str) -> PyResult<Py<PyAny>> {
    use pyo3::types::PyBytes;
    py.allow_threads(|| {
        // 1. Parse bucket & key from "s3://bucket/some/path.npz"
        let (bucket, key) = parse_s3_uri(s3_uri)?;

        // 2. Download object to memory (e.g. multi-part or single GET)
        let data = blocking_download_from_s3(&bucket, &key, region)?;

        // 3. Decompress / interpret .npz if needed OR just return raw bytes
        let py_data = PyBytes::new(py, &data);
        // or if you parse the npz into a NumPy array, return that
        Ok(py_data.into())
    })
}

// Similar approach for read_hdf5(...) or read_tfrecord(...)

// Helper stubs:
fn parse_s3_uri(uri: &str) -> Result<(String, String), SomeError> {
    // naive parse, e.g. remove "s3://", split at first slash
    // return (bucket, key)
}

fn blocking_download_from_s3(bucket: &str, key: &str, region: &str) -> Result<Vec<u8>, SomeError> {
    // sync or sync-wrapped: use the AWS Rust SDK's "get_object" or multi-part approach
}

// Optional: use a tokio runtime for async if you prefer
fn tokio_block_on<F: std::future::Future>(fut: F) -> F::Output {
    use once_cell::sync::Lazy;
    use tokio::runtime::Runtime;

    static RUNTIME: Lazy<Runtime> = Lazy::new(|| Runtime::new().unwrap());
    RUNTIME.block_on(fut)
}

#[pymodule]
fn dlio_s3_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(list_objects, m)?)?;
    m.add_function(wrap_pyfunction!(read_npz, m)?)?;
    // ...
    Ok(())
}
``` 

### Performance tips

-   **Reuse your  `Client`**. If you expect repeated calls, you can store the client/config in a static or global within  `list_objects`  or  `read_npz`  so you don’t re-initialize it on every call.
    
-   **`allow_threads`**. This ensures that Python doesn’t block on every network call. If DLIO spawns multiple Python threads or processes, each can proceed in parallel.
    
-   **Multipart downloads**. The AWS Rust SDK can automatically do multi-part requests for large objects. This is frequently faster than a single GET call.
    

Once compiled with  [maturin](https://github.com/PyO3/maturin), you’ll have a  `.whl`  you can install with  `pip install target/wheels/dlio_s3_rust-*.whl`.

----------

## 2. Implement a minimal “storage” plugin that calls your Rust list function

Even though Boto3 is omitted, DLIO’s code needs a class that implements its  `DataStorage`  interface so it can discover file names. You can do that in a few ways:

1.  **Hack**: Bypass  `storage_factory.py`  entirely and just pass a static list of s3 keys to your custom readers. This might be simplest if you control your own input lists in the config.  _But you lose some DLIO features like indexing or how it enumerates train/test files._
    
2.  **Cleaner**: Provide a custom plugin class that calls your Rust  `list_objects`  to do the enumerations. You’d do something like:
    

python

```
# s3_rust_plugin/storage.py
import os, re
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
import dlio_s3_rust  # <-- your Rust wheel

class S3RustStorage(DataStorage):
    def __init__(self, bucket, region="us-east-1", framework=None):
        super().__init__(framework)
        self.namespace = Namespace(bucket, NamespaceType.FLAT)
        self.region = region

    def get_uri(self, key):
        return f"s3://{os.path.join(self.namespace.name, key)}"

    def walk_node(self, prefix, use_pattern=False):
        all_keys = dlio_s3_rust.list_objects(self.namespace.name, prefix, self.region)
        if use_pattern:
            regex = re.compile(os.path.basename(prefix).replace("*", ".*"))
            all_keys = [k for k in all_keys if regex.match(os.path.basename(k))]
        return all_keys

    def get_node(self, key=""):
        return MetadataType.S3_OBJECT if key else MetadataType.DIRECTORY

    # If the rest of DLIO calls these, stub them out or implement
    def create_namespace(self, exist_ok=False): return True
    def create_node(self, id, exist_ok=False):  return True
    def delete_node(self, id):                  return True

```

### Register your plugin in DLIO

One approach is monkey-patching or hooking  `StorageFactory.get_storage`  so that if  `StorageType.S3`  is requested, it returns  `S3RustStorage`. But that might require patching. Another approach is just forking DLIO slightly (if that’s acceptable) or using the  `storage_type`  enum in your own local build:

python

```
# s3_rust_plugin/__init__.py
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.common.enumerations import StorageType
from .storage import S3RustStorage

orig_get_storage = StorageFactory.get_storage
def custom_get_storage(self, t, ns, fw=None):
    if t == StorageType.S3:
        return S3RustStorage(ns)
    return orig_get_storage(self, t, ns, fw)

StorageFactory.get_storage = custom_get_storage

```

**Result**: DLIO’s internal calls to  `walk_node`  etc. will talk to your S3RustStorage, which in turn calls your Rust code for listing S3 objects.

----------

## 3. Update the format‑specific Readers to call Rust for data

DLIO’s “reader” classes (for NPZ, TFRecord, HDF5) have an  `open()`  method that loads the data. You can subclass them so that if the path is  `s3://...`, you call your Rust function. That’s effectively the same as before, but skipping Boto:

python

```
# s3_rust_plugin/npz_s3_reader.py
from dlio_benchmark.reader.npz_reader import NPZReader
import dlio_s3_rust  # your Rust wheel

class NPZS3Reader(NPZReader):
    def open(self, filename):
        if filename.startswith("s3://"):
            # e.g. read_npz returns a NumPy array or raw bytes you can parse
            data = dlio_s3_rust.read_npz(filename, "us-east-1")
            return data
        else:
            return super().open(filename)

```

Then in your config:

yaml

```
workflow:
  storage_type: s3
  storage_root: my-s3-bucket
  format: npz

reader:
  reader_classname: s3_rust_plugin.npz_s3_reader.NPZS3Reader

```

**Done**—DLIO sees your S3 bucket, enumerates the S3 keys with your  `S3RustStorage`, and then your custom NPZ reader calls the Rust library to fetch and parse them at high speed.

----------

## 4. Pushing concurrency even further

-   **Threads in Rust**: If you want to download multiple files or splits in parallel from a single Python process, you can spin up a thread pool in Rust. Because you’ve used  `py.allow_threads()`, each Python thread can call into Rust without GIL contention, and the Rust code can do concurrent GETs with the official AWS S3 Rust SDK.
    
-   **Async in Rust**: The AWS SDK is async. If you want truly top speed, you can maintain your own async runtime in Rust and spawn multiple requests simultaneously.
    
-   **DLIO’s concurrency**: DLIO can also spawn multiple processes (with Horovod or MPI). Each rank can read from S3 in parallel. Just ensure your bucket region + credentials are correct so that every rank can run the same code.
    

----------

## Final Summary

1.  **Put all S3 logic in Rust**  using the AWS S3 Rust SDK (no boto3).
    
2.  **Expose**  two sets of PyO3 functions:
    
    -   `list_objects(bucket, prefix, region) -> [str, …]`
        
    -   `read_npz|read_hdf5|read_tfrecord(s3_uri, region) -> …`
        
3.  **Implement**  a custom DLIO “storage” class (`S3RustStorage`) that calls  `list_objects`  from Rust for file enumeration.
    
4.  **Subclass**  the existing DLIO readers (NPZReader, HDF5Reader, etc.) so that  `open("s3://…")`  delegates to your Rust “read” methods.
    
5.  **Configure**  DLIO to use  `StorageType.S3`  + your custom Reader.
    

This approach avoids Python overhead in the core data path (i.e., no Boto3), leverages the native AWS S3 Rust SDK for higher performance (including multi-part GET, concurrency, etc.), and minimally modifies the existing DLIO code structure. By concentrating all network I/O in a single Rust library, you can tune concurrency, caching, or specialized read patterns as needed without rewriting Python. Enjoy the performance boost!
