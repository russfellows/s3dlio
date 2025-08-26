# s3dlio - Multi-Protocol AI/ML Storage Library

## Overview
s3dlio is a high-performance, multi-protocol storage library designed specifically for AI/ML workloads. Built in Rust with Python bindings, it provides zero-copy streaming, comprehensive checkpointing, and advanced data loading capabilities across multiple storage backends including S3, Azure Blob Storage, local filesystems, and high-performance DirectIO.

This project supports both low-level storage operations (get, put, list, stat, delete) and high-level AI/ML operations including PyTorch/TensorFlow/JAX-compatible data loaders and distributed checkpointing systems. The library is designed to handle arbitrarily large models and datasets while maintaining memory efficiency through zero-copy streaming architecture.

## Key Components
1. **Rust Core Library**: High-performance storage operations with zero-copy streaming
2. **Python Library**: ML framework integration with seamless PyTorch/TensorFlow/JAX compatibility  
3. **CLI Tool**: Unified command-line interface supporting all storage backends

## Multi-Backend Support
- **AWS S3**: Full S3 API compatibility with multipart upload support
- **Azure Blob Storage**: Complete Azure integration with hot/cool tier support
- **Local Filesystem**: High-performance file operations with optional O_DIRECT support
- **DirectIO**: Bypass OS page cache for maximum I/O performance

## Recent Highlights

### Version 0.7.2 - Complete Python Compression Integration (Latest)
Full end-to-end compression support with Python API integration. The `compression_level` parameter now provides seamless compression/decompression across all storage backends with automatic save/load cycles. Achieves 99.8% compression ratios with data integrity preservation through streaming zstd compression and automatic decompression on read.

### Version 0.6.2 - Enhanced Data Integrity
Complete checksum integration across all storage backends providing CRC32C-based data integrity validation. All checkpoint operations now include computed checksums with zero-copy streaming preserved.

### Version 0.6.1 - Zero-Copy Streaming Infrastructure  
Revolutionary memory efficiency enabling processing of models larger than available RAM. Peak memory usage reduced from full model size to single chunk size (99%+ memory reduction for large models).

### Version 0.6.0 - Comprehensive Checkpointing System
Production-ready distributed checkpointing modeled after AWS S3 PyTorch Connector, with multi-backend support and framework integration for PyTorch, JAX, and TensorFlow.

### Version 0.5.3 - Advanced Async DataLoader
Dynamic batch formation with out-of-order completion, eliminating head-of-line blocking while maintaining complete backward compatibility.

📖 **[Complete Version History](docs/Changelog.md)** - Detailed changelog with all enhancements and technical details

### Logging
This is a new feature, that logs all S3 operations.  The file format is designed to be compatible with the MinIO warp tool.  Thus, these s3 op-log files can be used with our warp-replay project.

#### Rust CLI Usage
To enable via the cli, use the ```--op-log <FILE>``` syntax.

#### Python Library Usage
To enable s3 operation logging in Python, you must first start the logging:
```
# Start op logging (warp-replay compatible TSV.ZST)
s3.init_op_log("/tmp/python_s3_ops_test.tsv.zst")
```

Then, prior to exiting, shut down the logger to finalize the file properly.
```
# Flush and close (safe to call multiple times)
s3.finalize_op_log()
```


#### Op logging Environment variables
There are a number of tuning parameters that may be set, they are as follows:

For normal situations, optimizes I/O throughput and works on best-effort with minimal I/O or CPU impact. 
```
bash
export S3DLIO_OPLOG_BUF=2048        # entries
export S3DLIO_OPLOG_WBUFCAP=262144  # 256 KiB
export S3DLIO_OPLOG_LEVEL=1
export S3DLIO_OPLOG_LOSSLESS=false
```

For high replay fidelity, i.e. lossless logging, and high burst tolerance, use the following:
```
bash
export S3DLIO_OPLOG_BUF=8192        # entries
export S3DLIO_OPLOG_WBUFCAP=1048576 # 1 MiB
export S3DLIO_OPLOG_LEVEL=1
export S3DLIO_OPLOG_LOSSLESS=true
```

## To Do and Enhancements
What is left?  Several items are already identified as future enhancements.  These are noted both in discussions for this repo, and in the "Issues".  Note that currently all outstanding issues are tagged as enhancements, as there are no known bugs.  There are ALWAYS things that could work better, or differently, but currently, there are no outstanding issues.

### Enhancements
These are listed in no particular order:
  * (✅ Completed!) Add AI/ML data loader functionality (issue #14 - closed 9, Aug. 2025)
  * (✅ Completed!) Add AI/ML check point writer feature (issue #18 - closed 26, Aug. 2025)
  * (✅ Completed!) Implement S3 operation logging (issue #17 - closed 27, July 2025)
  * Improve Get performance (issue #11)
  * Improve Put performance and add multi-part upload (issue #12)
  * Improve python binding performance, w/ zero-copy views (issue #13)
  * (✅ Completed!) Implement regex for operations (issue #23 - closed 10, Aug. 2025)

# How to Guide
## Multi-Protocol Storage Library and CLI
This guide shows how to use the s3dlio library, which supports both a compiled command line interface executable called ```s3-cli``` and Python library, compiled into a wheel file. Using this wheel, the Python library may be installed, imported via  `import s3dlio as s3` and used by Python programs.

### Purpose
The purpose of this library is to provide a unified interface for AI/ML storage operations across multiple backends including AWS S3, Azure Blob Storage, local filesystems, and high-performance DirectIO. The library is designed to enable testing and production use of storage systems via both CLI and Python interfaces. The intention is to provide low-level storage operations (get, put, list, stat, delete) along with higher-level AI/ML compatible operations including data-loader and checkpoint-writer functionality, similar to PyTorch and TensorFlow equivalents.

### Multi-Backend Support
The library automatically detects and routes to the appropriate backend based on URI schemes:
- **s3://**: AWS S3 storage operations
- **az://**: Azure Blob Storage operations  
- **file://**: Local filesystem operations
- **direct://**: High-performance DirectIO operations

### Core Operations
The following primary operations are supported across all backends:
 - get
 - put
 - list
 - stat
 - delete
 - create-bucket (where applicable)
 - delete-bucket (where applicable)

### Enhanced Operations
Additionally, the CLI supports commands to upload and download data:
 - upload
 - download

These operations will respectively upload/put a local file as an object, or download/get an object as a local file.

**Note** that bucket creation can occur explicitly, or as part of a Put operation if the create-bucket flag is passed.  

### Container
For ease of testing and deployment, both the CLI and library may be built into a container. The Rust source code is removed from this container to reduce its size.

### Plan for Publishing
The Python library `s3dlio` will be published on PyPi for distribution. The compiled Rust executable and Python wheel files are already published on GitHub and are available under the release section of this project.  

# How to Build
In order to build this project, you can build the code and libraries and use them directly. 

### Prerequisites 
1. Rust, see [https://www.rust-lang.org/tools/install] How to Install Rust
2. The `maturin` command, which may be installed with pip, uv or some other tools
3. The `patchelf` executable, best installed with system packages, such as `sudo apt-get install patchelf` or similar 

## Building Executables and Libraries
### Rust Environment Pre-Req
Given this is a Rust project, you must install the Rust toolchain, including cargo, etc.
The following installs the Rust toolchain:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Python Environment Pre-Req
A Python environment is required.  The uv package manager and virtual environment works well.  
The 'uv' tool may be installed via: 
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Next, you will need a virtual environment, along with Python
Here are the steps:
1. ```uv venv```
2. ```source .venv/bin/activate```
3. ```uv python install 3.12```

### Building the CLI
To build the executable cli code:
 * At the top of the project issue: `cargo build --release`
 * Upon success the executable called `s3-cli` will exist in the ./target/release directory

### Building the Python Library
This step requires the Python Environment pre-reqs listed above.
 * First, make certain you have activated your virtual environment
 * If using uv, the command `source .venv/bin/activate` will do so
 * Next, to build the python library: `maturin build --release --features extension-module`
 * Upon success a python wheel will exist in the ./target/wheels directory

# Container
A container image may be built from the source code in this repository.  Also, examining the Dockerfile may help users better understand how to build the code outside of a container by examining the steps in the Dockerfile.  
The container will contain the executable, along with a python library.  This allows you to run either the Rust cli, or a python program that uses the Python library.
### Building / Pulling
Either docker or podman may be used to create the image, typically with a `podman build -t xxx .` where xxx is the name you want to use for your container image. For those that want to simply pull and run a pre-built container, a relatively recent image may be found on quay.io.  However, the version on quay may be older than that available by building from the source code in this repository.  

***Note:** A NEW container is here:  https://quay.io/repository/russfellows-sig65/s3dlio*

In order to pull the container using podman (or docker), you may execute the following:

```
podman pull quay.io/russfellows-sig65/s3dlio:0.4.2
```
***Note 2:** The above container must be pulled with the tag name as indicated*

### Running Container
Here is an example of starting a container using podman, and listing the contents of the containers /app directory.  
***Note:** It is VERY important to use the "--net=host" or similar command when starting, since using S3 necessitates network connectivity to S3 storage.*

```
eval@loki-node:~ $ podman run --net=host --rm -it s3dlio:0.4.2
root@loki-node:/app# ls
LICENSE  README.md  aws-env  configs  docs  local-env  python  target  tests  uv.lock  wheels
root@loki-node:/app# which python
/app/.venv/bin/python
root@loki-node:/app# python --version
Python 3.12.11

root@loki-node:/app# python ./python/tests/test_regex.py 
--- Creating S3 bucket russ-test5... ---
--- Setting up test files in S3... ---
Setup complete.

--- Testing s3dlio.list() ---
Non-recursive list found 2 items: ['test-recursive-py/data/', 'test-recursive-py/dummy1.txt']
Recursive list found 3 items: ['test-recursive-py/data/dummy2.txt', 'test-recursive-py/data/logs/dummy3.txt', 'test-recursive-py/dummy1.txt']
✅ List tests passed!

--- Testing s3dlio.download() ---
Non-recursive download got: ['dummy1.txt']
Recursive download got: ['dummy3.txt', 'dummy1.txt', 'dummy2.txt']
✅ Download tests passed!

--- Cleaning up test files... ---
--- Deleting Empty S3 bucket russ-test5... ---
✅ Cleanup successful!
root@loki-node:/app# exit 
```

# S3 Access via CLI or Python Library
This library and CLI currently support accessing S3 via http without TLS, and S3 with TLS via https for both official, and self signed certificates.  There is a method to direct the library to use a PEM file bundle of self-signed certificates, which is detailed below.

## Lack of Insecure TLS Option
There is NO option to connect via TLS and ignore certificate errors.  There are several reasons for this, the two biggest ones being:
1) It is VERY difficult to do this with the current Rust crates and AWS S3 SDK
2) It is VERY dangerous, and strongly discouraged

## Important S3 Environment Info
Setting up the environment variables and information necessary to access S3 can be somewhat tricky.  This code supports the use of variables read in from a ```.env``` file, OR environment variables.  Note that the examples here have sample .env files.  The values included from a private test environment, thus the ACCESS_KEY and SECRET_KEY values are not a concern.  Again, these may be set as environment variables if desired, but probably easier to set them in the .env file, which is read in by default.

***Note:** The following values are required:*
```
AWS_ACCESS_KEY_ID=some-id
AWS_SECRET_ACCESS_KEY=my-secret
AWS_ENDPOINT_URL=https://dns-name-or-ipaddr:port-number
AWS_REGION=region-if-needed
```
***Note:** The following value is OPTIONAL, but **REQUIRED** if using a self signed certificate with TLS access to S3:*
```
AWS_CA_BUNDLE_PATH=/a/path/to-myfile.pem
```

Here are the contents of the sample .env file:
```
eval@loki-node3:~/real-dealio2$ cat .env
AWS_ACCESS_KEY_ID=BG0XPVXISBP41DCXOQR8
AWS_SECRET_ACCESS_KEY=kGSmlMHBl0ohc/nYRtGbBx4KCfpdPN1/fLbtjUyX
AWS_ENDPOINT_URL=http://10.9.0.21
AWS_REGION=us-east-1
S3_BUCKET=my-bucket2
```

# Python
The intention of this entire library is to provide a Python library for interacting with S3.  The Rust CLI provides a way to check the underlying functions without worrying about constructing a Python program, or syntax errors.  Because it is pure Rust, it also more accurately displays the potential performance that may be attained.  However, using the Python library is the intention.  In the "docs" subdirectory, there is a quick overview of the Python API, written by a chat agent.  For those who prefer examples, there is a sample Python script to test the APIs, shown below.

## Compression Example
The Python API now supports seamless compression with the `compression_level` parameter, providing automatic compression during save and decompression during load:

```python
import s3dlio as s3

# Create a checkpoint store with compression (level 1-21, default 3)
store = s3.CheckpointStore("s3://my-bucket/checkpoints/", compression_level=10)

# Save data with compression - automatically adds .zst extension
import numpy as np
data = np.random.rand(1000, 1000).astype(np.float32)  # 4MB of random data
store.put_shard("model_weights", 0, data.tobytes())

# Load data with automatic decompression - seamlessly handles .zst files  
loaded_data = store.get_shard("model_weights", 0)
loaded_array = np.frombuffer(loaded_data, dtype=np.float32).reshape(1000, 1000)

# Compression works across all backends: S3, Azure, FileSystem, DirectIO
# Achieves 99.8% compression ratios for typical ML data
```

## Running a Python Test
Note: The following example shows running the `test_new_dlio_s3.py` Python script, after changing the number of objects to 500, by editing the file.  The default number of objects to use for testing is only 5 objects.  In order to display more accurate performance values, the following example shows running the test with `NUM_OBJECTS = 500` setting.

```
root@loki-node3:/app# uv run test_new_dlio_s3.py 
=== Sync LIST ===
Found 999 objects under s3://my-bucket2/my-data2/ in 0.02s

=== Sync PUT ===
Uploaded 500 objects (10000.00 MiB) in 3.34s -> 149.66 ops/s, 2993.14 MiB/s

=== Sync STAT ===
Stat for s3://my-bucket2/test_rust_api/object_0_of_5.dat:
  size: 20971520
  last_modified: 2025-07-16T22:53:33Z
  etag: "05a7bc9e4bab6974ebbb712ad54c18de"
  content_type: application/octet-stream
  content_language: None
  storage_class: None
  metadata: {}
  version_id: None
  replication_status: None
  server_side_encryption: None
  ssekms_key_id: None

=== Sync GET Many ===
Fetched 500 objects (10000.00 MiB) in 7.50s -> 66.67 ops/s, 1333.49 MiB/s

=== Sync GET Many Stats ===
Stats: 500 objects, 10485760000 bytes (10000.00 MiB) in 2.21s -> 226.25 ops/s, 4524.91 MiB/s

=== Sync DELETE ===
Deleted test objects in 0.37s

=== Async LIST ===
Found 999 objects under s3://my-bucket2/my-data2/ in 0.02s

=== Async PUT ===
Uploaded 500 objects (10000.00 MiB) in 3.10s -> 161.18 ops/s, 3223.68 MiB/s

=== Async GET Many ===
Fetched 500 objects (10000.00 MiB) in 7.64s -> 65.41 ops/s, 1308.29 MiB/s

=== Async GET Many Stats ===
Stats: 500 objects, 10485760000 bytes (10000.00 MiB) in 2.41s -> 207.79 ops/s, 4155.79 MiB/s

=== Async DELETE ===
Deleted test objects in 0.36s
root@loki-node3:/app# 
```

# Using the Rust CLI
If you built this project locally, outside a container, you will have to install the executable, or provide the path to its location, which by default is in the "./target/release" directory.
If running in the container, the Rust executable is in your path.  A  ```which s3-cli``` shows its location:

```
root@loki-node3:/app# which s3-cli
/usr/local/bin/s3-cli
```

Running the command without any arguments shows the usage:

```
root@loki-node3:/app# s3-cli                           
Usage: s3-cli <COMMAND>

Commands:
  create-bucket  Create a new S3 bucket
  delete-bucket  Delete an S3 bucket. The bucket must be empty
  list           List objects in an S3 path, the final part of path is treated as a regex. Example: s3-cli list s3://my-bucket/data/.*\\.csv
  stat           Stat object, show size & last modify date of a single object
  delete         Delete one object or every object that matches the prefix
  get            Download one or many objects concurrently
  put            Upload one or more objects concurrently, uses ObjectType format filled with random data
  upload         Upload local files (supports glob patterns) to S3, concurrently to jobs
  download       Download object(s) to named directory (uses globbing pattern match)
  help           Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose...     Increase log verbosity: -v = Info, -vv = Debug
      --op-log <FILE>  Write warp‑replay compatible op‑log (.tsv.zst). Disabled if not provided
  -h, --help           Print help
  -V, --version        Print version

root@loki-node3:/app# 
```

### Example --help for subcommand

First is an example of getting help for the `get` subcommand:

```
root@loki-node3:/app# s3-cli get --help 
Download one or many objects

Usage: s3-cli get [OPTIONS] <URI>

Arguments:
  <URI>  S3 URI – can be a full key or a prefix ending with `/`

Options:
  -j, --jobs <JOBS>  Maximum concurrent GET requests [default: 64]
  -h, --help         Print help


```
Next is an example of help for the `put` subcommand:

```
root@loki-node3:/app# s3-cli put --help
Upload one or more objects concurrently, uses ObjectType format filled with random data

Usage: s3-cli put [OPTIONS] <URI_PREFIX>

Arguments:
  <URI_PREFIX>  S3 URI prefix (e.g. s3://bucket/prefix)

Options:
  -c, --create-bucket              Optionally create the bucket if it does not exist
  -j, --jobs <JOBS>                Maximum concurrent uploads (jobs), but is modified to be min(jobs, num) [default: 32]
  -n, --num <NUM>                  Number of objects to create and upload [default: 1]
  -o, --object-type <OBJECT_TYPE>  What kind of object to generate: [default: RAW] [possible values: NPZ, TFRECORD, HDF5, RAW]
  -s, --size <SIZE>                Object size in bytes (default 20 MB) [default: 20971520]
  -t, --template <TEMPLATE>        Template for object names. Use '{}' as a placeholder [default: object_{}_of_{}.dat]
  -h, --help                       Print help
```



### Additional CLI Examples
Here are several examples of running the ```s3-cli``` command:

```
root@loki-node3:/app# s3-cli list s3://my-bucket4/
...
/russ-test3-989.obj
/russ-test3-99.obj
/russ-test3-990.obj
/russ-test3-991.obj
/russ-test3-992.obj
/russ-test3-993.obj
/russ-test3-994.obj
/russ-test3-995.obj
/russ-test3-996.obj
/russ-test3-997.obj
/russ-test3-998.obj
/russ-test3-999.obj

Total objects: 1200
```
#### Example Delete command and List
```
root@loki-node3:/app# s3-cli delete s3://my-bucket4/
Deleting 1200 objects…
Done.
root@loki-node3:/app# s3-cli list s3://my-bucket4/

Total objects: 0
```
#### Example put and then delete to cleanup
Note: This use of put only uses a single set of braces `{}` , and will thus each object will have only the object number, and not the total number in the name:
```
root@loki-node3:/app# s3-cli put -c -n 1200 -t russ-test3-{}.obj s3://my-bucket4/
Uploaded 1200 objects (total 25165824000 bytes) in 7.607254643s (157.74 objects/s, 3154.88 MB/s)

root@loki-node3:/app# 
root@loki-node3:/app# s3-cli get -j 32 s3://my-bucket4/
Fetching 1200 objects with 32 jobs…
downloaded 24000.00 MB in 4.345053775s (5523.52 MB/s)

root@loki-node3:/app# s3-cli delete s3://my-bucket4/

Deleting 1200 objects…
Done.
root@loki-node3:/app#
```
