# Overview
The goal of this project is to provide a Python library and CLI for S3 interaction, using very recent versions of AWS S3 Rust SDK (current version is documented in the Project overview).  This is designed to be a general purpose library, with AI/ML specific operations supported as well. Moreover, the intent is to provide low-level S3 operations (get, put, list, stat, delete), along with higher-level AI/ML compatible operations that perform data-loader and checkpoint-writer functionality, similar to PyTorch and TensorFlow data-loader and checkpoint-writer functions.  

As such, this project essentially has 3 components that can be utilized:
1. A Rust library "s3dlio"
2. A Python library built on top of the Rust library, also called "s3dlio"
3. An executable CLI, built using Rust

## What's New
This is in reverse order, newest first.
### Version 0.4.1
This version added a several python bindings to the "data-loader" interface, for use within PyTorch, TensorFlow and Jax AI/ML workflows.  At this stage this feature is deemed fully functional via the Python library interfaces. HOWEVER, additional testing is requested to ensure its interface and usage align with traditional AI/ML data-loaders.  Basic data-loader tests via python are available in the python/tests subdirectory, specifically using the files tf_smoke.py, jax_smoke.py and pytorch_smoke.py.    

### Version 0.4.0
This version added a high-level, "data-loader" interface, to enable use within AI/ML workflows.  The data loader work is on-going, but should have basic functionality, primarily through the Python library interface, where it is expected to receive most use.  This release will require testing to ensure its interface and usage align with traditional AI/ML data-loaders.   

### Version 0.3.x
These versions added features including new I/O tracing capabilities.  The file format is the same as the MinIO warp tool, thus enabling capture in a semi standard way.  With an I/O trace file, this enables the recording and playback of S3 I/O operations, with s3dlio providing the "recording" part of the equation.  Additional changes in these releases included improving and updating the Rust - Python binding, async I/O library utilized.  The async library changes should be invisible, but will ultimately enable higher multi-threading with lower overhead, and increased compatability with other libraries going forward.  

### Version 0.3.0
Since the prior project "dlio_s3_rust" was archived and all future work moved to this project, there have been a number of enhancements.  These include the following:
  * Enhanced data generation - now supports settable dedupe and compression ratios
  * Two new cli commands: upload and download, which help enhance the use as a testing tool
  * One new cli and Python library command, "stat"  
  * AI/ML object creation options:  This library now supports creation, and all S3 operations of 4 specific, AI/ML data / object types:
    - TensorRecord
    - HDF5
    - NPZ
    - Raw

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
  * (✅ Completed!) Add AI/ML data loader functionality (issue #14 - close 9, Aug. 2025)
  * Add AI/ML check point writer feature (issue #18)
  * (✅ Completed!) Implement S3 operation logging (issue #17 - closed 27, July 2025)
  * Improve Get performance (issue #11)
  * Improve Put performance and add multi-part upload (issue #12)
  * Improve python binding performance, w/ zero-copy views (issue #13)

# How to Guide
## S3 Rust Library and CLI
This guide shows how to use the Rust library, which supports both a compiled command line interface executable called ```s3-cli``` and Python library, compiled into a wheel file. Using this wheel, the Python library may be installed, imported via  `import s3dlio as s3` and used by Python programs.

### Purpose
The purpose of this library is to enable testing of S3 storage via both a cli and a Python library.  The intention is to create hooks to DLIO, so that it may access S3 storage during its testing.  The six primary operations for S3 are included: 
 - get
 - put
 - list
 - stat
 - delete
 - create-bucket.

Additionally, the cli supports commands to upload and download data.  These are:
 - upload
 - download

These operations will respectively upload/put a local file as an object, or download/get an object as a local file.  

**Note** that bucket creation currently occurs only as part of a Put operation.  

### Container
For ease of testing, both the cli and library may be built into a container as well.  The Rust source code is removed from this container in order to reduce its size.  

### Plan for Publishing
In the near future, the Python library `s3dlio` will be published on PyPi for distribution.  The compiled Rust executable and Python wheel files are already published on GitHub and are availble under the release section of this project.  

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

## Container
A container image may be built from the source code in this repository.  Also, examining the Dockerfile may help users better understand how to build the code outside of a container by examining the steps in the Dockerfile.  
The container will contain the executable, along with a python library.  This allows you to run either the Rust cli, or a python program that uses the Python library.
### Building / Pulling
Either docker or podman may be used to create the image, typically with a `podman build -t xxx .` where xxx is the name you want to use for your container image. For those that want to simply pull and run a pre-built container, a relatively recent image may be found on quay.io.  However, the version on quay may be older than that available by building from the source code in this repository.  

***Note:** An OLD container is here:  https://quay.io/repository/russfellows-sig65/dealio_s3_rust*

In order to pull the container using docker, you may execute the following:

```
docker pull quay.io/russfellows-sig65/s3dlio
```
***Note 2:** The above container does not exist currently*

### Running Container
Here is an example of starting a container using podman, and listing the contents of the containers /app directory.  
***Note:** It is VERY important to use the "--net=host" or similar command when starting, since using S3 necessitates network connectivity to S3 storage.*

```
eval@loki-node3:~/real-dealio2$ podman run --net=host --rm -it dealio_s3-rust 
root@loki-node3:/app# ls
README.md  dlio_s3_test3.py  target  test_all_dlio_s3.py
root@loki-node3:/app# which python
/app/.venv/bin/python
root@loki-node3:/app# python --version
Python 3.12.8
```

# S3 Access
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
  list      List keys that start with the given prefix
  stat      Stat object, show size & last modify date of a single object
  delete    Delete one object or every object that matches the prefix
  get       Download one or many objects concurrently
  put       Upload one or more objects concurrently, uses ObjectType format filled with random data
  upload    Upload local files (supports glob patterns) to S3, concurrently to jobs
  download  Download object(s) to named directory (uses globbing pattern match)
  help      Print this message or the help of the given subcommand(s)

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
