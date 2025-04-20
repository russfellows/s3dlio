# How to Guide

## S3 Rust Library and CLI
This guide shows how to use the Rust library, which supports both a compiled command line interface executable called ```s3Rust-cli``` and Python library, compiled into a wheel file.  

### Purpose
The purpose of this library is to enable testing of S3 storage via both a cli and a Python library.  The intention is to create hooks to DLIO, so that it may access S3 storage during its testing.  The five primary operations for S3 are included: Get, Put, List, Delete, and Create-Bucket.  Note that bucket creation currently occurs only as part of a Put operation.  

For ease of testing, both the cli and library may be built into a container as well.  The Rust source code is removed from this container in order to reduce its size.  

### Plan for Publishing
Currently, the plan is to publish the Python library on PyPi for distribution.  The compiled Rust executable may also be published on GitHub for general use.  The source code was just published, since the project was made public.  

# How to Build
In order to build this project, you can build the code and libraries and use them directly. 

### Prerequisites 
1. Rust, see [https://www.rust-lang.org/tools/install] How to Install Rust
2. The `maturin` command, which may be installed with pip, uv or some other tools
3. The `patchelf` executable, best installed with system packages, such as `sudo apt-get install patchelf` or similar 

## Building Executables and Libraries
To build the code, there is a two step process:
1. First build the Rust code and library. Use `cargo build --release` , which of course presumes you have Rust installed
2. Next, build the Python library, using maturin.  Command is: `maturin build --release --features extension-module`

## Installing Python Library
To install the python library you would use a `pip install --force-reinstall ./path/to/my/manylinux.whl` using the path of the .whl file just created
For testing purposes you can build and run the code on most Linux systems, including both the Rust cli and the python library.  For ease of use, a container may be used as well.  

## Container
A container image may be built from the source code in this repository.  Also, examining the Dockerfile may help users better understand how to build the code outside of a container by examining the steps in the Dockerfile.  
The container will contain the executable, along with a python library.  This allows you to run either the Rust cli, or a python program that uses the Python library.
### Building / Pulling
Either docker or podman may be used to create the image, typically with a `podman build -t xxx .` where xxx is the name you want to use for your container image. For those that want to simply pull and run a pre-built container, a relatively recent image may be found on quay.io.  However, the version on quay may be older than that available by building from the source code in this repository.  

The location of the container is here:  https://quay.io/repository/russfellows-sig65/dealio_s3_rust

In order to pull the container using docker, you may execute the following:
```
docker pull quay.io/russfellows-sig65/dealio_s3_rust
```

### Running Container
Here is an example of starting the container using podman, and listing the contents of the containers /app directory.  
***Note:** It is VERY important to use the "--net=host" or similar command when starting, since using S3 necessitates network connectivity to S3 storage.*

```
eval@loki-node3:~/real-dealio2$ podman run --net=host --rm -it dealio_s3-rust 
root@loki-node3:/app# ls
README.md  dlio_s3_test3.py  target  test_all_dlio_s3.py
root@loki-node3:/app# which python
/app/.venv/bin/python
root@loki-node3:/app# python --version
Python 3.13.2
```

## Important S3 Info
Setting up the variables and information necessary to access S3 can be somewhat tricky.  This code supports the use of both, or either of environment variables, and values in a ```.env``` file.  Note that this container includes a sample .env file.  The values are from a private test environment.  Thus the ACCESS_KEY and SECRET_KEY values are not a concern.  

These values are required, but may be set as environment variables if desired.  Probably easier to just set them in the .env file, which is read in by default.

***Note:** There is a sample file called "my-env" which may be edited, and saved as .env.  Since dot files are hidden, they are not incldued in Github repos by default.*

Here are the contents, which include several commented out test buckets:

```
eval@loki-node3:~/real-dealio2$ cat .env
AWS_ACCESS_KEY_ID=BG0XPVXISBP41DCXOQR8
AWS_SECRET_ACCESS_KEY=kGSmlMHBl0ohc/nYRtGbBx4KCfpdPN1/fLbtjUyX
AWS_ENDPOINT_URL=http://10.9.0.21
AWS_REGION=us-east-1
S3_BUCKET=my-bucket2
#S3_BUCKET=bucket1
#S3_BUCKET=warp-benchmark-bucket

```

## Using the Rust CLI
The Rust executable is in your path.  A  ```which s3Rust-cli``` shows its location:

```
root@loki-node3:/app# which s3Rust-cli
/usr/local/bin/s3Rust-cli
```

Running the command without any arguments shows the usage:

```
root@loki-node3:/app# s3Rust-cli                           
Usage: s3Rust-cli <COMMAND>

Commands:
  list    List keys that start with the given prefix
  get     Download one or many objects
  delete  Delete one object or every object that matches the prefix
  put     Upload one object with random data. Upload one or more  concurrently with random data
  help    Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version

root@loki-node3:/app#
```

### Example --help for subcommand

First is an example of getting help for the `get` subcommand:

```
root@loki-node3:/app# s3Rust-cli get --help 
Download one or many objects

Usage: s3Rust-cli get [OPTIONS] <URI>

Arguments:
  <URI>  S3 URI – can be a full key or a prefix ending with `/`

Options:
  -j, --jobs <JOBS>  Maximum concurrent GET requests [default: 64]
  -h, --help         Print help


```
Next is an example of help for the `put` subcommand:

```
root@loki-node3:/app# s3Rust-cli put --help
Upload one or more objects concurrently, uses ObjectType format filled with random data

Usage: s3Rust-cli put [OPTIONS] <URI_PREFIX>

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
Here are several examples of running the ```s3Rust-cli``` command:

```
root@loki-node3:/app# s3Rust-cli list s3://my-bucket4/
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
root@loki-node3:/app# s3Rust-cli delete s3://my-bucket4/
Deleting 1200 objects…
Done.
root@loki-node3:/app# s3Rust-cli list s3://my-bucket4/

Total objects: 0
```
#### Example put and then delete to cleanup
Note: This use of put only uses a single set of braces `{}` , and will thus each object will have only the object number, and not the total number in the name:
```
root@loki-node3:/app# s3Rust-cli put -c -n 1200 -t russ-test3-{}.obj s3://my-bucket4/
Uploaded 1200 objects (total 25165824000 bytes) in 7.607254643s (157.74 objects/s, 3154.88 MB/s)

root@loki-node3:/app# 
root@loki-node3:/app# s3Rust-cli get -j 32 s3://my-bucket4/
Fetching 1200 objects with 32 jobs…
downloaded 24000.00 MB in 4.345053775s (5523.52 MB/s)

root@loki-node3:/app# s3Rust-cli delete s3://my-bucket4/
Deleting 1200 objects…
Done.
root@loki-node3:/app#
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

