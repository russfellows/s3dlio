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

### Example get --help command

First is an example of getting help for the ```get``` subcommand:

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
#### Example put-many and then Delete to cleanup
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

### Running a Python Test

```
root@loki-node3:/app# python ./test_all_dlio_s3.py 
=== Listing objects from s3://my-bucket2/my-data2/ ===
First 25 objects:
['my-data2/file_1000_of_1000.npz', 'my-data2/file_100_of_1000.npz', 'my-data2/file_101_of_1000.npz', 'my-data2/file_102_of_1000.npz', 'my-data2/file_103_of_1000.npz', 'my-data2/file_104_of_1000.npz', 'my-data2/file_105_of_1000.npz', 'my-data2/file_106_of_1000.npz', 'my-data2/file_107_of_1000.npz', 'my-data2/file_108_of_1000.npz', 'my-data2/file_109_of_1000.npz', 'my-data2/file_10_of_1000.npz', 'my-data2/file_110_of_1000.npz', 'my-data2/file_111_of_1000.npz', 'my-data2/file_112_of_1000.npz', 'my-data2/file_113_of_1000.npz', 'my-data2/file_114_of_1000.npz', 'my-data2/file_115_of_1000.npz', 'my-data2/file_116_of_1000.npz', 'my-data2/file_117_of_1000.npz', 'my-data2/file_118_of_1000.npz', 'my-data2/file_119_of_1000.npz', 'my-data2/file_11_of_1000.npz', 'my-data2/file_120_of_1000.npz', 'my-data2/file_121_of_1000.npz']
Total objects: 999

Single download from s3://my-bucket2/my-data2/file_101_of_1000.npz: 296.15 MB/s (27377002 bytes in 0.09 s)

Multi-get: Downloaded 490 objects in 10.43 s
Throughput: 46.98 objects/s, 1226.50 MB/s

Single put: Uploaded 1 MB to s3://my-bucket2/test_rust_api/single_put_test.dat in 0.01 s (68.33 MB/s)

Get newly uploaded object: 1048576 bytes in 0.01 s

Put many: Uploaded 10 objects (total 10485760 bytes) in 0.04 s (232.88 MB/s)

Get many: Downloaded 10 objects (total 10485760 bytes) in 0.04 s (250.91 MB/s)

Delete: Removed all objects under s3://my-bucket2/test_rust_api/ in 0.01 s

After deletion, objects under s3://my-bucket2/test_rust_api/:
[]
Deletion verified: no objects remain.
root@loki-node3:/app# 
```

