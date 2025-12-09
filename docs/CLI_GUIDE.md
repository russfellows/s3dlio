# s3-cli Command Line Interface Guide

**Version:** 0.9.25  
**Last Updated:** December 9, 2025

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Storage URI Formats](#storage-uri-formats)
5. [Commands](#commands)
   - [ls - List Objects](#ls---list-objects)
   - [stat - Object Metadata](#stat---object-metadata)
   - [upload - Upload Files](#upload---upload-files)
   - [download - Download Objects](#download---download-objects)
   - [delete - Delete Objects](#delete---delete-objects)
   - [get - High-Performance Download](#get---high-performance-download)
   - [put - High-Performance Upload](#put---high-performance-upload)
6. [Regex Pattern Filtering](#regex-pattern-filtering)
7. [Operation Logging](#operation-logging)
8. [Environment Variables](#environment-variables)
9. [Examples by Backend](#examples-by-backend)
10. [Troubleshooting](#troubleshooting)

---

## Overview

`s3-cli` is a universal command-line tool for storage I/O operations across multiple cloud and local storage backends. It provides a consistent interface for S3, Azure Blob, Google Cloud Storage, local filesystems, and DirectIO.

**Key Features:**
- **5 Storage Backends**: S3, Azure Blob (`az://`), GCS (`gs://`), file (`file://`), DirectIO (`direct://`)
- **Regex Pattern Filtering**: Client-side filtering with full regex support
- **Progress Indicators**: Real-time progress with object counts and rates
- **Operation Logging**: Warp-compatible operation logs for replay and analysis
- **High Concurrency**: Parallel operations with configurable job counts

---

## Installation

### From Source (Recommended)

```bash
cd s3dlio
cargo build --release
# Binary at: target/release/s3-cli

# Optional: Add to PATH
cp target/release/s3-cli /usr/local/bin/
```

### Verify Installation

```bash
s3-cli --version
# Output: s3dlio 0.9.24

s3-cli --help
```

---

## Quick Start

```bash
# List objects in a bucket
s3-cli ls s3://my-bucket/prefix/

# List recursively with count
s3-cli ls -rc gs://my-bucket/data/

# Upload files
s3-cli upload /local/data/*.csv s3://bucket/data/

# Download objects
s3-cli download s3://bucket/prefix/ ./local-dir/

# Get object metadata
s3-cli stat s3://bucket/my-file.txt

# Delete objects
s3-cli delete s3://bucket/prefix/ -r
```

---

## Storage URI Formats

All commands use consistent URI formats across backends:

| Backend | URI Format | Example |
|---------|------------|---------|
| **S3** | `s3://bucket/key` | `s3://my-bucket/data/file.txt` |
| **Azure Blob** | `az://account/container/key` | `az://myaccount/mycontainer/data/` |
| **GCS** | `gs://bucket/key` | `gs://my-bucket/prefix/` |
| **Local File** | `file:///path` | `file:///home/user/data/` |
| **DirectIO** | `direct:///path` | `direct:///mnt/nvme/data/` |

**Prefix Listings:**
- Trailing slash indicates a prefix: `s3://bucket/prefix/`
- Lists all objects under that prefix
- Works consistently across all backends

---

## Commands

### ls - List Objects

List objects in any storage backend with optional filtering and counting.

```bash
s3-cli ls [OPTIONS] <URI>
```

**Options:**
| Option | Description |
|--------|-------------|
| `-r, --recursive` | List objects recursively (traverse subdirectories) |
| `-c, --count-only` | Count objects only, don't print URIs (faster for large sets) |
| `-p, --pattern <REGEX>` | Regex pattern to filter results (client-side) |

**Examples:**

```bash
# List all objects in bucket
s3-cli ls s3://bucket/

# List recursively
s3-cli ls -r gs://bucket/data/

# Count objects with progress indicator
s3-cli ls -rc az://account/container/
# Output: Total objects: 401,408 (28.147s, rate: 14,261 objects/s)

# Filter with regex pattern
s3-cli ls -r s3://bucket/ -p '.*\.txt$'        # Only .txt files
s3-cli ls -r gs://bucket/ -p '.*\.(csv|json)$' # CSV or JSON files
s3-cli ls -r az://acct/cont/ -p '.*/data_.*'   # Files with "data_" in path
```

**Progress Display (with `-c`):**
```
⠙ [00:00:05] 71,305 objects (14,261 obj/s)
```

---

### stat - Object Metadata

Get metadata for a single object.

```bash
s3-cli stat <URI>
```

**Example:**
```bash
s3-cli stat s3://bucket/data/file.npz
# Output:
# URI             : s3://bucket/data/file.npz
# Size            : 1048576
# LastModified    : 2025-12-09T10:30:00Z
# ETag            : "d41d8cd98f00b204e9800998ecf8427e"
# Content-Type    : application/octet-stream
```

---

### upload - Upload Files

Upload local files to any storage backend. Supports glob patterns.

```bash
s3-cli upload <FILES>... <DESTINATION_URI>
```

**Options:**
| Option | Description |
|--------|-------------|
| `-j, --jobs <N>` | Number of concurrent uploads (default: 32) |

**Examples:**

```bash
# Upload single file
s3-cli upload /local/data.csv s3://bucket/data/

# Upload multiple files with glob
s3-cli upload /local/logs/*.log az://account/container/logs/

# Upload with higher concurrency
s3-cli upload -j 64 /data/*.npz gs://bucket/training/
```

---

### download - Download Objects

Download objects from any storage backend to local filesystem.

```bash
s3-cli download <SOURCE_URI> <LOCAL_PATH>
```

**Options:**
| Option | Description |
|--------|-------------|
| `-j, --jobs <N>` | Number of concurrent downloads (default: 32) |

**Examples:**

```bash
# Download single object
s3-cli download s3://bucket/file.txt ./local/

# Download entire prefix
s3-cli download gs://bucket/data/ ./local-data/

# Download with high concurrency
s3-cli download -j 128 s3://bucket/large-dataset/ ./dataset/
```

---

### delete - Delete Objects

Delete one or more objects. Supports prefix deletion and regex filtering.

```bash
s3-cli delete [OPTIONS] <URI>
```

**Options:**
| Option | Description |
|--------|-------------|
| `-r, --recursive` | Delete all objects under prefix |
| `-j, --jobs <N>` | Concurrent deletion jobs (default: 32) |
| `-p, --pattern <REGEX>` | Only delete objects matching regex |

**Examples:**

```bash
# Delete single object
s3-cli delete s3://bucket/file.txt

# Delete all objects under prefix
s3-cli delete -r s3://bucket/temp-data/

# Delete only .log files
s3-cli delete -r s3://bucket/logs/ -p '.*\.log$'

# High-concurrency deletion
s3-cli delete -r -j 100 gs://bucket/old-data/
```

---

### get - High-Performance Download

Download objects with advanced options for maximum throughput.

```bash
s3-cli get [OPTIONS] <URI>
```

**Options:**
| Option | Description |
|--------|-------------|
| `-j, --jobs <N>` | Concurrent downloads (default: 32) |
| `-o, --output <PATH>` | Output path (default: current directory) |

---

### put - High-Performance Upload

Upload generated data for benchmarking and testing.

```bash
s3-cli put [OPTIONS] <URI>
```

---

## Regex Pattern Filtering

The `-p/--pattern` option accepts standard Rust regex patterns. Patterns are applied **client-side** after listing, so they work with all backends.

### Pattern Syntax

| Pattern | Meaning |
|---------|---------|
| `.` | Any single character |
| `.*` | Zero or more of any character |
| `\.` | Literal dot (escaped) |
| `$` | End of string |
| `^` | Start of string |
| `[0-9]` | Any digit |
| `[a-z]` | Any lowercase letter |
| `(a\|b)` | Either "a" or "b" |
| `+` | One or more of previous |
| `?` | Zero or one of previous |

### Common Patterns

```bash
# Match file extensions
-p '.*\.txt$'           # .txt files
-p '.*\.npz$'           # .npz files
-p '.*\.(csv|json)$'    # .csv or .json files
-p '.*\.(jpg|png|gif)$' # Image files

# Match filenames
-p '.*/data_.*'         # Files containing "data_" in path
-p '.*/[0-9]+\.bin$'    # Numeric filenames like 123.bin
-p '.*/file_[0-9]{4}\.txt$'  # file_0001.txt through file_9999.txt

# Match directories
-p '.*/train/.*'        # Files under any "train" directory
-p '.*/2025-12-.*'      # Files in December 2025 directories
```

### Shell Quoting

**Always use single quotes** to prevent shell expansion:

```bash
# Correct - single quotes prevent shell interpretation
s3-cli ls -r s3://bucket/ -p '.*\.txt$'

# Wrong - shell may expand * or interpret $
s3-cli ls -r s3://bucket/ -p ".*\.txt$"  # May work, but risky
s3-cli ls -r s3://bucket/ -p .*\.txt$    # Shell will expand *
```

---

## Operation Logging

Generate warp-compatible operation logs for analysis and replay.

```bash
s3-cli --op-log operations.tsv.zst ls -r s3://bucket/
```

**Log Format (TSV):**
```
start_ns	end_ns	op	uri	offset	length	status	first_byte_ns	client_id
1733745600123456789	1733745600234567890	GET	s3://bucket/file.bin	0	1048576	OK	1733745600150000000	agent-1
```

**Environment Variables for Logging:**
```bash
# Auto-sort log by timestamp at shutdown
export S3DLIO_OPLOG_SORT=1
s3-cli --op-log sorted.tsv.zst ls -r s3://bucket/
```

---

## Environment Variables

### S3 Configuration
| Variable | Description |
|----------|-------------|
| `AWS_PROFILE` | AWS credentials profile |
| `AWS_REGION` | AWS region |
| `AWS_ENDPOINT_URL` | Custom S3 endpoint (MinIO, Ceph, etc.) |

### Azure Configuration
| Variable | Description |
|----------|-------------|
| `AZURE_STORAGE_ACCOUNT` | Storage account name |
| `AZURE_STORAGE_ENDPOINT` | Custom endpoint (Azurite, etc.) |

### GCS Configuration
| Variable | Description |
|----------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account key path |
| `GCS_ENDPOINT_URL` | Custom endpoint |
| `STORAGE_EMULATOR_HOST` | GCS emulator endpoint |

### Logging
| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Log level (`error`, `warn`, `info`, `debug`, `trace`) |
| `S3DLIO_OPLOG_SORT` | Set to `1` to auto-sort operation logs |

---

## Examples by Backend

### S3

```bash
# List bucket
s3-cli ls s3://my-bucket/

# Upload to S3
s3-cli upload /data/*.csv s3://bucket/data/

# Download from S3
s3-cli download s3://bucket/data/ ./local/

# With custom endpoint (MinIO)
AWS_ENDPOINT_URL=http://localhost:9000 s3-cli ls s3://bucket/
```

### Azure Blob

```bash
# List container
s3-cli ls az://myaccount/mycontainer/

# Upload to Azure
s3-cli upload /logs/*.log az://account/container/logs/

# With Azurite emulator
AZURE_STORAGE_ENDPOINT=http://127.0.0.1:10000 s3-cli ls az://devstoreaccount1/test/
```

### Google Cloud Storage

```bash
# List bucket
s3-cli ls gs://my-bucket/

# Count objects recursively
s3-cli ls -rc gs://my-bucket/data/

# With fake-gcs-server
GCS_ENDPOINT_URL=http://localhost:4443 s3-cli ls gs://testbucket/
```

### Local Filesystem

```bash
# List directory
s3-cli ls file:///home/user/data/

# List recursively
s3-cli ls -r file:///var/log/

# Count files
s3-cli ls -rc file:///data/
```

### DirectIO

```bash
# List with DirectIO (bypasses page cache)
s3-cli ls direct:///mnt/nvme/data/

# Download with DirectIO for benchmarking
s3-cli download direct:///mnt/nvme/source/ ./dest/
```

---

## Troubleshooting

### "Invalid URI" Errors

Ensure URIs have proper format with scheme prefix:
```bash
# Wrong
s3-cli ls mybucket/prefix/

# Correct
s3-cli ls s3://mybucket/prefix/
```

### Permission Denied

Check credentials and permissions:
```bash
# S3 - verify AWS credentials
aws sts get-caller-identity

# Azure - verify login
az account show

# GCS - verify service account
gcloud auth list
```

### Slow Listing Performance

Use count-only mode for large result sets:
```bash
# Faster - doesn't print each URI
s3-cli ls -rc s3://bucket/

# Slower - prints every URI
s3-cli ls -r s3://bucket/
```

### Debug Logging

Enable verbose output:
```bash
# Info level
s3-cli -v ls s3://bucket/

# Debug level (caution: may hang with AWS SDK)
RUST_LOG=warn,s3dlio=debug s3-cli ls s3://bucket/
```

---

## See Also

- **[Python API Guide](PYTHON_API_GUIDE.md)** - Python library usage
- **[Multi-Endpoint Guide](MULTI_ENDPOINT_GUIDE.md)** - Load balancing configuration
- **[Operation Logging Guide](OPERATION_LOGGING.md)** - Detailed op-log documentation
- **[Changelog](Changelog.md)** - Version history

---

**Questions?** Check the main [README](../README.md) or open an issue on GitHub.
