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
   - [get - High-Performance Read](#get---high-performance-read)
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

**Which objects an `s3://` URI selects (#213).** The key part `K` of `s3://bucket/K` is a name, a
directory or an explicit glob -- never a regex and never a bare string prefix:

| URI key | Selects |
|---------|---------|
| `file.txt` (no `*`, no trailing `/`) | the object `file.txt` and, when descending (`-r`), everything under `file.txt/`. **Never** `file.txt.bak`, `file.txt2` or `file.txt.d/`. Without `-r`, `ls` shows `file.txt` and the directory entry `file.txt/` |
| `dl/` (trailing `/`) or empty | everything under `dl/` (or the whole bucket) |
| `file.txt*` | an explicit glob over the FULL key: `file.txt`, `file.txt.bak`, `file.txt2` (same directory level) |
| `data/**.npz` | every `.npz` under `data/`, at any depth |

In a glob `*` matches any run of characters except `/`, and `**` matches any run including `/`; every
other character (`.`, `(`, `[`, `+`, `$`, `?`) is literal. With a wildcard an object is selected iff
its full key matches, so `-r` makes no difference. A wildcard key that ends in `/` is refused.
Quote wildcard URIs in your shell. `*` also matches the empty string, so `s3://b/dir/*` includes a
folder-marker object `dir/` if one exists (`delete -r 's3://b/dir/*'` removes it too), and `a/**/b`
needs something between its slashes: it matches `a/x/b` and `a//b`, not `a/b`. There is no escape for a
literal `*` in a listing or recursive URI: a key literally named `a*b` is reached only by the exact,
non-recursive operations (`stat`, `delete` without `-r`), which take the key as written.
Use `-p/--pattern` to narrow the selection with a regex (see
[Regex Pattern Filtering](#regex-pattern-filtering)). The rule is implemented for `s3://` only;
`file://`, `direct://`, `gs://` and `az://` are unchanged (tracked in #220).

```bash
s3-cli ls [OPTIONS] <URI>
```

**Options:**
| Option | Description |
|--------|-------------|
| `-r, --recursive` | List objects recursively (traverse subdirectories) |
| `-c, --count-only` | Count objects only, don't print URIs (faster for large sets) |
| `-p, --pattern <REGEX>` | Regex pattern to filter results (client-side), matched against each entry relative to the listing base |

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
s3-cli ls -r az://acct/cont/ -p '(^|/)data_'  # Names starting with "data_" (pattern sees the path relative to the listing base)
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

On `s3://` the source URI selects as described under [ls](#ls---list-objects): `s3-cli download
s3://b/file.txt ./out/` downloads `file.txt` and NEVER `file.txt.bak`; `download -r s3://b/dl` downloads
everything under `dl/` (never `dl2/`); `download 's3://b/file.txt*'` downloads `file.txt`, `file.txt.bak`
and `file.txt2`. To filter with a regex use `-p/--pattern`
(see [Regex Pattern Filtering](#regex-pattern-filtering)); it is applied after the selection and matched
against the name relative to the listing base, so for `s3://b/dl/` and the key `dl/sub/c.npz` the pattern
sees `sub/c.npz`. The pattern is compiled before anything is listed: an invalid regex fails with nothing
sent and no directory created. `--pattern` works on every backend (`file://` and `direct://` included);
the selection rule is implemented for `s3://` only.

**Local layout (issue #284).** A download keeps each object's key path relative to the listing base:
`download -r s3://b/dl/ ./out/` writes `dl/a/x.bin` to `./out/a/x.bin` and `dl/b/x.bin` to `./out/b/x.bin`,
creating directories as needed. The listing base is the directory the URI selects (`dl/` for `dl/` and `dl`,
the directory before the first `*` for a glob, and the parent directory for a single exact object, so
`download s3://b/dir/file.txt ./out/` writes `./out/file.txt`). Zero-length "folder" objects (keys ending in
`/`) are skipped: no directory is created for them and they are not counted. `s3-cli cp -r <URI> <dir>` uses the
same layout.

An object is **refused** (counted as `failed` in the `DOWNLOAD summary` line, the command exits non-zero, and
nothing is written for it) when its relative key is empty or absolute, has an empty (`a//b`), `.` or `..`
component, contains a NUL byte or a component that is exactly a drive designator (`C:`; `a:b`, `x:y` and `v:1/x`
are ordinary names), or lies outside the listing base; a backslash counts as a separator on every platform, so
`..\evil` is refused too. The reason for each is logged as a WARN line
(the Python `download()` and the Rust `generic_download_objects` also repeat the first three in their error). Two objects that map to the
same local path are never merged: the first in listing order
is written and each later one fails with a collision error (an object and a directory of one name: below). A symlink that already exists below the
destination directory is not followed; the object fails instead, and on Unix the file itself is opened with
`O_NOFOLLOW`, so a symlink planted there between the check and the write is refused too. (An intermediate
directory swapped for a symlink by another local process in that instant is not detected. Names that differ
only in case can still merge on a case-insensitive file system, a file left by an earlier run is overwritten,
and a write that fails after its directories were created leaves those directories behind, empty.)

**Scope of these protections.** They reject ordinary hostile remote KEYS (traversal, absolute and drive-letter
forms, collisions, a pre-existing symlink). They are not containment against a hostile LOCAL directory, and they
are not a Windows file-name mapping: Windows is not a supported or tested platform of the release (the release
matrix is Linux), and alternate data streams, reserved device names and path aliases are not handled. Download
only to a destination whose directory entries are not being modified by an untrusted actor.

The base is one for the whole listing. `download -r s3://b/dl ./out/` where `dl` is both an object (typically a
zero-byte folder marker) and a directory (`dl/a`, `dl/b/c`) places everything relative to the parent: `dl/a`
goes to `./out/dl/a` and `dl/b/c` to `./out/dl/b/c`, never to `./out/a`. A file system cannot hold the
object `dl` and the directory `dl/`, so **the directory wins**: only the single object `dl` is refused (one
failed object, with the reason "object 'dl' and directory 'dl/' cannot both exist locally; ... fetch the
object alone (non-recursively), or use 's3://b/dl/' to download only the directory"). The same holds for a glob that
selects both. To get only the directory, add the trailing slash (`s3://b/dl/` writes `./out/a` and
`./out/b/c`); to get only the object, download it without `-r`. Only a directory with at least one downloadable object counts: if EVERY child under `dl/` is refused (unsafe key, collision), the object `dl` is downloaded as an ordinary file; in a chain (`p`, `p/q`, `p/q/r`) every object that has a valid child (`p` and `p/q`) is refused and only `p/q/r` is downloaded. `--pattern` is applied BEFORE this decision:
the layout base and the exact-object check see only the objects the pattern kept, so a pattern that filters
out `dl` itself makes the directory the base (`dl/a` goes to `./out/a`). `gs://`, `gcs://`, `az://` and the
Azure `https://<account>.blob.core.windows.net/<container>/...` spellings of a source all give the same
layout. For `gs://` and `az://` the listing is by RAW KEY PREFIX (the `s3://` selection rule is not implemented
there, tracked in #220), so `download -r gs://b/dl` can also list a sibling such as `dl.bak`; it is written
to `./out/dl.bak`, beside the directory's own files. Use `gs://b/dl/` to select the directory only. A file
that was only partly written (disk full, interrupted write) is NOT removed: the object fails, the command
exits non-zero, and the partial file stays in the destination for you to delete or overwrite. The CLI refuses per object (the rest is downloaded, the exit status is non-zero); the Rust library helpers
`s3dlio::s3_copy::download_objects*` are fail-fast: one bad key (or an object and directory of one name)
refuses the whole call before anything is created or fetched. Control characters in a remote key (newline,
ESC, bidi and other invisible format characters) are shown escaped (`\n`, `\u{1b}`, `\u{202e}`) in the error text, the failure reasons and the log lines of a download, including the cause text of a failed fetch, and credentials are redacted from every `scheme://` URI in that text (userinfo, signed query strings), not only from the object URI. This covers what `s3-cli download` and `s3-cli cp <uri> <dir>` print for a failed listing or a failed fetch, and the failure of the Rust `s3_copy::upload_files` and `download_objects*` helpers. Documented limits of the redaction: a schemeless `user:password@host` in the text is not recognised; a password containing whitespace or multiple slashes is not fully removed; one slash is handled for `http(s)` only, with the exceptions documented in the Python API guide. Raw `?` or `#` inside HTTP(S) userinfo removes that userinfo and the rest of the word; combining either character with a slash can leave part of the password; everything up to the last `@` of the authority (the text after `://` up to the first whitespace or `/`) is treated as userinfo, so a bare-host URI directly followed by text with an `@` and no whitespace (`https://h,user=a@b`) is over-redacted and `abfs://container@account` loses `container@`; a credential parameter is cut from `http(s)` URIs by dropping the whole query, but from other schemes (`s3://`, `gs://`, `az://`) only when it is one of the well-known names (`sig`, `signature`, `token`, `x-amz-*`, `x-goog-*`); and a cut query runs to the next whitespace, quote, backtick or angle bracket (`)`, `]`, `}`, `,` and `;` do not end it), the text after that boundary is kept, so compact JSON / XML keep their other fields. The same redaction and escaping is applied to ALL `s3-cli` error output (`ls`, `get`, `stat`, `put`, `delete`, `create-bucket`, `cp` upload, ...): the final error of every command is rendered through it before `main` prints it (`Error: ...` on stderr, non-zero exit status), and so are the per-object `GET failed` / `PUT failed` and `Failed to create bucket` log lines.

```bash
s3-cli download [OPTIONS] <SOURCE_URI> <LOCAL_PATH>
```

**Options:**
| Option | Description |
|--------|-------------|
| `-j, --jobs <N>` | Number of concurrent downloads (default: 64) |
| `-r, --recursive` | Download recursively |
| `-p, --pattern <REGEX>` | Regex (client-side) that narrows the selection; matched against each entry relative to the listing base |

**Examples:**

```bash
# Download one object (only file.txt; file.txt.bak is NOT downloaded)
s3-cli download s3://bucket/file.txt ./local/

# Download file.txt, file.txt.bak, file.txt2 ... (explicit wildcard; quote it)
s3-cli download 's3://bucket/file.txt*' ./local/

# Download entire prefix
s3-cli download gs://bucket/data/ ./local-data/

# Download with high concurrency
s3-cli download -j 128 s3://bucket/large-dataset/ ./dataset/

# Only the .npz objects under data/ (explicit regex; the URI itself is never a regex)
s3-cli download -r -p '\.npz$' s3://bucket/data/ ./npz-only/
```

---

### delete - Delete Objects

Delete one exact object, or, with `-r`, the objects an `s3://` URI selects (a name, a directory or a
`*` glob; see [ls](#ls---list-objects)). Alias: `rm`.

```bash
s3-cli delete [OPTIONS] <URI>
```

**The contract** (`bkt` is a bucket; `URI` may also be the host form `s3://host:port/bkt/...`,
which is sent to that server):

| Command | Behavior |
|---------|----------|
| `delete s3://bkt/foo` | Sends ONE delete request for exactly the key `foo`. Never `foobar`, never `foo/child`. |
| `delete s3://bkt/foo/` | Sends ONE delete request for exactly the key `foo/` (a zero-byte "folder marker"). A trailing slash alone never means "recursive"; the output reminds you to use `-r`. |
| either, object already missing | Success (exit 0), as with S3's own idempotent DELETE. Nothing is listed and nothing else is deleted. |
| either, the delete request fails (403, 5xx, network) | Non-zero exit, `failed to delete <uri>: <cause>`. A failure is never read as "already absent". |
| `delete -r s3://bkt/foo/` | Deletes the real objects whose key starts with `foo/`, nested ones included (`foo/a`, `foo/b/c`, and the marker `foo/` if present). Never `foobar`. |
| `delete -r s3://bkt/foo` | Deletes the object `foo` and everything under `foo/`. **Never `foobar`, `foo.bak` or `foo2/x`** (the same rule as `ls`). |
| `delete -r 's3://bkt/foo*'` | Explicit glob over the full key: `foo`, `foobar`, `foo.bak` (same directory level; not `foo/child`). `**` crosses `/`. A wildcard key ending in `/` is refused, nothing is deleted. |

How it works:

- **Exact delete.** One delete request for the named key, with no HEAD, no listing and no parent
  lookup, so it needs only delete permission. S3 reports success for a key that does not exist, so
  the command cannot tell "deleted" from "was already gone" and does not claim to. On other
  backends only the backend's own typed not-found error counts as success; every other error fails
  the command. On a **versioned bucket** a plain delete adds a delete marker and does not erase
  earlier versions.
- **Directories.** On `file://` and `direct://`, an exact delete of a path that is a directory fails
  with `<uri> is a directory; use -r / recursive=True to delete everything under it` and deletes
  nothing (use `-r` with a trailing `/`). A regular file is deleted; a missing path is success.
- **Recursive delete.** The selection is listed with no delimiter, so only real objects are returned
  (synthetic "directory" entries are never delete candidates), then deleted in batches of up to
  1,000 keys. A name selects itself and its `name/` directory; `*` / `**` are the only wildcards, and
  `.`, `[`, `?` and so on are never read as a pattern. Without `-r` a `*` is an ordinary key character.
- **`-p/--pattern`** is a client-side regex that narrows the `-r` selection (it is matched against
  each listed object URI, `s3://bkt/key`). It is an error without `-r`.
- **Reporting.** S3 reports a key that does not exist as "deleted", so the command never claims a
  deleted count. The messages, exactly:
  - exact delete: `Delete request succeeded: s3://bkt/foo (S3 does not report whether the object existed)`
    (other backends: `this backend does not report ...`). When the URI ends in `/` and `-r` was not
    given, the text continues `; only the object named by this exact URI was targeted, not what is
    under it (use -r to delete everything under a prefix)`;
  - `-r`, success: `Sent delete requests for N objects found under prefix <uri> (N listed; 0 reported errors)`;
  - `-r -p <regex>`, success: `Sent delete requests for N objects selected under prefix <uri> by pattern '<regex>' (M listed, N selected; 0 reported errors)`
    (`M` is counted before the pattern filter, `N` after it);
  - `-r` and nothing under the prefix: `No objects found under prefix: <uri>` (exit 0);
  - `-r -p <regex>` and nothing matches: `No objects under prefix <uri> match pattern '<regex>' (M listed)` (exit 0);
  - a listing that fails before anything is sent (for example a malformed URI such as `-r s3://bkt`,
    or a failing first page): `cannot list objects under prefix <uri>; nothing was deleted: <cause>`
    (exit non-zero, no delete summary);
  - a failure after requests were sent (a failed request, a per-key error inside a successful
    HTTP response, or a listing error part-way):
    `Sent delete requests for X of the L listed under prefix <uri>; ...` (with `-p`: `of the S selected
    (pattern '<regex>') of L listed`), followed by the number of failed requests, the keys that failed,
    and `K selected objects were NOT attempted` when batches were held back (a listing error also
    holds back the partly filled batch in which it occurred). Exit non-zero.
- **After a failure, nothing more is sent.** The check is made immediately before each DeleteObjects
  request is sent. Requests already in flight finish; later batches (including a partly filled one)
  are not sent and are counted as not attempted; the rest of the prefix is not listed.
- **Other backends.** `file://`, `direct://`, `gs://` and `az://` use their own existing listing.
  `-r` there requires a URI that ends in `/` and otherwise fails with an error (it never broadens
  into a prefix match). Without `-r`, the same single delete request is made; an already-missing
  object is success where the backend reports a typed not-found error (file backends, Azure
  `BlobNotFound`, GCS NOT_FOUND). On GCS a missing bucket gives the same status as a missing
  object, so it is not told apart.

**Options:**
| Option | Description |
|--------|-------------|
| `-r, --recursive` | Delete every object the URI selects (exact name and its directory, a directory, or a `*` glob; recursive listing) |
| `-p, --pattern <REGEX>` | With `-r` only: narrow the selection to URIs matching the regex |
| `-j, --jobs <N>` | Accepted for compatibility; currently not used (batches are 1,000 keys) |

**Examples:**

```bash
# Delete one object (one request; succeeds even if it was already gone)
s3-cli delete s3://bucket/dir/file.txt

# Delete the folder-marker object "dir/" only; dir/file.txt is untouched
s3-cli delete s3://bucket/dir/

# Delete everything under the folder dir/
s3-cli delete -r s3://bucket/temp-data/

# Deletes the object "temp" and everything under temp/ -- NOT temp-data/..., temporary.txt, tempest/...
s3-cli delete -r s3://bucket/temp

# Explicit wildcard: temp, temp-data, temporary.txt, tempest ... (same directory level; quote it)
s3-cli delete -r 's3://bucket/temp*'


# Delete only .log objects under a folder
s3-cli delete -r s3://bucket/logs/ -p '\.log$'

# Same, against a specific server
s3-cli delete -r s3://10.0.0.5:9000/bucket/logs/
```

**Migrating from earlier versions.** `delete s3://bkt/prefix/` (no `-r`) and
`delete s3://bkt/prefix` used to expand to the objects found under the name, with a fallback that
listed the parent "directory" (issues #167, #210: it could delete objects that were never named, and
it printed "Successfully deleted" when it deleted nothing). That behavior is gone. Add `-r` where you
mean "everything under".

---

### get - High-Performance Read

Read objects with concurrency for maximum throughput. `get` fetches the object bytes and reports the
byte count and throughput; it does **not** write them to a file and has no `--output` option. To save
objects to a local path use [download](#download---download-objects).

On `s3://` the URI selects as described under [ls](#ls---list-objects): `s3-cli get s3://b/file.txt`
reads `file.txt` and NEVER `file.txt.bak`; `get -r s3://b/dl` reads everything under `dl/`
(never `dl2/`); `get 's3://b/file.txt*'` reads `file.txt`, `file.txt.bak` and `file.txt2`. To filter
with a regex use `-p/--pattern`, applied after the selection and matched against the name relative to
the listing base (for `s3://b/dl/` and the key `dl/sub/c.npz` the pattern sees `sub/c.npz`). An invalid
pattern fails before any request is sent. `--pattern` cannot be combined with `--keylist` or
`--offset/--length`. The selection rule is implemented for `s3://` only; the other backends do not yet
follow it (tracked in #220).

```bash
s3-cli get [OPTIONS] [URI]
```

**Options** (exactly the options `s3-cli get --help` lists):
| Option | Description |
|--------|-------------|
| `[URI]` | A full key, a directory ending with `/`, or a `*` / `**` glob; optional when `--keylist` is given |
| `-p, --pattern <REGEX>` | Regex (client-side) that narrows the selection; matched against each entry relative to the listing base. Not valid with `--keylist` or `--offset/--length` |
| `-j, --jobs <JOBS>` | Maximum concurrent GET requests (default: 64, max 4096; accepts `k` and `m` suffixes) |
| `--concurrent <CONCURRENT>` | Alternative to `--jobs`, for `mp` compatibility (max 4096) |
| `--keylist <KEYLIST>` | File containing a list of URIs to read, one per line |
| `-r, --recursive` | Perform the operation recursively |
| `--offset <OFFSET>` | Byte offset for a range request (single-object GET only) |
| `--length <LENGTH>` | Number of bytes to read for a range request (to the end if omitted) |
| `--endpoints <HOST:PORT[,HOST:PORT...]>` | Comma-separated S3 endpoints for multi-endpoint GET, round-robin; requires an `s3://` URI |

```bash
# Only the .bin objects under data/
s3-cli get s3://bucket/data/ -p '\.bin$'
```

---

### put - High-Performance Upload

Upload generated data for benchmarking and testing.

```bash
s3-cli put [OPTIONS] <URI>
```

---

## Regex Pattern Filtering

The `-p/--pattern` option accepts standard Rust regex patterns. Patterns are applied **client-side** after listing, so they work with all backends.

For `s3://` URIs a regex only ever comes from `-p/--pattern`: the URI itself is never a regex (#213;
earlier versions read the last segment of an S3 URI as a regex). The pattern is applied AFTER the URI has
selected its objects (see [ls](#ls---list-objects)) and is matched (unanchored) against each entry
**relative to the listing base**, not against the whole URI. The base is the key itself when it ends with
`/`, the directory `K/` for an entry under a name `K` without a trailing `/`, the parent directory of `K`
for the object `K` itself, and, for a wildcard URI, the directory part of the text before the first `*`.
For a host-form URI (`s3://host:port/bkt/dl/`) the base is taken from the key parts:

| Listed URI | Entry | What the pattern sees |
|------------|-------|-----------------------|
| `s3://b/dl/` | `s3://b/dl/sub/c.npz` | `sub/c.npz` |
| `s3://b/dl` | `s3://b/dl/sub/c.npz` (under the directory `dl`) | `sub/c.npz` |
| `s3://b/data/run.1` | `s3://b/data/run.1` (the object itself) | `run.1` (relative to the parent directory) |
| `s3://b/data/file*` | `s3://b/data/file.txt.bak` | `file.txt.bak` (relative to `data/`) |

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

# The pattern sees the path RELATIVE to the listing base (e.g. "train/a/data_1.bin"),
# so a file directly in the base has no leading "/": write (^|/) rather than .*/
# Match filenames
-p '(^|/)data_'         # Names starting with "data_" (at any depth below the base)
-p '(^|/)[0-9]+\.bin$'  # Numeric filenames like 123.bin
-p '(^|/)file_[0-9]{4}\.txt$'  # file_0001.txt through file_9999.txt

# Match directories
-p '^train/'            # Files under the top-level "train" directory of the base
-p '(^|/)train/'        # Files under a "train" directory at any depth
-p '(^|/)2025-12-'      # Files in December 2025 directories
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
| `AZURE_STORAGE_ENDPOINT` | Custom endpoint (Azurite, etc.) |
| `S3DLIO_AZURE_CREDENTIAL` | `auto` (default), `cli`, `managed-identity`, `client-secret`, `client-certificate`, `workload-identity`, `sas`, `none` |
| `S3DLIO_AZURE_SAS_TOKEN` | SAS token for `sas` mode |

The account comes from the URI (`az://account/container/key`). Credentials: see
`Environment_Variables.md` (Azure Blob Storage Configuration) for the modes and the variables each reads;
`az login` / `azd auth login` is the `cli` mode.

### GCS Configuration
| Variable | Description |
|----------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account key path |
| `GCS_ENDPOINT_URL` | Custom endpoint |
| `STORAGE_EMULATOR_HOST` | GCS emulator endpoint |

### HTTP/2 Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_H2C` | *(not set)* | HTTP/2 cleartext mode for `http://` endpoints. Not set = auto-probe h2c on first connection, fall back to HTTP/1.1 if rejected. `1` (or `true`, `yes`, `on`, `enable`) = force h2c, no fallback. `0` (or `false`, `no`, `off`, `disable`) = always HTTP/1.1, skip probe. Has **no effect** on `https://` endpoints — those negotiate HTTP/2 automatically via TLS ALPN. |
| `S3DLIO_POOL_MAX_IDLE_PER_HOST` | `32` | Maximum idle connections per host in the reqwest connection pool. |
| `S3DLIO_POOL_IDLE_TIMEOUT_SECS` | `90` | Idle connection timeout in seconds before a pooled connection is closed. |
| `S3DLIO_H2_ADAPTIVE_WINDOW` | `1` (enabled) | HTTP/2 flow-control window mode (active when `S3DLIO_H2C=1`). `1` = adaptive BDP estimator: hyper sends H2 PING frames, measures RTT, computes `BDP = throughput × RTT`, and auto-issues `WINDOW_UPDATE` to eliminate stalls. **Completely overrides the two static variables below when enabled.** `0` = static windows. |
| `S3DLIO_H2_STREAM_WINDOW_MB` | `4` | Per-stream H2 flow-control window in MiB (static mode only). Controls how much response data the server may send on one stream before we must issue `WINDOW_UPDATE`. H2 spec default is 64 KB — too small for storage I/O. Ignored when `S3DLIO_H2_ADAPTIVE_WINDOW=1`. Max 256 MiB. |
| `S3DLIO_H2_CONN_WINDOW_MB` | `4×stream` | Connection-level H2 flow-control window in MiB (static mode only). Aggregate receive budget across all concurrent streams on one TCP connection. Must be ≥ stream window or the connection becomes the bottleneck. Defaults to 4× `S3DLIO_H2_STREAM_WINDOW_MB`. Ignored when adaptive. Max 256 MiB. |

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

# With custom endpoint over plain HTTP, forcing HTTP/2 cleartext (h2c)
# Use this for storage systems that require HTTP/2 on their http:// API endpoint
AWS_ENDPOINT_URL=http://storage-host:9000 S3DLIO_H2C=1 s3-cli stat s3://bucket/object

# With custom endpoint using TLS — HTTP/2 is negotiated automatically via ALPN
# (no S3DLIO_H2C needed for https:// endpoints)
AWS_ENDPOINT_URL=https://storage-host:9000 AWS_CA_BUNDLE=/path/to/ca.crt s3-cli stat s3://bucket/object
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
