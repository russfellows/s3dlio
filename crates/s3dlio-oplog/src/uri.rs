// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! URI reconstruction and translation for op-log rows
//!
//! An op-log row stores a URI as two columns, `endpoint` and `file`. s3dlio writes
//! them in these shapes:
//!
//! | writer | `endpoint` | `file` | URI the call used |
//! |---|---|---|---|
//! | `LoggedObjectStore`, S3 / Azure / GCS | `s3://bkt`, `az://acct`, `gs://bkt`, `https://acct.blob.core.windows.net` | `bkt/key`, `acct/cont/key`, `gs-bkt/key`, ... (everything after the scheme) | `scheme://` + `file` |
//! | `LoggedObjectStore`, local | `file://`, `direct://` | `/abs/path` | `endpoint` + `file` |
//! | legacy `S3Ops` (the "inner" row) | `AWS_ENDPOINT_URL` (`http://host:port`, or empty) | the bare key | **not recoverable**: the bucket is not recorded |
//!
//! [`row_uri`] rebuilds the exact URI for the first two shapes and returns a
//! [`RowUriError`] for anything else. It never guesses.

use anyhow::Result;
use s3dlio::s3_utils::{classify_s3_first_segment, S3FirstSegment};

/// Why the URI of an op-log row cannot be reconstructed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RowUriError {
    /// The `endpoint` column is empty (for example `AWS_ENDPOINT_URL` was unset on a legacy S3 row).
    #[error("op-log row has no endpoint (file '{file}'): the URI cannot be reconstructed")]
    NoEndpoint { file: String },

    /// The `endpoint` column is not a `scheme://...` URI (for example `unknown`).
    #[error("op-log row endpoint '{endpoint}' (file '{file}') is not a scheme://authority URI: the URI cannot be reconstructed")]
    NoScheme { endpoint: String, file: String },

    /// The scheme is not one s3dlio writes `endpoint` columns for.
    #[error(
        "op-log row endpoint '{endpoint}' (file '{file}') has an unsupported scheme '{scheme}'"
    )]
    UnsupportedScheme {
        scheme: String,
        endpoint: String,
        file: String,
    },

    /// A legacy S3 row: `endpoint` is the server (`http(s)://host:port`) and `file` the bare
    /// key. The bucket is not in the row, so the original URI is unknowable.
    #[error("bucket not recorded: endpoint '{endpoint}' with file '{file}' is a legacy S3 row (server and key only), so the original URI cannot be reconstructed")]
    BucketNotRecorded { endpoint: String, file: String },

    /// `file` does not begin with the authority named by `endpoint`, or is otherwise
    /// inconsistent with it.
    #[error("op-log row is inconsistent: file '{file}' does not start with the authority of endpoint '{endpoint}'")]
    Inconsistent { endpoint: String, file: String },
}

/// A row split into what it names.
struct RowParts<'a> {
    /// Exact URI the original call used.
    uri: String,
    /// Path below the bucket/container, as the original key (may carry leading slashes).
    relative: &'a str,
}

fn is_local_scheme(endpoint: &str) -> bool {
    endpoint == "file://" || endpoint == "direct://"
}

/// `file` starts with `endpoint` as a whole unit (the endpoint ends in `/`, or the next
/// character is `/` or the end of `file`).
fn starts_with_unit(file: &str, endpoint: &str) -> bool {
    match file.strip_prefix(endpoint) {
        None => false,
        Some(rest) => {
            endpoint.is_empty()
                || endpoint.ends_with('/')
                || rest.is_empty()
                || rest.starts_with('/')
        }
    }
}

fn split_row<'a>(endpoint: &str, file: &'a str) -> Result<RowParts<'a>, RowUriError> {
    if endpoint.is_empty() {
        return Err(RowUriError::NoEndpoint {
            file: file.to_string(),
        });
    }

    // file:// and direct://: `file` is the absolute path, the endpoint is only the scheme.
    if is_local_scheme(endpoint) {
        return Ok(RowParts {
            uri: format!("{endpoint}{file}"),
            relative: file,
        });
    }

    let Some((scheme, authority)) = endpoint.split_once("://") else {
        return Err(RowUriError::NoScheme {
            endpoint: endpoint.to_string(),
            file: file.to_string(),
        });
    };
    if authority.is_empty() || authority.contains('/') {
        return Err(RowUriError::NoScheme {
            endpoint: endpoint.to_string(),
            file: file.to_string(),
        });
    }

    // How many path segments below the authority name the container: s3 names the bucket
    // right after the authority unless the authority is a `host:port` server (then the
    // bucket follows it); Azure is account/container; GCS is the bucket.
    let container_segments = match scheme {
        "s3" => match classify_s3_first_segment(authority) {
            Ok(S3FirstSegment::Bucket(_)) => 0,
            Ok(S3FirstSegment::Server(_)) => 1,
            Err(_) => {
                return Err(RowUriError::Inconsistent {
                    endpoint: endpoint.to_string(),
                    file: file.to_string(),
                })
            }
        },
        "az" | "http" | "https" => 1,
        "gs" | "gcs" => 0,
        _ => {
            return Err(RowUriError::UnsupportedScheme {
                scheme: scheme.to_string(),
                endpoint: endpoint.to_string(),
                file: file.to_string(),
            })
        }
    };

    if !starts_with_unit(file, authority) {
        // A legacy S3Ops row: server in `endpoint`, bare key in `file`.
        return Err(if matches!(scheme, "http" | "https") {
            RowUriError::BucketNotRecorded {
                endpoint: endpoint.to_string(),
                file: file.to_string(),
            }
        } else {
            RowUriError::Inconsistent {
                endpoint: endpoint.to_string(),
                file: file.to_string(),
            }
        });
    }

    // `file` = authority [ "/" container-segments... "/" key ]
    let below_authority = file[authority.len()..].strip_prefix('/').unwrap_or("");
    let relative = if container_segments == 0 {
        below_authority
    } else {
        below_authority
            .split_once('/')
            .map_or("", |(_container, key)| key)
    };

    Ok(RowParts {
        uri: format!("{scheme}://{file}"),
        relative,
    })
}

/// Reconstruct the exact URI the original call used from an op-log row's
/// `endpoint` and `file` columns.
///
/// Handles every shape s3dlio writes for `s3://`, `az://`, `gs://`, Azure `https://`,
/// `file://` and `direct://` (see the module documentation). A row that does not
/// identify its URI unambiguously (a legacy S3 row without a bucket, an empty or
/// schemeless endpoint, a `file` that disagrees with `endpoint`) is a
/// [`RowUriError`]; it is never guessed and never dropped silently.
///
/// A hand-written row whose `file` already is the full URI (it begins with
/// `endpoint` and contains a scheme) is returned as `file`.
///
/// # Examples
/// ```
/// use s3dlio_oplog::uri::row_uri;
///
/// assert_eq!(row_uri("s3://srcbkt", "srcbkt/dir/obj1.bin").unwrap(), "s3://srcbkt/dir/obj1.bin");
/// assert_eq!(row_uri("file://", "/tmp/data/x.dat").unwrap(), "file:///tmp/data/x.dat");
/// // legacy S3 row: server + key, no bucket
/// assert!(row_uri("http://127.0.0.1:9000", "dir/obj1.bin").is_err());
/// ```
pub fn row_uri(endpoint: &str, file: &str) -> Result<String, RowUriError> {
    if !endpoint.is_empty() && file.contains("://") && starts_with_unit(file, endpoint) {
        return Ok(file.to_string());
    }
    split_row(endpoint, file).map(|p| p.uri)
}

/// Translate an op-log row to a new target backend
///
/// Simple 1:1 remapping that preserves the key below the original bucket/container:
/// - Row: `endpoint = s3://srcbkt`, `file = srcbkt/dir/obj1.bin`
/// - Target: `s3://replaybkt/`
/// - Result: `s3://replaybkt/dir/obj1.bin`
///
/// For the shapes s3dlio writes (see the module documentation) the bucket, container and
/// server are dropped and only the key is appended to the target. As in earlier versions,
/// a hand-written row whose `file` literally starts with `endpoint` has `endpoint`
/// stripped from it. Any other row (for example a legacy `S3Ops` row, `endpoint =
/// http://server`, `file = dir/key`) is translated exactly as earlier versions did: the
/// target supplies the destination, so the original URI is not needed and the row is not an
/// error. (Replay *without* a target does need the original URI; see [`row_uri`].)
///
/// # Arguments
/// * `file` - The row's `file` column
/// * `endpoint` - The row's `endpoint` column
/// * `target` - New target URI prefix
///
/// # Examples
/// ```
/// use s3dlio_oplog::uri::translate_uri;
///
/// let result = translate_uri("srcbkt/dir/obj1.bin", "s3://srcbkt", "s3://replaybkt/").unwrap();
/// assert_eq!(result, "s3://replaybkt/dir/obj1.bin");
///
/// // hand-written row: file carries the endpoint as a prefix
/// let result = translate_uri("/bucket/data/file.bin", "/bucket/", "s3://newbucket").unwrap();
/// assert_eq!(result, "s3://newbucket/data/file.bin");
/// ```
pub fn translate_uri(file: &str, endpoint: &str, target: &str) -> Result<String> {
    let relative = if !endpoint.is_empty() && starts_with_unit(file, endpoint) {
        &file[endpoint.len()..]
    } else if let Ok(parts) = split_row(endpoint, file) {
        parts.relative
    } else {
        // Not a shape s3dlio writes (a legacy `S3Ops` row with no bucket, an empty or schemeless
        // endpoint, ...). A target supplies the destination, so the original URI is not needed:
        // translate exactly as every earlier version did.
        file.strip_prefix(endpoint).unwrap_or(file)
    };
    let clean = relative.trim_start_matches('/');

    // Construct new URI with target
    let target_clean = target.trim_end_matches('/');
    Ok(format!("{}/{}", target_clean, clean))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_uri_basic() {
        let result = translate_uri("/bucket/data/file.bin", "/bucket/", "s3://newbucket").unwrap();
        assert_eq!(result, "s3://newbucket/data/file.bin");
    }

    #[test]
    fn test_translate_uri_with_trailing_slashes() {
        let result = translate_uri("/bucket/data/file.bin", "/bucket/", "s3://newbucket/").unwrap();
        assert_eq!(result, "s3://newbucket/data/file.bin");
    }

    #[test]
    fn test_translate_uri_no_prefix_match() {
        // When file doesn't start with endpoint, use file as-is (as in every earlier version)
        let result = translate_uri("/other/path/file.bin", "/bucket/", "s3://newbucket").unwrap();
        assert_eq!(result, "s3://newbucket/other/path/file.bin");
    }

    #[test]
    fn test_translate_uri_different_backends() {
        // File to S3
        let result = translate_uri(
            "file:///tmp/data/obj.dat",
            "file:///tmp/",
            "s3://bucket/prefix",
        )
        .unwrap();
        assert_eq!(result, "s3://bucket/prefix/data/obj.dat");

        // S3 to Azure
        let result = translate_uri(
            "s3://source/data/obj.dat",
            "s3://source/",
            "az://account/container",
        )
        .unwrap();
        assert_eq!(result, "az://account/container/data/obj.dat");

        // Azure to Direct I/O
        let result = translate_uri(
            "az://account/container/data/obj.dat",
            "az://account/container/",
            "direct:///nvme/cache",
        )
        .unwrap();
        assert_eq!(result, "direct:///nvme/cache/data/obj.dat");
    }

    #[test]
    fn test_prefix_must_end_on_a_boundary_for_row_uri() {
        // `s3://bkt` is a string prefix of `s3://bkt2/x` but not a path prefix, so the full-URI
        // shortcut does not apply and the inconsistent row cannot be rebuilt without a target.
        assert!(matches!(
            row_uri("s3://bkt", "s3://bkt2/x"),
            Err(RowUriError::Inconsistent { .. })
        ));
    }

    #[test]
    fn test_row_uri_writer_shapes() {
        assert_eq!(
            row_uri("s3://srcbkt", "srcbkt/dir/obj1.bin").unwrap(),
            "s3://srcbkt/dir/obj1.bin"
        );
        assert_eq!(
            row_uri("az://acct", "acct/cont/k").unwrap(),
            "az://acct/cont/k"
        );
        assert_eq!(row_uri("gs://b", "b/k").unwrap(), "gs://b/k");
        assert_eq!(row_uri("direct://", "/mnt/x").unwrap(), "direct:///mnt/x");
        // full URI in `file` (hand-written rows)
        assert_eq!(
            row_uri("s3://old", "s3://old/data/obj.dat").unwrap(),
            "s3://old/data/obj.dat"
        );
    }

    #[test]
    fn test_row_uri_errors_are_typed() {
        assert!(matches!(
            row_uri("", "key"),
            Err(RowUriError::NoEndpoint { .. })
        ));
        assert!(matches!(
            row_uri("unknown", "name"),
            Err(RowUriError::NoScheme { .. })
        ));
        assert!(matches!(
            row_uri("ftp://h", "h/k"),
            Err(RowUriError::UnsupportedScheme { .. })
        ));
        assert!(matches!(
            row_uri("http://127.0.0.1:19750", "dir/obj1.bin"),
            Err(RowUriError::BucketNotRecorded { .. })
        ));
        assert!(matches!(
            row_uri("s3://bkt", "other/k"),
            Err(RowUriError::Inconsistent { .. })
        ));
    }
}
