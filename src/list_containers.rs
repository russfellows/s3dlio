// src/list_containers.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Backend-agnostic "list top-level containers" for every storage backend.
//!
//! The concept of a "bucket" or "container" differs by backend:
//!
//! | Backend        | Concept                  | Example URI        |
//! |----------------|--------------------------|--------------------|
//! | AWS S3         | Bucket                   | `s3://`            |
//! | S3-compatible  | Bucket (same SDK)        | `s3://` + endpoint |
//! | Google GCS     | Bucket (in a project)    | `gs://`            |
//! | Azure Blob     | Container (in account)   | `az://account`     |
//! | Local file     | Top-level subdirectory   | `file:///path/`    |
//! | Direct I/O     | Top-level subdirectory   | `direct:///path/`  |
//!
//! # Usage
//!
//! ```ignore
//! // S3 (or S3-compatible via AWS_ENDPOINT_URL)
//! let containers = list_containers("s3://")?;
//!
//! // The buckets of one specific S3-compatible server (explicit port required)
//! let containers = list_containers("s3://minio.local:9000/")?;
//!
//! // GCS (requires GOOGLE_CLOUD_PROJECT or GCLOUD_PROJECT env var)
//! let containers = list_containers("gs://")?;
//!
//! // Azure (account name in URI)
//! let containers = list_containers("az://mystorageaccount")?;
//!
//! // Local filesystem (lists top-level subdirectories)
//! let containers = list_containers("file:///mnt/data")?;
//! ```

use anyhow::{bail, Context, Result};
#[cfg(feature = "backend-gcs")]
use std::sync::Arc;
#[cfg(feature = "backend-gcs")]
use tokio::sync::OnceCell;
use tracing::{debug, trace};

#[cfg(any(feature = "backend-gcs", feature = "backend-azure"))]
use crate::s3_client::run_on_global_rt;
use crate::s3_utils::{list_buckets_at, s3_server_of_first_segment, BucketInfo, S3Server};

#[cfg(feature = "backend-gcs")]
static GCS_OFFICIAL_LIST_CLIENT: OnceCell<Arc<crate::google_gcs_client::GcsClient>> =
    OnceCell::const_new();

#[cfg(feature = "backend-gcs")]
async fn get_list_gcs_official_client() -> Result<Arc<crate::google_gcs_client::GcsClient>> {
    let client = GCS_OFFICIAL_LIST_CLIENT
        .get_or_try_init(|| async {
            let gcs = crate::google_gcs_client::GcsClient::new()
                .await
                .context("Failed to initialise GCS official client")?;
            Ok::<Arc<crate::google_gcs_client::GcsClient>, anyhow::Error>(Arc::new(gcs))
        })
        .await?;
    Ok(Arc::clone(client))
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A named top-level storage container — bucket, container, or directory —
/// with its canonical URI and optional creation/modification date.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerInfo {
    /// Human-readable / short name (bucket name, container name, dir name).
    pub name: String,
    /// Fully-qualified URI that can be passed back to s3dlio operations.
    pub uri: String,
    /// Creation or last-modified date, if available.
    pub creation_date: Option<String>,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// List top-level containers (buckets, containers, or directories) for the
/// storage backend identified by `uri`.
///
/// The dispatch is based entirely on the URI **scheme**:
///
/// - `s3://` → AWS S3 or S3-compatible (reads `AWS_ENDPOINT_URL_S3` /
///   `AWS_ENDPOINT_URL` etc.)
/// - `s3://host:port/` (or `[ipv6]:port`) → the buckets of THAT server; a portless
///   IP address (`s3://10.0.0.5/`) is an error; `s3://bucket/` and other
///   portless names list the global server (issue #200)
/// - `gs://` / `gcs://` → Google Cloud Storage (reads `GOOGLE_CLOUD_PROJECT`)
/// - `az://account` → Azure Blob Storage containers for the given account
/// - `file://[/path]` → top-level subdirectories under `/path`
/// - `direct://[/path]` → same as `file://`
///
/// # Errors
///
/// Returns an error if the scheme is unrecognised, credentials are missing,
/// or the storage service returns a failure.
pub fn list_containers(uri: &str) -> Result<Vec<ContainerInfo>> {
    // Normalise: strip trailing slashes from any path component, but preserve
    // the scheme's double-slash.  We match on the *original* uri for scheme
    // detection so that bare "az://" or "file://" are handled correctly.
    let original = uri.trim();
    debug!("list_containers: dispatching uri={:?}", original);

    if original.is_empty()
        || original == "s3:"
        || original == "s3://"
        || original.starts_with("s3://")
    {
        list_s3_containers(original)
    } else if original == "gs:"
        || original == "gs://"
        || original == "gcs:"
        || original == "gcs://"
        || original.starts_with("gs://")
        || original.starts_with("gcs://")
    {
        list_gcs_containers()
    } else if original == "az:" || original == "az://" || original.starts_with("az://") {
        list_azure_containers(original)
    } else if original.starts_with("file://") {
        list_fs_dirs(original.strip_prefix("file://").unwrap_or("/"))
    } else if original.starts_with("direct://") {
        list_fs_dirs(original.strip_prefix("direct://").unwrap_or("/"))
    } else {
        bail!(
            "Unsupported URI scheme: '{}'\n\
             Supported: s3://, gs://, gcs://, az://account, file:///path, direct:///path",
            original
        )
    }
}

// ---------------------------------------------------------------------------
// S3 / S3-compatible
// ---------------------------------------------------------------------------

/// The server an S3 listing URI names, by the shared server-vs-bucket rule
/// ([`crate::s3_utils::classify_s3_first_segment`], issue #200):
///
/// - `s3://host:port/` (or `[ipv6]:port`) → `Some(server)`: list THAT server.
/// - `s3://10.0.0.5/` (a portless IP) → `Err`, the #178 error: add `:port`.
/// - `""`, `s3:`, `s3://`, `s3://bucket/`, `s3://data.example.com/` → `None`: list
///   the process-global server (`AWS_ENDPOINT_URL_S3` / `AWS_ENDPOINT_URL`, or AWS).
fn s3_listing_server(uri: &str) -> Result<Option<S3Server>> {
    let Some(rest) = uri.strip_prefix("s3://") else {
        return Ok(None);
    };
    let first_segment = rest.split('/').next().unwrap_or("");
    if first_segment.is_empty() {
        return Ok(None);
    }
    s3_server_of_first_segment(first_segment).with_context(|| format!("invalid S3 URI '{uri}'"))
}

fn list_s3_containers(uri: &str) -> Result<Vec<ContainerInfo>> {
    let server = s3_listing_server(uri)?;
    let buckets: Vec<BucketInfo> =
        list_buckets_at(server.as_ref()).context("Failed to list S3 buckets")?;
    // A bucket listed from a named server is addressed through that server, so the
    // URI can be passed straight back to s3dlio operations.
    let uri_prefix = match &server {
        Some(server) => format!("s3://{}/", server.authority),
        None => "s3://".to_string(),
    };
    Ok(buckets
        .into_iter()
        .map(|b| ContainerInfo {
            uri: format!("{uri_prefix}{}", b.name),
            name: b.name,
            creation_date: Some(b.creation_date),
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Google Cloud Storage
// ---------------------------------------------------------------------------

#[cfg(feature = "backend-gcs")]
fn list_gcs_containers() -> Result<Vec<ContainerInfo>> {
    crate::dotenv::apply();
    let project = std::env::var("GOOGLE_CLOUD_PROJECT")
        .or_else(|_| std::env::var("GCLOUD_PROJECT"))
        .or_else(|_| std::env::var("GCS_PROJECT"))
        .map_err(|_| {
            anyhow::anyhow!(
                "GCS bucket listing requires a project ID.\n\
             Set one of: GOOGLE_CLOUD_PROJECT, GCLOUD_PROJECT, or GCS_PROJECT"
            )
        })?;
    debug!("list_containers GCS: resolved project={:?}", project);

    run_on_global_rt(async move { list_gcs_async(&project).await })
}

#[cfg(not(feature = "backend-gcs"))]
fn list_gcs_containers() -> Result<Vec<ContainerInfo>> {
    bail!("GCS backend is not enabled in this build. Rebuild with feature 'backend-gcs' or 'full-backends'.")
}

/// A GCS bucket creation time as an RFC 3339 string (#263).
///
/// GCS reports it as Google's `Timestamp`, whose `Debug` form is a struct dump
/// (`Timestamp { seconds: .., nanos: .. }`), not a date.  Google's `String::from(Timestamp)`
/// gives RFC 3339 in UTC and keeps the fraction when there is one.  An absent time stays absent.
#[cfg(feature = "backend-gcs")]
fn bucket_time_rfc3339<T>(time: Option<T>) -> Option<String>
where
    T: Into<String>,
{
    time.map(Into::into)
}

#[cfg(feature = "backend-gcs")]
async fn list_gcs_async(project: &str) -> Result<Vec<ContainerInfo>> {
    use google_cloud_gax::paginator::ItemPaginator;

    debug!("GCS LIST BUCKETS (official/gRPC): project={}", project);
    let gcs = get_list_gcs_official_client().await?;

    let parent = format!("projects/{}", project);
    let mut containers = Vec::new();
    let mut pages = gcs
        .control_ref()
        .list_buckets()
        .set_parent(&parent)
        .by_item();

    while let Some(item) = pages.next().await {
        let bucket = item.map_err(|e| anyhow::anyhow!("GCS list_buckets: {}", e))?;
        let short_name = bucket
            .name
            .strip_prefix("projects/_/buckets/")
            .unwrap_or(&bucket.name)
            .to_string();
        trace!("GCS LIST BUCKETS: bucket={}", short_name);
        let date = bucket_time_rfc3339(bucket.create_time);
        containers.push(ContainerInfo {
            uri: format!("gs://{}", short_name),
            name: short_name,
            creation_date: date,
        });
    }

    debug!("GCS LIST BUCKETS: found {} buckets", containers.len());
    Ok(containers)
}

// ---------------------------------------------------------------------------
// Azure Blob Storage
// ---------------------------------------------------------------------------

#[cfg(feature = "backend-azure")]
fn list_azure_containers(uri: &str) -> Result<Vec<ContainerInfo>> {
    crate::dotenv::apply();
    let account = uri
        .strip_prefix("az://")
        .map(|s| s.trim_end_matches('/'))
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Azure container listing requires an account name in the URI.\n\
             Example: s3dlio list-buckets az://mystorageaccount"
            )
        })?
        .to_string();

    run_on_global_rt(async move { list_azure_async(&account).await })
}

#[cfg(not(feature = "backend-azure"))]
fn list_azure_containers(_uri: &str) -> Result<Vec<ContainerInfo>> {
    bail!("Azure backend is not enabled in this build. Rebuild with feature 'backend-azure' or 'full-backends'.")
}

#[cfg(feature = "backend-azure")]
async fn list_azure_async(account: &str) -> Result<Vec<ContainerInfo>> {
    debug!("Azure LIST CONTAINERS: account={}", account);
    let pairs = crate::azure_client::list_account_containers(account)
        .await
        .context("Failed to list Azure containers")?;
    let mut containers = Vec::with_capacity(pairs.len());
    for (name, date) in pairs {
        trace!("Azure LIST CONTAINERS: container={}", name);
        containers.push(ContainerInfo {
            uri: format!("az://{}/{}", account, name),
            name,
            creation_date: date,
        });
    }
    debug!(
        "Azure LIST CONTAINERS: {} container(s) found",
        containers.len()
    );
    Ok(containers)
}

// ---------------------------------------------------------------------------
// Local filesystem / Direct I/O
// ---------------------------------------------------------------------------

fn list_fs_dirs(raw_path: &str) -> Result<Vec<ContainerInfo>> {
    let path_str = if raw_path.is_empty() || raw_path == "/" {
        ".".to_string()
    } else {
        raw_path.to_string()
    };

    debug!("FS LIST DIRS: path={}", path_str);

    let mut entries: Vec<ContainerInfo> = std::fs::read_dir(&path_str)
        .with_context(|| format!("Failed to read directory '{}'", path_str))?
        .filter_map(|res| {
            let entry = res.ok()?;
            let meta = entry.metadata().ok()?;
            if !meta.is_dir() {
                return None;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                return None;
            }
            let abs = entry.path().to_string_lossy().into_owned();
            trace!("FS LIST DIRS: directory={:?}", name);
            Some(ContainerInfo {
                name,
                uri: format!("file://{}", abs),
                creation_date: None,
            })
        })
        .collect();

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    debug!("FS LIST DIRS: found {} directories", entries.len());
    Ok(entries)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_containers_unsupported_scheme() {
        let err = list_containers("ftp://someserver").unwrap_err();
        assert!(err.to_string().contains("Unsupported URI scheme"));
    }

    #[test]
    fn test_list_fs_dirs_nonexistent() {
        let err = list_fs_dirs("/nonexistent/path/s3dlio_test_xyz123").unwrap_err();
        assert!(err.to_string().contains("Failed to read directory"));
    }

    #[test]
    fn test_list_fs_dirs_temp() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir(root.join("alpha")).unwrap();
        std::fs::create_dir(root.join("beta")).unwrap();
        std::fs::write(root.join("file.txt"), b"data").unwrap();
        std::fs::create_dir(root.join(".hidden")).unwrap();

        let entries = list_fs_dirs(&root.to_string_lossy()).unwrap();
        let names: Vec<&str> = entries.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
        assert!(entries[0].uri.starts_with("file://"));
        assert!(entries[0].creation_date.is_none());
    }

    /// A bucket whose creation time is parsed from `rfc3339`, built without naming the timestamp
    /// type (it lives in a crate this one does not depend on directly).
    #[cfg(feature = "backend-gcs")]
    fn bucket_created(rfc3339: &str) -> google_cloud_storage::model::Bucket {
        let mut bucket = google_cloud_storage::model::Bucket::default();
        bucket.create_time = Some(rfc3339.try_into().expect("valid timestamp"));
        bucket
    }

    /// A synthetic UTC creation time is an RFC 3339 string, not a struct dump (#263).
    #[test]
    #[cfg(feature = "backend-gcs")]
    fn utc_bucket_time_is_rfc3339() {
        let bucket = bucket_created("2025-05-16T09:46:12Z");
        assert_eq!(
            bucket_time_rfc3339(bucket.create_time).as_deref(),
            Some("2025-05-16T09:46:12Z")
        );
    }

    /// The fraction is kept, down to the nanosecond.
    #[test]
    #[cfg(feature = "backend-gcs")]
    fn fractional_bucket_time_keeps_its_precision() {
        for (input, want) in [
            ("2025-05-16T09:46:12.5Z", "2025-05-16T09:46:12.5Z"),
            ("2025-05-16T09:46:12.123456Z", "2025-05-16T09:46:12.123456Z"),
            (
                "2025-05-16T09:46:12.123456789Z",
                "2025-05-16T09:46:12.123456789Z",
            ),
        ] {
            let bucket = bucket_created(input);
            assert_eq!(
                bucket_time_rfc3339(bucket.create_time).as_deref(),
                Some(want),
                "{input}"
            );
        }
    }

    /// The output parses back to the same instant, so nothing but formatting changed.
    #[test]
    #[cfg(feature = "backend-gcs")]
    fn bucket_time_round_trips_to_the_same_instant() {
        let bucket = bucket_created("2031-02-03T04:05:06.007008009Z");
        let text = bucket_time_rfc3339(bucket.create_time).expect("present");
        let parsed = chrono::DateTime::parse_from_rfc3339(&text).expect("RFC 3339");
        let want_secs = chrono::NaiveDate::from_ymd_opt(2031, 2, 3)
            .and_then(|d| d.and_hms_opt(4, 5, 6))
            .expect("valid date")
            .and_utc()
            .timestamp();
        assert_eq!(parsed.timestamp(), want_secs);
        assert_eq!(parsed.timestamp_subsec_nanos(), 7_008_009);
    }

    /// An absent creation time stays absent; it is never an empty string or a zero time.
    #[test]
    #[cfg(feature = "backend-gcs")]
    fn absent_bucket_time_stays_absent() {
        let bucket = google_cloud_storage::model::Bucket::default();
        assert_eq!(bucket_time_rfc3339(bucket.create_time), None);
    }

    #[test]
    #[cfg(feature = "backend-azure")]
    #[ignore = "requires AZURE_BLOB_ACCOUNT, live Azure credentials, and network access"]
    fn test_list_containers_azure_live() {
        let account = std::env::var("AZURE_BLOB_ACCOUNT")
            .expect("AZURE_BLOB_ACCOUNT must be set for the ignored live Azure test");
        let containers = list_containers(&format!("az://{account}"))
            .expect("live Azure container listing should succeed");
        let expected_prefix = format!("az://{account}/");
        for container in containers {
            // The message deliberately omits the account name and the URI (which embeds
            // it): CodeQL rust/cleartext-logging treats the account as sensitive.
            assert!(
                container.uri.starts_with(&expected_prefix),
                "an Azure container URI does not use the configured account prefix"
            );
        }
    }

    #[test]
    fn test_list_containers_gs_missing_project() {
        std::env::remove_var("GOOGLE_CLOUD_PROJECT");
        std::env::remove_var("GCLOUD_PROJECT");
        std::env::remove_var("GCS_PROJECT");
        let err = list_containers("gs://").unwrap_err();
        if cfg!(feature = "backend-gcs") {
            assert!(
                err.to_string().contains("project ID"),
                "should ask for project ID, got: {}",
                err
            );
        } else {
            assert!(
                err.to_string().contains("GCS backend is not enabled"),
                "should explain backend-gcs feature requirement, got: {}",
                err
            );
        }
    }

    #[test]
    fn test_list_containers_azure_no_account() {
        let err = list_containers("az://").unwrap_err();
        if cfg!(feature = "backend-azure") {
            assert!(
                err.to_string().contains("account name"),
                "should ask for account name, got: {}",
                err
            );
        } else {
            assert!(
                err.to_string().contains("Azure backend is not enabled"),
                "should explain backend-azure feature requirement, got: {}",
                err
            );
        }
    }

    #[test]
    fn test_list_containers_file_scheme() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("container")).unwrap();
        let uri = format!("file://{}", tmp.path().display());

        let containers = list_containers(&uri).expect("file:// URI should list directories");
        assert_eq!(containers.len(), 1);
        assert_eq!(containers[0].name, "container");
    }

    // ------------------------------------------------------------------
    // ContainerInfo fields — validates the contract used by the Python API
    // (issue #133: list_containers exposed to Python as list of dicts with
    // keys "name", "uri", "creation_date")
    // ------------------------------------------------------------------

    /// Every ContainerInfo returned from a file:// URI must have a non-empty
    /// name, a URI that starts with "file://", and creation_date == None
    /// (filesystem stat doesn't provide creation time on Linux).
    #[test]
    fn test_container_info_fields_file_uri() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir(root.join("bucket-a")).unwrap();
        std::fs::create_dir(root.join("bucket-b")).unwrap();

        let uri = format!("file://{}", root.display());
        let containers = list_containers(&uri).expect("list_containers should succeed for file://");

        assert!(
            !containers.is_empty(),
            "should return at least one container"
        );
        for c in &containers {
            assert!(!c.name.is_empty(), "name must be non-empty: {:?}", c);
            assert!(
                c.uri.starts_with("file://"),
                "uri must start with 'file://', got: {}",
                c.uri
            );
            // creation_date is None for filesystem entries (no creation time on Linux)
            assert!(
                c.creation_date.is_none(),
                "creation_date should be None for filesystem entries"
            );
        }

        // Names should be sorted and match the created directories
        let names: Vec<&str> = containers.iter().map(|c| c.name.as_str()).collect();
        assert!(
            names.contains(&"bucket-a"),
            "bucket-a missing from {:?}",
            names
        );
        assert!(
            names.contains(&"bucket-b"),
            "bucket-b missing from {:?}",
            names
        );
    }

    /// The URI field of each ContainerInfo must be re-usable as a storage URI
    /// (can be passed back to list_containers to list the container's contents).
    #[test]
    fn test_container_info_uri_is_reusable() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sub = root.join("my-bucket");
        std::fs::create_dir(&sub).unwrap();
        // Put a directory inside the sub-directory so re-listing is non-empty
        std::fs::create_dir(sub.join("nested")).unwrap();

        let uri = format!("file://{}", root.display());
        let containers = list_containers(&uri).expect("outer list should succeed");
        let bucket_entry = containers
            .iter()
            .find(|c| c.name == "my-bucket")
            .expect("my-bucket should appear in the listing");

        // Re-use the URI from ContainerInfo as input to list_containers
        let inner = list_containers(&bucket_entry.uri)
            .expect("ContainerInfo.uri must be a valid re-usable list_containers input");
        let inner_names: Vec<&str> = inner.iter().map(|c| c.name.as_str()).collect();
        assert!(
            inner_names.contains(&"nested"),
            "nested dir missing from re-listed container: {:?}",
            inner_names
        );
    }
}
