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
use crate::s3_utils::{list_buckets, BucketInfo};

#[cfg(all(feature = "backend-gcs", feature = "gcs-community"))]
static GCS_COMMUNITY_LIST_CLIENT: OnceCell<Arc<crate::gcs_client::GcsClient>> = OnceCell::const_new();

#[cfg(all(feature = "backend-gcs", feature = "gcs-community"))]
async fn get_list_gcs_community_client() -> Result<Arc<crate::gcs_client::GcsClient>> {
    let client = GCS_COMMUNITY_LIST_CLIENT
        .get_or_try_init(|| async {
            let gcs = crate::gcs_client::GcsClient::new()
                .await
                .context("Failed to initialise GCS community client")?;
            Ok::<Arc<crate::gcs_client::GcsClient>, anyhow::Error>(Arc::new(gcs))
        })
        .await?;
    Ok(Arc::clone(client))
}

#[cfg(all(feature = "backend-gcs", not(feature = "gcs-community")))]
static GCS_OFFICIAL_LIST_CLIENT: OnceCell<Arc<crate::google_gcs_client::GcsClient>> = OnceCell::const_new();

#[cfg(all(feature = "backend-gcs", not(feature = "gcs-community")))]
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
/// - `s3://` → AWS S3 or S3-compatible (reads `AWS_ENDPOINT_URL` etc.)
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
        list_s3_containers()
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

fn list_s3_containers() -> Result<Vec<ContainerInfo>> {
    let buckets: Vec<BucketInfo> = list_buckets().context("Failed to list S3 buckets")?;
    Ok(buckets
        .into_iter()
        .map(|b| ContainerInfo {
            uri: format!("s3://{}", b.name),
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
    dotenvy::dotenv().ok();
    let project = std::env::var("GOOGLE_CLOUD_PROJECT")
        .or_else(|_| std::env::var("GCLOUD_PROJECT"))
        .or_else(|_| std::env::var("GCS_PROJECT"))
        .map_err(|_| anyhow::anyhow!(
            "GCS bucket listing requires a project ID.\n\
             Set one of: GOOGLE_CLOUD_PROJECT, GCLOUD_PROJECT, or GCS_PROJECT"
        ))?;
    debug!("list_containers GCS: resolved project={:?}", project);

    run_on_global_rt(async move { list_gcs_async(&project).await })
}

#[cfg(not(feature = "backend-gcs"))]
fn list_gcs_containers() -> Result<Vec<ContainerInfo>> {
    bail!("GCS backend is not enabled in this build. Rebuild with feature 'backend-gcs' or 'full-backends'.")
}

#[cfg(all(feature = "backend-gcs", feature = "gcs-community"))]
async fn list_gcs_async(project: &str) -> Result<Vec<ContainerInfo>> {
    use gcloud_storage::http::buckets::list::ListBucketsRequest;

    debug!("GCS LIST BUCKETS (community): project={}", project);
    let gcs = get_list_gcs_community_client().await?;

    let mut containers = Vec::new();
    let mut page_token: Option<String> = None;

    loop {
        let req = ListBucketsRequest {
            project: project.to_string(),
            page_token: page_token.clone(),
            ..Default::default()
        };

        let resp = gcs
            .inner_client()
            .list_buckets(&req)
            .await
            .map_err(|e| anyhow::anyhow!("GCS list_buckets failed: {}", e))?;

        for bucket in resp.items {
            trace!("GCS LIST BUCKETS: bucket={}", bucket.name);
            let date = bucket
                .time_created
                .map(|t| t.to_string())
                .or_else(|| bucket.updated.map(|t| t.to_string()));
            containers.push(ContainerInfo {
                uri: format!("gs://{}", bucket.name),
                name: bucket.name,
                creation_date: date,
            });
        }

        match resp.next_page_token {
            Some(token) if !token.is_empty() => page_token = Some(token),
            _ => break,
        }
    }

    debug!("GCS LIST BUCKETS: found {} buckets", containers.len());
    Ok(containers)
}

#[cfg(all(feature = "backend-gcs", not(feature = "gcs-community")))]
async fn list_gcs_async(project: &str) -> Result<Vec<ContainerInfo>> {
    use google_cloud_gax::paginator::ItemPaginator;

    debug!("GCS LIST BUCKETS (official/gRPC): project={}", project);
    let gcs = get_list_gcs_official_client().await?;

    let parent = format!("projects/{}", project);
    let mut containers = Vec::new();
    let mut pages = gcs.control_ref().list_buckets().set_parent(&parent).by_item();

    while let Some(item) = pages.next().await {
        let bucket = item.map_err(|e| anyhow::anyhow!("GCS list_buckets: {}", e))?;
        let short_name = bucket
            .name
            .strip_prefix("projects/_/buckets/")
            .unwrap_or(&bucket.name)
            .to_string();
        trace!("GCS LIST BUCKETS: bucket={}", short_name);
        let date = bucket.create_time.map(|t| format!("{:?}", t));
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
    let account = uri
        .strip_prefix("az://")
        .map(|s| s.trim_end_matches('/'))
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!(
            "Azure container listing requires an account name in the URI.\n\
             Example: s3dlio list-buckets az://mystorageaccount"
        ))?
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
    debug!("Azure LIST CONTAINERS: {} container(s) found", containers.len());
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

    #[test]
    fn test_list_containers_s3_scheme_not_unsupported() {
        let result = list_containers("s3://");
        if let Err(e) = result {
            assert!(
                !e.to_string().contains("Unsupported URI scheme"),
                "s3:// must not hit unsupported-scheme error"
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
        // file:// with a real path should successfully list or fail with a path error, not scheme error.
        let result = list_containers("file:///tmp");
        if let Err(e) = result {
            assert!(
                !e.to_string().contains("Unsupported URI scheme"),
                "file:// must not hit unsupported-scheme error"
            );
        }
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

        assert!(!containers.is_empty(), "should return at least one container");
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
        assert!(names.contains(&"bucket-a"), "bucket-a missing from {:?}", names);
        assert!(names.contains(&"bucket-b"), "bucket-b missing from {:?}", names);
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
        let bucket_entry = containers.iter().find(|c| c.name == "my-bucket")
            .expect("my-bucket should appear in the listing");

        // Re-use the URI from ContainerInfo as input to list_containers
        let inner = list_containers(&bucket_entry.uri)
            .expect("ContainerInfo.uri must be a valid re-usable list_containers input");
        let inner_names: Vec<&str> = inner.iter().map(|c| c.name.as_str()).collect();
        assert!(inner_names.contains(&"nested"), "nested dir missing from re-listed container: {:?}", inner_names);
    }
}
