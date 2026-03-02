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
use tracing::{debug, trace};

use crate::s3_client::run_on_global_rt;
use crate::s3_utils::{list_buckets, BucketInfo};

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

#[cfg(feature = "gcs-community")]
async fn list_gcs_async(project: &str) -> Result<Vec<ContainerInfo>> {
    use gcloud_storage::http::buckets::list::ListBucketsRequest;

    debug!("GCS LIST BUCKETS (community): project={}", project);
    let gcs = crate::gcs_client::GcsClient::new()
        .await
        .context("Failed to initialise GCS community client")?;

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

#[cfg(not(feature = "gcs-community"))]
async fn list_gcs_async(project: &str) -> Result<Vec<ContainerInfo>> {
    use google_cloud_gax::paginator::ItemPaginator;

    debug!("GCS LIST BUCKETS (official/gRPC): project={}", project);
    let gcs = crate::google_gcs_client::GcsClient::new()
        .await
        .context("Failed to initialise GCS official client")?;

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
        assert!(
            err.to_string().contains("project ID"),
            "should ask for project ID, got: {}",
            err
        );
    }

    #[test]
    fn test_list_containers_azure_no_account() {
        let err = list_containers("az://").unwrap_err();
        assert!(
            err.to_string().contains("account name"),
            "should ask for account name, got: {}",
            err
        );
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
}
