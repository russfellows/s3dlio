// Azure backend tests - now always enabled

use anyhow::{Context, Result};
use bytes::Bytes;
use rand::{rng, Rng};
use rand::distr::Alphanumeric;
use std::env;
use s3dlio::azure_client::AzureBlob;

fn req_env(key: &str) -> Result<String> {
    env::var(key).with_context(|| format!("Missing required env var {}", key))
}

fn opt_env(key: &str) -> Option<String> {
    env::var(key).ok()
}

fn random_key(prefix: &str) -> String {
    let mut r = rng();
    let suffix: String = (&mut r)
        .sample_iter(&Alphanumeric)
        .take(10)
        .map(char::from)
        .collect();
    format!("{prefix}-{suffix}")
}

async fn mk_client() -> Result<AzureBlob> {
    let container = req_env("AZURE_BLOB_CONTAINER")?;
    if let Some(url) = opt_env("AZURE_BLOB_ACCOUNT_URL") {
        if let Some(acct) = opt_env("AZURE_BLOB_ACCOUNT") {
            // crude parse: https://{account}.blob.core.windows.net
            let host = url.strip_prefix("https://").unwrap_or(&url);
            let acc_from_url = host.split('.').next().unwrap_or("");
            anyhow::ensure!(acc_from_url == acct,
                "AZURE_BLOB_ACCOUNT_URL points to '{acc_from_url}', but AZURE_BLOB_ACCOUNT is '{acct}'");
        }
        Ok(AzureBlob::with_default_credential_from_url(&url, &container)?)
    } else {
        let acct = req_env("AZURE_BLOB_ACCOUNT")?;
        Ok(AzureBlob::with_default_credential(&acct, &container)?)
    }
}


#[tokio::test]
async fn multi_blob_roundtrip_and_cleanup() -> Result<()> {
    // Graceful skip if not configured
    if env::var("AZURE_BLOB_CONTAINER").is_err() ||
        (env::var("AZURE_BLOB_ACCOUNT").is_err() && env::var("AZURE_BLOB_ACCOUNT_URL").is_err())
    {
        eprintln!("SKIP: set AZURE_BLOB_CONTAINER and (AZURE_BLOB_ACCOUNT or AZURE_BLOB_ACCOUNT_URL)");
        return Ok(());
    }

    let client = mk_client().await?;
    let prefix = random_key("s3dlio-multi");
    let n_blobs = 5usize;

    // Build some sample contents of varying sizes
    let mut keys: Vec<String> = Vec::new();
    for i in 0..n_blobs {
        let key = format!("{}/blob-{:02}.bin", &prefix, i);
        let size = (i + 1) * 64 * 1024; // 64KiB, 128KiB, ...
        let mut data = vec![0u8; size];
        for (idx, b) in data.iter_mut().enumerate() { *b = ((idx + i) % 251) as u8; }
        client.put(&key, Bytes::from(data), true).await?;
        keys.push(key);
    }

    // List by prefix and verify we see exactly our blobs
    let listed = client.list(Some(&prefix)).await?;
    for k in &keys {
        assert!(listed.iter().any(|x| x == k), "missing {k} in list()");
    }

    // Download each and quick spot check length
    for (i, k) in keys.iter().enumerate() {
        let got = client.get(k).await?;
        let expected_len = (i + 1) * 64 * 1024;
        assert_eq!(got.len(), expected_len);
    }

    // Cleanup
    client.delete_objects(&keys).await?;

    // Verify deletion
    let listed_after = client.list(Some(&prefix)).await?;
    assert!(
        listed_after.is_empty(),
        "expected no keys under prefix {prefix}, got: {listed_after:?}"
    );

    Ok(())
}

