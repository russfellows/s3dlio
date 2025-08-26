// Azure backend tests - now always enabled

use anyhow::{Context, Result};
use bytes::Bytes;
use rand::{rng, Rng};
use rand::distr::Alphanumeric;
use std::{env, fs, path::Path};
use s3dlio::azure_client::AzureBlob;

const MANIFEST_PATH: &str = "target/azure_seq_manifest.txt";

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

fn write_manifest(prefix: &str, keys: &[String]) -> Result<()> {
    if let Some(dir) = Path::new(MANIFEST_PATH).parent() {
        fs::create_dir_all(dir)?;
    }
    let mut buf = String::new();
    buf.push_str(&format!("PREFIX={}\n", prefix));
    for k in keys {
        buf.push_str(&format!("KEY={}\n", k));
    }
    fs::write(MANIFEST_PATH, buf)?;
    Ok(())
}

fn read_manifest() -> Result<(String, Vec<String>)> {
    let s = fs::read_to_string(MANIFEST_PATH)
        .with_context(|| format!("missing manifest at {MANIFEST_PATH}. Run sequence_1_upload_and_list first."))?;
    let mut prefix = String::new();
    let mut keys = Vec::new();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("PREFIX=") {
            prefix = rest.to_string();
        } else if let Some(rest) = line.strip_prefix("KEY=") {
            keys.push(rest.to_string());
        }
    }
    Ok((prefix, keys))
}

/// Sequence #1:
/// - show initial listing size (by our upcoming prefix)
/// - upload a few blobs
/// - show listing again & GET one
/// - write a manifest for the cleanup test
#[tokio::test]
async fn sequence_1_upload_and_list() -> Result<()> {
    if env::var("AZURE_BLOB_CONTAINER").is_err() ||
        (env::var("AZURE_BLOB_ACCOUNT").is_err() && env::var("AZURE_BLOB_ACCOUNT_URL").is_err())
    {
        eprintln!("SKIP: set AZURE_BLOB_CONTAINER and (AZURE_BLOB_ACCOUNT or AZURE_BLOB_ACCOUNT_URL)");
        return Ok(());
    }

    let client = mk_client().await?;

    let prefix = random_key("s3dlio-seq");
    let before = client.list(Some(&prefix)).await?;
    println!("Sequence#1: initial count under prefix {prefix}: {}", before.len());

    // Upload 3 blobs (single-shot put)
    let mut created: Vec<String> = Vec::new();
    for i in 0..3 {
        let key = format!("{}/item-{}.bin", &prefix, i);
        // small content with a pattern
        let len = (i + 1) * 4096;
        let mut buf = vec![0u8; len];
        for (j, b) in buf.iter_mut().enumerate() { *b = ((j + i) % 251) as u8; }
        client.put(&key, Bytes::from(buf), true).await?;
        created.push(key);
    }

    // List again and print a brief diff
    let after = client.list(Some(&prefix)).await?;
    println!("Sequence#1: after upload, count under prefix {prefix}: {}", after.len());
    println!("Sequence#1: created keys: {:?}", &created);

    // GET one blob to validate
    let sample = &created[1];
    let got = client.get(sample).await?;
    println!("Sequence#1: got {} bytes from {}", got.len(), sample);

    // Persist manifest for the cleanup test
    write_manifest(&prefix, &created)?;

    Ok(())
}

/// Sequence #2:
/// - read the manifest produced by #1
/// - delete those blobs
/// - show the prefix count returns to zero
#[tokio::test]
async fn sequence_2_cleanup_and_verify() -> Result<()> {
    if env::var("AZURE_BLOB_CONTAINER").is_err() ||
        (env::var("AZURE_BLOB_ACCOUNT").is_err() && env::var("AZURE_BLOB_ACCOUNT_URL").is_err())
    {
        eprintln!("SKIP: set AZURE_BLOB_CONTAINER and (AZURE_BLOB_ACCOUNT or AZURE_BLOB_ACCOUNT_URL)");
        return Ok(());
    }

    let client = mk_client().await?;
    let (prefix, keys) = match read_manifest() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SKIP: {}", e);
            return Ok(());
        }
    };

    println!("Sequence#2: cleaning up {} objects under prefix {}", keys.len(), prefix);
    client.delete_objects(&keys).await?;

    let after = client.list(Some(&prefix)).await?;
    println!("Sequence#2: remaining under {prefix}: {}", after.len());
    assert!(after.is_empty(), "expected empty listing under {prefix}, got {:?}", after);

    // Optional: remove the manifest so repeated runs fail cleanly if step 1 wasn't run
    let _ = fs::remove_file(MANIFEST_PATH);

    Ok(())
}

