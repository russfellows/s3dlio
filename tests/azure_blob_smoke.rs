// Azure backend tests - now always enabled

use anyhow::{Context, Result};
use bytes::Bytes;
use crc32fast::Hasher as Crc32;
use futures::stream;
use rand::{Rng, rng};
use rand::distr::Alphanumeric;
use std::env;
use s3dlio::azure_client::AzureBlob; // direct client tests
use s3dlio::object_store::store_for_uri; // NEW: factory-based smoke

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

fn crc32(bytes: &[u8]) -> u32 {
    let mut h = Crc32::new();
    h.update(bytes);
    h.finalize()
}

#[tokio::test]
async fn put_get_list_stat_delete_smoke() -> Result<()> {
    // Skip with a friendly note if not configured
    if env::var("AZURE_BLOB_CONTAINER").is_err()
        || (env::var("AZURE_BLOB_ACCOUNT").is_err() && env::var("AZURE_BLOB_ACCOUNT_URL").is_err())
    {
        eprintln!("SKIP: set AZURE_BLOB_CONTAINER and (AZURE_BLOB_ACCOUNT or AZURE_BLOB_ACCOUNT_URL) to run azure blob tests");
        return Ok(());
    }

    let client = mk_client().await?;

    let key = random_key("s3dlio-azure-smoke");
    let payload = "hello azure blob 👋".as_bytes();
    let crc_in = crc32(payload);

    // put
    client.put(&key, Bytes::from_static(payload), true).await?;

    // get
    let got = client.get(&key).await?;
    assert_eq!(&got[..], &payload[..], "roundtrip mismatch");
    assert_eq!(crc32(&got), crc_in, "crc mismatch after GET");

    // stat
    let st = client.stat(&key).await?;
    assert_eq!(st.content_length as usize, payload.len());

    // list (by prefix)
    let pref = key.split('-').next().unwrap().to_string(); // "s3dlio"
    let listed = client.list(Some(&pref)).await?;
    assert!(listed.iter().any(|k| k == &key), "key not found in list() results for prefix {pref}");

    // range get (middle slice)
    let start = 6u64;
    let end = 10u64; // inclusive
    let rng = client.get_range(&key, start, Some(end)).await?;
    assert_eq!(&rng[..], &payload[start as usize..=end as usize]);

    // cleanup
    client.delete_objects(&vec![key.clone()]).await?;
    Ok(())
}

#[tokio::test]
async fn multipart_stream_upload_roundtrip() -> Result<()> {
    // Skip if not configured
    if env::var("AZURE_BLOB_CONTAINER").is_err()
        || (env::var("AZURE_BLOB_ACCOUNT").is_err() && env::var("AZURE_BLOB_ACCOUNT_URL").is_err())
    {
        eprintln!("SKIP: set AZURE_BLOB_CONTAINER and (AZURE_BLOB_ACCOUNT or AZURE_BLOB_ACCOUNT_URL) to run azure blob tests");
        return Ok(());
    }

    let client = mk_client().await?;
    let key = random_key("s3dlio-azure-mp");

    // Generate ~32 MiB of data and split into ~4 MiB parts
    let total_len: usize = 32 * 1024 * 1024;
    let part_size: usize = 4 * 1024 * 1024;
    let max_in_flight: usize = 16;

    let mut data = vec![0u8; total_len];
    // Fill with deterministic pattern (cheap & cache-friendly)
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let crc_in = crc32(&data);

    // Build a stream of Bytes chunks
    let chunks: Vec<Bytes> = data
        .chunks(part_size)
        .map(|c| Bytes::copy_from_slice(c))
        .collect();

    let stream = stream::iter(chunks);

    // upload via high-throughput multipart stream
    client
        .upload_multipart_stream(&key, stream, part_size, max_in_flight)
        .await?;

    // download and verify
    let got = client.get(&key).await?;
    assert_eq!(got.len(), total_len);
    assert_eq!(crc32(&got), crc_in, "crc mismatch after multipart upload");

    // spot-check a couple of ranges across boundaries
    let ranges = [
        (0u64, Some(1023u64)),
        (3_900_000u64, Some(4_100_000u64)), // crosses a part boundary
        ((total_len as u64) - 4096, None),   // tail open-ended
    ];
    for (start, end) in ranges {
        let slice = client.get_range(&key, start, end).await?;
        let end_incl = end.unwrap_or((total_len as u64) - 1);
        let expected = &data[start as usize..=end_incl as usize];
        assert_eq!(&slice[..], expected);
    }

    // cleanup
    client.delete_objects(&vec![key.clone()]).await?;
    Ok(())
}

/// NEW: validate the ObjectStore factory with both `az://...` and full HTTPS forms.
#[tokio::test]
async fn factory_roundtrip_smoke() -> Result<()> {
    // Skip if not configured
    if env::var("AZURE_BLOB_CONTAINER").is_err()
        || (env::var("AZURE_BLOB_ACCOUNT").is_err() && env::var("AZURE_BLOB_ACCOUNT_URL").is_err())
    {
        eprintln!("SKIP: set AZURE_BLOB_CONTAINER and (AZURE_BLOB_ACCOUNT or AZURE_BLOB_ACCOUNT_URL) to run azure blob tests");
        return Ok(());
    }

    let account = req_env("AZURE_BLOB_ACCOUNT")?;
    let container = req_env("AZURE_BLOB_CONTAINER")?;
    let base_https = opt_env("AZURE_BLOB_ACCOUNT_URL")
        .unwrap_or_else(|| format!("https://{}.blob.core.windows.net", account));

    let payload = b"hello via ObjectStore";

    // 1) az:// form
    let key1 = random_key("s3dlio-factory-az");
    let uri1 = format!("az://{}/{}/{}", account, container, key1);
    let store1 = store_for_uri(&uri1)?;
    store1.put(&uri1, payload).await?;
    let got1 = store1.get(&uri1).await?;
    assert_eq!(&got1[..], payload);
    store1.delete(&uri1).await?;

    // 2) full https form
    let key2 = random_key("s3dlio-factory-https");
    let uri2 = format!("{}/{}/{}", base_https.trim_end_matches('/'), container, key2);
    let store2 = store_for_uri(&uri2)?;
    store2.put(&uri2, payload).await?;
    let got2 = store2.get(&uri2).await?;
    assert_eq!(&got2[..], payload);
    store2.delete(&uri2).await?;

    Ok(())
}

