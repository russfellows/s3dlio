//
// Copyright, 2025.  Signal65 / Futurum Group.
// 

use anyhow::{Context, Result};
use futures::{stream::FuturesUnordered, StreamExt};
use std::sync::Arc;
use tokio::{sync::{mpsc, Semaphore}, task::JoinHandle};

use crate::s3_utils::{get_object, parse_s3_uri};

/// Item type returned to the consumer.
pub type ObjectData = (String, Vec<u8>);

/// Handle returned to the caller.
pub struct Prefetcher {
    rx: mpsc::Receiver<ObjectData>,
    _bg: JoinHandle<()>,      // keeps the producer alive
}

impl Prefetcher {
    /// Blocking iterator interface for Rust callers.
    pub fn next_blocking(&mut self) -> Option<ObjectData> {
        crate::s3_utils::block_on(self.rx.recv())
    }
}

/// Spawn the background producer and return the consumer handle.
///
/// * `uris` – list of S3 URIs (s3://bucket/key) to download  
/// * `jobs` – number of concurrent GETs  
/// * `queue_depth` – channel capacity (controls pre‑fetch window)
pub fn start_prefetch(
    uris: Vec<String>,
    jobs: usize,
    queue_depth: usize,
) -> Prefetcher {
    let (tx, rx) = mpsc::channel::<ObjectData>(queue_depth);
    let bg: JoinHandle<()> = tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(jobs));
        let mut futs = FuturesUnordered::new();

        for uri in uris {
            let permit = sem.clone().acquire_owned().await.unwrap();
            let tx = tx.clone();
            futs.push(tokio::spawn(async move {
                let _permit = permit;                 // keep permit until end
                let (bucket, key) = parse_s3_uri(&uri)?;
                let bytes = get_object(&bucket, &key)?;
                tx.send((uri.clone(), bytes)).await.ok();
                Ok::<(), anyhow::Error>(())
            }));
        }
        // Wait for all to finish; ignore individual errors (already sent via tx).
        while let Some(_)= futs.next().await {}
        // drop tx so receiver sees EOF
    });
    Prefetcher { rx, _bg: bg }
}

