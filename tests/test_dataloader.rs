//! Integration tests for the Stage-1 DataLoader.
//!
//! We use small, in-memory mock datasets so the tests are deterministic
//! and do not need S3 or any external services.

use s3dlio::{DataLoader, Dataset, DatasetError, LoaderOptions};

use async_trait::async_trait;
use futures_util::StreamExt; // for `next()`

// ────────────────────────────────────────────────────────────────────────────
// Helper 1: Map-style dataset with a backing Vec<T>
// ────────────────────────────────────────────────────────────────────────────
#[derive(Clone)]
struct VecDataset {
    data: Vec<i32>,
}

#[async_trait]
impl Dataset for VecDataset {
    type Item = i32;

    fn len(&self) -> Option<usize> {
        Some(self.data.len())
    }

    async fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        self.data
            .get(index)
            .copied()
            .ok_or(DatasetError::IndexOutOfRange(index))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Helper 2: Iterable-only dataset implemented as an async stream
// ────────────────────────────────────────────────────────────────────────────
struct StreamDataset {
    n: usize,
}

#[async_trait]
impl Dataset for StreamDataset {
    type Item = usize;

    fn len(&self) -> Option<usize> {
        None // unknown a priori
    }

    async fn get(&self, _index: usize) -> Result<Self::Item, DatasetError> {
        Err(DatasetError::Unsupported)
    }

    fn as_stream(&self) -> Option<s3dlio::dataset::DynStream<Self::Item>> {
        use futures_util::stream;
        let n = self.n;
        let s = stream::iter(0..n).map(Ok); // Result<Item, DatasetError>
        Some(Box::pin(s))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Helper 3: Unknown-length map dataset (no `len`, but supports `get`)
// ────────────────────────────────────────────────────────────────────────────
struct UnknownLenDataset {
    n: usize,
}

#[async_trait]
impl Dataset for UnknownLenDataset {
    type Item = usize;

    fn len(&self) -> Option<usize> {
        None
    }

    async fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        if index < self.n {
            Ok(index)
        } else {
            Err(DatasetError::IndexOutOfRange(index))
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Baseline Stage 1 Tests
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn map_dataset_batches() {
    let ds = VecDataset {
        data: (0..100).collect(),
    };
    let opts = LoaderOptions::default().with_batch_size(32);
    let loader = DataLoader::new(ds, opts);

    let mut batches = loader.stream();
    let mut flat = Vec::new();
    while let Some(batch) = batches.next().await {
        flat.extend(batch.expect("no error"));
    }

    assert_eq!(flat.len(), 100);
    assert_eq!(flat, (0..100).collect::<Vec<_>>());
}

#[tokio::test]
async fn map_dataset_drop_last() {
    let ds = VecDataset {
        data: (0..100).collect(),
    };
    let opts = LoaderOptions::default()
        .with_batch_size(32)
        .drop_last(true);
    let loader = DataLoader::new(ds, opts);

    let batches: Vec<_> = loader
        .stream()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();

    assert_eq!(batches.len(), 3); // 3 * 32 = 96; last 4 items dropped
    assert_eq!(batches[0].len(), 32);
    assert_eq!(batches[2][31], 95);
}

#[tokio::test]
async fn iterable_dataset() {
    let ds = StreamDataset { n: 55 };
    let loader = DataLoader::new(ds, LoaderOptions::default().with_batch_size(20));

    let collected: Vec<_> = loader
        .stream()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .flat_map(Result::unwrap)
        .collect();

    assert_eq!(collected, (0..55).collect::<Vec<_>>());
}


#[tokio::test]
async fn unknown_len_dataset() {
    let ds = UnknownLenDataset { n: 10 };

    // Use builder-style options to avoid breakage when LoaderOptions grows fields.
    let opts = LoaderOptions::default()
        .with_batch_size(3)
        .drop_last(false)
        .shuffle(false, 0)
        .num_workers(0)
        .prefetch(0);

    let loader = DataLoader::new(ds, opts);

    let collected: Vec<_> = loader
        .stream()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .flat_map(Result::unwrap)
        .collect();

    assert_eq!(collected, (0..10).collect::<Vec<_>>());
}

#[tokio::test]
async fn empty_dataset() {
    let ds = VecDataset { data: vec![] };
    let loader = DataLoader::new(ds, LoaderOptions::default());

    let mut stream = loader.stream();
    assert!(stream.next().await.is_none(), "stream should be empty");
}

// ────────────────────────────────────────────────────────────────────────────
// Stage 2 Tests
// ────────────────────────────────────────────────────────────────────────────

/// Two runs with the same shuffle seed must produce identical sequences,
/// and that sequence must differ from the unshuffled order.
#[tokio::test]
async fn shuffle_determinism() {
    let base: VecDataset = VecDataset { data: (0..50).collect() };

    // Unshuffled (seed ignored)
    let uns_opt = LoaderOptions::default()
        .with_batch_size(1)
        .shuffle(false, 123);
    let uns: Vec<i32> = DataLoader::new(base.clone(), uns_opt)
        .stream()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        //.map(Result::unwrap)
        .flat_map(Result::unwrap)
        .collect();

    // Two shuffled runs with the same seed
    let shuf_opt = LoaderOptions::default()
        .with_batch_size(1)
        .shuffle(true, 42);
    let shuf1: Vec<i32> = DataLoader::new(base.clone(), shuf_opt.clone())
        .stream()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        //.map(Result::unwrap)
        .flat_map(Result::unwrap)
        .collect();
    let shuf2: Vec<i32> = DataLoader::new(base, shuf_opt)
        .stream()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        //.map(Result::unwrap)
        .flat_map(Result::unwrap)
        .collect();

    assert_eq!(shuf1, shuf2, "shuffled outputs with same seed must match");
    assert_ne!(uns, shuf1, "shuffled sequence must differ from unshuffled");
}

/// Parallel workers + prefetch must not change the output compared to
/// the default (single-threaded, no prefetch) loader.
#[tokio::test]
async fn parallel_prefetch_equivalence() {
    let ds = VecDataset { data: (0..100).collect() };

    // Default loader
    let default_out: Vec<i32> = DataLoader::new(ds.clone(), LoaderOptions::default().with_batch_size(10))
        .stream()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        //.map(Result::unwrap)
        .flat_map(Result::unwrap)
        .collect();

    // Parallel loader
    let parallel_opts = LoaderOptions::default()
        .with_batch_size(10)
        .num_workers(4)
        .prefetch(16);
    let parallel_out: Vec<i32> = DataLoader::new(ds, parallel_opts)
        .stream()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        //.map(Result::unwrap)
        .flat_map(Result::unwrap)
        .collect();

    assert_eq!(default_out, parallel_out, "parallel loader output must match default");
}
