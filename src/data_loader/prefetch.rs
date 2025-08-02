//! Prefetch helper for data loader (Stage 2).
//!
//! Spawn a simple async prefetcher over a producer function.
//! Returns a `Receiver` that yields up to `cap` items ahead of consumption.

use std::future::Future;
use tokio::sync::mpsc::{channel, Receiver};
use crate::DatasetError;

/// Spawn an async prefetcher.
///
/// `producer` is called repeatedly (it returns a Future yielding `T`).
/// The returned `Receiver` yields `Result<T, DatasetError>` up to `cap` in flight.
pub fn spawn_prefetch<F, Fut, T>(
    cap: usize,
    mut producer: F,
) -> Receiver<Result<T, DatasetError>>
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, DatasetError>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = channel(cap.max(1));
    tokio::spawn(async move {
        // keep producing until `producer` errors or the channel closes
        loop {
            match producer().await {
                Ok(item) => {
                    if tx.send(Ok(item)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    break;
                }
            }
        }
    });
    rx
}

    /*
    #[test]
    fn shuffle_yields_all_in_random_order() {
        let mut s = ShuffleSampler::new(5, 42);
        let got: Vec<_> = std::iter::from_fn(|| s.next_index()).collect();
        assert_eq!(got.len(), 5);
        assert!(got.iter().all(|&i| i < 5));
        assert_eq!(s.remaining(), Some(0));
    }
    */
