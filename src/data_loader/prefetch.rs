//! Prefetch helper for data loader (Stage 2).
//!
//! Spawn a simple async prefetcher over a producer function.
//! Returns a `Receiver` that yields up to `cap` items ahead of consumption.

use std::future::Future;
use tokio::sync::mpsc::{channel, Receiver};
use tokio_util::sync::CancellationToken;
use crate::DatasetError;

/// Spawn an async prefetcher with optional cancellation support.
///
/// `producer` is called repeatedly (it returns a Future yielding `T`).
/// The returned `Receiver` yields `Result<T, DatasetError>` up to `cap` in flight.
/// 
/// `cancel_token` enables graceful shutdown - when cancelled, the prefetch loop
/// will exit cleanly without producing new items.
pub fn spawn_prefetch<F, Fut, T>(
    cap: usize,
    mut producer: F,
    cancel_token: Option<CancellationToken>,
) -> Receiver<Result<T, DatasetError>>
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, DatasetError>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = channel(cap.max(1));
    tokio::spawn(async move {
        // keep producing until `producer` errors, the channel closes, or cancellation
        loop {
            // Check cancellation before producing
            if let Some(ref token) = cancel_token {
                if token.is_cancelled() {
                    break;  // Clean exit
                }
            }
            
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
