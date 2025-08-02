//! src/data_loader/sampler.rs
//! Samplers for the data loader (Stage 2).
//!
//! A `Sampler` produces a stream of indices into a map-style dataset.
//! Two implementations are provided:
//!  * `SequentialSampler` – yields 0..end in order.
//!  * `ShuffleSampler`    – yields 0..len in a deterministic shuffled order.

use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};

/// Trait for index producers.
pub trait Sampler {
    /// Return the next index to fetch, or `None` when exhausted.
    fn next_index(&mut self) -> Option<usize>;
    /// (Optional) remaining items hint.
    fn remaining(&self) -> Option<usize> { None }
}

/// Yields `0, 1, 2, …, end-1` once.
#[derive(Debug, Clone)]
pub struct SequentialSampler {
    curr: usize,
    end: usize,
}

impl SequentialSampler {
    /// Create a sequential sampler over `[0, end)`.
    pub fn new(end: usize) -> Self {
        Self { curr: 0, end }
    }
}

impl Sampler for SequentialSampler {
    fn next_index(&mut self) -> Option<usize> {
        if self.curr < self.end {
            let i = self.curr;
            self.curr += 1;
            Some(i)
        } else {
            None
        }
    }

    fn remaining(&self) -> Option<usize> {
        Some(self.end.saturating_sub(self.curr))
    }
}

/// Yields all indices `0..len` in a deterministic shuffled order.
#[derive(Debug, Clone)]
pub struct ShuffleSampler {
    indices: Vec<usize>,
    pos: usize,
}

impl ShuffleSampler {
    /// Create a shuffled sampler for `len` items, using `seed`.
    pub fn new(len: usize, seed: u64) -> Self {
        // initialize sequential indices
        let mut indices: Vec<usize> = (0..len).collect();
        // seed a ChaCha20 RNG (from rand_core v0.6)
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // === Manual Fisher–Yates shuffle ===
        for i in (1..len).rev() {
            // generate a u32 and reduce to [0..=i]
            let j = (rng.next_u32() as usize) % (i + 1);
            indices.swap(i, j);
        }

        Self { indices, pos: 0 }
    }
}

impl Sampler for ShuffleSampler {
    fn next_index(&mut self) -> Option<usize> {
        if self.pos < self.indices.len() {
            let i = self.indices[self.pos];
            self.pos += 1;
            Some(i)
        } else {
            None
        }
    }

    fn remaining(&self) -> Option<usize> {
        Some(self.indices.len().saturating_sub(self.pos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_yields_all_in_order() {
        let mut s = SequentialSampler::new(5);
        let got: Vec<_> = std::iter::from_fn(|| s.next_index()).collect();
        assert_eq!(got, vec![0,1,2,3,4]);
        assert_eq!(s.remaining(), Some(0));
    }

    #[test]
    fn shuffle_is_deterministic() {
        let mut a = ShuffleSampler::new(10, 42);
        let mut b = ShuffleSampler::new(10, 42);
        let av: Vec<_> = std::iter::from_fn(|| a.next_index()).collect();
        let bv: Vec<_> = std::iter::from_fn(|| b.next_index()).collect();
        assert_eq!(av, bv);           // same seed -> same order
        assert_ne!(av, (0..10).collect::<Vec<_>>()); // not the identity
    }
}
