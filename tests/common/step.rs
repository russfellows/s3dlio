// tests/common/step.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Deterministic "mutate the world between two polls" helpers. A local `get_into` reads one
// bounded chunk per poll and yields between chunks, so polling its future ONCE by hand (no
// runtime needed) reads exactly the first chunk; the test can then change the file and poll the
// future to the end. That is a deterministic interleaving of a concurrent writer with the read,
// without any hook in the library. Included with `#[path = "common/step.rs"] mod step;`.

#![allow(dead_code)]

use futures::task::noop_waker_ref;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Poll `f` exactly once.
pub fn poll_once<F: Future + ?Sized>(f: Pin<&mut F>) -> Poll<F::Output> {
    f.poll(&mut Context::from_waker(noop_waker_ref()))
}

/// Poll `f` until it is ready (bounded, so a bug cannot hang the test).
pub fn run_to_end<F: Future>(mut f: Pin<&mut F>) -> F::Output {
    for _ in 0..100_000 {
        if let Poll::Ready(v) = poll_once(f.as_mut()) {
            return v;
        }
    }
    panic!("the future did not finish in 100000 polls");
}
