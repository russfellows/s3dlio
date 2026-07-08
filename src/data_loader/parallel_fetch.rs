// src/data_loader/parallel_fetch.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Small shared utility for the task-level-parallelism pattern used by the
// data loaders (issue #148 Phase 2). The core piece is `DropCancel`: a
// stack-held guard that cancels a `CancellationToken` on any drop.
//
// When a producer task holds a `DropCancel(cancel_token)` on its own
// stack and hands clones of `cancel_token` to each spawned fetch task
// (via a `tokio::select!` cancellation arm), the following invariant
// holds: if the producer exits for ANY reason — normal completion,
// early `break` from a receiver-drop, panic on the producer's own
// frame, or the enclosing runtime shutting the producer down — every
// in-flight spawned fetch immediately sees the cancellation and drops
// its own future (cancelling the in-flight I/O). No `JoinHandle`
// tracking, no explicit `.abort()`, and no operating-system-specific
// behavior — pure tokio primitives that work identically from 1-core
// WSL to 256-core HPC.

use tokio_util::sync::CancellationToken;

/// Cancels the wrapped `CancellationToken` when dropped.
///
/// See the module-level docs for the invariant this maintains for
/// task-level-parallel producers.
pub(crate) struct DropCancel(pub CancellationToken);

impl Drop for DropCancel {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
