// src/range_version.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! One object version per logical GET (Blue stage G review, item G-F2).
//!
//! The RangeEngine reads one object as several independent ranged requests, each started
//! separately. Pinning the retries of ONE range to the version it first saw does not pin its
//! siblings: if the object is replaced by another of the same length between two ranges, the
//! body is assembled from two versions and no size check can see it. A mixed body is never
//! acceptable; a consistently pinned older version, a complete restart on the newer one, or an
//! error all are.
//!
//! An [`ObjectVersion`] is the identity every range of one logical GET shares (an Azure strong
//! ETag, a GCS generation). The first range to start is its LEADER: it is sent unpinned, and as
//! soon as its response HEADERS arrive (before its body is received) it publishes the identity
//! the response reports. Every other range waits for that, which costs about one round trip and
//! not one range's transfer, and is then sent pinned to it, so the service itself refuses a
//! different version. The leader never waits for anyone, so the scheme cannot deadlock however
//! the ranges are scheduled.
//!
//! * A leader whose response carries no usable identity publishes "unavailable": unpinned ranges
//!   are never combined, so every other range fails at once and the caller reads the object as
//!   ONE response instead (see `ObjectVersion::is_unavailable`).
//! * A leader that fails, is cancelled or panics before it published releases the others with an
//!   error that names the reason, so a sibling can never wait for ever.
//!
//! The type holds no payload and copies nothing.

use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::watch;

/// What the leader has reported so far.
#[derive(Clone, Debug)]
enum State<T> {
    /// The leader's response has not arrived yet.
    Pending,
    /// The identity every range must carry.
    Known(T),
    /// The leader's response carried no identity that can pin a request.
    Unavailable,
    /// The leader failed or was dropped before it reported (the reason, for the error text).
    Failed(String),
}

/// The identity shared by the ranges of one logical GET.
#[derive(Debug)]
pub(crate) struct ObjectVersion<T> {
    sender: watch::Sender<State<T>>,
    receiver: watch::Receiver<State<T>>,
    /// Set by the first range to ask: that range is the leader.
    claimed: AtomicBool,
}

/// What a range does, as [`ObjectVersion::join`] decides.
#[derive(Debug)]
pub(crate) enum Role<'a, T> {
    /// Send unpinned, then report the identity of the response with [`Leader::publish`] or
    /// [`Leader::unavailable`].
    Leader(Leader<'a, T>),
    /// Send pinned to this identity.
    Follower(T),
}

/// The leader's handle. Dropping it before it reported releases the followers with an error.
#[derive(Debug)]
pub(crate) struct Leader<'a, T> {
    owner: &'a ObjectVersion<T>,
}

impl<T: Clone> ObjectVersion<T> {
    /// A version nobody has reported yet: the first [`join`](Self::join) leads.
    pub(crate) fn new() -> Self {
        let (sender, receiver) = watch::channel(State::Pending);
        Self {
            sender,
            receiver,
            claimed: AtomicBool::new(false),
        }
    }

    /// Take a part in the logical GET. The first caller leads; every other caller waits for the
    /// leader's report and returns the identity to pin to, or an error when there is none.
    pub(crate) async fn join(&self) -> Result<Role<'_, T>> {
        if !self.claimed.swap(true, Ordering::AcqRel) {
            return Ok(Role::Leader(Leader { owner: self }));
        }
        let mut receiver = self.receiver.clone();
        // The sender lives in `self`, which this borrow keeps alive: the wait cannot see it gone.
        let state = receiver
            .wait_for(|state| !matches!(state, State::Pending))
            .await
            .map_err(|_| anyhow!("the object version was abandoned"))?
            .clone();
        match state {
            State::Known(identity) => Ok(Role::Follower(identity)),
            State::Unavailable => Err(anyhow!(
                "the first response carried no version identity that can pin a request, so \
                 the ranges of this object cannot be combined"
            )),
            State::Failed(reason) => Err(anyhow!(
                "the first range, which establishes the object's version for the others, did \
                 not complete: {reason}"
            )),
            State::Pending => Err(anyhow!("the object version is still pending")),
        }
    }

    /// Whether the leader found no usable identity: the object cannot be read as pinned ranges.
    pub(crate) fn is_unavailable(&self) -> bool {
        matches!(*self.receiver.borrow(), State::Unavailable)
    }

    /// Move from `Pending` to `next`; a report that is already made is never replaced.
    fn settle(&self, next: State<T>) {
        self.sender.send_if_modified(|state| {
            if matches!(state, State::Pending) {
                *state = next;
                true
            } else {
                false
            }
        });
    }
}

impl<T: Clone> Leader<'_, T> {
    /// Report the identity of the leader's response; the others may now start, pinned to it.
    pub(crate) fn publish(&self, identity: T) {
        self.owner.settle(State::Known(identity));
    }

    /// Report that the leader's response carried no usable identity.
    pub(crate) fn unavailable(&self) {
        self.owner.settle(State::Unavailable);
    }

    /// Report that the leader failed before it could report; `reason` reaches the others. Has no
    /// effect once an identity or "unavailable" was reported.
    pub(crate) fn fail(&self, reason: String) {
        self.owner.settle(State::Failed(reason));
    }
}

impl<T> Drop for Leader<'_, T> {
    /// A leader that is cancelled (its future dropped) or panics reports failure, so a follower
    /// waiting for it is released instead of waiting for ever. A no-op after any report.
    fn drop(&mut self) {
        self.owner.sender.send_if_modified(|state| {
            if matches!(state, State::Pending) {
                *state = State::Failed("the first range was cancelled".to_string());
                true
            } else {
                false
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    const DEADLINE: Duration = Duration::from_secs(10);

    async fn followed(version: &ObjectVersion<String>) -> Result<String> {
        match version.join().await? {
            Role::Follower(identity) => Ok(identity),
            Role::Leader(_) => Err(anyhow!("expected a follower")),
        }
    }

    #[tokio::test]
    async fn the_first_range_leads_and_the_others_follow_its_identity() {
        let version = Arc::new(ObjectVersion::<String>::new());
        let Role::Leader(leader) = version.join().await.expect("first") else {
            panic!("the first range must lead");
        };
        let waiting: Vec<_> = (0..3)
            .map(|_| {
                let version = Arc::clone(&version);
                tokio::spawn(async move { followed(&version).await })
            })
            .collect();
        // The followers cannot finish before the leader reports.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(waiting.iter().all(|task| !task.is_finished()));
        leader.publish("etag-a".to_string());
        for task in waiting {
            let identity = tokio::time::timeout(DEADLINE, task)
                .await
                .expect("released")
                .expect("task")
                .expect("an identity");
            assert_eq!(identity, "etag-a");
        }
        // A range that starts later sees the identity at once.
        assert_eq!(followed(&version).await.expect("late"), "etag-a");
        assert!(!version.is_unavailable());
    }

    #[tokio::test]
    async fn only_the_first_report_counts() {
        let version = ObjectVersion::<String>::new();
        let Role::Leader(leader) = version.join().await.expect("first") else {
            panic!("lead");
        };
        leader.publish("first".to_string());
        leader.publish("second".to_string());
        leader.unavailable();
        leader.fail("late".to_string());
        drop(leader);
        assert_eq!(followed(&version).await.expect("identity"), "first");
    }

    #[tokio::test]
    async fn an_unavailable_identity_fails_every_follower_and_is_visible_to_the_caller() {
        let version = ObjectVersion::<String>::new();
        let Role::Leader(leader) = version.join().await.expect("first") else {
            panic!("lead");
        };
        leader.unavailable();
        let err = followed(&version).await.expect_err("no identity");
        assert!(
            format!("{err:#}").contains("no version identity"),
            "{err:#}"
        );
        assert!(version.is_unavailable());
    }

    #[tokio::test]
    async fn a_failed_leader_releases_the_followers_with_its_reason() {
        let version = Arc::new(ObjectVersion::<String>::new());
        let Role::Leader(leader) = version.join().await.expect("first") else {
            panic!("lead");
        };
        let waiting = {
            let version = Arc::clone(&version);
            tokio::spawn(async move { followed(&version).await })
        };
        leader.fail("403 denied".to_string());
        drop(leader);
        let err = tokio::time::timeout(DEADLINE, waiting)
            .await
            .expect("released")
            .expect("task")
            .expect_err("the leader failed");
        assert!(format!("{err:#}").contains("403 denied"), "{err:#}");
        assert!(!version.is_unavailable());
    }

    #[tokio::test]
    async fn a_dropped_leader_releases_the_followers() {
        let version = Arc::new(ObjectVersion::<String>::new());
        let leader_version = Arc::clone(&version);
        let leading = tokio::spawn(async move {
            let Role::Leader(_leader) = leader_version.join().await.expect("first") else {
                panic!("lead");
            };
            // Never reports: the task is aborted below while it holds the handle.
            std::future::pending::<()>().await;
        });
        // Let the leader claim the role before anyone else asks.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let waiting = {
            let version = Arc::clone(&version);
            tokio::spawn(async move { followed(&version).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiting.is_finished());
        leading.abort();
        let err = tokio::time::timeout(DEADLINE, waiting)
            .await
            .expect("released by the dropped leader")
            .expect("task")
            .expect_err("the leader was cancelled");
        assert!(format!("{err:#}").contains("cancelled"), "{err:#}");
    }
}
