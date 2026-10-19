// tests/caller_buffer_store_support/mod.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// The single Azure fake of a `caller_buffer_store_*` test binary. Included with `#[path]` by a
// binary that also includes `common/azure_fake.rs` as `mod azure_fake`.
//
// The process environment is set ONCE per binary, before the first test touches the network:
// the fake runs on its own thread and runtime (so it outlives every test's runtime) and the
// Azure environment is pointed at it. A test never changes the environment itself, so no test
// changes it while another test's I/O reads it. Each binary calls [`azure`] from the single
// setup step that every test begins with.

use crate::azure_fake::{clear_azure_env, start, State};
use std::sync::OnceLock;

static FAKE: OnceLock<State> = OnceLock::new();

/// The fake Blob service of this binary. The first call starts it and sets the Azure
/// environment; later calls only return it, so call it from the binary's once-only setup before
/// any test starts I/O. Tests that change the fake's shared state (forbidden accounts, missing
/// containers) hold `azure_fake::ENV_LOCK` and put it back.
pub fn azure() -> &'static State {
    FAKE.get_or_init(|| {
        clear_azure_env();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("azure-fake".to_string())
            .spawn(move || {
                let rt = tokio::runtime::Runtime::new().expect("the azure-fake runtime");
                rt.block_on(async move {
                    let (state, port) = start().await;
                    tx.send((state, port)).expect("send the azure-fake address");
                    std::future::pending::<()>().await;
                });
            })
            .expect("spawn the azure-fake thread");
        let (state, port) = rx.recv().expect("receive the azure-fake address");
        // The fake's request cap (a storm guard) counts every request of the binary here, not
        // one test's: leave room for all of them. A storm still ends in a 403.
        state.lock().expect("the fake's lock").cap = 5000;
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");
        std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
        state
    })
}
