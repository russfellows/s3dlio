# GCS BidiReadObject Hang Analysis

> Consolidation note (v0.9.70): this is the deep-dive technical source of truth.
> For triage-oriented incident notes and quick status, use [GCS_DEBUG_SUMMARY.md](GCS_DEBUG_SUMMARY.md).

**Date**: March 14, 2026  
**Symptom**: `sai3-bench` GCS GET on RAPID bucket hangs permanently near end of large runs  
**Root**: `google-cloud-storage` crate upgrade: vendored v1.8 → fork at v1.9 (`russfellows/google-cloud-rust`, branch `s3dlio-patches`)  
**Status**: Investigation complete, fix NOT yet implemented

---

## 1. Observed Behavior

| Object count | Result |
|---|---|
| 6000 | Always hangs |
| 2000 | Always hangs |
| 1000 | Always hangs |
| 500  | Hangs most of the time, rarely succeeds |
| 100  | Always succeeds |

- Hang is **permanent** — no timeout ever fires, process never terminates
- Last log line before hang: `GCS GET (gRPC) success: X bytes in N chunk(s)` — objects complete successfully, then the run stalls
- No error is logged; no panic; process is alive but frozen
- Degradation starts at ~500 concurrent objects — consistent with a connection/flow-control threshold

---

## 2. The v1.8 → v1.9 Regression

The previous vendor dir (`s3dlio/vendor/`) was based on `google-cloud-storage v1.8`. The current fork (`/home/eval/Documents/Code/google-cloud-rust`, branch `s3dlio-patches`) is at v1.9.

In v1.9, `open_object()` was completely rewritten. It now establishes a **persistent bidirectional gRPC stream** (`BidiReadObject`) with a background `Worker` tokio task. In v1.8 the implementation was simpler and did not create this persistent-worker architecture.

The s3dlio usage pattern is **fire-and-forget per object** — open, read the entire object once, drop. This does not match the intended v1.9 design (long-lived descriptor, multiple sequential range reads). Under high concurrency this mismatch exposes a hang path that did not exist in v1.8.

---

## 3. Call Stack from sai3-bench to the gRPC Stream

```
sai3-bench: get_object_multi_backend(uri)
  └─ GcsObjectStore::get(uri)                   [object_store.rs:2135]
       └─ GcsClient::get_object(bucket, object) [google_gcs_client.rs:357]
            └─ GcsClient::get_object_via_grpc() [google_gcs_client.rs:367]
                 └─ Storage::open_object(bucket, object)
                      .send_and_read(ReadRange::all()).await
                         │
                         └─ ObjectDescriptorTransport::new(connector, ranges)
                              ├─ connector.connect(proto_ranges)    → establishes bidi gRPC stream
                              ├─ tokio::spawn(worker.run(conn, rx)) → BACKGROUND TASK (JoinHandle DROPPED)
                              └─ returns (descriptor, reader)       → reader = RangeReader
```

---

## 4. v1.9 BidiReadObject Architecture (Key Types)

### `ObjectDescriptorTransport` — `bidi/transport.rs`

Created per `open_object()` call. Holds:
- `tx: Sender<ActiveRead>` — request channel to the Worker (capacity 100)
- `object: Arc<Object>` — metadata from the initial handshake
- `headers: HeaderMap`

**Critical**: `let _handle = tokio::spawn(worker.run(connection, rx));`  
The `JoinHandle` is **immediately dropped** (`_handle`). Worker panics are invisible to callers.

### `RangeReader` — `bidi/range_reader.rs`

Returned to `get_object_via_grpc()` as the iterator over object chunks.

Fields:
- `inner: Receiver<Result<Bytes, ReadError>>` — chunks arrive here from the Worker
- `object: Arc<Object>` — metadata
- `_tx: Sender<ActiveRead>` — **clone of the Worker's request channel sender**

The `_tx` field is explicitly commented:
> "Unused, holding to a copy prevents the worker task from terminating early."

As long as any `RangeReader` is alive, its `_tx` keeps the Worker's `requests` channel open.

`next()` implementation:
```rust
async fn next(&mut self) -> Option<crate::Result<bytes::Bytes>> {
    let msg = self.inner.recv().await?;   // blocks until data or channel closed
    Some(msg.map_err(Error::io))
}
```

### `Worker` — `bidi/worker.rs`

Runs in a background task. Main loop (`run()`):

```rust
let error = loop {
    tokio::select! {
        m = rx.next_message() => {
            match self.handle_response(m).await {
                None          => break None,           // clean EOF from server
                Some(Err(e))  => break Some(e),        // unrecoverable error
                Some(Ok(None))=> {},                   // message processed, continue
                Some(Ok(Some(conn))) => { ... }        // reconnected, update rx/tx
            }
        },
        r = requests.recv_many(&mut ranges, 16) => {
            if r == 0 { break None; }                  // all Senders dropped
            self.insert_ranges(tx.clone(), ...).await;
        },
    }
};
```

**Worker exits only when:**
1. `requests` channel has zero senders (all `_tx` clones dropped) → `recv_many()` returns 0 → `break None`
2. Server sends clean EOF → `rx.next_message()` returns `Ok(None)` → `break None`
3. Unrecoverable error → `break Some(e)` → notifies remaining readers via `interrupted()`

### `ActiveRead` — `bidi/active_read.rs`

One per in-flight range. Held in `Worker.ranges: HashMap<i64, ActiveRead>`.

Each `ActiveRead` holds its own `sender: Sender<Result<Bytes, ReadError>>` — the write side of `RangeReader.inner`.

When the server sends `range_end=true`, `handle_range_data()`:
```rust
let mut pending = ranges.lock().await.remove(&range.read_id)...; // drops ActiveRead
pending.handle_data(..., true)?                                   // sends final chunk
// ActiveRead is dropped here → sender is dropped → RangeReader.inner.recv() returns None
```

---

## 5. The Normal (Working) Completion Sequence

For a single object at low concurrency, the correct happy-path is:

1. `open_object().send_and_read()` → Worker spawned, `(descriptor, reader)` returned
2. `while let Some(chunk) = reader.next().await` — loops collecting chunks
3. Server sends chunks. For the final chunk, `range_end=true` is set
4. Worker calls `handle_range_data()` with `range_end=true`:
   - `ActiveRead` removed from HashMap, dropped
   - `ActiveRead.sender` dropped → `RangeReader.inner` channel closed
5. `reader.next().await` returns `None` — loop exits
6. `get_object_via_grpc()` returns `Ok(bytes)`
7. `reader` is dropped → `reader._tx` (clone of Worker's requests sender) dropped
8. `descriptor` is dropped → `ObjectDescriptorTransport.tx` (original Worker requests sender) dropped
9. Worker's `requests` channel now has zero senders → `recv_many()` returns 0 → `break None`
10. Worker calls `drop(rx); drop(tx)` → gRPC stream closed
11. Worker task exits

**This sequence works correctly at low concurrency (≤100 objects).**

---

## 6. The Hang Path

Under high concurrency (500–6000 objects), HTTP/2 flow control pressure causes some bidi streams to **stall** — the server stops sending data mid-transfer, before it has sent `range_end=true`.

The resulting deadlock:

```
get_object_via_grpc()               Worker task
        │                                │
reader.next().await ─────────────>  inner.recv() blocked
        │                            (waiting for data)
        │                                │
        │   rx.next_message() waiting ───┘
        │   (gRPC stream stalled: server sent no EOF, no data)
        │
        │   requests.recv_many() blocked
        │   (reader._tx + descriptor.tx both still alive)
        │
        ▼
DEADLOCK: reader.next() waits for Worker data; Worker waits for gRPC server;
          server stalled; no timeout anywhere; Worker requests channel kept
          alive by _tx in the blocking RangeReader
```

**There is no timeout on `reader.next().await` in `get_object_via_grpc()`.**  
**There is no timeout on `rx.next_message()` in the Worker select loop.**

The only timeout in this code path is `bidi_attempt_timeout = 60s` (default), which wraps **only `connect_attempt()`** — the initial stream handshake. It does not apply to ongoing data delivery.

---

## 7. Why 100 Objects Succeeds, 1000 Hangs

With 64 gRPC subchannels (default configuration) and 64 concurrent jobs:

- Each subchannel is one HTTP/2 TCP connection
- Each HTTP/2 connection multiplexes all streams sharing that connection's flow-control window
- **100 objects at 64 concurrency**: ~1.5 objects per subchannel max in flight — low pressure, streams drain before the window saturates
- **1000 objects at 64 concurrency**: steady-state ~16 streams per subchannel — each stream competes for 1/16th of the connection window; back-pressure stalls some streams, which never complete, which hangs the process

The threshold at ~500 objects is consistent with HTTP/2 `MAX_CONCURRENT_STREAMS` pressure and per-connection flow control window exhaustion. When chunks cannot be delivered because the receive window is full and the application-side consumer is stalled (because the tokio task is waiting but the flow-control credits are not being sent back because **another stalled stream is blocking the executor**), the server legitimately stops sending.

---

## 8. Timeout Coverage Map

| Timeout | Location | Applies To |
|---|---|---|
| `bidi_attempt_timeout` = 60s | `connector.connect()` wrapping `connect_attempt()` | Initial stream handshake ONLY |
| None | `connector.reconnect()` | Reconnect policy: unbounded retries on transient errors (`Recommended` policy) |
| None | `worker.run()` select branches | gRPC stream data delivery and range request handling |
| None | `reader.next().await` | Per-chunk data delivery to `get_object_via_grpc()` |
| None | `ObjectDescriptorTransport::new()` overall | Entire open+read lifecycle |

The `Recommended` resume policy (`read_resume_policy.rs`) retries on `io`, `transport`, `timeout`, `Unavailable`, `ResourceExhausted`, `Internal`, `DeadlineExceeded` — **unbounded retries** (no `with_attempt_limit()` applied by default). If a stall causes a timeout/RST at the transport layer, the worker will attempt to reconnect indefinitely.

---

## 9. Code Locations of Interest

| File | Lines | Notes |
|---|---|---|
| `s3dlio/src/google_gcs_client.rs` | 367–430 | `get_object_via_grpc()` — no timeout on `reader.next()` |
| `google-cloud-rust/.../bidi/transport.rs` | 34–70 | `ObjectDescriptorTransport::new()` — `_handle` discarded |
| `google-cloud-rust/.../bidi/transport.rs` | 95–110 | `read_range()` — panics if worker died (`expect`) |
| `google-cloud-rust/.../bidi/worker.rs` | 60–115 | `Worker::run()` — the actual select loop |
| `google-cloud-rust/.../bidi/worker.rs` | 216–245 | `handle_range_data()` — `range_end` processing |
| `google-cloud-rust/.../bidi/range_reader.rs` | 28–52 | `RangeReader` — `_tx` hold explained in comment |
| `google-cloud-rust/.../bidi/active_read.rs` | 47–68 | `handle_data()` — data delivery and end-of-range |
| `google-cloud-rust/.../bidi/connector.rs` | 88–115 | `connect()` — bidi_attempt_timeout scope |
| `google-cloud-rust/.../storage/request_options.rs` | 30, 132 | `DEFAULT_BIDI_ATTEMPT_TIMEOUT = 60s` |
| `google-cloud-rust/.../read_resume_policy.rs` | 134–145 | `Recommended` policy — transient reconnect, no limit |
| `s3dlio/src/s3_utils.rs` | 1125–1150 | `get_objects_parallel()` — `Semaphore` + `FuturesUnordered` |
| `s3dlio/src/object_store.rs` | 2134–2170 | `GcsObjectStore::get()` routing to `get_object_via_grpc` |

---

## 10. Secondary Issue: Discarded JoinHandle

In `transport.rs`:
```rust
let _handle = tokio::spawn(worker.run(connection, rx));
```

The `JoinHandle` is discarded. If the Worker panics mid-transfer (e.g., due to an `expect` in an internal bidi path), the panic is silently swallowed by tokio. The caller's `reader.next().await` would then block forever because the data channel sender inside the Worker is gone (the Worker thread died before sending `range_end=true`). This is a secondary mechanism that could contribute to hangs, distinct from the flow-control stall.

---

## 11. Proposed Fixes (NOT YET EVALUATED OR IMPLEMENTED)

### Option A — Add timeout to `reader.next().await` in s3dlio (minimal, no fork changes)

Wrap the while loop in `get_object_via_grpc()`:
```rust
while let Some(chunk) = tokio::time::timeout(Duration::from_secs(120), reader.next())
    .await
    .map_err(|_| anyhow!("GCS GET timeout for gs://{}/{}", bucket, object))?
    .transpose()...
```
**Pro**: No fork changes. Converts hang to an error.  
**Con**: Still loses the object (error, not retry). Root cause unaddressed; may mask real bugs.

### Option B — Revert s3dlio to HTTP for RAPID reads

For whole-object reads, use `read_object()` (HTTP/server-streaming) instead of `open_object()` (BidiReadObject). The BidiReadObject API is designed for long-lived, multi-range descriptors; for single whole-object reads, `read_object()` is the correct API.

**Requires verifying**: Does `read_object()` work on RAPID buckets? The RAPID distinction may be write-only (appendable writes). This needs testing — if RAPID buckets accept `ReadObject` RPCs, this is the cleanest fix.

### Option C — Cap concurrency at the `open_object()` call site

Add a dedicated semaphore in `GcsObjectStore::get()` (or `get_object_via_grpc()`) that limits concurrent bidi streams independently of the outer s3dlio concurrency. For example, cap at `subchannel_count * 4` concurrent BidiRead streams. This prevents flow-control saturation.

**Pro**: Directly addresses the resource exhaustion.  
**Con**: May reduce throughput below what v1.8 achieved.

### Option D — Add per-read-chunk timeout inside the Worker (fork change)

Patch `bidi/worker.rs` to add a timeout on `rx.next_message()`. If the stream delivers no data for N seconds, treat it as a transient error and trigger a reconnect (which the existing `reconnect()` method handles by resuming from the current `RemainingRange`).

**Pro**: Correct behavior — stalled streams reconnect instead of hanging.  
**Con**: Requires fork maintenance.

---

## 12. Outstanding Questions Before Implementing a Fix

1. Does `read_object()` work on RAPID buckets? (Can we avoid BidiRead for whole-object reads entirely?)
2. What was v1.8's implementation of `open_object()` — was it also BidiRead or a simpler RPC?
   - Check: `find s3dlio/vendor -name "*.rs" | xargs grep -l "open_object\|BidiRead"`
3. At what HTTP/2 stream count does GCS RAPID start stalling? (Is there a server-side `MAX_CONCURRENT_STREAMS` hint we can read?)
4. Is the `Recommended` resume policy with no attempt limit making things worse? (Could an infinite reconnect loop against a stalled stream consume resources and starve other streams?)

## Additional Thinking
Read transport.rs, lines 1 to 130

Now I have the complete picture. Let me carefully trace the data flow:

In transport.rs::ObjectDescriptorTransport::new():

let (tx, rx) = tokio::sync::mpsc::channel(100) — this is the requests channel (Sender<ActiveRead>), used to send new range requests to the Worker
In map_ranges():

For each range, creates let (tx, rx) = tokio::sync::mpsc::channel(100) — this is the data channel (Sender<ReadResultbytes::Bytes>), used to deliver data to the RangeReader

Each ActiveRead gets the sender side of the data channel, while the RangeReader gets the receiver side plus a clone of the requests sender to communicate back with the Worker

So the lifetime chain is: ObjectDescriptorTransport holds one Sender<ActiveRead> for the requests channel, each RangeReader gets its own clone of that sender, and the Worker holds the corresponding Receiver that listens for all incoming ActiveRead requests across all RangeReaders.

For the per-range data channel, the ActiveRead stores the sender side of the ReadResult channel in the hashmap, while RangeReader holds the receiver side. Now I'm tracing through the exact scenario where handler.send().await could deadlock: when a range completes and gets removed from the hashmap, the pending ActiveRead is dropped as a local variable, which closes its sender side of the data channel before the handler can actually send the result through it. drop at end of handle_range_data, then handler.send().await completes and drops the cloned sender, followed by pending dropping the original sender — at that point all senders are gone, so reader.inner.recv() returns None and the iteration ends. The send operation itself shouldn't block since the channel has capacity 100 and we're only sending one final chunk, but back-pressure could still occur if the receiver isn't consuming fast enough. I'm realizing the actual deadlock: the Worker task fills the channel with 100 items via handler.send().await, which blocks because get_object_via_grpc() hasn't consumed them yet — but they're in separate Tokio tasks, so if the runtime is over-subscribed, the receiver never gets scheduled to drain the channel. Let me trace through the call chain to confirm whether they're actually running concurrently. I'm realizing the producer-consumer relationship here: the Worker tasks are sending data through the channel while the sai3-bench workers are trying to receive it, and with thousands of concurrent tasks, the event loop might be so overwhelmed that neither side gets scheduled enough to make progress. Let me dig deeper into what's actually causing the bottleneck. the handle_range_data carefully again:

I'm checking the lock behavior in the range_end = false branch — the guard is held during handle_data, but it drops before handler.send().await is called, so there's no deadlock risk there either.

Let me trace through the actual implementation from worker.rs to verify the full flow...

Looking at handle_ranges(), I see there's a TODO comment about parallelizing the loop, which means right now the responses are being processed sequentially in a for loop.

So handle_range_data calls aren't concurrent for the same stream. But when range_end is false, the code acquires the lock, gets the pending handler, and calls handle_data on it — and that's where I think the actual deadlock is happening.

The handler gets returned while the lock is still held, then handler.send().await tries to push data onto a channel. If that channel is full (capped at 100 items), the send blocks. Meanwhile, the consumer task that's supposed to drain the channel is stuck waiting to be scheduled, but with high concurrency — say 64 concurrent objects being read — there are 64 worker tasks and 64 consumer tasks all competing for CPU time. The channel fills up, the send blocks while holding the lock, and nothing can progress.

Actually, wait — in an async runtime this shouldn't be a true deadlock since send().await yields control and lets other tasks run. The consumer should wake up, read an item, and free up space for the sender to continue. That's the whole point of async channels — they apply backpressure without blocking the executor.

So if there's a hang, it's not the data channel itself. Let me consider what else might be blocking — maybe it's an HTTP/2 flow control issue instead.

GCS uses gRPC over HTTP/2, and the server limits concurrent streams per connection with SETTINGS_MAX_CONCURRENT_STREAMS, typically set to 100. With s3dlio's default setup of 64 subchannels on a 16-CPU machine, that gives 6400 total concurrent streams across all connections. When sai3-bench runs 64 concurrent workers each opening a BidiReadObject stream, we're well within that limit, so stream exhaustion shouldn't be the bottleneck either.

The real question is what concurrency level the user actually ran their tests at. They mentioned testing with 1000, 2000, and 6000 objects, but those are total object counts, not concurrent workers. The actual concurrency is controlled by sai3-bench's semaphore setting, which I need to check in their config to understand what they were really testing with. the critical question. The GCS BidiReadObject streams don't close after delivering all ranges—they stay open waiting for more requests, which is the whole point of bidirectional streaming. So when the worker loops back after finishing all ranges, it blocks on the next message from the server while the request channel might be waiting to signal shutdown, creating a deadlock where neither side can proceed.

The real issue is timing: I need to trace through when both senders actually get dropped. In the get_object_via_grpc() function, the reader and descriptor are only dropped when the function returns, which happens after the while loop exits and the final result is returned. So the senders stay alive until that exact moment—but by then the worker is already blocked in select! waiting for the next server message. ends, which drops reader and descriptor.

The timing question: After reader.next() returns None, what task is doing what?

get_object_via_grpc() got None from reader.next()
reader.next() returned None because reader.inner.recv() returned None
reader.inner.recv() returned None because all senders to the data channel closed
The data channel closes when the ActiveRead is removed from ranges AND the Handler's cloned sender is dropped (both happen when handle_range_data processes range_end=true)
After step 4, handle_range_data returns, then handle_ranges, then handle_response_success, then handle_response returns Some(Ok(None)). The Worker loops back to select! and waits there. But here's the key: get_object_via_grpc() hasn't actually returned yet—it still needs to log and finish up. That means reader and descriptor are still in scope, so both reader._tx and ObjectDescriptorTransport.tx are still alive, keeping the requests channel open with 2 active senders. When recv_many() tries to receive, it won't get anything but the channel stays open because those senders haven't been dropped. Both branches are blocking, so the Worker just waits. Eventually when get_object_via_grpc() finishes and drops reader and descriptor, the requests channel closes, recv_many() returns 0, and the Worker exits cleanly. But that doesn't explain the hang. Let me trace through what actually happens when reader.next() returns None — that occurs when the data channel's senders are all dropped. In map_ranges(), a channel is created and the sender is moved into an ActiveRead wrapper, which is then passed to RangeReader.

The issue is that ActiveRead isn't the only holder of the sender. When handle_data creates a Handler, it clones the sender, so now there are two senders in play — one in the ActiveRead and one in the Handler. After the handler sends its data, the sender inside it gets dropped, but the one in ActiveRead is still alive, so the receiver never sees the channel as closed.

Then when ActiveRead itself is dropped from the ranges map, that final sender drops too, and only then does the receiver get the None signal. But the real problem is that the last data chunk with range_end=true gets sent through the handler first, so the consumer might receive it before the channel closes, creating a race condition or ordering issue. the consumer loop processes the final chunk successfully and exits when the channel closes, but the real stall is happening somewhere else—likely the producer is blocking before it ever gets to send that last data.

The Worker's event loop uses tokio::select! to handle incoming messages and batch requests, but when it calls insert_ranges to send a new read request to the server, something's getting stuck there instead of completing and allowing the gRPC stream to continue flowing. I'm realizing the key constraint here: the tx.send() call on this bounded 100-capacity channel could theoretically block if the gRPC layer isn't consuming from rx fast enough, but in practice for a single read-all request, we only send one initial message so the channel never fills up. The real question is whether the ranges are baked into that initial request or sent incrementally afterward.

Looking at the hang pattern—where 100 objects succeed but 1000+ consistently hangs near the end—this points to a resource exhaustion issue rather than a channel capacity problem.

My hypothesis is that under high concurrency, GCS is closing connections with GOAWAY or RST_STREAM frames, which triggers reconnection attempts. But when the reconnect logic hits server-side errors like RESOURCE_EXHAUSTED or UNAVAILABLE, those retries may be getting stuck or exhausting some finite resource, leaving a few tasks unable to complete. reconnect fails permanently (after retry exhaustion), close_readers() is called to interrupt all active reads by sending errors through their data channels. The consumer receives these errors via reader.next() and should propagate them up through the error handling in get_object_via_grpc(). But I'm not seeing errors in the logs—just a hang. I'm realizing that if the ranges hashmap is already empty (because the read completed), close_readers() does nothing, which might explain why we're stuck waiting instead of failing. no interrupts are sent.

But then the Worker would exit via an error handler that closes all readers and returns the error, which eventually propagates up to the run() method. There it breaks out of the loop and enters a final section that sends the error to any pending requests before the channel closes.

If there are no active ranges, requests.recv() would block waiting for new ones, but since descriptor and reader are dropped, the channel closes and the loop terminates. However, if the reconnect succeeds but then stalls with no incoming data, the Worker gets stuck in the select! loop waiting for a message that never arrives.

I think I've found the actual issue. Let me trace through the reconnect() logic again—it's collecting the current ranges and attempting to reconnect with them.

Examining the reconnect implementation more closely...

Continuing through the connector code...

So when the resume policy allows continuing, it reconnects with whatever ranges are pending—or an empty list if there are none. The server sends back an initial response with no data, the handler processes the empty range list successfully, and the Worker gets a fresh stream. Now it's sitting in the select loop waiting for either new messages from the receiver or data from this newly established stream.

But here's the critical issue: the reconnect happens while get_object_via_grpc() is still blocked in its loop waiting for responses. The senders on the requests channel are still alive, so recv_many() doesn't return empty—it just keeps waiting. Meanwhile, the fresh stream is idle with no pending ranges, so rx.next_message() blocks too. Both branches of the select are blocked, and nothing can unblock them because the original request is stuck upstream. That's where the deadlock occurs.

Actually, I'm wondering if there's a race condition here. If the stream delivered the final range_end=true message and the handler sent the data to reader.inner, but then the network error happened before get_object_via_grpc() actually polled that channel, the data would be sitting there unread.

So the real question is: when does reader.inner.recv() actually block forever? It only happens if the channel is empty AND at least one sender is still alive. Since each range creates exactly one sender in map_ranges(), and handle_data might clone it, I need to figure out which senders could still be holding a reference when the channel should be drained.

The Handler objects that receive cloned senders are temporary — they send their data and then drop the clone. The original sender lives in ActiveRead and only gets dropped when range_end=true arrives. So if that signal never comes because the gRPC stream stalls partway through a range, and the reconnect logic doesn't properly re-request the remaining data, then recv() would block indefinitely waiting for a sender that will never close.

Checking how reconnect handles partial reads...

The RemainingRange methods show that when reconnecting, we request only the bytes we haven't received yet, not the full original range. So if the stream stalls silently — no error, just hanging — then rx.next_message() is stuck waiting on the gRPC stream that's gone unresponsive.

The HTTP/2 keepalive would eventually timeout after 20+ minutes, but no reconnect triggers because there's no actual error. The reader just hangs indefinitely waiting for data that never arrives. This explains the observed behavior perfectly.

The key difference between v1.8 and v1.9 is the streaming model: v1.8 likely used simple server-streaming where the server closes the stream after sending all data, whereas v1.9's bidirectional streaming keeps the connection open, making it vulnerable to silent stalls where the client never realizes the server has stopped responding.

In v1.9, the proper cleanup should happen when all Sender<ActiveRead> clones get dropped, which would cause the Worker to drop the gRPC request sender and trigger the HTTP/2 stream closure. But something's preventing this cleanup path from executing.

Examining the run() function...

Handling the cleanup logic... closes the response stream, which causes rx.next_message() to return Ok(None)... but wait, rx is already dropped, this doesn't matter.

Actually, the issue is: what if recv_many() returns 0 but the Worker task is ALREADY blocked on rx.next_message() and never gets to poll recv_many()?

In tokio::select!, both futures are polled. When one of them is ready, the select picks it (randomly if both ready). If rx.next_message() and requests.recv_many() are both pending, both are registered for wakeup.

When requests.recv_many() becomes ready (because all senders dropped), it wakes the Worker task. The Worker task is scheduled. It enters select!, polls both branches. recv_many() is ready with 0 items, rx.next_message() might or might not be ready. select! picks a ready branch...

If only recv_many() is ready (server still idle), it picks that branch → r = 0 → break None. If both are ready, it randomly picks one. Say it picks rx.next_message() first. After processing that message, it loops and recv_many() is still ready with 0 items, so next iteration it picks recv_many() → breaks.

So this should work correctly. The real question is whether all senders to the requests channel are actually dropped. Let me trace through the Sender clones: there's ObjectDescriptorTransport.tx created in Object... After reader.next() returns None and the function logs and returns the data, both descriptor and reader drop out of scope, which means both senders get dropped and the worker's recv_many() sees zero senders and breaks the loop. This should all work correctly, so the hang must be elsewhere — let me trace through handle_range_data to see if reader.next() is actually returning None or if something's keeping a sender alive.

When range_end=false, the lock guard holds the mutex while handle_data is called on the pending read, then the handler sends the data...

but I'm realizing there's a potential issue if handle_data errors when range_end=true — the pending read gets removed from the hashmap before we know if the operation succeeds. When an error bubbles up through handle_ranges, it gets wrapped in an Arc and passed to close_readers to notify all pending reads, then propagates back through handle_response as Some(Err(e)). Finally, in the main loop, this error causes the entire operation to break and return the error to the caller. When pending drops due to the early return from the ? operator, the sender gets dropped too, which closes the data channel and makes the consumer think the read completed normally instead of erroring mid-stream — this is a real bug in the library's error handling, though not the hang we're looking for. Let me step back and check what the Google Cloud Rust maintainers have documented about this API and any known issues, since the original hypothesis about the _tx sender keep-alive might actually be on the right track.

The core problem is that near the end of reading 1000+ objects, some reads just hang with no error — and looking at the v1.9 architecture, the stream stays open until all sender references are dropped, which s3dlio does in one shot. If the stall happens before range_end=true arrives, the Worker is stuck in select! waiting for the next gRPC message.

The server likely stops sending due to HTTP/2 flow control — the client's receive window fills up because the get_object_via_grpc() task isn't being scheduled fast enough to drain data and send WINDOW_UPDATE frames back to the server, causing the server to halt transmission. doesn't send WINDOW_UPDATE, and the server's flow control kicks in, stalling the stream.

But this would only be TEMPORARY — once the Tokio runtime schedules the task, it reads data and sends WINDOW_UPDATE, resuming the stream.

Unless... the Tokio runtime has a livelock/priority inversion where tasks never get scheduled. This is possible if the tokio::spawn() tasks (Workers) are consuming all the runtime threads, starving the inline async tasks.

Actually, in sai3-bench, are the get_object_via_grpc() calls spawned or inline?

From sai3-bench workload.rs, it looks like each worker is a spawned task. The get_object_multi_backend() is called from within a spawned future. So ALL tasks are spawned.

OK so both the Worker tasks and the consumer tasks are spawned, so the Tokio runtime should handle these fairly.

I'm thinking this is actually a permanent high-concurrency HTTP/2 flow control stall. With 1000 objects being read concurrently, each stream has its own HTTP/2 flow control window, but they all share the connection-level window. When all streams are actively receiving, the connection-level window fills up and the server stops sending until the client drains it and sends WINDOW_UPDATE back. If the Tokio runtime gets starved or tasks aren't scheduled frequently enough to drain the buffers, the connection deadlocks.

The real deadlock is likely between the gRPC layer and the application layer: the Worker task's tx.send().await blocks when the channel buffer (capacity 100) is full, which prevents the gRPC reading task from progressing since it's waiting for the Worker to complete so it can read the next chunk.

But actually, the data has already been received from gRPC and is sitting in the Worker's buffer — the Worker just needs to forward it downstream without getting stuck on a full channel.

The consumer should be waiting on recv(), and when the Worker sends data, the consumer wakes up, processes it, and calls recv() again, freeing up channel capacity. This is a standard async producer-consumer pattern that should work fine.

The only real concern would be if both tasks are somehow starving each other on a single-threaded executor, but sai3-bench uses the default multi-threaded Tokio runtime, so the Worker task can block on send().await without preventing the consumer task from running on a different thread.

Now I'm focusing on the actual symptoms — the user said the hang happens near the end after seeing success messages for some objects, which suggests it's not a simple resource leak that would fail from the start. Instead, it looks like the later objects are getting stuck, so I need to trace through what's different about their timing or state.

The threshold behavior (≥1000 consistently fails, 100 always works) is pointing me toward bounded resource exhaustion — either HTTP/2 MAX_CONCURRENT_STREAMS, file descriptors, or some connection pool. The HTTP/2 hypothesis seems strongest since GCS uses HTTP/2 and typically sets MAX_CONCURRENT_STREAMS to 100-500, and with s3dlio's 64 subchannels by default, that could theoretically support thousands of concurrent streams, but the actual server-side limit might be the bottleneck.

The key issue is that each open_object() call creates a persistent BidiReadObject stream that only gets freed when get_object_via_grpc() returns and drops its references. With 64 concurrent workers all holding open streams simultaneously, I'm likely hitting the server's MAX_CONCURRENT_STREAMS limit partway through the workload, causing the tail-end hangs.

Even though the math suggests 64 concurrent streams should fit within typical limits (100 per connection × 64 connections = 6400 total), the real problem might be that streams aren't closing promptly after the data is fully received. The function logs something after reader.next() returns None, but there's likely a delay before the stream actually gets cleaned up on the server side.I think I've found the actual issue. Let me trace through the reconnect() logic again—it's collecting the current ranges and attempting to reconnect with them.

Examining the reconnect implementation more closely...

Continuing through the connector code...

So when the resume policy allows continuing, it reconnects with whatever ranges are pending—or an empty list if there are none. The server sends back an initial response with no data, the handler processes the empty range list successfully, and the Worker gets a fresh stream. Now it's sitting in the select loop waiting for either new messages from the receiver or data from this newly established stream.

But here's the critical issue: the reconnect happens while get_object_via_grpc() is still blocked in its loop waiting for responses. The senders on the requests channel are still alive, so recv_many() doesn't return empty—it just keeps waiting. Meanwhile, the fresh stream is idle with no pending ranges, so rx.next_message() blocks too. Both branches of the select are blocked, and nothing can unblock them because the original request is stuck upstream. That's where the deadlock occurs.

Actually, I'm wondering if there's a race condition here. If the stream delivered the final range_end=true message and the handler sent the data to reader.inner, but then the network error happened before get_object_via_grpc() actually polled that channel, the data would be sitting there unread.

So the real question is: when does reader.inner.recv() actually block forever? It only happens if the channel is empty AND at least one sender is still alive. Since each range creates exactly one sender in map_ranges(), and handle_data might clone it, I need to figure out which senders could still be holding a reference when the channel should be drained.

The Handler objects that receive cloned senders are temporary — they send their data and then drop the clone. The original sender lives in ActiveRead and only gets dropped when range_end=true arrives. So if that signal never comes because the gRPC stream stalls partway through a range, and the reconnect logic doesn't properly re-request the remaining data, then recv() would block indefinitely waiting for a sender that will never close.

Checking how reconnect handles partial reads...

The RemainingRange methods show that when reconnecting, we request only the bytes we haven't received yet, not the full original range. So if the stream stalls silently — no error, just hanging — then rx.next_message() is stuck waiting on the gRPC stream that's gone unresponsive.

The HTTP/2 keepalive would eventually timeout after 20+ minutes, but no reconnect triggers because there's no actual error. The reader just hangs indefinitely waiting for data that never arrives. This explains the observed behavior perfectly.

The key difference between v1.8 and v1.9 is the streaming model: v1.8 likely used simple server-streaming where the server closes the stream after sending all data, whereas v1.9's bidirectional streaming keeps the connection open, making it vulnerable to silent stalls where the client never realizes the server has stopped responding.


