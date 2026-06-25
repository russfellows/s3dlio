# Draft reply for mlcommons/storage#506

> This is a *draft* sitting in `docs/issue506-draft-reply.md` — review
> and post when ready.  Not committed to any history beyond the s3dlio
> branch `fix/506-sdk-error-chain`.

---

@austingnanaraj @FileSystemGuy thanks for the detailed report and
analysis.  We're landing fixes in `russfellows/s3dlio` and
`mlcommons/DLIO_local_changes` and there are a few things that will
materially help on the next run.

## What "dispatch failure" actually means

It's worth flagging upfront: the bare string `RuntimeError: dispatch
failure` is **misleading**.  It looks like a Tokio scheduling failure,
but `SdkError::DispatchFailure` is the AWS Smithy runtime's name for
"the HTTP request couldn't be completed at the transport layer."  That
covers I/O errors, connect timeouts, TLS handshake failures, DNS
errors, and connection refused — none of which involve Tokio's
scheduler at all.

The reason that distinction never reached the Python side is a missing
`format_sdk_error()` call on the GET / HEAD / PUT / range hot paths.
The helper has been in `s3_utils.rs` for a while and is already wired
on the LIST path, but match arms on the other paths were converting
`SdkError → anyhow` straight via `?` or `.context()`, which drops
everything below the outer message.

**Patch landed on `fix/506-sdk-error-chain`** — wraps every SdkError
leak site (s3_ops.rs ×7, s3_utils.rs ×6, object_store.rs ×5,
multipart.rs ×3) with `format_sdk_error()` so the next run will say
e.g.:

```
RuntimeError: S3 GET for s3://retinanet3/.../img.jpg failed:
  dispatch failure (connection timeout: connect error: connection
  attempt timed out after 5 seconds) — server did not respond within
  the connect timeout — verify AWS_ENDPOINT_URL is correct and the
  host is reachable
```

instead of just `dispatch failure`.  A `cargo test` regression guard
(`test_sdk_anyhow_keeps_context_in_top_message`) locks in the
contract.

## Five of the env vars in your mpirun command silently do nothing

While auditing, I found that several `S3DLIO_*` names you passed
through `--mpi-params -x` are not actually read by any code in s3dlio.
Specifically:

| You set | Reality |
|---|---|
| `S3DLIO_MAX_CONCURRENCY=32` | name doesn't exist in s3dlio source |
| `S3DLIO_MULTIPART_THRESHOLD=1073741824` | doesn't exist (the *similar* name `S3DLIO_MULTIPART_THRESHOLD_MB` IS read, but only by DLIO's `obj_store_lib.py`, not by s3dlio; and your value is in bytes, the MB version expects MB) |
| `S3DLIO_PART_SIZE=268435456` | doesn't exist |
| `S3DLIO_CONNECTION_TIMEOUT=300` | doesn't exist; you probably meant the documented `S3DLIO_CONNECT_TIMEOUT_SECS`, **which itself was a doc promise that wasn't wired until now** |
| `S3DLIO_READ_TIMEOUT=300` | doesn't exist; the wired equivalent is `S3DLIO_OPERATION_TIMEOUT_SECS` (default 60 s — your 300 s would have helped if the name had matched) |

So in your last run:

- You probably hit a 5 s SDK-layer connect timeout (yes, **5 seconds** —
  hardcoded), not your intended 5 minutes.
- You had the AWS-SDK default operation timeout of 60 s, not 300 s.

Both of these are plausibly causal for the cold-start dispatch
failures.  2 048 concurrent connects on a single endpoint at warmup is
not unreasonable for these machines, but if the endpoint's TCP accept
queue takes more than 5 s to drain that burst, every late-arriving
worker's connect times out and returns `dispatch failure`.

**Patch landed on `fix/506-sdk-error-chain`** — `S3DLIO_CONNECT_TIMEOUT_SECS`
is now actually honored (default 10 s; both the reqwest transport and
the SDK `TimeoutConfig` agree on it now).  Plus an updated docs block
listing the dead names explicitly so they stop propagating.

## What I'd ask you to try next

Once the patched s3dlio build is in hand:

1. **Add `RUST_BACKTRACE=full`** to the mpirun env.  With the new error
   formatting that's strictly redundant for the top-level cause string,
   but it gives us the full Tokio frame in case the failure is *not* a
   transport error.

2. **Use the right env-var names** (or just delete the inert ones from
   your script):
   ```bash
   # Real, wired knobs that matter for cold-start fan-out:
   export S3DLIO_CONNECT_TIMEOUT_SECS=30           # was effectively 5 s before
   export S3DLIO_OPERATION_TIMEOUT_SECS=120        # was the default 60 s
   export S3DLIO_POOL_MAX_IDLE_PER_HOST=0          # unlimited; keep warm conns
   export S3DLIO_POOL_IDLE_TIMEOUT_SECS=300        # don't tear them down
   export S3DLIO_RT_THREADS=16                     # per process; tune to ranks_per_node
   # Drop these — they do nothing:
   # S3DLIO_MAX_CONCURRENCY, S3DLIO_CONNECTION_TIMEOUT, S3DLIO_READ_TIMEOUT,
   # S3DLIO_MULTIPART_THRESHOLD, S3DLIO_PART_SIZE
   ```

3. **If it still fails**, narrow with the recipe @FileSystemGuy
   already wrote up:
   - `++workload.storage.storage_options.prefetch_window=8` to drop
     per-worker fanout from 64 → 8 and see if the failure moves.
   - Test with one S3 endpoint (`unset S3_ENDPOINT_URIS;
     AWS_ENDPOINT_URL=<one of the four>`) to isolate the round-robin
     path.
   - Confirm `ulimit -n` on the client hosts is ≥ 65 536 per process
     (not just per shell).

## Side note on the 20-minute listing

You're correct that `++workload.dataset.skip_listing=true` should have
made the 20-minute listing phase disappear.  That turns out to be a
separate bug in DLIO_local_changes: `LoadConfig()` was reading every
other `dataset.*` field out of the Hydra config tree but had no
handler for `skip_listing` or `listing_validation_interval`, so the
field stayed at its default `False` no matter what the override said.

**Fix landed on DLIO branch `fix/504-skip-listing-loadconfig`** (also
addresses #504 directly).  11 new regression tests cover every value
shape Hydra can emit, including the lowercase string form you used.

---

Pinging @russfellows for review of all of the above before this gets
posted to the issue thread.
