# s3dlio — rules for coding agents

## RULE #1 — ZERO COPY, EVERYWHERE

**VERY IMPORTANT: The s3dlio repository is ALWAYS Zero Copy EVERYWHERE, unless OTHERWISE
EXPLICITLY noted in the documentation, and comments.**

The entire reason s3dlio exists is to be faster than libraries that copy payload data around
in memory. Every copy of payload bytes is a defect unless it is one of the few documented
exceptions below.

- **Never copy payload data** to convert, buffer, "simplify", satisfy a type signature or make
  ownership easier: no `to_vec()`, `Bytes::copy_from_slice`, `extend_from_slice` into an
  accumulation buffer, `Vec::from(&[u8])`, `.clone()` of a buffer, `PyBytes::new(py, &data)`,
  `bytes(obj)`, `.tobytes()` or `read()` into a new buffer on a payload path. Hand data along as
  `Bytes` (refcounted slices), borrowed exports kept alive by a guard, or a caller's buffer.
- **Accepting another data type never means copying it.** numpy arrays, bytearray, memoryview,
  `BytesView` and other buffer-protocol objects are read in place through the buffer protocol.
  Results go back as `BytesView` (or into the caller's buffer), never as a new `bytes` copy.
- **The only acceptable copies** are the unavoidable ones: the single copy from the network /
  SDK chunk (or the kernel) into the final destination, a transform (compression, encryption,
  format decode), O_DIRECT bounce buffers for unaligned memory, a non-contiguous source that must
  be gathered, an object that only offers `__bytes__`, an API that explicitly returns `bytes` for
  compatibility, or a copy the caller explicitly opts into. Each such copy MUST be documented in
  the user documentation AND in a comment at the copy site saying why it cannot be avoided.
- **Every change to a data path proves it is zero-copy** with a test (pointer identity, a
  refcount/owner check, or a peak-memory bound), not with a claim.
- **Why copies matter: performance.** Each copy of the payload costs roughly half the throughput
  of the path it sits on; memory use is secondary. So the test of any copy is measured speed: a
  single copy is acceptable when it makes the path measurably FASTER (for example coalescing many
  tiny writes into one part buffer instead of tracking thousands of borrowed buffers). Such a copy
  must be bounded, measured (benchmark numbers in the commit or plan), documented and commented
  like the other exceptions.
- If you think a copy is unavoidable, it is not your call: ask the maintainer first.

## The other rules

[CLAUDE.md](CLAUDE.md) holds the rest of this repository's rules (no push/PR/publish without an
explicit instruction, RED/GREEN tests for every fix, quality gates, zero warnings, build and
Python test environment). They apply to every agent, not only Claude.
