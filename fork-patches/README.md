# AWS Smithy HTTP Client Fork Patches

This directory contains patches to expose HTTP client configuration that AWS SDK doesn't currently provide publicly.

## What We're Patching

The `aws-smithy-http-client` crate has `hyper_builder()` methods marked `pub(crate)` which prevents us from configuring:
- HTTP connection pool size
- Connection keep-alive settings  
- HTTP/2 multiplexing
- Other performance-critical hyper client settings

## Minimal Patch Strategy

Instead of maintaining a full fork, we'll create minimal patches to expose just what we need:
1. Change `pub(crate) fn hyper_builder()` to `pub fn hyper_builder()`
2. Change `pub(crate) fn set_hyper_builder()` to `pub fn set_hyper_builder()`

This gives us access to configure hyper's connection pooling for 10+ GB/s performance.

## Files to Patch

From aws-smithy-http-client v1.1.0:
- `src/client.rs` lines 358 and 369

## Implementation Plan

1. Create a custom [patch] section in Cargo.toml
2. Use git patch or direct replacement
3. Add HTTP connection pool optimization to s3dlio
4. Test performance improvements
5. Consider submitting upstream PR to AWS

## Target Performance

Goal: Achieve 10+ GB/s GET performance (warp benchmark shows 11.38 GB/s is possible with optimized HTTP client)
