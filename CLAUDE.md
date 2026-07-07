# s3dlio — session rules

The parent [/home/eval/Documents/Code/CLAUDE.md](../CLAUDE.md) Prime Directives
apply here in full and are not repeated. The rules below are additive.

## RED/GREEN test discipline (applies to every bug fix, every project)

Every bug fix must land alongside tests that provably fail against unmodified
code and pass after the fix. The workflow is:

1. **RED.** Write the new test(s) *first*. Run them against the current tree
   with the fix *not applied*. Confirm they FAIL for the reason the bug says
   they should. If a test can't be made to go RED before the fix is applied,
   it isn't a regression test — figure out why (test hitting the wrong code
   path, wrong assertion, wrong fixture) before continuing.
2. **GREEN.** Apply the fix. Re-run the same tests. Confirm they now PASS.
3. **Commit both together.** Tests and fix land in the same commit (or same
   PR at minimum), so the RED-then-GREEN transition is bisectable.
4. **Report the RED-then-GREEN transition** in the PR description or commit
   message, not just "added tests" — a reviewer should be able to see that
   the test would have caught the bug had it existed earlier.

This applies to *every* project under `/home/eval/Documents/Code/`, not just
s3dlio — the parent CLAUDE.md carries the same rule.

## s3dlio-specific rules

### Version bumps sync with downstream pins

Any change to `Cargo.toml`'s `version` must be paired, *in the same pass*, with
updates to:

- `DLIO_local_changes/pyproject.toml`'s `s3dlio>=X.Y.Z` pin (and its
  `[tool.uv.sources]` local wheel path if used).
- `mlp-storage/pyproject.toml`'s `s3dlio` pin (same reasoning).

Never let `Cargo.toml` drift ahead of the downstream pyproject pins — testing
DLIO/mlp-storage against a stale wheel that doesn't have the behavior the pin
promises has bitten us before.

Also update `docs/Changelog.md` in the same commit as the version bump.

### Pre-push quality gate

Before pushing any code change, run in this order and only push if all three
are clean:

1. `cargo fmt --all -- --check`
2. `cargo clippy --all-targets --all-features -- -D warnings`
3. `cargo test --lib` (unit tests) — plus any integration tests specifically
   touching the changed area.

For Python wheel changes, additionally rebuild via `maturin` and smoke-test
against the wheel in a `uv` env.

### Stability first — s3dlio is approaching 1.0

Correctness > performance > new features. Do not destabilize working code for
marginal performance wins. When a change plausibly touches a hot path or a
retry/error-handling loop, the required test bar is a *fault-injection* test
that exercises the failure mode, not just a happy-path assertion.

### Test-suite scope

- `tests/` contains many integration tests; some require live cloud endpoints
  or specific env vars. When in doubt, prefer `cargo test --lib` (unit tests)
  or specific `--test <name>` invocations for the area you touched — do not
  blindly run the whole `tests/` suite as a smoke test.
- The lurking-issue audit for issue #148 lives at
  [docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md](docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md);
  it is the authoritative plan for that work until superseded.

### Branch/tag conventions

- Tags (`vX.Y.Z`) are for released versions only, cut when the wheel is
  published to PyPI. Do not tag mid-work checkpoints.
- Feature branches: `<area>/<issue>-<slug>` (e.g. `perf/148-phase1-buffer-copies`).
- Umbrella branches for multi-phase work land as one merge to `main` at the
  end (or as several sequential PRs, at maintainer discretion).
