# s3dlio — session rules

The parent [/home/eval/Documents/Code/CLAUDE.md](../CLAUDE.md) Prime Directives
apply here in full and are not repeated. In particular, remember Prime
Directive #1 (never push without an explicit instruction) and Prime
Directive #2 (only touch the repo you were asked to change).

The rules below are additive.

## No pushes, no PRs, no publishes — s3dlio-specific reinforcement

- `git push`, `git push --tags`, `gh pr create`, `gh release create`,
  `cargo publish`, `maturin publish`, and `twine upload` are all
  forbidden without an explicit instruction from the user in the current
  turn. "Bump the version," "commit the fix," "run the tests" — none
  of these authorize a push.
- Version bumps in `Cargo.toml` / `pyproject.toml` are **local file
  edits**. They do not entitle you to push, tag, publish a wheel, or
  touch a downstream repo's pin. Those are all separate explicit steps.
- Approaching a 1.0 release makes this rule *more* important, not less.
  An accidental publish of a broken pre-1.0 wheel is harder to undo than
  a bad commit on a local branch.

## Version-bump scope

When the user says "bump the version" or "we need to go to v0.9.X",
that refers to s3dlio's OWN version files ONLY:

- `s3dlio/Cargo.toml` — `[workspace.package] version = "..."` (single
  source of truth; the subcrate `crates/s3dlio-oplog/Cargo.toml`
  inherits via `version.workspace = true`).
- `s3dlio/pyproject.toml` — the `[project] version = "..."` line for
  the Python wheel.
- `s3dlio/docs/Changelog.md` — new version heading and change notes.
- `s3dlio/docs/Environment_Variables.md` — Version History entry.
- In-code doc comments in `src/` that cite the new version.

Do **not** touch downstream pin files in sibling repos
(`DLIO_local_changes/pyproject.toml`, `mlp-storage/pyproject.toml`, or
any other) unless the user explicitly names those repos in the current
turn. The version-sync memory rule reminds you those files *will need*
updating eventually — that reminder is a prompt to ask, not
pre-authorization to act.

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

### Python build + test environment — always uv, always ./build_pyo3.sh

- **All Python testing runs under `uv`.** Do not invoke `python` /
  `python3` / `pytest` directly on the system interpreter, and do not
  create your own venv. Use `uv run pytest ...`, `uv run python ...`,
  etc. — `uv` reads `pyproject.toml` and resolves the correct
  interpreter and dependency set for this project.
- **To build the s3dlio Python wheel, use the in-repo script
  [`./build_pyo3.sh`](build_pyo3.sh)**, not raw `maturin build`. The
  script handles platform detection (x86_64/aarch64, Linux/Darwin),
  feature-set selection (`default`/`slim`/`full`), and produces the
  wheel under `target/wheels/`. Do NOT call `maturin` directly —
  arguments and feature flags in the script are the tested combination
  for this repo.
- Example:
  ```bash
  ./build_pyo3.sh          # AWS + file/direct backends (default/slim)
  ./build_pyo3.sh full     # + Azure + GCS
  uv run pytest tests/...  # run Python tests against the built wheel
  ```

### Pre-push quality gate

Before pushing any code change, run in this order and only push if all three
are clean:

1. `cargo fmt --all -- --check`
2. `cargo clippy --all-targets --all-features -- -D warnings`
3. `cargo test --lib` (unit tests) — plus any integration tests specifically
   touching the changed area.

For Python wheel changes, additionally rebuild via `./build_pyo3.sh` (see
above) and smoke-test against the wheel via `uv run pytest`.

For any change touching `python/` (including `python/s3dlio/...` and
`python/tests/...`), additionally run (per the parent CLAUDE.md rule #7):

4. `ruff check python/`
5. `ruff format --check python/`
6. `uv run python -m compileall -q python/s3dlio python/tests`

All three must be clean before considering the Python side of a change done.

### Zero warnings — no exceptions, no underscore hacks

Per the parent Prime Directive #4, never leave a warning behind. In this
project specifically:

- `cargo test`, `cargo build`, `cargo clippy` must all produce zero warnings
  from crate-local code. Warnings from dependencies are out of scope; ours
  are not.
- **Never prefix an unused variable, argument, or field with `_` to silence
  the compiler.** In s3dlio, that pattern has historically hidden
  Send/Sync bound violations that were only meant to be caught at
  compile-time, missed `.await` chains that dropped in-flight tasks, and
  buffer offsets that were computed but never written to. If a binding is
  unused, delete it — or find the missing call site and wire it up. Do
  not rename it to hide.
- If a warning fires from `tests/common/mod.rs` in a specific test binary,
  that means the test binary is `mod common;`-ing utilities it doesn't
  use. Fix it by narrowing what the test binary imports, not by
  broad-brush `#[allow(dead_code)]` on the shared module.
- `#[allow(...)]` on a specific item is acceptable only when the warning
  is a documented false positive AND the reason is written in a
  one-line comment right above the attribute.

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
