# Release Checklist

## ⚠️ CRITICAL: Always Complete Before Committing a Release

This checklist must be completed **BEFORE** committing and pushing any release branch.

### Pre-Commit Checklist

- [ ] **Update Version Numbers**
  - [ ] `Cargo.toml` - Update `[workspace.package] version`
  - [ ] `pyproject.toml` - Update `[project] version`
  - [ ] Verify with: `cargo metadata --format-version 1 | jq -r '.packages[] | select(.name | startswith("s3dlio")) | "\(.name): \(.version)"'`

- [ ] **Update Documentation** ⚠️ REQUIRED
  - [ ] `docs/Changelog.md` - Add release notes with date, version, and changes
  - [ ] `README.md` - Update version badge and test count if changed
  - [ ] Update any API documentation if interfaces changed

- [ ] **Code Quality**
  - [ ] Run `cargo build --release` - ensure ZERO warnings
  - [ ] Run `cargo clippy --package s3dlio-oplog -- -D warnings` - fix all issues
  - [ ] Run `cargo test --package s3dlio-oplog` - all tests pass
  - [ ] Run example if added: `cargo run --example <name> --package s3dlio-oplog`

- [ ] **Python Integration** (if applicable)
  - [ ] Run `./build_pyo3.sh && ./install_pyo3_wheel.sh`
  - [ ] Test Python imports: `python -c "import s3dlio; print('Version:', s3dlio.__version__)"`

- [ ] **Git Workflow**
  - [ ] Create feature branch: `git checkout -b feature/<name>-v0.x.y`
  - [ ] Stage all changes: `git add -A`
  - [ ] Verify staged files: `git status`
  - [ ] Commit with detailed message (see template below)
  - [ ] Push branch: `git push -u origin feature/<name>-v0.x.y`

- [ ] **Create Pull Request**
  - [ ] Create PR with comprehensive description
  - [ ] Link any related issues
  - [ ] Request review if working in team

### Post-Merge Checklist

- [ ] **Tag Release**
  - [ ] Switch to main: `git checkout main`
  - [ ] Pull latest: `git pull`
  - [ ] Create tag: `git tag -a v0.x.y -m "Release v0.x.y"`
  - [ ] Push tag: `git push origin v0.x.y`

- [ ] **Verify Release**
  - [ ] Check CI/CD passes (if configured)
  - [ ] Test installation from clean environment
  - [ ] Verify documentation renders correctly

### Commit Message Template

```
<type>: <short summary> (v0.x.y)

## <Feature/Fix Name>

### What Changed
- Key change 1
- Key change 2
- Key change 3

### Implementation Details
- Technical detail 1
- Technical detail 2

### Documentation
- Added/Updated docs/...
- Added/Updated README.md
- Added/Updated Changelog.md

### Files Modified
- File 1: description
- File 2: description

### Testing
- X tests passing
- Zero warnings

### Breaking Changes
- None / List breaking changes

Version: 0.x.y
```

## Common Mistakes to Avoid

❌ **DO NOT**:
- Commit without updating `docs/Changelog.md`
- Commit without updating `README.md` version badge
- Push without running tests
- Merge with compiler warnings
- Forget to update both Rust and Python versions
- Skip documentation for new features

✅ **ALWAYS**:
- Update Changelog.md with date and release notes
- Update README.md if version, features, or test count changed
- Run full test suite before committing
- Verify zero warnings with `cargo build --release`
- Document all new features and breaking changes
- Test Python integration if Rust API changed

## Quick Version Update Commands

```bash
# 1. Update versions (2 files)
# Edit Cargo.toml: [workspace.package] version = "0.x.y"
# Edit pyproject.toml: version = "0.x.y"

# 2. Update documentation (REQUIRED!)
# Edit docs/Changelog.md - add release notes
# Edit README.md - update version badge and features

# 3. Verify versions
cargo metadata --format-version 1 2>/dev/null | \
  jq -r '.packages[] | select(.name | startswith("s3dlio")) | "\(.name): \(.version)"'

# 4. Run tests
cargo build --release 2>&1 | grep -i warning  # Should be empty
cargo test --package s3dlio-oplog

# 5. Commit and push
git checkout -b feature/<name>-v0.x.y
git add -A
git status  # Verify docs/Changelog.md and README.md are staged
git commit -m "feat: <description> (v0.x.y)"
git push -u origin feature/<name>-v0.x.y
```

## Automation Ideas (Future)

Consider adding to CI/CD:
- Pre-commit hook to check if Changelog.md updated when version changes
- Automated version badge updates in README.md
- Automated test count updates
- Git hook to verify documentation updates
