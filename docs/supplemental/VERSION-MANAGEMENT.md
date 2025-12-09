# Version Management

## Workspace-Level Version Inheritance (v0.8.15+)

Starting with version 0.8.15, s3dlio uses **workspace-level version inheritance** to manage versions across all Rust crates from a single location.

### How It Works

**Root `Cargo.toml`** defines the workspace version:
```toml
[workspace]
members = [".", "crates/s3dlio-oplog"]

[workspace.package]
version = "0.8.15"        # ← Single source of truth for Rust crates
edition = "2021"
authors = ["Russ Fellows"]
license = "MIT"
```

**Member crates** inherit the version:
```toml
[package]
name = "s3dlio"
version.workspace = true   # ← Inherits from workspace.package
edition.workspace = true
```

```toml
[package]
name = "s3dlio-oplog"
version.workspace = true   # ← Inherits from workspace.package
edition.workspace = true
```

### Updating Versions

To update the version for a release, you only need to change **TWO files**:

1. **`Cargo.toml`** (root workspace):
   ```toml
   [workspace.package]
   version = "0.8.16"  # ← Update here
   ```

2. **`pyproject.toml`** (Python package):
   ```toml
   [project]
   version = "0.8.16"  # ← Update here
   ```

All Rust crates (`s3dlio`, `s3dlio-oplog`, etc.) automatically inherit the workspace version.

### Verification

Check all crate versions:
```bash
cargo metadata --format-version 1 2>/dev/null | \
  jq -r '.packages[] | select(.name | startswith("s3dlio")) | "\(.name): \(.version)"'
```

Expected output:
```
s3dlio: 0.8.15
s3dlio-oplog: 0.8.15
```

### Benefits

✅ **Single source of truth** for Rust crate versions  
✅ **No more version drift** between workspace members  
✅ **Easier releases** - update version in one place  
✅ **Less prone to errors** - can't forget to update a crate  

### Notes

- **Python version** (`pyproject.toml`) must still be updated manually (no Rust/Python version sync)
- **Available since Rust 1.64** - workspace inheritance is stable
- **Other inheritable fields**: `authors`, `license`, `edition`, `repository`, `homepage`, etc.

### Release Checklist

When releasing a new version:

- [ ] Update `[workspace.package] version` in root `Cargo.toml`
- [ ] Update `[project] version` in `pyproject.toml`
- [ ] Update `docs/Changelog.md` with release notes
- [ ] Update README.md version badge (optional)
- [ ] Run `cargo build --release` to verify
- [ ] Run `./build_pyo3.sh && ./install_pyo3_wheel.sh` for Python
- [ ] Commit with message: `chore: bump version to 0.8.X`
- [ ] Tag release: `git tag v0.8.X`
