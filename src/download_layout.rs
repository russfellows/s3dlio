// src/download_layout.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Where a downloaded object is written (issue #284).
//!
//! A recursive download keeps each object's key path RELATIVE TO THE LISTING BASE:
//! `s3-cli download -r s3://b/dl/ out` writes `dl/a/x.bin` to `out/a/x.bin` and `dl/b/x.bin`
//! to `out/b/x.bin`. The listing base is the one [`crate::list_pattern::relative_to_listing_base`]
//! computes (the rule `--pattern` already uses): the directory a `dir/` or `dir` URI selects,
//! the directory before the first `*` of a glob, or the parent directory of a single exact
//! object (so a single object lands in `<dest>/<object name>`).
//!
//! # One base per listing
//!
//! The base is decided once for the whole listing, never per object. When a URI without a
//! trailing `/` selects an object AND the directory of the same name (`s3://b/dl` with the
//! objects `dl`, e.g. a zero-byte folder marker, and `dl/a`, `dl/b/c`), the base is the PARENT
//! of `dl` for every object: `dl/a` goes to `<dest>/dl/a` and `dl/b/c` to `<dest>/dl/b/c`,
//! never quietly relocated to `<dest>/a`. Without the object itself (only `dl/...` entries)
//! the directory is the base, as for `dl/`. This changes only where files go; WHICH objects a
//! URI selects is decided by the listing, and `--pattern` still sees the key relative to the
//! base [`relative_to_listing_base`] computes.
//!
//! # Object and directory of one name
//!
//! A file system cannot hold the object `dl` and the directory `dl/`. The DIRECTORY'S
//! CONTENTS WIN, in any listing order and for a glob as for a plain URI: the objects below
//! `dl/` are planned first and only the single object `dl` is refused, with the reason
//! "object 'dl' and directory 'dl/' cannot both exist locally ..." that names the way to get
//! the directory alone (`s3://b/dl/`) or the object alone (non-recursively). The per-object
//! download paths count it as one failed object; the fail-fast `s3_copy` helpers refuse the
//! whole call with the same message.
//!
//! Only a directory that holds at least one object that is itself downloadable counts: if
//! EVERY child under `dl/` is refused (unsafe key, collision), the directory never exists
//! locally and the object `dl` is downloaded as an ordinary file. In a chain (`p`, `p/q`,
//! `p/q/r`) every object that has a valid child below it is refused (`p` and `p/q`); only
//! the deepest, `p/q/r`, is downloaded.
//!
//! # Source spellings
//!
//! The stores return their entries in one canonical form: `gs://bucket/key` for Google Cloud
//! Storage (also for a `gcs://` source) and `az://account/container/key` for Azure (also for
//! the `https://<account>.blob.core.windows.net/<container>/key` source). The source URI is
//! mapped to that form before the relative path is computed, with the same parsers the stores
//! use. An Azure account name is canonicalised like the store does: lower-cased for the
//! default endpoint, kept as written for a custom endpoint (an emulator).
//!
//! # Safety
//!
//! A key is data from a remote server, so it is never trusted as a path. The relative key is
//! split on `/` AND `\` (a backslash is a separator on every platform, so the result is the
//! same everywhere) and the object is REFUSED -- as a per-object error, never a panic, and
//! nothing is written for it -- when the relative key
//!
//! * is empty, absolute (leading `/` or `\`), or has an empty component (`a//b`);
//! * has a `.` or `..` component;
//! * contains a NUL byte;
//! * has a component that is exactly a Windows drive designator (one ASCII letter and a colon:
//!   `C:`). Only that exact form is refused: `a:b`, `x:y` and `v:1/x` are ordinary keys;
//! * does not lie under the listing base at all (the entry is then still a whole URI).
//!
//! Every accepted component is appended to the destination one at a time, so the result is
//! always `<dest_dir>/<normal components>`. The components below `dest_dir` that already exist
//! are checked with `symlink_metadata`, and a pre-existing symlink among them (or as the file
//! itself) is refused rather than followed. On Unix the file itself is then opened with
//! `O_NOFOLLOW` (create, truncate, write), so a symlink planted at the final component
//! between the check and the write is refused too. The limit that remains: an INTERMEDIATE
//! directory swapped for a symlink by another local process between the check and the write is
//! not detected (std offers no portable `openat` walk). `dest_dir` itself may be a symlink;
//! that is the caller's own choice. The checks, the directory creation and the write for one
//! object run in a single blocking section.
//!
//! A write that fails after the parent directories were created (disk full, permission) leaves
//! those directories behind, empty; a partly written file is not removed either.
//!
//! # Collisions
//!
//! Two objects that map to the same local path (`a/x` and `a/x` listed twice) are decided
//! BEFORE anything is downloaded, in listing order: the first keeps the path, every later one
//! fails with a collision error (an object and a directory of one name: see above). Nothing is
//! ever silently overwritten within one run. Case-insensitive filesystems can still merge
//! names that differ only in case, and a file that already exists in `dest_dir` from an
//! earlier run is replaced, exactly as before.
//!
//! # Directory markers
//!
//! Entries ending in `/` (zero-length "folder" objects, S3 common prefixes) are skipped:
//! no directory is created for them and they are not counted as attempted.
//!
//! # Messages
//!
//! Failure reasons are meant to be shown to the caller; `redact::redact_uri` (the one
//! redactor, also used by the Python errors) removes `userinfo@` and signed-URL query strings
//! from the URIs in them, and every control character of a remote key is escaped
//! (`redact::escape_control`, which also escapes invisible bidi / format characters), so a
//! hostile key cannot forge log lines or terminal output or spoof the text. The cause chain of
//! a failed fetch quotes the raw key, so the final text of each failure reason, of the
//! returned error and of the warning logged for it is escaped where it is rendered.

use anyhow::{anyhow, bail, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::list_pattern::relative_to_listing_base;
use crate::object_store::{infer_scheme, Scheme};
use crate::redact::{escape_control, redact_and_escape};

/// One object to download and where it goes, decided up front.
#[derive(Debug)]
pub(crate) struct PlannedDownload {
    /// The listed entry (URI or key).
    pub entry: String,
    /// The validated components below the destination directory, or why the object is refused.
    pub target: Result<Vec<String>>,
}

/// The account name as the Azure store canonicalises it (so as it appears in the entries).
fn store_account_name(account: &str) -> String {
    #[cfg(feature = "backend-azure")]
    {
        crate::azure_client::canonical_account_name(account).unwrap_or_else(|_| account.to_string())
    }
    #[cfg(not(feature = "backend-azure"))]
    {
        account.to_ascii_lowercase()
    }
}

/// `src_uri` spelled the way the store that lists it spells its entries: `gs://bucket/key`
/// for `gs://` and `gcs://`, `az://account/container/key` for `az://` and the Azure blob host
/// form (account canonicalised by `canonical_account`). Every other URI, and one the
/// parsers refuse (the listing fails first anyway), is returned unchanged.
fn canonical_source(src_uri: &str, canonical_account: &dyn Fn(&str) -> String) -> String {
    match infer_scheme(src_uri) {
        Scheme::Gcs => match crate::capabilities::parse_gs_uri_pure(src_uri) {
            Ok((bucket, key)) => format!("gs://{bucket}/{key}"),
            Err(_) => src_uri.to_string(),
        },
        Scheme::Azure => match crate::capabilities::parse_azure_uri_pure(src_uri) {
            Ok((account, container, key)) => {
                format!("az://{}/{container}/{key}", canonical_account(&account))
            }
            Err(_) => src_uri.to_string(),
        },
        _ => src_uri.to_string(),
    }
}

/// The listing prefix the relative paths of a download are computed against: the canonical
/// spelling of `src_uri` ([`canonical_source`]) and, when `src_uri` has no trailing `/` and
/// the listing contains the object of exactly that name, the parent of that name (one base
/// for the object and for the `name/` directory beside it; see the module docs).
fn layout_prefix(
    src_uri: &str,
    entries: &[String],
    canonical_account: &dyn Fn(&str) -> String,
) -> String {
    let prefix = canonical_source(src_uri, canonical_account);
    if prefix.ends_with('/') {
        return prefix;
    }
    let is_glob = prefix.contains('*') && (!prefix.contains("://") || prefix.starts_with("s3://"));
    if is_glob {
        return prefix;
    }
    if let Some((_, after_scheme)) = prefix.split_once("://") {
        if !after_scheme.contains('/') {
            return prefix; // bucket / authority only: no key to be an object
        }
    }
    let (parent, name) = prefix.split_at(prefix.rfind('/').map_or(0, |slash| slash + 1));
    if name.is_empty() {
        return prefix;
    }
    if entries
        .iter()
        .any(|entry| relative_to_listing_base(parent, entry) == name)
    {
        return parent.to_string();
    }
    prefix
}

/// The validated components of `entry` relative to the listing base of `listing_prefix`.
pub(crate) fn relative_components(listing_prefix: &str, entry: &str) -> Result<Vec<String>> {
    let rel = relative_to_listing_base(listing_prefix, entry);
    if rel.len() == entry.len() && entry.contains("://") {
        bail!(
            "{}: not under the listing base of {}",
            redact_and_escape(entry),
            redact_and_escape(listing_prefix)
        );
    }
    let shown = redact_and_escape(entry);
    if rel.is_empty() {
        bail!("{shown}: empty path relative to the listing base");
    }
    if rel.contains('\0') {
        bail!("{shown}: key contains a NUL byte");
    }
    if rel.starts_with('/') || rel.starts_with('\\') {
        bail!("{shown}: absolute key path {rel:?} refused");
    }
    let mut out = Vec::new();
    for part in rel.split(['/', '\\']) {
        match part {
            "" => bail!("{shown}: empty path component in {rel:?} refused"),
            "." | ".." => bail!("{shown}: {part:?} path component in {rel:?} refused"),
            _ => {}
        }
        // Exactly `<ASCII letter>:` is a Windows drive designator; `x:y` and `v:1` are not.
        if let [letter, b':'] = part.as_bytes() {
            if letter.is_ascii_alphabetic() {
                bail!("{shown}: drive designator component {part:?} in {rel:?} refused");
            }
        }
        out.push(part.to_string());
    }
    Ok(out)
}

/// Plan every non-directory entry of a listing of `listing_prefix`: map it to its local
/// components and settle collisions (see the module docs: the first claim of a path wins, and
/// the contents of a directory win over an object of the same name).
pub(crate) fn plan_downloads(listing_prefix: &str, entries: &[String]) -> Vec<PlannedDownload> {
    plan_downloads_with(listing_prefix, entries, &store_account_name)
}

fn plan_downloads_with(
    listing_prefix: &str,
    entries: &[String],
    canonical_account: &dyn Fn(&str) -> String,
) -> Vec<PlannedDownload> {
    let base = layout_prefix(listing_prefix, entries, canonical_account);
    let mut mapped: Vec<(&String, Option<Result<Vec<String>>>)> = entries
        .iter()
        .filter(|entry| !entry.ends_with('/'))
        .map(|entry| (entry, Some(relative_components(&base, entry))))
        .collect();

    // Every directory that holds at least one validly mapped object. An object whose local
    // path is such a directory (a zero-byte "folder marker" `dl` next to `dl/a`) is planned
    // AFTER all the others, so the directory's contents win and only that object is refused.
    let mut directories: HashSet<String> = HashSet::new();
    for (_, target) in &mapped {
        if let Some(Ok(comps)) = target {
            let mut ancestor = String::new();
            for comp in &comps[..comps.len().saturating_sub(1)] {
                if !ancestor.is_empty() {
                    ancestor.push('/');
                }
                ancestor.push_str(comp);
                directories.insert(ancestor.clone());
            }
        }
    }
    let is_directory_too = |target: &Option<Result<Vec<String>>>| matches!(target, Some(Ok(comps)) if directories.contains(&comps.join("/")));

    let mut files: HashSet<String> = HashSet::new();
    let mut dirs: HashSet<String> = HashSet::new();
    let mut settled: Vec<Option<Result<Vec<String>>>> = (0..mapped.len()).map(|_| None).collect();
    for deferred_pass in [false, true] {
        for (index, (entry, target)) in mapped.iter_mut().enumerate() {
            if is_directory_too(&*target) != deferred_pass {
                continue;
            }
            let Some(target) = target.take() else {
                continue;
            };
            settled[index] = Some(target.and_then(|comps| {
                claim(&mut files, &mut dirs, &comps, entry.as_str())
                    .map_err(|e| anyhow!("{}: {e}", redact_and_escape(entry.as_str())))
                    .map(|()| comps)
            }));
        }
    }
    mapped
        .into_iter()
        .zip(settled)
        .filter_map(|((entry, _), target)| {
            target.map(|target| PlannedDownload {
                entry: entry.clone(),
                target,
            })
        })
        .collect()
}

/// Record `comps` as a file, or say which earlier object it collides with. `entry` is the
/// listed entry, only used to spell the way to download the directory alone.
fn claim(
    files: &mut HashSet<String>,
    dirs: &mut HashSet<String>,
    comps: &[String],
    entry: &str,
) -> Result<()> {
    let joined = comps.join("/");
    if files.contains(&joined) {
        bail!("collides with an earlier object that maps to the same local path {joined:?}");
    }
    if dirs.contains(&joined) {
        let name = escape_control(&joined);
        bail!(
            "object '{name}' and directory '{name}/' cannot both exist locally; the directory \
             takes precedence; fetch the object alone \
             (non-recursively), or use '{}/' to download only the directory",
            redact_and_escape(entry)
        );
    }
    let mut ancestor = String::new();
    for comp in &comps[..comps.len().saturating_sub(1)] {
        if !ancestor.is_empty() {
            ancestor.push('/');
        }
        ancestor.push_str(comp);
        if files.contains(&ancestor) {
            bail!(
                "local path {ancestor:?} is an earlier object's file, not a directory (file/directory collision)"
            );
        }
    }
    // Only now that nothing conflicts, record it.
    let mut ancestor = String::new();
    for comp in &comps[..comps.len().saturating_sub(1)] {
        if !ancestor.is_empty() {
            ancestor.push('/');
        }
        ancestor.push_str(comp);
        dirs.insert(ancestor.clone());
    }
    files.insert(joined);
    Ok(())
}

/// Write `bytes` to `<dest_dir>/<comps>`, creating parent directories, refusing a
/// pre-existing symlink at or above the file below `dest_dir` and (on Unix) a symlink
/// planted at the file itself. One blocking section does the checks, the directory creation
/// and the write.
pub(crate) async fn write_object(
    dest_dir: &Path,
    comps: &[String],
    bytes: bytes::Bytes,
) -> Result<PathBuf> {
    let dest_dir = dest_dir.to_path_buf();
    let comps = comps.to_vec();
    tokio::task::spawn_blocking(move || write_object_blocking(&dest_dir, &comps, &bytes))
        .await
        .map_err(|e| anyhow!("write task failed: {e}"))?
}

fn write_object_blocking(dest_dir: &Path, comps: &[String], bytes: &[u8]) -> Result<PathBuf> {
    let target = check_target(dest_dir, comps)?;
    write_checked(&target, bytes)?;
    Ok(target)
}

/// Step 1 of a write: the path `<dest_dir>/<comps>`, after refusing a pre-existing symlink
/// at or above the file below `dest_dir`. Nothing is created.
fn check_target(dest_dir: &Path, comps: &[String]) -> Result<PathBuf> {
    let mut cur = dest_dir.to_path_buf();
    let mut exists = true;
    for comp in comps {
        cur.push(comp);
        if !exists {
            continue;
        }
        match std::fs::symlink_metadata(&cur) {
            Ok(meta) if meta.file_type().is_symlink() => {
                bail!("refusing to follow the symlink {cur:?} below the destination")
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => exists = false,
            Err(e) => return Err(anyhow!("cannot inspect {cur:?}: {e}")),
        }
    }
    Ok(cur)
}

/// Step 2 of a write: create the parent directories of `target` and write `bytes` through
/// [`open_for_write`], which refuses a symlink at the final component even when it was
/// planted after [`check_target`] ran.
fn write_checked(target: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;

    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("cannot create the directory {parent:?}: {e}"))?;
    }
    let mut file = open_for_write(target)?;
    file.write_all(bytes)
        .map_err(|e| anyhow!("cannot write {target:?}: {e}"))?;
    Ok(())
}

/// Create or truncate `path` for writing. On Unix the open fails (`ELOOP`) when the final
/// component is a symlink, even one created after the caller's own check.
pub(crate) fn open_for_write(path: &Path) -> Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    options.open(path).map_err(|e| {
        #[cfg(unix)]
        if e.raw_os_error() == Some(libc::ELOOP) {
            return anyhow!("refusing to follow the symlink {path:?} below the destination");
        }
        anyhow!("cannot open {path:?} for writing: {e}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comps(prefix: &str, entry: &str) -> Vec<String> {
        relative_components(prefix, entry).expect("accepted")
    }

    /// The components `entry` gets in a plan of `prefix` (the source is canonicalised first).
    fn planned(prefix: &str, entry: &str) -> Vec<String> {
        let plans = plan_downloads(prefix, &[entry.to_string()]);
        plans[0].target.as_ref().expect("accepted").clone()
    }

    fn refused(prefix: &str, entry: &str) -> String {
        format!(
            "{:#}",
            relative_components(prefix, entry).expect_err("must be refused")
        )
    }

    #[test]
    fn keeps_the_path_relative_to_the_listing_base() {
        assert_eq!(comps("s3://b/dl/", "s3://b/dl/a/x.bin"), ["a", "x.bin"]);
        assert_eq!(
            comps("s3://b/dl", "s3://b/dl/a/b/c.bin"),
            ["a", "b", "c.bin"]
        );
        assert_eq!(
            comps("s3://b/dl/*.npz", "s3://b/dl/sub/c.npz"),
            ["sub", "c.npz"]
        );
        assert_eq!(comps("file:///d/", "file:///d/a/x"), ["a", "x"]);
    }

    #[test]
    fn a_single_exact_object_goes_to_its_name() {
        assert_eq!(
            comps("s3://b/dir/file.txt", "s3://b/dir/file.txt"),
            ["file.txt"]
        );
        assert_eq!(comps("file:///d/f.bin", "file:///d/f.bin"), ["f.bin"]);
        assert_eq!(comps("s3://b/top", "s3://b/top"), ["top"]);
    }

    #[test]
    fn host_form_prefix_maps_bucket_form_entries() {
        assert_eq!(
            comps("s3://127.0.0.1:9000/bkt/dl/", "s3://bkt/dl/sub/c.npz"),
            ["sub", "c.npz"]
        );
    }

    #[test]
    fn raw_keys_work_like_uris() {
        assert_eq!(comps("dl/", "dl/a/x"), ["a", "x"]);
        assert_eq!(comps("", "a/x"), ["a", "x"]);
    }

    #[test]
    fn traversal_and_unsafe_keys_are_refused() {
        for rel in [
            "../evil",
            "a/../../evil",
            "a/..",
            "/abs",
            "a//b",
            "./x",
            "a/./b",
            "a\\..\\..\\evil",
            "..\\evil",
            "\\abs",
            "C:\\evil",
            "C:/evil",
            "a/C:",
            "a\0b",
        ] {
            let entry = format!("s3://b/dl/{rel}");
            let msg = refused("s3://b/dl/", &entry);
            assert!(
                msg.contains("refused") || msg.contains("NUL"),
                "{rel}: {msg}"
            );
        }
        // The same through raw keys and a file:// base.
        refused("dl/", "dl/../evil");
        refused("file:///d/", "file:///d/a/../../evil");
    }

    #[test]
    fn an_entry_outside_the_base_is_refused() {
        let msg = refused("s3://b/dl/", "s3://b/other/x");
        assert!(msg.contains("not under the listing base"), "{msg}");
    }

    #[test]
    fn dir_markers_are_skipped_and_empty_relative_keys_refused() {
        let plans = plan_downloads(
            "s3://b/dl/",
            &["s3://b/dl/".into(), "s3://b/dl/sub/".into()],
        );
        assert!(plans.is_empty());
        refused("s3://b/dl/", "s3://b/dl/");
    }

    #[test]
    fn same_local_path_collides_first_wins() {
        let plans = plan_downloads(
            "s3://b/dl/",
            &[
                "s3://b/dl/a/x".into(),
                "s3://b/dl/b/x".into(),
                "s3://b/dl/a/x".into(),
            ],
        );
        assert!(plans[0].target.is_ok());
        assert!(plans[1].target.is_ok());
        let msg = format!("{:#}", plans[2].target.as_ref().expect_err("collision"));
        assert!(msg.contains("collides"), "{msg}");
    }

    #[test]
    fn file_and_directory_collide_in_either_order() {
        // The directory's contents win in either order; only the single object is refused.
        let a_first = plan_downloads("s3://b/", &["s3://b/a".into(), "s3://b/a/b".into()]);
        let msg = format!("{:#}", a_first[0].target.as_ref().expect_err("collision"));
        assert!(msg.contains("cannot both exist locally"), "{msg}");
        assert_eq!(a_first[1].target.as_ref().expect("planned"), &["a", "b"]);

        let b_first = plan_downloads("s3://b/", &["s3://b/a/b".into(), "s3://b/a".into()]);
        assert_eq!(b_first[0].target.as_ref().expect("planned"), &["a", "b"]);
        let msg = format!("{:#}", b_first[1].target.as_ref().expect_err("collision"));
        assert!(msg.contains("cannot both exist locally"), "{msg}");
    }

    #[test]
    fn a_refused_object_claims_nothing() {
        let plans = plan_downloads(
            "s3://b/",
            &[
                "s3://b/a/../x".into(),
                "s3://b/x".into(),
                "s3://b/a/y".into(),
                "s3://b/a".into(),
            ],
        );
        assert!(plans[0].target.is_err());
        assert!(
            plans[1].target.is_ok(),
            "the refused a/../x claimed nothing"
        );
        assert!(plans[2].target.is_ok());
        assert!(plans[3].target.is_err(), "file a loses to the directory a/");
    }

    #[tokio::test]
    async fn write_creates_parents_and_refuses_symlinks() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("dest");
        std::fs::create_dir_all(&dest).expect("mkdir");
        let written = write_object(
            &dest,
            &["a".into(), "b".into(), "f".into()],
            bytes::Bytes::from_static(b"hi"),
        )
        .await
        .expect("write");
        assert_eq!(written, dest.join("a/b/f"));
        assert_eq!(std::fs::read(&written).expect("read"), b"hi");

        #[cfg(unix)]
        {
            let outside = dir.path().join("outside");
            std::fs::create_dir_all(&outside).expect("mkdir");
            std::os::unix::fs::symlink(&outside, dest.join("link")).expect("symlink");
            let err = write_object(
                &dest,
                &["link".into(), "evil".into()],
                bytes::Bytes::from_static(b"x"),
            )
            .await
            .expect_err("symlinked directory refused");
            assert!(format!("{err:#}").contains("symlink"), "{err:#}");
            assert!(!outside.join("evil").exists());

            let victim = dir.path().join("victim");
            std::fs::write(&victim, b"orig").expect("write");
            std::os::unix::fs::symlink(&victim, dest.join("flink")).expect("symlink");
            write_object(&dest, &["flink".into()], bytes::Bytes::from_static(b"x"))
                .await
                .expect_err("symlinked file refused");
            assert_eq!(std::fs::read(&victim).expect("read"), b"orig");
        }
    }

    // ---- review round: #284 follow-ups ----

    #[test]
    fn gcs_and_azure_https_sources_map_against_canonical_entries() {
        // The stores return `gs://bucket/key` and `az://account/container/key`, whatever
        // spelling the source used.
        assert_eq!(
            planned("gcs://bkt/dl/", "gs://bkt/dl/sub/c.npz"),
            ["sub", "c.npz"]
        );
        assert_eq!(planned("gcs://bkt/dl", "gs://bkt/dl/a/x"), ["a", "x"]);
        assert_eq!(planned("gs://bkt/dl/", "gs://bkt/dl/a/x"), ["a", "x"]);
        assert_eq!(
            planned("gcs://bkt/dir/f.bin", "gs://bkt/dir/f.bin"),
            ["f.bin"]
        );
        assert_eq!(planned("gcs://bkt", "gs://bkt/top/x"), ["top", "x"]);
        assert_eq!(planned("gcs://bkt/", "gs://bkt/top/x"), ["top", "x"]);
        assert_eq!(
            planned(
                "https://acct.blob.core.windows.net/ctr/dl/",
                "az://acct/ctr/dl/sub/c.npz"
            ),
            ["sub", "c.npz"]
        );
        assert_eq!(
            planned(
                "http://acct.blob.core.windows.net/ctr/dl",
                "az://acct/ctr/dl/a/x"
            ),
            ["a", "x"]
        );
        assert_eq!(
            planned("az://acct/ctr/dl/", "az://acct/ctr/dl/a/x"),
            ["a", "x"]
        );
        assert_eq!(
            planned("az://acct/ctr", "az://acct/ctr/top/x"),
            ["top", "x"]
        );
        // A plan (what the downloaders use) agrees.
        let plans = plan_downloads(
            "https://acct.blob.core.windows.net/ctr/dl/",
            &["az://acct/ctr/dl/a/x".into()],
        );
        assert_eq!(plans[0].target.as_ref().expect("planned"), &["a", "x"]);
        let plans = plan_downloads("gcs://bkt/dl/", &["gs://bkt/dl/a/x".into()]);
        assert_eq!(plans[0].target.as_ref().expect("planned"), &["a", "x"]);
    }

    #[test]
    fn keys_with_a_colon_are_ordinary_only_an_exact_drive_designator_is_refused() {
        assert_eq!(comps("s3://b/dl/", "s3://b/dl/x:y"), ["x:y"]);
        assert_eq!(comps("s3://b/dl/", "s3://b/dl/v:1/x"), ["v:1", "x"]);
        assert_eq!(comps("s3://b/dl/", "s3://b/dl/a:b:c"), ["a:b:c"]);
        assert_eq!(comps("s3://b/x:y", "s3://b/x:y"), ["x:y"]);
        assert_eq!(comps("s3://b/dl/", "s3://b/dl/1:x"), ["1:x"]);
        for rel in ["C:", "c:", "a/C:", "C:/evil", "C:\\evil"] {
            let msg = refused("s3://b/dl/", &format!("s3://b/dl/{rel}"));
            assert!(msg.contains("drive"), "{rel}: {msg}");
        }
    }

    #[test]
    fn one_base_for_a_listing_with_an_exact_object_that_is_also_a_directory() {
        // `s3://b/run.1` selects the object `run.1` AND the directory `run.1/`. One base for
        // the whole listing (the parent): `run.1` -> `run.1`, `run.1/child` -> `run.1/child`,
        // which collides with the file `run.1` and is reported, never relocated to `child`.
        let plans = plan_downloads(
            "s3://b/run.1",
            &["s3://b/run.1".into(), "s3://b/run.1/child".into()],
        );
        let msg = format!(
            "{:#}",
            plans[0].target.as_ref().expect_err("the object loses")
        );
        assert!(msg.contains("cannot both exist locally"), "{msg}");
        assert!(msg.contains("'run.1'"), "{msg}");
        assert_eq!(
            plans[1].target.as_ref().expect("planned"),
            &["run.1", "child"]
        );

        // Without the object itself the directory is the base, as for `dl` and `dl/`.
        let plans = plan_downloads(
            "s3://b/run.1",
            &["s3://b/run.1/child".into(), "s3://b/run.1/z/w".into()],
        );
        assert_eq!(plans[0].target.as_ref().expect("planned"), &["child"]);
        assert_eq!(plans[1].target.as_ref().expect("planned"), &["z", "w"]);

        // The object alone is unchanged.
        let plans = plan_downloads("s3://b/data/run.1", &["s3://b/data/run.1".into()]);
        assert_eq!(plans[0].target.as_ref().expect("planned"), &["run.1"]);

        // Raw keys (the s3_copy callers) agree.
        let plans = plan_downloads("run.1", &["run.1".into(), "run.1/child".into()]);
        assert!(plans[0].target.is_err());
        assert_eq!(
            plans[1].target.as_ref().expect("planned"),
            &["run.1", "child"]
        );
    }

    #[test]
    fn an_azure_account_is_canonicalised_like_the_store_does() {
        // Default endpoint: the store lower-cases the account, so its entries are
        // `az://myacct/...` whatever the source spelled; a custom endpoint keeps the name.
        let lower = |a: &str| a.to_ascii_lowercase();
        let keep = |a: &str| a.to_string();
        let entries = ["az://myacct/ctr/dl/a/x".to_string()];
        for src in [
            "az://MyAcct/ctr/dl/",
            "https://MyAcct.blob.core.windows.net/ctr/dl/",
        ] {
            let plans = plan_downloads_with(src, &entries, &lower);
            assert_eq!(
                plans[0].target.as_ref().expect("planned"),
                &["a", "x"],
                "{src}"
            );
            let plans = plan_downloads_with(src, &entries, &keep);
            assert!(
                plans[0].target.is_err(),
                "{src}: a kept name does not match"
            );
        }
        let kept = ["az://MyAcct/ctr/dl/a/x".to_string()];
        let plans = plan_downloads_with("az://MyAcct/ctr/dl/", &kept, &keep);
        assert_eq!(plans[0].target.as_ref().expect("planned"), &["a", "x"]);
    }

    #[test]
    fn the_exact_object_base_works_for_host_form_and_bucket_only_sources() {
        let plans = plan_downloads(
            "s3://127.0.0.1:9000/bkt/run.1",
            &["s3://bkt/run.1".into(), "s3://bkt/run.1/child".into()],
        );
        assert!(plans[0].target.is_err(), "the object run.1 loses");
        assert_eq!(
            plans[1].target.as_ref().expect("planned"),
            &["run.1", "child"]
        );

        let plans = plan_downloads(
            "s3://127.0.0.1:9000/bkt/run.1",
            &["s3://bkt/run.1/child".into()],
        );
        assert_eq!(plans[0].target.as_ref().expect("planned"), &["child"]);

        // A bucket-only source has no key that could be an object.
        for src in ["s3://b", "s3://b/", "gcs://b", "gs://b/"] {
            let canonical = if src.starts_with("s3") {
                "s3://b"
            } else {
                "gs://b"
            };
            let plans = plan_downloads(src, &[format!("{canonical}/top/x")]);
            assert_eq!(
                plans[0].target.as_ref().expect("planned"),
                &["top", "x"],
                "{src}"
            );
        }
    }

    #[test]
    fn messages_do_not_carry_credentials() {
        assert_eq!(
            redact_and_escape("https://acct.blob.core.windows.net/c/k?sv=1&sig=SECRET"),
            "https://acct.blob.core.windows.net/c/k"
        );
        assert_eq!(
            redact_and_escape("https://user:pw@host/c/k#frag"),
            "https://host/c/k"
        );
        assert_eq!(
            redact_and_escape("s3://b/k?versionId=3"),
            "s3://b/k?versionId=3"
        );
        assert_eq!(
            redact_and_escape("s3://b/k?X-Amz-Signature=abc"),
            "s3://b/k"
        );
        assert_eq!(redact_and_escape("dl/a@b"), "dl/a@b");
        let msg = refused(
            "https://acct.blob.core.windows.net/ctr/dl/?sig=SECRET",
            "az://acct/ctr/other/x",
        );
        assert!(!msg.contains("SECRET"), "{msg}");
    }

    #[cfg(unix)]
    #[test]
    fn the_final_component_is_opened_without_following_a_symlink() {
        // What a symlink planted between the check and the write would hit.
        let dir = tempfile::tempdir().expect("tempdir");
        let victim = dir.path().join("victim");
        std::fs::write(&victim, b"orig").expect("write");
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&victim, &link).expect("symlink");
        let err = open_for_write(&link).expect_err("a symlink is refused");
        assert!(format!("{err:#}").contains("symlink"), "{err:#}");
        assert_eq!(std::fs::read(&victim).expect("read"), b"orig");

        // A dangling symlink would be created through by a plain create+open.
        let dangling = dir.path().join("dangling");
        std::os::unix::fs::symlink(dir.path().join("target"), &dangling).expect("symlink");
        open_for_write(&dangling).expect_err("a dangling symlink is refused");
        assert!(!dir.path().join("target").exists());

        // An ordinary file is created / truncated.
        let plain = dir.path().join("plain");
        std::fs::write(&plain, b"long old content").expect("write");
        drop(open_for_write(&plain).expect("plain file"));
        assert_eq!(std::fs::read(&plain).expect("read"), b"");
    }

    // ---- review round 2: directory wins, one redactor, control characters, O_NOFOLLOW ----

    #[test]
    fn an_object_is_downloaded_when_every_child_of_its_directory_is_refused() {
        let plans = plan_downloads(
            "s3://b/dl",
            &["s3://b/dl".into(), "s3://b/dl/../evil".into()],
        );
        assert_eq!(plans[0].target.as_ref().expect("planned"), &["dl"]);
        assert!(plans[1].target.is_err());
    }

    #[test]
    fn in_a_chain_every_object_with_a_valid_child_is_refused() {
        let plans = plan_downloads(
            "s3://b/p",
            &[
                "s3://b/p".into(),
                "s3://b/p/q".into(),
                "s3://b/p/q/r".into(),
            ],
        );
        assert!(plans[0].target.is_err(), "p has the valid child p/q/r");
        assert!(plans[1].target.is_err(), "p/q has the valid child p/q/r");
        assert_eq!(plans[2].target.as_ref().expect("planned"), &["p", "q", "r"]);
    }

    #[test]
    fn a_folder_marker_object_loses_to_its_directory() {
        // `s3://b/dl` selects the object `dl` (a zero-byte folder marker) and `dl/...`.
        let plans = plan_downloads(
            "s3://b/dl",
            &[
                "s3://b/dl".into(),
                "s3://b/dl/a".into(),
                "s3://b/dl/b/c".into(),
            ],
        );
        let msg = format!(
            "{:#}",
            plans[0].target.as_ref().expect_err("the object loses")
        );
        assert!(msg.contains("object 'dl' and directory 'dl/'"), "{msg}");
        assert!(msg.contains("cannot both exist locally"), "{msg}");
        assert!(
            msg.contains("'s3://b/dl/'"),
            "names the trailing-slash alternative: {msg}"
        );
        assert!(
            msg.contains("fetch the object alone (non-recursively)"),
            "{msg}"
        );
        assert_eq!(plans[1].target.as_ref().expect("planned"), &["dl", "a"]);
        assert_eq!(
            plans[2].target.as_ref().expect("planned"),
            &["dl", "b", "c"]
        );

        // The order of the listing does not matter.
        let plans = plan_downloads("s3://b/dl", &["s3://b/dl/a".into(), "s3://b/dl".into()]);
        assert_eq!(plans[0].target.as_ref().expect("planned"), &["dl", "a"]);
        assert!(plans[1].target.is_err());

        // A glob that selects both behaves the same.
        let plans = plan_downloads("s3://b/dl*", &["s3://b/dl".into(), "s3://b/dl/a".into()]);
        assert!(plans[0].target.is_err());
        assert_eq!(plans[1].target.as_ref().expect("planned"), &["dl", "a"]);

        // Nested: both `a` and `a/b` are objects with something below them.
        let plans = plan_downloads(
            "s3://b/",
            &[
                "s3://b/a".into(),
                "s3://b/a/b".into(),
                "s3://b/a/b/c".into(),
            ],
        );
        assert!(plans[0].target.is_err());
        assert!(plans[1].target.is_err());
        assert_eq!(plans[2].target.as_ref().expect("planned"), &["a", "b", "c"]);
    }

    #[test]
    fn a_directory_whose_children_are_all_refused_does_not_hide_the_object() {
        // `dl/../x` is refused, so nothing is written under `dl/` and the object `dl` is fine.
        let plans = plan_downloads("s3://b/dl", &["s3://b/dl".into(), "s3://b/dl/../x".into()]);
        assert_eq!(plans[0].target.as_ref().expect("planned"), &["dl"]);
        assert!(plans[1].target.is_err());
    }

    #[test]
    fn control_characters_in_a_key_never_reach_a_message() {
        let hostile = "x\ny\u{1b}[31mFORGED\r\u{85}\u{2028}";
        let entry = format!("s3://b/dl/../{hostile}");
        let msg = refused("s3://b/dl/", &entry);
        assert!(msg.contains("refused"), "{msg}");
        let outside = refused("s3://b/dl/", &format!("s3://b/other/{hostile}"));
        assert!(outside.contains("not under the listing base"), "{outside}");
        let dup = format!("s3://b/dl/{hostile}");
        let plans = plan_downloads("s3://b/dl/", &[dup.clone(), dup.clone()]);
        let collision = format!("{:#}", plans[1].target.as_ref().expect_err("collision"));
        for text in [&msg, &outside, &collision] {
            for forbidden in ['\n', '\r', '\u{1b}', '\u{85}', '\u{2028}'] {
                assert!(!text.contains(forbidden), "{forbidden:?} in {text:?}");
            }
            assert!(
                text.contains("FORGED"),
                "the key is still identifiable: {text:?}"
            );
        }
        assert!(msg.contains("\\n") && msg.contains("\\u{1b}"), "{msg}");
    }

    #[test]
    fn the_directory_wins_message_escapes_the_key_too() {
        let hostile = "x\ny\u{1b}[31mFORGED";
        let marker = plan_downloads(
            "s3://b/dl",
            &[
                format!("s3://b/dl{hostile}"),
                format!("s3://b/dl{hostile}/a"),
            ],
        );
        let msg = format!("{:#}", marker[0].target.as_ref().expect_err("object loses"));
        assert!(!msg.contains('\n') && !msg.contains('\u{1b}'), "{msg:?}");
        assert!(msg.contains("FORGED") && msg.contains("\\n"), "{msg}");
    }
    #[test]
    fn download_messages_use_the_shared_redactor() {
        // A password holding a `/`, and a SigV2 `Signature=` marker, passed through the
        // download code's own redactor; the one shared redactor removes both.
        assert_eq!(
            redact_and_escape("https://user:p/w@host/x"),
            "https://host/x"
        );
        assert_eq!(redact_and_escape("gs://b/o?Signature=SECRET"), "gs://b/o");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_planted_after_the_check_is_refused_by_the_final_open() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("dest");
        std::fs::create_dir_all(&dest).expect("mkdir");
        let victim = dir.path().join("victim");
        std::fs::write(&victim, b"orig").expect("write");

        // Step 1 passes: nothing is at the target yet.
        let target = check_target(&dest, &["sub".into(), "f".into()]).expect("check passes");
        // Another local process plants a symlink at the final component.
        std::fs::create_dir_all(dest.join("sub")).expect("mkdir");
        std::os::unix::fs::symlink(&victim, &target).expect("symlink");
        // Step 2 must refuse it; a plain create would write through it.
        let err = write_checked(&target, b"EVIL").expect_err("the final open refuses a symlink");
        assert!(format!("{err:#}").contains("symlink"), "{err:#}");
        assert_eq!(std::fs::read(&victim).expect("read"), b"orig");

        // The same through a dangling link: nothing is created at its destination.
        let missing = dir.path().join("created-by-evil");
        let target = check_target(&dest, &["d".into()]).expect("check passes");
        std::os::unix::fs::symlink(&missing, &target).expect("symlink");
        write_checked(&target, b"EVIL").expect_err("a dangling symlink is refused");
        assert!(!missing.exists());
    }
}
