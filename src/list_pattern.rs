// src/list_pattern.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! The explicit, client-side regex filter for listings (issue #213).
//!
//! A storage URI given to a listing is a LITERAL key prefix: `s3://b/data/run.1`
//! lists every key that starts with `data/run.1` (so `data/run.1.old` too, but
//! never `data/run_1` or `data/prerun.1`) and `.`, `(`, `[`, `+`, `$`, `?` in it
//! are ordinary characters. A regular expression is applied only when the caller
//! passes one explicitly: Python `list(uri, pattern=...)`, `s3-cli ls --pattern`
//! or [`crate::s3_copy::download_objects_matching`]. All of them use this one
//! type, so they agree on what the pattern is matched against.
//!
//! # What the pattern sees
//!
//! The pattern is matched (unanchored, like [`Regex::is_match`]) against the entry
//! RELATIVE TO THE LISTING BASE, never against the whole URI. The base of a
//! listing prefix `P` is:
//!
//! 1. `P` itself when it ends with `/` (`s3://b/dl/` -> `sub/c.npz`);
//! 2. otherwise `P/` for an entry that lies under the directory `P`
//!    (`s3://b/dl` and `s3://b/dl/sub/c.npz` -> `sub/c.npz`), and
//! 3. otherwise the parent directory of `P` -- everything up to and including its
//!    last `/` -- for the entries that merely START WITH `P`
//!    (`s3://b/data/run.1` and `s3://b/data/run.1.old` -> `run.1.old`).
//!
//! An entry that does not start with the base is matched as it is.

use anyhow::{Context, Result};
use regex::Regex;

/// A compiled listing filter. Compile it BEFORE listing, so an invalid pattern
/// fails without a single request being sent.
#[derive(Debug, Clone)]
pub struct ListPattern {
    regex: Regex,
}

impl ListPattern {
    /// Compile `pattern` (Rust `regex` syntax).
    pub fn new(pattern: &str) -> Result<Self> {
        let regex =
            Regex::new(pattern).with_context(|| format!("Invalid regex pattern: '{pattern}'"))?;
        Ok(Self { regex })
    }

    /// Whether `entry` (a URI or key returned by a listing of `listing_prefix`)
    /// is selected: the pattern is matched against the part of `entry` after the
    /// listing base (see the [module docs](self)).
    pub fn is_match(&self, listing_prefix: &str, entry: &str) -> bool {
        self.regex
            .is_match(relative_to_listing_base(listing_prefix, entry))
    }

    /// Keep only the entries of a listing of `listing_prefix` that match.
    pub fn retain(&self, listing_prefix: &str, entries: &mut Vec<String>) {
        entries.retain(|entry| self.is_match(listing_prefix, entry));
    }
}

/// `entry` relative to the base of a listing of `listing_prefix` (see the
/// [module docs](self) for the rule).
pub fn relative_to_listing_base<'a>(listing_prefix: &str, entry: &'a str) -> &'a str {
    if listing_prefix.ends_with('/') {
        return entry.strip_prefix(listing_prefix).unwrap_or(entry);
    }
    if let Some(inside) = entry
        .strip_prefix(listing_prefix)
        .and_then(|rest| rest.strip_prefix('/'))
    {
        return inside;
    }
    match listing_prefix.rfind('/') {
        Some(slash) => entry
            .strip_prefix(&listing_prefix[..=slash])
            .unwrap_or(entry),
        None => entry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_of_a_directory_prefix_is_the_prefix() {
        assert_eq!(
            relative_to_listing_base("s3://b/dl/", "s3://b/dl/sub/c.npz"),
            "sub/c.npz"
        );
        assert_eq!(relative_to_listing_base("s3://b/dl/", "s3://b/dl/"), "");
    }

    #[test]
    fn base_of_a_prefix_without_slash_is_the_directory_for_entries_under_it() {
        assert_eq!(
            relative_to_listing_base("s3://b/dl", "s3://b/dl/sub/c.npz"),
            "sub/c.npz"
        );
        assert_eq!(relative_to_listing_base("s3://b/dl", "s3://b/dl/"), "");
    }

    #[test]
    fn base_of_a_prefix_without_slash_is_the_parent_for_entries_that_merely_start_with_it() {
        assert_eq!(
            relative_to_listing_base("s3://b/data/run.1", "s3://b/data/run.1.old"),
            "run.1.old"
        );
        assert_eq!(
            relative_to_listing_base("s3://b/data/run.1", "s3://b/data/run.1"),
            "run.1"
        );
        assert_eq!(
            relative_to_listing_base("s3://b/foo", "s3://b/foobar"),
            "foobar"
        );
    }

    #[test]
    fn raw_keys_follow_the_same_rule() {
        assert_eq!(relative_to_listing_base("dl", "dl/a.npz"), "a.npz");
        assert_eq!(relative_to_listing_base("dl/", "dl/a.npz"), "a.npz");
        assert_eq!(relative_to_listing_base("run.1", "run.1.old"), "run.1.old");
        assert_eq!(relative_to_listing_base("", "dl/a.npz"), "dl/a.npz");
    }

    #[test]
    fn an_entry_outside_the_base_is_matched_whole() {
        assert_eq!(
            relative_to_listing_base("s3://b/dl/", "s3://b/other/x"),
            "s3://b/other/x"
        );
    }

    #[test]
    fn pattern_matches_the_relative_path_unanchored() {
        let p = ListPattern::new(r"\.npz$").unwrap();
        assert!(p.is_match("s3://b/dl/", "s3://b/dl/sub/c.npz"));
        assert!(!p.is_match("s3://b/dl/", "s3://b/dl/b.txt"));
        let anchored = ListPattern::new("^sub/").unwrap();
        assert!(anchored.is_match("s3://b/dl/", "s3://b/dl/sub/c.npz"));
        assert!(!anchored.is_match("s3://b/dl/", "s3://b/dl/a.npz"));
    }

    #[test]
    fn retain_keeps_only_matches() {
        let p = ListPattern::new("^a").unwrap();
        let mut entries = vec![
            "s3://b/dl/a.npz".to_string(),
            "s3://b/dl/b.txt".to_string(),
            "s3://b/dl/sub/c.npz".to_string(),
        ];
        p.retain("s3://b/dl", &mut entries);
        assert_eq!(entries, vec!["s3://b/dl/a.npz"]);
    }

    #[test]
    fn invalid_pattern_is_an_error_naming_the_pattern() {
        let err = ListPattern::new("(").unwrap_err();
        assert!(
            format!("{err:#}").contains("Invalid regex pattern: '('"),
            "{err:#}"
        );
    }
}
