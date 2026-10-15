// src/s3_selection.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Which objects an `s3://` URI selects (issue #213).
//!
//! The key part `K` of an `s3://bucket/K` URI names a SELECTION, not a bare string
//! prefix. Every listing-style operation on S3 -- `s3-cli ls` / `download` / `get` /
//! `delete -r`, Python `list` / `download` / `delete(recursive=True)`,
//! `ObjectStore::list` / `list_stream` / `list_objects_with_prefix`,
//! `S3BytesDataset::from_prefix` -- goes through [`select_key_stream`], so they all
//! agree:
//!
//! | `K` | selects |
//! |---|---|
//! | empty | the whole bucket |
//! | ends in `/`, no `*` | every key under `K` |
//! | no `*`, not ending in `/` | the object whose key is exactly `K`, and (when the operation descends) every key under `K/` -- never `K.bak`, `K2`, `K.d/` |
//! | contains `*` | a glob: an object is selected iff its FULL key matches `K` |
//! | contains `*` and ends in `/` | an error (not defined; see [`S3Selection::parse`]) |
//!
//! In a glob `*` matches any run of characters EXCEPT `/`, and `**` matches any run
//! including `/`. Every other character, `.` `(` `[` `+` `$` `?` included, is
//! literal. The S3 request prefix is the text before the first `*`, so the server does
//! the narrowing; the glob is then applied to the full key on the client.
//!
//! An explicit regular expression (`--pattern` / `pattern=`, [`crate::list_pattern`])
//! is applied AFTER this selection.
//!
//! # Requests
//!
//! * `K` empty or ending in `/`: one `ListObjectsV2` per page of the listing.
//! * `K` exact, descending (recursive listing, `download -r`, `delete -r`): ONE probe
//!   (`prefix=K`, `max-keys=1`; `K` sorts before every other key that starts with `K`,
//!   so it is the first key returned when the object exists) plus the pages of the
//!   listing of `K/`. No sibling (`K.0`, `K.1`, ...) is ever paged through.
//! * `K` exact, not descending (non-recursive `ls`): the same probe, plus one more
//!   probe (`prefix=K/`, `max-keys=1`) that decides whether the directory entry `K/`
//!   is shown. Two requests in total.
//! * glob: the pages of the listing of the request prefix; when the glob cannot cross
//!   a `/` after that prefix (no `**`, no `/` after the prefix) the listing uses the
//!   `/` delimiter so the subtree below the prefix is not read.
//!
//! No `HEAD` is ever sent.
//!
//! This applies to `s3://` only; the other backends are tracked in #220.

use anyhow::{bail, Context, Result};
use futures::Stream;
use regex::Regex;
use std::pin::Pin;

use crate::s3_utils::format_sdk_error;

/// A stream of raw object keys, plus (non-recursive listings) directory entries that
/// end in `/`.
pub(crate) type KeyStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

/// What the key part of an `s3://` URI selects. See the [module docs](self).
#[derive(Debug, Clone)]
pub enum S3Selection {
    /// The bucket root (empty key), or a key ending in `/` with no `*`: everything
    /// under it. Holds the key (S3 request prefix).
    Dir(String),
    /// A key with no `*` that does not end in `/`: the object with exactly this key
    /// and, when the operation descends, everything under `key/`.
    Exact(String),
    /// A key containing `*`: an explicit glob over the full key.
    Glob(GlobSelection),
}

/// A compiled glob over full keys (see the [module docs](self)).
#[derive(Debug, Clone)]
pub struct GlobSelection {
    glob: String,
    request_prefix: String,
    matcher: Regex,
    one_level: bool,
}

impl S3Selection {
    /// Classify `key`, the key part of an `s3://` URI.
    ///
    /// A key that contains `*` AND ends in `/` (`data/*/`) is refused: the glob rule
    /// ("selected iff the full key matches") would match only objects whose key ends
    /// in `/`, while the directory rule would read the `*` literally, and neither is
    /// clearly what was meant. Nothing is listed; drop the trailing `/`, or use `**`
    /// to select everything below a directory.
    pub fn parse(key: &str) -> Result<Self> {
        if !key.contains('*') {
            if key.is_empty() || key.ends_with('/') {
                return Ok(Self::Dir(key.to_string()));
            }
            return Ok(Self::Exact(key.to_string()));
        }
        if key.ends_with('/') {
            bail!(
                "the key '{key}' contains a wildcard and ends in '/'; that combination is not \
                 defined (nothing was listed). Drop the trailing '/', or end the key with '**' \
                 to select everything below a directory"
            );
        }
        Ok(Self::Glob(GlobSelection::new(key)?))
    }

    /// The S3 `prefix` a listing for this selection starts from, for the pattern
    /// base only: the text before the first `*`, or the whole key.
    pub fn request_prefix(&self) -> &str {
        match self {
            Self::Dir(k) | Self::Exact(k) => k,
            Self::Glob(g) => &g.request_prefix,
        }
    }

    /// Whether `key` (a raw object key) is selected by this selection, ignoring
    /// whether the operation descends: a `Dir` selects what starts with it, an
    /// `Exact` the key itself and everything under `key/`, a `Glob` the full-key
    /// matches.
    pub fn selects(&self, key: &str) -> bool {
        match self {
            Self::Dir(k) => key.starts_with(k.as_str()),
            Self::Exact(k) => {
                key == k
                    || key
                        .strip_prefix(k.as_str())
                        .is_some_and(|rest| rest.starts_with('/'))
            }
            Self::Glob(g) => g.matcher.is_match(key),
        }
    }
}

impl GlobSelection {
    fn new(glob: &str) -> Result<Self> {
        let first_star = glob.find('*').unwrap_or(glob.len());
        let request_prefix = glob[..first_star].to_string();
        let one_level = !glob.contains("**") && !glob[first_star..].contains('/');
        Ok(Self {
            glob: glob.to_string(),
            request_prefix,
            matcher: glob_to_regex(glob)?,
            one_level,
        })
    }

    /// The glob text.
    pub fn glob(&self) -> &str {
        &self.glob
    }
}

/// Compile `glob` to an anchored regex: `*` = `[^/]*`, a run of two or more `*` =
/// anything (`/` included), every other character literal.
fn glob_to_regex(glob: &str) -> Result<Regex> {
    let mut re = String::with_capacity(glob.len() * 2 + 8);
    re.push_str("^(?:");
    let mut literal = String::new();
    let mut chars = glob.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '*' {
            literal.push(c);
            continue;
        }
        let mut stars = 1usize;
        while chars.peek() == Some(&'*') {
            chars.next();
            stars += 1;
        }
        re.push_str(&regex::escape(&literal));
        literal.clear();
        re.push_str(if stars >= 2 { "(?s:.*)" } else { "[^/]*" });
    }
    re.push_str(&regex::escape(&literal));
    re.push_str(")$");
    Regex::new(&re).with_context(|| format!("cannot use '{glob}' as a wildcard key"))
}

/// One entry of a raw listing page.
enum Entry {
    Object(String),
    Dir(String),
}

/// The pages of ONE `ListObjectsV2` listing of `prefix`: objects, and with
/// `delimiter` the common prefixes (directories) too.
///
/// With `strict`, a truncated page that carries no continuation token is an error
/// (a silently short listing must never drive a delete); otherwise the listing just
/// ends there.
fn pages(
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
    delimiter: bool,
    strict: bool,
) -> Pin<Box<dyn Stream<Item = Result<Entry>> + Send>> {
    Box::pin(async_stream::stream! {
        let mut cont: Option<String> = None;
        loop {
            let mut req = client
                .list_objects_v2()
                .bucket(bucket.as_str())
                .prefix(prefix.as_str());
            if delimiter {
                req = req.delimiter("/");
            }
            if let Some(ref token) = cont {
                req = req.continuation_token(token.as_str());
            }
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(anyhow::anyhow!(
                        "list_objects_v2 failed for bucket '{}' prefix '{}': {}",
                        bucket,
                        prefix,
                        format_sdk_error(&e)
                    ));
                    return;
                }
            };
            for obj in resp.contents() {
                if let Some(key) = obj.key() {
                    yield Ok(Entry::Object(key.to_string()));
                }
            }
            if delimiter {
                for cp in resp.common_prefixes() {
                    if let Some(p) = cp.prefix() {
                        yield Ok(Entry::Dir(p.to_string()));
                    }
                }
            }
            match resp.next_continuation_token() {
                Some(next) => cont = Some(next.to_string()),
                None if strict && resp.is_truncated().unwrap_or(false) => {
                    yield Err(anyhow::anyhow!(
                        "list_objects_v2 for bucket '{}' prefix '{}' was truncated but gave \
                         no continuation token; the listing is incomplete",
                        bucket,
                        prefix
                    ));
                    return;
                }
                None => return,
            }
        }
    })
}

/// The keys of ONE `ListObjectsV2` request with `max-keys=1` for `prefix`.
///
/// A real server returns at most one key (the first in key order); the caller must
/// not rely on that, only on "the first key in order is in the answer if it exists".
async fn probe(client: &aws_sdk_s3::Client, bucket: &str, prefix: &str) -> Result<Vec<String>> {
    let resp = client
        .list_objects_v2()
        .bucket(bucket)
        .prefix(prefix)
        .max_keys(1)
        .send()
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "list_objects_v2 failed for bucket '{}' prefix '{}': {}",
                bucket,
                prefix,
                format_sdk_error(&e)
            )
        })?;
    Ok(resp
        .contents()
        .iter()
        .filter_map(|o| o.key().map(str::to_string))
        .collect())
}

/// Stream what the key part `key` of `s3://bucket/key` selects (see the
/// [module docs](self)): raw object keys, plus, when `recursive` is false, the
/// directory entries (keys ending in `/`) the non-recursive listing shows.
///
/// `strict` turns a truncated page without a continuation token into an error; use it
/// for a listing that drives a delete. An unusable `key` (see [`S3Selection::parse`])
/// is the stream's first and only item, an `Err`, before any request is sent.
pub(crate) fn select_key_stream(
    client: aws_sdk_s3::Client,
    bucket: String,
    key: &str,
    recursive: bool,
    strict: bool,
) -> KeyStream {
    let selection = match S3Selection::parse(key) {
        Ok(s) => s,
        Err(e) => return Box::pin(futures::stream::once(async move { Err(e) })),
    };
    Box::pin(async_stream::stream! {
        use futures::StreamExt;
        match selection {
            S3Selection::Dir(prefix) => {
                let mut s = pages(client, bucket, prefix, !recursive, strict);
                while let Some(item) = s.next().await {
                    match item {
                        Ok(Entry::Object(k)) | Ok(Entry::Dir(k)) => yield Ok(k),
                        Err(e) => { yield Err(e); return; }
                    }
                }
            }
            S3Selection::Exact(exact) => {
                match probe(&client, &bucket, &exact).await {
                    Ok(found) => {
                        if found.contains(&exact) {
                            yield Ok(exact.clone());
                        }
                    }
                    Err(e) => { yield Err(e); return; }
                }
                let children = format!("{exact}/");
                if recursive {
                    let mut s = pages(client, bucket, children, false, strict);
                    while let Some(item) = s.next().await {
                        match item {
                            Ok(Entry::Object(k)) | Ok(Entry::Dir(k)) => yield Ok(k),
                            Err(e) => { yield Err(e); return; }
                        }
                    }
                } else {
                    match probe(&client, &bucket, &children).await {
                        Ok(found) => {
                            if !found.is_empty() {
                                yield Ok(children);
                            }
                        }
                        Err(e) => { yield Err(e); }
                    }
                }
            }
            S3Selection::Glob(glob) => {
                let mut s = pages(
                    client,
                    bucket,
                    glob.request_prefix.clone(),
                    glob.one_level,
                    strict,
                );
                while let Some(item) = s.next().await {
                    match item {
                        Ok(Entry::Object(k)) => {
                            if glob.matcher.is_match(&k) {
                                yield Ok(k);
                            }
                        }
                        // A directory entry is not an object; a glob selects objects.
                        Ok(Entry::Dir(_)) => {}
                        Err(e) => { yield Err(e); return; }
                    }
                }
            }
        }
    })
}

/// Stream the keys under the LITERAL prefix `prefix` (no wildcard reading, no exact
/// rule): what `ObjectStore::delete_prefix` needs, whose prefix always ends in `/`.
pub(crate) fn literal_prefix_key_stream(
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
) -> KeyStream {
    use futures::StreamExt;
    Box::pin(
        pages(client, bucket, prefix, false, true).map(|item| match item {
            Ok(Entry::Object(k)) | Ok(Entry::Dir(k)) => Ok(k),
            Err(e) => Err(e),
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glob(g: &str) -> Regex {
        glob_to_regex(g).expect("glob compiles")
    }

    #[test]
    fn classification() {
        assert!(matches!(S3Selection::parse("").unwrap(), S3Selection::Dir(k) if k.is_empty()));
        assert!(matches!(
            S3Selection::parse("dl/").unwrap(),
            S3Selection::Dir(_)
        ));
        assert!(matches!(
            S3Selection::parse("file.txt").unwrap(),
            S3Selection::Exact(_)
        ));
        assert!(matches!(
            S3Selection::parse("a/b.c").unwrap(),
            S3Selection::Exact(_)
        ));
        assert!(matches!(
            S3Selection::parse("file.txt*").unwrap(),
            S3Selection::Glob(_)
        ));
        assert!(matches!(
            S3Selection::parse("*").unwrap(),
            S3Selection::Glob(_)
        ));
        // `?` and the regex characters are ordinary characters.
        assert!(matches!(
            S3Selection::parse("a?b(c)[d]+$").unwrap(),
            S3Selection::Exact(_)
        ));
    }

    #[test]
    fn wildcard_ending_in_slash_is_refused() {
        let err = S3Selection::parse("data/*/").unwrap_err();
        assert!(format!("{err}").contains("not defined"), "{err}");
    }

    #[test]
    fn request_prefix_is_the_text_before_the_first_star() {
        for (k, p) in [
            ("file.txt*", "file.txt"),
            ("data/**.npz", "data/"),
            ("data/a*b*c", "data/a"),
            ("*", ""),
        ] {
            assert_eq!(S3Selection::parse(k).unwrap().request_prefix(), p, "{k}");
        }
    }

    #[test]
    fn single_star_does_not_cross_a_slash() {
        let re = glob("data/*.npz");
        assert!(re.is_match("data/a.npz"));
        assert!(re.is_match("data/.npz"));
        assert!(!re.is_match("data/sub/a.npz"));
        assert!(!re.is_match("data/a.npz.bak"));
        assert!(!re.is_match("xdata/a.npz"));
    }

    #[test]
    fn double_star_crosses_a_slash() {
        let re = glob("data/**.npz");
        assert!(re.is_match("data/a.npz"));
        assert!(re.is_match("data/sub/deep/a.npz"));
        assert!(!re.is_match("data/a.txt"));
        // a run of three stars is still "anything"
        assert!(glob("data/***").is_match("data/x/y"));
    }

    #[test]
    fn other_characters_are_literal() {
        let re = glob("run.1*");
        assert!(re.is_match("run.1"));
        assert!(re.is_match("run.1.old"));
        assert!(!re.is_match("run_1"));
        assert!(!re.is_match("prerun.1"));
        let re = glob("a(b)[c]+$?*");
        assert!(re.is_match("a(b)[c]+$?zzz"));
        assert!(!re.is_match("abbc"));
        // a key with a newline in it
        assert!(glob("a*").is_match("a\nb"));
    }

    #[test]
    fn one_level_detection() {
        let one = |k: &str| match S3Selection::parse(k).unwrap() {
            S3Selection::Glob(g) => g.one_level,
            _ => panic!("not a glob"),
        };
        assert!(one("file.txt*"));
        assert!(one("data/*.npz"));
        assert!(!one("data/**.npz"));
        assert!(!one("data/*/x"));
    }

    #[test]
    fn selects_follows_the_rule() {
        let exact = S3Selection::parse("run.1").unwrap();
        assert!(exact.selects("run.1"));
        assert!(exact.selects("run.1/x"));
        assert!(!exact.selects("run.1.old"));
        assert!(!exact.selects("run.1.d/x"));
        assert!(!exact.selects("run_1"));
        assert!(!exact.selects("prerun.1"));
        let dir = S3Selection::parse("dl/").unwrap();
        assert!(dir.selects("dl/a"));
        assert!(!dir.selects("dl2/x"));
        let g = S3Selection::parse("file.txt*").unwrap();
        assert!(g.selects("file.txt") && g.selects("file.txt.bak") && g.selects("file.txt2"));
        assert!(!g.selects("file.txt/x"));
    }
}
