// src/redact.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Credential redaction for URIs that are shown in error text and log lines.
//!
//! One implementation for every message that quotes a URI: the Python batch errors and the
//! download code (issue #284) both call [`redact_uri`]. [`sanitized_error`] is re-exported (hidden) at
//! the crate root for the `s3-cli` binary; everything else here is crate-private. Ungated on purpose: the messages exist
//! in every build.

/// `uri` for error text: without `userinfo@` credentials and, for `http(s)`, without the
/// `?query` / `#fragment` (an Azure SAS token, a presigned signature). Ungated on purpose: the
/// Azure helper of the same job is built only with `backend-azure`, and these messages exist in
/// every build. Never panics: every cut is at an ASCII delimiter, so always a char boundary.
///
/// The text may hold more than the URI. The scheme is the maximal run of
/// `[A-Za-z][A-Za-z0-9+.-]*` immediately before the first `://`; everything before that run
/// (whitespace, quotes, `url=`, `x `, ...) is kept verbatim as a prefix, and the rest is
/// redacted as a URI. Only that first URI is redacted: a second URI later in the text is not.
///
/// Rules, for `scheme://authority/rest`:
/// * Userinfo is everything after `://` up to the LAST `@` before the first `/`, and is removed.
///   If that userinfo holds a `?` or `#` (a password holding one, or a host followed by a query
///   whose value holds an `@`: `https://h?e=a@b&sig=S`), the text behind the removed `@` is
///   taken for the URI's query / fragment: for `http(s)` it is removed too (`https://h?e=a@b&sig=S`
///   and `https://user:p?w@host/x?sig=1` -> `https://`); for other schemes it is removed only
///   when it holds a credential parameter (`s3://user:p?w@host/b/k?x` -> `s3://host/b/k?x`).
///   Without a `?` / `#` in the userinfo nothing of this applies (`https://h/x?e=a@b&sig=S` ->
///   `https://h/x`).
///   `abfs://` / `abfss://` keep their `container@account` (a container is not userinfo) ONLY
///   when the part before the `@` holds no `:`; `token:secret@account` is userinfo and is removed.
/// * `http(s)` only: an authority without `@` that looks like `user:password` (a `:` followed
///   by something that is not a port number) while the first path segment (up to the next `/`,
///   `?` or `#`) holds an `@` is a password that contained a `/` (`https://user:p/w@host/x`): everything up to that `@` is
///   removed. A numeric password is indistinguishable from a port, and a password holding two
///   or more `/` is not caught either (documented limits). Other schemes are left alone:
///   `s3://b/key@v1`, `file://localhost:abc/a@b` and `http://host:9000/key@v1` are untouched.
/// * Malformed slashes after `http(s):` (`https:/u:SECRET@h/`, `https:///u:SECRET@h/`): any
///   `@` in the first segment, which ends at the first `/`, ends the userinfo, as above
///   (`https:/h?e=a@b&sig=S` -> `https:/`; a password holding a `/` is not caught here). Other schemes are not treated this way (`file:///tmp/a@b`).
/// * `http(s)`: `?query` / `#fragment` are cut. Every other scheme keeps its path verbatim, `?`
///   being an ordinary key character in `s3://`, `gs://`, `az://`, ..., EXCEPT when the text
///   after the first `?` holds a credential parameter ([`CREDENTIAL_QUERY_MARKERS`], matched
///   case-insensitively anywhere in that text, so `sv=...&sig=...` is covered by `sig=` and a
///   SigV2 presigned `...&Signature=...` by `signature=`): the
///   URI is then cut at that `?`. Trade-off: a literal key such as `s3://b/a?x&sig=1` loses its
///   tail in the message, which beats showing a signature; parameters named differently from
///   the markers are not recognised.
/// * Documented limits: see [`redact_all_uris`] (rule 4 is applied in the same cases and with
///   the same exceptions).
/// * Without a scheme, `x@` is removed only for `user:password@` (a `:` before the first `@`
///   within the first `/`-segment) and for an explicit `//authority@`: `user:pw@host/x` ->
///   `host/x`, `//user@host/x` -> `//host/x`; `photo@2x.png` and `a@b` are unchanged.
pub(crate) fn redact_uri(uri: &str) -> String {
    for (colon, _) in uri.match_indices(":/") {
        let start = scheme_start(&uri[..colon]);
        if start == colon {
            continue; // no scheme in front of this `:/`
        }
        let (prefix, scheme) = (&uri[..start], &uri[start..colon]);
        let after = &uri[colon + 1..]; // the slashes and everything behind them
        let slashes = after.bytes().take_while(|&b| b == b'/').count();
        let web = is_web_scheme(scheme);
        let redacted = if slashes == 2 || (slashes > 2 && !web) {
            redact_scheme_uri(scheme, &after[2..])
        } else if web {
            redact_malformed_web_uri(scheme, &uri[colon..=colon + slashes], &after[slashes..])
        } else {
            continue; // `x:/y`: not a URI of a kind this function knows
        };
        return format!("{prefix}{redacted}");
    }
    redact_schemeless_uri(uri)
}

/// Query text that marks a signed / token-carrying URI (compared in lower case).
const CREDENTIAL_QUERY_MARKERS: [&str; 8] = [
    "sig=",
    "signature=",
    "x-amz-signature=",
    "x-amz-credential=",
    "x-amz-security-token=",
    "x-goog-signature=",
    "x-goog-credential=",
    "token=",
];

/// Byte index in `before` where the scheme that ends `before` starts (`before.len()` if none):
/// the maximal trailing run of scheme characters, trimmed to begin with a letter.
fn scheme_start(before: &str) -> usize {
    let bytes = before.as_bytes();
    let mut start = bytes.len();
    while start > 0
        && (bytes[start - 1].is_ascii_alphanumeric() || b"+-.".contains(&bytes[start - 1]))
    {
        start -= 1;
    }
    while start < bytes.len() && !bytes[start].is_ascii_alphabetic() {
        start += 1;
    }
    start
}

fn is_web_scheme(scheme: &str) -> bool {
    scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
}

/// `host/path` split at the first `/`, `?` or `#`: (`host`, `/path`), (`host`, `?query`), or
/// (`host`, `""`): the part that `looks_like_user_and_password` and the abfs / `@` handling
/// look at once the `?` / `#`-in-userinfo case has been dealt with.
fn split_authority(rest: &str) -> (&str, &str) {
    match rest.find(['/', '?', '#']) {
        Some(slash) => rest.split_at(slash),
        None => (rest, ""),
    }
}

/// `true` for `user:password`-shaped text: a `:` followed by something that is not a port
/// number, outside an IPv6 literal.
fn looks_like_user_and_password(authority: &str) -> bool {
    if authority.starts_with('[') {
        return false;
    }
    match authority.rsplit_once(':') {
        Some((_, tail)) => !tail.is_empty() && !tail.bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

/// Cut `out` at the first `?` or `#` at or after byte `from`.
fn cut_query_and_fragment(out: &mut String, from: usize) {
    if let Some(cut) = out[from..].find(['?', '#']) {
        out.truncate(from + cut);
    }
}

/// `true` if `text` holds a credential parameter ([`CREDENTIAL_QUERY_MARKERS`], case-insensitive).
fn holds_credential_marker(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    CREDENTIAL_QUERY_MARKERS.iter().any(|m| lower.contains(m))
}

fn redact_scheme_uri(scheme: &str, rest: &str) -> String {
    let lower = scheme.to_ascii_lowercase();
    let web = lower == "http" || lower == "https";
    // Userinfo ends at the LAST `@` before the first `/`. A `?` or `#` inside it (a password
    // holding one, or a host followed by a query whose value holds an `@`) makes the text
    // behind that `@` the URI's query / fragment: `http(s)` drops it, any other scheme drops
    // it only when it holds a credential parameter.
    let whole_end = rest.find('/').unwrap_or(rest.len());
    if let Some(at) = rest[..whole_end].rfind('@') {
        if rest[..at].contains(['?', '#']) {
            let tail = &rest[at + 1..];
            if web || holds_credential_marker(tail) {
                return format!("{scheme}://");
            }
            return format!("{scheme}://{tail}");
        }
    }
    let (mut authority, mut path) = split_authority(rest);
    // `abfs(s)://container@account` keeps its authority unless the part before the last `@`
    // holds a `:` (`token:secret@account`): then it is userinfo like any other.
    let container_form = (lower == "abfs" || lower == "abfss")
        && authority
            .rfind('@')
            .is_none_or(|at| !authority[..at].contains(':'));
    if !container_form {
        match authority.rfind('@') {
            Some(at) => authority = &authority[at + 1..],
            None if web && looks_like_user_and_password(authority) && path.starts_with('/') => {
                // The password held a `/`: the "path" starts with the rest of it, up to an `@`.
                let segment = &path[1..];
                let segment_end = segment.find(['/', '?', '#']).unwrap_or(segment.len());
                if let Some(at) = segment[..segment_end].rfind('@') {
                    (authority, path) = split_authority(&segment[at + 1..]);
                }
            }
            None => {}
        }
    }
    let mut out = format!("{scheme}://{authority}{path}");
    // Keep the `scheme://` prefix: it holds no `?` or `#`.
    let from = scheme.len() + 3;
    if web {
        cut_query_and_fragment(&mut out, from);
    } else if let Some(q) = out[from..].find('?') {
        if holds_credential_marker(&out[from + q + 1..]) {
            out.truncate(from + q);
        }
    }
    out
}

/// `http(s):` followed by one slash or by three or more (`sep` is the `:` and the slashes):
/// the userinfo ends at the last `@` of the first segment of `rest`, which ends at the first
/// `/`. If that userinfo holds a `?` or `#`, the text behind the `@` is a query / fragment and
/// is removed with it (as in [`redact_scheme_uri`]).
fn redact_malformed_web_uri(scheme: &str, sep: &str, rest: &str) -> String {
    let segment_end = rest.find('/').unwrap_or(rest.len());
    let tail = match rest[..segment_end].rfind('@') {
        Some(at) if rest[..at].contains(['?', '#']) => return format!("{scheme}{sep}"),
        Some(at) => &rest[at + 1..],
        None => rest,
    };
    let mut out = format!("{scheme}{sep}{tail}");
    cut_query_and_fragment(&mut out, scheme.len() + sep.len());
    out
}

fn redact_schemeless_uri(uri: &str) -> String {
    let (lead, rest) = match uri.strip_prefix("//") {
        Some(rest) => ("//", rest),
        None => ("", uri),
    };
    let segment = &rest[..rest.find('/').unwrap_or(rest.len())];
    match (segment.find('@'), segment.rfind('@')) {
        (Some(first), Some(last)) if !lead.is_empty() || segment[..first].contains(':') => {
            format!("{lead}{}", &rest[last + 1..])
        }
        _ => uri.to_string(),
    }
}

/// `uri` redacted by [`redact_uri`] with every control character escaped, for a message that
/// quotes a REMOTE-CONTROLLED string (an object key): a key holding a newline or an escape
/// sequence must not forge log lines or terminal output. Printable text, including non-ASCII
/// text and backslashes, is left as it is.
pub(crate) fn redact_and_escape(uri: &str) -> String {
    escape_control(&redact_uri(uri))
}

/// `text` with the credentials of EVERY `scheme://` URI removed, for text that quotes URIs it
/// does not control: the cause chain of a provider error (an S3 service message, an HTTP client
/// error that quotes the request URL, a JSON / XML error body, ...). [`redact_uri`] alone
/// handles only the first URI of its input. Text that is not part of such a URI is returned
/// unchanged.
///
/// Deliberately simple, conservative rules (security over diagnostic fidelity), applied in ONE
/// left-to-right pass, linear in the length of `text` (every search resumes where the previous
/// one stopped; nothing is rescanned per separator). Never panics: every cut is at an ASCII
/// byte or at a boundary found by a `char` search.
///
/// 1. A URI starts at the maximal `[A-Za-z][A-Za-z0-9+.-]*` run in front of a `://`. That run
///    never reaches back into text that an earlier URI's redaction already consumed.
/// 2. The *userinfo span* is the text after `://` up to the first whitespace or `/`; for
///    `http(s)` a run of extra slashes directly after `://` is skipped first (`https:///u:S@h/x`
///    -> `https:///h/x`, as in [`redact_uri`]). If it
///    holds an `@`, everything from after `://` through the LAST `@` of the span is userinfo
///    and is removed. If the removed userinfo holds a `?` or `#` (a password holding one, or a
///    host followed by a query whose value holds an `@`: `https://h?e=a@b&sig=S`), the text
///    behind that `@` is this URI's query / fragment (there is no kept path in front of it) and
///    runs to the end of the whitespace word, NOT to a structural boundary, so a quote in the
///    path cannot hide the real query (`https://u:p?w@h/it's.jpg?sig=S` -> `https://`):
///    `http(s)` always removes it, other schemes when a credential parameter starts anywhere in
///    it (`https://u:ab?c@h` -> `https://`, `s3://u:ab?c@h/k` -> `s3://h/k`,
///    `s3://u:ab?c@h/k?sig=S` -> `s3://`). There is no digit, port or separator heuristic: a password may
///    hold any other character, quotes, brackets, `,`, `:` and `@` included
///    (`https://u:a:,S@h/x`, `https://u:12"S@h/x`).
/// 3. *Query / fragment.* The first `?` (`http(s)`: the first `?` or `#`) after the userinfo
///    and before the next whitespace starts it. Its text runs to the next *structural
///    boundary*: whitespace, `"`, `'`, `<`, `>` or a backtick (a boundary that is the very
///    first character of the query belongs to the query: `https://h/x?"S`). `)` `]` `}` `,`
///    `;` are NOT boundaries, so a signature or password cannot hide behind them. For
///    `http(s)` the query / fragment is always removed; for every other scheme it is removed
///    only when it holds a credential parameter ([`CREDENTIAL_QUERY_MARKERS`], case-insensitive;
///    `?` is an ordinary key character in `s3://b/what?.csv`). The text from the boundary on is
///    always kept, so compact JSON / XML keep their other fields:
///    `{"url":"https://h/b?a=1","code":"X"}` -> `{"url":"https://h/b","code":"X"}`. A quote or
///    `<` in the PATH does not hide the query behind it (`https://h/it's.jpg?sig=S`).
///    If the removed text holds a `://` (a later URI glued into the same word, whose own
///    quote / `<` would otherwise end the cut inside it and leave its tail as text:
///    `https://a/x?y=1,https://h/it's?sig=S`), the cut extends to the end of the whitespace
///    word. Trade-off, favouring redaction: a cut query that holds a nested URI removes the
///    rest of the word (`{"u":"https://h/b?next=https://x/y","c":"X"}` -> `{"u":"https://h/b`).
/// 4. `http(s)` only: an authority without `@` that looks like `user:password` (a `:` followed
///    by something that is not a port number) while the first path segment (up to the next
///    `/`, `?`, `#` or whitespace: an `@` in a query or fragment is not userinfo,
///    `https://h:x/p?e=a@b&sig=S` -> `https://h:x/p`) holds an `@` is a password that contained ONE `/`
///    (`https://u:p/w@h/x`): everything up to that `@` is removed.
/// 5. URIs nested in the kept text (a `next=https://u:S@h/p` inside a kept query) are redacted
///    by the same rules; those inside removed text are gone with it. A `?` / `#` that lies
///    after a later URI's `://` belongs to that later URI (or one after it), not to the URI in
///    front: that URI does not cut there, and each URI is redacted by its own iteration of the
///    pass (`["https://a/x","https://u:S@b/y?z=1"]` -> `["https://a/x","https://b/y"]`;
///    `https://h/p,https://u:S?w@h/x` -> `https://h/p,https://`). So the path a cut keeps never
///    holds a URI, and nothing is scanned recursively.
///
/// Documented limits and accepted trade-offs:
/// * A password holding `/` ends the userinfo span there. One `/` is handled for `http(s)`
///   only (rule 4), two or more are not. A password holding `?` or `#` is removed whole, but
///   so is everything behind it up to the end of the whitespace word for `http(s)`
///   (`https://u:ab?c@h/x` -> `https://`); a URI whose host is followed by a query with an `@` in
///   it is read the same way (`https://h?e=a@b&sig=S` -> `https://`), over-redacting.
/// * A password holding both a `?` / `#` and a `/` is not removed whole: the userinfo span ends
///   at the `/`, so the cut starts at the `?` / `#` and the part of the password in front of it
///   stays (`https://u:S?w/x@h/p` -> `https://u:S`).
/// * Only `scheme://` is recognised: a malformed `https:/h?e=a@b&sig=S` in free text is not
///   (`redact_uri` handles it for a single URI).
/// * Rule 4 is not applied, and the password keeps its part after the `/`, when the part before
///   the `/` ends in `:` followed by digits (it reads as a port) or in a bare `:`, when the
///   user holds an `@`, or when it starts with `[` (an IPv6 literal): `https://u:a:12/w@h`,
///   `https://u:S::/S@h/x`.
/// * A bare-host URI directly followed by text that holds an `@` but no whitespace, `/`, `?` or
///   `#` is taken for userinfo and over-redacted: `endpoint=https://h,user=a@b` ->
///   `endpoint=https://b`, `{"e":"https://h","o":"bob@example.com"}` ->
///   `{"e":"https://example.com"}`.
/// * `abfs(s)://container@account` loses its `container@` (no scheme-specific cases).
/// * A schemeless `user:password@host` in running text is NOT recognised (free text has no safe
///   boundary for it: `note: see a@b` would be damaged).
/// * A password holding whitespace is not recognised (the authority ends there).
/// * A credential parameter outside [`CREDENTIAL_QUERY_MARKERS`] is removed only from `http(s)`
///   URIs (which lose the whole query), not from other schemes.
/// * The scheme is the MAXIMAL run of letters, digits, `+`, `.` and `-` in front of `://`, so
///   such characters glued in front of a scheme join it: `b.http://h/#k=S` is read as the
///   non-web scheme `b.http`. For non-web schemes only a `?` starts a query: a `#...` fragment
///   is never cut, even with a credential parameter (`s3://b/k#sig=S` is kept; `#` is an
///   ordinary key character).
/// * A raw `"` / `'` / `<` / `>` / backtick ANYWHERE in a query (but not at its very start)
///   before a credential parameter ends the removed part there: what follows stays, so
///   `https://h/x?name=O'Brien&X-Amz-Signature=SIG` keeps `'Brien&X-Amz-Signature=SIG`. Real
///   signed URLs percent-encode these characters, so none occurs in a real signature.
/// * When an `http(s)` URI without a query is followed in the same whitespace-free word by an
///   unrelated `?` (`{"u":"https://h/b","q":"what?","c":"X"}`), that `?` starts the cut (the
///   text is damaged around it, nothing leaks).
pub(crate) fn redact_all_uris(text: &str) -> String {
    let mut scan = Scan::new(text);
    let mut out = String::new();
    let mut copied = 0; // text[..copied] is already in `out` or removed
    let mut floor = 0; // text before `floor` is consumed: no scheme can start there
    for (at, _) in text.match_indices("://") {
        if at < floor {
            continue; // inside text that was removed
        }
        let start = floor + scheme_start(&text[floor..at]);
        let body = at + 3;
        floor = body;
        if start == at {
            continue; // no scheme in front of this `://`
        }
        let web = is_web_scheme(&text[start..at]);

        // Userinfo: up to the last `@` before the first whitespace or `/`. For `http(s)` the
        // authority starts behind any run of extra slashes (`https:///u:S@h/x`, as in
        // `redact_uri`); they stay in the output.
        let auth = if web {
            body + text[body..].bytes().take_while(|&b| b == b'/').count()
        } else {
            body
        };
        let span_end = text[auth..]
            .find(|c: char| c.is_whitespace() || c == '/')
            .map_or(text.len(), |i| auth + i);
        let mut rest = body;
        if let Some(i) = text[auth..span_end].rfind('@') {
            rest = auth + i + 1;
            if text[auth..rest - 1].contains(['?', '#']) {
                // A `?` / `#` in the userinfo: the text behind the `@` is this URI's query /
                // fragment, with no kept path in front of it. It runs to the end of the
                // whitespace word, NOT to a structural boundary: a quote in the path must not
                // hide the real query behind it (`https://u:p?w@h/it's.jpg?sig=S`). Web URIs
                // lose it; other schemes lose it when a credential marker starts anywhere in
                // it.
                out.push_str(&text[copied..auth]);
                let end = scan.word_end(rest);
                if web || scan.has_credential_marker_from_userinfo(rest, end) {
                    copied = end;
                    floor = end;
                } else {
                    copied = rest;
                    floor = rest;
                }
                continue;
            }
        } else if web
            && text[span_end..].starts_with('/')
            && !text[auth..span_end].contains(['?', '#'])
            && looks_like_user_and_password(&text[auth..span_end])
        {
            // The password held a `/`: the first path segment starts with the rest of it.
            let segment = span_end + 1;
            let segment_end = text[segment..]
                .find(|c: char| c.is_whitespace() || matches!(c, '/' | '?' | '#'))
                .map_or(text.len(), |i| segment + i);
            if let Some(i) = text[segment..segment_end].rfind('@') {
                rest = segment + i + 1;
            }
        }
        if rest > body {
            out.push_str(&text[copied..auth]);
            copied = rest;
            floor = rest;
        }

        // Query / fragment: the first `?` (`#`) before the end of the word.
        let word_end = scan.word_end(rest);
        let mark = if web {
            scan.query_or_fragment(rest)
        } else {
            scan.query(rest)
        };
        let Some(mark) = mark.filter(|&m| m < word_end) else {
            continue;
        };
        // A boundary right behind the mark is the first character of the query.
        let from = match text.as_bytes().get(mark + 1) {
            Some(b'"' | b'\'' | b'<' | b'>' | b'`') => mark + 2,
            _ => mark + 1,
        };
        let end = scan.boundary(from);
        if web || scan.has_credential_marker(mark + 1, end) {
            // A later URI starts in front of the mark (`https://h/p,https://u:S?w@h/x`): the mark
            // belongs to that URI or a later one (here it lies in the later URI's userinfo), not
            // to this one. Do not cut for this URI and consume nothing past `rest`: the main loop
            // reaches the later URI, whose own rules (userinfo, query cut) handle the mark. So the
            // path span that a cut keeps never holds a URI. Linear: the search stops at the first
            // URI start, the next URI's own search begins behind it (disjoint regions); when it
            // finds none the region `rest..mark` is consumed by the cut below (`floor = end`).
            if holds_uri_start(&text[rest..mark]) {
                continue;
            }
            // The boundary can lie inside a LATER URI glued into this word
            // (`...?y=1,https://h/it's`): cutting there would drop that URI's scheme and keep its
            // tail as plain text. Remove the rest of the word instead (`word_end` >= `end`: a
            // whitespace is a boundary). Linear: this containment scan runs only when the cut
            // fires, and then the scanned region is consumed (`floor = end`), so each byte is
            // scanned once. When the cut does not fire nothing is removed, so a later URI in that
            // region is handled by its own iteration (no leak) and no rescan is needed.
            let end = if text[mark..end].contains("://") {
                word_end
            } else {
                end
            };
            out.push_str(&text[copied..mark]);
            copied = end;
            floor = end;
        }
    }
    if copied == 0 {
        return text.to_string();
    }
    out.push_str(&text[copied..]);
    out
}

/// Whether `span` holds the start of a URI: a `://` with a scheme in front of it.
fn holds_uri_start(span: &str) -> bool {
    span.match_indices("://")
        .any(|(at, _)| scheme_start(&span[..at]) < at)
}

/// The monotone searches of [`redact_all_uris`]. Every search resumes where the last one for
/// the same pattern stopped, so the work of one whole pass is linear in the text.
struct Scan<'a> {
    text: &'a str,
    /// Lower-case copy of the text (ASCII only, so the offsets are the same), made on demand.
    lower: Option<String>,
    word: Cursor,
    query: Cursor,
    query_or_fragment: Cursor,
    markers: [Cursor; CREDENTIAL_QUERY_MARKERS.len()],
    /// Cursors of [`Scan::has_credential_marker_from_userinfo`] only (its `from` sequence differs
    /// from the one of `markers`).
    userinfo_markers: [Cursor; CREDENTIAL_QUERY_MARKERS.len()],
    /// `(from, end)`: no structural boundary in `text[from..end]`, and one at `end`.
    boundary: Option<(usize, usize)>,
}

impl<'a> Scan<'a> {
    fn new(text: &'a str) -> Self {
        Scan {
            text,
            lower: None,
            word: Cursor::default(),
            query: Cursor::default(),
            query_or_fragment: Cursor::default(),
            markers: Default::default(),
            userinfo_markers: Default::default(),
            boundary: None,
        }
    }

    /// First whitespace at or after `from`, or the end of the text.
    fn word_end(&mut self, from: usize) -> usize {
        let text = self.text;
        self.word
            .find(from, |tail| tail.find(char::is_whitespace), text)
            .unwrap_or(text.len())
    }

    /// First `?` at or after `from`.
    fn query(&mut self, from: usize) -> Option<usize> {
        self.query.find(from, |tail| tail.find('?'), self.text)
    }

    /// First `?` or `#` at or after `from`.
    fn query_or_fragment(&mut self, from: usize) -> Option<usize> {
        self.query_or_fragment
            .find(from, |tail| tail.find(['?', '#']), self.text)
    }

    /// First structural boundary (whitespace, `"`, `'`, `<`, `>`, backtick) at or after `from`,
    /// or the end of the text.
    fn boundary(&mut self, from: usize) -> usize {
        if let Some((start, end)) = self.boundary {
            if start <= from && from <= end {
                return end;
            }
        }
        let end = self.text[from..]
            .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | '`'))
            .map_or(self.text.len(), |i| from + i);
        self.boundary = Some((from, end));
        end
    }

    /// `true` if a credential parameter starts in `text[from..end]` (case-insensitive).
    fn has_credential_marker(&mut self, from: usize, end: usize) -> bool {
        let text = self.text;
        let lower = self.lower.get_or_insert_with(|| text.to_ascii_lowercase());
        CREDENTIAL_QUERY_MARKERS
            .iter()
            .zip(self.markers.iter_mut())
            .any(|(marker, cursor)| {
                cursor
                    .find(from, |tail| tail.find(*marker), lower)
                    .is_some_and(|at| at < end)
            })
    }

    /// `true` if a credential parameter starts in `text[from..end]` (case-insensitive): the check
    /// of the userinfo-holding-`?`/`#` branch of [`redact_all_uris`], with its own cursors.
    /// `from` is the position behind that branch's `@` and does not decrease between calls: the
    /// userinfo before it is consumed (removed, or the branch leaves `floor` at `from`), so the
    /// next `://` that reaches this branch is at or after `floor`, its authority starts after
    /// that `://`, and so does its `@`. The search runs to the end of the text, not to `end`, so
    /// a cut that does not fire costs nothing when repeated (no rescan per URI).
    fn has_credential_marker_from_userinfo(&mut self, from: usize, end: usize) -> bool {
        let text = self.text;
        let lower = self.lower.get_or_insert_with(|| text.to_ascii_lowercase());
        CREDENTIAL_QUERY_MARKERS
            .iter()
            .zip(self.userinfo_markers.iter_mut())
            .any(|(marker, cursor)| {
                cursor
                    .find(from, |tail| tail.find(*marker), lower)
                    .is_some_and(|at| at < end)
            })
    }
}

/// "First match at or after `from`" for queries whose `from` never decreases: the last match
/// is remembered, and a search that found nothing is never repeated.
#[derive(Default)]
struct Cursor {
    hit: Option<usize>,
    exhausted: bool,
}

impl Cursor {
    /// Byte index of the first match in `hay[from..]` (`search` returns a match offset in the
    /// tail it is given); `from` must not be smaller than in an earlier call.
    fn find(
        &mut self,
        from: usize,
        search: impl FnOnce(&str) -> Option<usize>,
        hay: &str,
    ) -> Option<usize> {
        if let Some(hit) = self.hit.filter(|&hit| hit >= from) {
            return Some(hit);
        }
        if self.exhausted {
            return None;
        }
        self.hit = search(&hay[from..]).map(|i| from + i);
        self.exhausted = self.hit.is_none();
        self.hit
    }
}

/// [`redact_all_uris`] and then [`escape_control`]: the text of a REMOTE-CONTROLLED error
/// message (a cause chain quoting object keys and provider messages) made safe to return,
/// show and log. Credentials of every `scheme://` URI in it are gone and no control, line
/// separator or bidi / format character is left raw.
pub(crate) fn redact_and_escape_text(text: &str) -> String {
    escape_control(&redact_all_uris(text))
}

/// `err` rebuilt from its sanitised rendering, for an error that is RETURNED to a caller (a
/// listing or store-construction failure whose cause chain may quote a provider message with
/// signed URLs or `user:password@`): the text of the whole chain (`{err:#}`) goes through
/// [`redact_and_escape_text`], so it is safe to show and log.
///
/// The original error types are lost, with one exception: a typed
/// [`ObjectError`](crate::object_error::ObjectError) (its kind and provider, and its own
/// message, which holds only a status and a sanitised service code) is attached again over
/// the sanitised text, so `object_error(&result)` still classifies the failure (not-found,
/// permission, transport, ...) exactly as it did for `err`. The typed layer's text is shown
/// ONCE, as the outermost layer, wherever it stood in the chain (a bare typed error gives the
/// same text and classification back).
pub fn sanitized_error(err: &anyhow::Error) -> anyhow::Error {
    use crate::object_error::{attach_object_error, ObjectError};
    let parts: Vec<String> = err.chain().map(|c| c.to_string()).collect();
    let Some(typed) = crate::object_error::object_error(err) else {
        return anyhow::anyhow!(redact_and_escape_text(&parts.join(": ")));
    };
    // The typed layer shows in the chain as its own `Display`; the attached layer re-creates
    // that text, so the layer is left out of the rendered rest.
    let shown = typed.to_string();
    let at = parts.iter().position(|p| *p == shown);
    let rest: Vec<&str> = parts
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != at)
        .map(|(_, p)| p.as_str())
        .collect();
    // `Display` is `<provider> <kind>[ (length)][: <message>]`: take the message back.
    let message = match at {
        Some(_) => shown
            .split_once(": ")
            .map_or(String::new(), |(_, m)| redact_and_escape_text(m)),
        None => String::new(),
    };
    if rest.is_empty() {
        return anyhow::Error::new(ObjectError::new(typed.kind, typed.provider, message));
    }
    let rebuilt = anyhow::anyhow!(redact_and_escape_text(&rest.join(": ")));
    attach_object_error(rebuilt, typed.kind, typed.provider, message)
}

/// `text` with each control character, Unicode line / paragraph separator and invisible
/// format character (bidi embeddings, overrides and isolates, zero-width and directional
/// marks, the byte-order mark, ...: the "Trojan Source" spoofing set) replaced by a visible
/// escape: `\n`, `\r`, `\t`, or `\u{..}`. Letters, marks and symbols of every script stay.
pub(crate) fn escape_control(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() || matches!(c, '\u{2028}' | '\u{2029}') || is_format_char(c) => {
                out.push_str(&format!("\\u{{{:x}}}", u32::from(c)));
            }
            c => out.push(c),
        }
    }
    out
}

/// `true` for the characters of Unicode general category Cf (format): invisible, yet able to
/// reorder or hide the text around them. The std library has no category test, so the ranges
/// (Unicode 16) are listed here.
fn is_format_char(c: char) -> bool {
    matches!(u32::from(c),
        0x00AD
        | 0x0600..=0x0605
        | 0x061C
        | 0x06DD
        | 0x070F
        | 0x0890..=0x0891
        | 0x08E2
        | 0x180E
        | 0x200B..=0x200F
        | 0x202A..=0x202E
        | 0x2060..=0x2064
        | 0x2066..=0x206F
        | 0xFEFF
        | 0xFFF9..=0xFFFB
        | 0x110BD
        | 0x110CD
        | 0x13430..=0x1343F
        | 0x1BCA0..=0x1BCA3
        | 0x1D173..=0x1D17A
        | 0xE0001
        | 0xE0020..=0xE007F)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_uri_rules() {
        for (input, want) in [
            // Plain URIs of every scheme.
            ("s3://b/k", "s3://b/k"),
            ("file:///tmp/a.bin", "file:///tmp/a.bin"),
            ("direct:///tmp/a.bin", "direct:///tmp/a.bin"),
            ("gs://b/o", "gs://b/o"),
            ("gcs://b/o", "gcs://b/o"),
            ("az://acct/c/o", "az://acct/c/o"),
            ("s3://", "s3://"),
            ("", ""),
            // `?` and `#` are key characters outside http(s): the path is kept verbatim.
            ("s3://b/what?.csv", "s3://b/what?.csv"),
            ("s3://b/k#frag?x=1", "s3://b/k#frag?x=1"),
            ("gs://b/a?b#c", "gs://b/a?b#c"),
            ("az://acct/c/o?x=1", "az://acct/c/o?x=1"),
            ("file:///tmp/what?.csv", "file:///tmp/what?.csv"),
            ("direct:///tmp/a#b", "direct:///tmp/a#b"),
            ("s3://b/a%3Fb", "s3://b/a%3Fb"),
            ("s3://b/key@v1", "s3://b/key@v1"),
            ("s3://host:9000/key@v1", "s3://host:9000/key@v1"),
            // Userinfo goes, whatever the scheme.
            ("s3://user:pw@host:9000/b/k", "s3://host:9000/b/k"),
            (
                "s3://user:pw@host:9000/b/k?X-Amz-Signature=abc",
                "s3://host:9000/b/k",
            ),
            ("s3://tok@bucket", "s3://bucket"),
            ("file://user:pw@host/x", "file://host/x"),
            ("a://b@c@d/x", "a://d/x"),
            // `abfs://container@account`: the container is not userinfo.
            (
                "abfss://container@acct.dfs.core.windows.net/p/q",
                "abfss://container@acct.dfs.core.windows.net/p/q",
            ),
            ("abfs://c@acct/p?x", "abfs://c@acct/p?x"),
            (
                "abfss://tok:SECRET@acct.dfs.core.windows.net/p",
                "abfss://acct.dfs.core.windows.net/p",
            ),
            // A signature parameter outside http(s): the URI is cut at the `?`.
            (
                "gs://b/o?GoogleAccessId=x&Expires=1&Signature=SECRET",
                "gs://b/o",
            ),
            (
                "s3://b/k?AWSAccessKeyId=AK&Signature=SECRET&Expires=1",
                "s3://b/k",
            ),
            ("s3://b/a?design=1", "s3://b/a?design=1"),
            // http(s): the query and the fragment go (SAS tokens, presigned signatures).
            (
                "https://acct.blob.core.windows.net/c/k?sv=2022&sig=SECRET%3D&se=1",
                "https://acct.blob.core.windows.net/c/k",
            ),
            ("http://h/k#frag", "http://h/k"),
            ("HTTPS://h/x?sig=1", "HTTPS://h/x"),
            ("https://h?sig=1", "https://h"),
            ("https://h/p%3Fx", "https://h/p%3Fx"),
            ("https://h/key@v1", "https://h/key@v1"),
            ("https://h:9000/key@v1", "https://h:9000/key@v1"),
            ("https://[::1]:9000/key@v1", "https://[::1]:9000/key@v1"),
            // Passwords that hold `/`; one holding `?` or `#` is userinfo too, and the text
            // behind its `@` is a query / fragment that goes with it (http(s)).
            ("https://user:p?w@host/x?sig=1", "https://"),
            ("https://user:p#w@host/x", "https://"),
            ("https://user:p/w@host/x", "https://host/x"),
            // Only http(s) get the "password holding a slash" heuristic.
            ("s3://user:p/w@host/b/k", "s3://user:p/w@host/b/k"),
            ("s3://user:p?w@host/b/k?x", "s3://host/b/k?x"),
            ("s3://u:ab?c@h/k", "s3://h/k"),
            ("s3://u:ab?c@h/k?sig=1", "s3://"),
            ("https://u:ab?c@h", "https://"),
            ("abfs://c?w@acct/p", "abfs://acct/p"),
            // No `?` / `#` in the userinfo: unchanged.
            ("https://u:p@h/x?sig=1", "https://h/x"),
            ("https://h?e=1", "https://h"),
            ("https://h/x?e=a@b&sig=S", "https://h/x"),
            ("s3://h?e=a@b&sig=S", "s3://"),
            // IPv6.
            ("http://u:p@[::1]:9000/b/k", "http://[::1]:9000/b/k"),
            ("http://u:p@[::1]:9000/b/k?sig=1#f", "http://[::1]:9000/b/k"),
            ("s3://[::1]:9000/b/k", "s3://[::1]:9000/b/k"),
            // Non-ASCII.
            ("s3://bücket/ключ?x", "s3://bücket/ключ?x"),
            ("https://üser:pä@höst/ключ?sig=ü#é", "https://höst/ключ"),
            ("https://üser:pä/w@höst/x", "https://höst/x"),
            ("usér:pw@hóst/päth", "hóst/päth"),
            // No scheme.
            ("user:pw@host/path", "host/path"),
            ("//user:pw@host/x", "//host/x"),
            ("//host/x", "//host/x"),
            ("/plain/path?x=1", "/plain/path?x=1"),
            ("/a/key@v1", "/a/key@v1"),
            ("rel/dir@x/f", "rel/dir@x/f"),
            ("just-a-name", "just-a-name"),
            ("://x", "://x"),
            // Text before the scheme is kept verbatim; the URI after it is redacted.
            (" https://u:SECRET@h/x", " https://h/x"),
            ("\thttps://u:SECRET@h/x?sig=1", "\thttps://h/x"),
            ("\"https://u:SECRET@h/x?sig=1\"", "\"https://h/x"),
            ("'s3://u:SECRET@h/b/k'", "'s3://h/b/k'"),
            ("url=https://u:SECRET@h/x", "url=https://h/x"),
            ("x https://u:SECRET@h/x?sig=1", "x https://h/x"),
            ("(az://u:SECRET@h/c/o?sig=1)", "(az://h/c/o"),
            ("1https://u:SECRET@h/", "1https://h/"),
            ("ü https://üser:pä@höst/ключ?sig=ü", "ü https://höst/ключ"),
            // Wrong number of slashes after an http(s) scheme: a `user:pass@` before the first
            // `/` is still removed.
            ("https:/u:SECRET@h/", "https:/h/"),
            ("https:///u:SECRET@h/", "https:///h/"),
            ("https:/u:SECRET@h/x?sig=1#f", "https:/h/x"),
            ("HTTP:/tok@h/x", "HTTP:/h/x"),
            ("x https:/u:SECRET@h/", "x https:/h/"),
            ("https:/h/key@v1", "https:/h/key@v1"),
            ("https:///h/key@v1", "https:///h/key@v1"),
            ("file:///tmp/a@b", "file:///tmp/a@b"),
            ("file:///tmp/u:p@h", "file:///tmp/u:p@h"),
            // Non-http schemes keep `?`, unless the text after it holds a credential parameter.
            ("az://acct/c/o?sig=SECRET", "az://acct/c/o"),
            ("az://acct/c/o?sv=2022&sig=SECRET", "az://acct/c/o"),
            ("az://acct/c/o?SV=2022&SIG=SECRET", "az://acct/c/o"),
            ("s3://b/k?X-Amz-Signature=abc", "s3://b/k"),
            ("s3://b/k?a=1&x-amz-credential=AK%2F2030", "s3://b/k"),
            ("s3://b/k?X-Amz-Security-Token=T", "s3://b/k"),
            ("gs://b/o?X-Goog-Signature=1", "gs://b/o"),
            ("gcs://b/o?x-goog-credential=1", "gcs://b/o"),
            ("s3://b/k?token=T", "s3://b/k"),
            ("file:///t/a?sig=1", "file:///t/a"),
            ("direct:///t/a?Token=1", "direct:///t/a"),
            ("s3://b/k?x=1#sig=2", "s3://b/k"),
            ("s3://u:pw@h/b/k?sig=1", "s3://h/b/k"),
            ("s3://b/what?.csv", "s3://b/what?.csv"),
            ("s3://b/a?b=c", "s3://b/a?b=c"),
            ("az://acct/c/o?sv=2022", "az://acct/c/o?sv=2022"),
            ("s3://b/k?signal=1", "s3://b/k?signal=1"),
            ("s3://b/k?x-amz-expires=60", "s3://b/k?x-amz-expires=60"),
            // No scheme: `x@` goes only for `user:password@` (or an explicit `//` authority).
            ("photo@2x.png", "photo@2x.png"),
            ("a@b", "a@b"),
            ("a@b/c", "a@b/c"),
            ("dir/a:b@c", "dir/a:b@c"),
            ("//tok@host/x", "//host/x"),
            ("user:pw@host", "host"),
            ("u:p@ss@host/x", "host/x"),
            // `user:password` heuristic for an authority without `@`: http(s) only.
            ("file://localhost:abc/a@b", "file://localhost:abc/a@b"),
            ("s3://localhost:abc/a@b", "s3://localhost:abc/a@b"),
            ("http://localhost:9000/a@b", "http://localhost:9000/a@b"),
        ] {
            assert_eq!(redact_uri(input), want, "input {input:?}");
        }
    }

    /// Every prefix of a URI full of multi-byte characters is redacted without a panic.
    #[test]
    fn redact_uri_never_panics_on_any_prefix() {
        for uri in [
            "https://üser:pä?#/@w@höst:9000/ключ@v?sig=ü#é",
            "s3://b/日本語?語#語@語",
            "ü@ö://ä@ß/é?",
            "//ü:ö@[::ä]/x",
            "é\"ü https:/üser:pä@höst/é?sig=ü#é",
            "ü https:///ä@ö://ß/?token=é",
            "日本 s3://b/日本語?X-Amz-Signature=語",
        ] {
            for (end, _) in uri.char_indices().chain(std::iter::once((uri.len(), ' '))) {
                let _ = redact_uri(&uri[..end]);
            }
        }
    }

    #[test]
    fn escape_control_hides_control_characters_only() {
        assert_eq!(escape_control("a\nb\rc\td"), "a\\nb\\rc\\td");
        assert_eq!(escape_control("x\u{1b}[31m"), "x\\u{1b}[31m");
        assert_eq!(escape_control("nul\0end"), "nul\\u{0}end");
        assert_eq!(escape_control("l\u{2028}p\u{2029}"), "l\\u{2028}p\\u{2029}");
        // Printable text stays as it is.
        assert_eq!(escape_control("dir\\file ключ 日本"), "dir\\file ключ 日本");
    }

    #[test]
    fn escape_control_hides_bidi_and_format_characters() {
        // Trojan-Source style spoofing: bidi embeddings / overrides / isolates, zero-width and
        // directional marks, the byte-order mark, the soft hyphen and the C1 NEL stay visible.
        for c in [
            '\u{202a}',
            '\u{202b}',
            '\u{202c}',
            '\u{202d}',
            '\u{202e}',
            '\u{2066}',
            '\u{2067}',
            '\u{2068}',
            '\u{2069}',
            '\u{200b}',
            '\u{200c}',
            '\u{200d}',
            '\u{200e}',
            '\u{200f}',
            '\u{feff}',
            '\u{00ad}',
            '\u{2060}',
            '\u{061c}',
            '\u{85}',
            '\u{180e}',
            '\u{600}',
            '\u{206f}',
            '\u{e0041}',
        ] {
            let want = format!("a\\u{{{:x}}}b", u32::from(c));
            assert_eq!(
                escape_control(&format!("a{c}b")),
                want,
                "U+{:04X}",
                u32::from(c)
            );
        }
        // Ordinary non-ASCII letters, marks and symbols are not format characters.
        assert_eq!(
            escape_control("ключ 日本 é ü ß \u{301} \u{1f600}"),
            "ключ 日本 é ü ß \u{301} \u{1f600}"
        );
    }

    #[test]
    fn redact_and_escape_does_both() {
        assert_eq!(
            redact_and_escape("https://u:SECRET@h/x\ny?sig=1"),
            "https://h/x\\ny"
        );
        assert_eq!(redact_and_escape("s3://b/k\u{1b}[2J"), "s3://b/k\\u{1b}[2J");
    }

    // ---- redact_all_uris / redact_and_escape_text (review F-F1) ----

    /// None of the synthetic secrets of the F-F1 tests may survive `text`.
    fn assert_clean(text: &str) {
        for secret in [
            "SECRET1", "SECRET2", "SIG1", "SIG2", "SIG3", "TOK4", "user:", "admin:",
        ] {
            assert!(!text.contains(secret), "`{secret}` survived in {text:?}");
        }
    }

    #[test]
    fn redact_all_uris_handles_every_uri_not_only_the_first() {
        // The first URI is already redacted (the outer context); two credential-bearing
        // URIs follow in the "cause".
        let text = "s3://bkt/key: failed: see https://user:SECRET1@h/x?X-Amz-Signature=SIG1 and \
                    s3://b/k?sig=SIG2 end";
        let out = redact_all_uris(text);
        assert_clean(&out);
        assert_eq!(
            out,
            "s3://bkt/key: failed: see https://h/x and s3://b/k end"
        );
        // `redact_uri` alone is not enough: that is the bug.
        assert!(redact_uri(text).contains("SECRET1"));
    }

    #[test]
    fn redact_all_uris_handles_urls_in_quotes_parentheses_and_brackets() {
        let out = redact_all_uris(
            "e: \"https://user:SECRET1@h/x?X-Amz-Signature=SIG1\" (s3://b/k?sig=SIG2), \
             [https://h/y?sig=SIG3]; <https://u2:SECRET2@h2/z>.",
        );
        assert_clean(&out);
        assert_eq!(
            out,
            // `)` `]` `;` `,` are not boundaries: the word after a cut is consumed (a signature
            // may hold them), so `), ` and `]; ` go with the query.
            "e: \"https://h/x\" (s3://b/k [https://h/y <https://h2/z>."
        );
    }

    #[test]
    fn redact_all_uris_keeps_text_that_is_not_a_credential_uri() {
        for text in [
            "",
            "no uri here, only a@b and photo@2x.png: done.",
            "s3://bucket/dir/key.bin",
            "see (s3://bucket/dir/key.bin).",
            "a ://b and :// alone and x:/y",
            "file:///tmp/a@b",
            "http://host:9000/key@v1",
            "ключ 日本 é s3://b/ключ ok",
        ] {
            assert_eq!(redact_all_uris(text), text, "{text:?}");
        }
        // Trailing prose punctuation is kept when nothing was removed.
        assert_eq!(redact_all_uris("at s3://b/k."), "at s3://b/k.");
    }

    #[test]
    fn redact_all_uris_passwords_may_hold_terminator_characters() {
        // A quote / bracket inside the userinfo does not end the URI before the `@`.
        let out = redact_all_uris("x https://user:SEC)R'ET1@h/x?sig=SIG1 y");
        assert_eq!(out, "x https://h/x y");
        // A quote in the path does not hide a later signature.
        let out = redact_all_uris("x s3://b/k'q?sig=SIG2 y");
        assert_clean(&out);
        assert_eq!(out, "x s3://b/k'q y");
        // Two URIs glued by a delimiter, no space between.
        let out = redact_all_uris(
            "{\"a\":\"https://h/a?sig=SIG1\",\"b\":\"https://user:SECRET2@h2/b?sig=SIG2\"}",
        );
        assert_clean(&out);
        assert!(
            out.contains("https://h/a") && out.contains("https://h2/b"),
            "{out}"
        );
    }

    #[test]
    fn redact_all_uris_handles_a_uri_inside_a_query_and_odd_schemes() {
        let out = redact_all_uris("go https://h/redirect?next=https://u:SECRET1@h2/p?sig=SIG1 now");
        assert_clean(&out);
        assert!(out.starts_with("go https://h/redirect"), "{out}");
        assert!(out.ends_with(" now"), "{out}");
        let out = redact_all_uris("abfss://tok:SECRET1@acct.dfs/x and az://c/b?sv=1&sig=SIG1");
        assert_clean(&out);
        assert_eq!(out, "abfss://acct.dfs/x and az://c/b");
        // Uppercase markers and schemes.
        let out = redact_all_uris("HTTPS://U:SECRET1@H/X?X-AMZ-SIGNATURE=SIG1");
        assert_clean(&out);
    }

    #[test]
    fn redact_all_uris_is_safe_on_hostile_text() {
        // Prefixes of a hostile string never panic (char boundaries, odd delimiters).
        let hostile = "é://é\u{202e}a://@@://:// s3://@/ \u{2028}http://u:p@/ ünï://ç?sig= \
                       https://:@:// ((((s3://b/k?sig=)))) 日本://x \
                       https://[é:@[,\"@]://[ http://u:p@[::1\"}@ä,ü]:9\"x?#(ö)sig=ß,é";
        for end in (0..=hostile.len()).filter(|&i| hostile.is_char_boundary(i)) {
            let _ = redact_all_uris(&hostile[..end]);
            let _ = redact_and_escape_text(&hostile[..end]);
        }
        // Many URIs and long runs stay fast (linear): this would time out if quadratic.
        let many = "s3://b/k?sig=SIG1 ".repeat(20_000);
        assert!(!redact_all_uris(&many).contains("SIG1"));
        let glued = "u://".repeat(50_000);
        let _ = redact_all_uris(&glued);
        let quotes = format!("https://h/{}?sig=SIG1", "'".repeat(100_000));
        assert!(!redact_all_uris(&quotes).contains("SIG1"));
    }

    #[test]
    fn redact_and_escape_text_redacts_every_uri_and_escapes_control_text() {
        let text =
            "s3://b/k\tx: GET failed: denied https://admin:SECRET1@h/p?X-Amz-Signature=SIG1 \
                    and (s3://o/k?sig=SIG2) \u{202e}end\nnext\u{85}line";
        let out = redact_and_escape_text(text);
        assert_clean(&out);
        assert_eq!(
            out,
            "s3://b/k\\tx: GET failed: denied https://h/p and (s3://o/k \\u{202e}end\\nnext\\u{85}line"
        );
        assert!(!out.chars().any(|c| c.is_control() || c == '\u{202e}'));
        // A control character inside the userinfo does not stop the redaction.
        let out = redact_and_escape_text("https://u:SEC\u{1b}RET1@h/x?sig=SIG1");
        assert_clean(&out);
        assert_eq!(out, "https://h/x");
    }

    /// Review S1: a web URI whose path holds a terminator character (or whose host is an IPv6
    /// literal) still loses its query: an unlisted credential parameter must not survive.
    #[test]
    fn redact_all_uris_cuts_web_queries_behind_path_terminators_and_ipv6_hosts() {
        for (input, want) in [
            ("https://[::1]:9000/x?apikey=SECRET", "https://[::1]:9000/x"),
            (
                "https://h/photo(1).jpg?apikey=SECRET",
                "https://h/photo(1).jpg",
            ),
            (
                "https://h/it's.jpg?apikey=SECRET rest",
                "https://h/it's.jpg rest",
            ),
            ("https://u:S@h/a(1).jpg?apikey=SECRET", "https://h/a(1).jpg"),
            ("https://h/a}b?password=SECRET", "https://h/a}b"),
            ("https://h/a<b?auth=SECRET", "https://h/a<b"),
            (
                "http://u:S@[::1]:9000/b/k?apikey=SECRET",
                "http://[::1]:9000/b/k",
            ),
            ("https://[::1]:9000?apikey=SECRET", "https://[::1]:9000"),
            ("s3://[::1]:9000/b/k?sig=SECRET", "s3://[::1]:9000/b/k"),
            // The `)` behind the query is part of the removed word (no bracket is a boundary).
            ("x (https://h/p(1)?apikey=SECRET) y", "x (https://h/p(1) y"),
            // Nothing to cut: the text is untouched, however odd the path.
            ("https://[::1]:9000/x", "https://[::1]:9000/x"),
            (
                "see https://h/photo(1).jpg, ok",
                "see https://h/photo(1).jpg, ok",
            ),
        ] {
            let out = redact_all_uris(input);
            assert!(!out.contains("SECRET"), "{input:?} -> {out:?}");
            assert_eq!(out, want, "input {input:?}");
        }
    }

    /// Review S2: a cut inside a signature must not leave the tail of that signature behind.
    #[test]
    fn redact_all_uris_does_not_leak_the_tail_of_a_cut_signature() {
        for (input, want) in [
            ("https://h/x?sig=AB)CD", "https://h/x"),
            ("https://h/x?sig=AB]CD ok", "https://h/x ok"),
            ("s3://b/k?sig=AB,CD;EF end", "s3://b/k end"),
            ("(s3://b/k?sig=AB)CD)", "(s3://b/k"),
            // No bracket is a boundary: the `)` that closes the text before the URI goes with
            // the query (it may be part of the signature). A quote IS a boundary.
            ("(s3://b/k?sig=AB)", "(s3://b/k"),
            ("(s3://b/k?sig=AB).", "(s3://b/k"),
            ("\"https://h/x?sig=AB\",", "\"https://h/x\","),
        ] {
            let out = redact_all_uris(input);
            assert!(
                !out.contains("CD") && !out.contains("AB"),
                "{input:?} -> {out:?}"
            );
            assert_eq!(out, want, "input {input:?}");
        }
        // Documented limit: a quote INSIDE a signature (not at the start of the query) ends
        // the removed part; real signatures are percent-encoded and never hold one.
        assert_eq!(redact_all_uris("s3://b/k?sig=AB'CD"), "s3://b/k'CD");
        assert_eq!(
            redact_all_uris("s3://b/k?sig=AB\"CD,EF end"),
            "s3://b/k\"CD,EF end"
        );
    }

    /// Review S3, changed by the simple authority rule: an `@` behind whitespace belongs to
    /// other text and the host is not rewritten; WITHOUT whitespace, `/`, `?` or `#` in between
    /// the `@` is taken for userinfo (the accepted over-redaction, see
    /// `redact_all_uris_over_redacts_a_bare_host_followed_by_an_at`).
    #[test]
    fn redact_all_uris_does_not_rewrite_a_host_for_an_at_behind_a_boundary() {
        for text in [
            "{'e': 'https://s3.example.com', 'o': 'bob@example.com'}",
            "endpoint=https://h user=a@b",
            "endpoint=https://h/x,user=a@b",
            "{\"e\":\"https://h/x\",\"o\":\"bob@example.com\"}",
        ] {
            assert_eq!(redact_all_uris(text), text, "{text:?}");
        }
        // Was unchanged before the simple rule: now the host and what is in front of the `@`
        // go (accepted trade-off).
        for (input, want) in [
            ("endpoint=https://h,user=a@b", "endpoint=https://b"),
            ("[\"https://h\",\"a@b\"]", "[\"https://b\"]"),
            ("{e=https://h}a@b", "{e=https://b"),
            ("endpoint=https://h:9000,user=a@b", "endpoint=https://b"),
            ("endpoint=https://[::1]:9000,user=a@b", "endpoint=https://b"),
        ] {
            assert_eq!(redact_all_uris(input), want, "input {input:?}");
        }
        // Passwords may hold the separators.
        for (input, want) in [
            ("https://user:pa,ss@h/x", "https://h/x"),
            ("https://user:pa'ss@h/x", "https://h/x"),
            ("x https://user:SEC)R'ET1@h/x?sig=SIG1 y", "x https://h/x y"),
            ("https://tok@h,other=a@b", "https://b"),
        ] {
            let out = redact_all_uris(input);
            assert!(!out.contains("pa,ss") && !out.contains("SEC"), "{out:?}");
            assert_eq!(out, want, "input {input:?}");
        }
    }

    /// Review N1: the separator between two glued URIs stays.
    #[test]
    fn redact_all_uris_keeps_the_separator_between_glued_uris() {
        for (input, want) in [
            (
                "a,https://u:S@h/x;https://v:S2@h2/y",
                "a,https://h/x;https://h2/y",
            ),
            ("https://u:S@h/x,s3://v:S2@h2/y", "https://h/x,s3://h2/y"),
            ("see https://u:S@h/x.", "see https://h/x."),
            (
                "https://u:S@h/x;https://h2/y?sig=Z",
                "https://h/x;https://h2/y",
            ),
        ] {
            let out = redact_all_uris(input);
            assert!(!out.contains(":S"), "{out:?}");
            assert_eq!(out, want, "input {input:?}");
        }
    }

    /// Review N4: the text of the typed layer appears once, whatever its place in the chain.
    #[test]
    fn sanitized_error_says_the_typed_layer_once() {
        use crate::object_error::{
            attach_object_error, object_error, ObjectError, ObjectErrorKind,
        };
        let count = |hay: &str| hay.matches("s3 forbidden").count();
        // The typed error is the ROOT of the chain (a bare error, no source under it).
        let root = anyhow::Error::new(ObjectError::new(
            ObjectErrorKind::Forbidden,
            "s3",
            "HTTP 403 AccessDenied",
        ));
        let clean = sanitized_error(&root);
        let shown = format!("{clean:#}");
        assert_eq!(shown, "s3 forbidden: HTTP 403 AccessDenied");
        assert_eq!(count(&shown), 1);
        let t = object_error(&clean).expect("still typed");
        assert_eq!((t.kind, t.provider), (ObjectErrorKind::Forbidden, "s3"));
        // The same root under context.
        let wrapped = root.context("outer https://u:SECRET2@h2/x");
        let clean = sanitized_error(&wrapped);
        let shown = format!("{clean:#}");
        assert_clean(&shown);
        assert_eq!(count(&shown), 1, "{shown}");
        assert!(shown.contains("HTTP 403 AccessDenied") && shown.contains("outer https://h2/x"));
        assert!(object_error(&clean).is_some());
        // A typed layer in the middle of a chain keeps its message and its classification.
        let middle = attach_object_error(
            anyhow::anyhow!("denied: https://h/p?sig=SIG1"),
            ObjectErrorKind::Forbidden,
            "s3",
            "HTTP 403",
        )
        .context("outer");
        let clean = sanitized_error(&middle);
        let shown = format!("{clean:#}");
        assert_clean(&shown);
        assert_eq!(count(&shown), 1, "{shown}");
        assert_eq!(shown, "s3 forbidden: HTTP 403: outer: denied: https://h/p");
        assert!(object_error(&clean).is_some());
    }

    #[test]
    fn sanitized_error_cleans_the_chain_and_keeps_the_classification() {
        use crate::object_error::{attach_object_error, object_error, ObjectErrorKind};
        let leaky = "denied: https://admin:SECRET1@h/p?X-Amz-Signature=SIG1 and s3://o/k?sig=SIG2";
        // Untyped: text cleaned, context kept.
        let plain = anyhow::anyhow!(leaky.to_string()).context("s3://b/dl/: LIST failed\u{202e}");
        let clean = sanitized_error(&plain);
        let shown = format!("{clean:#}");
        assert_clean(&shown);
        assert_eq!(
            shown,
            "s3://b/dl/: LIST failed\\u{202e}: denied: https://h/p and s3://o/k"
        );
        assert!(object_error(&clean).is_none());
        // Typed: classification and its message survive, once.
        let typed = attach_object_error(
            anyhow::anyhow!(leaky.to_string()),
            ObjectErrorKind::Forbidden,
            "s3",
            "HTTP 403 AccessDenied",
        );
        let clean = sanitized_error(&typed);
        let shown = format!("{clean:#}");
        assert_clean(&shown);
        let t = object_error(&clean).expect("still typed");
        assert_eq!((t.kind, t.provider), (ObjectErrorKind::Forbidden, "s3"));
        assert_eq!(
            shown,
            "s3 forbidden: HTTP 403 AccessDenied: denied: https://h/p and s3://o/k"
        );
        // A typed error that is not the top layer keeps its classification too.
        let under = typed.context("outer https://u:SECRET2@h2/x");
        let clean = sanitized_error(&under);
        assert_clean(&format!("{clean:#}"));
        assert!(object_error(&clean).is_some());
    }

    // ---- the simple, linear free-text rules (issue: adversarial leaks) ----

    /// A password may start with, or hold, any character that is not whitespace or `/`:
    /// everything up to the LAST `@` of the userinfo span is userinfo.
    #[test]
    fn redact_all_uris_removes_passwords_that_start_with_or_hold_separators() {
        for (input, want) in [
            ("https://u:,S@h/x", "https://h/x"),
            ("https://u:\"S@h/x", "https://h/x"),
            ("https://u:12'S@h/x", "https://h/x"),
            ("https://user:1234}S@h/x", "https://h/x"),
            ("https://u:a:,S@h/x", "https://h/x"),
            ("https://u:]S@h/x", "https://h/x"),
            ("https://u:12\"SEKRET@h/x", "https://h/x"),
            ("https://u:a:,SEKRET@h/x", "https://h/x"),
            ("s3://u:,S@h/x", "s3://h/x"),
            ("s3://u:\"S@h/x", "s3://h/x"),
            ("abfss://u:,S@h/x", "abfss://h/x"),
            // The password starts with `#`: the userinfo holds a `#`, so the text behind its
            // `@` is the fragment and goes with it.
            ("https://usr:##+$:]S@h", "https://"),
            // IPv6 host behind the userinfo, signature in the query.
            ("https://u:S@[::1]:9000/x?sig=S2", "https://[::1]:9000/x"),
            ("s3://u:,S@[::1]:9000/b/k?sig=S2", "s3://[::1]:9000/b/k"),
            // Surrounding text is kept.
            ("x (https://u:,S@h/x) y", "x (https://h/x) y"),
            ("url=https://u:\"S@h/x, next", "url=https://h/x, next"),
        ] {
            let out = redact_all_uris(input);
            assert!(!out.contains('S'), "{input:?} -> {out:?}");
            assert_eq!(out, want, "input {input:?}");
        }
    }

    /// A query that starts with a bracket or a quote is still the query (the trimmed `?`
    /// of the old rules left it behind).
    #[test]
    fn redact_all_uris_cuts_queries_that_start_with_punctuation() {
        for (input, want) in [
            ("https://h/x?)S", "https://h/x"),
            ("https://h/x?)SEKRET", "https://h/x"),
            ("https://h/x?\"S", "https://h/x"),
            ("https://h/x?'S", "https://h/x"),
            ("https://h/x?]S ok", "https://h/x ok"),
            ("https://h/x#}S ok", "https://h/x ok"),
            ("s3://b/k?sig=)S", "s3://b/k"),
            ("s3://b/k?\"sig=S", "s3://b/k"),
        ] {
            let out = redact_all_uris(input);
            assert!(!out.contains('S'), "{input:?} -> {out:?}");
            assert_eq!(out, want, "input {input:?}");
        }
    }

    /// A quote or bracket in the path does not hide the query behind it.
    #[test]
    fn redact_all_uris_cuts_the_query_behind_a_quote_in_the_path() {
        for (input, want) in [
            (
                "https://h/it's.jpg?apikey=SEKRET rest",
                "https://h/it's.jpg rest",
            ),
            ("s3://b/k'q?sig=SEKRET y", "s3://b/k'q y"),
            ("https://h/a<b?auth=SEKRET", "https://h/a<b"),
            ("https://h/a\"b?token=SEKRET", "https://h/a\"b"),
        ] {
            let out = redact_all_uris(input);
            assert!(!out.contains("SEKRET"), "{input:?} -> {out:?}");
            assert_eq!(out, want, "input {input:?}");
        }
    }

    /// Compact JSON and XML keep their other fields: only the credential-bearing part of the
    /// URI goes; the text after the structural boundary is kept.
    #[test]
    fn redact_all_uris_keeps_the_rest_of_compact_json_and_xml() {
        for (input, want) in [
            (
                r#"{"url":"https://h/b?a=1","code":"X","message":"boom"}"#,
                r#"{"url":"https://h/b","code":"X","message":"boom"}"#,
            ),
            (
                r#"{"url":"https://u:SEKRET@h/b?sig=SEKRET2","code":"X","message":"boom"}"#,
                r#"{"url":"https://h/b","code":"X","message":"boom"}"#,
            ),
            (
                r#"{"url":"s3://u:SEKRET@h/b/k?X-Amz-Signature=SEKRET2","code":"X"}"#,
                r#"{"url":"s3://h/b/k","code":"X"}"#,
            ),
            (
                "{'url': 'https://h/b?sig=SEKRET', 'code': 'X'}",
                "{'url': 'https://h/b', 'code': 'X'}",
            ),
            (
                "<Error><Code>X</Code><Url>https://u:SEKRET@h/b?X-Amz-Signature=SEKRET2</Url>\
                 <Message>boom</Message></Error>",
                "<Error><Code>X</Code><Url>https://h/b</Url><Message>boom</Message></Error>",
            ),
            (
                "<Url>https://u:SEKRET@h</Url><Message>boom</Message>",
                "<Url>https://h</Url><Message>boom</Message>",
            ),
            // Nothing to remove: the text is returned as it was.
            (
                r#"{"url":"https://h/b","code":"X","message":"boom"}"#,
                r#"{"url":"https://h/b","code":"X","message":"boom"}"#,
            ),
            (
                "<Url>s3://b/k</Url><Email>bob@example.com</Email>",
                "<Url>s3://b/k</Url><Email>bob@example.com</Email>",
            ),
        ] {
            let out = redact_all_uris(input);
            assert!(!out.contains("SEKRET"), "{input:?} -> {out:?}");
            assert_eq!(out, want, "input {input:?}");
        }
    }

    /// An AWS-style `<Message>` that quotes two URLs loses both signatures and keeps the prose.
    #[test]
    fn redact_all_uris_handles_an_aws_message_with_two_urls() {
        let text = "<Error><Code>AccessDenied</Code><Message>Denied for \
                    https://u:SEKRET@h/a?X-Amz-Signature=SEKRET2 while copying to \
                    s3://b/k?X-Amz-Signature=SEKRET3&x=1 (retry)</Message></Error>";
        let out = redact_all_uris(text);
        assert!(!out.contains("SEKRET"), "{out}");
        assert_eq!(
            out,
            "<Error><Code>AccessDenied</Code><Message>Denied for https://h/a while copying to \
             s3://b/k (retry)</Message></Error>"
        );
        // Without whitespace between the URLs.
        let out = redact_all_uris("<M>https://h/a?sig=SEKRET2<N>s3://u:SEKRET@h/b</N></M>");
        assert!(!out.contains("SEKRET"), "{out}");
        assert_eq!(out, "<M>https://h/a<N>s3://h/b</N></M>");
    }

    /// Ordinary diagnostics survive: an `@` in a path, a schemeless `photo@2x.png`, an e-mail
    /// address that is not behind a scheme.
    #[test]
    fn redact_all_uris_keeps_ordinary_diagnostics() {
        for text in [
            "s3://b/a@b.txt",
            "failed s3://b/a@b.txt: not found",
            "photo@2x.png",
            "bob@example.com",
            "contact bob@example.com about s3://b/k, then a@b",
            "s3://b/photo@2x.png and https://h:9000/key@v1",
            "https://[::1]:9000/key@v1",
            "s3://b/what?.csv",
            "az://acct/c/o?sv=2022",
            "s3://b/k?signal=1",
        ] {
            assert_eq!(redact_all_uris(text), text, "{text:?}");
        }
    }

    /// ACCEPTED trade-off: a bare-host URI directly followed by text that holds an `@` and no
    /// whitespace, `/`, `?` or `#` is taken for `host,user=a@` userinfo and over-redacted.
    #[test]
    fn redact_all_uris_over_redacts_a_bare_host_followed_by_an_at() {
        assert_eq!(
            redact_all_uris("endpoint=https://h,user=a@b"),
            "endpoint=https://b"
        );
        assert_eq!(
            redact_all_uris("{\"e\":\"https://h\",\"o\":\"bob@example.com\"}"),
            "{\"e\":\"https://example.com\"}"
        );
    }

    /// A URI nested in the kept part of another one is still redacted.
    #[test]
    fn redact_all_uris_redacts_a_nested_uri_in_a_kept_query() {
        let out = redact_all_uris("s3://b/k?next=https://u:SEKRET@h2/p end");
        assert!(!out.contains("SEKRET"), "{out}");
        assert_eq!(out, "s3://b/k?next=https://h2/p end");
        // A credential parameter anywhere in the query removes all of it, nested URI included.
        let out = redact_all_uris("s3://b/k?next=https://u:SEKRET@h2/p?sig=SEKRET2 end");
        assert!(!out.contains("SEKRET"), "{out}");
        assert_eq!(out, "s3://b/k end");
    }

    /// A credentialed URI that FOLLOWS another URI in the same whitespace-free word, and carries
    /// its own `?` / `#`, is still redacted: the cut of the first URI's query keeps the path
    /// span in front of it, and that span is redacted by the same rules.
    #[test]
    fn redact_all_uris_redacts_a_uri_that_follows_another_before_a_cut() {
        let cases: [(&str, &str, &[&str]); 7] = [
            (
                r#"["https://a/x","https://u:SECRETPW@b/y?z=1"]"#,
                r#"["https://a/x","https://b/y"]"#,
                &["SECRETPW", "PW@"],
            ),
            (
                "urls=https://a/x,https://u:SECRETPW@b/y?z=1",
                "urls=https://a/x,https://b/y",
                &["SECRETPW", "PW@"],
            ),
            (
                "https://proxy.example/https://u:SECRETPW@h/x?y=1",
                "https://proxy.example/https://h/x",
                &["SECRETPW", "PW@"],
            ),
            (
                "https://h/x'y,https://v:SECRET2@g/y?z",
                "https://h/x'y,https://g/y",
                &["SECRET2"],
            ),
            (
                "https://h/a|http://SECRETTOK@h/b#f",
                "https://h/a|http://h/b",
                &["SECRETTOK"],
            ),
            (
                "s3://a/x,https://u:PW@h/y?sig=S",
                "s3://a/x,https://h/y",
                &["PW@"],
            ),
            // Three levels: a non-web URI whose marker cut fires, a web URI in its path that
            // cuts at a `#`, and a credentialed URI in front of that `#`.
            (
                "s3://a/x,https://h/y,https://u:SECRETPW@c/z#f?sig=S",
                "s3://a/x,https://h/y,https://c/z",
                &["SECRETPW", "PW@"],
            ),
        ];
        for (input, expected, secrets) in cases {
            let out = redact_all_uris(input);
            for secret in secrets {
                assert!(!out.contains(secret), "`{secret}` survived in {out:?}");
            }
            assert_eq!(out, expected, "input {input:?}");
        }
    }

    /// The first `?` (web: `?` / `#`) after URI A can lie inside the USERINFO of a LATER URI B
    /// glued into the same whitespace word. The mark belongs to B, so A must not cut there: B's
    /// own iteration removes its userinfo (and the query behind it).
    #[test]
    fn redact_all_uris_does_not_cut_a_at_the_query_char_of_a_later_uris_userinfo() {
        let cases: [(&str, &str); 7] = [
            ("https://h/p,https://u:SECRET?w@h/x", "https://h/p,https://"),
            (
                r#"{"a":"https://h/p","b":"https://u:SECRET?w@h/x"}"#,
                r#"{"a":"https://h/p","b":"https://"#,
            ),
            (
                r##"{"a":"https://h","b":"s3://u:SECRET#w@h/x"}"##,
                r##"{"a":"https://h","b":"s3://h/x"}"##,
            ),
            (
                "https://h/p,https://u:SECRET#frag@h/x",
                "https://h/p,https://",
            ),
            ("https://h/x\"https://u:SECRET?w@h", "https://h/x\"https://"),
            ("https://h/p;http://SECRET:p#@h", "https://h/p;http://"),
            ("s3://b/k,https://u:SECRET?w@h/x?sig=1", "s3://b/k,https://"),
        ];
        for (input, expected) in cases {
            let out = redact_all_uris(input);
            assert!(!out.contains("SECRET"), "`SECRET` survived in {out:?}");
            assert_eq!(out, expected, "input {input:?}");
        }
        // A `://` with no scheme in front of it is no URI start: the cut still fires.
        assert_eq!(redact_all_uris("https://h/1://x?sig=S"), "https://h/1://x");
        // Documented limit: a password holding both `?` / `#` and a `/` is not removed whole.
        assert_eq!(redact_all_uris("https://u:S?w/x@h/p"), "https://u:S");
    }

    /// A query / fragment cut ends at the next structural boundary; when that boundary lies in a
    /// LATER URI glued into the same whitespace-free word, the removed region holds a `://` and
    /// the cut extends to the end of the word, so the later URI's tail is not left as text.
    #[test]
    fn redact_all_uris_does_not_leak_a_later_uri_behind_a_cut() {
        let cases: [(&str, &str, &[&str]); 5] = [
            (
                "https://a/x?y=1,https://h/it's?sig=S",
                "https://a/x",
                &["sig=S", "=S", "it's"],
            ),
            (
                "https://a/x?y=1,https://h/a\"b?sig=S",
                "https://a/x",
                &["sig=S", "=S", "a\"b"],
            ),
            (
                "s3://b/k?sig=S,https://u:pa'ssWORD@h/z",
                "s3://b/k",
                &["ssWORD", "pa'", "WORD"],
            ),
            (
                "https://a/x?y=1,https://u:pa'ssWORD@h/z",
                "https://a/x",
                &["ssWORD", "pa'", "WORD"],
            ),
            (
                "https://a/x?y=1,https://u:pa\"ssWORD@h/z",
                "https://a/x",
                &["ssWORD", "pa\"", "WORD"],
            ),
        ];
        for (input, expected, secrets) in cases {
            let out = redact_all_uris(input);
            for secret in secrets {
                assert!(!out.contains(secret), "`{secret}` survived in {out:?}");
            }
            assert_eq!(out, expected, "input {input:?}");
        }
        // The rest of the word goes, the text behind the whitespace stays.
        assert_eq!(
            redact_all_uris("https://a/x?y=1,https://h/it's?sig=S next=1"),
            "https://a/x next=1"
        );
        // Documented trade-off: a cut query that holds a nested URI removes the rest of the word.
        assert_eq!(
            redact_all_uris(r#"{"u":"https://h/b?next=https://x/y","c":"X"}"#),
            r#"{"u":"https://h/b"#
        );
    }

    /// Rule 4 (a password holding ONE `/`): the first path segment ends at whitespace, `/`, `?`
    /// or `#`, so an `@` in a query or fragment is not taken for the end of the userinfo.
    #[test]
    fn rule_4_segment_ends_at_a_query_or_fragment() {
        for (input, want) in [
            ("https://h:x/p?e=a@b&X-Amz-Signature=SIG", "https://h:x/p"),
            ("https://h:x/p?e=a@b&sig=S", "https://h:x/p"),
            ("https://h:x/p#a@b&sig=S", "https://h:x/p"),
            ("https://h:x/?@a=1&b=SEC", "https://h:x/"),
        ] {
            for out in [redact_all_uris(input), redact_uri(input)] {
                for secret in ["SIG", "sig=S", "SEC", "a=1", "e=a"] {
                    assert!(
                        !out.contains(secret),
                        "`{secret}` in {out:?} from {input:?}"
                    );
                }
                assert_eq!(out, want, "input {input:?}");
            }
        }
        // The real case of rule 4 still works, and a port is still a port.
        for (input, want) in [
            ("https://u:p/w@h/x", "https://h/x"),
            ("https://h:9000/p?e=a@b&sig=S", "https://h:9000/p"),
        ] {
            for out in [redact_all_uris(input), redact_uri(input)] {
                assert!(!out.contains("u:p/w"), "{out:?} from {input:?}");
                assert_eq!(out, want, "input {input:?}");
            }
        }
    }

    /// The userinfo span (for the malformed `https:/` forms, the first segment) ends at the
    /// first `/`. A `?` / `#` inside the removed userinfo makes the text behind its `@` a
    /// query / fragment that is removed with it, so no credential query is left behind.
    /// `redact_uri` and `redact_all_uris` agree.
    #[test]
    fn authority_ends_at_a_query_or_fragment_for_redact_uri_and_redact_all_uris() {
        let mut leaks = Vec::new();
        for input in [
            "https:/h?e=a@b&sig=S",
            "https:///p?e=a@b&sig=S",
            "https:////x#e=a@b&token=S",
            "https:/h#e=a@b&sig=S",
            "https:///?x=a@b&X-Amz-Signature=S",
            "https://h?e=a@b&sig=S",
            "https://h:80?e=a@b&sig=S",
            "https://h#e=a@b&sig=S",
            "https://u:S@h?e=a@b&sig=T",
            "https://h?e=a@b&sig=S/z",
        ] {
            // `redact_all_uris` looks for `://` only, so it does not see the `https:/h` form.
            let all = if input.contains("://") {
                redact_all_uris(input)
            } else {
                redact_uri(input)
            };
            for (name, out) in [("redact_uri", redact_uri(input)), ("redact_all_uris", all)] {
                for secret in ["sig=S", "sig=T", "token=S", "Signature=S", "u:S"] {
                    if out.contains(secret) {
                        leaks.push(format!("{name}: `{secret}` in {out:?} from {input:?}"));
                    }
                }
            }
        }
        assert!(leaks.is_empty(), "leaks:\n{}", leaks.join("\n"));
        // The two redactors agree on these inputs: a `?` / `#` in the userinfo span makes the
        // text behind the last `@` a query / fragment, which goes with the userinfo.
        for (input, want) in [
            ("https://h?e=a@b&sig=S", "https://"),
            ("https://u:S@h?e=a@b&sig=T", "https://"),
            ("https://h:80?e=a@b&sig=S", "https://"),
            ("https://h#e=a@b&sig=S", "https://"),
            ("s3://h?e=a@b&sig=S", "s3://"),
            ("https://user:p?w@host/x?sig=1", "https://"),
            ("https://user:p#w@host/x", "https://"),
            ("s3://user:p?w@host/b/k?x", "s3://host/b/k?x"),
            ("s3://u:ab?c@h/k", "s3://h/k"),
            ("https://u:ab?c@h", "https://"),
            ("https://u:p@h/x?sig=1", "https://h/x"),
            ("https://h?e=1", "https://h"),
            ("https://h/x?e=a@b&sig=S", "https://h/x"),
        ] {
            assert_eq!(redact_uri(input), want, "redact_uri {input:?}");
            assert_eq!(redact_all_uris(input), want, "redact_all_uris {input:?}");
        }
        // The malformed forms exist for `redact_uri` only.
        for input in [
            "https:/h?e=a@b&sig=S",
            "https:///p?e=a@b&sig=S",
            "https:////x#e=a@b&token=S",
        ] {
            let out = redact_uri(input);
            assert!(
                !out.contains('@') && !out.contains("=S"),
                "{out:?} from {input:?}"
            );
        }
    }

    /// In running text the userinfo span ends at whitespace or `/`; the removed query ends at
    /// the whitespace word end (a quote in the path must not hide the real query), so the rest
    /// of the line stays but the rest of the word (a JSON field's tail) goes.
    #[test]
    fn redact_all_uris_userinfo_with_query_characters_in_text() {
        for (input, want) in [
            ("a https://u:ab?c@h/x?sig=1 b", "a https:// b"),
            ("a s3://u:ab?c@h/k b", "a s3://h/k b"),
            (
                "{\"u\":\"https://u:p#w@h/x\",\"c\":\"X\"}",
                "{\"u\":\"https://",
            ),
            ("x https://h?e=a@b&sig=S next=1", "x https:// next=1"),
        ] {
            assert_eq!(redact_all_uris(input), want, "input {input:?}");
        }
    }

    /// A `?` / `#` in the removed userinfo makes the text behind the `@` the query: it runs to
    /// the end of the whitespace word, so a quote in the PATH cannot hide the real signed query.
    #[test]
    fn redact_all_uris_userinfo_with_query_characters_cuts_to_the_word_end() {
        for (input, want) in [
            ("https://u:p?w@h/it's.jpg?X-Amz-Signature=SEC", "https://"),
            ("https://h?e=a@b/it's?sig=SEC", "https://"),
            ("https://u:p#w@h/it's?sig=SEC", "https://"),
            ("https://a?b@h:80/a<b?sig=SEC", "https://"),
            ("s3://u:p?w@h/it's.csv?token=SEC", "s3://"),
            ("s3://u:p?w@h/\"x?sig=SEC", "s3://"),
            ("s3://u:p?w@h/x>y?sig=SEC", "s3://"),
            ("s3://h?e=a@b/it's?sig=SEC", "s3://"),
        ] {
            assert_eq!(redact_all_uris(input), want, "redact_all_uris {input:?}");
            assert_eq!(redact_uri(input), want, "redact_uri {input:?}");
        }
        // The word end stops at whitespace: the text behind it stays.
        assert_eq!(
            redact_all_uris("a https://u:p?w@h/it's?sig=SEC b"),
            "a https:// b"
        );
    }

    /// Extra slashes after `://` do not hide the userinfo of a web URI.
    #[test]
    fn redact_all_uris_removes_userinfo_behind_extra_slashes() {
        for (input, want) in [
            ("https:///u:SEC@h/x", "https:///h/x"),
            ("HTTPS:////u:SEC@h/x?sig=SEC", "HTTPS:////h/x"),
        ] {
            assert_eq!(redact_all_uris(input), want, "redact_all_uris {input:?}");
            assert_eq!(redact_uri(input), want, "redact_uri {input:?}");
            assert!(!want.contains("SEC") && !want.contains("u:SEC"));
        }
        // Non-web schemes keep their slashes and paths (`file:///tmp/a@b`).
        assert_eq!(redact_all_uris("file:///tmp/a@b"), "file:///tmp/a@b");
    }

    /// Run `f` on a thread and fail if it takes 2 s or more in this (debug) build, or does not
    /// finish within 20 s (a quadratic rule would not).
    fn assert_fast(what: &str, f: impl FnOnce() + Send + 'static) {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let t = std::time::Instant::now();
            f();
            let _ = tx.send(t.elapsed());
        });
        let elapsed = rx
            .recv_timeout(std::time::Duration::from_secs(20))
            .unwrap_or_else(|_| panic!("{what}: still running after 20 s (quadratic?)"));
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "{what}: {elapsed:?}"
        );
    }

    #[test]
    fn redact_all_uris_is_linear_on_adversarial_text() {
        const N: usize = 500_000; // about 1 MB per input
        assert_fast("numeric-led password then separators", || {
            let text = format!("https://u:{}a{}", "1".repeat(N), ",".repeat(N));
            assert_eq!(redact_all_uris(&text), text);
            let with_host = format!("{text}@h/x");
            assert_eq!(redact_all_uris(&with_host), "https://h/x");
        });
        assert_fast("at-bracket runs", || {
            let text = format!("https://{}", "@[".repeat(N));
            let out = redact_all_uris(&text);
            assert_eq!(out, "https://[");
        });
        assert_fast("scheme-like runs", || {
            let text = "a://".repeat(N / 2);
            let _ = redact_all_uris(&text);
        });
        assert_fast("path then scheme-like runs", || {
            let text = format!("https://h/{}", "a://".repeat(N / 2));
            let _ = redact_all_uris(&text);
        });
        assert_fast("many non-web URIs sharing one later query mark", || {
            // No cut fires (no marker, not a web scheme): nothing is consumed, so a rescan per
            // URI would be quadratic. Nothing is redacted, so the output is the input.
            let text = format!("{}?{}", "s3://a/".repeat(N / 10), "x".repeat(N));
            assert_eq!(redact_all_uris(&text), text);
        });
        assert_fast("nested kept queries", || {
            let text = "s3://b/?".repeat(N / 4);
            assert_eq!(redact_all_uris(&text), text);
        });
        assert_fast("nested kept queries with a marker at the end", || {
            let text = format!("{}sig=S", "s3://b/?".repeat(N / 4));
            assert_eq!(redact_all_uris(&text), "s3://b/");
        });
        assert_fast("many query marks, one slash segment each", || {
            let text = format!("{}x", "https://u:a/".repeat(N / 12));
            let _ = redact_all_uris(&text);
        });
        assert_fast("userinfo-looking segments", || {
            let text = format!("https://u:a/{}", "@/".repeat(N));
            let _ = redact_all_uris(&text);
        });
        assert_fast("quotes and brackets", || {
            let path = "'\"<>)]}".repeat(N / 7);
            let text = format!("https://h/{path}?sig=S");
            assert_eq!(redact_all_uris(&text), format!("https://h/{path}"));
        });
        assert_fast("credentialed URIs nested in a path before a cut", || {
            let text = format!("{}?x", "https://a/https://u:p@b/".repeat(N / 24));
            let out = redact_all_uris(&text);
            assert!(!out.contains("u:p@"), "userinfo survived");
            assert_eq!(out, "https://a/https://b/".repeat(N / 24));
        });
        assert_fast("non-web, web and fragment levels before a cut", || {
            let text = format!("{}#f?sig=S", "s3://a/https://u:p@b/".repeat(N / 22));
            let out = redact_all_uris(&text);
            assert!(!out.contains("u:p@"), "userinfo survived");
        });
        assert_fast(
            "cut query then a later URI with a boundary in its path",
            || {
                let text = "https://a/x?y=1,https://h/it's".repeat(N / 30);
                let out = redact_all_uris(&text);
                assert!(!out.contains("it's"), "tail survived");
            },
        );
        assert_fast("userinfo holding a query character, repeated", || {
            let text = "s3://u:a?b@h/k,".repeat(N * 2 / 15);
            let out = redact_all_uris(&text);
            assert!(!out.contains("u:a"), "userinfo survived");
            assert_eq!(out, "s3://h/k,".repeat(N * 2 / 15));
        });
        assert_fast(
            "userinfo holding a query character, no marker, quote in path",
            || {
                let text = "s3://u:a?b@h/it's,".repeat(N * 2 / 15);
                let out = redact_all_uris(&text);
                assert_eq!(out, "s3://h/it's,".repeat(N * 2 / 15));
            },
        );
        assert_fast(
            "web URI then a later URI whose userinfo holds a query char",
            || {
                let text = "https://h/p,https://u:S?w@h/x,".repeat(N / 30);
                let out = redact_all_uris(&text);
                assert!(!out.contains("u:S"), "userinfo survived");
            },
        );
        assert_fast("escape and redact together", || {
            let text = "s3://u:p@h/\n?sig=1 ".repeat(N / 20);
            let _ = redact_and_escape_text(&text);
        });
    }
}
