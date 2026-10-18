// src/redact.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Credential redaction for URIs that are shown in error text and log lines.
//!
//! One implementation for every message that quotes a URI: the Python batch errors and the
//! download code (issue #284) both call [`redact_uri`]. Ungated on purpose: the messages exist
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
/// * The authority ends at the first `/` after `://` (not at `?` or `#`: a password may hold
///   those). Userinfo is everything up to the LAST `@` inside the authority and is removed.
///   `abfs://` / `abfss://` keep their `container@account` (a container is not userinfo) ONLY
///   when the part before the `@` holds no `:`; `token:secret@account` is userinfo and is removed.
/// * `http(s)` only: an authority without `@` that looks like `user:password` (a `:` followed
///   by something that is not a port number) while the first path segment holds an `@` is a
///   password that contained a `/` (`https://user:p/w@host/x`): everything up to that `@` is
///   removed. A numeric password is indistinguishable from a port, and a password holding two
///   or more `/` is not caught either (documented limits). Other schemes are left alone:
///   `s3://b/key@v1`, `file://localhost:abc/a@b` and `http://host:9000/key@v1` are untouched.
/// * Malformed slashes after `http(s):` (`https:/u:SECRET@h/`, `https:///u:SECRET@h/`): any
///   `@` in the first `/`-separated segment ends the userinfo, as above (a password holding a
///   `/` is not caught here). Other schemes are not treated this way (`file:///tmp/a@b`).
/// * `http(s)`: `?query` / `#fragment` are cut. Every other scheme keeps its path verbatim, `?`
///   being an ordinary key character in `s3://`, `gs://`, `az://`, ..., EXCEPT when the text
///   after the first `?` holds a credential parameter ([`CREDENTIAL_QUERY_MARKERS`], matched
///   case-insensitively anywhere in that text, so `sv=...&sig=...` is covered by `sig=` and a
///   SigV2 presigned `...&Signature=...` by `signature=`): the
///   URI is then cut at that `?`. Trade-off: a literal key such as `s3://b/a?x&sig=1` loses its
///   tail in the message, which beats showing a signature; parameters named differently from
///   the markers are not recognised.
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

/// `host/path` split at the first `/`: (`host`, `/path`), or (`host`, `""`).
fn split_authority(rest: &str) -> (&str, &str) {
    match rest.find('/') {
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

fn redact_scheme_uri(scheme: &str, rest: &str) -> String {
    let lower = scheme.to_ascii_lowercase();
    let web = lower == "http" || lower == "https";
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
                let segment_end = segment.find('/').unwrap_or(segment.len());
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
        let query = out[from + q + 1..].to_ascii_lowercase();
        if CREDENTIAL_QUERY_MARKERS.iter().any(|m| query.contains(m)) {
            out.truncate(from + q);
        }
    }
    out
}

/// `http(s):` followed by one slash or by three or more (`sep` is the `:` and the slashes):
/// the userinfo ends at the last `@` of the first `/`-separated segment of `rest`.
fn redact_malformed_web_uri(scheme: &str, sep: &str, rest: &str) -> String {
    let segment_end = rest.find('/').unwrap_or(rest.len());
    let tail = match rest[..segment_end].rfind('@') {
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
            // Passwords that hold `?`, `#` or `/`.
            ("https://user:p?w@host/x?sig=1", "https://host/x"),
            ("https://user:p#w@host/x", "https://host/x"),
            ("https://user:p/w@host/x", "https://host/x"),
            // Only http(s) get the "password holding a slash" heuristic.
            ("s3://user:p/w@host/b/k", "s3://user:p/w@host/b/k"),
            ("s3://user:p?w@host/b/k?x", "s3://host/b/k?x"),
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
}
