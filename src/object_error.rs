// src/object_error.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Typed object errors shared by the caller-buffer GET and typed HEAD paths
// (plan docs/implementation-plans/v0.10.0-caller-buffer-get-plan.md, §4.5 A2).
//
// Design
// ──────
// * An [`ObjectError`] is attached as an `anyhow` *context* over the retained
//   source error: `source.context(ObjectError { .. })`.  The SDK error stays
//   recoverable for diagnostics (`err.downcast_ref::<SdkFailure>()`, the Azure
//   and GCS typed errors, ...), but the **outermost** classification is
//   authoritative: a `VersionChanged` attached over an SDK 404 stays
//   `VersionChanged`.
// * Lookup is [`object_error`], i.e. `err.downcast_ref::<ObjectError>()`.
//   anyhow's downcast is context-aware and finds a context payload at any
//   depth, nearest-the-top first.  Walking `err.chain()` would NOT find it,
//   because a context layer appears there as anyhow's private `ContextError`
//   type.  Every consumer (`RetryClassify`, `delete_contract::is_not_found`,
//   the Python exception mapping) uses this one lookup; a source-chain walk
//   remains only as the fallback for errors that carry no `ObjectError`.
// * The message inside an `ObjectError` is built by the attaching site from
//   the HTTP status and a sanitised service code only ([`sanitize_service_code`]:
//   `[A-Za-z0-9._-]`, at most 64 characters), never from the source's own text.
//   That covers the `ObjectError`'s own `Display` and `Debug`.  It does NOT
//   redact the retained source error: printing the chain (`{:#}`, `{:?}`) also
//   prints the source, which this module does not touch.  For Azure the source
//   is the already-redacted `AzureRequestError`; for S3 and GCS it is the
//   flattened SDK text or the gax error as the caller produced it.

use std::fmt;

/// What kind of failure an object operation reported.
///
/// The kind is a *semantic* classification, independent of the provider SDK.
/// New kinds may be added, so a match outside this crate needs a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ObjectErrorKind {
    /// The object does not exist (first request, or a typed HEAD).
    NotFound,
    /// The received length does not match the expected length.
    /// `actual` is `None` when it is unknown.
    LengthMismatch { expected: u64, actual: Option<u64> },
    /// The object changed (or disappeared) after the read pinned its version.
    VersionChanged,
    /// The operation is not supported by this store or build.
    Unsupported,
    /// The service refused the request (authorization, not credential expiry).
    Forbidden,
    /// The service asked the caller to slow down or try again later.
    Throttled,
    /// No usable response arrived (connect, timeout, I/O, body stream).
    Transport,
    /// The service failed (5xx and transient service codes).
    Server,
    /// Anything else, including definite client errors with no dedicated kind.
    Other,
}

impl ObjectErrorKind {
    /// Short stable name, used in the `Display` text.
    pub fn name(&self) -> &'static str {
        match self {
            Self::NotFound => "not found",
            Self::LengthMismatch { .. } => "length mismatch",
            Self::VersionChanged => "version changed",
            Self::Unsupported => "unsupported",
            Self::Forbidden => "forbidden",
            Self::Throttled => "throttled",
            Self::Transport => "transport failure",
            Self::Server => "server failure",
            Self::Other => "error",
        }
    }

    /// The kind to report for a failure on a read whose version is already pinned.
    ///
    /// A pinned read never reports absence: once an ETag or generation is pinned,
    /// a "not found" on a later retry, range or repair means the pinned version
    /// is gone, which is [`ObjectErrorKind::VersionChanged`].  `NotFound` comes
    /// only from the first request, or from HEAD.  Every other kind is unchanged.
    pub fn for_pinned_read(self) -> Self {
        match self {
            Self::NotFound => Self::VersionChanged,
            other => other,
        }
    }
}

/// A typed object error, attached as an `anyhow` context (see the module docs).
///
/// `message` is private.  The attaching helpers build it from the HTTP status and a
/// sanitised service code only (see [`sanitize_service_code`]); it holds no URL, token
/// or source error text.  That is a statement about this message alone: the source error
/// retained under the context is not redacted by this module.
pub struct ObjectError {
    /// The classification.
    pub kind: ObjectErrorKind,
    /// The provider that produced the failure (`"s3"`, `"az"`, `"gs"`, ...).
    pub provider: &'static str,
    message: String,
}

impl ObjectError {
    /// Build an error.  `message` must contain no secrets (no URLs, tokens or
    /// source error text); the attaching helpers pass the status and a sanitised
    /// code only.
    pub fn new(kind: ObjectErrorKind, provider: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            provider,
            message: message.into(),
        }
    }
}

impl fmt::Display for ObjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.provider, self.kind.name())?;
        if let ObjectErrorKind::LengthMismatch { expected, actual } = self.kind {
            match actual {
                Some(actual) => write!(f, " (expected {expected} bytes, got {actual})")?,
                None => write!(f, " (expected {expected} bytes, actual size unknown)")?,
            }
        }
        if !self.message.is_empty() {
            write!(f, ": {}", self.message)?;
        }
        Ok(())
    }
}

impl fmt::Debug for ObjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectError")
            .field("kind", &self.kind)
            .field("provider", &self.provider)
            .field("message", &self.message)
            .finish()
    }
}

impl std::error::Error for ObjectError {}

/// The typed [`ObjectError`] an error carries, if any.
///
/// This is `err.downcast_ref::<ObjectError>()`: anyhow's context-aware lookup,
/// which finds the payload at any depth and returns the **outermost** one when
/// several are attached.  It deliberately does not walk `err.chain()`.
pub fn object_error(err: &anyhow::Error) -> Option<&ObjectError> {
    err.downcast_ref::<ObjectError>()
}

/// The longest service error code (in characters) kept in an [`ObjectError`] message.
const MAX_SERVICE_CODE_CHARS: usize = 64;

/// A server-supplied service error code made safe to put in an [`ObjectError`] message.
///
/// Keeps only `[A-Za-z0-9._-]` and at most 64 characters, so a hostile or malformed
/// response cannot inject newlines, control characters or an unbounded string into an error
/// message.  Real S3 and Azure codes (`NoSuchKey`, `BlobNotFound`, `Throttling`) pass
/// unchanged.  `None` when nothing is left.
pub(crate) fn sanitize_service_code(code: &str) -> Option<String> {
    let kept: String = code
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        .take(MAX_SERVICE_CODE_CHARS)
        .collect();
    (!kept.is_empty()).then_some(kept)
}

/// The [`ObjectError`] message for a failure with an optional HTTP `status` and optional
/// service `code`: `status 404, code NoSuchKey`, `status 404`, `code Foo` or
/// `no HTTP response`.  The code goes through [`sanitize_service_code`].
pub(crate) fn status_code_message(status: Option<u16>, code: Option<&str>) -> String {
    match (status, code.and_then(sanitize_service_code)) {
        (Some(s), Some(c)) => format!("status {s}, code {c}"),
        (Some(s), None) => format!("status {s}"),
        (None, Some(c)) => format!("code {c}"),
        (None, None) => "no HTTP response".to_string(),
    }
}

/// Attach a classification over `source`, keeping `source` recoverable.
///
/// The returned error's top-level text is the [`ObjectError`]'s; the source's
/// own message stays one layer down (`{:#}` prints both).
///
/// The `ObjectError`'s message is whatever the caller passes; the source is kept as is
/// and is not redacted here.
pub fn attach_object_error(
    source: anyhow::Error,
    kind: ObjectErrorKind,
    provider: &'static str,
    message: impl Into<String>,
) -> anyhow::Error {
    source.context(ObjectError::new(kind, provider, message))
}

/// Attach a classification over `source` for a read whose version is already
/// pinned: a `NotFound` becomes `VersionChanged` (see
/// [`ObjectErrorKind::for_pinned_read`]).
pub fn attach_pinned_read_error(
    source: anyhow::Error,
    kind: ObjectErrorKind,
    provider: &'static str,
    message: impl Into<String>,
) -> anyhow::Error {
    attach_object_error(source, kind.for_pinned_read(), provider, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[derive(Debug)]
    struct SdkSource;
    impl fmt::Display for SdkSource {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("sdk said 404")
        }
    }
    impl std::error::Error for SdkSource {}

    fn nested(kind: ObjectErrorKind) -> anyhow::Error {
        let sdk = anyhow::Error::new(SdkSource);
        let typed = attach_object_error(sdk, kind, "s3", "status 404 code NoSuchKey");
        Err::<(), _>(typed)
            .context("ranged read of part 3")
            .context("caller-buffer GET failed")
            .unwrap_err()
    }

    /// The typed context is found inside nested string contexts, and the
    /// preserved SDK source is still recoverable below it.
    #[test]
    fn typed_context_found_through_string_contexts_and_source_survives() {
        let e = nested(ObjectErrorKind::NotFound);
        let o = object_error(&e).expect("typed context");
        assert_eq!(o.kind, ObjectErrorKind::NotFound);
        assert_eq!(o.provider, "s3");
        assert!(
            e.downcast_ref::<SdkSource>().is_some(),
            "source recoverable"
        );
        let full = format!("{e:#}");
        assert!(full.contains("caller-buffer GET failed"), "{full}");
        assert!(full.contains("status 404 code NoSuchKey"), "{full}");
        assert!(full.contains("sdk said 404"), "{full}");
    }

    /// `chain()` never yields the context payload; the downcast does.  This is
    /// why the lookup rule is `downcast_ref`, not a chain walk.
    #[test]
    fn chain_walk_does_not_find_the_context_but_downcast_does() {
        let e = nested(ObjectErrorKind::Forbidden);
        assert!(e.chain().all(|c| c.downcast_ref::<ObjectError>().is_none()));
        assert!(object_error(&e).is_some());
    }

    /// The outermost classification is authoritative: a `VersionChanged`
    /// attached over an error that already carries `NotFound` wins, and the
    /// inner classification is not what the lookup reports.
    #[test]
    fn outer_classification_is_authoritative() {
        let inner = nested(ObjectErrorKind::NotFound);
        let outer = attach_object_error(inner, ObjectErrorKind::VersionChanged, "s3", "pinned");
        let outer = Err::<(), _>(outer).context("outer string").unwrap_err();
        assert_eq!(
            object_error(&outer).expect("typed").kind,
            ObjectErrorKind::VersionChanged
        );
        // Both remain recoverable in the chain text; the SDK source too.
        assert!(outer.downcast_ref::<SdkSource>().is_some());
    }

    #[test]
    fn plain_errors_carry_no_object_error() {
        assert!(object_error(&anyhow::anyhow!("boom")).is_none());
        assert!(object_error(&anyhow::Error::new(SdkSource)).is_none());
    }

    /// A pinned read never reports absence.
    #[test]
    fn pinned_read_maps_not_found_to_version_changed_only() {
        use ObjectErrorKind::*;
        assert_eq!(NotFound.for_pinned_read(), VersionChanged);
        for k in [
            VersionChanged,
            Unsupported,
            Forbidden,
            Throttled,
            Transport,
            Server,
            Other,
            LengthMismatch {
                expected: 1,
                actual: None,
            },
        ] {
            assert_eq!(k.for_pinned_read(), k);
        }
        let e =
            attach_pinned_read_error(anyhow::Error::new(SdkSource), NotFound, "s3", "status 404");
        assert_eq!(object_error(&e).expect("typed").kind, VersionChanged);
    }

    #[test]
    fn display_names_length_and_never_contains_source_text() {
        let e = attach_object_error(
            anyhow::Error::new(SdkSource),
            ObjectErrorKind::LengthMismatch {
                expected: 10,
                actual: Some(7),
            },
            "az",
            "",
        );
        let o = object_error(&e).expect("typed");
        let text = o.to_string();
        assert!(
            text.contains("expected 10") && text.contains("got 7"),
            "{text}"
        );
        assert!(!text.contains("sdk said"), "{text}");
    }

    #[test]
    fn object_error_is_send_sync_static_error() {
        fn assert_traits<T: std::error::Error + Send + Sync + 'static>() {}
        assert_traits::<ObjectError>();
    }

    /// Newlines, control characters and spaces are dropped; a long code is cut to 64.
    #[test]
    fn server_codes_are_sanitised_before_they_reach_a_message() {
        assert_eq!(
            sanitize_service_code("NoSuchKey").as_deref(),
            Some("NoSuchKey")
        );
        assert_eq!(
            sanitize_service_code("Blob.Not_Found-1").as_deref(),
            Some("Blob.Not_Found-1")
        );
        let hostile = "Bad\r\nInjected: header\x1b[31m \u{202e}x";
        assert_eq!(
            sanitize_service_code(hostile).as_deref(),
            Some("BadInjectedheader31mx")
        );
        let long = "A".repeat(500);
        assert_eq!(sanitize_service_code(&long).map(|s| s.len()), Some(64));
        assert_eq!(sanitize_service_code("\n\t "), None);
        assert_eq!(sanitize_service_code(""), None);

        // The message built from them carries none of it.
        let msg = status_code_message(Some(400), Some(hostile));
        assert_eq!(msg, "status 400, code BadInjectedheader31mx");
        assert!(!msg.contains('\n') && !msg.contains('\x1b'), "{msg:?}");
        let msg = status_code_message(Some(400), Some(&long));
        assert_eq!(msg.len(), "status 400, code ".len() + 64);
        assert_eq!(status_code_message(None, None), "no HTTP response");
        assert_eq!(status_code_message(Some(503), None), "status 503");
        assert_eq!(status_code_message(None, Some("X")), "code X");
        // A code that is all junk is as good as absent.
        assert_eq!(status_code_message(Some(500), Some("\n\n")), "status 500");
    }
}
