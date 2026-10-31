// tests/python_future_bodies_no_attach_302.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2026 Russ Fellows <russ.fellows@gmail.com>

//! Structural barrier for issues #302 and #304 (std only: no Python, no pyo3, no feature flag).
//!
//! The rule (`src/python_api/async_bridge.rs`, module docs): a thread that is not a Python thread
//! must never attach to Python on its own, because on CPython 3.14 a thread state created while the
//! interpreter is finalizing corrupts finalization (#302), and every call into Python that an async
//! entry point makes from Rust must sit inside an exit-gate scope (#304). So the places that attach
//! or re-take the GIL are an explicit, small allow-list, each marked in the source with
//! `// exit-gate: allowed (<reason>)` on the line or within the three lines above it.
//!
//! The scan covers the PRODUCTION part of EVERY `src/**/*.rs` file, not just the arguments of
//! `future_into_py`, so a helper function, a future built outside the call, an alias, a raw FFI call
//! or a call split over lines is caught wherever it is. Whitespace is ignored when matching, comments
//! and string literals are blanked. It is a lexical barrier (it cannot follow a call into another crate
//! or a macro); the behavioural barrier is `python/tests/test_exit_daemon_threads_298.py`.

use std::fs;
use std::path::{Path, PathBuf};

/// Whitespace-free spellings that must not appear in production code outside the allow-list.
const BANNED: &[&str] = &[
    "::attach(",
    "::with_gil(",
    "::try_attach(",
    "attach_unchecked",
    "assume_attached",
    "assume_gil_acquired",
    "PyGILState_Ensure",
    "PyEval_RestoreThread",
    "PyEval_AcquireThread",
    "PyEval_AcquireLock",
    "PyThreadState_New",
    "PyThreadState_Swap",
    // An alias would hide `Python::attach` behind another name.
    "Pythonas",
    "pyo3::Pythonas",
    // The bridge's own entry point is the only way to build an awaitable (it holds the gate).
    "generic::future_into_py",
    "tokio::future_into_py",
    "::into_future",
    "::into_future_with_locals",
    "future_into_py_with_locals",
];

/// `src` with `//` comments and string/char-literal contents blanked out (same length, newlines kept).
fn blank_comments_and_strings(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let (mut i, n) = (0, b.len());
    while i < n {
        if b[i] == b'/' && i + 1 < n && b[i + 1] == b'/' {
            while i < n && b[i] != b'\n' {
                out[i] = b' ';
                i += 1;
            }
        } else if b[i] == b'"' {
            i += 1;
            while i < n && b[i] != b'"' {
                if b[i] == b'\\' && i + 1 < n {
                    out[i] = b' ';
                    i += 1;
                }
                if b[i] != b'\n' {
                    out[i] = b' ';
                }
                i += 1;
            }
            i += 1;
        } else {
            i += 1;
        }
    }
    String::from_utf8(out).expect("blanking ASCII bytes keeps the text valid UTF-8")
}

/// Production part of a source file: everything before its first `#[cfg(test)] mod`.
fn production_part(src: &str) -> &str {
    let mut from = 0;
    while let Some(rel) = src[from..].find("#[cfg(") {
        let at = from + rel;
        let Some(close) = src[at..].find(']') else {
            break;
        };
        let attr = &src[at..at + close + 1];
        from = at + 1;
        // `#[cfg(test)]`, `#[cfg(all(test, feature = ..))]`, ...
        if !attr.contains("test") || attr.contains("not(test") {
            continue;
        }
        // Skip further attributes, then look for `mod` / `pub mod`.
        let mut rest = src[at + close + 1..].trim_start();
        while rest.starts_with("#[") {
            match rest.find(']') {
                Some(e) => rest = rest[e + 1..].trim_start(),
                None => break,
            }
        }
        let rest = rest.strip_prefix("pub ").unwrap_or(rest);
        if rest.starts_with("mod ") {
            return &src[..at];
        }
    }
    src
}

/// Unmarked occurrences of a banned spelling in `raw`: `(1-based line, spelling)`.
/// Also returns the number of marked (allowed) occurrences.
fn scan(raw: &str) -> (Vec<(usize, &'static str)>, usize) {
    let prod = production_part(raw);
    let code = blank_comments_and_strings(prod);
    // Whitespace-free copy with a map back to byte offsets in `code`.
    let mut squeezed = String::new();
    let mut map = Vec::new();
    for (i, ch) in code.char_indices() {
        if !ch.is_whitespace() {
            squeezed.push(ch);
            for _ in 0..ch.len_utf8() {
                map.push(i);
            }
        }
    }
    let lines: Vec<&str> = prod.split('\n').collect();
    let line_of = |off: usize| code[..off].matches('\n').count() + 1;
    let marked = |line: usize| {
        let lo = line.saturating_sub(4);
        (lo..line).any(|l| {
            lines
                .get(l)
                .is_some_and(|t| t.contains("exit-gate: allowed") && t.contains('('))
        })
    };
    let (mut bad, mut ok) = (Vec::new(), 0);
    for &tok in BANNED {
        let mut from = 0;
        while let Some(rel) = squeezed[from..].find(tok) {
            let at = from + rel;
            from = at + 1;
            let line = line_of(map[at]);
            if marked(line) {
                ok += 1;
            } else {
                bad.push((line, tok));
            }
        }
    }
    bad.sort();
    (bad, ok)
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display())) {
        let p = e.expect("dir entry").path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn only_allow_listed_code_attaches_to_python() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rs_files(&root, &mut files);
    assert!(files.len() > 50, "scanned only {} files", files.len());
    let (mut allowed_total, mut report) = (0, String::new());
    for f in &files {
        let raw = fs::read_to_string(f).unwrap_or_else(|e| panic!("read {}: {e}", f.display()));
        let (bad, ok) = scan(&raw);
        allowed_total += ok;
        for (line, tok) in bad {
            report += &format!(
                "  {}:{line}: `{tok}`\n",
                f.strip_prefix(&root).unwrap_or(f).display()
            );
        }
    }
    assert!(
        report.is_empty(),
        "RED (#302/#304): attach / GIL re-take / awaitable construction outside the exit gate.\n\
         A thread that is not a Python thread must never attach on its own (CPython 3.14 \
         corrupts finalization), and a Python call made from Rust needs a `python_call_scope()`. \
         Return a plain Rust value from the future (an `IntoPyObject`) and use `future_into_py`; \
         if this is a genuinely safe site (the gated delivery, a sync pyfunction on a Python thread), \
         mark it `// exit-gate: allowed (reason)`.\n{report}"
    );
    // Non-vacuous: the known allow-listed sites were really seen.
    assert!(
        allowed_total >= 5,
        "only {allowed_total} allow-listed sites seen"
    );
}

// ---- mutation checks of the scanner itself (the reviewer's mutations M2 to M7 and a few more) ----

fn flagged(src: &str) -> bool {
    !scan(src).0.is_empty()
}

#[test]
fn the_scanner_catches_the_known_bypasses() {
    // M1: inline attach in a body.
    assert!(flagged(
        "fn f(){ future_into_py(py, async move { Python::attach(|py| 1) }) }"
    ));
    // M2: a helper function that attaches (the body only calls it).
    assert!(flagged("fn h(b:Bytes)->PyResult<Py<PyAny>>{ Python::attach(|py| Ok(Py::new(py,V(b))?.into_any())) }\nfn f(){ future_into_py(py, async move { h(b) }) }"));
    // M3: the future is built outside the call and passed by name.
    assert!(flagged(
        "fn f(){ let task = async move { Python::attach(|py| 1) }; seams.bridge(py, task) }"
    ));
    // M4: an alias.
    assert!(flagged(
        "use pyo3::Python as P;\nfn f(){ P::attach(|py| 1) }"
    ));
    assert!(flagged("use pyo3::marker::Python as Py3;"));
    // M5: raw FFI.
    assert!(flagged(
        "fn f(){ unsafe { pyo3::ffi::PyGILState_Ensure() }; }"
    ));
    assert!(flagged(
        "fn f(){ unsafe { ffi::PyEval_RestoreThread(t) }; }"
    ));
    // M6: split over lines.
    assert!(flagged("fn f(){ Python\n    ::attach(|py| 1) }"));
    assert!(flagged("fn f(){ Python :: with_gil(|py| 1) }"));
    // M7: spawn_blocking helper.
    assert!(flagged(
        "async fn h(){ tokio::task::spawn_blocking(|| Python::attach(|py| 1)).await; }"
    ));
    // Building an awaitable without the bridge (no gate scope, issue #304).
    assert!(flagged(
        "fn f(){ pyo3_async_runtimes::tokio::future_into_py(py, fut) }"
    ));
    assert!(flagged(
        "fn f(){ pyo3_async_runtimes::generic::future_into_py::<R,_,T>(py, fut) }"
    ));
    // try_attach counts too.
    assert!(flagged("fn f(){ Python::try_attach(|py| 1); }"));
}

#[test]
fn the_scanner_accepts_marked_sites_and_ignores_prose_and_tests() {
    let marked = "// exit-gate: allowed (sync pyfunction on a Python thread)\nfn f(){ Python::attach(|py| 1) }";
    let (bad, ok) = scan(marked);
    assert!(bad.is_empty() && ok == 1, "{bad:?} {ok}");
    // A marker without a reason does not count.
    assert!(flagged(
        "// exit-gate: allowed\nfn f(){ Python::attach(|py| 1) }"
    ));
    // A marker too far above does not count.
    assert!(flagged(
        "// exit-gate: allowed (x)\n\n\n\n\nfn f(){ Python::attach(|py| 1) }"
    ));
    // Comments and string literals are prose.
    assert!(!flagged(
        "// Python::attach is banned\nfn f(){ let s = \"Python::attach\"; }"
    ));
    // Test modules are not production code.
    assert!(!flagged(
        "fn p(){}\n#[cfg(test)]\nmod tests { fn t(){ Python::attach(|py| 1); } }"
    ));
    assert!(!flagged("fn p(){}\n#[cfg(all(test, feature = \"x\"))]\nmod tests { fn t(){ Python::attach(|py| 1); } }"));
    assert!(!flagged(
        "fn p(){}\n#[cfg(test)]\n#[allow(unused)]\nmod tests { fn t(){ Python::attach(|py| 1); } }"
    ));
    // But a `#[cfg(test)]` fn is not a module: code after it is still production.
    assert!(flagged(
        "#[cfg(test)]\nfn helper(){}\nfn p(){ Python::attach(|py| 1); }"
    ));
}
