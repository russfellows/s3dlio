// tests/python_future_bodies_no_attach_302.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2026 Russ Fellows <russ.fellows@gmail.com>

//! Structural guard for issue #302 (std only: no Python, no pyo3, no `extension-module` feature).
//!
//! The body of a `future_into_py` / `guarded_put` call runs on a Tokio worker, AHEAD of the exit
//! gate of `python_api/async_bridge.rs`. If it attaches to Python itself (`Python::attach`,
//! `Python::with_gil`, `Python::try_attach`) it can do so while the interpreter is finalizing, and on
//! CPython 3.14 that corrupts finalization (SIGSEGV, "double free or corruption", "Fatal Python
//! error: PyInterpreterState_Delete: remaining threads"). A body must return a plain Rust value (any
//! `IntoPyObject`, e.g. `PyBytesView`); PyO3 converts it inside the gated delivery.
//!
//! This test scans every `future_into_py(` / `guarded_put(` call site in the production part of the
//! Python API sources and fails if one contains such an attach. The behavioural check is the
//! `python/tests/test_exit_daemon_threads_298.py` scenarios.

use std::path::PathBuf;

/// `src` with `//` comments and string-literal contents blanked out (same length), so a scan for
/// code patterns and balanced parentheses cannot be fooled by prose.
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

/// The production part of a python_api source file: everything before its first test module.
fn production_part(src: &str) -> &str {
    let end = src
        .find("\n#[cfg(test)]\nmod ")
        .expect("a test module marks the end of the production code");
    &src[..end]
}

fn read(name: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "src", "python_api", name]
        .iter()
        .collect();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Every call site's argument span (text between the call's parentheses) in `code`.
fn call_spans<'a>(code: &'a str, call: &str) -> Vec<&'a str> {
    let mut spans = Vec::new();
    let mut from = 0;
    while let Some(rel) = code[from..].find(call) {
        let start = from + rel;
        let open = start + call.len();
        from = open;
        // Skip the definition (`fn future_into_py(`), only call sites matter.
        if code[..start].trim_end().ends_with("fn") {
            continue;
        }
        let mut depth = 1usize;
        let mut end = None;
        for (k, ch) in code[open..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(open + k);
                        break;
                    }
                }
                _ => {}
            }
        }
        spans.push(&code[open..end.expect("unbalanced parentheses at a call site")]);
    }
    spans
}

#[test]
fn future_bodies_never_attach_to_python() {
    let mut checked = 0;
    for name in [
        "python_core_api.rs",
        "python_aiml_api.rs",
        "python_advanced_api.rs",
    ] {
        let raw = read(name);
        let code = blank_comments_and_strings(production_part(&raw));
        for call in ["future_into_py(", "guarded_put("] {
            for span in call_spans(&code, call) {
                checked += 1;
                for banned in ["Python::attach", "Python::with_gil", "Python::try_attach"] {
                    assert!(
                        !span.contains(banned),
                        "RED (#302): a `{call}` body in {name} calls {banned}. Return a plain Rust \
                         value (an `IntoPyObject` type such as `PyBytesView`) and let the gated \
                         delivery build the Python object.\n{span}"
                    );
                }
            }
        }
    }
    // The scan must really have seen the call sites (a blanking bug must not make it vacuous).
    assert!(checked >= 25, "only {checked} future bodies were scanned");
}

#[test]
fn the_scanner_catches_an_attaching_body() {
    // Mutation check of the guard itself.
    let bad = "fn f() { future_into_py(py, async move { let x = work().await; Python::attach(|py| x) }) }";
    let blanked = blank_comments_and_strings(bad);
    let spans = call_spans(&blanked, "future_into_py(");
    assert_eq!(spans.len(), 1);
    assert!(spans[0].contains("Python::attach"));
    // Prose in comments and strings is ignored.
    let ok = "fn f() { future_into_py(py, async move { // Python::attach would be wrong\n Ok(\"Python::attach\") }) }";
    let blanked = blank_comments_and_strings(ok);
    let spans = call_spans(&blanked, "future_into_py(");
    assert_eq!(spans.len(), 1);
    assert!(!spans[0].contains("Python::attach"));
}
