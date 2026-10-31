// tests/python_future_bodies_no_attach_302.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2026 Russ Fellows <russ.fellows@gmail.com>

//! Structural barrier for issues #302 and #304 (std only: no Python, no pyo3, no feature flag).
//!
//! PARTIAL BY DESIGN: this is a lexical scan, not a type check. It cannot follow a call into another
//! crate or a macro, and it does not know which values are `PyErr`s. The behavioural barrier is
//! `python/tests/test_exit_daemon_threads_298.py` (every async entry point exits under load).
//!
//! Rules (all over the PRODUCTION part of every `src/**/*.rs`; comments and string literals blanked):
//!
//! 1. No attach, anywhere. The identifiers `attach`, `try_attach`, `with_gil`, `attach_unchecked`,
//!    `assume_attached`, `assume_gil_acquired`, raw `PyGILState_Ensure` / `PyEval_RestoreThread` / ...,
//!    an alias of `Python`, and building an awaitable without the bridge (`generic::future_into_py`,
//!    `tokio::future_into_py`, `into_future*`) are banned unless the line, or the nearest code line
//!    above it, carries `// exit-gate: allowed (<reason>)` (a marker covers ONE line). A whole-identifier
//!    match also catches `use ..::attach;`, `use ..::attach as a`, a bare `attach(..)`, a turbofish
//!    `attach::<..>(..)` and `.map(Python::attach)`.
//! 2. A future body must not format or unwrap a possibly-`PyErr` value. `Display`, `Debug` and
//!    `to_string()` of a `PyErr` attach to Python inside PyO3 (`err/mod.rs`), which is exactly #302 with
//!    no banned token in the source. Inside every `async` block / `async fn` and every
//!    `future_into_py(..)` / `guarded_put(..)` argument list the scan bans `to_string()`, `format!`,
//!    `write!`, the print/log/panic macros and `unwrap` / `expect` (which Debug-format the error). Applies to
//!    `src/python_api/*.rs`; an `async` block there that is not an argument of those calls is checked when it
//!    mentions `PyErr`, `py_err` or `PyResult` (a pure-Rust producer task is not a future body).
//!    (Making bodies return a non-`PyErr` error type, converted in the gated delivery, was evaluated and is
//!    too invasive: about 100 `map_err(py_err)` sites share a helper with the synchronous API.)
//! 3. (#304) A function that calls into Python from Rust through the awaitable machinery
//!    (`create_future`, `get_running_loop`, `add_done_callback`, the guard's `self.class.call1(`,
//!    `from_code(`, `generic::future_into_py`, `call_method0("cancel")`) must call
//!    `python_call_scope()` BEFORE the first such call.

use std::fs;
use std::path::{Path, PathBuf};

const ATTACH_IDENTS: &[&str] = &[
    "attach",
    "try_attach",
    "with_gil",
    "attach_unchecked",
    "assume_attached",
    "assume_gil_acquired",
    "PyGILState_Ensure",
    "PyEval_RestoreThread",
    "PyEval_AcquireThread",
    "PyEval_AcquireLock",
    "PyThreadState_New",
    "PyThreadState_Swap",
    "future_into_py_with_locals",
    "into_future",
    "into_future_with_locals",
];

/// Whitespace-free spellings (paths and aliases).
const SQUEEZED_BANNED: &[&str] = &[
    "Pythonas",
    "generic::future_into_py",
    "tokio::future_into_py",
];

/// Tokens that must not appear inside a future body (rule 2).
const BODY_BANNED: &[&str] = &[
    ".to_string()",
    "format!",
    "write!",
    "writeln!",
    "print!",
    "println!",
    "eprint!",
    "eprintln!",
    "panic!",
    "unreachable!",
    "todo!",
    "debug!",
    "info!",
    "warn!",
    "error!",
    "trace!",
    ".unwrap(",
    ".expect(",
    ".unwrap_err(",
    ".expect_err(",
];

/// Calls into Python made from Rust by the awaitable machinery (rule 3).
const PYTHON_CALLS: &[&str] = &[
    "create_future",
    "get_running_loop",
    "add_done_callback",
    "generic::future_into_py",
    "from_code(",
    "self.class.call1(",
    "call_method0(\"cancel\")",
];

fn blank_comments_and_strings_keep(src: &str, keep_strings: bool) -> String {
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
                    if !keep_strings {
                        out[i] = b' ';
                    }
                    i += 1;
                }
                if b[i] != b'\n' && !keep_strings {
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

/// `src` with `//` comments and string-literal contents blanked out (same length, newlines kept).
fn blank_comments_and_strings(src: &str) -> String {
    blank_comments_and_strings_keep(src, false)
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
        if !attr.contains("test") || attr.contains("not(test") {
            continue;
        }
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

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn line_of(code: &str, off: usize) -> usize {
    code[..off].matches('\n').count() + 1
}

/// A marker covers its own line and the NEXT code line only: the marker is on the same line as the
/// token, or on the nearest non-blank line above it.
fn marked(lines: &[&str], line: usize) -> bool {
    let has = |l: usize| {
        lines
            .get(l)
            .is_some_and(|t| t.contains("exit-gate: allowed") && t.contains('('))
    };
    if has(line - 1) {
        return true;
    }
    let mut l = line - 1;
    while l > 0 {
        l -= 1;
        if !lines[l].trim().is_empty() {
            return has(l);
        }
    }
    false
}

/// Balanced span after an opening delimiter at `open` (the byte after it), up to its closer.
fn balanced(code: &str, open: usize, o: u8, c: u8) -> Option<usize> {
    let b = code.as_bytes();
    let mut depth = 1usize;
    let mut i = open;
    while i < b.len() {
        if b[i] == o {
            depth += 1;
        } else if b[i] == c {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Rule 1: unmarked banned identifiers / spellings. Returns `(line, token)` and the marked count.
fn scan_attach(raw: &str) -> (Vec<(usize, String)>, usize) {
    let prod = production_part(raw);
    let code = blank_comments_and_strings(prod);
    let lines: Vec<&str> = prod.split('\n').collect();
    let b = code.as_bytes();
    let (mut bad, mut ok) = (Vec::new(), 0);
    let mut note = |line: usize, tok: &str| {
        if marked(&lines, line) {
            ok += 1;
        } else {
            bad.push((line, tok.to_string()));
        }
    };
    // Whole identifiers.
    let mut i = 0;
    while i < b.len() {
        if is_ident(b[i]) && (i == 0 || !is_ident(b[i - 1])) {
            let mut j = i;
            while j < b.len() && is_ident(b[j]) {
                j += 1;
            }
            let word = &code[i..j];
            if ATTACH_IDENTS.contains(&word) {
                note(line_of(&code, i), word);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    // Whitespace-free spellings.
    let mut squeezed = String::new();
    let mut map = Vec::new();
    for (k, ch) in code.char_indices() {
        if !ch.is_whitespace() {
            squeezed.push(ch);
            for _ in 0..ch.len_utf8() {
                map.push(k);
            }
        }
    }
    for &tok in SQUEEZED_BANNED {
        let mut from = 0;
        while let Some(rel) = squeezed[from..].find(tok) {
            let at = from + rel;
            from = at + 1;
            note(line_of(&code, map[at]), tok);
        }
    }
    bad.sort();
    (bad, ok)
}

/// Spans of every future body: `async` blocks, `async fn` bodies and the argument lists of
/// `future_into_py(` / `guarded_put(`. `(start, end)` byte offsets into `code`.
fn future_spans(code: &str) -> Vec<(usize, usize, bool, usize)> {
    let b = code.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if is_ident(b[i]) && (i == 0 || !is_ident(b[i - 1])) {
            let mut j = i;
            while j < b.len() && is_ident(b[j]) {
                j += 1;
            }
            let word = &code[i..j];
            if word == "async" {
                // `async move {`, `async {`, `async fn name(..) .. {`, `async |..|` (closure) skipped.
                let rest = code[j..].trim_start();
                let rest_off = code.len() - rest.len();
                let next = rest
                    .split(|c: char| !is_ident(c as u8))
                    .next()
                    .unwrap_or("");
                let brace = if rest.starts_with('{')
                    || next == "move" && rest[4..].trim_start().starts_with('{')
                {
                    code[rest_off..].find('{').map(|p| rest_off + p)
                } else if next == "fn" {
                    // body brace: first `{` after the parameter list closes.
                    code[rest_off..].find('(').and_then(|p| {
                        let open = rest_off + p + 1;
                        balanced(code, open, b'(', b')')
                            .and_then(|close| code[close..].find('{').map(|q| close + q))
                    })
                } else {
                    None
                };
                if let Some(bo) = brace {
                    if let Some(end) = balanced(code, bo + 1, b'{', b'}') {
                        spans.push((bo, end, false, i));
                    }
                }
            } else if (word == "future_into_py" || word == "guarded_put")
                && code[j..].trim_start().starts_with('(')
                && !code[..i].trim_end().ends_with("fn")
            {
                let po = j + code[j..].find('(').expect("paren");
                if let Some(end) = balanced(code, po + 1, b'(', b')') {
                    spans.push((po, end, true, po));
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    spans
}

/// Rule 2: banned formatting / unwrapping inside future bodies.
fn scan_bodies(raw: &str) -> Vec<(usize, String)> {
    let prod = production_part(raw);
    let code = blank_comments_and_strings(prod);
    let lines: Vec<&str> = prod.split('\n').collect();
    let mut bad = Vec::new();
    for (s, e, is_call, sig) in future_spans(&code) {
        let span = &code[s..e];
        // An `async` block / fn outside a `future_into_py(..)` call is a future body only if it deals in
        // PyO3 errors or results (a pure-Rust producer task, formatting its own errors, is not one).
        let whole = &code[sig..e];
        if !is_call
            && !(whole.contains("PyErr") || whole.contains("py_err") || whole.contains("PyResult"))
        {
            continue;
        }
        for &tok in BODY_BANNED {
            let mut from = 0;
            while let Some(rel) = span[from..].find(tok) {
                let at = s + from + rel;
                from += rel + 1;
                let line = line_of(&code, at);
                if !marked(&lines, line) {
                    bad.push((line, tok.to_string()));
                }
            }
        }
    }
    bad.sort();
    bad.dedup();
    bad
}

/// Rule 3: a function that calls into Python through the machinery must hold the scope first.
fn scan_scope(raw: &str) -> Vec<(usize, String)> {
    let prod = production_part(raw);
    let code = blank_comments_and_strings(prod);
    // For `call_method0("cancel")` the string content matters: look in the unblanked text, but only at
    // offsets where the blanked text still has the call (so comments are ignored).
    let keep = blank_comments_and_strings_keep(prod, true);

    let b = code.as_bytes();
    let mut bad = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if is_ident(b[i]) && (i == 0 || !is_ident(b[i - 1])) {
            let mut j = i;
            while j < b.len() && is_ident(b[j]) {
                j += 1;
            }
            if &code[i..j] == "fn" {
                // fn name ( params ) ... {
                if let Some(p) = code[j..].find('(') {
                    let open = j + p + 1;
                    if let Some(close) = balanced(&code, open, b'(', b')') {
                        // the body brace, unless a `;` (trait method without body) comes first
                        let tail = &code[close..];
                        let semi = tail.find(';');
                        let brace = tail.find('{');
                        if let (Some(br), true) = (brace, brace < semi || semi.is_none()) {
                            let bo = close + br;
                            if let Some(end) = balanced(&code, bo + 1, b'{', b'}') {
                                let body = &code[bo..end];
                                let kbody = &keep[bo..end];
                                let scope_at = body.find("python_call_scope()");
                                let mut first: Option<(usize, &str)> = None;
                                for &tok in PYTHON_CALLS {
                                    let hay = if tok.contains('"') { kbody } else { body };
                                    let mut from = 0;
                                    while let Some(rel) = hay[from..].find(tok) {
                                        let at = from + rel;
                                        from = at + 1;
                                        // ignore uses that are only comments in `body`
                                        let head = tok.split('(').next().unwrap_or(tok);
                                        if !body[at..].starts_with(head) {
                                            continue;
                                        }
                                        if first.is_none_or(|(f, _)| at < f) {
                                            first = Some((at, tok));
                                        }
                                        break;
                                    }
                                }
                                if let Some((at, tok)) = first {
                                    let ok = scope_at.is_some_and(|sa| sa < at);
                                    let line = line_of(&code, bo + at);
                                    if !ok {
                                        // (no marker exemption: it would let the `generic::future_into_py` marker of rule 1 hide a missing scope)
                                        bad.push((line, tok.to_string()));
                                    }
                                }
                                // continue scanning INSIDE the body too (nested fns): move past `fn`.
                            }
                        }
                    }
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    bad.sort();
    bad
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

fn all_sources() -> (PathBuf, Vec<(PathBuf, String)>) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rs_files(&root, &mut files);
    assert!(files.len() > 50, "scanned only {} files", files.len());
    let srcs = files
        .into_iter()
        .map(|f| {
            let t = fs::read_to_string(&f).unwrap_or_else(|e| panic!("read {}: {e}", f.display()));
            (f, t)
        })
        .collect();
    (root, srcs)
}

fn report(root: &Path, f: &Path, items: &[(usize, String)], out: &mut String) {
    for (line, tok) in items {
        *out += &format!(
            "  {}:{line}: `{tok}`\n",
            f.strip_prefix(root).unwrap_or(f).display()
        );
    }
}

#[test]
fn only_allow_listed_code_attaches_to_python() {
    let (root, srcs) = all_sources();
    let (mut allowed_total, mut out) = (0, String::new());
    for (f, raw) in &srcs {
        let (bad, ok) = scan_attach(raw);
        allowed_total += ok;
        report(&root, f, &bad, &mut out);
    }
    assert!(
        out.is_empty(),
        "RED (#302/#304): attach / GIL re-take / awaitable construction outside the exit gate.\n\
         A thread that is not a Python thread must never attach on its own (CPython 3.14 corrupts \
         finalization). Return a plain Rust value from the future (an `IntoPyObject`) and use \
         `future_into_py`; if this is a genuinely safe site (the gated delivery, a sync pyfunction on a \
         Python thread), put `// exit-gate: allowed (reason)` on the line or the line above.\n{out}"
    );
    assert!(
        allowed_total >= 5,
        "only {allowed_total} allow-listed sites seen"
    );
}

#[test]
fn future_bodies_do_not_format_or_unwrap() {
    let (root, srcs) = all_sources();
    let (mut spans, mut out) = (0, String::new());
    for (f, raw) in &srcs {
        let prod = production_part(raw);
        if !f.components().any(|c| c.as_os_str() == "python_api") {
            continue; // the rule is about the Python API's futures
        }
        spans += future_spans(&blank_comments_and_strings(prod)).len();
        report(&root, f, &scan_bodies(raw), &mut out);
    }
    assert!(
        out.is_empty(),
        "RED (#302): a future body formats or unwraps a value. If that value is a `PyErr`, PyO3 attaches to \
         Python inside Display/Debug/to_string, on a Tokio worker. Return the error and let the gated \
         delivery convert it; if the value is certainly not a `PyErr`, mark the line \
         `// exit-gate: allowed (reason)`.\n{out}"
    );
    assert!(spans >= 30, "only {spans} future bodies were scanned");
}

#[test]
fn python_calls_from_rust_hold_the_exit_gate_scope() {
    let (root, srcs) = all_sources();
    let mut out = String::new();
    for (f, raw) in &srcs {
        report(&root, f, &scan_scope(raw), &mut out);
    }
    assert!(
        out.is_empty(),
        "RED (#304): a function calls into Python from Rust (create_future / get_running_loop / \
         add_done_callback / the guard constructor / cancel / the guard import) without taking \
         `python_call_scope()` first. On CPython < 3.13.8 a GIL re-take inside that call during \
         finalization aborts the process.\n{out}"
    );
    // Non-vacuous: the bridge's own functions are really seen.
    let bridge = srcs
        .iter()
        .find(|(f, _)| f.ends_with("python_api/async_bridge.rs"))
        .expect("async_bridge.rs");
    assert!(production_part(&bridge.1).contains("python_call_scope()"));
}

// ---- mutation checks of the rules themselves (the reviewer's mutations and more) ----

fn attach_flagged(src: &str) -> bool {
    !scan_attach(src).0.is_empty()
}
fn body_flagged(src: &str) -> bool {
    !scan_bodies(src).is_empty()
}
fn scope_flagged(src: &str) -> bool {
    !scan_scope(src).is_empty()
}

#[test]
fn rule1_catches_the_known_bypasses() {
    assert!(attach_flagged(
        "fn f(){ future_into_py(py, async move { Python::attach(|py| 1) }) }"
    ));
    assert!(attach_flagged("fn h(b:Bytes)->PyResult<Py<PyAny>>{ Python::attach(|py| Ok(Py::new(py,V(b))?.into_any())) }"));
    assert!(attach_flagged(
        "fn f(){ let task = async move { Python::attach(|py| 1) }; seams.bridge(py, task) }"
    ));
    assert!(attach_flagged(
        "use pyo3::Python as P;\nfn f(){ P::attach(|py| 1) }"
    ));
    assert!(attach_flagged("use pyo3::marker::Python as Py3;"));
    assert!(attach_flagged(
        "fn f(){ unsafe { pyo3::ffi::PyGILState_Ensure() }; }"
    ));
    assert!(attach_flagged(
        "fn f(){ unsafe { ffi::PyEval_RestoreThread(t) }; }"
    ));
    assert!(attach_flagged("fn f(){ Python\n    ::attach(|py| 1) }"));
    assert!(attach_flagged("fn f(){ Python :: with_gil(|py| 1) }"));
    assert!(attach_flagged(
        "async fn h(){ tokio::task::spawn_blocking(|| Python::attach(|py| 1)).await; }"
    ));
    assert!(attach_flagged(
        "fn f(){ pyo3_async_runtimes::tokio::future_into_py(py, fut) }"
    ));
    assert!(attach_flagged(
        "fn f(){ pyo3_async_runtimes::generic::future_into_py::<R,_,T>(py, fut) }"
    ));
    assert!(attach_flagged("fn f(){ Python::try_attach(|py| 1); }"));
    // Round 2 holes: use-import then a bare attach, an aliased import, a turbofish, a function path.
    assert!(attach_flagged(
        "use pyo3::marker::Python::attach;\nfn f(){ attach(|py| 1) }"
    ));
    assert!(attach_flagged(
        "use pyo3::marker::Python::attach as a;\nfn f(){ a(|py| 1) }"
    ));
    assert!(attach_flagged(
        "use pyo3::marker::Python::{attach, with_gil};"
    ));
    assert!(attach_flagged("fn f(){ Python::attach::<_, i32>(|py| 1) }"));
    assert!(attach_flagged("fn f(){ attach::<_, i32>(|py| 1) }"));
    assert!(attach_flagged(
        "fn f(){ let g = Python::attach; g(|py| 1) }"
    ));
    assert!(attach_flagged("fn f(){ xs.iter().map(Python::attach) }"));
}

#[test]
fn rule1_markers_cover_exactly_one_line() {
    let (bad, ok) = scan_attach("// exit-gate: allowed (sync pyfunction on a Python thread)\nfn f(){ Python::attach(|py| 1) }");
    assert!(bad.is_empty() && ok == 1, "{bad:?} {ok}");
    // Same-line marker.
    assert!(!attach_flagged(
        "fn f(){ Python::attach(|py| 1) } // exit-gate: allowed (x)"
    ));
    // No reason: not a marker.
    assert!(attach_flagged(
        "// exit-gate: allowed\nfn f(){ Python::attach(|py| 1) }"
    ));
    // A blank line between marker and code: still the nearest non-blank line.
    assert!(!attach_flagged(
        "// exit-gate: allowed (x)\n\nfn f(){ Python::attach(|py| 1) }"
    ));
    // Leakage: code under a marked line is NOT covered (the old window was 4 lines).
    assert!(attach_flagged("// exit-gate: allowed (x)\nfn f(){ Python::attach(|py| 1) }\nfn g(){ Python::attach(|py| 2) }"));
    assert!(attach_flagged(
        "// exit-gate: allowed (x)\nfn f(){\n    Python::attach(|py| 1)\n}"
    ));
    // Prose and tests are not production code.
    assert!(!attach_flagged(
        "// Python::attach is banned\nfn f(){ let s = \"Python::attach\"; }"
    ));
    assert!(!attach_flagged(
        "fn p(){}\n#[cfg(test)]\nmod tests { fn t(){ Python::attach(|py| 1); } }"
    ));
    assert!(!attach_flagged("fn p(){}\n#[cfg(all(test, feature = \"x\"))]\nmod tests { fn t(){ Python::attach(|py| 1); } }"));
    assert!(attach_flagged(
        "#[cfg(test)]\nfn helper(){}\nfn p(){ Python::attach(|py| 1); }"
    ));
    // Identifiers that merely contain the word are fine.
    assert!(!attach_flagged(
        "fn f(){ let releaser_attaches = 1; let n = reattach(); AttachGate::new(); }"
    ));
}

#[test]
fn rule2_catches_pyerr_formatting_in_future_bodies() {
    // M8/M9 of the review: Display / Debug / to_string of a PyErr inside a body.
    assert!(body_flagged("fn f(){ future_into_py(py, async move { let e = work().await.map_err(py_err)?; let m = e.to_string(); Ok(()) }) }"));
    assert!(body_flagged("fn f(){ future_into_py(py, async move { if let Err(e) = r { tracing::warn!(\"{e}\"); } Ok(()) }) }"));
    assert!(body_flagged(
        "fn f(){ future_into_py(py, async move { let s = format!(\"{e:?}\"); Ok(()) }) }"
    ));
    assert!(body_flagged(
        "fn f(){ future_into_py(py, async move { work().await.expect(\"x\"); Ok(()) }) }"
    ));
    assert!(body_flagged(
        "fn f(){ guarded_put(py, latch, move || async move { r.unwrap(); Ok(None) }) }"
    ));
    // A future built outside the call, and an async fn.
    assert!(body_flagged("fn f(){ let task = async move { r.map_err(py_err).unwrap_err().to_string() }; seams.bridge(py, task) }"));
    assert!(body_flagged(
        "async fn quiesce() -> PyResult<()> { println!(\"x\"); Ok(()) }"
    ));
    assert!(body_flagged(
        "pub async fn q(a: A) -> PyResult<B> { let x = e.to_string(); Ok(x) }"
    ));
    // Marked: allowed. Outside a body: fine. Clean bodies: fine.
    assert!(!body_flagged("fn f(){ future_into_py(py, async move {\n // exit-gate: allowed (a String, not a PyErr)\n let s = name.to_string();\n Ok(()) }) }"));
    assert!(!body_flagged(
        "fn f(){ let s = name.to_string(); println!(\"x\"); }"
    ));
    // A pure-Rust producer task (no PyO3 error in it) may format its own errors.
    assert!(!body_flagged(
        "async fn produce() { tracing::info!(\"x\"); let s = e.to_string(); }"
    ));
    assert!(!body_flagged("fn f(){ future_into_py(py, async move { let b = store.get(&u).await.map_err(py_err)?; Ok(PyBytesView::new(b)) }) }"));
}

#[test]
fn rule3_requires_the_scope_before_python_calls() {
    // RED: calls Python through the machinery with no scope, or the scope only after the call.
    assert!(scope_flagged(
        "fn f(py: Python) { let fut = loop_.call_method0(\"create_future\"); create_future(py) }"
    ));
    assert!(scope_flagged(
        "fn f(py: Python) { let l = get_running_loop(py)?; }"
    ));
    assert!(scope_flagged(
        "fn f(py: Python) { fut.add_done_callback(cb); }"
    ));
    assert!(scope_flagged(
        "fn f(py: Python) { let m = PyModule::from_code(py, a, b, c)?; }"
    ));
    assert!(scope_flagged("fn g(&self) { self.class.call1((a, b)) }"));
    assert!(scope_flagged(
        "fn c(f: &Bound<PyAny>) { f.call_method0(\"cancel\"); }"
    ));
    assert!(scope_flagged(
        "fn f(py: Python) { let l = get_running_loop(py)?; let _s = python_call_scope()?; }"
    ));
    assert!(scope_flagged(
        "fn f(py: Python) { pyo3_async_runtimes::generic::future_into_py::<R,_,T>(py, fut) }"
    ));
    // GREEN: the scope comes first (the shape of the bridge, also after the fix/297 merge).
    assert!(!scope_flagged("fn f(py: Python) { let _s = python_call_scope()?; let l = get_running_loop(py)?; let fut = create_future(l)?; fut.add_done_callback(cb); }"));
    assert!(!scope_flagged(
        "fn c(f: &Bound<PyAny>) { let _s = python_call_scope()?; f.call_method0(\"cancel\"); }"
    ));
    // Prose, strings and other fns are not calls.
    assert!(!scope_flagged(
        "// create_future is called below\nfn f() { let s = \"get_running_loop\"; }"
    ));
    assert!(!scope_flagged("trait T { fn instantiate(&self) -> R; }\nfn g() { let _s = python_call_scope()?; self.class.call1((a,)) }"));
}
