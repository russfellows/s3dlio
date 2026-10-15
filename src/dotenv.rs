// src/dotenv.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! `.env` file handling (issue #243): ONE switch, ONE helper, called from every place
//! s3dlio used to load `.env` on its own.
//!
//! # The switch
//!
//! `S3DLIO_LOAD_DOTENV`, read from the **real process environment** (never from a
//! `.env` file, which would be circular), once per process:
//!
//! | value                  | meaning                                                         |
//! |------------------------|-----------------------------------------------------------------|
//! | `1` / `true` / `yes` / `on`  | load `.env` from the working directory (or the nearest parent) |
//! | `0` / `false` / `no` / `off` | never load a `.env` file                                       |
//! | anything else          | a file path: load exactly that file                             |
//! | unset / empty          | the default of the running program (below)                      |
//!
//! The words are case-insensitive. Surrounding whitespace is ignored.
//!
//! # Defaults
//!
//! * the `s3-cli` binary: **on** (`.env` from the working directory), as it always was;
//! * library use (Rust and Python): **off**. A `.env` file in whatever directory a host
//!   process happened to be started from silently changes credentials, endpoints and
//!   every other `S3DLIO_*` / `AWS_*` setting of the whole process, so an embedding
//!   application has to ask for it. When a `.env` exists but is not loaded because of
//!   this default, a one-time INFO log says how to enable it.
//!
//! # Precedence
//!
//! A variable that is already set in the real environment is **never** overridden by a
//! `.env` file (`dotenvy`'s default); the file only fills in what is missing.
//!
//! # When the file is read
//!
//! Once per process, at the first of these (so before s3dlio reads any setting):
//!
//! * `s3-cli`: start-up, before the command line is parsed and before the Tokio runtime
//!   is built;
//! * Python: `import s3dlio`: the extension module's initialisation, the first thing it
//!   does, which also runs before the optional `torch` import inside `s3dlio/torch.py`
//!   (a `torch` the application imported before `import s3dlio` is outside s3dlio's
//!   reach, so torch-side settings are not covered);
//! * Rust: the first use of any entry point that reads a setting: the global runtime,
//!   `configure_thread_pools`, `store_for_uri*`, an S3 / Azure / GCS client, the S3
//!   dataset constructors, `Logger::new`, the process-wide endpoint, a bucket or
//!   container listing, the Arrow backend set-up, and the latched tuning knobs
//!   (`S3DLIO_UNSIGNED_PAYLOAD`, `S3DLIO_SKIP_HEAD`, `S3DLIO_ENABLE_RANGE_OPTIMIZATION`,
//!   `S3DLIO_RANGE_THRESHOLD_MB`, `S3DLIO_SIZE_CACHE_TTL_SECS`, `S3DLIO_GCS_RAPID`).
//!
//! **What this does not promise.** s3dlio cannot know whether its host is multi-threaded:
//! a Python application, an inference server or a Rust program may already have threads
//! when it first calls s3dlio, and setting an environment variable is not safe against
//! another thread that is inside C `getenv` at that moment. Only `s3-cli` (which loads the
//! file as its first act, on its main thread) has no other threads. A host that needs the
//! file's values, or a race-free environment, before its own threads start loads the file
//! itself (or sets the real environment) at the very start of the process.
//!
//! An application that wants a file's values even earlier (before it calls anything in
//! s3dlio) can call [`load_env_file`] itself, or in Python load it with `python-dotenv`
//! before `import s3dlio`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use tracing::{info, warn};

/// The environment variable that switches `.env` loading (see the module docs).
pub const LOAD_DOTENV_ENV: &str = "S3DLIO_LOAD_DOTENV";

/// What `S3DLIO_LOAD_DOTENV` resolved to for this process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DotenvMode {
    /// Never load a `.env` file.
    Off,
    /// Load `.env` from the working directory (or the nearest parent, as `dotenvy` does).
    WorkingDir,
    /// Load exactly this file.
    File(PathBuf),
}

/// Which program is running, which decides the default when the switch is unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DotenvDefault {
    /// An embedding application (Rust or Python): off.
    Library,
    /// The `s3-cli` binary: on.
    Cli,
}

/// Parse the value of `S3DLIO_LOAD_DOTENV`; `None` means "not set" (unset or empty).
pub(crate) fn parse_switch(value: &str) -> Option<DotenvMode> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else if ["1", "true", "yes", "on"]
        .iter()
        .any(|t| value.eq_ignore_ascii_case(t))
    {
        Some(DotenvMode::WorkingDir)
    } else if ["0", "false", "no", "off"]
        .iter()
        .any(|t| value.eq_ignore_ascii_case(t))
    {
        Some(DotenvMode::Off)
    } else {
        Some(DotenvMode::File(PathBuf::from(value)))
    }
}

/// The mode for a given switch value and program default. Pure; the process-wide
/// resolution below only feeds it the real environment.
pub(crate) fn resolve_mode(switch: Option<&str>, default: DotenvDefault) -> DotenvMode {
    switch.and_then(parse_switch).unwrap_or(match default {
        DotenvDefault::Cli => DotenvMode::WorkingDir,
        DotenvDefault::Library => DotenvMode::Off,
    })
}

/// What to tell a user whose credentials are missing while a `.env` file exists that
/// s3dlio did not load because of the library default; `None` when that is not the
/// situation (the file is loaded, loading was switched off explicitly, or no `.env`).
pub(crate) fn not_loaded_hint() -> Option<String> {
    if *process_mode() != DotenvMode::Off || switch_is_set() || dotenvy::dotenv_iter().is_err() {
        return None;
    }
    Some(NOT_LOADED_HINT.to_string())
}

const NOT_LOADED_HINT: &str =
    "a .env file was found in the working directory (or a parent) but is \
    not loaded: s3dlio does not read .env files when used as a library. Set \
    S3DLIO_LOAD_DOTENV=1 to load it, S3DLIO_LOAD_DOTENV=/path/to/file to load a specific \
    file, or call s3dlio::dotenv::load_env_file()";

/// True when the switch is set to a non-empty value (an explicit decision).
fn switch_is_set() -> bool {
    std::env::var(LOAD_DOTENV_ENV)
        .map(|v| parse_switch(&v).is_some())
        .unwrap_or(false)
}

/// Set by [`load_for_cli`] before anything resolves the mode.
static CLI_DEFAULT: AtomicBool = AtomicBool::new(false);
/// The switch, read once.
static MODE: OnceLock<DotenvMode> = OnceLock::new();
/// The load itself happens once per process too; holds the problem it met, if any.
static APPLIED: OnceLock<Option<String>> = OnceLock::new();

fn process_mode() -> &'static DotenvMode {
    MODE.get_or_init(|| {
        let default = if CLI_DEFAULT.load(Ordering::SeqCst) {
            DotenvDefault::Cli
        } else {
            DotenvDefault::Library
        };
        let switch = std::env::var(LOAD_DOTENV_ENV).ok();
        resolve_mode(switch.as_deref(), default)
    })
}

/// Load `path` as a `.env` file: every variable it defines that is **not already set**
/// in the process environment is set. Variables that are already set keep their value.
///
/// This is the explicit way for an application to load a `.env` file (the default for
/// library use is not to load one; see the module docs). It works whatever
/// `S3DLIO_LOAD_DOTENV` says.
///
/// # Errors
/// The file cannot be opened, or a line in it cannot be parsed.
pub fn load_env_file<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    dotenvy::from_path(path).with_context(|| format!("loading env file {}", path.display()))
}

/// What the `s3-cli` binary calls first, before it parses its arguments or builds any
/// client: make `.env` loading the default (the switch still wins) and apply it.
///
/// Returns the problem the load met (a file named by `S3DLIO_LOAD_DOTENV` that cannot be
/// read), because no log subscriber exists yet at that point: the caller prints it.
#[doc(hidden)]
pub fn load_for_cli() -> Option<String> {
    CLI_DEFAULT.store(true, Ordering::SeqCst);
    apply_inner().clone()
}

/// Apply the `.env` policy of this process. Called at the start of every library entry
/// point (and from the sites that used to call `dotenvy::dotenv()` themselves); cheap and
/// idempotent (the switch is read once and the file is loaded at most once per process).
pub(crate) fn apply() {
    apply_inner();
}

fn apply_inner() -> &'static Option<String> {
    APPLIED.get_or_init(|| {
        let problem = apply_mode(process_mode());
        if let Some(p) = &problem {
            warn!("{p}");
        }
        problem
    })
}

/// Do what `mode` says; returns the problem, if any.
fn apply_mode(mode: &DotenvMode) -> Option<String> {
    match mode {
        DotenvMode::WorkingDir => {
            // Same as before #243: a missing file is not an error.
            let _ = dotenvy::dotenv();
            None
        }
        DotenvMode::File(path) => load_env_file(path)
            .err()
            .map(|e| format!("{LOAD_DOTENV_ENV} names {}: {e:#}", path.display())),
        DotenvMode::Off => {
            // Only the unset (default) case is worth a hint; an explicit `0` is a decision.
            if !switch_is_set() && dotenvy::dotenv_iter().is_ok() {
                info!("{NOT_LOADED_HINT}.");
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::{lock, EnvRestore};
    use std::fs;

    #[test]
    fn switch_values_parse() {
        for v in [
            "1", "true", "TRUE", "True", " 1 ", "\ttrue\n", "yes", "YES", "on", "On",
        ] {
            assert_eq!(parse_switch(v), Some(DotenvMode::WorkingDir), "{v:?}");
        }
        for v in [
            "0", "false", "FALSE", "False", " 0 ", "no", "No", "off", "OFF",
        ] {
            assert_eq!(parse_switch(v), Some(DotenvMode::Off), "{v:?}");
        }
        for v in ["", "   ", "\n"] {
            assert_eq!(parse_switch(v), None, "{v:?}");
        }
        assert_eq!(
            parse_switch("/etc/s3dlio/prod.env"),
            Some(DotenvMode::File(PathBuf::from("/etc/s3dlio/prod.env")))
        );
        assert_eq!(
            parse_switch("./local.env"),
            Some(DotenvMode::File(PathBuf::from("./local.env")))
        );
        // Only the documented spellings are booleans: anything else is a path.
        assert_eq!(
            parse_switch("maybe"),
            Some(DotenvMode::File(PathBuf::from("maybe")))
        );
    }

    #[test]
    fn unset_switch_follows_the_program_default() {
        assert_eq!(
            resolve_mode(None, DotenvDefault::Library),
            DotenvMode::Off,
            "library use must not read .env by default"
        );
        assert_eq!(
            resolve_mode(None, DotenvDefault::Cli),
            DotenvMode::WorkingDir,
            "s3-cli keeps reading .env by default"
        );
        // An empty value counts as unset.
        assert_eq!(
            resolve_mode(Some(""), DotenvDefault::Library),
            DotenvMode::Off
        );
        assert_eq!(
            resolve_mode(Some("  "), DotenvDefault::Cli),
            DotenvMode::WorkingDir
        );
    }

    #[test]
    fn an_explicit_switch_beats_either_default() {
        for default in [DotenvDefault::Library, DotenvDefault::Cli] {
            assert_eq!(
                resolve_mode(Some("1"), default),
                DotenvMode::WorkingDir,
                "{default:?}"
            );
            assert_eq!(
                resolve_mode(Some("0"), default),
                DotenvMode::Off,
                "{default:?}"
            );
            assert_eq!(
                resolve_mode(Some("/x/.env"), default),
                DotenvMode::File(PathBuf::from("/x/.env")),
                "{default:?}"
            );
        }
    }

    /// A scratch directory under target/tmp, removed when dropped (also when the test
    /// fails and unwinds).
    struct ScratchDir(PathBuf);

    impl ScratchDir {
        fn new(name: &str) -> Self {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("tmp")
                .join(format!("dotenv243-{name}-{}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn load_env_file_sets_missing_variables_and_never_overrides_set_ones() {
        let _g = lock();
        let _restore = EnvRestore::capture(&["S3DLIO_243_UNIT_NEW", "S3DLIO_243_UNIT_SET"]);
        let dir = ScratchDir::new("explicit");
        let file = dir.path().join("custom.env");
        fs::write(
            &file,
            "S3DLIO_243_UNIT_NEW=from-file\nS3DLIO_243_UNIT_SET=from-file\n",
        )
        .unwrap();
        std::env::remove_var("S3DLIO_243_UNIT_NEW");
        std::env::set_var("S3DLIO_243_UNIT_SET", "from-real-env");

        load_env_file(&file).unwrap();

        assert_eq!(std::env::var("S3DLIO_243_UNIT_NEW").unwrap(), "from-file");
        assert_eq!(
            std::env::var("S3DLIO_243_UNIT_SET").unwrap(),
            "from-real-env",
            "the real environment wins over the file"
        );
    }

    #[test]
    fn load_env_file_reports_a_missing_file() {
        let err = load_env_file("/nonexistent/s3dlio-243/none.env").unwrap_err();
        assert!(
            format!("{err:#}").contains("/nonexistent/s3dlio-243/none.env"),
            "{err:#}"
        );
    }

    #[test]
    fn file_mode_loads_exactly_that_file() {
        let _g = lock();
        let _restore = EnvRestore::capture(&["S3DLIO_243_UNIT_FILE_MODE"]);
        let dir = ScratchDir::new("filemode");
        let file = dir.path().join("named.env");
        fs::write(&file, "S3DLIO_243_UNIT_FILE_MODE=named\n").unwrap();
        std::env::remove_var("S3DLIO_243_UNIT_FILE_MODE");

        assert_eq!(apply_mode(&DotenvMode::File(file)), None);
        assert_eq!(std::env::var("S3DLIO_243_UNIT_FILE_MODE").unwrap(), "named");

        // A missing file named by the switch is a warning, not a panic or an error.
        let problem = apply_mode(&DotenvMode::File(dir.path().join("missing.env")));
        let problem = problem.expect("a missing named file is reported");
        assert!(problem.contains("S3DLIO_LOAD_DOTENV"), "{problem}");
        assert!(problem.contains("missing.env"), "{problem}");
        // `Off` never touches the environment and has no problem to report.
        assert_eq!(apply_mode(&DotenvMode::Off), None);
    }
}
