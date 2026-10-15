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
//! | `1` / `true`           | load `.env` from the working directory (or the nearest parent)  |
//! | `0` / `false`          | never load a `.env` file                                        |
//! | anything else          | a file path: load exactly that file                             |
//! | unset / empty          | the default of the running program (below)                      |
//!
//! `1`/`true`/`0`/`false` are case-insensitive. Surrounding whitespace is ignored.
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
//! `s3-cli` loads it at start-up, before anything else reads the environment. A library
//! loads it (once per process) the first time s3dlio builds an S3 client, lists buckets
//! or containers, or sets up the Arrow backend. Settings that s3dlio reads *earlier* than
//! that (for example the runtime thread count) do not see values that come only from the
//! `.env` file; an application that needs the file's values everywhere should call
//! [`load_env_file`] itself at start-up.

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
    } else if value == "1" || value.eq_ignore_ascii_case("true") {
        Some(DotenvMode::WorkingDir)
    } else if value == "0" || value.eq_ignore_ascii_case("false") {
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

/// Set by [`load_for_cli`] before anything resolves the mode.
static CLI_DEFAULT: AtomicBool = AtomicBool::new(false);
/// The switch, read once.
static MODE: OnceLock<DotenvMode> = OnceLock::new();
/// The load itself happens once per process too.
static APPLIED: OnceLock<()> = OnceLock::new();

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
pub fn load_for_cli() {
    CLI_DEFAULT.store(true, Ordering::SeqCst);
    apply();
}

/// Apply the `.env` policy of this process. Called from every site that used to call
/// `dotenvy::dotenv()` itself; cheap and idempotent (the switch is read once and the
/// file is loaded at most once per process).
pub(crate) fn apply() {
    APPLIED.get_or_init(|| apply_mode(process_mode()));
}

fn apply_mode(mode: &DotenvMode) {
    match mode {
        DotenvMode::WorkingDir => {
            // Same as before #243: a missing file is not an error.
            let _ = dotenvy::dotenv();
        }
        DotenvMode::File(path) => {
            if let Err(e) = load_env_file(path) {
                warn!("{LOAD_DOTENV_ENV} names {}: {e:#}", path.display());
            }
        }
        DotenvMode::Off => {
            // Only the unset (default) case is worth a hint; an explicit `0` is a decision.
            if std::env::var(LOAD_DOTENV_ENV)
                .map(|v| parse_switch(&v).is_none())
                .unwrap_or(true)
                && dotenvy::dotenv_iter().is_ok()
            {
                info!(
                    "a .env file was found in the working directory (or a parent) but is not \
                     loaded: s3dlio does not read .env files when used as a library. Set \
                     {LOAD_DOTENV_ENV}=1 to load it, {LOAD_DOTENV_ENV}=/path/to/file to load a \
                     specific file, or call s3dlio::dotenv::load_env_file()."
                );
            }
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
        for v in ["1", "true", "TRUE", "True", " 1 ", "\ttrue\n"] {
            assert_eq!(parse_switch(v), Some(DotenvMode::WorkingDir), "{v:?}");
        }
        for v in ["0", "false", "FALSE", "False", " 0 "] {
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
            parse_switch("yes"),
            Some(DotenvMode::File(PathBuf::from("yes")))
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

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("tmp")
            .join(format!("dotenv243-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn load_env_file_sets_missing_variables_and_never_overrides_set_ones() {
        let _g = lock();
        let _restore = EnvRestore::capture(&["S3DLIO_243_UNIT_NEW", "S3DLIO_243_UNIT_SET"]);
        let dir = scratch_dir("explicit");
        let file = dir.join("custom.env");
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
        fs::remove_dir_all(&dir).ok();
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
        let dir = scratch_dir("filemode");
        let file = dir.join("named.env");
        fs::write(&file, "S3DLIO_243_UNIT_FILE_MODE=named\n").unwrap();
        std::env::remove_var("S3DLIO_243_UNIT_FILE_MODE");

        apply_mode(&DotenvMode::File(file));
        assert_eq!(std::env::var("S3DLIO_243_UNIT_FILE_MODE").unwrap(), "named");

        // A missing file named by the switch is a warning, not a panic or an error.
        apply_mode(&DotenvMode::File(dir.join("missing.env")));
        // `Off` never touches the environment.
        apply_mode(&DotenvMode::Off);
        fs::remove_dir_all(&dir).ok();
    }
}
