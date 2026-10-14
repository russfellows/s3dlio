// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Core types for operation log entries and operation types
//!
//! These types represent the fundamental data structures used across
//! the s3dlio ecosystem for operation logging and replay.

use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Operation type from op-log
///
/// Represents the different types of storage operations that can be logged and replayed.
///
/// Covers every operation name s3dlio's op-log writer emits. `GET`, `PUT`, `DELETE`, `LIST`
/// and `STAT` (written as `HEAD` by the object-store layer, which parses as `STAT`) can be
/// replayed exactly; see [`OpType::is_exactly_replayable`] for why the others cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
// Variant names are the op-log column text (`GET_RANGE`, `DELETE_BATCH`, ...).
#[allow(non_camel_case_types)]
#[non_exhaustive]
pub enum OpType {
    GET,
    PUT,
    DELETE,
    LIST,
    STAT,
    /// Ranged read. The row records the bytes returned, not the offset or length.
    #[serde(rename = "GET_RANGE")]
    GET_RANGE,
    /// Batch delete. The row records only the first URI of the batch and the object count.
    #[serde(rename = "DELETE_BATCH")]
    DELETE_BATCH,
    /// Prefix delete. The row records the prefix, not the objects that were removed.
    #[serde(rename = "DELETE_PREFIX")]
    DELETE_PREFIX,
    /// Bucket/container creation. The row records a bare name with no scheme.
    #[serde(rename = "CREATE_CONTAINER")]
    CREATE_CONTAINER,
    /// Bucket/container deletion. The row records a bare name with no scheme.
    #[serde(rename = "DELETE_CONTAINER")]
    DELETE_CONTAINER,
}

impl OpType {
    /// Whether a replay can re-issue this operation exactly from what the row records.
    ///
    /// The compound operations (`GET_RANGE`, `DELETE_BATCH`, `DELETE_PREFIX`,
    /// `CREATE_CONTAINER`, `DELETE_CONTAINER`) are parsed so a log containing them loads,
    /// but their rows do not carry enough to repeat them (see each variant), so replay
    /// reports them as unsupported instead of executing an approximation.
    pub fn is_exactly_replayable(self) -> bool {
        matches!(
            self,
            OpType::GET | OpType::PUT | OpType::DELETE | OpType::LIST | OpType::STAT
        )
    }
}

impl FromStr for OpType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Ok(OpType::GET),
            "PUT" => Ok(OpType::PUT),
            "DELETE" => Ok(OpType::DELETE),
            "LIST" => Ok(OpType::LIST),
            "STAT" | "HEAD" => Ok(OpType::STAT),
            "GET_RANGE" => Ok(OpType::GET_RANGE),
            "DELETE_BATCH" => Ok(OpType::DELETE_BATCH),
            "DELETE_PREFIX" => Ok(OpType::DELETE_PREFIX),
            "CREATE_CONTAINER" => Ok(OpType::CREATE_CONTAINER),
            "DELETE_CONTAINER" => Ok(OpType::DELETE_CONTAINER),
            _ => bail!("Unknown operation type: {}", s),
        }
    }
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpType::GET => write!(f, "GET"),
            OpType::PUT => write!(f, "PUT"),
            OpType::DELETE => write!(f, "DELETE"),
            OpType::LIST => write!(f, "LIST"),
            OpType::STAT => write!(f, "STAT"),
            OpType::GET_RANGE => write!(f, "GET_RANGE"),
            OpType::DELETE_BATCH => write!(f, "DELETE_BATCH"),
            OpType::DELETE_PREFIX => write!(f, "DELETE_PREFIX"),
            OpType::CREATE_CONTAINER => write!(f, "CREATE_CONTAINER"),
            OpType::DELETE_CONTAINER => write!(f, "DELETE_CONTAINER"),
        }
    }
}

/// Single operation entry from op-log
///
/// This structure captures the essential information needed for both logging
/// and replay of storage operations across all backends (file://, s3://, az://, direct://).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpLogEntry {
    /// Sequential operation index (0-based)
    pub idx: u64,

    /// Operation type
    pub op: OpType,

    /// Data transferred in bytes (0 for metadata operations)
    pub bytes: u64,

    /// Storage backend endpoint (e.g., "s3://", "file://", "az://", "direct://")
    pub endpoint: String,

    /// Full object path/key
    pub file: String,

    /// Operation start timestamp
    pub start: DateTime<Utc>,

    /// Optional duration in nanoseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ns: Option<u64>,

    /// Optional error message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_type_parse() {
        assert_eq!("GET".parse::<OpType>().unwrap(), OpType::GET);
        assert_eq!("get".parse::<OpType>().unwrap(), OpType::GET);
        assert_eq!("PUT".parse::<OpType>().unwrap(), OpType::PUT);
        assert_eq!("DELETE".parse::<OpType>().unwrap(), OpType::DELETE);
        assert_eq!("LIST".parse::<OpType>().unwrap(), OpType::LIST);
        assert_eq!("STAT".parse::<OpType>().unwrap(), OpType::STAT);
        assert_eq!("HEAD".parse::<OpType>().unwrap(), OpType::STAT);
        assert!("UNKNOWN".parse::<OpType>().is_err());
    }

    #[test]
    fn test_op_type_display() {
        assert_eq!(OpType::GET.to_string(), "GET");
        assert_eq!(OpType::PUT.to_string(), "PUT");
        assert_eq!(OpType::DELETE.to_string(), "DELETE");
        assert_eq!(OpType::LIST.to_string(), "LIST");
        assert_eq!(OpType::STAT.to_string(), "STAT");
    }
}
