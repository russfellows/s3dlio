// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! `OpType` is matched exhaustively, with no wildcard, by downstream crates (sai3-bench's
//! `replay_streaming.rs` handles exactly GET, PUT, DELETE, LIST and STAT). Adding a variant or
//! marking the enum `#[non_exhaustive]` breaks them at compile time, so this file is the guard:
//! it stops compiling if `OpType` stops being exactly those five variants.

use s3dlio_oplog::OpType;

/// Exhaustive match over exactly the five original variants, deliberately without a wildcard arm.
/// (`#[non_exhaustive]` on the enum would make a wildcard-free match a compile error here.)
fn name_of(op: OpType) -> &'static str {
    match op {
        OpType::GET => "GET",
        OpType::PUT => "PUT",
        OpType::DELETE => "DELETE",
        OpType::LIST => "LIST",
        OpType::STAT => "STAT",
    }
}

#[test]
fn op_type_is_exactly_the_five_original_variants() {
    for op in [
        OpType::GET,
        OpType::PUT,
        OpType::DELETE,
        OpType::LIST,
        OpType::STAT,
    ] {
        assert_eq!(name_of(op), op.to_string());
        assert_eq!(
            name_of(op.to_string().parse().expect("round trip")),
            name_of(op)
        );
    }
    assert_eq!("HEAD".parse::<OpType>().expect("HEAD"), OpType::STAT);
}

#[test]
fn op_type_serde_names_are_unchanged() {
    assert_eq!(
        serde_json::to_string(&OpType::STAT).expect("ser"),
        "\"STAT\""
    );
    assert_eq!(
        serde_json::from_str::<OpType>("\"DELETE\"").expect("de"),
        OpType::DELETE
    );
}
