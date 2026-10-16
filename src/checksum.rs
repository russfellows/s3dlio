// src/checksum.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! The checksum every s3dlio writer reports: CRC-32C (Castagnoli), issue #256.
//!
//! Writers have always labelled their checksum `crc32c:<hex>`; until #256 the value was
//! CRC-32 (IEEE, `crc32fast`), which differs from CRC-32C for almost every input (both are
//! zero for empty input) and so did not match a GCS `crc32c` or an S3
//! `x-amz-checksum-crc32c`. The value is now real CRC-32C (CRC-32/ISCSI).
//!
//! # Which implementation, and why
//!
//! `crc-fast` (already in the dependency tree through the AWS SDK's checksum crate). It
//! picks its code at run time from the CPU features (PCLMULQDQ / VPCLMULQDQ with AVX-512
//! on x86-64, PMULL and the CRC32 instructions on aarch64) with a software fallback, so
//! no `-C target-feature` / `target-cpu` build flag is needed and a wheel gets the fast
//! path. The `crc32c` crate was measured first (`examples/bench_crc32c.rs`) and is about
//! half as fast as the `crc32fast` this replaces on x86-64 without build flags: it detects
//! SSE4.2 at run time, but its hardware loop makes a non-inlinable call per 8 bytes (the
//! intrinsic is `#[target_feature]`), so it only runs well when the whole build enables
//! `+sse4.2`. That would have made every `write_chunk` slower than before #256. The
//! `crc32c` crate is kept as a dev-dependency, as an independent oracle for the tests. On the benchmark host `crc-fast` is
//! 3x to 4x faster than `crc32fast` for buffers that fit in cache and as fast as it when
//! memory-bound. The ZIP/NPZ container CRC and nothing else still uses CRC-32 (that format
//! requires it).

use crc_fast::{CrcAlgorithm, Digest};

/// A streaming CRC-32C hasher. Same shape as the `crc32fast::Hasher` it replaces.
#[derive(Clone, Debug)]
pub struct Crc32c(Digest);

impl Default for Crc32c {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32c {
    /// A hasher over no data yet.
    pub fn new() -> Self {
        Self(Digest::new(CrcAlgorithm::Crc32Iscsi))
    }

    /// Feed more data.
    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    /// The CRC-32C of everything fed so far. The hasher can keep being used.
    pub fn finalize(&self) -> u32 {
        // CRC-32/ISCSI is a 32-bit CRC: the digest's u64 holds it in the low half.
        self.0.finalize() as u32
    }
}

/// Which CRC-32C code this CPU runs, for diagnostics and run records (for example
/// `x86_64-avx512-vpclmulqdq`, or a table/software name when no SIMD path applies).
/// Chosen by `crc-fast` at run time; never depends on build flags.
pub fn crc32c_implementation() -> String {
    crc_fast::get_calculator_target(CrcAlgorithm::Crc32Iscsi)
}

/// Continue a CRC-32C: `crc` is a value this module returned (a finalised CRC) for some
/// earlier bytes, and the result is the CRC-32C of those bytes followed by `data`.
/// `crc32c_append(0, data)` is `crc32c_of(data)`.
pub fn crc32c_append(crc: u32, data: &[u8]) -> u32 {
    // The digest state of CRC-32/ISCSI is the complement of the finalised value.
    let mut digest = Digest::new_with_init_state(CrcAlgorithm::Crc32Iscsi, u64::from(!crc));
    digest.update(data);
    digest.finalize() as u32
}

/// CRC-32C of `data` in one call.
pub fn crc32c_of(data: &[u8]) -> u32 {
    crc_fast::crc32_iscsi(data)
}

/// What to add to every checksum-mismatch message: the algorithm, and the one release
/// boundary that makes an old stored value look wrong (issue #256).
pub const MISMATCH_NOTE: &str = "CRC-32C; checksums written before the #256 fix were CRC-32";

/// The CRC-32C an expected-checksum string asks for (issue #259).
///
/// The canonical form is `<algorithm>:<hex>`, the one writers report and checkpoint
/// manifests store (`crc32c:e3069283`). The algorithm name is matched case-insensitively
/// and only `crc32c` is supported: any other algorithm is an error that says so, never a
/// "mismatch". A string with no `:` is bare hex and is read as CRC-32C, for callers that
/// compute the value themselves.
///
/// The hex part must be 1 to 8 hexadecimal digits (either case), nothing else: a `0x`
/// prefix, an empty value, a ninth digit, a further `:` or any other character is a
/// "malformed checksum" error. Fewer than 8 digits are read as the number they spell, so
/// `crc32c:0` is the checksum `crc32c:00000000` (the empty object's).
pub fn parse_expected_crc32c(expected: &str) -> anyhow::Result<u32> {
    let expected_trimmed = expected.trim();
    let hex = match expected_trimmed.split_once(':') {
        None => expected_trimmed,
        Some((alg, hex)) if alg.trim().eq_ignore_ascii_case("crc32c") => hex.trim(),
        Some((alg, _)) => anyhow::bail!(
            "unsupported checksum algorithm '{}' in expected checksum '{}' (supported: crc32c)",
            alg.trim(),
            expected_trimmed
        ),
    };
    if hex.is_empty() || hex.len() > 8 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        anyhow::bail!(
            "malformed checksum '{expected_trimmed}': expected 'crc32c:' followed by 1 to 8 \
             hexadecimal digits (or just the digits)"
        );
    }
    // 1..=8 hex digits always fit a u32.
    Ok(u32::from_str_radix(hex, 16)?)
}

/// Check `data` against an expected-checksum string (see [`parse_expected_crc32c`]).
/// `what` names the object in the error. A malformed or unsupported `expected` is
/// reported as such, not as a mismatch.
pub fn verify_crc32c(data: &[u8], expected: &str, what: &str) -> anyhow::Result<()> {
    let wanted = parse_expected_crc32c(expected)?;
    let actual = crc32c_of(data);
    if actual != wanted {
        anyhow::bail!(
            "Integrity validation failed for {what}: expected {expected}, got {actual:08x} ({MISMATCH_NOTE})"
        );
    }
    Ok(())
}

/// The label writers report: `crc32c:` plus eight lowercase hex digits.
pub fn crc32c_label(crc: u32) -> String {
    format!("crc32c:{crc:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard check input; CRC-32C = 0xe3069283, CRC-32 (IEEE) = 0xcbf43926.
    #[test]
    fn known_answers() {
        assert_eq!(crc32c_of(b"123456789"), 0xe306_9283);
        assert_ne!(crc32c_of(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32c_of(b""), 0);
        // RFC 3720 (iSCSI) B.4: 32 bytes of zeros, and 32 bytes of 0xff.
        assert_eq!(crc32c_of(&[0u8; 32]), 0x8a91_36aa);
        assert_eq!(crc32c_of(&[0xffu8; 32]), 0x62a8_ab43);
    }

    #[test]
    fn streaming_equals_one_shot_at_every_split() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();
        let whole = crc32c_of(&data);
        for split in [0, 1, 7, 8, 63, 64, 999, 1000] {
            let mut h = Crc32c::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finalize(), whole, "split at {split}");
        }
    }

    #[test]
    fn finalize_does_not_consume_the_state() {
        let mut h = Crc32c::new();
        h.update(b"1234");
        let _ = h.clone().finalize();
        h.update(b"56789");
        assert_eq!(h.finalize(), 0xe306_9283);
    }

    #[test]
    fn label_is_eight_lowercase_hex_digits() {
        assert_eq!(crc32c_label(0xe306_9283), "crc32c:e3069283");
        assert_eq!(crc32c_label(0), "crc32c:00000000");
        assert_eq!(crc32c_label(0xab), "crc32c:000000ab");
    }

    /// A bitwise CRC-32C with no table and no crate: the arbiter in the tests below.
    fn bitwise(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = (crc >> 1) ^ if crc & 1 == 1 { 0x82F6_3B78 } else { 0 };
            }
        }
        !crc
    }

    fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    /// Every length around the SIMD block boundaries (16, 32, 64, 128, 256, 512 bytes and a
    /// few KiB), against the bitwise arbiter and the independent `crc32c` crate.
    #[test]
    fn agrees_with_two_independent_implementations_at_every_boundary_length() {
        let data = pseudo_random(5000, 0x1234_5678_9abc_def1);
        for len in (0..=600).chain([1023, 1024, 1025, 2047, 2048, 2049, 4096, 4999, 5000]) {
            let want = bitwise(&data[..len]);
            assert_eq!(crc32c_of(&data[..len]), want, "one-shot, len {len}");
            assert_eq!(
                crc32c::crc32c(&data[..len]),
                want,
                "crc32c crate, len {len}"
            );
            let mut h = Crc32c::new();
            h.update(&data[..len]);
            assert_eq!(h.finalize(), want, "streaming, len {len}");
        }
    }

    /// A large buffer, fed in uneven pieces, equals the one-shot value.
    #[test]
    fn large_buffer_in_uneven_pieces() {
        let data = pseudo_random(3 << 20, 42);
        let want = crc32c::crc32c(&data);
        assert_eq!(crc32c_of(&data), want);
        let mut h = Crc32c::new();
        let mut rest = &data[..];
        let mut piece = 1usize;
        while !rest.is_empty() {
            let n = piece.min(rest.len());
            h.update(&rest[..n]);
            rest = &rest[n..];
            piece = piece * 3 + 1;
        }
        assert_eq!(h.finalize(), want);
    }

    /// RFC 3720 appendix B.4, through every entry point (one shot, streaming, and append
    /// from 0), plus the check value.
    #[test]
    fn rfc3720_vectors_through_every_entry_point() {
        let ascending: Vec<u8> = (0u8..32).collect();
        let descending: Vec<u8> = (0u8..32).rev().collect();
        let cases: [(&[u8], u32); 5] = [
            (&[0u8; 32], 0x8a91_36aa),
            (&[0xffu8; 32], 0x62a8_ab43),
            (&ascending, 0x46dd_794e),
            (&descending, 0x113f_db5c),
            (b"123456789", 0xe306_9283),
        ];
        for (data, want) in cases {
            assert_eq!(crc32c_of(data), want, "one shot");
            assert_eq!(crc32c_append(0, data), want, "append from 0");
            let mut h = Crc32c::new();
            h.update(data);
            assert_eq!(h.finalize(), want, "streaming");
        }
    }

    /// Every length 0..=2200 (all SIMD block boundaries) against the independent crate.
    #[test]
    fn agrees_with_the_crc32c_crate_at_every_length_up_to_2200() {
        let data = pseudo_random(2200, 0xfeed_beef_1234_5678);
        for len in 0..=2200 {
            assert_eq!(
                crc32c_of(&data[..len]),
                crc32c::crc32c(&data[..len]),
                "len {len}"
            );
        }
    }

    /// A 33 MiB buffer (more than any SIMD block) and an unaligned start.
    #[test]
    fn agrees_with_the_crc32c_crate_on_a_33_mib_buffer() {
        let data = pseudo_random(33 << 20, 7);
        assert_eq!(crc32c_of(&data), crc32c::crc32c(&data));
        assert_eq!(crc32c_of(&data[3..]), crc32c::crc32c(&data[3..]));
    }

    /// Resuming from a FINALISED value (not from the internal state) at many splits.
    #[test]
    fn append_continues_from_a_finalised_crc() {
        let data = pseudo_random(5000, 99);
        let whole = crc32c::crc32c(&data);
        for split in (0..=300).chain([511, 512, 513, 1000, 2047, 2048, 4096, 4999, 5000]) {
            let head = crc32c_of(&data[..split]);
            assert_eq!(head, crc32c::crc32c(&data[..split]), "head, split {split}");
            assert_eq!(crc32c_append(head, &data[split..]), whole, "split {split}");
            assert_eq!(
                crc32c_append(head, &data[split..]),
                crc32c::crc32c_append(head, &data[split..]),
                "agrees with the crate's append, split {split}"
            );
        }
    }

    #[test]
    fn reports_which_implementation_the_cpu_selected() {
        let name = crc32c_implementation();
        assert!(!name.trim().is_empty());
        println!("CRC-32C implementation: {name}");
    }
}
