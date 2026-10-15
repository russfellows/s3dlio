// src/checksum.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! The checksum every s3dlio writer reports: CRC-32C (Castagnoli), issue #256.
//!
//! Writers have always labelled their checksum `crc32c:<hex>`; until #256 the value was
//! CRC-32 (IEEE, `crc32fast`), which differs from CRC-32C for every input and so never
//! matched a GCS `crc32c` or an S3 `x-amz-checksum-crc32c`. The value is now real
//! CRC-32C, computed with the hardware-accelerated `crc32c` crate (SSE4.2 / ARMv8 CRC,
//! with a software fallback). The ZIP/NPZ container CRC and nothing else still uses
//! CRC-32 (that format requires it).

/// A streaming CRC-32C hasher. Same shape as the `crc32fast::Hasher` it replaces.
#[derive(Clone, Debug, Default)]
pub struct Crc32c(u32);

impl Crc32c {
    /// A hasher over no data yet.
    pub fn new() -> Self {
        Self(0)
    }

    /// Feed more data.
    pub fn update(&mut self, data: &[u8]) {
        self.0 = crc32c::crc32c_append(self.0, data);
    }

    /// The CRC-32C of everything fed so far. The hasher can keep being used.
    pub fn finalize(&self) -> u32 {
        self.0
    }
}

/// CRC-32C of `data` in one call.
pub fn crc32c_of(data: &[u8]) -> u32 {
    crc32c::crc32c(data)
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
}
