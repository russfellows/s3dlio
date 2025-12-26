// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use anyhow::Result;
use bytes::Bytes;

pub fn build_raw(data: &[u8]) -> Result<Bytes> {
    Ok(Bytes::copy_from_slice(data))
}

