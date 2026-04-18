// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

#[cfg(feature = "hdf5")]
pub mod hdf5;
pub mod npz;
pub mod raw;
pub mod tfrecord;

#[cfg(feature = "hdf5")]
pub use hdf5::build_hdf5;
pub use npz::{
    array_to_npy_bytes, build_multi_npz, build_npz, list_npz_arrays, read_npy_bytes, read_npz_array,
};
pub use raw::build_raw;
pub use tfrecord::{build_tfrecord, build_tfrecord_with_index, TfRecordWithIndex};
