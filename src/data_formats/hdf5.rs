use anyhow::{bail, Result};
use bytes::Bytes;
use hdf5_metno::File;
use hdf5_metno_sys::h5f::H5Fget_file_image;    // ← import the C function
use std::os::raw::c_void;

/// Build an HDF5 file entirely in RAM (Core VFD, no backing store),
/// write your dataset, then extract its in-memory image via H5Fget_file_image.
pub fn build_hdf5(
    elements: usize,
    element_size: usize,
    data: &[u8],
) -> Result<Bytes> {
    // 1) Set up a core-driver FAPL with no backing store
    let mut builder = File::with_options();
    {
        let fapl = builder.fapl();         // &mut FileAccessBuilder
        fapl.core();                       // use the Memory (CORE) driver
        fapl.core_filebacked(false);      // do NOT write to disk
    }

    // 2) Create the (anonymous) in-memory file
    let file = builder.create("unused.h5")?;

    // 3) Write one flat 1-D dataset called "data"
    file.new_dataset::<u8>()
        .shape([elements * element_size])
        .create("data")?
        .write(data)?;
    file.flush()?; // ensure all data lands in driver buffers

    // 4) Pull the raw HDF5 image via the C API
    let fid = file.id();
    // First call: get required buffer size
    //let image_size = unsafe { sys::H5Fget_file_image(fid, std::ptr::null_mut(), 0) };
    let image_size = unsafe { H5Fget_file_image(fid, std::ptr::null_mut(), 0) };
    if image_size < 0 {
        bail!("H5Fget_file_image failed to query size");
    }
    let image_size = image_size as usize;

    // Allocate and fill
    let mut buf = Vec::with_capacity(image_size);
    unsafe {
        buf.set_len(image_size);
        //let ret = sys::H5Fget_file_image(
        let ret = H5Fget_file_image(
            fid,
            buf.as_mut_ptr() as *mut c_void,
            image_size,
        );
        if ret < 0 {
            bail!("H5Fget_file_image failed to retrieve image");
        }
    }

    Ok(Bytes::from(buf))
}
