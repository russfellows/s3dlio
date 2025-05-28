use clap::ValueEnum;

/// Valid object formats.
#[clap(rename_all = "UPPERCASE")] // CLI shows NPZ, TFRECORD, HDF5, RAW
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ObjectType {
    Npz,
    TfRecord,
    Hdf5,
    Raw,
}

/// Runtime parameters used by `data_gen`.
#[derive(Debug, Clone)]
pub struct Config {
    pub object_type:    ObjectType, // ← enum of "NPZ" | "HDF5" | "TFRecord" | "RAW"
    pub elements:       usize,       // number of elements per object
    pub element_size:   usize,   // bytes per element
                               
    // ---- knobs for data pattern generation ----
    pub use_controlled: bool,    // true => call generate_controlled_data()
    pub dedup_factor:   usize,       // e.g. 1 (all unique), 3 (1/3 unique)
    pub compress_factor: usize,    // e.g. 1 (random), 2 (≈50 % zeros)
}
