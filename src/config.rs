use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Valid object formats.
#[derive(Clone, Copy, Debug, ValueEnum)]
#[clap(rename_all = "UPPERCASE")] // CLI shows NPZ, TFRECORD, HDF5, RAW
pub enum ObjectType {
    Npz,
    TfRecord,
    Hdf5,
    Raw,
}

/// S3-specific configuration
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct S3Config {
    pub endpoint: Option<String>,
    pub region: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
}

/// Azure Blob Storage configuration
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AzureConfig {
    pub account: String,
    pub container: Option<String>,
    pub access_key: Option<String>,
    pub connection_string: Option<String>,
}

/// File system configuration
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct FileConfig {
    pub base_path: Option<String>,
    pub use_direct_io: bool,
}

/// Backend-specific configuration
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum BackendConfig {
    S3(S3Config),
    Azure(AzureConfig),
    File(FileConfig),
}

/// Main configuration structure
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CliConfig {
    pub backend: Option<BackendConfig>,
    pub default_jobs: Option<usize>,
    pub log_level: Option<String>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            backend: None,
            default_jobs: Some(32),
            log_level: Some("info".to_string()),
        }
    }
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
