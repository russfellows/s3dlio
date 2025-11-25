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

/// Data generation algorithm selection.
#[derive(Clone, Copy, Debug, ValueEnum, Deserialize, Serialize)]
#[clap(rename_all = "kebab-case")] // CLI shows random, prand
#[serde(rename_all = "snake_case")] // JSON shows random, prand
pub enum DataGenAlgorithm {
    /// Random data generation (variable performance 1-7 GB/s depending on size)
    Random,
    /// Pseudo-random using BASE_BLOCK algorithm (consistent 3-4 GB/s, CPU-efficient)
    Prand,
}

impl Default for DataGenAlgorithm {
    fn default() -> Self {
        Self::Random
    }
}

/// Data generation modes for performance optimization.
#[derive(Clone, Copy, Debug, ValueEnum, Deserialize, Serialize)]
#[clap(rename_all = "kebab-case")] // CLI shows streaming, single-pass
#[serde(rename_all = "snake_case")] // JSON shows streaming, single_pass
pub enum DataGenMode {
    /// Streaming generation with configurable chunk size (default, faster for most workloads)
    Streaming,
    /// Single-pass generation (legacy, faster for 16-32MB objects)
    SinglePass,
}

impl Default for DataGenMode {
    fn default() -> Self {
        // Default to streaming based on performance benchmarks showing 64% win rate
        Self::Streaming
    }
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
    pub compress_factor: usize,    // e.g. 1 (random), 2 (≈50 % zeros)
    
    // ---- performance optimization ----
    pub data_gen_algorithm: DataGenAlgorithm, // random vs prand algorithm
    pub data_gen_mode: DataGenMode,  // streaming vs single-pass generation  
    pub chunk_size: usize,           // chunk size for streaming mode (default 256KB)
}

impl Config {
    /// Create a new Config with optimal defaults based on performance benchmarks
    pub fn new_with_defaults(
        object_type: ObjectType,
        elements: usize, 
        element_size: usize,
        dedup_factor: usize,
        compress_factor: usize
    ) -> Self {
        Self {
            object_type,
            elements,
            element_size,
            use_controlled: dedup_factor != 1 || compress_factor != 1,
            dedup_factor,
            compress_factor,
            data_gen_algorithm: DataGenAlgorithm::default(), // Random by default
            data_gen_mode: DataGenMode::default(), // Streaming by default
            chunk_size: 256 * 1024, // 256KB optimal from benchmarks
        }
    }
    
    /// Override data generation algorithm (random vs prand)
    pub fn with_data_gen_algorithm(mut self, algorithm: DataGenAlgorithm) -> Self {
        self.data_gen_algorithm = algorithm;
        self
    }
    
    /// Override data generation mode for specific performance requirements
    pub fn with_data_gen_mode(mut self, mode: DataGenMode) -> Self {
        self.data_gen_mode = mode;
        self
    }
    
    /// Override chunk size for streaming mode
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }
}