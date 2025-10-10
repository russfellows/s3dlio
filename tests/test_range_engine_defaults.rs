// tests/test_range_engine_defaults.rs
//
// Test to verify RangeEngine is disabled by default across all backends (v0.9.6+)

#[cfg(test)]
mod range_engine_defaults {
    use s3dlio::object_store::{AzureConfig, GcsConfig};
    use s3dlio::file_store::FileSystemConfig;
    use s3dlio::file_store_direct::FileSystemConfig as DirectIOConfig;
    use s3dlio::constants::DEFAULT_RANGE_ENGINE_THRESHOLD;

    #[test]
    fn test_azure_config_default_range_engine_disabled() {
        let config = AzureConfig::default();
        assert_eq!(
            config.enable_range_engine, 
            false,
            "AzureConfig::default() should have enable_range_engine=false (v0.9.6+)"
        );
        println!("✅ AzureConfig: RangeEngine disabled by default");
    }

    #[test]
    fn test_gcs_config_default_range_engine_disabled() {
        let config = GcsConfig::default();
        assert_eq!(
            config.enable_range_engine,
            false,
            "GcsConfig::default() should have enable_range_engine=false (v0.9.6+)"
        );
        println!("✅ GcsConfig: RangeEngine disabled by default");
    }

    #[test]
    fn test_file_config_default_range_engine_disabled() {
        let config = FileSystemConfig::default();
        assert_eq!(
            config.enable_range_engine,
            false,
            "FileSystemConfig::default() should have enable_range_engine=false (v0.9.6+)"
        );
        println!("✅ FileSystemConfig: RangeEngine disabled by default");
    }

    #[test]
    fn test_directio_config_default_range_engine_disabled() {
        let config = DirectIOConfig::default();
        assert_eq!(
            config.enable_range_engine,
            false,
            "DirectIOConfig::default() should have enable_range_engine=false (v0.9.6+)"
        );
        println!("✅ DirectIOConfig (default): RangeEngine disabled");
    }

    #[test]
    fn test_directio_config_direct_io_range_engine_disabled() {
        let config = DirectIOConfig::direct_io();
        assert_eq!(
            config.enable_range_engine,
            false,
            "DirectIOConfig::direct_io() should have enable_range_engine=false (v0.9.6+)"
        );
        println!("✅ DirectIOConfig (direct_io): RangeEngine disabled");
    }

    #[test]
    fn test_directio_config_high_performance_range_engine_disabled() {
        let config = DirectIOConfig::high_performance();
        assert_eq!(
            config.enable_range_engine,
            false,
            "DirectIOConfig::high_performance() should have enable_range_engine=false (v0.9.6+)"
        );
        println!("✅ DirectIOConfig (high_performance): RangeEngine disabled");
    }

    #[test]
    fn test_configs_can_enable_range_engine() {
        let azure = AzureConfig {
            enable_range_engine: true,
            ..Default::default()
        };
        let gcs = GcsConfig {
            enable_range_engine: true,
            ..Default::default()
        };
        let file = FileSystemConfig {
            enable_range_engine: true,
            ..Default::default()
        };
        let mut directio = DirectIOConfig::direct_io();
        directio.enable_range_engine = true;
        
        assert!(azure.enable_range_engine);
        assert!(gcs.enable_range_engine);
        assert!(file.enable_range_engine);
        assert!(directio.enable_range_engine);
        
        println!("✅ All configs can explicitly enable RangeEngine");
    }

    #[test]
    fn test_default_threshold_is_16mib() {
        const EXPECTED: u64 = 16 * 1024 * 1024;
        assert_eq!(DEFAULT_RANGE_ENGINE_THRESHOLD, EXPECTED);
        println!("✅ DEFAULT_RANGE_ENGINE_THRESHOLD is 16 MiB");
    }

    #[test]
    fn test_all_configs_use_16mib_threshold() {
        let azure = AzureConfig::default();
        let gcs = GcsConfig::default();
        let file = FileSystemConfig::default();
        let directio = DirectIOConfig::default();
        
        const EXPECTED: u64 = 16 * 1024 * 1024;
        
        assert_eq!(azure.range_engine.min_split_size, EXPECTED);
        assert_eq!(gcs.range_engine.min_split_size, EXPECTED);
        assert_eq!(file.range_engine.min_split_size, EXPECTED);
        assert_eq!(directio.range_engine.min_split_size, EXPECTED);
        
        println!("✅ All configs use 16 MiB threshold");
    }
}
