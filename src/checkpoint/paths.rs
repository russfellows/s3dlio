use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy { 
    Flat, 
    RoundRobin, 
    Binary 
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Strategy::Flat => write!(f, "flat"),
            Strategy::RoundRobin => write!(f, "round_robin"),
            Strategy::Binary => write!(f, "binary"),
        }
    }
}

impl Strategy {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_lowercase().as_str() {
            "flat" => Ok(Strategy::Flat),
            "round_robin" | "roundrobin" => Ok(Strategy::RoundRobin),
            "binary" => Ok(Strategy::Binary),
            _ => anyhow::bail!("Unknown strategy: {}. Valid options: flat, round_robin, binary", s),
        }
    }
}

pub struct KeyLayout {
    pub base: String,        // normalized without trailing '/'
    pub step: u64,
    pub ts: String,          // yyyy-mm-ddThh:mm:ssZ or epoch-ms
}

impl KeyLayout {
    /// Create a new key layout with current timestamp
    pub fn new(base: String, step: u64) -> Self {
        let ts = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
        Self {
            base: base.trim_end_matches('/').to_string(),
            step,
            ts,
        }
    }

    /// Create a new key layout with custom timestamp
    pub fn with_timestamp(base: String, step: u64, ts: String) -> Self {
        Self {
            base: base.trim_end_matches('/').to_string(),
            step,
            ts,
        }
    }

    /// Generate the manifest key path
    pub fn manifest_key(&self) -> String {
        format!("manifests/ckpt-{}-{}.json", self.step, self.ts)
    }

    /// Generate a shard key path based on the strategy
    pub fn shard_key(&self, rank: u32, strat: Strategy) -> String {
        let shard_dir = match strat {
            Strategy::Flat => {
                format!("shards/ckpt-{}-{}", self.step, self.ts)
            }
            Strategy::RoundRobin => {
                // Distribute across 16 prefixes by default to avoid hot spots
                let partition = rank % 16;
                format!("shards/rr/{:02x}/ckpt-{}-{}", partition, self.step, self.ts)
            }
            Strategy::Binary => {
                // Use first 4 bits of rank as hex prefix for partitioning
                let prefix = format!("{:08b}", rank);
                let partition = &prefix[0..4];
                format!("shards/bin/{}/ckpt-{}-{}", partition, self.step, self.ts)
            }
        };
        format!("{}/rank-{}.bin", shard_dir, rank)
    }

    /// Generate the latest pointer key
    pub fn latest_key(&self) -> String {
        "latest.json".to_string()
    }

    /// Generate a directory prefix for listing all shards
    pub fn shard_dir_prefix(&self, strat: Strategy) -> String {
        match strat {
            Strategy::Flat => {
                format!("shards/ckpt-{}-{}/", self.step, self.ts)
            }
            Strategy::RoundRobin => {
                "shards/rr/".to_string()
            }
            Strategy::Binary => {
                "shards/bin/".to_string()
            }
        }
    }

    /// Extract step and timestamp from a manifest key
    pub fn parse_manifest_key(key: &str) -> anyhow::Result<(u64, String)> {
        // Expected format: {base}/manifests/ckpt-{step}-{timestamp}.json
        let filename = key.rsplit('/').next()
            .ok_or_else(|| anyhow::anyhow!("Invalid manifest key format"))?;
        
        if !filename.starts_with("ckpt-") || !filename.ends_with(".json") {
            anyhow::bail!("Manifest key must match pattern: ckpt-{{step}}-{{timestamp}}.json");
        }
        
        let without_ext = filename.strip_suffix(".json").unwrap();
        let without_prefix = without_ext.strip_prefix("ckpt-").unwrap();
        
        let parts: Vec<&str> = without_prefix.splitn(2, '-').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid manifest key format: expected ckpt-{{step}}-{{timestamp}}.json");
        }
        
        let step = parts[0].parse::<u64>()
            .map_err(|e| anyhow::anyhow!("Invalid step in manifest key: {}", e))?;
        let timestamp = parts[1].to_string();
        
        Ok((step, timestamp))
    }

    /// Generate a full URI from base URI and relative key
    pub fn to_uri(&self, base_uri: &str, relative_key: &str) -> String {
        format!("{}/{}", base_uri.trim_end_matches('/'), relative_key)
    }

    /// Extract relative key from full URI
    pub fn from_uri(&self, base_uri: &str, full_uri: &str) -> Option<String> {
        let base_trimmed = base_uri.trim_end_matches('/');
        let expected_prefix = format!("{}/", base_trimmed);
        
        if full_uri.starts_with(&expected_prefix) {
            Some(full_uri[expected_prefix.len()..].to_string())
        } else if full_uri == base_trimmed {
            Some("".to_string())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_layout_creation() {
        let layout = KeyLayout::new("s3://bucket/checkpoints".to_string(), 100);
        assert_eq!(layout.base, "s3://bucket/checkpoints");
        assert_eq!(layout.step, 100);
        assert!(!layout.ts.is_empty());
    }

    #[test]
    fn test_manifest_key() {
        let layout = KeyLayout::with_timestamp(
            "s3://bucket/checkpoints".to_string(), 
            100, 
            "20240101-120000".to_string()
        );
        assert_eq!(
            layout.manifest_key(),
            "manifests/ckpt-100-20240101-120000.json"
        );
    }

    #[test]
    fn test_shard_keys() {
        let layout = KeyLayout::with_timestamp(
            "s3://bucket/ckpt".to_string(), 
            50, 
            "test".to_string()
        );

        // Test flat strategy
        assert_eq!(
            layout.shard_key(0, Strategy::Flat),
            "shards/ckpt-50-test/rank-0.bin"
        );

        // Test round-robin strategy
        assert_eq!(
            layout.shard_key(0, Strategy::RoundRobin),
            "shards/rr/00/ckpt-50-test/rank-0.bin"
        );
        assert_eq!(
            layout.shard_key(16, Strategy::RoundRobin),
            "shards/rr/00/ckpt-50-test/rank-16.bin"
        );
        assert_eq!(
            layout.shard_key(1, Strategy::RoundRobin),
            "shards/rr/01/ckpt-50-test/rank-1.bin"
        );

        // Test binary strategy
        assert_eq!(
            layout.shard_key(0, Strategy::Binary),
            "shards/bin/0000/ckpt-50-test/rank-0.bin"
        );
        assert_eq!(
            layout.shard_key(8, Strategy::Binary),
            "shards/bin/0000/ckpt-50-test/rank-8.bin"
        );
    }

    #[test]
    fn test_latest_key() {
        let layout = KeyLayout::new("s3://bucket/checkpoints".to_string(), 100);
        assert_eq!(layout.latest_key(), "latest.json");
    }

    #[test]
    fn test_parse_manifest_key() {
        let result = KeyLayout::parse_manifest_key("s3://bucket/manifests/ckpt-100-20240101-120000.json");
        assert!(result.is_ok());
        let (step, ts) = result.unwrap();
        assert_eq!(step, 100);
        assert_eq!(ts, "20240101-120000");

        // Test invalid formats
        assert!(KeyLayout::parse_manifest_key("invalid").is_err());
        assert!(KeyLayout::parse_manifest_key("ckpt-100.json").is_err());
        assert!(KeyLayout::parse_manifest_key("ckpt-abc-def.json").is_err());
    }

    #[test]
    fn test_strategy_from_str() {
        assert_eq!(Strategy::from_str("flat").unwrap(), Strategy::Flat);
        assert_eq!(Strategy::from_str("round_robin").unwrap(), Strategy::RoundRobin);
        assert_eq!(Strategy::from_str("roundrobin").unwrap(), Strategy::RoundRobin);
        assert_eq!(Strategy::from_str("binary").unwrap(), Strategy::Binary);
        assert!(Strategy::from_str("invalid").is_err());
    }

    #[test]
    fn test_uri_conversion() {
        let layout = KeyLayout::new("s3://bucket/ckpt".to_string(), 100);
        
        let full_uri = layout.to_uri("s3://bucket/ckpt", "manifests/test.json");
        assert_eq!(full_uri, "s3://bucket/ckpt/manifests/test.json");
        
        let relative = layout.from_uri("s3://bucket/ckpt", "s3://bucket/ckpt/manifests/test.json");
        assert_eq!(relative, Some("manifests/test.json".to_string()));
        
        let relative_none = layout.from_uri("s3://bucket/ckpt", "s3://other/test.json");
        assert_eq!(relative_none, None);
    }
}
