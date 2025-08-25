use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ShardMeta {
    pub rank: u32,
    pub key: String,
    pub size: u64,
    pub etag: Option<String>,          // from ObjectMetadata.e_tag
    pub checksum: Option<String>,      // e.g., "crc32c:abcd..." (optional)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Manifest {
    pub format_version: u32,           // start at 1
    pub framework: String,             // "torch" | "flax" | "tf" | "generic"
    pub global_step: u64,
    pub epoch: u64,
    pub wall_time: String,             // RFC3339
    pub world_size: u32,
    pub shards: Vec<ShardMeta>,
    pub status: String,                // "complete"
    pub user_meta: Option<serde_json::Value>,
}

impl Manifest {
    /// Create a new manifest with default values
    pub fn new(
        framework: String,
        global_step: u64,
        epoch: u64,
        world_size: u32,
    ) -> Self {
        Self {
            format_version: 1,
            framework,
            global_step,
            epoch,
            wall_time: chrono::Utc::now().to_rfc3339(),
            world_size,
            shards: Vec::new(),
            status: "incomplete".to_string(),
            user_meta: None,
        }
    }

    /// Mark the manifest as complete
    pub fn mark_complete(&mut self) {
        self.status = "complete".to_string();
    }

    /// Add a shard to the manifest
    pub fn add_shard(&mut self, shard: ShardMeta) {
        self.shards.push(shard);
    }

    /// Sort shards by rank for consistent ordering
    pub fn sort_shards(&mut self) {
        self.shards.sort_by_key(|s| s.rank);
    }

    /// Validate manifest consistency
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.format_version == 0 {
            anyhow::bail!("Invalid format_version: must be >= 1");
        }
        
        if self.framework.is_empty() {
            anyhow::bail!("Framework cannot be empty");
        }
        
        if self.world_size == 0 {
            anyhow::bail!("World size must be > 0");
        }
        
        // Check for duplicate ranks
        let mut ranks = std::collections::HashSet::new();
        for shard in &self.shards {
            if !ranks.insert(shard.rank) {
                anyhow::bail!("Duplicate rank found: {}", shard.rank);
            }
        }
        
        // For complete manifests, ensure we have shards for all ranks
        if self.status == "complete" {
            let expected_ranks: std::collections::HashSet<u32> = (0..self.world_size).collect();
            let actual_ranks: std::collections::HashSet<u32> = self.shards.iter().map(|s| s.rank).collect();
            if expected_ranks != actual_ranks {
                anyhow::bail!("Missing shards for some ranks. Expected: {:?}, Got: {:?}", 
                    expected_ranks, actual_ranks);
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_creation() {
        let manifest = Manifest::new("torch".to_string(), 100, 5, 4);
        assert_eq!(manifest.format_version, 1);
        assert_eq!(manifest.framework, "torch");
        assert_eq!(manifest.global_step, 100);
        assert_eq!(manifest.epoch, 5);
        assert_eq!(manifest.world_size, 4);
        assert_eq!(manifest.status, "incomplete");
        assert!(manifest.shards.is_empty());
    }

    #[test]
    fn test_manifest_validation() {
        let mut manifest = Manifest::new("torch".to_string(), 100, 5, 2);
        
        // Should pass validation for incomplete manifest
        assert!(manifest.validate().is_ok());
        
        // Add shards
        manifest.add_shard(ShardMeta {
            rank: 0,
            key: "shard-0".to_string(),
            size: 1024,
            etag: None,
            checksum: None,
        });
        manifest.add_shard(ShardMeta {
            rank: 1,
            key: "shard-1".to_string(),
            size: 2048,
            etag: None,
            checksum: None,
        });
        
        // Mark complete and validate
        manifest.mark_complete();
        assert!(manifest.validate().is_ok());
        
        // Test duplicate rank detection
        manifest.add_shard(ShardMeta {
            rank: 1,
            key: "shard-1-dup".to_string(),
            size: 1024,
            etag: None,
            checksum: None,
        });
        assert!(manifest.validate().is_err());
    }
}
