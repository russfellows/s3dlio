// src/hardware.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Hardware detection and optimization utilities
//!
//! This module provides runtime hardware detection for CPU affinity, NUMA topology,
//! and other system characteristics. All functions work at RUNTIME regardless of
//! how the library was compiled, allowing binaries built on one system to optimize
//! for hardware on another system.
//!
//! # Design Philosophy
//!
//! - **Runtime Detection**: No compile-time feature flags gate detection code
//! - **Cross-Platform**: Graceful degradation on unsupported platforms
//! - **Zero Dependencies**: Core CPU detection uses only std library
//! - **Optional NUMA**: NUMA topology requires the `numa` feature but detection
//!   code is always available (returns None if feature disabled)
//!
//! # Usage
//!
//! ```rust
//! use s3dlio::hardware;
//!
//! // Get CPU count respecting process affinity mask
//! let cpu_count = hardware::get_affinity_cpu_count();
//!
//! // Get total system CPUs (ignores affinity)
//! let total_cpus = hardware::total_cpus();
//!
//! // Check if NUMA is available (requires 'numa' feature)
//! if hardware::is_numa_available() {
//!     println!("NUMA system detected");
//! }
//! ```

use std::fs;

/// Get CPU count from current process affinity mask
///
/// This function respects CPU affinity masks set by:
/// - Python multiprocessing
/// - taskset / cpuset
/// - Docker/Kubernetes CPU limits
/// - NUMA node pinning
///
/// Falls back to total system CPU count if affinity cannot be determined.
///
/// # Platform Support
///
/// - **Linux**: Reads /proc/self/status for Cpus_allowed_list
/// - **Other platforms**: Falls back to num_cpus::get()
///
/// # Examples
///
/// ```rust
/// use s3dlio::hardware::get_affinity_cpu_count;
///
/// // Returns CPUs available to this process
/// let cpus = get_affinity_cpu_count();
/// println!("Available CPUs: {}", cpus);
/// ```
pub fn get_affinity_cpu_count() -> usize {
    #[cfg(target_os = "linux")]
    {
        // Try to read /proc/self/status to get Cpus_allowed_list
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("Cpus_allowed_list:") {
                    if let Some(cpus) = line.split(':').nth(1) {
                        let cpus = cpus.trim();
                        let count = parse_cpu_list(cpus);
                        if count > 0 {
                            tracing::debug!("CPU affinity mask: {} CPUs ({})", count, cpus);
                            return count;
                        }
                    }
                }
            }
        }
    }
    
    // Fallback to system CPU count
    num_cpus::get()
}

/// Parse Linux CPU list format (e.g., "0-23" or "0-11,24-35")
///
/// This function parses the format used by Linux in /proc/self/status
/// for Cpus_allowed_list and similar fields.
///
/// # Format
///
/// - Single CPU: "5" → 1
/// - Range: "0-7" → 8
/// - Multiple ranges: "0-7,16-23" → 16
/// - Sparse: "0,2,4,6" → 4
///
/// # Examples
///
/// ```rust
/// use s3dlio::hardware::parse_cpu_list;
///
/// assert_eq!(parse_cpu_list("0-23"), 24);
/// assert_eq!(parse_cpu_list("0-11,24-35"), 24);
/// assert_eq!(parse_cpu_list("0,2,4,6"), 4);
/// ```
///
/// # Platform Support
///
/// This function is available on all platforms but only meaningful on Linux.
/// On other platforms, it's provided for completeness but may not be useful.
#[cfg(target_os = "linux")]
pub fn parse_cpu_list(cpu_list: &str) -> usize {
    let mut count = 0;
    
    for part in cpu_list.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        
        if let Some((start, end)) = part.split_once('-') {
            // Range format: "0-23"
            if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                count += (end - start) + 1;
            }
        } else {
            // Single CPU: "5"
            if part.parse::<usize>().is_ok() {
                count += 1;
            }
        }
    }
    
    count
}

/// Get total system CPU count (ignores affinity)
///
/// This returns the total number of logical CPUs available on the system,
/// ignoring any affinity masks or restrictions.
///
/// Use `get_affinity_cpu_count()` if you want to respect process affinity.
///
/// # Examples
///
/// ```rust
/// use s3dlio::hardware::total_cpus;
///
/// let cpus = total_cpus();
/// println!("System has {} CPUs", cpus);
/// ```
pub fn total_cpus() -> usize {
    num_cpus::get()
}

/// Check if NUMA (Non-Uniform Memory Access) is available
///
/// Returns true if:
/// - The `numa` feature is enabled at compile time
/// - NUMA topology can be detected at runtime
/// - System has more than 1 NUMA node
///
/// # Feature Requirements
///
/// This function requires the `numa` feature. If the feature is not enabled,
/// it always returns false.
///
/// # Examples
///
/// ```rust
/// use s3dlio::hardware::is_numa_available;
///
/// if is_numa_available() {
///     println!("NUMA optimizations available");
/// } else {
///     println!("Using UMA (single memory domain)");
/// }
/// ```
pub fn is_numa_available() -> bool {
    #[cfg(feature = "numa")]
    {
        // Check if NUMA topology is available and has multiple nodes
        if let Some(topology) = detect_numa_topology() {
            return topology.num_nodes > 1;
        }
    }
    
    false
}

/// Detect NUMA topology (if numa feature enabled)
///
/// Returns NUMA topology information if:
/// - The `numa` feature is enabled
/// - NUMA can be detected on this system
/// - System has at least 1 NUMA node
///
/// Returns None if NUMA is not available or feature is disabled.
///
/// # Feature Requirements
///
/// This function requires the `numa` feature. If the feature is not enabled,
/// it always returns None.
///
/// # Examples
///
/// ```rust
/// use s3dlio::hardware::detect_numa_topology;
///
/// if let Some(topology) = detect_numa_topology() {
///     println!("NUMA nodes: {}", topology.num_nodes);
///     for node in &topology.nodes {
///         println!("  Node {}: {} CPUs", node.node_id, node.cpus.len());
///     }
/// }
/// ```
#[cfg(feature = "numa")]
pub fn detect_numa_topology() -> Option<crate::numa::NumaTopology> {
    match crate::numa::NumaTopology::detect() {
        Ok(topology) => {
            tracing::debug!(
                "NUMA topology detected: {} nodes, {} physical cores, {} logical CPUs, UMA={}",
                topology.num_nodes,
                topology.physical_cores,
                topology.logical_cpus,
                topology.is_uma
            );
            Some(topology)
        }
        Err(e) => {
            tracing::debug!("NUMA topology detection failed: {}", e);
            None
        }
    }
}

#[cfg(not(feature = "numa"))]
pub fn detect_numa_topology() -> Option<()> {
    // When numa feature is disabled, return None (no topology available)
    // Using Option<()> instead of custom type to avoid having to define
    // NumaTopology when feature is disabled
    None
}

/// Get recommended thread count for data generation
///
/// This function provides a sensible default thread count based on:
/// - Process CPU affinity (respects taskset, Docker limits, etc.)
/// - NUMA node pinning (if specified)
/// - Physical core count (avoids hyperthreading overhead for data generation)
///
/// # Arguments
///
/// * `numa_node` - Optional NUMA node ID to pin to (requires `numa` feature)
/// * `max_threads` - Optional maximum thread count override
///
/// # Returns
///
/// Recommended thread count for parallel data generation
///
/// # Examples
///
/// ```rust
/// use s3dlio::hardware::recommended_data_gen_threads;
///
/// // Use default detection
/// let threads = recommended_data_gen_threads(None, None);
///
/// // Limit to specific NUMA node (requires numa feature)
/// let threads = recommended_data_gen_threads(Some(0), None);
///
/// // Override with specific count
/// let threads = recommended_data_gen_threads(None, Some(8));
/// ```
#[cfg(feature = "numa")]
pub fn recommended_data_gen_threads(numa_node: Option<usize>, max_threads: Option<usize>) -> usize {
    if let Some(node_id) = numa_node {
        if let Some(topology) = detect_numa_topology() {
            if let Some(node) = topology.nodes.iter().find(|n| n.node_id == node_id) {
                // Limit threads to cores available on this NUMA node
                let node_cores = node.cpus.len();
                let requested_threads = max_threads.unwrap_or(node_cores);
                return requested_threads.min(node_cores);
            }
        }
        // NUMA node requested but not available, use affinity
        max_threads.unwrap_or_else(get_affinity_cpu_count)
    } else {
        // No specific NUMA node, use all cores
        max_threads.unwrap_or_else(num_cpus::get)
    }
}

#[cfg(not(feature = "numa"))]
pub fn recommended_data_gen_threads(_numa_node: Option<usize>, max_threads: Option<usize>) -> usize {
    // Without NUMA feature, ignore numa_node parameter
    max_threads.unwrap_or_else(get_affinity_cpu_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_cpus() {
        let cpus = total_cpus();
        assert!(cpus > 0, "Should detect at least 1 CPU");
        assert!(cpus <= 1024, "Sanity check: shouldn't report >1024 CPUs");
    }

    #[test]
    fn test_get_affinity_cpu_count() {
        let cpus = get_affinity_cpu_count();
        assert!(cpus > 0, "Should detect at least 1 CPU");
        assert!(cpus <= total_cpus(), "Affinity CPUs shouldn't exceed total");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpu_list() {
        assert_eq!(parse_cpu_list("0-23"), 24);
        assert_eq!(parse_cpu_list("0-11,24-35"), 24);
        assert_eq!(parse_cpu_list("0,2,4,6"), 4);
        assert_eq!(parse_cpu_list("5"), 1);
        assert_eq!(parse_cpu_list(""), 0);
        assert_eq!(parse_cpu_list("0-7,16-23"), 16);
    }

    #[test]
    fn test_is_numa_available() {
        // Just ensure it doesn't panic - actual value depends on hardware
        let _ = is_numa_available();
    }

    #[test]
    fn test_recommended_data_gen_threads() {
        // Default case
        let threads = recommended_data_gen_threads(None, None);
        assert!(threads > 0, "Should recommend at least 1 thread");

        // With max_threads override
        let threads = recommended_data_gen_threads(None, Some(4));
        assert_eq!(threads, 4, "Should respect max_threads override");
    }
}
