// Example: Using s3dlio hardware detection in sai3-bench
//
// This shows how external tools can leverage s3dlio's hardware detection
// for optimal data generation performance.

use s3dlio::hardware;

fn main() {
    // Get CPU count respecting affinity (taskset, Docker limits, etc.)
    let affinity_cpus = hardware::get_affinity_cpu_count();
    println!("CPUs available to this process: {}", affinity_cpus);

    // Get total system CPUs (ignores affinity)
    let total_cpus = hardware::total_cpus();
    println!("Total system CPUs: {}", total_cpus);

    // Check if NUMA is available
    if hardware::is_numa_available() {
        println!("NUMA system detected");
        
        #[cfg(feature = "numa")]
        if let Some(topology) = hardware::detect_numa_topology() {
            println!("NUMA nodes: {}", topology.num_nodes);
            for node in &topology.nodes {
                println!("  Node {}: {} CPUs", node.node_id, node.cpus.len());
            }
        }
    } else {
        println!("UMA (single memory domain) system");
    }

    // Get recommended thread count for data generation
    // This respects CPU affinity and NUMA topology
    let threads = hardware::recommended_data_gen_threads(None, None);
    println!("Recommended data gen threads: {}", threads);

    // Override with specific count
    let threads_limited = hardware::recommended_data_gen_threads(None, Some(8));
    println!("Limited to 8 threads: {}", threads_limited);

    // Pin to specific NUMA node (requires numa feature)
    #[cfg(feature = "numa")]
    {
        let threads_numa = hardware::recommended_data_gen_threads(Some(0), None);
        println!("Threads for NUMA node 0: {}", threads_numa);
    }
}
