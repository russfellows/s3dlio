// Comprehensive performance comparison between old and new data generation algorithms
//
// Measures:
// 1. Throughput (MB/s) - PRIMARY CONCERN
// 2. CPU time - SECONDARY CONCERN
// 3. Memory allocation - TERTIARY CONCERN

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::{Duration, Instant};

// Import both old and new implementations
use s3dlio::data_gen::{generate_controlled_data, ObjectGen as ObjectGenOld, DataGenerator};
use s3dlio::data_gen_alt::{generate_controlled_data_alt, ObjectGenAlt};

/// Memory tracking helper
struct MemorySnapshot {
    start_rss: usize,
    peak_rss: usize,
}

impl MemorySnapshot {
    fn new() -> Self {
        Self {
            start_rss: Self::get_rss(),
            peak_rss: 0,
        }
    }

    fn update_peak(&mut self) {
        let current = Self::get_rss();
        if current > self.peak_rss {
            self.peak_rss = current;
        }
    }

    fn delta_mb(&self) -> f64 {
        (self.peak_rss.saturating_sub(self.start_rss)) as f64 / 1_048_576.0
    }

    #[cfg(target_os = "linux")]
    fn get_rss() -> usize {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024; // Convert to bytes
                        }
                    }
                }
            }
        }
        0
    }

    #[cfg(not(target_os = "linux"))]
    fn get_rss() -> usize {
        0 // Fallback for non-Linux
    }
}

/// CPU time measurement (user + system)
fn get_cpu_time() -> Duration {
    use std::time::Duration;
    
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(stat) = fs::read_to_string("/proc/self/stat") {
            let fields: Vec<&str> = stat.split_whitespace().collect();
            if fields.len() > 14 {
                let utime = fields[13].parse::<u64>().unwrap_or(0);
                let stime = fields[14].parse::<u64>().unwrap_or(0);
                let ticks_per_sec = 100; // Usually 100 on Linux
                let total_ticks = utime + stime;
                return Duration::from_millis(total_ticks * 1000 / ticks_per_sec);
            }
        }
    }
    
    Duration::from_secs(0)
}

/// Comprehensive performance test
fn run_performance_test<F>(
    name: &str,
    size: usize,
    iterations: usize,
    f: F,
) -> (f64, f64, f64) // (throughput_mb_s, cpu_seconds, memory_mb)
where
    F: Fn(),
{
    let mut mem = MemorySnapshot::new();
    let cpu_start = get_cpu_time();
    let wall_start = Instant::now();

    for _ in 0..iterations {
        f();
        mem.update_peak();
    }

    let wall_elapsed = wall_start.elapsed();
    let cpu_elapsed = get_cpu_time() - cpu_start;

    let total_bytes = (size * iterations) as f64;
    let throughput = total_bytes / wall_elapsed.as_secs_f64() / 1_048_576.0;
    let cpu_seconds = cpu_elapsed.as_secs_f64();
    let memory_mb = mem.delta_mb();

    println!("{}: {:.2} MB/s, CPU: {:.3}s, Mem: {:.2} MB", 
             name, throughput, cpu_seconds, memory_mb);

    (throughput, cpu_seconds, memory_mb)
}

/// Single-pass generation comparison
fn bench_single_pass_generation(c: &mut Criterion) {
    let sizes = vec![
        ("1MB", 1_048_576),
        ("4MB", 4_194_304),
        ("16MB", 16_777_216),
        ("64MB", 67_108_864),
    ];

    for (name, size) in sizes {
        let mut group = c.benchmark_group(format!("single_pass_{}", name));
        group.throughput(Throughput::Bytes(size as u64));
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(10));

        // OLD: generate_controlled_data (compress=1, incompressible)
        group.bench_with_input(
            BenchmarkId::new("OLD_compress1", name),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data = generate_controlled_data(size, 1, 1);
                    black_box(data);
                });
            },
        );

        // NEW: generate_controlled_data_alt (compress=1, incompressible)
        group.bench_with_input(
            BenchmarkId::new("NEW_compress1", name),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data = generate_controlled_data_alt(size, 1, 1);
                    black_box(data);
                });
            },
        );

        // OLD: generate_controlled_data (compress=5)
        group.bench_with_input(
            BenchmarkId::new("OLD_compress5", name),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data = generate_controlled_data(size, 1, 5);
                    black_box(data);
                });
            },
        );

        // NEW: generate_controlled_data_alt (compress=5)
        group.bench_with_input(
            BenchmarkId::new("NEW_compress5", name),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data = generate_controlled_data_alt(size, 1, 5);
                    black_box(data);
                });
            },
        );

        group.finish();
    }
}

/// Streaming generation comparison
fn bench_streaming_generation(c: &mut Criterion) {
    let total_size = 16_777_216; // 16MB
    let chunk_sizes = vec![
        ("64KB", 65536),
        ("256KB", 262144),
        ("1MB", 1_048_576),
    ];

    for (name, chunk_size) in chunk_sizes {
        let mut group = c.benchmark_group(format!("streaming_{}", name));
        group.throughput(Throughput::Bytes(total_size as u64));
        group.sample_size(20);

        // OLD: ObjectGen streaming
        group.bench_with_input(
            BenchmarkId::new("OLD_streaming", name),
            &chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    let mut data_gen = DataGenerator::new();
                    let mut obj_gen = data_gen.begin_object(total_size, 1, 1);
                    let mut buffer = vec![0u8; chunk_size];
                    
                    while !obj_gen.is_complete() {
                        obj_gen.fill_chunk(&mut buffer);
                        black_box(&buffer);
                    }
                });
            },
        );

        // NEW: ObjectGenAlt streaming
        group.bench_with_input(
            BenchmarkId::new("NEW_streaming", name),
            &chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    let mut obj_gen = ObjectGenAlt::new(total_size, 1, 1);
                    let mut buffer = vec![0u8; chunk_size];
                    
                    while !obj_gen.is_complete() {
                        obj_gen.fill_chunk(&mut buffer);
                        black_box(&buffer);
                    }
                });
            },
        );

        group.finish();
    }
}

/// Deduplication comparison
fn bench_with_dedup(c: &mut Criterion) {
    let size = 16_777_216; // 16MB (256 blocks)
    let dedup_factors = vec![1, 2, 4, 8];

    let mut group = c.benchmark_group("dedup_16MB");
    group.throughput(Throughput::Bytes(size as u64));
    group.sample_size(20);

    for dedup in dedup_factors {
        // OLD
        group.bench_with_input(
            BenchmarkId::new("OLD", format!("dedup{}", dedup)),
            &dedup,
            |b, &dedup| {
                b.iter(|| {
                    let data = generate_controlled_data(size, dedup, 1);
                    black_box(data);
                });
            },
        );

        // NEW
        group.bench_with_input(
            BenchmarkId::new("NEW", format!("dedup{}", dedup)),
            &dedup,
            |b, &dedup| {
                b.iter(|| {
                    let data = generate_controlled_data_alt(size, dedup, 1);
                    black_box(data);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_pass_generation,
    bench_streaming_generation,
    bench_with_dedup
);
criterion_main!(benches);
