use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::SeedableRng;
use s3dlio::{
    data_gen::{generate_controlled_data, DataGenerator},
    constants::BLK_SIZE,
};

fn bench_single_pass_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pass_generation");
    
    for size in [BLK_SIZE, BLK_SIZE * 4, BLK_SIZE * 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("single_pass", size), 
            size, 
            |b, &size| {
                b.iter(|| {
                    black_box(generate_controlled_data(size, 1, 1))
                })
            }
        );
    }
    
    group.finish();
}

fn bench_streaming_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_generation");
    
    for size in [BLK_SIZE, BLK_SIZE * 4, BLK_SIZE * 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("streaming", size), 
            size, 
            |b, &size| {
                b.iter(|| {
                    let generator = DataGenerator::new();
                    let mut object_gen = generator.begin_object(size, 1, 1);
                    black_box(object_gen.fill_remaining())
                })
            }
        );
    }
    
    group.finish();
}

fn bench_streaming_chunked_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_chunked_generation");
    
    for chunk_size in [4096, 16384, 65536].iter() {
        group.bench_with_input(
            BenchmarkId::new("chunked", chunk_size), 
            chunk_size, 
            |b, &chunk_size| {
                b.iter(|| {
                    let size = BLK_SIZE * 4;
                    let generator = DataGenerator::new();
                    let mut object_gen = generator.begin_object(size, 1, 1);
                    let mut total_data = Vec::new();
                    
                    while let Some(chunk) = object_gen.fill_chunk(chunk_size) {
                        total_data.extend_from_slice(&chunk);
                    }
                    
                    black_box(total_data)
                })
            }
        );
    }
    
    group.finish();
}

fn bench_rng_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng_performance");
    
    // Benchmark different RNG approaches
    group.bench_function("SmallRng_per_block", |b| {
        b.iter(|| {
            let mut data = vec![0u8; BLK_SIZE];
            for i in 0..16 {  // 16 blocks worth
                let seed = i as u64;
                let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
                use rand::RngCore;
                rng.fill_bytes(&mut data[..4096]);
                black_box(&data);
            }
        })
    });
    
    group.bench_function("SmallRng_reused", |b| {
        b.iter(|| {
            let mut data = vec![0u8; BLK_SIZE];
            let mut rng = rand::rngs::SmallRng::seed_from_u64(12345);
            for _ in 0..16 {  // 16 blocks worth
                use rand::RngCore;
                rng.fill_bytes(&mut data[..4096]);
                black_box(&data);
            }
        })
    });
    
    group.finish();
}

criterion_group!(benches, 
    bench_single_pass_generation,
    bench_streaming_generation, 
    bench_streaming_chunked_generation,
    bench_rng_performance
);
criterion_main!(benches);