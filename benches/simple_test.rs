use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn simple_benchmark(c: &mut Criterion) {
    c.bench_function("simple_test", |b| {
        b.iter(|| {
            let data = vec![42u8; 1024];
            black_box(data);
        });
    });
}

criterion_group!(benches, simple_benchmark);
criterion_main!(benches);
