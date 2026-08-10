use criterion::{Criterion, criterion_group, criterion_main};

fn dummy_benchmark(_c: &mut Criterion) {}

criterion_group!(benches, dummy_benchmark);
criterion_main!(benches);
