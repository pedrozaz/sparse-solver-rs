use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sparse_solver_rs::poisson::assemble_poisson_2d;
use sparse_solver_rs::solvers::SolverOptions;
use sparse_solver_rs::solvers::cg::ConjugateGradient;
use sparse_solver_rs::solvers::multigrid::MultigridSolver;

fn bench_cg_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("CG Solver");
    let options = SolverOptions {
        max_iter: 5000,
        tol: 1e-8,
    };

    for &n in &[7, 15, 31, 63] {
        let (a, b) = assemble_poisson_2d(n);
        group.bench_with_input(BenchmarkId::new("Poisson2D", n), &n, |bench, _| {
            bench.iter(|| ConjugateGradient::solve(&a, &b, &options));
        });
    }
    group.finish();
}

fn bench_multigrid_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multigrid V-Cycle");
    let mg_solver = MultigridSolver::new();
    let options = SolverOptions {
        max_iter: 50,
        tol: 1e-8,
    };

    for &n in &[7, 15, 31, 63] {
        let (_, b) = assemble_poisson_2d(n);
        group.bench_with_input(BenchmarkId::new("Poisson2D", n), &n, |bench, &n| {
            bench.iter(|| mg_solver.solve(n, &b, &options));
        });
    }
    group.finish();
}

fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpMV");

    for &n in &[7, 15, 31, 63] {
        let (a, _) = assemble_poisson_2d(n);
        let x = vec![1.0; n * n];
        group.bench_with_input(BenchmarkId::new("Poisson2D", n), &n, |bench, _| {
            bench.iter(|| a.spmv(&x));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_spmv, bench_cg_solver, bench_multigrid_solver);
criterion_main!(benches);
