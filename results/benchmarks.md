# Benchmarks (M4 Pro, jax 0.11.0, mlx vs jax CPU backend)

Measured with `jax.jit` + forced evaluation (the runtime calls `mx.eval` at
execute time, so these are real execution times, not lazy-graph build times).

| workload | mlx | cpu | mlx/cpu |
|---|---|---|---|
| matmul 1024x1024 | 1.21 ms | 1.25 ms | 0.97x |
| matmul 2048x2048 | 5.30 ms | 12.02 ms | 0.44x |
| elementwise chain, 1M elems | 0.71 ms | 0.55 ms | 1.28x |
| tiny op (add, 10 elems) | 0.21 ms | 0.002 ms | ~107x |
| MLP fwd+bwd (128x256, 256x256) | 0.52 ms | 0.13 ms | 4.0x |

Interpretation: large matmuls win on the GPU today; everything small is
dominated by the per-executable Python interpreter dispatch + eval overhead
(~0.2 ms). The known fixes are caching `mx.compile`d closures per executable
and batching evaluation (see README follow-ups). Memory is stable: 150
distinct compilations grew maxRSS by ~85 MB with no unbounded growth.

Correctness at scale: a 100-step MLP training loop produces losses matching
the CPU backend to <1e-3 at every checkpoint; attention blocks, layernorm,
512x512 matmuls, and bf16/f16 matmul/reductions all match CPU within
dtype-appropriate tolerances.
