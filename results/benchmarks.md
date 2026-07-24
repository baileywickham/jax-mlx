# Benchmarks (M4 Pro, jax 0.11.0, mlx vs jax CPU backend)

Measured with `jax.jit` + forced evaluation (the runtime calls `mx.eval` at
execute time, so these are real execution times, not lazy-graph build times).

With the `mx.compile` fast path (executables without data-dependent control
flow replay a fused MLX graph instead of the Python op loop):

| workload | mlx | cpu | mlx/cpu | pre-mx.compile |
|---|---|---|---|---|
| matmul 1024x1024 | 1.16 ms | 1.09 ms | 1.07x | 0.97x |
| matmul 2048x2048 | 3.83 ms | 13.53 ms | 0.28x | 0.44x |
| elementwise chain, 1M elems | 0.33 ms | 0.55 ms | 0.60x | 1.28x |
| tiny op (add, 10 elems) | 0.17 ms | 0.002 ms | ~84x | ~107x |
| MLP fwd+bwd (128x256, 256x256) | 0.29 ms | 0.14 ms | 2.0x | 4.0x |

Interpretation: fusion moves elementwise workloads from GPU-slower to
GPU-faster, and matmul 2048 now matches the MPSGraph-backed jax-metal plugin
(3.8 ms vs 3.4 ms). Remaining gap: fixed per-dispatch overhead (~0.17 ms)
from the C-bridge -> Python -> mx round trip, which dominates tiny ops and
many-small-op programs; batched/async evaluation is the next lever.
Programs containing `stablehlo.while` (e.g. jax.random) are statically kept
on the interpreter — tracing would silently bake placeholder loop counts. Memory is stable: 150
distinct compilations grew maxRSS by ~85 MB with no unbounded growth.

Correctness at scale: a 100-step MLP training loop produces losses matching
the CPU backend to <1e-3 at every checkpoint; attention blocks, layernorm,
512x512 matmuls, and bf16/f16 matmul/reductions all match CPU within
dtype-appropriate tolerances.
