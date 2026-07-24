| test | status | detail |
|---|---|---|
| `dtype_float32` | PASS |  |
| `dtype_float16` | PASS |  |
| `dtype_bfloat16` | PASS |  |
| `dtype_int32` | PASS |  |
| `dtype_int16` | PASS |  |
| `dtype_int8` | PASS |  |
| `dtype_uint8` | PASS |  |
| `dtype_uint32` | PASS |  |
| `dtype_int64` | PASS |  |
| `dtype_float64` | PASS |  |
| `dtype_complex64` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported ir dtype: complex<f32> |
| `dtype_bool` | PASS |  |
| `matmul_f32` | PASS |  |
| `matmul_f16` | PASS |  |
| `matmul_bf16` | PASS |  |
| `reduce_sum` | PASS |  |
| `reduce_max` | PASS |  |
| `argmax` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: general reduce region ['stablehlo.compare', 'stablehlo.compare', 'stablehlo.or', 'stablehlo.compare', 'stablehlo. |
| `transpose_reshape` | PASS |  |
| `slice_concat` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.dynamic_slice |
| `pad` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.pad |
| `gather_take` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.gather |
| `scatter_at_set` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.scatter |
| `sort` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.sort |
| `top_k` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.composite |
| `cumsum` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.reduce_window |
| `where_select` | PASS |  |
| `exp_log_tanh` | PASS |  |
| `erf` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.composite |
| `rsqrt_pow` | PASS |  |
| `einsum` | PASS |  |
| `conv2d` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.convolution |
| `random_normal` | PASS |  |
| `random_uniform` | PASS |  |
| `random_split` | PASS |  |
| `scan` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: <jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x10f1147b0> |
| `while_loop` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: <jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x10b6a6770> |
| `cond` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.case |
| `fori_loop` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: <jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x10eb9f670> |
| `grad_simple` | PASS |  |
| `grad_mlp` | PASS |  |
| `jvp` | PASS |  |
| `vjp` | PASS |  |
| `vmap` | PASS |  |
| `jit_donate` | PASS |  |
| `cholesky` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.cholesky |
| `qr` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported custom_call Qr |
| `svd` | FAIL | NotImplementedError: MLIR translation rule for primitive 'eigh' not found for platform mlx |
| `triangular_solve` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.triangular_solve |
| `fft` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported ir dtype: complex<f32> |
| `mlp_training_step` | PASS |  |
