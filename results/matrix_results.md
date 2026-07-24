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
| `sort` | CRASH | F0723 19:12:09.054891  598617 pjrt_c_api_status_utils.cc:139] Unexpected error status <built-in method __enter__ of _thread.lock object at 0x10c32fb00> returned |
| `top_k` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.composite |
| `cumsum` | CRASH | F0723 19:12:09.672193  598721 pjrt_c_api_status_utils.cc:139] Unexpected error status <built-in method __enter__ of _thread.lock object at 0x10e997f80> returned |
| `where_select` | PASS |  |
| `exp_log_tanh` | PASS |  |
| `erf` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.composite |
| `rsqrt_pow` | PASS |  |
| `einsum` | PASS |  |
| `conv2d` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.convolution |
| `random_normal` | PASS |  |
| `random_uniform` | PASS |  |
| `random_split` | PASS |  |
| `scan` | CRASH | F0723 19:12:13.395515  599326 pjrt_c_api_status_utils.cc:139] Unexpected error status <built-in method __enter__ of _thread.lock object at 0x10ab40040> returned |
| `while_loop` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: <jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x10b503eb0> |
| `cond` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.case |
| `fori_loop` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: <jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x10f726f30> |
| `grad_simple` | PASS |  |
| `grad_mlp` | PASS |  |
| `jvp` | PASS |  |
| `vjp` | PASS |  |
| `vmap` | PASS |  |
| `jit_donate` | PASS |  |
| `cholesky` | CRASH | F0723 19:12:16.970543  599871 pjrt_c_api_status_utils.cc:139] Unexpected error status <built-in method __enter__ of _thread.lock object at 0x10beecdc0> returned |
| `qr` | CRASH | F0723 19:12:17.365233  599918 pjrt_c_api_status_utils.cc:139] Unexpected error status <built-in method __enter__ of _thread.lock object at 0x10d4bc6c0> returned |
| `svd` | CRASH | F0723 19:12:17.782044  599980 pjrt_c_api_status_utils.cc:139] Unexpected error status <built-in method __enter__ of _thread.lock object at 0x10ccc3b00> returned |
| `triangular_solve` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported op stablehlo.triangular_solve |
| `fft` | FAIL | jax.errors.JaxRuntimeError: INTERNAL: jax-mlx: unsupported ir dtype: complex<f32> |
| `mlp_training_step` | PASS |  |
