# jax-mlx

An open-source [PJRT](https://openxla.org/xla/pjrt) plugin that runs [JAX](https://github.com/jax-ml/jax)
programs on Apple Silicon GPUs, using [MLX](https://github.com/ml-explore/mlx) as the execution engine
instead of XLA:GPU or Metal shaders written from scratch. Register the plugin and `jax.jit`, `jax.grad`,
and `jax.random` run on the `"mlx"` device with no other code changes.

This project is **not affiliated with Apple or Google**. It is an independent, community plugin.

## Architecture

A thin C "bridge" dylib implements the PJRT C API and forwards every interesting call (compile,
execute, buffer transfer) into Python via a registered dispatcher — the plugin runs inside the host
CPython process, so the bridge just acquires the GIL and calls back into Python. The Python core
deserializes the StableHLO program using jaxlib's own MLIR bindings (which guarantees version
agreement with whatever jaxlib is installed) and interprets the module directly as MLX operations.
Buffers are `mx.array`s held in a Python-side registry; the C side caches dtype/shape so hot PJRT
queries never have to touch Python.

```
 JAX program
     │  jax.jit(...)
     ▼
 XLA/PJRT compiles to StableHLO, calls into the bridge via the PJRT C API
     │
     ▼
 bridge/pjrt_mlx_bridge.c  (C, implements PJRT_Api, forwards to Python)
     │  ctypes dispatch, GIL held throughout
     ▼
 src/jax_plugins/mlx_plugin/
     translator.py   StableHLO module -> Python call graph
     ops.py           one handler per StableHLO op -> mx.* calls
     runtime.py       buffer / executable registries, dispatch() entry point
     dtypes.py        PJRT <-> mlx <-> numpy dtype mapping
     │
     ▼
 mlx.core executes on the GPU (Metal), results held as mx.array
```

Single device, single process, synchronous execution: there is no multi-device sharding or async
dispatch in v0.

## Install

```bash
pip install git+https://github.com/baileywickham/jax-mlx.git
```

This builds `bridge/pjrt_mlx_bridge.c` into a dylib at install time (via a `setup.py` build hook) and
packages it alongside the Python plugin. macOS on Apple Silicon (arm64) only — the loader is a no-op
on other platforms. Requires `jax>=0.7` and `mlx>=0.30`.

```python
import jax, jax.numpy as jnp
print(jax.devices())          # [MlxDevice(id=0)]  -- mlx registers at priority 500
a = jnp.ones((32, 32))
print((a @ a).sum())          # runs on the mlx device
```

## Op coverage (v0)

Elementwise: `add`, `subtract`, `multiply`, `divide`, `negate`, `abs`, `exponential`, `log`,
`log_plus_one`, `tanh`, `logistic`, `sqrt`, `rsqrt`, `sign`, `floor`, `ceil`, `cosine`, `sine`,
`power`, `maximum`, `minimum`, `and`, `or`, `xor`, `not`, `shift_left`, `shift_right_logical`,
`shift_right_arithmetic`, `select`, `remainder`, `compare`, and `chlo.erf_inv`.

Structural: `constant`, `convert`, `bitcast_convert`, `iota`, `reshape`, `transpose`,
`broadcast_in_dim`, `slice`, `concatenate`, `dot_general`, `reduce` (sum/product/max/min/any/all).

Control flow: `stablehlo.while` is interpreted (needed for `jax.random`'s threefry2x32 mixing loop);
`func.call`/`func.return` are supported as part of the interpreter's block-execution model.

Other: `stablehlo.custom_call @Sharding` is passed through as a no-op (JAX emits this to pin down
sharding on programs that don't otherwise need it). `jax.random` (`random.uniform`, `random.normal`)
works end-to-end.

The loader also disables JAX's Shardy partitioner (`jax_use_shardy_partitioner=False`) before
registering the plugin. Current JAX embeds `sdy` dialect attributes in the StableHLO it emits, even
for single-device programs, and this interpreter doesn't (and doesn't need to) understand that
dialect; without the disable, `jax.random` fails to deserialize.

## Non-goals for v0

The following are explicitly out of scope for this release and are tracked as follow-up work:

- `gather` / `scatter` (general forms)
- `sort`, `top_k`
- `reduce_window`
- `convolution`
- `case` / `cond` (structured control flow beyond `while`)
- `float64` and `complex64`/`complex128` dtypes (Metal/MLX limitation)
- Multi-device sharding, async dispatch

Programs that lower to any of these will raise `NotImplementedError` naming the missing op or dtype.

## Adding an op

1. Add a handler in `src/jax_plugins/mlx_plugin/ops.py`. For simple elementwise ops, use the
   `_elementwise("stablehlo.<name>", fn)` helper; for anything that needs attributes or multiple
   results, use `@register("stablehlo.<name>")` and read `op.attributes` / `op.operands` directly
   (see `_reduce` or `_dot_general` for examples of structural ops).
2. Add a golden test in `tests/test_translator_math.py` using `tests/harness.py`'s `check(...)`,
   which lowers the same JAX expression on CPU and asserts the mlx-backed result matches.
3. Run the full suite: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/ -v`.

## License

MIT. See `LICENSE`. `pjrt_c_api.h` is vendored from [openxla/xla](https://github.com/openxla/xla)
and is licensed under Apache 2.0; see the header in that file for its original copyright notice.
