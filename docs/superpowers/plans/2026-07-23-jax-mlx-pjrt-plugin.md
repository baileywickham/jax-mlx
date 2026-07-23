# jax-mlx PJRT Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An open-source PJRT plugin that runs real JAX programs on Apple Silicon GPUs using MLX as the execution engine.

**Architecture:** A thin C "bridge" dylib implements the PJRT C API and forwards every interesting call (compile, execute, buffer transfer) into Python via a registered dispatcher — the plugin runs inside the host CPython process, so the bridge just acquires the GIL and calls back. The Python core deserializes StableHLO using jaxlib's own MLIR bindings (guaranteed version agreement) and interprets the module as MLX operations. Buffers are `mx.array`s held in a Python-side registry; the C side caches dtype/shape so hot PJRT queries never touch Python.

**Tech Stack:** C (bridge, no deps beyond `pjrt_c_api.h` + Python C API), Python ≥3.10, `mlx` (pip), `jax`/`jaxlib` ≥0.7 (tested against 0.11), pytest, clang.

## Global Constraints

- macOS on Apple Silicon (arm64) only; loader must no-op on other platforms.
- No bazel, no LLVM/XLA source builds — the only C dependency is the vendored `pjrt_c_api.h` (already at repo root, copied from openxla/xla, Apache 2.0).
- Bridge compiles with: `clang -O2 -Wall -shared -undefined dynamic_lookup -I<python-include> -o src/jax_plugins/mlx_plugin/pjrt_mlx_bridge.dylib bridge/pjrt_mlx_bridge.c` (`-undefined dynamic_lookup` resolves Python C API symbols from the host process — never link libpython).
- Platform name is exactly `"mlx"` (lowercase) everywhere: `xb.register_plugin("mlx", ...)`, `jax.devices("mlx")`.
- PJRT error codes follow absl: 12 = UNIMPLEMENTED, 13 = INTERNAL.
- Single device, single process, synchronous execution: `num_replicas = num_partitions = 1`, all PJRT events are created already-ready.
- Unsupported dtypes raise `NotImplementedError` with the dtype name (float64, complex — MLX/Metal limitation).
- v0 op scope is exactly the ops listed in Tasks 3–4. `while`, `case`/`cond`, `gather`, `scatter`, `sort`, `top_k`, `reduce_window`, `convolution` are FOLLOW-UP plans — do not add speculative stubs for them.
- License MIT. Repo root: `~/workspace/jax-mlx`.
- All pytest runs use the repo venv: `.venv/bin/python -m pytest` (Task 1 creates it with `jax`, `mlx`, `numpy`, `pytest`).
- Tests that load the bridge into jaxlib run as **subprocess scripts** (a C bug must fail a test, not kill pytest).

## File Structure

```
jax-mlx/
  pjrt_c_api.h                          # vendored (already present)
  bridge/pjrt_mlx_bridge.c              # the entire C bridge (single file)
  src/jax_plugins/mlx_plugin/
    __init__.py                         # loader: build-path lookup, dispatcher install, register_plugin
    dtypes.py                           # PJRT enum <-> mlx dtype <-> numpy mapping
    translator.py                       # StableHLO module -> python callable of mx ops
    ops.py                              # op registry: one handler per StableHLO op
    runtime.py                          # buffer/executable registries + dispatch() entry point
  tests/
    test_dtypes.py
    test_translator_basic.py
    test_translator_math.py
    harness.py                          # golden-test helper (lower via jax CPU, compare)
    test_runtime.py
    integration/
      check_devices.py                  # subprocess scripts, exit 0 on success
      check_buffers.py
      check_execute.py
    test_integration.py                 # pytest wrapper running the scripts
  pyproject.toml, setup.py, README.md, LICENSE, .gitignore
```

---

### Task 1: Repo scaffold + dtype mapping

**Files:**
- Create: `pyproject.toml`, `setup.py`, `.gitignore`, `LICENSE` (MIT, © 2026 Bailey Wickham)
- Create: `src/jax_plugins/mlx_plugin/__init__.py` (empty for now), `src/jax_plugins/mlx_plugin/dtypes.py`
- Test: `tests/test_dtypes.py`

**Interfaces:**
- Produces: `dtypes.PJRT` (IntEnum of PJRT_Buffer_Type values), `dtypes.pjrt_to_mlx(code: int) -> mx.Dtype`, `dtypes.mlx_to_pjrt(dt: mx.Dtype) -> int`, `dtypes.pjrt_to_np(code: int) -> np.dtype`, `dtypes.np_to_pjrt(dt) -> int`, `dtypes.ir_type_to_pjrt(ir_type_str: str) -> int` (maps MLIR element type strings like `"f32"`, `"i32"`, `"i1"`, `"bf16"`, `"ui8"`).

- [ ] **Step 1: venv + deps**

```bash
cd ~/workspace/jax-mlx && git init -q
python3.12 -m venv .venv && .venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q jax mlx numpy pytest ml_dtypes
```

- [ ] **Step 2: scaffold files**

`.gitignore`:
```
__pycache__/
*.egg-info/
build/
dist/
.venv/
src/jax_plugins/mlx_plugin/*.dylib
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "jax-mlx"
version = "0.1.0"
description = "JAX on Apple Silicon GPUs via a PJRT plugin backed by MLX"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
dependencies = ["jax>=0.7", "mlx>=0.30", "numpy", "ml_dtypes"]

[tool.setuptools]
package-dir = { "" = "src" }
packages = ["jax_plugins.mlx_plugin"]

[tool.setuptools.package-data]
"jax_plugins.mlx_plugin" = ["*.dylib"]
```

`setup.py`:
```python
import pathlib, subprocess, sys, sysconfig
from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = pathlib.Path(__file__).resolve().parent

class BuildBridge(build_py):
    def run(self):
        if sys.platform == "darwin":
            out = ROOT / "src/jax_plugins/mlx_plugin/pjrt_mlx_bridge.dylib"
            subprocess.check_call([
                "clang", "-O2", "-Wall", "-shared", "-undefined", "dynamic_lookup",
                f"-I{sysconfig.get_paths()['include']}", f"-I{ROOT}",
                str(ROOT / "bridge/pjrt_mlx_bridge.c"), "-o", str(out)])
        super().run()

setup(cmdclass={"build_py": BuildBridge})
```

- [ ] **Step 3: write the failing dtype test**

`tests/test_dtypes.py`:
```python
import mlx.core as mx
import numpy as np
from jax_plugins.mlx_plugin import dtypes


def test_roundtrip_mlx():
    for dt in [mx.float32, mx.float16, mx.bfloat16, mx.int32, mx.int64,
               mx.int16, mx.int8, mx.uint8, mx.uint16, mx.uint32, mx.uint64,
               mx.bool_]:
        assert dtypes.pjrt_to_mlx(dtypes.mlx_to_pjrt(dt)) == dt


def test_np_f32():
    code = dtypes.np_to_pjrt(np.dtype("float32"))
    assert code == dtypes.PJRT.F32 == 11
    assert dtypes.pjrt_to_np(code) == np.dtype("float32")


def test_ir_strings():
    assert dtypes.ir_type_to_pjrt("f32") == dtypes.PJRT.F32
    assert dtypes.ir_type_to_pjrt("i1") == dtypes.PJRT.PRED
    assert dtypes.ir_type_to_pjrt("i32") == dtypes.PJRT.S32
    assert dtypes.ir_type_to_pjrt("ui8") == dtypes.PJRT.U8
    assert dtypes.ir_type_to_pjrt("bf16") == dtypes.PJRT.BF16


def test_unsupported_raises():
    import pytest
    with pytest.raises(NotImplementedError, match="f64"):
        dtypes.ir_type_to_pjrt("f64")
```

- [ ] **Step 4: run to verify failure**

Run: `cd ~/workspace/jax-mlx && PYTHONPATH=src .venv/bin/python -m pytest tests/test_dtypes.py -v`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError`)

- [ ] **Step 5: implement `dtypes.py`**

The enum values are the declaration order of `PJRT_Buffer_Type` in `pjrt_c_api.h` (verify: `grep -n -A20 "typedef enum PJRT_Buffer_Type" pjrt_c_api.h` — INVALID=0, PRED=1, S8..S64=2..5, U8..U64=6..9, F16=10, F32=11, F64=12, BF16=13, C64=14, C128=15, then F8 types).

```python
import enum
import mlx.core as mx
import numpy as np
import ml_dtypes


class PJRT(enum.IntEnum):
    INVALID = 0; PRED = 1
    S8 = 2; S16 = 3; S32 = 4; S64 = 5
    U8 = 6; U16 = 7; U32 = 8; U64 = 9
    F16 = 10; F32 = 11; F64 = 12; BF16 = 13
    C64 = 14; C128 = 15


_MLX = {
    PJRT.PRED: mx.bool_, PJRT.S8: mx.int8, PJRT.S16: mx.int16,
    PJRT.S32: mx.int32, PJRT.S64: mx.int64, PJRT.U8: mx.uint8,
    PJRT.U16: mx.uint16, PJRT.U32: mx.uint32, PJRT.U64: mx.uint64,
    PJRT.F16: mx.float16, PJRT.F32: mx.float32, PJRT.BF16: mx.bfloat16,
}
_MLX_INV = {v: k for k, v in _MLX.items()}

_NP = {
    PJRT.PRED: np.dtype("bool"), PJRT.S8: np.dtype("int8"),
    PJRT.S16: np.dtype("int16"), PJRT.S32: np.dtype("int32"),
    PJRT.S64: np.dtype("int64"), PJRT.U8: np.dtype("uint8"),
    PJRT.U16: np.dtype("uint16"), PJRT.U32: np.dtype("uint32"),
    PJRT.U64: np.dtype("uint64"), PJRT.F16: np.dtype("float16"),
    PJRT.F32: np.dtype("float32"), PJRT.BF16: np.dtype(ml_dtypes.bfloat16),
}
_NP_INV = {v: k for k, v in _NP.items()}

_IR = {
    "i1": PJRT.PRED, "i8": PJRT.S8, "i16": PJRT.S16, "i32": PJRT.S32,
    "i64": PJRT.S64, "ui8": PJRT.U8, "ui16": PJRT.U16, "ui32": PJRT.U32,
    "ui64": PJRT.U64, "f16": PJRT.F16, "f32": PJRT.F32, "bf16": PJRT.BF16,
}


def _lookup(table, key, kind):
    try:
        return table[key]
    except KeyError:
        raise NotImplementedError(f"jax-mlx: unsupported {kind} dtype: {key}")


def pjrt_to_mlx(code): return _lookup(_MLX, PJRT(code), "pjrt")
def mlx_to_pjrt(dt): return int(_lookup(_MLX_INV, dt, "mlx"))
def pjrt_to_np(code): return _lookup(_NP, PJRT(code), "pjrt")
def np_to_pjrt(dt): return int(_lookup(_NP_INV, np.dtype(dt), "numpy"))
def ir_type_to_pjrt(s): return int(_lookup(_IR, s, "ir"))
```

- [ ] **Step 6: run tests to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_dtypes.py -v`
Expected: 4 PASS

- [ ] **Step 7: commit**

```bash
git add -A && git commit -m "feat: scaffold jax-mlx repo with dtype mapping"
```

---

### Task 2: StableHLO parsing + golden-test harness

**Files:**
- Create: `src/jax_plugins/mlx_plugin/translator.py` (parsing half), `tests/harness.py`
- Test: `tests/test_translator_basic.py`

**Interfaces:**
- Consumes: `dtypes.ir_type_to_pjrt`
- Produces:
  - `translator.parse_module(payload: bytes | str) -> ir.Module` — accepts StableHLO portable-artifact bytes, raw MLIR bytecode, or MLIR text.
  - `translator.main_func(module) -> ir.Operation` — the `func.func @main` op.
  - `translator.result_specs(module) -> list[tuple[int, tuple[int, ...]]]` — `(pjrt_dtype, shape)` per result of `@main`.
  - `harness.lower(f, *args) -> str` — StableHLO text of `jax.jit(f).lower(*args)` on the CPU backend.
  - `harness.check(f, *args, rtol=1e-5, atol=1e-6)` — lowers `f`, translates+runs on MLX (uses `translator.compile_module`, implemented in Task 3), compares against `jax.jit(f)(*args)` with `np.testing.assert_allclose`. Write it now; it stays red until Task 3.

- [ ] **Step 1: write failing tests**

`tests/test_translator_basic.py`:
```python
import jax
import jax.numpy as jnp
from jax_plugins.mlx_plugin import translator
from tests.harness import lower


def test_parse_text_and_main():
    text = lower(lambda x: x + x, jnp.ones((3,), jnp.float32))
    module = translator.parse_module(text)
    fn = translator.main_func(module)
    assert fn.name == "func.func"


def test_result_specs():
    text = lower(lambda x: (x + x).astype(jnp.int32), jnp.ones((2, 3), jnp.float32))
    module = translator.parse_module(text)
    assert translator.result_specs(module) == [(4, (2, 3))]  # 4 == PJRT.S32


def test_parse_portable_artifact():
    from jaxlib.mlir.dialects import stablehlo
    text = lower(lambda x: x * 2, jnp.ones((2,), jnp.float32))
    artifact = stablehlo.serialize_portable_artifact_str(
        text, stablehlo.get_current_version())
    module = translator.parse_module(bytes(artifact))
    assert translator.main_func(module) is not None
```

`tests/harness.py`:
```python
import jax
import numpy as np


def lower(f, *args):
    return str(jax.jit(f).lower(*args).compiler_ir())


def check(f, *args, rtol=1e-5, atol=1e-6):
    from jax_plugins.mlx_plugin import translator
    import mlx.core as mx
    expected = jax.jit(f)(*args)
    expected = expected if isinstance(expected, (list, tuple)) else [expected]
    compiled = translator.compile_module(translator.parse_module(lower(f, *args)))
    got = compiled([mx.array(np.asarray(a)) for a in args])
    assert len(got) == len(expected)
    for e, g in zip(expected, got):
        np.testing.assert_allclose(
            np.asarray(e), np.array(g, copy=False), rtol=rtol, atol=atol)
```

- [ ] **Step 2: run to verify failure**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_translator_basic.py -v`
Expected: FAIL (`translator` has no `parse_module`)

- [ ] **Step 3: implement the parsing half of `translator.py`**

```python
"""StableHLO -> MLX. Parsing/introspection here; op semantics in ops.py."""
from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from . import dtypes

_PORTABLE_MAGIC = b"ML\xefR"  # MLIR bytecode magic; artifacts are vhlo bytecode


def parse_module(payload):
    if isinstance(payload, str):
        data = payload
    else:
        data = bytes(payload)
        if data.startswith(_PORTABLE_MAGIC):
            # Portable artifact (vhlo). jaxlib's own deserializer upgrades it
            # to current StableHLO -- version agreement is guaranteed because
            # the same jaxlib produced it.
            data = stablehlo.deserialize_portable_artifact_str(data)
    with make_ir_context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(data)
        module._jaxmlx_ctx = ctx  # keep context alive with the module
        return module


def main_func(module):
    for op in module.body.operations:
        if op.name == "func.func" and ir.StringAttr(op.attributes["sym_name"]).value == "main":
            return op
    raise ValueError("no @main in module")


def _tensor_spec(t):
    rt = ir.RankedTensorType(t)
    return (dtypes.ir_type_to_pjrt(str(rt.element_type)), tuple(rt.shape))


def result_specs(module):
    fn = main_func(module)
    ftype = ir.FunctionType(ir.TypeAttr(fn.attributes["function_type"]).value)
    return [_tensor_spec(t) for t in ftype.results]
```

- [ ] **Step 4: run tests to verify pass**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_translator_basic.py -v`
Expected: 3 PASS. If `module._jaxmlx_ctx` assignment fails (nanobind slots), wrap instead: return a small `ParsedModule` dataclass holding both `module` and `ctx`, and update `main_func`/`result_specs` to take it — keep the same public names.

- [ ] **Step 5: commit**

```bash
git add -A && git commit -m "feat: StableHLO parsing and golden-test harness"
```

---

### Task 3: Interpreter core + elementwise/structural ops

**Files:**
- Create: `src/jax_plugins/mlx_plugin/ops.py`
- Modify: `src/jax_plugins/mlx_plugin/translator.py` (add `compile_module`)
- Test: `tests/test_translator_math.py`

**Interfaces:**
- Consumes: `parse_module`, `main_func`, `harness.check`
- Produces:
  - `translator.compile_module(module) -> Callable[[list[mx.array]], list[mx.array]]`
  - `ops.HANDLERS: dict[str, Callable[[op, list[mx.array]], mx.array | list[mx.array]]]` — key is the full op name (`"stablehlo.add"`); handler receives the `ir.Operation` and already-evaluated operand arrays, returns result array(s).
  - `ops.register(name)` decorator.

- [ ] **Step 1: write failing golden tests**

`tests/test_translator_math.py` (first batch):
```python
import jax
import jax.numpy as jnp
import numpy as np
from tests.harness import check

x32 = jnp.linspace(0.5, 2.0, 12, dtype=jnp.float32)
m = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4)


def test_add_mul_sub_div(): check(lambda x: (x + x) * x - x / 2, x32)
def test_neg_abs(): check(lambda x: -abs(x - 1), x32)
def test_exp_log_tanh_sqrt(): check(lambda x: jnp.tanh(jnp.log(jnp.exp(x))) + jnp.sqrt(x))
def test_rsqrt_pow(): check(lambda x: jax.lax.rsqrt(x) ** 2, x32)
def test_max_min(): check(lambda x: jnp.maximum(x, 1.0) + jnp.minimum(x, 1.0), x32)
def test_compare_select(): check(lambda x: jnp.where(x > 1.0, x, -x), x32)
def test_convert(): check(lambda x: x.astype(jnp.int32).astype(jnp.float32), x32)
def test_constant_splat(): check(lambda x: x + 3.0, x32)
def test_constant_dense(): check(lambda x: x + jnp.array([1.0, 2.0, 3.0, 4.0]), m)
def test_iota(): check(lambda: jnp.arange(10))
def test_reshape_transpose(): check(lambda a: a.reshape(4, 3).T, m)
def test_broadcast(): check(lambda a: a + jnp.ones((1, 4)), m)
def test_slice(): check(lambda x: x[2:9:2], x32)
def test_concat(): check(lambda x: jnp.concatenate([x, x]), x32)
def test_bitwise_shift():
    u = jnp.arange(8, dtype=jnp.uint32)
    check(lambda a: ((a ^ 21) | 3) & (a << 2) ^ (a >> 1), u)
def test_bitcast():
    u = jnp.arange(4, dtype=jnp.uint32)
    check(lambda a: jax.lax.bitcast_convert_type(a, jnp.int32), u)
def test_multiple_results(): check(lambda x: (x + 1, x * 2), x32)
```

- [ ] **Step 2: run to verify failure**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_translator_math.py -v`
Expected: all FAIL (`translator` has no `compile_module`)

- [ ] **Step 3: implement interpreter core in `translator.py`**

Append:
```python
def compile_module(module):
    from . import ops
    fn = main_func(module)
    body = fn.regions[0].blocks[0]

    def run(inputs):
        env = {}
        for arg, val in zip(body.arguments, inputs):
            env[arg] = val
        outs = None
        for op in body.operations:
            if op.name == "func.return":
                outs = [env[v] for v in op.operands]
                break
            handler = ops.HANDLERS.get(op.name)
            if handler is None:
                raise NotImplementedError(f"jax-mlx: unsupported op {op.name}")
            results = handler(op, [env[v] for v in op.operands])
            if not isinstance(results, list):
                results = [results]
            for res_value, res_array in zip(op.results, results):
                env[res_value] = res_array
        return outs

    return run
```

(`ir.Value` is hashable by identity in the MLIR nanobind bindings; `test_reshape_transpose` exercises multi-step chains and will catch it if not — fallback is keying `env` on `v.get_name()` strings.)

- [ ] **Step 4: implement `ops.py`**

```python
"""One handler per StableHLO op. Handlers get (ir.Operation, [mx.array])."""
import mlx.core as mx
import numpy as np
from jaxlib.mlir import ir

from . import dtypes

HANDLERS = {}


def register(name):
    def deco(fn):
        HANDLERS[name] = fn
        return fn
    return deco


def _elementwise(name, fn):
    HANDLERS[name] = lambda op, args: fn(*args)


_elementwise("stablehlo.add", lambda a, b: a + b)
_elementwise("stablehlo.subtract", lambda a, b: a - b)
_elementwise("stablehlo.multiply", lambda a, b: a * b)
_elementwise("stablehlo.divide", lambda a, b: a / b)
_elementwise("stablehlo.negate", lambda a: -a)
_elementwise("stablehlo.abs", mx.abs)
_elementwise("stablehlo.exponential", mx.exp)
_elementwise("stablehlo.log", mx.log)
_elementwise("stablehlo.tanh", mx.tanh)
_elementwise("stablehlo.logistic", mx.sigmoid)
_elementwise("stablehlo.sqrt", mx.sqrt)
_elementwise("stablehlo.rsqrt", mx.rsqrt)
_elementwise("stablehlo.sign", mx.sign)
_elementwise("stablehlo.floor", mx.floor)
_elementwise("stablehlo.ceil", mx.ceil)
_elementwise("stablehlo.cosine", mx.cos)
_elementwise("stablehlo.sine", mx.sin)
_elementwise("stablehlo.power", mx.power)
_elementwise("stablehlo.maximum", mx.maximum)
_elementwise("stablehlo.minimum", mx.minimum)
_elementwise("stablehlo.and", lambda a, b: a & b)
_elementwise("stablehlo.or", lambda a, b: a | b)
_elementwise("stablehlo.xor", lambda a, b: a ^ b)
_elementwise("stablehlo.not", lambda a: ~a)
_elementwise("stablehlo.shift_left", mx.left_shift)
_elementwise("stablehlo.shift_right_logical", mx.right_shift)  # note below
_elementwise("stablehlo.shift_right_arithmetic", mx.right_shift)
_elementwise("stablehlo.select", lambda p, t, f: mx.where(p, t, f))
_elementwise("stablehlo.remainder", mx.remainder)

# shift_right_logical on signed ints must zero-fill: view as unsigned first.
def _srl(op, args):
    a, s = args
    if a.dtype in (mx.int8, mx.int16, mx.int32, mx.int64):
        u = {mx.int8: mx.uint8, mx.int16: mx.uint16,
             mx.int32: mx.uint32, mx.int64: mx.uint64}[a.dtype]
        return mx.right_shift(a.view(u), s.view(u)).view(a.dtype)
    return mx.right_shift(a, s)
HANDLERS["stablehlo.shift_right_logical"] = _srl


_CMP = {"EQ": lambda a, b: a == b, "NE": lambda a, b: a != b,
        "LT": lambda a, b: a < b, "LE": lambda a, b: a <= b,
        "GT": lambda a, b: a > b, "GE": lambda a, b: a >= b}

@register("stablehlo.compare")
def _compare(op, args):
    direction = str(op.attributes["comparison_direction"]).split("<")[1].split(" ")[1].rstrip(">")
    return _CMP[direction](*args)


def _result_type(op, i=0):
    return ir.RankedTensorType(op.results[i].type)


def _mlx_result_dtype(op, i=0):
    return dtypes.pjrt_to_mlx(
        dtypes.ir_type_to_pjrt(str(_result_type(op, i).element_type)))


@register("stablehlo.constant")
def _constant(op, args):
    attr = op.attributes["value"]
    dense = ir.DenseElementsAttr(attr)
    shape = tuple(_result_type(op).shape)
    np_dt = dtypes.pjrt_to_np(
        dtypes.ir_type_to_pjrt(str(_result_type(op).element_type)))
    if dense.is_splat:
        val = np.array(dense.get_splat_value(), dtype=np_dt)
        arr = np.broadcast_to(val, shape)
    else:
        arr = np.array(dense, copy=False).astype(np_dt, copy=False).reshape(shape)
    return mx.array(np.ascontiguousarray(arr))


@register("stablehlo.convert")
def _convert(op, args):
    return args[0].astype(_mlx_result_dtype(op))


@register("stablehlo.bitcast_convert")
def _bitcast(op, args):
    return args[0].view(_mlx_result_dtype(op))


@register("stablehlo.iota")
def _iota(op, args):
    rt = _result_type(op)
    dim = ir.IntegerAttr(op.attributes["iota_dimension"]).value
    n = rt.shape[dim]
    r = mx.arange(n, dtype=_mlx_result_dtype(op))
    view = [1] * len(rt.shape)
    view[dim] = n
    return mx.broadcast_to(r.reshape(view), tuple(rt.shape))


@register("stablehlo.reshape")
def _reshape(op, args):
    return args[0].reshape(tuple(_result_type(op).shape))


def _i64_array_attr(op, name):
    return [ir.IntegerAttr(a).value
            for a in ir.ArrayAttr(op.attributes[name])] \
        if isinstance(op.attributes[name], ir.ArrayAttr) \
        else list(ir.DenseI64ArrayAttr(op.attributes[name]))


@register("stablehlo.transpose")
def _transpose(op, args):
    return mx.transpose(args[0], _i64_array_attr(op, "permutation"))


@register("stablehlo.broadcast_in_dim")
def _broadcast_in_dim(op, args):
    out_shape = tuple(_result_type(op).shape)
    bdims = _i64_array_attr(op, "broadcast_dimensions")
    view = [1] * len(out_shape)
    for src, dst in enumerate(bdims):
        view[dst] = args[0].shape[src]
    return mx.broadcast_to(args[0].reshape(view), out_shape)


@register("stablehlo.slice")
def _slice(op, args):
    starts = _i64_array_attr(op, "start_indices")
    limits = _i64_array_attr(op, "limit_indices")
    strides = _i64_array_attr(op, "strides")
    idx = tuple(slice(s, l, st) for s, l, st in zip(starts, limits, strides))
    return args[0][idx]


@register("stablehlo.concatenate")
def _concatenate(op, args):
    dim = ir.IntegerAttr(op.attributes["dimension"]).value
    return mx.concatenate(args, axis=dim)
```

- [ ] **Step 5: run tests, iterate on attribute-access details**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_translator_math.py -v`
Expected: all PASS. Two known fragile spots to fix by inspection if red (use
`PYTHONPATH=src:. .venv/bin/python -c "from tests.harness import lower; import jax.numpy as jnp; print(lower(lambda x: x > 1.0, jnp.ones(3)))"` to see real attribute syntax):
- `comparison_direction` parsing — the attribute prints like `#stablehlo<comparison_direction GT>`; adjust the string split to match what `str(attr)` actually returns.
- `permutation`/`broadcast_dimensions` may be `DenseI64ArrayAttr` (`array<i64: 1, 0>`) rather than `ArrayAttr` depending on version — `_i64_array_attr` handles both; verify both branches compile.

- [ ] **Step 6: commit**

```bash
git add -A && git commit -m "feat: MLX interpreter core with elementwise and structural ops"
```

---

### Task 4: dot_general + reduce (+ PRNG golden test)

**Files:**
- Modify: `src/jax_plugins/mlx_plugin/ops.py`
- Test: append to `tests/test_translator_math.py`

**Interfaces:**
- Consumes: `ops.register`, `harness.check`
- Produces: handlers for `stablehlo.dot_general`, `stablehlo.reduce`; after this task the golden suite covers matmul, reductions, grad, and `jax.random`.

- [ ] **Step 1: write failing tests**

Append to `tests/test_translator_math.py`:
```python
def test_matmul(): check(lambda a: a @ a.T, m)
def test_batched_matmul():
    b = jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4)
    check(lambda a: jnp.einsum("bij,bkj->bik", a, a), b)
def test_reduce_sum(): check(lambda a: a.sum(), m)
def test_reduce_axis(): check(lambda a: a.sum(axis=1), m)
def test_reduce_max(): check(lambda a: a.max(axis=0), m)
def test_reduce_prod(): check(lambda a: (a + 1).prod(axis=1) / 1e4, m)
def test_reduce_any():
    check(lambda a: (a > 5.0).any(axis=0), m)
def test_grad(): check(jax.grad(lambda x: jnp.tanh(x).sum()), x32)
def test_mlp_grad():
    w = jnp.ones((4, 8), jnp.float32) * 0.1
    check(jax.grad(lambda w: jnp.tanh(m @ w).sum()), w)
def test_random_uniform():
    check(lambda: jax.random.uniform(jax.random.key(0), (8,)))
def test_random_normal():
    check(lambda: jax.random.normal(jax.random.key(1), (4, 4)))
```

- [ ] **Step 2: run to verify failure**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_translator_math.py -k "matmul or reduce or grad or random" -v`
Expected: FAIL with `NotImplementedError: jax-mlx: unsupported op stablehlo.dot_general` (and `stablehlo.reduce`)

- [ ] **Step 3: implement dot_general via einsum construction**

Append to `ops.py`:
```python
import string


@register("stablehlo.dot_general")
def _dot_general(op, args):
    lhs, rhs = args
    dn = ir.Attribute(op.attributes["dot_dimension_numbers"])
    # Robust across binding versions: parse the printed form
    # "#stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
    #   lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>"
    import re
    text = str(dn)
    def dims(key):
        mobj = re.search(key + r"\s*=\s*\[([0-9,\s]*)\]", text)
        return [int(x) for x in mobj.group(1).split(",")] if mobj and mobj.group(1).strip() else []
    lb, rb = dims("lhs_batching_dimensions"), dims("rhs_batching_dimensions")
    lc, rc = dims("lhs_contracting_dimensions"), dims("rhs_contracting_dimensions")

    letters = iter(string.ascii_letters)
    lhs_l = [None] * len(lhs.shape)
    rhs_l = [None] * len(rhs.shape)
    out_l = []
    for i, j in zip(lb, rb):
        c = next(letters); lhs_l[i] = rhs_l[j] = c; out_l.append(c)
    for i, j in zip(lc, rc):
        c = next(letters); lhs_l[i] = rhs_l[j] = c
    for i in range(len(lhs.shape)):
        if lhs_l[i] is None:
            lhs_l[i] = next(letters); out_l.append(lhs_l[i])
    for j in range(len(rhs.shape)):
        if rhs_l[j] is None:
            rhs_l[j] = next(letters); out_l.append(rhs_l[j])
    spec = f"{''.join(lhs_l)},{''.join(rhs_l)}->{''.join(out_l)}"
    return mx.einsum(spec, lhs, rhs).astype(_mlx_result_dtype(op))


_REDUCERS = {
    "stablehlo.add": mx.sum, "stablehlo.multiply": mx.prod,
    "stablehlo.maximum": mx.max, "stablehlo.minimum": mx.min,
    "stablehlo.or": mx.any, "stablehlo.and": mx.all,
}


@register("stablehlo.reduce")
def _reduce(op, args):
    n = len(op.results)
    operands, _inits = args[:n], args[n:]
    dims = _i64_array_attr(op, "dimensions")
    block = op.regions[0].blocks[0]
    body = [o for o in block.operations if o.name != "stablehlo.return"]
    if n != 1 or len(body) != 1 or body[0].name not in _REDUCERS:
        names = [o.name for o in body]
        raise NotImplementedError(f"jax-mlx: general reduce region {names}")
    return _REDUCERS[body[0].name](operands[0], axis=tuple(dims))
```

- [ ] **Step 4: run tests**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_translator_math.py -v`
Expected: all PASS. Likely iteration points: (a) `jax.random` lowering may include `stablehlo.dynamic_slice` or `stablehlo.pad` — if so, print the module (`harness.lower`) and add those two handlers (`dynamic_slice`: clamp start operands with `mx.clip(int(start), 0, dim-size)` then slice; `pad`: `mx.pad` with `edge_padding_low/high`, rejecting nonzero `interior_padding` with `NotImplementedError`); (b) reduce over `bool` with `or` — mx.any returns bool, matches.

- [ ] **Step 5: commit**

```bash
git add -A && git commit -m "feat: dot_general and reduce; golden suite covers matmul, grad, PRNG"
```

---

### Task 5: Runtime registries + dispatcher

**Files:**
- Create: `src/jax_plugins/mlx_plugin/runtime.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `translator.parse_module`, `translator.compile_module`, `translator.result_specs`, `dtypes.*`
- Produces (this is the exact contract the C bridge calls — signatures are frozen here):
  - `runtime.dispatch(method: str, args: tuple) -> object` — module-level entry point handed to C.
  - `compile(code: bytes) -> tuple[int, list[tuple[int, tuple[int, ...]]]]` — returns `(exec_id, [(pjrt_dtype, dims), ...])`.
  - `execute(exec_id: int, arg_ids: tuple[int, ...]) -> list[tuple[int, int, tuple[int, ...]]]` — returns `[(buf_id, pjrt_dtype, dims), ...]`.
  - `buffer_from_host(data: bytes, pjrt_dtype: int, dims: tuple[int, ...]) -> int`
  - `buffer_to_host(buf_id: int) -> bytes`
  - `buffer_delete(buf_id: int) -> None`
  - `stablehlo_version() -> tuple[int, int, int]`

- [ ] **Step 1: write failing tests**

`tests/test_runtime.py`:
```python
import numpy as np
import jax.numpy as jnp
from jax_plugins.mlx_plugin import runtime
from tests.harness import lower


def test_buffer_roundtrip():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    bid = runtime.dispatch("buffer_from_host", (a.tobytes(), 11, (2, 3)))
    out = runtime.dispatch("buffer_to_host", (bid,))
    np.testing.assert_array_equal(np.frombuffer(out, np.float32).reshape(2, 3), a)
    runtime.dispatch("buffer_delete", (bid,))


def test_compile_execute():
    text = lower(lambda x: (x @ x.T).sum(), jnp.ones((3, 4), jnp.float32))
    exec_id, out_specs = runtime.dispatch("compile", (text.encode(),))
    assert out_specs == [(11, ())]
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    bid = runtime.dispatch("buffer_from_host", (a.tobytes(), 11, (3, 4)))
    [(rid, dtype, dims)] = runtime.dispatch("execute", (exec_id, (bid,)))
    assert (dtype, dims) == (11, ())
    val = np.frombuffer(runtime.dispatch("buffer_to_host", (rid,)), np.float32)
    np.testing.assert_allclose(val[0], (a @ a.T).sum(), rtol=1e-5)


def test_error_propagates():
    import pytest
    with pytest.raises(Exception):
        runtime.dispatch("compile", (b"not a module",))
```

- [ ] **Step 2: run to verify failure**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_runtime.py -v`
Expected: FAIL (no `runtime` module)

- [ ] **Step 3: implement `runtime.py`**

```python
"""Object registries + the dispatch() entry point called from the C bridge."""
import itertools
import threading

import mlx.core as mx
import numpy as np

from . import dtypes, translator

_lock = threading.Lock()
_ids = itertools.count(1)
_buffers: dict[int, mx.array] = {}
_executables: dict[int, tuple] = {}  # id -> (callable, out_specs)


def buffer_from_host(data, pjrt_dtype, dims):
    np_dt = dtypes.pjrt_to_np(pjrt_dtype)
    arr = np.frombuffer(data, dtype=np_dt).reshape(dims)
    a = mx.array(arr)
    with _lock:
        bid = next(_ids)
        _buffers[bid] = a
    return bid


def buffer_to_host(buf_id):
    a = _buffers[buf_id]
    return np.array(a, copy=False).tobytes()


def buffer_delete(buf_id):
    with _lock:
        _buffers.pop(buf_id, None)


def compile(code):
    module = translator.parse_module(code)
    fn = translator.compile_module(module)
    out_specs = translator.result_specs(module)
    with _lock:
        eid = next(_ids)
        _executables[eid] = (fn, out_specs)
    return eid, out_specs


def execute(exec_id, arg_ids):
    fn, out_specs = _executables[exec_id]
    outs = fn([_buffers[i] for i in arg_ids])
    results = []
    with _lock:
        for a, (want_dtype, want_dims) in zip(outs, out_specs):
            bid = next(_ids)
            _buffers[bid] = a
            results.append((bid, dtypes.mlx_to_pjrt(a.dtype), tuple(a.shape)))
    return results


def stablehlo_version():
    from jaxlib.mlir.dialects import stablehlo
    return tuple(int(x) for x in stablehlo.get_current_version().split("."))


_METHODS = {f.__name__: f for f in
            [buffer_from_host, buffer_to_host, buffer_delete,
             compile, execute, stablehlo_version]}


def dispatch(method, args):
    return _METHODS[method](*args)
```

- [ ] **Step 4: run tests to verify pass**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_runtime.py tests/ -v`
Expected: full suite PASS

- [ ] **Step 5: commit**

```bash
git add -A && git commit -m "feat: runtime registries and C-facing dispatch contract"
```

---

### Task 6: C bridge part A — errors, events, client, device, memory

**Files:**
- Create: `bridge/pjrt_mlx_bridge.c`
- Create: `tests/integration/check_devices.py`, `tests/test_integration.py`
- Modify: `src/jax_plugins/mlx_plugin/__init__.py` (loader)

**Interfaces:**
- Consumes: `runtime.dispatch`, `runtime.stablehlo_version`
- Produces:
  - exported C symbols: `GetPjrtApi(void)`, `JaxMlxInstallDispatcher(PyObject*)`, `JaxMlxSetStablehloVersion(long, long, long)`
  - `jax_plugins.mlx_plugin.initialize()` — the JAX auto-discovery hook: loads the dylib via ctypes, installs the dispatcher, sets the stablehlo version, calls `xb.register_plugin("mlx", priority=500, library_path=...)`.
  - C helpers used by Task 7/8 in the same file: `err_new(int, const char*)`, `event_new(void)`, `call_py(const char*, PyObject* args_tuple_stolen, PyObject** out_new_ref)`

- [ ] **Step 1: write the failing integration test**

`tests/integration/check_devices.py`:
```python
import jax

devs = jax.devices("mlx")
assert len(devs) == 1, devs
d = devs[0]
assert d.platform == "mlx", d.platform
assert d.id == 0
assert d.default_memory().kind == "device"
print("DEVICES-OK")
```

`tests/test_integration.py`:
```python
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_check(name):
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}")
    p = subprocess.run(
        [sys.executable, str(ROOT / "tests/integration" / name)],
        capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == 0, f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    return p.stdout


def test_devices():
    assert "DEVICES-OK" in run_check("check_devices.py")
```

- [ ] **Step 2: run to verify failure**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_integration.py -v`
Expected: FAIL (no dylib / `jax.devices("mlx")` raises)

- [ ] **Step 3: write the bridge (part A)**

`bridge/pjrt_mlx_bridge.c`:
```c
// jax-mlx PJRT bridge: implements the PJRT C API and forwards compile /
// execute / buffer traffic to Python (runtime.dispatch) via the CPython API.
// The plugin runs inside the host CPython process; -undefined dynamic_lookup
// resolves Py* symbols at load time.
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pjrt_c_api.h"

// ---------- errors ----------
typedef struct { int code; char msg[1024]; } Error;

static PJRT_Error* err_new(int code, const char* msg) {
  Error* e = calloc(1, sizeof(Error));
  e->code = code;
  snprintf(e->msg, sizeof(e->msg), "%s", msg ? msg : "(null)");
  return (PJRT_Error*)e;
}
static void Bridge_Error_Destroy(PJRT_Error_Destroy_Args* a) { free(a->error); }
static void Bridge_Error_Message(PJRT_Error_Message_Args* a) {
  Error* e = (Error*)a->error;
  a->message = e->msg; a->message_size = strlen(e->msg);
}
static PJRT_Error* Bridge_Error_GetCode(PJRT_Error_GetCode_Args* a) {
  a->code = (PJRT_Error_Code)((Error*)a->error)->code; return NULL;
}

// ---------- always-ready events ----------
static PJRT_Event* event_new(void) { return (PJRT_Event*)calloc(1, 8); }
static PJRT_Error* Bridge_Event_Destroy(PJRT_Event_Destroy_Args* a) { free(a->event); return NULL; }
static PJRT_Error* Bridge_Event_IsReady(PJRT_Event_IsReady_Args* a) { a->is_ready = true; return NULL; }
static PJRT_Error* Bridge_Event_Error(PJRT_Event_Error_Args* a) { (void)a; return NULL; }
static PJRT_Error* Bridge_Event_Await(PJRT_Event_Await_Args* a) { (void)a; return NULL; }
static PJRT_Error* Bridge_Event_OnReady(PJRT_Event_OnReady_Args* a) {
  a->callback(NULL, a->user_arg); return NULL;
}

// ---------- python dispatch ----------
static PyObject* g_dispatch = NULL;
static int64_t g_shlo_cur[3] = {1, 0, 0};
static const int64_t g_shlo_min[3] = {0, 9, 0};

__attribute__((visibility("default")))
void JaxMlxInstallDispatcher(PyObject* fn) { Py_XINCREF(fn); g_dispatch = fn; }

__attribute__((visibility("default")))
void JaxMlxSetStablehloVersion(long a, long b, long c) {
  g_shlo_cur[0] = a; g_shlo_cur[1] = b; g_shlo_cur[2] = c;
}

// Steals `args` (a new-ref tuple or NULL). On success stores a new ref in *out.
static PJRT_Error* call_py(const char* method, PyObject* args, PyObject** out) {
  if (!g_dispatch) return err_new(13, "jax-mlx: dispatcher not installed");
  PyGILState_STATE st = PyGILState_Ensure();
  PJRT_Error* err = NULL;
  PyObject* res = PyObject_CallFunction(
      g_dispatch, "sO", method, args ? args : Py_None);
  if (!res) {
    PyObject *t, *v, *tb;
    PyErr_Fetch(&t, &v, &tb);
    PyObject* s = v ? PyObject_Str(v) : NULL;
    err = err_new(13, s ? PyUnicode_AsUTF8(s) : "python error");
    Py_XDECREF(s); Py_XDECREF(t); Py_XDECREF(v); Py_XDECREF(tb);
  } else {
    *out = res;
  }
  Py_XDECREF(args);
  PyGILState_Release(st);
  return err;
}

// ---------- singletons: client / device / description / memory ----------
static int g_client_o, g_device_o, g_desc_o, g_memory_o;
#define CLIENT ((PJRT_Client*)&g_client_o)
#define DEVICE ((PJRT_Device*)&g_device_o)
#define DESC   ((PJRT_DeviceDescription*)&g_desc_o)
#define MEMORY ((PJRT_Memory*)&g_memory_o)
static PJRT_Device* g_devices[1];
static PJRT_Memory* g_memories[1];

static PJRT_Error* Bridge_Plugin_Initialize(PJRT_Plugin_Initialize_Args* a) { (void)a; return NULL; }

static PJRT_NamedValue g_attrs[2];
static PJRT_Error* Bridge_Plugin_Attributes(PJRT_Plugin_Attributes_Args* a) {
  g_attrs[0] = (PJRT_NamedValue){ .struct_size = PJRT_NamedValue_STRUCT_SIZE,
      .name = "stablehlo_current_version", .name_size = 25,
      .type = PJRT_NamedValue_kInt64List,
      .int64_array_value = g_shlo_cur, .value_size = 3 };
  g_attrs[1] = (PJRT_NamedValue){ .struct_size = PJRT_NamedValue_STRUCT_SIZE,
      .name = "stablehlo_minimum_version", .name_size = 25,
      .type = PJRT_NamedValue_kInt64List,
      .int64_array_value = g_shlo_min, .value_size = 3 };
  a->attributes = g_attrs; a->num_attributes = 2;
  return NULL;
}

static PJRT_Error* Bridge_Client_Create(PJRT_Client_Create_Args* a) {
  g_devices[0] = DEVICE; g_memories[0] = MEMORY;
  a->client = CLIENT; return NULL;
}
static PJRT_Error* Bridge_Client_Destroy(PJRT_Client_Destroy_Args* a) { (void)a; return NULL; }
static PJRT_Error* Bridge_Client_PlatformName(PJRT_Client_PlatformName_Args* a) {
  a->platform_name = "mlx"; a->platform_name_size = 3; return NULL;
}
static PJRT_Error* Bridge_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* a) {
  a->process_index = 0; return NULL;
}
static PJRT_Error* Bridge_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* a) {
  a->platform_version = "jax-mlx 0.1.0"; a->platform_version_size = 13; return NULL;
}
static PJRT_Error* Bridge_Client_Devices(PJRT_Client_Devices_Args* a) {
  a->devices = g_devices; a->num_devices = 1; return NULL;
}
static PJRT_Error* Bridge_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args* a) {
  a->addressable_devices = g_devices; a->num_addressable_devices = 1; return NULL;
}
static PJRT_Error* Bridge_Client_AddressableMemories(PJRT_Client_AddressableMemories_Args* a) {
  a->addressable_memories = g_memories; a->num_addressable_memories = 1; return NULL;
}
static PJRT_Error* Bridge_Client_LookupDevice(PJRT_Client_LookupDevice_Args* a) {
  if (a->id != 0) return err_new(3, "jax-mlx: only device 0 exists");
  a->device = DEVICE; return NULL;
}
static PJRT_Error* Bridge_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* a) {
  if (a->local_hardware_id != 0) return err_new(3, "jax-mlx: only device 0");
  a->addressable_device = DEVICE; return NULL;
}
static PJRT_Error* Bridge_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args* a) {
  for (size_t i = 0; i < a->default_assignment_size; ++i) a->default_assignment[i] = 0;
  return NULL;
}

static PJRT_Error* Bridge_Device_GetDescription(PJRT_Device_GetDescription_Args* a) {
  a->device_description = DESC; return NULL;
}
static PJRT_Error* Bridge_Device_IsAddressable(PJRT_Device_IsAddressable_Args* a) {
  a->is_addressable = true; return NULL;
}
static PJRT_Error* Bridge_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* a) {
  a->local_hardware_id = 0; return NULL;
}
static PJRT_Error* Bridge_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* a) {
  a->memories = g_memories; a->num_memories = 1; return NULL;
}
static PJRT_Error* Bridge_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* a) {
  a->memory = MEMORY; return NULL;
}

static PJRT_Error* Bridge_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* a) {
  a->id = 0; return NULL;
}
static PJRT_Error* Bridge_DeviceDescription_ProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* a) { a->process_index = 0; return NULL; }
static PJRT_Error* Bridge_DeviceDescription_Attributes(
    PJRT_DeviceDescription_Attributes_Args* a) {
  a->attributes = NULL; a->num_attributes = 0; return NULL;
}
static PJRT_Error* Bridge_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* a) {
  a->device_kind = "mlx"; a->device_kind_size = 3; return NULL;
}
static PJRT_Error* Bridge_DeviceDescription_DebugString(
    PJRT_DeviceDescription_DebugString_Args* a) {
  a->debug_string = "MlxDevice(id=0)"; a->debug_string_size = 15; return NULL;
}
static PJRT_Error* Bridge_DeviceDescription_ToString(
    PJRT_DeviceDescription_ToString_Args* a) {
  a->to_string = "MlxDevice(id=0)"; a->to_string_size = 15; return NULL;
}

static PJRT_Error* Bridge_Memory_Id(PJRT_Memory_Id_Args* a) { a->id = 0; return NULL; }
static PJRT_Error* Bridge_Memory_Kind(PJRT_Memory_Kind_Args* a) {
  a->kind = "device"; a->kind_size = 6; return NULL;
}
static PJRT_Error* Bridge_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* a) {
  a->kind_id = 1; return NULL;
}
static PJRT_Error* Bridge_Memory_DebugString(PJRT_Memory_DebugString_Args* a) {
  a->debug_string = "mlx unified memory"; a->debug_string_size = 18; return NULL;
}
static PJRT_Error* Bridge_Memory_ToString(PJRT_Memory_ToString_Args* a) {
  a->to_string = "device"; a->to_string_size = 6; return NULL;
}
static PJRT_Error* Bridge_Memory_AddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* a) {
  a->devices = g_devices; a->num_devices = 1; return NULL;
}

// ---------- api table ----------
static PJRT_Error* GenericUnimplemented(void* a) {
  (void)a; return err_new(12, "jax-mlx: PJRT function not implemented");
}

static PJRT_Api g_api;

__attribute__((visibility("default")))
const PJRT_Api* GetPjrtApi(void) {
  memset(&g_api, 0, sizeof(g_api));
  g_api.struct_size = PJRT_Api_STRUCT_SIZE;
  g_api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
  g_api.pjrt_api_version.major_version = PJRT_API_MAJOR;
  g_api.pjrt_api_version.minor_version = PJRT_API_MINOR;
  // Fill every function slot with a safe UNIMPLEMENTED handler, then override.
  for (void** p = (void**)&g_api.PJRT_Error_Destroy;
       p < (void**)((char*)&g_api + sizeof(g_api)); ++p)
    *p = (void*)GenericUnimplemented;

#define SET(name) g_api.PJRT_##name = Bridge_##name
  SET(Error_Destroy); SET(Error_Message); SET(Error_GetCode);
  SET(Event_Destroy); SET(Event_IsReady); SET(Event_Error);
  SET(Event_Await); SET(Event_OnReady);
  SET(Plugin_Initialize); SET(Plugin_Attributes);
  SET(Client_Create); SET(Client_Destroy); SET(Client_PlatformName);
  SET(Client_ProcessIndex); SET(Client_PlatformVersion); SET(Client_Devices);
  SET(Client_AddressableDevices); SET(Client_AddressableMemories);
  SET(Client_LookupDevice); SET(Client_LookupAddressableDevice);
  SET(Client_DefaultDeviceAssignment);
  SET(Device_GetDescription); SET(Device_IsAddressable);
  SET(Device_LocalHardwareId); SET(Device_AddressableMemories);
  SET(Device_DefaultMemory);
  SET(DeviceDescription_Id); SET(DeviceDescription_ProcessIndex);
  SET(DeviceDescription_Attributes); SET(DeviceDescription_Kind);
  SET(DeviceDescription_DebugString); SET(DeviceDescription_ToString);
  SET(Memory_Id); SET(Memory_Kind); SET(Memory_Kind_Id);
  SET(Memory_DebugString); SET(Memory_ToString); SET(Memory_AddressableByDevices);
#undef SET
  return &g_api;
}
```

Note: field names above (`PJRT_Event_Error`, `major_version`, etc.) must be verified against the vendored header before compiling: `grep -n "PJRT_Event_Error\|major_version\|PJRT_API_MAJOR" pjrt_c_api.h`. Adjust to the header's exact names — the header is the source of truth.

- [ ] **Step 4: write the loader**

`src/jax_plugins/mlx_plugin/__init__.py`:
```python
"""JAX auto-discovery loader for the jax-mlx PJRT plugin."""
import ctypes
import pathlib
import sys

_dll = None  # keep alive


def initialize():
    global _dll
    if sys.platform != "darwin":
        return
    here = pathlib.Path(__file__).resolve().parent
    dylib = here / "pjrt_mlx_bridge.dylib"
    if not dylib.exists():
        import warnings
        warnings.warn(f"jax-mlx: bridge dylib missing at {dylib}")
        return

    from . import runtime
    _dll = ctypes.CDLL(str(dylib))
    _dll.JaxMlxInstallDispatcher.argtypes = [ctypes.py_object]
    _dll.JaxMlxInstallDispatcher.restype = None
    _dll.JaxMlxInstallDispatcher(runtime.dispatch)
    _dll.JaxMlxSetStablehloVersion.argtypes = [ctypes.c_long] * 3
    _dll.JaxMlxSetStablehloVersion(*runtime.stablehlo_version())

    import jax._src.xla_bridge as xb
    xb.register_plugin("mlx", priority=500, library_path=str(dylib))


def version():
    return "0.1.0"
```

- [ ] **Step 5: build and iterate until the integration test passes**

```bash
clang -O2 -Wall -shared -undefined dynamic_lookup \
  -I"$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" -I. \
  bridge/pjrt_mlx_bridge.c -o src/jax_plugins/mlx_plugin/pjrt_mlx_bridge.dylib
PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_integration.py -v
```

Expected: `test_devices` PASS. If jaxlib fails client creation, the failing PJRT call name is in the error message (our `GenericUnimplemented` text plus jaxlib's context) — implement that one function following the same singleton pattern and re-run. Budget for 2–4 such iterations (likely candidates: `PJRT_Client_TopologyDescription` — return UNIMPLEMENTED is tolerated; `PJRT_DeviceDescription_MemoryDescriptions` — return empty list).

- [ ] **Step 6: commit**

```bash
git add -A && git commit -m "feat: C bridge part A -- jax.devices('mlx') works"
```

---

### Task 7: C bridge part B — buffers

**Files:**
- Modify: `bridge/pjrt_mlx_bridge.c`
- Create: `tests/integration/check_buffers.py`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: `runtime.dispatch("buffer_from_host" | "buffer_to_host" | "buffer_delete")`, `call_py`, `event_new`, `err_new`
- Produces: C struct `Buf { int64_t id; int dtype; size_t ndim; int64_t dims[8]; size_t nbytes; }` and a `buf_new(...)` helper reused by Task 8's execute.

- [ ] **Step 1: failing integration test**

`tests/integration/check_buffers.py`:
```python
import jax
import numpy as np

d = jax.devices("mlx")[0]
a = np.arange(24, dtype=np.float32).reshape(4, 6)
buf = jax.device_put(a, d)
assert buf.dtype == np.float32 and buf.shape == (4, 6)
np.testing.assert_array_equal(np.asarray(buf), a)

b16 = jax.device_put(np.ones((3,), np.int16), d)
np.testing.assert_array_equal(np.asarray(b16), np.ones((3,), np.int16))
print("BUFFERS-OK")
```

Append to `tests/test_integration.py`:
```python
def test_buffers():
    assert "BUFFERS-OK" in run_check("check_buffers.py")
```

- [ ] **Step 2: run to verify failure**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_integration.py::test_buffers -v`
Expected: FAIL with our UNIMPLEMENTED message naming `BufferFromHostBuffer`

- [ ] **Step 3: implement buffer functions in the bridge**

Add to `bridge/pjrt_mlx_bridge.c` (before `GetPjrtApi`), and add the `SET(...)` lines listed at the end:

```c
// ---------- buffers ----------
static const size_t kDtypeBytes[] = {0, 1, 1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8, 2, 8, 16};

typedef struct {
  int64_t id; int dtype; size_t ndim; int64_t dims[8]; size_t nbytes;
} Buf;

static Buf* buf_new(int64_t id, int dtype, size_t ndim, const int64_t* dims) {
  Buf* b = calloc(1, sizeof(Buf));
  b->id = id; b->dtype = dtype; b->ndim = ndim;
  size_t n = 1;
  for (size_t i = 0; i < ndim; ++i) { b->dims[i] = dims[i]; n *= (size_t)dims[i]; }
  b->nbytes = n * kDtypeBytes[dtype];
  return b;
}

static PJRT_Error* Bridge_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* a) {
  if (a->num_byte_strides != 0) {
    // Only dense row-major accepted in v0; jax sends dense for np arrays.
    size_t expect = kDtypeBytes[a->type];
    for (size_t i = a->num_dims; i-- > 0;) {
      if ((size_t)a->byte_strides[i] != expect)
        return err_new(12, "jax-mlx: non-dense host strides unsupported");
      expect *= (size_t)a->dims[i];
    }
  }
  size_t n = kDtypeBytes[a->type];
  for (size_t i = 0; i < a->num_dims; ++i) n *= (size_t)a->dims[i];

  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* bytes = PyBytes_FromStringAndSize((const char*)a->data, (Py_ssize_t)n);
  PyObject* dims = PyTuple_New((Py_ssize_t)a->num_dims);
  for (size_t i = 0; i < a->num_dims; ++i)
    PyTuple_SET_ITEM(dims, (Py_ssize_t)i, PyLong_FromLongLong(a->dims[i]));
  PyObject* args = PyTuple_Pack(3, bytes, PyLong_FromLong(a->type), dims);
  Py_DECREF(bytes); Py_DECREF(dims);
  PyGILState_Release(st);

  PyObject* res = NULL;
  PJRT_Error* err = call_py("buffer_from_host", args, &res);
  if (err) return err;
  st = PyGILState_Ensure();
  int64_t id = PyLong_AsLongLong(res);
  Py_DECREF(res);
  PyGILState_Release(st);

  a->buffer = (PJRT_Buffer*)buf_new(id, a->type, a->num_dims, a->dims);
  a->done_with_host_buffer = event_new();
  return NULL;
}

static PJRT_Error* Bridge_Buffer_Destroy(PJRT_Buffer_Destroy_Args* a) {
  Buf* b = (Buf*)a->buffer;
  PyObject* res = NULL;
  PJRT_Error* err = call_py("buffer_delete",
      PyGILState_Check() ? PyTuple_Pack(1, PyLong_FromLongLong(b->id))
                         : NULL, &res);
  // NOTE: build the tuple under the GIL; see step text.
  Py_XDECREF(res);
  free(b);
  return err;
}

static PJRT_Error* Bridge_Buffer_ElementType(PJRT_Buffer_ElementType_Args* a) {
  a->type = (PJRT_Buffer_Type)((Buf*)a->buffer)->dtype; return NULL;
}
static PJRT_Error* Bridge_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* a) {
  Buf* b = (Buf*)a->buffer; a->dims = b->dims; a->num_dims = b->ndim; return NULL;
}
static PJRT_Error* Bridge_Buffer_UnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args* a) {
  Buf* b = (Buf*)a->buffer;
  a->unpadded_dims = b->dims; a->num_dims = b->ndim; return NULL;
}
static PJRT_Error* Bridge_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* a) {
  a->on_device_size_in_bytes = ((Buf*)a->buffer)->nbytes; return NULL;
}
static PJRT_Error* Bridge_Buffer_Device(PJRT_Buffer_Device_Args* a) {
  a->device = DEVICE; return NULL;
}
static PJRT_Error* Bridge_Buffer_Memory(PJRT_Buffer_Memory_Args* a) {
  a->memory = MEMORY; return NULL;
}
static PJRT_Error* Bridge_Buffer_Delete(PJRT_Buffer_Delete_Args* a) {
  (void)a; return NULL;  // actual free happens in Destroy
}
static PJRT_Error* Bridge_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* a) {
  a->is_deleted = false; return NULL;
}
static PJRT_Error* Bridge_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* a) {
  a->is_on_cpu = false; return NULL;
}
static PJRT_Error* Bridge_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* a) {
  a->event = event_new(); return NULL;
}

static PJRT_Error* Bridge_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* a) {
  Buf* b = (Buf*)a->src;
  if (a->dst == NULL) { a->dst_size = b->nbytes; return NULL; }
  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* args = PyTuple_Pack(1, PyLong_FromLongLong(b->id));
  PyGILState_Release(st);
  PyObject* res = NULL;
  PJRT_Error* err = call_py("buffer_to_host", args, &res);
  if (err) return err;
  st = PyGILState_Ensure();
  char* data; Py_ssize_t len;
  PyBytes_AsStringAndSize(res, &data, &len);
  if ((size_t)len > a->dst_size) {
    Py_DECREF(res); PyGILState_Release(st);
    return err_new(13, "jax-mlx: host buffer too small");
  }
  memcpy(a->dst, data, (size_t)len);
  Py_DECREF(res);
  PyGILState_Release(st);
  a->event = event_new();
  return NULL;
}
```

Fix the sloppy tuple-building in `Bridge_Buffer_Destroy` — the pattern is: `PyGILState_Ensure`, build the tuple, `PyGILState_Release`, then `call_py` (which re-acquires). Write it exactly like `Bridge_Buffer_ToHostBuffer` does.

Add to `GetPjrtApi`:
```c
  SET(Client_BufferFromHostBuffer);
  SET(Buffer_Destroy); SET(Buffer_ElementType); SET(Buffer_Dimensions);
  SET(Buffer_UnpaddedDimensions); SET(Buffer_OnDeviceSizeInBytes);
  SET(Buffer_Device); SET(Buffer_Memory); SET(Buffer_Delete);
  SET(Buffer_IsDeleted); SET(Buffer_IsOnCpu); SET(Buffer_ReadyEvent);
  SET(Buffer_ToHostBuffer);
```

- [ ] **Step 4: rebuild + run**

```bash
clang -O2 -Wall -shared -undefined dynamic_lookup \
  -I"$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" -I. \
  bridge/pjrt_mlx_bridge.c -o src/jax_plugins/mlx_plugin/pjrt_mlx_bridge.dylib
PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_integration.py -v
```
Expected: `test_devices` and `test_buffers` PASS. Likely iteration: jaxlib may call `PJRT_Buffer_GetMemoryLayout` — implement returning a dense major-to-minor `PJRT_Buffer_MemoryLayout` with `type = PJRT_Buffer_MemoryLayout_Type_Tiled`, `minor_to_major = {ndim-1, ..., 0}` stored in a static per-call buffer inside `Buf`.

- [ ] **Step 5: commit**

```bash
git add -A && git commit -m "feat: C bridge part B -- device_put roundtrip works"
```

---

### Task 8: C bridge part C — compile and execute

**Files:**
- Modify: `bridge/pjrt_mlx_bridge.c`
- Create: `tests/integration/check_execute.py`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: `runtime.dispatch("compile" | "execute")`, `Buf`/`buf_new`, `call_py`, `event_new`
- Produces: working `jax.jit` end-to-end on the `mlx` device.

- [ ] **Step 1: failing integration test**

`tests/integration/check_execute.py`:
```python
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platforms", "mlx,cpu")

x = jnp.arange(10)
assert "mlx" in str(list(x.devices())[0]).lower() or x.devices() == {jax.devices("mlx")[0]}
np.testing.assert_array_equal(np.asarray(x), np.arange(10))

a = jnp.ones((64, 64), jnp.float32)
np.testing.assert_allclose(float((a @ a).sum()), 64.0**3)

g = jax.grad(lambda w: jnp.tanh(w * w).sum())(jnp.linspace(0.1, 1.0, 8))
expected = jax.jit(jax.grad(lambda w: jnp.tanh(w * w).sum()),
                   backend="cpu")(jnp.linspace(0.1, 1.0, 8))
np.testing.assert_allclose(np.asarray(g), np.asarray(expected), rtol=1e-5)

k = jax.random.normal(jax.random.key(0), (4,))
kc = jax.jit(lambda: jax.random.normal(jax.random.key(0), (4,)), backend="cpu")()
np.testing.assert_allclose(np.asarray(k), np.asarray(kc), rtol=1e-5)
print("EXECUTE-OK")
```

Append to `tests/test_integration.py`:
```python
def test_execute():
    assert "EXECUTE-OK" in run_check("check_execute.py")
```

- [ ] **Step 2: run to verify failure**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_integration.py::test_execute -v`
Expected: FAIL naming `Client_Compile` as unimplemented

- [ ] **Step 3: implement compile/execute in the bridge**

Add to `bridge/pjrt_mlx_bridge.c`:

```c
// ---------- executables ----------
typedef struct {
  int64_t id;
  size_t num_outputs;
  int out_dtypes[64];
  size_t out_ndims[64];
  int64_t out_dims[64][8];
} Exec;

static PJRT_Error* Bridge_Client_Compile(PJRT_Client_Compile_Args* a) {
  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* code = PyBytes_FromStringAndSize(
      a->program->code, (Py_ssize_t)a->program->code_size);
  PyObject* args = PyTuple_Pack(1, code);
  Py_DECREF(code);
  PyGILState_Release(st);

  PyObject* res = NULL;
  PJRT_Error* err = call_py("compile", args, &res);
  if (err) return err;

  st = PyGILState_Ensure();
  Exec* e = calloc(1, sizeof(Exec));
  e->id = PyLong_AsLongLong(PyTuple_GetItem(res, 0));
  PyObject* specs = PyTuple_GetItem(res, 1);
  e->num_outputs = (size_t)PyList_Size(specs);
  for (size_t i = 0; i < e->num_outputs && i < 64; ++i) {
    PyObject* spec = PyList_GetItem(specs, (Py_ssize_t)i);
    e->out_dtypes[i] = (int)PyLong_AsLong(PyTuple_GetItem(spec, 0));
    PyObject* dims = PyTuple_GetItem(spec, 1);
    e->out_ndims[i] = (size_t)PyTuple_Size(dims);
    for (size_t j = 0; j < e->out_ndims[i]; ++j)
      e->out_dims[i][j] = PyLong_AsLongLong(PyTuple_GetItem(dims, (Py_ssize_t)j));
  }
  Py_DECREF(res);
  PyGILState_Release(st);
  a->executable = (PJRT_LoadedExecutable*)e;
  return NULL;
}

static PJRT_Error* Bridge_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* a) { free(a->executable); return NULL; }
static PJRT_Error* Bridge_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* a) {
  a->executable = (PJRT_Executable*)a->loaded_executable; return NULL;
}
static PJRT_Error* Bridge_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* a) {
  a->addressable_devices = g_devices; a->num_addressable_devices = 1; return NULL;
}
static PJRT_Error* Bridge_LoadedExecutable_Delete(
    PJRT_LoadedExecutable_Delete_Args* a) { (void)a; return NULL; }
static PJRT_Error* Bridge_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* a) { a->is_deleted = false; return NULL; }

static PJRT_Error* Bridge_Executable_Destroy(PJRT_Executable_Destroy_Args* a) {
  (void)a; return NULL;  // freed via LoadedExecutable_Destroy
}
static PJRT_Error* Bridge_Executable_Name(PJRT_Executable_Name_Args* a) {
  a->executable_name = "jax_mlx_exec"; a->executable_name_size = 12; return NULL;
}
static PJRT_Error* Bridge_Executable_NumReplicas(PJRT_Executable_NumReplicas_Args* a) {
  a->num_replicas = 1; return NULL;
}
static PJRT_Error* Bridge_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args* a) { a->num_partitions = 1; return NULL; }
static PJRT_Error* Bridge_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* a) {
  a->num_outputs = ((Exec*)a->executable)->num_outputs; return NULL;
}

static PJRT_Error* Bridge_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* a) {
  Exec* e = (Exec*)a->executable;
  if (a->num_devices != 1)
    return err_new(12, "jax-mlx: multi-device execution unsupported");

  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* ids = PyTuple_New((Py_ssize_t)a->num_args);
  for (size_t i = 0; i < a->num_args; ++i)
    PyTuple_SET_ITEM(ids, (Py_ssize_t)i,
        PyLong_FromLongLong(((Buf*)a->argument_lists[0][i])->id));
  PyObject* args = PyTuple_Pack(2, PyLong_FromLongLong(e->id), ids);
  Py_DECREF(ids);
  PyGILState_Release(st);

  PyObject* res = NULL;
  PJRT_Error* err = call_py("execute", args, &res);
  if (err) return err;

  st = PyGILState_Ensure();
  Py_ssize_t n = PyList_Size(res);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* t = PyList_GetItem(res, i);
    int64_t id = PyLong_AsLongLong(PyTuple_GetItem(t, 0));
    int dtype = (int)PyLong_AsLong(PyTuple_GetItem(t, 1));
    PyObject* dims = PyTuple_GetItem(t, 2);
    size_t ndim = (size_t)PyTuple_Size(dims);
    int64_t cdims[8];
    for (size_t j = 0; j < ndim; ++j)
      cdims[j] = PyLong_AsLongLong(PyTuple_GetItem(dims, (Py_ssize_t)j));
    a->output_lists[0][i] = (PJRT_Buffer*)buf_new(id, dtype, ndim, cdims);
  }
  Py_DECREF(res);
  PyGILState_Release(st);

  if (a->device_complete_events) a->device_complete_events[0] = event_new();
  return NULL;
}
```

Add to `GetPjrtApi`:
```c
  SET(Client_Compile);
  SET(LoadedExecutable_Destroy); SET(LoadedExecutable_GetExecutable);
  SET(LoadedExecutable_AddressableDevices); SET(LoadedExecutable_Delete);
  SET(LoadedExecutable_IsDeleted); SET(LoadedExecutable_Execute);
  SET(Executable_Destroy); SET(Executable_Name); SET(Executable_NumReplicas);
  SET(Executable_NumPartitions); SET(Executable_NumOutputs);
```

- [ ] **Step 4: rebuild + run, iterate**

Same build command as Task 7. Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_integration.py -v`
Expected: all 3 PASS. Known likely iterations, in order of probability:
1. jaxlib queries `PJRT_Executable_OutputElementTypes` / `PJRT_Executable_OutputDimensions` — implement from `Exec` meta (types array = `e->out_dtypes`; dims flattened with a `dim_sizes` array).
2. jaxlib queries `PJRT_Executable_OutputMemoryKinds` — leave UNIMPLEMENTED (converts to default kinds; our Memory kind is `"device"`, consistent).
3. jaxlib queries `PJRT_Executable_Fingerprint` — implement returning an empty string.
4. If `jnp.arange(10)` produces `int32` on CPU lowering but jax expects weak types — trust the golden comparison in the check script.

- [ ] **Step 5: run the full suite**

Run: `PYTHONPATH=src:. .venv/bin/python -m pytest tests/ -v`
Expected: everything PASS.

- [ ] **Step 6: commit**

```bash
git add -A && git commit -m "feat: C bridge part C -- jax.jit end-to-end on MLX device"
```

---

### Task 9: Packaging + fresh-venv smoke test + README

**Files:**
- Modify: `pyproject.toml` (no changes expected — verify), `setup.py` (verify build hook)
- Create: `README.md`

**Interfaces:**
- Consumes: everything
- Produces: `pip install .` works in a clean venv and `jax.devices("mlx")` runs.

- [ ] **Step 1: fresh-venv install test**

```bash
python3.12 -m venv /tmp/jaxmlx-e2e && /tmp/jaxmlx-e2e/bin/pip install -q --upgrade pip
/tmp/jaxmlx-e2e/bin/pip install -q ~/workspace/jax-mlx
/tmp/jaxmlx-e2e/bin/python -c "
import jax, jax.numpy as jnp, numpy as np
print(jax.devices('mlx'))
a = jnp.ones((32, 32))
assert float((a @ a).sum()) == 32.0**3
print('SMOKE-OK')"
```
Expected: `[MlxDevice(id=0)]` then `SMOKE-OK`. If the dylib is missing from the wheel, the `build_py` hook or `package-data` glob is wrong — fix and re-run.

- [ ] **Step 2: write README.md**

Content requirements (write actual prose, ~60 lines): what it is (open-source PJRT plugin for JAX on Apple Silicon backed by MLX), architecture diagram (the C-bridge + Python-core sketch from the plan header), install (`pip install git+...`), current op coverage (list the Task 3/4 ops), explicit non-goals for v0 (control flow, gather/scatter, sort, conv — link follow-up section), how to add an op (register a handler in `ops.py`, add a golden test using `harness.check`), license MIT, "not affiliated with Apple/Google", credit `pjrt_c_api.h` to openxla (Apache 2.0).

- [ ] **Step 3: commit**

```bash
git add -A && git commit -m "docs: README and verified clean-venv install"
```

---

### Task 10: Compatibility matrix + results

**Files:**
- Create: `tests/matrix.py` (adapted copy of `~/workspace/jax-metal-fork/tests/test_matrix.py`)
- Create: `results/matrix_results.md` (generated)

**Interfaces:**
- Consumes: the installed plugin (auto-discovery)
- Produces: a committed pass/fail table comparing every case against the CPU backend.

- [ ] **Step 1: adapt the matrix runner**

Copy `~/workspace/jax-metal-fork/tests/test_matrix.py` to `tests/matrix.py` with two changes: (a) in `run_child`, replace the METAL device lookup with `metal = jax.devices("mlx")[0]` (rename the variable `mlx_dev`); (b) `RESULTS` dir stays `results/` relative to repo root (same computation). Keep every test case — the FAIL rows for out-of-scope ops (cond, gather-heavy cases, sort, conv, fft, linalg) are the honest coverage statement for v0.

- [ ] **Step 2: run it**

Run: `PYTHONPATH=src:. .venv/bin/python -u tests/matrix.py`
Expected: PASS for everything Tasks 3–4 cover (dtypes except f64/complex, matmuls, reductions, transforms, random, autodiff, MLP step, fori/scan — scan lowers to `stablehlo.while`... if `scan`/`fori_loop`/`while_loop` rows FAIL with `unsupported op stablehlo.while`, that is the expected v0 result). Record whatever comes out; a FAIL row with a clean `NotImplementedError` is a correct v0 outcome, a CRASH row is a bug to fix now.

- [ ] **Step 3: sanity-check the numbers, commit, push**

If PASS count < 30, something structural is wrong — investigate before committing.

```bash
git add -A && git commit -m "test: compatibility matrix results for v0"
gh repo create jax-mlx --public --source=. --push \
  --description "JAX on Apple Silicon GPUs via an open-source PJRT plugin backed by MLX"
```

---

## Follow-up plans (explicitly out of v0 scope)

1. **Control flow**: `stablehlo.while`, `stablehlo.case` — host-driven interpretation (evaluate cond region, loop in Python, `mx.eval` per iteration). Unlocks `scan`/`while_loop`/`fori_loop`/`cond`, which unlocks most real training loops.
2. **Indexing**: `gather`/`scatter`/`dynamic_slice`/`dynamic_update_slice` general forms (`mx.take_along_axis`, `at[].set` equivalents).
3. **`sort`, `top_k` (composite), `reduce_window`, `convolution`** (`mx.conv_general`).
4. **Performance**: cache `mx.compile`d closures per executable, buffer donation, zero-copy host transfers via `mx.array(memoryview)`.
5. **Robustness**: run `jax`'s own test suite subsets; multi-output executables with tokens; `jax.debug.print` via host callbacks.

## Self-review notes

- Spec coverage: parsing (T2), translation (T3–4), runtime contract (T5), PJRT surface (T6–8), packaging (T9), honest coverage reporting (T10) — all present. Control flow explicitly deferred with rationale.
- Type consistency: `runtime.dispatch` signatures in T5 match every `call_py` marshalling site in T6–8 (`compile` → `(int, list[tuple[int, tuple]])`; `execute` → `list[(int, int, tuple)]`; `buffer_from_host(bytes, int, tuple) -> int`). `dtypes.PJRT.F32 == 11` used consistently in T1/T5 tests.
- Known-risk steps are marked with expected iteration loops (T6 step 5, T7 step 4, T8 step 4) rather than pretending the first compile will pass — the header's exact field names are the source of truth and the plan says to verify before building.
