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
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.asarray(a).tobytes()


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


def executable_delete(exec_id):
    with _lock:
        _executables.pop(exec_id, None)


def execute(exec_id, arg_ids):
    fn, out_specs = _executables[exec_id]
    outs = fn([_buffers[i] for i in arg_ids])
    if len(outs) != len(out_specs):
        raise RuntimeError(
            f"jax-mlx: executable returned {len(outs)} outputs, expected {len(out_specs)}")
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
             compile, execute, executable_delete, stablehlo_version]}


def dispatch(method, args):
    return _METHODS[method](*args)
