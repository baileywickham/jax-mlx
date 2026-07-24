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


def test_buffer_roundtrip_dtypes():
    import ml_dtypes
    for np_dt, code in [(np.float16, 10), (np.int64, 5), (np.uint8, 6),
                        (np.dtype(ml_dtypes.bfloat16), 13), (np.bool_, 1)]:
        a = np.arange(6).reshape(2, 3).astype(np_dt)
        bid = runtime.dispatch("buffer_from_host", (a.tobytes(), code, (2, 3)))
        out = runtime.dispatch("buffer_to_host", (bid,))
        np.testing.assert_array_equal(np.frombuffer(out, np_dt).reshape(2, 3), a)
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


def test_executable_delete():
    import pytest
    text = lower(lambda x: (x @ x.T).sum(), jnp.ones((3, 4), jnp.float32))
    exec_id, out_specs = runtime.dispatch("compile", (text.encode(),))
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    bid = runtime.dispatch("buffer_from_host", (a.tobytes(), 11, (3, 4)))
    runtime.dispatch("execute", (exec_id, (bid,)))

    runtime.dispatch("executable_delete", (exec_id,))
    assert exec_id not in runtime._executables

    with pytest.raises(KeyError):
        runtime.dispatch("execute", (exec_id, (bid,)))
