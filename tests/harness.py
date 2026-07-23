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
