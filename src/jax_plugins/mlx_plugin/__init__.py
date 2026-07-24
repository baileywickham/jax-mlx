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

    # Our StableHLO interpreter (translator.py/ops.py) doesn't handle the
    # `sdy` (shardy) dialect that jax's default partitioner now embeds even
    # for single-device programs -- deserializing the portable artifact
    # fails with "dialect 'sdy' is unknown" before we ever get to run
    # anything (jax.random's threefry lowering hits this). Disable it here,
    # in the auto-discovery loader itself, so every consumer of the plugin
    # gets a working jax.random without having to know this workaround.
    from jax._src import config as jax_config
    jax_config.config.update("jax_use_shardy_partitioner", False)

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
