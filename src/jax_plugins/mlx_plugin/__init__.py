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
