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
