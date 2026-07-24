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
