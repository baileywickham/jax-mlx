"""Session-wide pytest setup.

Not part of the Task 6 brief's file list, but required to keep the pre-existing
43 tests green once the mlx PJRT plugin is registered (Task 6): the loader
registers "mlx" via `xb.register_plugin("mlx", priority=500, ...)` per the
brief's frozen interface, and JAX picks the *highest-priority* initialized
platform as the default backend for eager ops / `jax.jit` execution. Priority
500 beats cpu's built-in priority (0), so every existing test that runs actual
computation through the default backend (e.g. module-level `jnp.linspace(...)`
in test_translator_math.py, or `jax.jit(f)(*args)` in tests/harness.py's
`check()`) would silently start executing against the still-unimplemented mlx
compile/execute path (Task 7/8) and fail with UNIMPLEMENTED.

Setting JAX_PLATFORMS explicitly reorders backend priority by list position
(first = highest) without disabling the mlx platform: cpu remains the default
for existing tests, while "mlx" stays discoverable for
tests/integration/check_devices.py's `jax.devices("mlx")` (that script inherits
this process's environment as its subprocess env base, so it must still list
"mlx" here). Only sets it if the user/CI hasn't already chosen a value.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu,mlx")
