import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platforms", "mlx,cpu")
# The shardy-partitioner disable now lives in the plugin loader itself
# (src/jax_plugins/mlx_plugin/__init__.py's initialize()), so it's applied
# automatically for every consumer of the plugin, including this script via
# the normal jax auto-discovery path -- no workaround needed here.

x = jnp.arange(10)
assert x.devices() == {jax.devices("mlx")[0]}
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
