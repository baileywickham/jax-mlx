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
