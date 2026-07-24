import jax
import jax.numpy as jnp
import numpy as np

d = jax.devices("mlx")[0]
a = np.arange(24, dtype=np.float32).reshape(4, 6)
buf = jax.device_put(a, d)
assert buf.dtype == np.float32 and buf.shape == (4, 6)
np.testing.assert_array_equal(np.asarray(buf), a)

b16 = jax.device_put(np.ones((3,), np.int16), d)
np.testing.assert_array_equal(np.asarray(b16), np.ones((3,), np.int16))

# Rank > 8 must raise a clean exception, not silently corrupt the Buf (which
# only has room for 8 dims) and round-trip wrong data.
rank9 = np.zeros((1,) * 9, dtype=np.float32)
try:
    bad = jax.device_put(rank9, d)
    np.testing.assert_array_equal(np.asarray(bad), rank9)
    raise AssertionError("rank-9 device_put should have raised, not succeeded")
except AssertionError:
    raise
except Exception:
    pass  # jaxlib may wrap the PJRT error -- any clean exception is correct.

# f8 dtypes (PJRT_Buffer_Type enum values >= 16) must not overrun the
# 16-entry kDtypeBytes table -- must raise cleanly, not crash.
if hasattr(jnp, "float8_e4m3fn"):
    try:
        f8 = np.zeros(2, dtype=np.float32).astype(jnp.float8_e4m3fn)
        bad_f8 = jax.device_put(f8, d)
        np.testing.assert_array_equal(np.asarray(bad_f8), f8)
        raise AssertionError("f8 device_put should have raised, not succeeded")
    except AssertionError:
        raise
    except Exception:
        pass

print("BUFFERS-OK")
