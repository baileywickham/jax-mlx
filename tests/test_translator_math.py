import jax
import jax.numpy as jnp
import numpy as np
from tests.harness import check

x32 = jnp.linspace(0.5, 2.0, 12, dtype=jnp.float32)
m = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4)


def test_add_mul_sub_div(): check(lambda x: (x + x) * x - x / 2, x32)
def test_neg_abs(): check(lambda x: -abs(x - 1), x32)
def test_exp_log_tanh_sqrt(): check(lambda x: jnp.tanh(jnp.log(jnp.exp(x))) + jnp.sqrt(x), x32)
def test_rsqrt_pow(): check(lambda x: jax.lax.rsqrt(x) ** 2, x32)
def test_max_min(): check(lambda x: jnp.maximum(x, 1.0) + jnp.minimum(x, 1.0), x32)
def test_compare_select(): check(lambda x: jnp.where(x > 1.0, x, -x), x32)
def test_convert(): check(lambda x: x.astype(jnp.int32).astype(jnp.float32), x32)
def test_constant_splat(): check(lambda x: x + 3.0, x32)
def test_constant_dense(): check(lambda x: x + jnp.array([1.0, 2.0, 3.0, 4.0]), m)
def test_iota(): check(lambda: jnp.arange(10))
def test_reshape_transpose(): check(lambda a: a.reshape(4, 3).T, m)
def test_broadcast(): check(lambda a: a + jnp.ones((1, 4)), m)
def test_slice(): check(lambda x: x[2:9:2], x32)
def test_concat(): check(lambda x: jnp.concatenate([x, x]), x32)
def test_bitwise_shift():
    u = jnp.arange(8, dtype=jnp.uint32)
    check(lambda a: ((a ^ 21) | 3) & (a << 2) ^ (a >> 1), u)
def test_bitcast():
    u = jnp.arange(4, dtype=jnp.uint32)
    check(lambda a: jax.lax.bitcast_convert_type(a, jnp.int32), u)
def test_multiple_results(): check(lambda x: (x + 1, x * 2), x32)
def test_matmul(): check(lambda a: a @ a.T, m)
def test_batched_matmul():
    b = jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4)
    check(lambda a: jnp.einsum("bij,bkj->bik", a, a), b)
def test_reduce_sum(): check(lambda a: a.sum(), m)
def test_reduce_axis(): check(lambda a: a.sum(axis=1), m)
def test_reduce_max(): check(lambda a: a.max(axis=0), m)
def test_reduce_prod(): check(lambda a: (a + 1).prod(axis=1) / 1e4, m)
def test_reduce_any():
    check(lambda a: (a > 5.0).any(axis=0), m)
def test_grad(): check(jax.grad(lambda x: jnp.tanh(x).sum()), x32)
def test_mlp_grad():
    w = jnp.ones((4, 8), jnp.float32) * 0.1
    check(jax.grad(lambda w: jnp.tanh(m @ w).sum()), w)
def test_random_uniform():
    check(lambda: jax.random.uniform(jax.random.key(0), (8,)))
def test_random_normal():
    check(lambda: jax.random.normal(jax.random.key(1), (4, 4)))
