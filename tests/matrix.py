"""jax-mlx compatibility matrix.

Requires jax-mlx to be installed (the "mlx" backend is picked up via JAX's
jax_plugins discovery). Parent mode spawns itself once per case (a crashing
case must not kill the run), collects PASS / FAIL / CRASH, and writes
results/matrix_results.md. Child mode (argv[1] = case name) runs the case on
mlx and on CPU and compares results.
"""
import os
import subprocess
import sys

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CASES = {}
def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco

# ---- dtypes ----
def _dtype_case(dtype):
    def fn(jnp, jax, np):
        x = jnp.arange(8).astype(dtype)
        return (x + x) * x
    return fn

for dt in ["float32", "float16", "bfloat16", "int32", "int16", "int8",
           "uint8", "uint32", "int64", "float64", "complex64", "bool"]:
    if dt == "bool":
        CASES["dtype_bool"] = lambda jnp, jax, np: jnp.array([True, False]) & jnp.array([True, True])
    else:
        CASES[f"dtype_{dt}"] = _dtype_case(dt)

# ---- basic ops ----
case("matmul_f32")(lambda jnp, jax, np: jnp.ones((64, 64)) @ jnp.ones((64, 64)))
case("matmul_f16")(lambda jnp, jax, np: (jnp.ones((32, 32), jnp.float16) @ jnp.ones((32, 32), jnp.float16)))
case("matmul_bf16")(lambda jnp, jax, np: (jnp.ones((32, 32), jnp.bfloat16) @ jnp.ones((32, 32), jnp.bfloat16)).astype(jnp.float32))
case("reduce_sum")(lambda jnp, jax, np: jnp.arange(100.0).sum())
case("reduce_max")(lambda jnp, jax, np: jnp.arange(100.0).max())
case("argmax")(lambda jnp, jax, np: jnp.argmax(jnp.array([1.0, 5.0, 3.0])))
case("transpose_reshape")(lambda jnp, jax, np: jnp.arange(24.0).reshape(2, 3, 4).transpose(2, 0, 1).reshape(-1))
case("slice_concat")(lambda jnp, jax, np: jnp.concatenate([jnp.arange(10.0)[2:5], jnp.arange(3.0)]))
case("pad")(lambda jnp, jax, np: jnp.pad(jnp.ones((3, 3)), 1))
case("gather_take")(lambda jnp, jax, np: jnp.take(jnp.arange(10.0) * 2, jnp.array([1, 3, 5])))
case("scatter_at_set")(lambda jnp, jax, np: jnp.zeros(10).at[jnp.array([2, 4])].set(1.0))
case("sort")(lambda jnp, jax, np: jnp.sort(jnp.array([3.0, 1.0, 2.0, -1.0])))
case("top_k")(lambda jnp, jax, np: jax.lax.top_k(jnp.array([1.0, 9.0, 3.0, 7.0]), 2)[0])
case("cumsum")(lambda jnp, jax, np: jnp.cumsum(jnp.arange(10.0)))
case("where_select")(lambda jnp, jax, np: jnp.where(jnp.arange(6) % 2 == 0, 1.0, -1.0))
case("exp_log_tanh")(lambda jnp, jax, np: jnp.tanh(jnp.log(jnp.exp(jnp.linspace(0.1, 2.0, 8)))))
case("erf")(lambda jnp, jax, np: jax.scipy.special.erf(jnp.linspace(-2, 2, 8)))
case("rsqrt_pow")(lambda jnp, jax, np: jax.lax.rsqrt(jnp.linspace(1.0, 4.0, 8)) ** 2)
case("einsum")(lambda jnp, jax, np: jnp.einsum("ij,jk->ik", jnp.ones((8, 8)), jnp.ones((8, 8))))
case("conv2d")(lambda jnp, jax, np: jax.lax.conv_general_dilated(
    jnp.ones((1, 1, 8, 8)), jnp.ones((1, 1, 3, 3)), (1, 1), "SAME"))

# ---- random ----
case("random_normal")(lambda jnp, jax, np: jax.random.normal(jax.random.key(0), (4, 4)).sum() * 0)
case("random_uniform")(lambda jnp, jax, np: jax.random.uniform(jax.random.key(1), (4,)).sum() * 0)
case("random_split")(lambda jnp, jax, np: jnp.stack([k.sum() * 0 for k in
    (lambda ks: [jax.random.normal(k, (2,)) for k in jax.random.split(ks, 2)])(jax.random.key(0))]))

# ---- control flow ----
case("scan")(lambda jnp, jax, np: jax.lax.scan(lambda c, x: (c + x, c), 0.0, jnp.arange(5.0))[0])
case("while_loop")(lambda jnp, jax, np: jax.lax.while_loop(lambda v: v < 10, lambda v: v + 3, 0))
case("cond")(lambda jnp, jax, np: jax.lax.cond(True, lambda: jnp.array(1.0), lambda: jnp.array(2.0)))
case("fori_loop")(lambda jnp, jax, np: jax.lax.fori_loop(0, 5, lambda i, v: v + i, 0))

# ---- autodiff / vmap ----
case("grad_simple")(lambda jnp, jax, np: jax.grad(lambda w: (w * w).sum())(jnp.arange(4.0)))
case("grad_mlp")(lambda jnp, jax, np: jax.grad(
    lambda w: jnp.tanh(jnp.ones((4, 8)) @ w).sum())(jnp.ones((8, 4)) * 0.1))
case("jvp")(lambda jnp, jax, np: jax.jvp(jnp.sin, (jnp.arange(4.0),), (jnp.ones(4),))[1])
case("vjp")(lambda jnp, jax, np: jax.vjp(jnp.sin, jnp.arange(4.0))[1](jnp.ones(4))[0])
case("vmap")(lambda jnp, jax, np: jax.vmap(lambda x: x @ x)(jnp.ones((3, 4, 4))))
case("jit_donate")(lambda jnp, jax, np: jax.jit(lambda x: x + 1)(jnp.arange(4.0)))

# ---- linalg / fft ----
case("cholesky")(lambda jnp, jax, np: jnp.linalg.cholesky(jnp.eye(4) * 4))
case("qr")(lambda jnp, jax, np: jnp.linalg.qr(jnp.eye(4))[0])
case("svd")(lambda jnp, jax, np: jnp.linalg.svd(jnp.eye(4))[1])
case("triangular_solve")(lambda jnp, jax, np: jax.scipy.linalg.solve_triangular(
    jnp.eye(4), jnp.ones((4, 1)), lower=True))
case("fft")(lambda jnp, jax, np: jnp.abs(jnp.fft.fft(jnp.arange(8.0))))

# ---- small training loop ----
@case("mlp_training_step")
def _mlp(jnp, jax, np):
    def loss(params, x, y):
        h = jnp.tanh(x @ params["w1"])
        p = h @ params["w2"]
        return ((p - y) ** 2).mean()
    params = {"w1": jnp.ones((4, 16)) * 0.1, "w2": jnp.ones((16, 1)) * 0.1}
    x, y = jnp.ones((8, 4)), jnp.zeros((8, 1))
    g = jax.grad(loss)(params, x, y)
    new = jax.tree.map(lambda p, gg: p - 0.1 * gg, params, g)
    return loss(new, x, y)


def run_child(name):
    # This repo's plugin auto-registers via jax_plugins discovery (no manual
    # xb.register_plugin needed). The matrix child runs as a bare subprocess
    # (no conftest.py), so replicate the platform ordering conftest.py sets
    # for the rest of the test suite before importing jax.
    os.environ.setdefault("JAX_PLATFORMS", "cpu,mlx")
    import jax
    import jax.numpy as jnp
    import numpy as np

    fn = CASES[name]
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        expected = np.asarray(fn(jnp, jax, np))
    mlx_dev = jax.devices("mlx")[0]
    with jax.default_device(mlx_dev):
        got = np.asarray(fn(jnp, jax, np))
    if expected.dtype != got.dtype:
        print(f"DTYPE-MISMATCH cpu={expected.dtype} mlx={got.dtype}")
        sys.exit(3)
    if not np.allclose(expected.astype(np.float64) if expected.dtype.kind in "fc" else expected,
                       got.astype(np.float64) if got.dtype.kind in "fc" else got,
                       rtol=2e-2, atol=2e-2):
        print(f"VALUE-MISMATCH cpu={expected.ravel()[:6]} mlx={got.ravel()[:6]}")
        sys.exit(4)
    print("PASS")
    sys.exit(0)


def run_parent():
    results = {}
    for name in CASES:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), name],
            capture_output=True, text=True, timeout=180)
        tail = [l for l in (p.stdout + p.stderr).splitlines()
                if l.strip() and not l.startswith(("I0000", "W0000", "WARNING",
                                                   "Metal device", "systemMemory",
                                                   "maxCacheSize", "Platform"))]
        if p.returncode == 0:
            results[name] = ("PASS", "")
        elif p.returncode in (3, 4):
            msg = next((l for l in tail if "MISMATCH" in l), "")
            results[name] = ("WRONG-RESULT", msg[:160])
        elif p.returncode < 0 or p.returncode == 134:  # signal / abort
            err = next((l for l in tail if "error" in l.lower() or "RAW:" in l), "")
            results[name] = ("CRASH", err[:160])
        else:
            err = next((l for l in reversed(tail)
                        if "Error" in l or "error" in l), tail[-1] if tail else "")
            results[name] = ("FAIL", err[:160])
        status, msg = results[name]
        print(f"{name:24s} {status:12s} {msg}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "matrix_results.md"), "w") as f:
        f.write("| test | status | detail |\n|---|---|---|\n")
        for name, (status, msg) in results.items():
            f.write(f"| `{name}` | {status} | {msg.replace('|', '/')} |\n")
    counts = {}
    for s, _ in results.values():
        counts[s] = counts.get(s, 0) + 1
    print("\nSUMMARY:", counts)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_child(sys.argv[1])
    else:
        run_parent()
