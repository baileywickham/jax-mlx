import jax

devs = jax.devices("mlx")
assert len(devs) == 1, devs
d = devs[0]
assert d.platform == "mlx", d.platform
assert d.id == 0
assert d.default_memory().kind == "device"
print("DEVICES-OK")
