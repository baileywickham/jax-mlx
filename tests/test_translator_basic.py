import jax
import jax.numpy as jnp
from jax_plugins.mlx_plugin import translator
from tests.harness import lower


def test_parse_text_and_main():
    text = lower(lambda x: x + x, jnp.ones((3,), jnp.float32))
    module = translator.parse_module(text)
    fn = translator.main_func(module)
    assert fn.name == "func.func"


def test_result_specs():
    text = lower(lambda x: (x + x).astype(jnp.int32), jnp.ones((2, 3), jnp.float32))
    module = translator.parse_module(text)
    assert translator.result_specs(module) == [(4, (2, 3))]  # 4 == PJRT.S32


def test_parse_portable_artifact():
    from jaxlib.mlir.dialects import stablehlo
    text = lower(lambda x: x * 2, jnp.ones((2,), jnp.float32))
    artifact = stablehlo.serialize_portable_artifact_str(
        text, stablehlo.get_current_version())
    module = translator.parse_module(bytes(artifact))
    assert translator.main_func(module) is not None
