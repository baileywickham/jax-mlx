"""StableHLO -> MLX. Parsing/introspection here; op semantics in ops.py."""
import dataclasses

from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from . import dtypes

_PORTABLE_MAGIC = b"ML\xefR"  # MLIR bytecode magic; artifacts are vhlo bytecode


@dataclasses.dataclass
class ParsedModule:
    """Wraps an ir.Module together with the ir.Context that keeps it alive.

    nanobind's ir.Module does not allow attaching arbitrary attributes
    (unlike the pybind11 build), so we can't stash the context directly on
    the module -- hold both here instead.
    """
    module: ir.Module
    ctx: ir.Context

    @property
    def body(self):
        return self.module.body


def parse_module(payload):
    if isinstance(payload, str):
        data = payload
    else:
        data = bytes(payload)
        if data.startswith(_PORTABLE_MAGIC):
            # Portable artifact (vhlo). jaxlib's own deserializer upgrades it
            # to current StableHLO -- version agreement is guaranteed because
            # the same jaxlib produced it.
            data = stablehlo.deserialize_portable_artifact_str(data)
    with make_ir_context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(data)
        return ParsedModule(module=module, ctx=ctx)


def main_func(module):
    for op in module.body.operations:
        # `op` here comes back as the dialect-specialized OpView (e.g.
        # func.FuncOp), whose `.name` property is shadowed to mean the
        # symbol name rather than the operation name. Go through
        # `.operation` to get the generic op, whose `.name` is the plain
        # "dialect.op" string (e.g. "func.func") that callers expect.
        genop = op.operation
        if genop.name == "func.func" and ir.StringAttr(genop.attributes["sym_name"]).value == "main":
            return genop
    raise ValueError("no @main in module")


def _tensor_spec(t):
    rt = ir.RankedTensorType(t)
    return (dtypes.ir_type_to_pjrt(str(rt.element_type)), tuple(rt.shape))


def result_specs(module):
    fn = main_func(module)
    ftype = ir.FunctionType(ir.TypeAttr(fn.attributes["function_type"]).value)
    return [_tensor_spec(t) for t in ftype.results]
