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


def _exec_func(fn, inputs, funcs, ops):
    """Interpret one func.func body; recurses into `funcs` for func.call."""
    body = fn.regions[0].blocks[0]
    env = {}
    for arg, val in zip(body.arguments, inputs):
        env[arg] = val
    for op in body.operations:
        genop = op.operation
        if genop.name == "func.return":
            return [env[v] for v in genop.operands]
        if genop.name == "func.call":
            # jnp.where (and other jax-level helpers) lower to a private
            # callee invoked via `func.call` rather than inlining -- not
            # covered by the brief's ops.py, which only handles stablehlo.*
            # ops. Resolve the callee by symbol name and recurse.
            callee = ir.FlatSymbolRefAttr(genop.attributes["callee"]).value
            results = _exec_func(funcs[callee], [env[v] for v in genop.operands], funcs, ops)
        else:
            handler = ops.HANDLERS.get(genop.name)
            if handler is None:
                raise NotImplementedError(f"jax-mlx: unsupported op {genop.name}")
            results = handler(genop, [env[v] for v in genop.operands])
        if not isinstance(results, list):
            results = [results]
        for res_value, res_array in zip(genop.results, results):
            env[res_value] = res_array
    raise ValueError("function body fell through without a func.return")


def compile_module(module):
    from . import ops
    fn = main_func(module)
    funcs = {}
    for op in module.body.operations:
        genop = op.operation
        if genop.name == "func.func":
            name = ir.StringAttr(genop.attributes["sym_name"]).value
            funcs[name] = genop

    def run(inputs):
        # `module` (the ParsedModule, which owns the ir.Context keeping the
        # whole IR tree alive) isn't otherwise referenced in this closure --
        # only `fn`/`funcs` are. Without this line nothing keeps a Python
        # reference to `module` once compile_module() returns, so its
        # context can be torn down before `run` executes, silently emptying
        # `body.operations` (or segfaulting, observed empirically).
        _keep_module_alive = module
        return _exec_func(fn, inputs, funcs, ops)

    return run
