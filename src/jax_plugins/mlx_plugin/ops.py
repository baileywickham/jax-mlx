"""One handler per StableHLO op. Handlers get (ir.Operation, [mx.array])."""
import mlx.core as mx
import numpy as np
from jaxlib.mlir import ir

from . import dtypes

HANDLERS = {}


def register(name):
    def deco(fn):
        HANDLERS[name] = fn
        return fn
    return deco


def _elementwise(name, fn):
    HANDLERS[name] = lambda op, args: fn(*args)


_elementwise("stablehlo.add", lambda a, b: a + b)
_elementwise("stablehlo.subtract", lambda a, b: a - b)
_elementwise("stablehlo.multiply", lambda a, b: a * b)
_elementwise("stablehlo.divide", lambda a, b: a / b)
_elementwise("stablehlo.negate", lambda a: -a)
_elementwise("stablehlo.abs", mx.abs)
_elementwise("stablehlo.exponential", mx.exp)
_elementwise("stablehlo.log", mx.log)
_elementwise("stablehlo.log_plus_one", mx.log1p)
_elementwise("stablehlo.tanh", mx.tanh)
_elementwise("stablehlo.logistic", mx.sigmoid)
_elementwise("stablehlo.sqrt", mx.sqrt)
_elementwise("stablehlo.rsqrt", mx.rsqrt)
_elementwise("stablehlo.sign", mx.sign)
_elementwise("stablehlo.floor", mx.floor)
_elementwise("stablehlo.ceil", mx.ceil)
_elementwise("stablehlo.cosine", mx.cos)
_elementwise("stablehlo.sine", mx.sin)
_elementwise("stablehlo.power", mx.power)
_elementwise("stablehlo.maximum", mx.maximum)
_elementwise("stablehlo.minimum", mx.minimum)
_elementwise("stablehlo.and", lambda a, b: a & b)
_elementwise("stablehlo.or", lambda a, b: a | b)
_elementwise("stablehlo.xor", lambda a, b: a ^ b)
_elementwise("stablehlo.not", lambda a: ~a)
_elementwise("stablehlo.shift_left", mx.left_shift)
_elementwise("stablehlo.shift_right_logical", mx.right_shift)  # note below
_elementwise("stablehlo.shift_right_arithmetic", mx.right_shift)
_elementwise("stablehlo.select", lambda p, t, f: mx.where(p, t, f))
_elementwise("stablehlo.remainder", mx.remainder)
# Not stablehlo, but jax.random.normal's lowering emits it directly (CHLO
# dialect, used for ops without a StableHLO equivalent); needed empirically.
_elementwise("chlo.erf_inv", mx.erfinv)

# StableHLO defines shift amounts >= the operand's bit width to saturate
# (0 for shift_left/shift_right_logical, sign-extension for
# shift_right_arithmetic) -- but MLX's underlying shift instruction instead
# wraps the count modulo the bit width (e.g. `int32(1) >> 32 == 1`, not `0`).
# Surfaced empirically by jax.random's key-splitting, which computes
# `seed >> 32` on an i32 seed to extract the (always-zero) high key word.
_BITWIDTH = {mx.int8: 8, mx.uint8: 8, mx.int16: 16, mx.uint16: 16,
             mx.int32: 32, mx.uint32: 32, mx.int64: 64, mx.uint64: 64}


def _shift_clamp(a, s, in_range, fill):
    oob = s >= mx.array(_BITWIDTH[a.dtype], dtype=s.dtype)
    return mx.where(oob, fill, in_range)


def _shift_left(op, args):
    a, s = args
    return _shift_clamp(a, s, mx.left_shift(a, s), mx.zeros_like(a))
HANDLERS["stablehlo.shift_left"] = _shift_left


# shift_right_logical on signed ints must zero-fill: view as unsigned first.
def _srl(op, args):
    a, s = args
    if a.dtype in (mx.int8, mx.int16, mx.int32, mx.int64):
        u = {mx.int8: mx.uint8, mx.int16: mx.uint16,
             mx.int32: mx.uint32, mx.int64: mx.uint64}[a.dtype]
        r = mx.right_shift(a.view(u), s.view(u)).view(a.dtype)
    else:
        r = mx.right_shift(a, s)
    return _shift_clamp(a, s, r, mx.zeros_like(a))
HANDLERS["stablehlo.shift_right_logical"] = _srl


def _sra(op, args):
    a, s = args
    fill = mx.where(a < mx.array(0, dtype=a.dtype),
                     mx.array(-1, dtype=a.dtype), mx.array(0, dtype=a.dtype))
    return _shift_clamp(a, s, mx.right_shift(a, s), fill)
HANDLERS["stablehlo.shift_right_arithmetic"] = _sra


_CMP = {"EQ": lambda a, b: a == b, "NE": lambda a, b: a != b,
        "LT": lambda a, b: a < b, "LE": lambda a, b: a <= b,
        "GT": lambda a, b: a > b, "GE": lambda a, b: a >= b}

@register("stablehlo.compare")
def _compare(op, args):
    direction = str(op.attributes["comparison_direction"]).split("<")[1].split(" ")[1].rstrip(">")
    return _CMP[direction](*args)


def _result_type(op, i=0):
    return ir.RankedTensorType(op.results[i].type)


def _mlx_result_dtype(op, i=0):
    return dtypes.pjrt_to_mlx(
        dtypes.ir_type_to_pjrt(str(_result_type(op, i).element_type)))


@register("stablehlo.constant")
def _constant(op, args):
    attr = op.attributes["value"]
    dense = ir.DenseElementsAttr(attr)
    shape = tuple(_result_type(op).shape)
    np_dt = dtypes.pjrt_to_np(
        dtypes.ir_type_to_pjrt(str(_result_type(op).element_type)))
    if dense.is_splat:
        val = np.array(dense.get_splat_value(), dtype=np_dt)
        arr = np.broadcast_to(val, shape)
    else:
        arr = np.array(dense, copy=False).astype(np_dt, copy=False).reshape(shape)
    return mx.array(np.ascontiguousarray(arr))


@register("stablehlo.convert")
def _convert(op, args):
    return args[0].astype(_mlx_result_dtype(op))


@register("stablehlo.bitcast_convert")
def _bitcast(op, args):
    return args[0].view(_mlx_result_dtype(op))


@register("stablehlo.iota")
def _iota(op, args):
    rt = _result_type(op)
    dim = ir.IntegerAttr(op.attributes["iota_dimension"]).value
    n = rt.shape[dim]
    r = mx.arange(n, dtype=_mlx_result_dtype(op))
    view = [1] * len(rt.shape)
    view[dim] = n
    return mx.broadcast_to(r.reshape(view), tuple(rt.shape))


@register("stablehlo.reshape")
def _reshape(op, args):
    return args[0].reshape(tuple(_result_type(op).shape))


def _i64_array_attr(op, name):
    return [ir.IntegerAttr(a).value
            for a in ir.ArrayAttr(op.attributes[name])] \
        if isinstance(op.attributes[name], ir.ArrayAttr) \
        else list(ir.DenseI64ArrayAttr(op.attributes[name]))


@register("stablehlo.transpose")
def _transpose(op, args):
    return mx.transpose(args[0], _i64_array_attr(op, "permutation"))


@register("stablehlo.broadcast_in_dim")
def _broadcast_in_dim(op, args):
    out_shape = tuple(_result_type(op).shape)
    bdims = _i64_array_attr(op, "broadcast_dimensions")
    view = [1] * len(out_shape)
    for src, dst in enumerate(bdims):
        view[dst] = args[0].shape[src]
    return mx.broadcast_to(args[0].reshape(view), out_shape)


@register("stablehlo.slice")
def _slice(op, args):
    starts = _i64_array_attr(op, "start_indices")
    limits = _i64_array_attr(op, "limit_indices")
    strides = _i64_array_attr(op, "strides")
    idx = tuple(slice(s, l, st) for s, l, st in zip(starts, limits, strides))
    return args[0][idx]


@register("stablehlo.concatenate")
def _concatenate(op, args):
    dim = ir.IntegerAttr(op.attributes["dimension"]).value
    return mx.concatenate(args, axis=dim)


import string


@register("stablehlo.dot_general")
def _dot_general(op, args):
    lhs, rhs = args
    dn = ir.Attribute(op.attributes["dot_dimension_numbers"])
    # Robust across binding versions: parse the printed form
    # "#stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
    #   lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>"
    import re
    text = str(dn)
    def dims(key):
        mobj = re.search(key + r"\s*=\s*\[([0-9,\s]*)\]", text)
        return [int(x) for x in mobj.group(1).split(",")] if mobj and mobj.group(1).strip() else []
    lb, rb = dims("lhs_batching_dimensions"), dims("rhs_batching_dimensions")
    lc, rc = dims("lhs_contracting_dimensions"), dims("rhs_contracting_dimensions")

    letters = iter(string.ascii_letters)
    lhs_l = [None] * len(lhs.shape)
    rhs_l = [None] * len(rhs.shape)
    out_l = []
    for i, j in zip(lb, rb):
        c = next(letters); lhs_l[i] = rhs_l[j] = c; out_l.append(c)
    for i, j in zip(lc, rc):
        c = next(letters); lhs_l[i] = rhs_l[j] = c
    for i in range(len(lhs.shape)):
        if lhs_l[i] is None:
            lhs_l[i] = next(letters); out_l.append(lhs_l[i])
    for j in range(len(rhs.shape)):
        if rhs_l[j] is None:
            rhs_l[j] = next(letters); out_l.append(rhs_l[j])
    spec = f"{''.join(lhs_l)},{''.join(rhs_l)}->{''.join(out_l)}"
    return mx.einsum(spec, lhs, rhs).astype(_mlx_result_dtype(op))


_REDUCERS = {
    "stablehlo.add": mx.sum, "stablehlo.multiply": mx.prod,
    "stablehlo.maximum": mx.max, "stablehlo.minimum": mx.min,
    "stablehlo.or": mx.any, "stablehlo.and": mx.all,
}


@register("stablehlo.reduce")
def _reduce(op, args):
    n = len(op.results)
    operands, _inits = args[:n], args[n:]
    dims = _i64_array_attr(op, "dimensions")
    block = op.regions[0].blocks[0]
    body = [o for o in block.operations if o.name != "stablehlo.return"]
    if n != 1 or len(body) != 1 or body[0].name not in _REDUCERS:
        names = [o.name for o in body]
        raise NotImplementedError(f"jax-mlx: general reduce region {names}")
    return _REDUCERS[body[0].name](operands[0], axis=tuple(dims))


@register("stablehlo.custom_call")
def _custom_call(op, args):
    # Not in the brief; needed empirically by jax.random (jax.random.key /
    # jax.random.normal emit `stablehlo.custom_call @Sharding` to pin down
    # replication for sharding propagation -- a metadata annotation with no
    # runtime effect, XLA itself lowers it to an identity copy). Passing the
    # operand(s) through unchanged is correct for this target; anything else
    # is a real gap, so it still raises.
    target = ir.StringAttr(op.attributes["call_target_name"]).value
    if target == "Sharding":
        return list(args)
    raise NotImplementedError(f"jax-mlx: unsupported custom_call {target}")


@register("stablehlo.while")
def _while(op, args):
    # Not in the brief (which only anticipated dynamic_slice/pad); needed
    # empirically by jax.random's threefry2x32, whose 5-round mixing loop
    # lowers to stablehlo.while with a func.call'd body. Interpret via
    # translator.run_block so the cond/body regions get the same func.call
    # support as top-level function bodies.
    from . import translator
    cond_block = op.regions[0].blocks[0]
    body_block = op.regions[1].blocks[0]
    vals = list(args)
    while bool(translator.run_block(cond_block, vals)[0]):
        vals = translator.run_block(body_block, vals)
    return vals
