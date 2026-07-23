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

# shift_right_logical on signed ints must zero-fill: view as unsigned first.
def _srl(op, args):
    a, s = args
    if a.dtype in (mx.int8, mx.int16, mx.int32, mx.int64):
        u = {mx.int8: mx.uint8, mx.int16: mx.uint16,
             mx.int32: mx.uint32, mx.int64: mx.uint64}[a.dtype]
        return mx.right_shift(a.view(u), s.view(u)).view(a.dtype)
    return mx.right_shift(a, s)
HANDLERS["stablehlo.shift_right_logical"] = _srl


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
