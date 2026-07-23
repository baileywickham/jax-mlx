import enum
import mlx.core as mx
import numpy as np
import ml_dtypes


class PJRT(enum.IntEnum):
    INVALID = 0; PRED = 1
    S8 = 2; S16 = 3; S32 = 4; S64 = 5
    U8 = 6; U16 = 7; U32 = 8; U64 = 9
    F16 = 10; F32 = 11; F64 = 12; BF16 = 13
    C64 = 14; C128 = 15


_MLX = {
    PJRT.PRED: mx.bool_, PJRT.S8: mx.int8, PJRT.S16: mx.int16,
    PJRT.S32: mx.int32, PJRT.S64: mx.int64, PJRT.U8: mx.uint8,
    PJRT.U16: mx.uint16, PJRT.U32: mx.uint32, PJRT.U64: mx.uint64,
    PJRT.F16: mx.float16, PJRT.F32: mx.float32, PJRT.BF16: mx.bfloat16,
}
_MLX_INV = {v: k for k, v in _MLX.items()}

_NP = {
    PJRT.PRED: np.dtype("bool"), PJRT.S8: np.dtype("int8"),
    PJRT.S16: np.dtype("int16"), PJRT.S32: np.dtype("int32"),
    PJRT.S64: np.dtype("int64"), PJRT.U8: np.dtype("uint8"),
    PJRT.U16: np.dtype("uint16"), PJRT.U32: np.dtype("uint32"),
    PJRT.U64: np.dtype("uint64"), PJRT.F16: np.dtype("float16"),
    PJRT.F32: np.dtype("float32"), PJRT.BF16: np.dtype(ml_dtypes.bfloat16),
}
_NP_INV = {v: k for k, v in _NP.items()}

_IR = {
    "i1": PJRT.PRED, "i8": PJRT.S8, "i16": PJRT.S16, "i32": PJRT.S32,
    "i64": PJRT.S64, "ui8": PJRT.U8, "ui16": PJRT.U16, "ui32": PJRT.U32,
    "ui64": PJRT.U64, "f16": PJRT.F16, "f32": PJRT.F32, "bf16": PJRT.BF16,
}


def _lookup(table, key, kind):
    try:
        return table[key]
    except KeyError:
        name = key.name if isinstance(key, PJRT) else key
        raise NotImplementedError(f"jax-mlx: unsupported {kind} dtype: {name}")


def _to_pjrt(code):
    try:
        return PJRT(code)
    except ValueError:
        raise NotImplementedError(f"jax-mlx: unsupported pjrt dtype: {code}")


def pjrt_to_mlx(code): return _lookup(_MLX, _to_pjrt(code), "pjrt")
def mlx_to_pjrt(dt): return int(_lookup(_MLX_INV, dt, "mlx"))
def pjrt_to_np(code): return _lookup(_NP, _to_pjrt(code), "pjrt")
def np_to_pjrt(dt): return int(_lookup(_NP_INV, np.dtype(dt), "numpy"))
def ir_type_to_pjrt(s): return int(_lookup(_IR, s, "ir"))
