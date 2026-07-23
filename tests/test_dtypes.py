import mlx.core as mx
import numpy as np
from jax_plugins.mlx_plugin import dtypes


def test_roundtrip_mlx():
    for dt in [mx.float32, mx.float16, mx.bfloat16, mx.int32, mx.int64,
               mx.int16, mx.int8, mx.uint8, mx.uint16, mx.uint32, mx.uint64,
               mx.bool_]:
        assert dtypes.pjrt_to_mlx(dtypes.mlx_to_pjrt(dt)) == dt


def test_np_f32():
    code = dtypes.np_to_pjrt(np.dtype("float32"))
    assert code == dtypes.PJRT.F32 == 11
    assert dtypes.pjrt_to_np(code) == np.dtype("float32")


def test_ir_strings():
    assert dtypes.ir_type_to_pjrt("f32") == dtypes.PJRT.F32
    assert dtypes.ir_type_to_pjrt("i1") == dtypes.PJRT.PRED
    assert dtypes.ir_type_to_pjrt("i32") == dtypes.PJRT.S32
    assert dtypes.ir_type_to_pjrt("ui8") == dtypes.PJRT.U8
    assert dtypes.ir_type_to_pjrt("bf16") == dtypes.PJRT.BF16


def test_unsupported_raises():
    import pytest
    with pytest.raises(NotImplementedError, match="f64"):
        dtypes.ir_type_to_pjrt("f64")


def test_unsupported_pjrt_mlx_includes_name():
    import pytest
    with pytest.raises(NotImplementedError, match="F64"):
        dtypes.pjrt_to_mlx(dtypes.PJRT.F64)


def test_unsupported_pjrt_np_includes_name():
    import pytest
    with pytest.raises(NotImplementedError, match="C64"):
        dtypes.pjrt_to_np(dtypes.PJRT.C64)


def test_unmodeled_code_raises_notimplementederror():
    import pytest
    with pytest.raises(NotImplementedError, match="16"):
        dtypes.pjrt_to_mlx(16)
