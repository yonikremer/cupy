
import pytest
import numpy as np
import cupy
import cupyx.scipy.signal
from cupy import testing
import scipy.signal

@testing.with_requires('scipy>=1.16.3')
class TestEnvelope:

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_envelope_basic(self, xp, scp, dtype):
        x = testing.shaped_random((100,), xp, dtype=dtype)
        return scp.signal.envelope(x)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_envelope_axis(self, xp, scp, dtype):
        x = testing.shaped_random((10, 20), xp, dtype=dtype)
        return scp.signal.envelope(x, axis=0)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_envelope_squared(self, xp, scp, dtype):
        x = testing.shaped_random((50,), xp, dtype=dtype)
        return scp.signal.envelope(x, squared=True)

    # TODO: Add more tests covering n_out, bp_in, residual options once we are confident basic logic works.
    
    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_envelope_residual_none(self, xp, scp, dtype):
        x = testing.shaped_random((50,), xp, dtype=dtype)
        return scp.signal.envelope(x, residual=None)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-5, atol=1e-5)
    def test_envelope_n_out(self, xp, scp, dtype):
        x = testing.shaped_random((50,), xp, dtype=dtype)
        return scp.signal.envelope(x, n_out=30)
