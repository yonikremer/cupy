
import cupy
import cupyx.scipy.fft as sp_fft
from cupyx.scipy.signal._resample import resample

def envelope(z, bp_in=(1, None), *,
             n_out=None, squared=False,
             residual='lowpass',
             axis=-1):
    """Compute the envelope of a real- or complex-valued signal.

    Parameters
    ----------
    z : cupy.ndarray
        Real- or complex-valued input signal, which is assumed to be made up of ``n``
        samples and having sampling interval ``T``. `z` may also be a multidimensional
        array with the time axis being defined by `axis`.
    bp_in : tuple[int | None, int | None], optional
        2-tuple defining the frequency band ``bp_in[0]:bp_in[1]`` of the input filter.
        The corner frequencies are specified as integer multiples of ``1/(n*T)`` with
        ``-n//2 <= bp_in[0] < bp_in[1] <= (n+1)//2`` being the allowed frequency range.
        ``None`` entries are replaced with ``-n//2`` or ``(n+1)//2`` respectively. The
        default of ``(1, None)`` removes the mean value as well as the negative
        frequency components.
    n_out : int | None, optional
        If not ``None`` the output will be resampled to `n_out` samples. The default
        of ``None`` sets the output to the same length as the input `z`.
    squared : bool, optional
        If set, the square of the envelope is returned. The bandwidth of the squared
        envelope is often smaller than the non-squared envelope bandwidth due to the
        nonlinear nature of the utilized absolute value function. I.e., the embedded
        square root function typically produces addiational harmonics.
        The default is ``False``.
    residual : Literal['lowpass', 'all', None], optional
        This option determines what kind of residual, i.e., the signal part which the
        input bandpass filter removes, is returned. ``'all'`` returns everything except
        the contents of the frequency band ``bp_in[0]:bp_in[1]``, ``'lowpass'``
        returns the contents of the frequency band ``< bp_in[0]``. If ``None`` then
        only the envelope is returned. Default: ``'lowpass'``.
    axis : int, optional
       Axis of `z` over which to compute the envelope. Default is last the axis.

    Returns
    -------
    cupy.ndarray
        If parameter `residual` is ``None`` then an array ``z_env`` with the same shape
        as the input `z` is returned, containing its envelope. Otherwise, an array with
        shape ``(2, *z.shape)``, containing the arrays ``z_env`` and ``z_res``, stacked
        along the first axis, is returned.
        It allows unpacking, i.e., ``z_env, z_res = envelope(z, residual='all')``.
        The residual ``z_res`` contains the signal part which the input bandpass filter
        removed, depending on the parameter `residual`. Note that for real-valued
        signals, a real-valued residual is returned. Hence, the negative frequency
        components of `bp_in` are ignored.
    """
    
    if not (-z.ndim <= axis < z.ndim):
        raise ValueError(f"Invalid parameter {axis=} for {z.shape=}!")
    if not (z.shape[axis] > 0):
        raise ValueError(f"z.shape[axis] not > 0 for {z.shape=}, {axis=}!")
    if len(bp_in) != 2 or not all((isinstance(b_, int) or b_ is None) for b_ in bp_in):
        raise ValueError(f"{bp_in=} isn't a 2-tuple of type (int | None, int | None)!")
    if not ((isinstance(n_out, int) and 0 < n_out) or n_out is None):
        raise ValueError(f"{n_out=} is not a positive integer or None!")
    if residual not in ('lowpass', 'all', None):
        raise ValueError(f"{residual=} not in ['lowpass', 'all', None]!")

    n = z.shape[axis]  # number of time samples of input
    n_out = n if n_out is None else n_out
    fak = n_out / n  # scaling factor for resampling

    bp = slice(bp_in[0] if bp_in[0] is not None else -(n//2),
               bp_in[1] if bp_in[1] is not None else (n+1)//2)
    if not (-n//2 <= bp.start < bp.stop <= (n+1)//2):
        raise ValueError("`-n//2 <= bp_in[0] < bp_in[1] <= (n+1)//2` does not hold " +
                         f"for n={z.shape[axis]=} and {bp_in=}!")

    # moving active axis to end allows to use `...` for indexing:
    z = cupy.moveaxis(z, axis, -1)

    if cupy.iscomplexobj(z):
        Z = sp_fft.fft(z)
    else:  # avoid calculating negative frequency bins for real signals:
        # For real inputs, we use rfft. 
        # Note: Scipy implementation does something slightly different for rfft initialization
        # but the logic is to get the full spectrum or half spectrum.
        # Scipy uses: dt = sp_fft.rfft(z[..., :1]).dtype; Z = xp.zeros_like(z, dtype=dt)
        # This implies Z should be same size as z but complex.
        
        # Let's follow logic: we need Z to hold FFT result.
        # If z is real, we want Z to be full complex FFT size to manipulate it easily?
        # Actually scipy code: 
        # Z = xp.zeros_like(z, dtype=dt) 
        # Z[..., :n//2 + 1] = sp_fft.rfft(z)
        # This implies Z has same shape as z (spatial domain) but holds freq domain data?
        # That seems odd if n is odd/even difference for rfft.
        # rfft output size is n//2 + 1. 
        # z size is n.
        # So Z having shape of z works for storage if n >= n//2 + 1, which is true for n>=1.
        
        # Determine complex dtype
        if z.dtype == cupy.float32:
            dt = cupy.complex64
        else:
            dt = cupy.complex128
            
        Z = cupy.zeros_like(z, dtype=dt)
        Z[..., :n//2 + 1] = sp_fft.rfft(z)
        
        if bp.start > 0:  # make signal analytic within bp_in band:
            Z[..., bp] *= 2
        elif bp.stop > 0:
            Z[..., 1:bp.stop] *= 2
            
    if not (bp.start <= 0 < bp.stop):  # envelope is invariant to freq. shifts.
        z_bb = sp_fft.ifft(Z[..., bp], n=n_out) * fak  # baseband signal
    else:
        bp_shift = slice(bp.start + n//2, bp.stop + n//2)
        # fftshift on last axis
        Z_shifted = sp_fft.fftshift(Z, axes=-1)
        z_bb = sp_fft.ifft(Z_shifted[..., bp_shift], n=n_out) * fak

    z_env = cupy.abs(z_bb) if not squared else cupy.real(z_bb) ** 2 + cupy.imag(z_bb) ** 2
    z_env = cupy.moveaxis(z_env, -1, axis)

    # Calculate the residual from the input bandpass filter:
    if residual is None:
        return z_env
        
    # Modify Z to represent the residual content
    if not (bp.start <= 0 < bp.stop):
        Z[..., bp] = 0
    else:
        Z[..., :bp.stop] = 0
        Z[..., bp.start:] = 0
        
    if residual == 'lowpass':
        if bp.stop > 0:
            Z[..., bp.stop:(n+1) // 2] = 0
        else:
            Z[..., bp.start:] = 0
            Z[..., 0:(n + 1) // 2] = 0

    if cupy.iscomplexobj(z):
        # resample in frequency domain
        if n_out > n:
            new_Z = cupy.zeros(z.shape[:-1] + (n_out,), dtype=Z.dtype)
            new_Z[..., :n//2] = Z[..., :n//2]
            new_Z[..., n_out - (n - n//2):] = Z[..., n//2:]
        elif n_out < n:
            new_Z = cupy.concatenate(
                (Z[..., :n_out//2], Z[..., n - (n_out - n_out//2):]), axis=-1)
        else:
            new_Z = Z
        z_res = sp_fft.ifft(new_Z) * (n_out / n)
    else:  # account for unpaired bin at m//2 before doing irfft():
        m = min(n, n_out)
        if n_out != n and m % 2 == 0:
            Z[..., m//2] *= 2 if n_out < n else 0.5
        z_res = fak * sp_fft.irfft(Z, n=n_out)

    return cupy.stack((z_env, cupy.moveaxis(z_res, -1, axis)), axis=0)
