"""
defines functions that allow time-evolution
"""

import numpy as _np
import copy
from eig_system import matrix_system, eig_pairs
import copy as _copy
import xarray as _xr
import xrft as _xrft




def coeff_project(_phi, _y, symmetri='even'):
	"""Takes a 1D-array and returns the Fourier coefficients that reproduce phi.

	Input:
		_phi: 1D np.array, complex. phi=phi(y)
		_y: np.array.
		symmetry: string. default is 'even'. If default then _phi has only real Fourier coefficients
			and the Fourier series is even.
	output:
		coeffs = 1D np.array. Complex.

	"""
	if np.max(_y) == _np.pi:
		nf = 2
	elif np.max(_y) == 2*_np.pi:
		nf = 1

	fac = _np.pi  # divides the Fourier coefficients (xrft scales them).

	L = len(_y)  # number of wavenumbers
	if (len(L) - 1) % 2 == 0:  # number should be odd.
		Nl = int((L - 1) / 2)

    da_phi_r = _xr.DataArray(_phi.real, dims=('y',), coords={'y': nf*_y})
    da_phi_i = _xr.DataArray(_phi.imag, dims=('y',), coords={'y': nf*_y})
    da_dft_phi_r = _xrft.fft(da_phi_r, true_phase=True, true_amplitude=True) # Fourier Transform w/ consideration of phase
    da_dft_phi_i = _xrft.fft(da_phi_i, true_phase=True, true_amplitude=True) # Fourier Transform w/ consideration of phase

    if default:
	    da_dft_phi_r = da_dft_phi_r.real.rename({'freq_y':'l'})
    	da_dft_phi_i = da_dft_phi_i.real.rename({'freq_y':'l'})

    else:
	    da_dft_phi_r = da_dft_phi_r.rename({'freq_y':'l'})
    	da_dft_phi_i = da_dft_phi_i.rename({'freq_y':'l'})

    gauss_alphs_a = list(da_dft_phi_r.isel(l=slice(nL,-1)).data/fac) + [da_dft_phi_r.data[-1]/fac]
    gauss_alphs_a[0] = gauss_alphs_a[0] / 2

    gauss_alphs_b = list(da_dft_phi_i.isel(l=slice(nL,-1)).data/fac) + [da_dft_phi_i.data[-1]/fac]
    gauss_alphs_b[0] = gauss_alphs_b[0] / 2
    
    rcoords = {'r':range(len(gauss_alphs_b))}
    gauss_alpsA = _xr.DataArray(gauss_alphs_a, coords=rcoords, dims='r')
    gauss_alpsB = _xr.DataArray(gauss_alphs_b, coords=rcoords, dims='r')
    gauss_coeffs = gauss_alpsA + (1j)*gauss_alpsB
    return gauss_coeffs



