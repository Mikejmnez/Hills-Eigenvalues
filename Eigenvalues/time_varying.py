"""
defines functions that allow time-evolution
"""

import numpy as _np
import copy
from Hills_eigenfunctions import complement_dot
import copy as _copy
import xarray as _xr
import xrft as _xrft




def coeff_project(_phi, _y, symmetry='even'):
	"""Takes a 1D-array and returns the Fourier coefficients that reproduce phi.

	Input:
		_phi: 1D np.array, complex. phi=phi(y)
		_y: np.array.
		symmetry: string. default is 'even'. If default then _phi has only real Fourier coefficients
			and the Fourier series is even.
	output:
		coeffs = 1D np.array. Complex.

	"""
	if _np.max(_y) == _np.pi:
		nf = 2
	elif _np.max(_y) == 2*_np.pi:
		nf = 1

	fac = _np.pi  # divides the Fourier coefficients (xrft scales them).

	L = len(_y)  # number of wavenumbers
	if (L - 1) % 2 == 0:  # number should be odd.
		nL = int((L - 1) / 2)

	da_phi_r = _xr.DataArray(_phi.real, dims=('y',), coords={'y': nf * _y})
	da_phi_i = _xr.DataArray(_phi.imag, dims=('y',), coords={'y': nf * _y})
	da_dft_phi_r = _xrft.fft(da_phi_r, true_phase=True, true_amplitude=True) # Fourier Transform w/ consideration of phase
	da_dft_phi_i = _xrft.fft(da_phi_i, true_phase=True, true_amplitude=True) # Fourier Transform w/ consideration of phase

	if symmetry == 'even':
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


def evolve_ds_modal(_dAs, _K, _alpha0, _Pe, _X, _Y, _time, _tf=0):
    """Constructs the modal solution to the IVP with uniform cross-jet initial condition """
    coords = {"t": copy.deepcopy(_time),
              "y": 2 * _Y[:, 0],
              "x": _X[0, :]}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["t", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    for i in range(len(_time)):
        exp_arg = (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
        PHI2n = _xr.dot(2*_dAs['A_2r'].isel(r=0), _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*(_time[i]-_tf)), dims='n')
        PHI2n = PHI2n.sel(k=_K).expand_dims({'x':_X[0, :]}).transpose('y', 'x')
        T0 = (PHI2n * _np.exp((2*_np.pi* _K * _X) * (1j))).real
        ds['Theta'].data[i, :, :] = T0.data
    return ds, PHI2n.isel(x=0).drop_vars({'x', 'r', 'k'})  # return the eigenfunction sum


def evolve_ds_modal_gaussian(_dAs, _K, _alpha0, _Pe, _gauss_alps, _facs, _X, _Y, _time, _tf=0):
    """Constructs the modal solution to the IVP that is localized across the jet."""

    coords = {"t": copy.deepcopy(_time),
              "y": 2 * _Y[:, 0],
              "x": _X[0, :]}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["t", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    Nr = len(_dAs.n) # length of truncated array
    ndAs = complement_dot(_facs * _gauss_alps, _dAs)  # has final size in n (sum in p)
    for i in range(len(_time)):
        exp_arg =  (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
        if Nr < len(_facs):  # Is this necessary?
            PHI2n = _xr.dot(ndAs.isel(n=slice(Nr)), _dAs['phi_2n'].isel(n=slice(Nr)) * _np.exp(-(0.25*_dAs['a_2n'].isel(n=slice(Nr)) + exp_arg)*(_time[i]-_tf)), dims='n')
            PHI2n = PHI2n + _xr.dot(ndAs[Nr:], _dAs['phi_2n'][Nr:] * _np.exp(-(0.25*_dAs['a_2n'][Nr:] + exp_arg)*(_time[i]-_tf)), dims='n')
        else:
            PHI2n = _xr.dot(ndAs, _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*(_time[i]-_tf)), dims='n')
        PHI2n = PHI2n.sel(k=_K).expand_dims({'x':_X[0, :]}).transpose('y', 'x')
        T0 = (PHI2n * _np.exp((2*_np.pi* _K * _X) * (1j))).real
        ds['Theta'].data[i, :, :] = T0.data
    return ds, PHI2n.isel(x=0).drop_vars({'x', 'k'})  # return the eigenfunction sum



## definition of time-varying shear flows (jets)

def time_reverse(_jet, _nt, _y, _t, _samp):
	"""Takes a jet and adds time periodicity, which reverses its orientation periodically.
	input:
		_jet: 1d np.numpy array. For now, a jet.
		_nt: frequency (how many time it reverses sign).
		_y: 1d np.numpy array.
		_t: 1d npumpy array. 
		_samp: int, sampling. Discretizes a single period of the time-varying function.
	"""
	if _np.max(_y) == _np.pi:
		nf = 2
	elif _np.max(_y) == 2*_np.pi:
		nf = 1

	sig = _nt * 2*_np.pi  # periodicity of time-variation.
	return sig


def re_sample(ft, nt=0):
	"""Samples a time-periodic function and returns a new function of same length with discrete values sampled from the original periodic signal.
	Input:
		ft: time-periodic function. 1d numpy array.
		nt: int. Defines how many values a periodic function can take between 0 and 1. nt=0 is default, and only values {-1, 0, 1} are considered. 
			nt=1 implies {-1, -0.5, 0, 0.5, 1} are consideredm, and so on.
	output:
		nft: time-periodic function. 1d numpy array. same lenght as original, but with only discrete values sampled from {0, 1/i, i in nt>=1}.
	"""
	KK = _np.arange(0, 1.000001, 1 / (2**nt))
	mids =[]
	for i in range(1, len(KK)):
		mids.append((KK[i] - KK[i-1]) / 2 + KK[i-1])

	nft = _np.ones(_np.shape(ft))  # new function

	for i in range(len(mids)):
		if i == 0:
			l = _np.where(abs(ft) < mids[i])[0]
			nft[l] = KK[i]
		else:
			l = _np.where(_np.logical_and(ft <= mids[i], ft >= mids[i-1]))[0]
			nft[l] = KK[i]

    # then last one
	l = _np.where(ft > mids[-1])[0]
	nft[l] = KK[-1]


    # now reverse sign
	for i in range(1, len(mids)):
		l = _np.where(_np.logical_and(ft <= -mids[i-1], ft >= -mids[i]))[0]
		nft[l] = -KK[i]
        
    # then last one
	l = _np.where(ft < -mids[-1])[0]
	nft[l] = -KK[-1]

	vals = list(-KK[::-1][:-1]) + list(KK)  ## all values

	ivals = loc_vals(nft, vals)

	return nft, vals, ivals


def loc_vals(_flip, _vals):
	"""returns an array of indices that reference the source its numerical value (i.e. to the value in _vals). The map from _vals to flip. 
	"""
	ind_vals = _np.nan*_np.ones(_np.shape(_flip))  # initial array of nans.
	for i in range(len(_vals)):  # iterate over all values
		l = _np.where(_flip == _vals[i])[0]
		ind_vals[l] = int(i)
	return ind_vals




