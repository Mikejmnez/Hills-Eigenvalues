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

	fac = _np.pi/2  # divides the Fourier coefficients (xrft scales them).

	L = len(_y)  # number of wavenumbers
	if (L - 1) % 2 == 0:  # number should be odd.
		nL = int((L - 1) / 2)

	da_phi_r = _xr.DataArray(_phi.real, dims=_phi.dims, coords=_phi.coords)
	da_phi_i = _xr.DataArray(_phi.imag, dims=_phi.dims, coords=_phi.coords)
	da_dft_phi_r = _xrft.fft(da_phi_r, dim='y', true_phase=True, true_amplitude=True) # Fourier Transform w/ consideration of phase
	da_dft_phi_i = _xrft.fft(da_phi_i, dim='y', true_phase=True, true_amplitude=True) # Fourier Transform w/ consideration of phase

	if symmetry == 'even':
		da_dft_phi_r = da_dft_phi_r.real.rename({'freq_y':'r'})
		da_dft_phi_i = da_dft_phi_i.real.rename({'freq_y':'r'})
	else:
		da_dft_phi_r = da_dft_phi_r.rename({'freq_y':'r'})
		da_dft_phi_i = da_dft_phi_i.rename({'freq_y':'r'})

	if len(da_dft_phi_r.dims) == 2:
		gauss_alphs_A = copy.deepcopy(da_dft_phi_r[:, nL:])
		gauss_alphs_A.data = gauss_alphs_A.data / fac
		gauss_alphs_A.data[:, 0] = gauss_alphs_A.data[:, 0] / 2

		gauss_alphs_B = copy.deepcopy(da_dft_phi_i[:, nL:])
		gauss_alphs_B.data = gauss_alphs_B.data / fac
		gauss_alphs_B.data[:, 0] = gauss_alphs_B.data[:, 0] / 2

		coords = {'r':range(len(gauss_alphs_A.r)), 'k':gauss_alphs_A.k.data}

	else:
		gauss_alphs_A = list(da_dft_phi_r.isel(r=slice(nL,-1)).data/fac) + [da_dft_phi_r.data[-1]/fac]
		gauss_alphs_A[0] = gauss_alphs_A[0] / 2
		coords = {'r':range(len(gauss_alphs_A))}


		gauss_alphs_B = list(da_dft_phi_i.isel(r=slice(nL,-1)).data/fac) + [da_dft_phi_i.data[-1]/fac]
		gauss_alphs_B[0] = gauss_alphs_B[0] / 2

	gauss_alphs_a = _xr.DataArray(gauss_alphs_A, coords=coords, dims=gauss_alphs_A.dims)
	gauss_alphs_b = _xr.DataArray(gauss_alphs_B, coords=coords, dims=gauss_alphs_B.dims)

	gauss_coeffs = gauss_alphs_a + (1j)*gauss_alphs_b
	# gauss_coeffs.r.data = _np.round(gauss_coeffs.r.data)  # round?

	return gauss_coeffs


def evolve_ds_modal(_dAs, _K, _alpha0, _Pe, _X, _Y, _time, _tf=0):
    """Constructs the modal solution to the IVP with uniform cross-jet initial condition """
    # something wrong is hapenning?
    coords = {"t": copy.deepcopy(_time),
              "y": 2 * _Y[:, 0],
              "x": _X[0, :]}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["t", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    for i in range(len(_time)):
        exp_arg = (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
        ndAs = 2*_dAs['A_2r'].isel(r=0)
        PHI2n = _xr.dot(ndAs, _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*(_time[i]-_tf)), dims='n')
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
        # if Nr < len(_facs):  # Is this necessary?
        #     PHI2n = _xr.dot(ndAs.isel(n=slice(Nr)), _dAs['phi_2n'].isel(n=slice(Nr)) * _np.exp(-(0.25*_dAs['a_2n'].isel(n=slice(Nr)) + exp_arg)*(_time[i]-_tf)), dims='n')
        #     PHI2n = PHI2n + _xr.dot(ndAs[Nr:], _dAs['phi_2n'][Nr:] * _np.exp(-(0.25*_dAs['a_2n'][Nr:] + exp_arg)*(_time[i]-_tf)), dims='n')
        # else:
        PHI2n = _xr.dot(ndAs, _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*(_time[i]-_tf)), dims='n')
        PHI2n = PHI2n.sel(k=_K).expand_dims({'x':_X[0, :]}).transpose('y', 'x')
        T0 = (PHI2n * _np.exp((2*_np.pi* _K * _X) * (1j))).real
        ds['Theta'].data[i, :, :] = T0.data
    return ds, PHI2n.isel(x=0).drop_vars({'x'})  # return the eigenfunction sum


def evolve_ds_gaussian_time_oscillate(_dAs, _da_xrft, _K, _alpha0, _Pe, _gauss_alps, _facs, _x, _y, _time, _tf=0):
    """Constructs the solution to the IVP"""
    ## Initialize the array.
    coords = {"t": _time, "y": 2 * _y, "x": _x}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["t", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    Nr = len(_dAs.n) # length of truncated array
    ndAs = complement_dot(_facs*_gauss_alps, _dAs)  # has final size in n (sum in p)
    for i in range(len(_time)):
        exp_arg =  (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
#         if Nr < len(facs):
#             PHI2n = xr.dot(ndAs.isel(n=slice(Nr)), dAs['phi_2n'].isel(n=slice(Nr)) * np.exp(-(0.25*dAs['a_2n'].isel(n=slice(Nr)) + exp_arg)*t[i]), dims='n')
#             PHI2n = PHI2n + xr.dot(ndAs[Nr:], dAs['phi_2n'][Nr:] * np.exp(-(0.25*dAs['a_2n'][Nr:] + exp_arg)*t[i]), dims='n')
#         else:
        PHI2n = _xr.dot(ndAs, _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*(_time[i] - _tf)), dims='n')
        T0 = _xrft.ifft(_da_xrft * PHI2n, dim='k', true_phase=True, true_amplitude=True).real # Signal in direct space
        nT0 = T0.rename({'freq_k':'x'}).transpose('y', 'x')
        ds['Theta'].data[i, :, :] = nT0.data
    return ds, PHI2n




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

	vals = list(-KK[::-1][:-1]) + list(KK)  ## all values - order with phase of nft

	ivals = loc_vals(nft, vals)

	return nft, vals, ivals


def loc_vals(_flip, _vals):
	"""returns an array of indices that reference the source its numerical value (i.e. to the value in _vals). The map from _vals to flip. 
	"""
	ind_vals = _np.nan*_np.ones(_np.shape(_flip))  # initial array of nans.

	for i in range(len(_vals)):  # iterate over all values
		l = _np.where(_flip == _vals[i])[0]
		ind_vals[l] = int(i)

	_ivals = [int(i) for i in ind_vals]  # turn into a list of integers
	return _ivals


def indt_intervals(_ivals):
	"""identifies the length of time that the tracer is advected by the same shear flow, which has
	functional dependence that is piecewise constant in time. 
	Input:
		_ivals: list contanining the mapping of indexes. When two contiguous indexes are repeated, 
			it implies shear flow is constant in time.
		N: int. Number of steady flows that approximate a time varying flow.
	output:
		indt: list. len(indt) = N. Each element is an pair of [start,end] time indexes.
		
	"""
	ll = _np.where(abs(_np.array(_ivals)[1:] - _np.array(_ivals)[:-1])==1)[0]
	Nf = len(ll) + 1

	t0 = 0
	indt = []
	for jj in range(Nf):
		for kk in range(t0, len(_ivals)-1):
			if _ivals[kk] != _ivals[kk+1]:  # only when two continuous indexes are different, stop. test this
				indt.append([t0, kk+1])
				break
		t0 = kk + 1
	t0 = indt[-1][-1]
	tf = len(_ivals)
	indt.append([t0, tf])
	return indt


def get_order(_nft, _indt, _vals):
	""" returns the maping from _vals (all possible values) into _nft, the time-oscillating
	piece-wise constant approximation to a continuous periodic fn.
	"""
	ordered_ind = []
	for i in range(len(_indt)):
		el = _nft[_indt[i][0]:_indt[i][1]][0] 
		ordered_ind.append(_np.where(_np.array(_vals) == el)[0][0])
	return ordered_ind










