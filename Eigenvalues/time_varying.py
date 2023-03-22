"""
defines functions that allow time-evolution
"""

import numpy as _np
from Hills_eigenfunctions import complement_dot
import copy as _copy
import xarray as _xr
import xrft as _xrft



def coeff_project(_phi, _y, dim='y', phi_old=_np.pi, phi_new=0):
	"""Takes a numpy-array and returns the even and odd Fourier coefficients that together,
		recreate the original function as a Fourier series.

	Input:
		_phi: 1D np.array, complex. phi=phi(y)
		_y: np.array. 
		phi_old = default pi. Refers to the original location (centering) of the flow. 
		phi_new: default 0. fase shift of the velocity field with respect to original location 
				(any positive values imply a negative shift for tracer, which is how we actually
				represent the shift in velocity).

	output:
		even_coeffs = Even coefficients.
		odd_coeffs = Odd coefficients
		phi_new
		phi_old
	"""

	frac = round((_y[-1] - _y[0]) / (2 * _np.pi), 1)  # unity if y in [0, 2\pi].

	phi_old = phi_old + phi_new
	if dim == 'y':
		if phi_old > 2*_np.pi:
			phi_old = phi_old - 2*_np.pi
		elif phi_old < 0:
			phi_old = phi_old + 2*_np.pi
	elif dim=='x':
		if phi_old > _np.pi:
			phi_old = phi_old - 2*_np.pi
		elif phi_old < -_np.pi:
			phi_old = phi_old + 2*_np.pi


	fac = _np.pi * frac   # divides the Fourier coefficients (xrft scales them).

	L = len(_y)  # number of wavenumbers
	if (L - 1) % 2 == 0:  # number should be odd.
		nL = int((L - 1) / 2)

	da_phi = _xr.DataArray(_phi, dims=_phi.dims, coords=_phi.coords)
	da_dft_phi = _xrft.fft(da_phi, dim=dim)
	da_dft_phi = da_dft_phi.rename({'freq_'+dim:'r'})

	_dims = da_dft_phi.dims
    
	if len(da_dft_phi.dims) == 1:  # no k dependence

		e_coeffs = 0.5 * (da_dft_phi.isel(r=slice(nL, L)).data + da_dft_phi.isel(r=slice(0, nL+1)).data[::-1]) / fac
		e_coeffs[0] = e_coeffs[0] / 2
		o_coeffs = 0.5 * (da_dft_phi.isel(r=slice(nL, L)).data - da_dft_phi.isel(r=slice(0, nL+1)).data[::-1]) / (-1j * fac)

		phi_cos = _np.ones(_np.shape(e_coeffs))
		phi_cos = _np.array([phi_cos[l] * _np.cos(-l*phi_new) for l in range(len(e_coeffs))])


		e_coords = {'r':range(len(e_coeffs))}
		o_coords = {'r':range(len(o_coeffs) - 1)}

		phi_sin = _np.ones(_np.shape(o_coeffs))
		phi_sin = _np.array([phi_sin[l] * _np.sin(-l*phi_new) for l in range(len(o_coeffs))])

		da_odd = o_coeffs * phi_cos + e_coeffs * phi_sin

		da_even = e_coeffs * phi_cos - o_coeffs * phi_sin

		odd_coeffs = _xr.DataArray(da_odd[1:], coords=o_coords, dims=_dims)

		even_coeffs = _xr.DataArray(da_even, coords=e_coords, dims=_dims)

	elif len(da_dft_phi.dims) == 2:

		e_coeffs = 0.5 * (da_dft_phi.isel(r=slice(nL, L)).data + da_dft_phi.isel(r=slice(0, nL+1)).data[:, ::-1]) / fac
		e_coeffs[:, 0] = e_coeffs[:, 0] / 2

		phi_cos = _np.ones(_np.shape(e_coeffs))
		phi_cos = _np.array([phi_cos[:, l] * _np.cos(-l*phi_new) for l in range(len(e_coeffs[0, :]))])

		o_coeffs = 0.5 * (da_dft_phi.isel(r=slice(nL, L)).data - da_dft_phi.isel(r=slice(0, nL+1)).data[:, ::-1]) / (-1j * fac)

		e_coords = {_dims[0]: da_phi[_dims[0]].data, 'r':range(len(e_coeffs[0, :]))}
		o_coords = {_dims[0]: da_phi[_dims[0]].data, 'r':range(len(o_coeffs[0, :]) - 1)}

		phi_sin = _np.ones(_np.shape(o_coeffs))
		phi_sin = _np.array([phi_sin[:, l] * _np.sin(-l*phi_new) for l in range(len(o_coeffs[0, :]))])

		da_odd = o_coeffs * _np.transpose(phi_cos) + e_coeffs * _np.transpose(phi_sin)
		da_even = e_coeffs * _np.transpose(phi_cos) - o_coeffs * _np.transpose(phi_sin)

		odd_coeffs = _xr.DataArray(da_odd[:, 1:], coords=o_coords, dims=_dims)
		even_coeffs = _xr.DataArray(da_even, coords=e_coords, dims=_dims)

	return even_coeffs, odd_coeffs, phi_new, phi_old


def _new_gamma(_gammas, _L, _M):
    """Defines the Fourier coefficients of a rotated shear flow based on the aspect ratio of domain. It 
    does so by keeping the scale of the original shear flow intact, albeit on a different (scaled) domain.
    Parameter:
        _gammas: 1d-array like.
            original Fourier coefficients that defined (via Fourier series) a shear flow.
        _L: Int.
            Lenght of (new) rotated coordinate.
        _M: Int.
            Length of (old) coordinate.
    Output:
        _gamma_new: 1d array-like.
            New Fourier coefficients that defined a rotated (by 90 degres) shear flow. Its dimensional scale
            remains invariant. In the trivial case M=L, gammas_new = gammas
    """
    ll = int(_L / _M) # must be an integer!
    
    _gammas_new = [0*m for m in range(ll*len(_gammas))]
    
    for i in range(len(_gammas)):
        _gammas_new[int((i+1)*ll) -1] = _gammas[i]
    return _gammas_new


def evolve_ds_modal_uniform(_dAs, _K, _alpha0, _Pe, _X, _Y, _t, _tf=0):
    """Constructs the modal solution to the IVP with uniform cross-jet initial condition """
    # something wrong is hapenning?
    coords = {"time": _copy.deepcopy(_t),
              "y": _Y[:, 0],
              "x": _X[0, :]}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["time", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    for i in range(len(_time)):
        exp_arg = (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
        ndAs = 2*_dAs['A_2r'].isel(r=0)
        PHI2n = _xr.dot(ndAs, _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*(_t[i]-_tf)), dims='n')
        PHI2n = PHI2n.sel(k=_K).expand_dims({'x':_X[0, :]}).transpose('y', 'x')
        T0 = (PHI2n * _np.exp((2*_np.pi* _K * _X) * (1j))).real
        ds['Theta'].data[i, :, :] = T0.data
    return ds, PHI2n.isel(x=0).drop_vars({'x', 'r', 'k'})  # return the eigenfunction sum


def evolve_ds_modal(_dAs, _K, _alpha0, _Pe, _gauss_alps, _facs, _X, _Y, _t, _tf=0):
	"""Constructs the modal solution to the IVP that is localized across the jet."""

	coords = {"time": _t, "y": _Y[:, 0], "x": _X[0, :]}
	Temp = _xr.DataArray(_np.nan, coords=coords, dims=["time", 'y', 'x'])
	ds = _xr.Dataset({'Theta': Temp})
	Nr = len(_dAs.n) # length of truncated array
	ndAs = complement_dot(_facs * _gauss_alps, _dAs)  # has final size in n (sum in p)
	for i in range(len(_t)):
		exp_arg =  (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		PHI2n = _xr.dot(ndAs, _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*(_t[i]-_tf)), dims='n')
		PHI2n = PHI2n.sel(k=_K).expand_dims({'x':_X[0, :]}).transpose('y', 'x')
		T0 = (PHI2n * _np.exp((2*_np.pi* _K * _X) * (1j))).real
		ds['Theta'].data[i, :, :] = T0.data
	return ds, PHI2n.isel(x=0).drop_vars({'x'})  # return the eigenfunction sum


def evolve_ds_modal_off(_dAs, _dBs, _K, _alpha0, _Pe, _a_alps, _afacs, _b_alps, _bfacs, _X, _Y, _t, _tf=0):
	"""Constructs the modal solution to the IVP that is localized across the jet,
	with arbitrary location in y"""

	coords = {"time": _copy.deepcopy(_t), "y": _Y[:, 0], "x": _X[0, :]}
	Temp = _xr.DataArray(_np.nan, coords=coords, dims=["time", 'y', 'x'])
	ds = _xr.Dataset({'Theta': Temp})
	_ndAs = _xr.dot(_afacs * _a_alps, _dAs['A_2r'], dims='r')
	_ndBs = _xr.dot(_bfacs * _b_alps, _dBs['B_2r'], dims='r')
	for i in range(len(_t)):    	
		arg_e = 0.25*_dAs['a_2n'] + (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		arg_o = 0.25*_dBs['b_2n'] + (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		_PHI2n_e = _xr.dot(_ndAs, _dAs['phi_2n'] * _np.exp(- arg_e*(_t[i] - _tf)), dims='n')
		_PHI2n_o = _xr.dot(_ndBs, _dBs['phi_2n'] * _np.exp(- arg_o*(_t[i] - _tf)), dims='n')
		_PHI2n = _PHI2n_e + _PHI2n_o
		_PHI2n = _PHI2n.sel(k=_K).expand_dims({'x':_X[0, :]}).transpose('y', 'x')
		T0 = (_PHI2n * _np.exp((2*_np.pi* _K * _X) * (1j))).real
		ds['Theta'].data[i, :, :] = T0.data
	return ds, PHI2n.isel(x=0).drop_vars({'x'})  # return the eigenfunction sum



def evolve_ds(_dAs, _da_xrft, _Kn, _alpha0, _Pe, _a_alps, _afacs, _x, _y, _t, _tf=0):
    """Constructs the solution to the IVP"""
    ## Initialize the array.
    coords = {"time": _t, "y": _y, "x": _x}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["time", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    Nr = len(_dAs.n) # length of truncated array
    ndAs = complement_dot(_afacs*_a_alps, _dAs)  # has final size in n (sum in p)
    for i in range(len(_t)):
        exp_arg =  (1j)*_alpha0*(2*_np.pi*_Kn)*_Pe + (2*_np.pi*_Kn)**2
        # if Nr < len(_facs):  # Is this necessary?
        #     PHI2n = _xr.dot(ndAs.isel(n=slice(Nr)), _dAs['phi_2n'].isel(n=slice(Nr)) * _np.exp(-(0.25*_dAs['a_2n'].isel(n=slice(Nr)) + exp_arg)*(_time[i]-_tf)), dims='n')
        #     PHI2n = PHI2n + _xr.dot(ndAs[Nr:], _dAs['phi_2n'][Nr:] * _np.exp(-(0.25*_dAs['a_2n'][Nr:] + exp_arg)*(_time[i]-_tf)), dims='n')
        # else:
        PHI2n = _xr.dot(ndAs, _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*(_t[i] - _tf)), dims='n')
        T0 = _xrft.ifft(_da_xrft * PHI2n, dim='k', true_phase=True, true_amplitude=True).real # Signal in direct space
        nT0 = T0.rename({'freq_k':'x'}).transpose('y', 'x')
        ds['Theta'].data[i, :, :] = nT0.data
    return ds, PHI2n



def evolve_ds_off(_dAs, _dBs, _da_xrft, _Kn, _alpha0, _Pe, _a_alps, _afacs, _b_alps, _bfacs, _x, _y, _t, _tf=0):
    """Constructs the solution to the IVP"""
    ## Initialize the array
    coords = {"time": _t, "y": _y, "x": _x}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["time", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    _ndAs = _xr.dot(_afacs * _a_alps, _dAs['A_2r'], dims='r')
    _ndBs = _xr.dot(_bfacs * _b_alps, _dBs['B_2r'], dims='r')
    for i in range(len(_t)):
    	arg_e = 0.25*_dAs['a_2n'] + (1j)*_alpha0*(2*_np.pi*_Kn)*_Pe + (2*_np.pi*_Kn)**2
    	arg_o = 0.25*_dBs['b_2n'] + (1j)*_alpha0*(2*_np.pi*_Kn)*_Pe + (2*_np.pi*_Kn)**2
    	_PHI2n_e = _xr.dot(_ndAs, _dAs['phi_2n'] * _np.exp(- arg_e*(_t[i] - _tf)), dims='n')
    	_PHI2n_o = _xr.dot(_ndBs, _dBs['phi_2n'] * _np.exp(- arg_o*(_t[i] - _tf)), dims='n')
    	_PHI2n = _PHI2n_e + _PHI2n_o
    	T0 = _xrft.ifft(_da_xrft * _PHI2n, dim='k', true_phase=True, true_amplitude=True).real
    	nT0 = T0.rename({'freq_k':'x'}).transpose('y', 'x')
    	ds['Theta'].data[i, :, :] = nT0.data

    return ds, _PHI2n



def evolve_ds_modal_time(_DAS, _indt, _order, _vals, _K0, _ALPHA0, _Pe, _gauss_alps, _facs, _X, _Y, _t):
	"""
	Evolve an initial condition defined in Fourier space by its y-F. coefficients, and the along-flow wavenumber k.

	Input:
		_DAS : List. each element contains a reference to the spectrum of the advection diffusion operator.
		_indt: index in time.
		_order:
		_vals:

	"""
	DS = []
	ncoeffs = _copy.deepcopy(_gauss_alps)
	for i in range(len(_indt)):
		if i == 0:
			tf = 0
		else:
			tf =_t[_indt[i - 1][1] - 1]
		ds, phi = evolve_ds_modal_gaussian(_DAS[_order[i]], _K0, _ALPHA0[_order[i]], abs(_vals[_order[i]])*_Pe, ncoeffs, _facs, _X, _Y, _t[_indt[i][0]:_indt[i][1]], tf)
		DS.append(ds)
		ncoeffs, odd_coeffs, phi_new, phi_old  = coeff_project(phi, _Y[:, 0])
	
	for i in range(len(DS)):
		if i ==0:
			ds_f = DS[i]
		else:
			ds_f = ds_f.combine_first(DS[i])
	
	return ds_f


def evolve_ds_time(_DAS, _indt, _order, _vals, _Kn, _ALPHA0, _Pe, _da_dft, _a_alps, _facs, _x, _y, _t):
	"""
	evolves a localized initial condition defined by its 2d Fourier coefficients.
	"""
	DS = []
	ncoeffs = _copy.deepcopy(_a_alps)
	for i in range(len(_indt)):
		if i == 0:
			tf = 0
		else:
			tf =_t[_indt[i - 1][1] - 1]
		ds, Phi2n = evolve_ds(_DAS[_order[i]], _da_dft, _Kn, _ALPHA0[_order[i]], abs(_vals[_order[i]])*_Pe, ncoeffs, _facs, _x, _y, _t[_indt[i][0]:_indt[i][1]],  t0)
		DS.append(ds)
		ncoeffs, odd_coeffs, phi_new, phi_old  = coeff_project(Phi2n, _y)
    
	for i in range(len(DS)):
		if i ==0:
			ds_f = DS[i]
		else:
			ds_f = ds_f.combine_first(DS[i])
	return ds_f


def evolve_off_ds_time(_DAS, _DBS, _indt, _order, _vals, _Kn, _ALPHA0, _Pe, _da_dft, _a_alps, _afacs, _b_alps, _bfacs, _x, _y, _t, _shift=0):
	"""evolves a localized initial condition defined by its 2d Fourier coefficients.

	shift must be a list


	"""
	DS = []
	PHI_NEW = []
	PHI_OLD = []
	ecoeffs = _copy.deepcopy(_a_alps)
	ocoeffs = _copy.deepcopy(_b_alps)
	if len(_shift) == 1:
		val = _shift[0]
		_shift = [val for i in range(len(_t))]
	for i in range(len(_indt)):
		phi_new = _shift[_indt[i][0]]  # only sample first - they are all the same
		if i == 0:
			t1 = _t[_indt[i][0]:_indt[i][1]]
			t0 = 0
			phi_old = _np.pi
			ndAs = _xr.dot(_afacs * ecoeffs, _DAS[_order[i]]['A_2r'])
			ndBs = _xr.dot(_bfacs * ocoeffs, _DBS[_order[i]]['B_2r'])
			PHI2n_e = _xr.dot(ndAs, _DAS[_order[i]]['phi_2n'], dims='n')
			PHI2n_o = _xr.dot(ndBs, _DBS[_order[i]]['phi_2n'], dims='n')
			Phi2n = PHI2n_e + PHI2n_o
		else:
			t0 =_t[_indt[i - 1][1] - 1]
			t1 = _t[_indt[i][0]-1:_indt[i][1]]
			ecoeffs, ocoeffs, phi_new, phi_old  = coeff_project(Phi2n, _y/2, phi_old=phi_old, phi_new=phi_new)  # will have to modify here
		ds, Phi2n = evolve_ds_off(_DAS[_order[i]], _DBS[_order[i]], _da_dft, _Kn, _ALPHA0[_order[i]], abs(_vals[_order[i]])*_Pe, ecoeffs, _afacs, ocoeffs, _bfacs,  _x, _y, t1, t0)
		DS.append(_copy.deepcopy(ds))
		PHI_NEW.append(phi_new)
		PHI_OLD.append(phi_old)
	for i in range(len(DS)):
		if i == 0:
			ds_f = DS[i]
		else:
			jump = abs(PHI_OLD[i] - PHI_OLD[0])
			dsign = int(_np.sign(PHI_OLD[i] - PHI_OLD[0]))
			diff = abs(_y - jump)
			ii = dsign * _np.where(diff == _np.min(diff))[0][0]
			ds_f = ds_f.combine_first(DS[i].roll(y=ii, roll_coords=False))
	return ds_f, DS, PHI_NEW, PHI_OLD


def evolve_ds_rot(_dAs, _da_xrft, _L, _alpha0, _Pe, _a_alps, _afacs, _x, _y, _t,  _tf=0):
    """Constructs the solution to the IVP. Shear flow aligned with y"""
    
    coords = {"time": _t, "y": _y, "x": _x}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["time", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    _ndAs = _xr.dot(_afacs * _a_alps, _dAs['A_2r'], dims='r')
    for i in range(len(_t)):
        arg = 0.25*_dAs['a_2n'] + (1j)*_alpha0*(2*_np.pi*_L)*_Pe + (2*_np.pi*_L)**2
        _PHI2n = _xr.dot(_ndAs, _dAs['phi_2n'] * _np.exp(- arg*(_t[i] - _tf)), dims='n')
        T0 = _xrft.ifft(_da_xrft * _PHI2n, dim='l', true_phase=True, true_amplitude=True).real
        nT0 = T0.rename({'freq_l':'y'})
        ds['Theta'].data[i, :, :] = nT0.data
    return ds, _PHI2n


def evolve_ds_off_rot(_dAs, _dBs, _da_xrft, _L, _alpha0, _Pe, _a_alps, _afacs, _b_alps, _bfacs, _x, _y, _t, _tf=0):
    """Constructs the solution to the IVP"""
    ## Initialize the array
    coords = {"time": _t, "y": _y, "x": _x}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["time", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    _ndAs = _xr.dot(_afacs * _a_alps, _dAs['A_2r'], dims='r')
    _ndBs = _xr.dot(_bfacs * _b_alps, _dBs['B_2r'], dims='r')
    for i in range(len(_t)):
    	arg_e = 0.25*_dAs['a_2n'] + (1j)*_alpha0*(2*_np.pi*_L)*_Pe + (2*_np.pi*_L)**2
    	arg_o = 0.25*_dBs['b_2n'] + (1j)*_alpha0*(2*_np.pi*_L)*_Pe + (2*_np.pi*_L)**2
    	_PHI2n_e = _xr.dot(_ndAs, _dAs['phi_2n'] * _np.exp(- arg_e*(_t[i] - _tf)), dims='n')
    	_PHI2n_o = _xr.dot(_ndBs, _dBs['phi_2n'] * _np.exp(- arg_o*(_t[i] - _tf)), dims='n')
    	_PHI2n = _PHI2n_e + _PHI2n_o
    	T0 = _xrft.ifft(_da_xrft * _PHI2n, dim='l', true_phase=True, true_amplitude=True).real
    	nT0 = T0.rename({'freq_l':'y'})
    	ds['Theta'].data[i, :, :] = nT0.data

    return ds, _PHI2n


def evolve_ds_rot_time(_DAS, _indt, _order, _vals, _Ln, _ALPHA0, _Pe, _da_dft, _a_alps, _afacs, _x, _y, _t):
	"""
	evolves a localized initial condition defined by its 2d Fourier coefficients.
	"""
	DS = []
	ncoeffs = _copy.deepcopy(_a_alps)
	for i in range(len(_indt)):
		if i == 0:
			tf = 0
		else:
			tf =_t[_indt[i - 1][1] - 1]
		ds, Phi2n = evolve_ds_rot(_DAS[_order[i]], _da_dft, _Ln, _ALPHA0[_order[i]], abs(_vals[_order[i]])*_Pe, ncoeffs, _afacs, _x, _y, _t[_indt[i][0]:_indt[i][1]], tf)
		DS.append(ds)
		ncoeffs, odd_coeffs, phi_new, phi_old  = coeff_project(Phi2n, _x / 2, dim='x')
    
	for i in range(len(DS)):
		if i ==0:
			ds_f = DS[i]
		else:
			ds_f = ds_f.combine_first(DS[i])

	return ds_f, phi_new, phi_old


def evolve_off_ds_rot_time(_DAS, _DBS, _indt, _order, _vals, _Ln, _ALPHA0, _Pe, _da_dft, _a_alps, _afacs, _b_alps, _bfacs, _x, _y, _t, _shift=0):
	"""evolves a localized initial condition defined by its 2d Fourier coefficients."""
	DS = []
	PHI_NEW = []
	PHI_OLD = []
	ecoeffs = _copy.deepcopy(_a_alps)
	ocoeffs = _copy.deepcopy(_b_alps)
	if len(_shift) == 1:
		val = _shift[0]
		_shift = [val for i in range(len(_t))]
	for i in range(len(_indt)):
		phi_new = _shift[_indt[i][0]]  # only sample first - they are all the same
		if i == 0:
			t0 = 0
			t1 = _t[_indt[i][0]:_indt[i][1]]
			phi_old = 0
			ndAs = _xr.dot(_afacs * ecoeffs, _DAS[_order[i]]['A_2r'])
			ndBs = _xr.dot(_bfacs * ocoeffs, _DBS[_order[i]]['B_2r'])
			PHI2n_e = _xr.dot(ndAs, _DAS[_order[i]]['phi_2n'], dims='n')
			PHI2n_o = _xr.dot(ndBs, _DBS[_order[i]]['phi_2n'], dims='n')
			Phi2n = PHI2n_e + PHI2n_o
		else:
			t0 =_t[_indt[i - 1][1] - 1]
			t1 = _t[_indt[i][0]-1:_indt[i][1]]
		ecoeffs, ocoeffs, phi_new, phi_old  = coeff_project(Phi2n, _x/2, phi_old=phi_old, phi_new=phi_new, dim='x')  # will have to modify here
		ds, Phi2n = evolve_ds_off_rot(_DAS[_order[i]], _DBS[_order[i]], _da_dft, _Ln, _ALPHA0[_order[i]], abs(_vals[_order[i]])*_Pe, ecoeffs, _afacs, ocoeffs, _bfacs,  _x, _y, t1, t0)
		DS.append(_copy.deepcopy(ds))
		PHI_NEW.append(phi_new)
		PHI_OLD.append(phi_old)
	for i in range(len(DS)):
		if i == 0:
			ds_f = DS[i]
		else:
			jump = abs(PHI_OLD[i] - PHI_OLD[0])
			dsign = int(_np.sign(PHI_OLD[i] - PHI_OLD[0]))
			diff = abs(_x - jump)
			ii = dsign * _np.where(diff == _np.min(diff))[0][0]
			ds_f = ds_f.combine_first(DS[i].roll(x=ii, roll_coords=False))
	return ds_f, DS, PHI_NEW, PHI_OLD


def evolve_ds_serial(_dAs, _Kn, _alpha0, _Pe, _gauss_alps, _facs, _x, _y, _t, _tf=0, _dim='k'):
	"""Constructs the modal solution to the IVP that is localized across the jet."""
	coords = {"t": _t, "y": _y, _dim: _Kn, 'x': _x}
	Temp = _xr.DataArray(_np.nan, coords=coords, dims=["t", 'y', 'x', _dim])
	DS = []
	if _dim == 'k':
		phi_arg = {'x':_x}
	elif _dim == 'l':
		phi_arg = {'y':_y}
	for kk in range(len(_Kn)):
		_K = _Kn[kk]
		k_args = {_dim: _K}
		ds = _xr.Dataset({'Theta': Temp})
		if _dim in _gauss_alps.dims:
			ndAs = _xr.dot(_facs * _gauss_alps.sel(**k_args), _dAs['A_2r'].sel(**k_args))
		else:
			ndAs = _xr.dot(_facs * _gauss_alps, _dAs['A_2r'].sel(**k_args))
		exp_arg =  0.25*_dAs['a_2n'].sel(**k_args) + (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		for i in range(len(_t)):
			PHI2n = _xr.dot(ndAs, _dAs['phi_2n'].sel(**k_args) * _np.exp(-exp_arg*(_t[i]-_tf)), dims='n')
			if _dim == 'k':
				PHI2n = PHI2n.expand_dims(**phi_arg).transpose('y', 'x')
				mode =  _np.exp((2*_np.pi* _K * _x) * (1j))
			else:
				PHI2n = PHI2n.expand_dims(**phi_arg)
				mode =  _np.exp((2*_np.pi* _K * _y) * (1j))
			T0 = (PHI2n * mode).real
			ds['Theta'].data[i, :, :, kk] = T0.data
	ll = int(_np.where(_Kn==0)[0][0]) # single zero
	dk = _Kn[ll + 1] / 2
	da = ds['Theta'].sum(dim=_dim) * dk
	coords = {"time": _t, "y": _Y[:, 0], 'x': _X[0, :]}
	da_final = _xr.DataArray(da.real, coords=coords, dims=['time', 'y', 'x'])
	ds_final = _xr.Dataset({'Theta': da_final})
	return ds_final


def evolve_ds_serial_off(_dAs, _dBs, _Kn, _alpha0, _Pe, _a_alps, _afacs, _b_alps, _bfacs, _x, _y, _t, _tf=0, _dim='k'):
	"""Constructs the modal solution to the IVP that is localized across the jet."""
	coords = {"t": _t, "y": _y, _dim: _Kn, 'x': _x}
	_X, _Y = _np.meshgrid(_x, _y)
	Temp = _xr.DataArray(coords=coords, dims=["t", 'y', 'x', _dim])
	DS = []
	if _dim == 'k':
		phi_arg = {'x':_x}
	elif _dim == 'l':
		phi_arg = {'y':_y}
	for kk in range(len(_Kn)):
		_K = _Kn[kk]
		k_args = {_dim: kk}
		ds = _xr.Dataset({'Theta': Temp})
		if _dim in _a_alps.dims:
			ndAs = _xr.dot(_afacs * _a_alps.isel(**k_args), _dAs['A_2r'].isel(**k_args))
			ndBs = _xr.dot(_bfacs * _b_alps.isel(**k_args), _dBs['B_2r'].isel(**k_args))
		else:
			ndAs = _xr.dot(_afacs * _a_alps, _dAs['A_2r'].isel(**k_args))
			ndBs = _xr.dot(_bfacs * _b_alps, _dAs['B_2r'].isel(**k_args))
		exp_arg_e =  0.25*_dAs['a_2n'].isel(**k_args) + (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		exp_arg_o =  0.25*_dBs['b_2n'].isel(**k_args) + (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		for i in range(len(_t)):
			PHI2n_e = _xr.dot(ndAs, _dAs['phi_2n'].isel(**k_args) * _np.exp(-exp_arg_e*(_t[i]-_tf)), dims='n')
			PHI2n_o = _xr.dot(ndBs, _dBs['phi_2n'].isel(**k_args) * _np.exp(-exp_arg_o*(_t[i]-_tf)), dims='n')
			PHI2n = PHI2n_e + PHI2n_o
			if _dim == 'k':
				PHI2n = PHI2n.expand_dims(**phi_arg).transpose('y', 'x')
				mode =  _np.exp((2*_np.pi* _K * _X) * (1j))
			else:
				PHI2n = PHI2n.expand_dims(**phi_arg)
				mode =  _np.exp((2*_np.pi* _K * _Y) * (1j))
			T0 = (PHI2n * mode).real
			ds['Theta'].data[i, :, :, kk] = T0.data
	ll = int(_np.where(_Kn==0)[0][0]) # single zero
	dk = _Kn[ll + 1] / 2
	da = ds['Theta'].sum(dim=_dim) * dk
	coords = {"time": _t, "y": _y, 'x': _x}
	da_final = _xr.DataArray(da.real, coords=coords, dims=['time', 'y', 'x'])
	ds_final = _xr.Dataset({'Theta': da_final})
	return ds_final


def renewing_evolve(_dAs, _dBs, _dAs_rot,_dBs_rot, _alpha0, _Pe, _Theta0, _vals, _order, _indt, _x, _y, _t):
	"""Computes the evolution of a passive scalar in the case the velocity field is renewing. Square domain.
	By construction, the velocity field begins with an along- x orientation.
	Input:
		_dAs, _dBs: datasets of non-rotated spectra associated with non-rotated shear flow.
		_dAs_rot, _dBs_rot: datasets with spectra associated with rotatede shear flows.
		_alpha0: Mean velocity.
		_Pe: float. Peclet number.
		_Theta0: Initial condition. xarray.DataArray.
		_X, _Y: 2d arrays (each). Together define the grid, assumed to be square domains
		_t: 1d array. Time with t[0]=0.
		_tau: float, element of _t. defines the frequency of velocity rotation.
	Output:
		ds: xarray.dataset 
	"""
	xt = _x / 2
	yt = _y / 2

	IND, ORDER, Time = split_signal(_vals, _order, _indt, _t)
	NT = len(ORDER)
    
	da_dft = _xrft.fft(_Theta0.transpose(), dim='x', true_phase=True, true_amplitude=True)
	da_dft = da_dft.rename({'freq_x':'k'})
	Kn = _copy.deepcopy(da_dft['k'].values)
    
	da_y = _xrft.fft(_Theta0, dim='y', true_phase=True, true_amplitude=True)
	da_y = da_y.rename({'freq_y':'l'})
	Ln = _copy.deepcopy(da_y['l'].values)
    
	even_coeffs, odd_coeffs, phi_new, phi_old = coeff_project(da_dft, yt)

	acoords = {'r':range(len(even_coeffs))}
	bcoords = {'r':range(len(odd_coeffs)-1)}
	afacs = _np.ones(_np.shape(range(len(even_coeffs))))
	afacs[0] = 2

	bfacs = _np.ones(_np.shape(range(len(odd_coeffs) - 1)))

	afacs = _xr.DataArray(afacs, coords=acoords, dims='r')
	bfacs = _xr.DataArray(bfacs, coords=bcoords, dims='r')

#     Initialize evolution
	d0 = evolve_ds_serial_off(_dAs, _dBs, Kn, _alpha0, _Pe, even_coeffs, afacs, odd_coeffs, bfacs, _x, _y, Time[0])
    
	for i in range(1, NT):
		da_step = d0['Theta'].isel(time=-1)
		t0 = Time[i-1][-1]
		t1 = Time[i]
		if i % 2 != 0:  # if odd number.
			da_dft = _xrft.fft(da_step, dim='y', true_phase=True, true_amplitude=True)
			da_dft = da_dft.rename({'freq_y':'l'})
			even_coeffs, odd_coeffs, phi_new, phi_old = coeff_project(da_dft, xt, dim='x')
			d1 = evolve_ds_serial_off(_dAs_rot, _dBs_rot, Ln, _alpha0, _Pe, even_coeffs, afacs, odd_coeffs, bfacs, _x, _y, t1, t0, _dim='l')
		else:
			da_dft = _xrft.fft(da_step.transpose(), dim='x', true_phase=True, true_amplitude=True)
			da_dft = da_dft.rename({'freq_x':'k'})
			even_coeffs, odd_coeffs, phi_new, phi_old = coeff_project(da_dft, yt)
			d1 = evolve_ds_serial_off(_dAs,_dBs, Kn, _alpha0, _Pe, even_coeffs, afacs, odd_coeffs, bfacs, _x, _y, t1, t0, _dim='k')
		d0 = d0.combine_first(d1)
	return d0,


def renewing_evolve_new(_DAS, _DBS, _DAS_rot, _DBS_rot, _ALPHA0,  _Pe, _vals, _order, _indt, _Theta0,  _x, _y, _t):

	xt = _x / 2
	yt = _y / 2

	IND, ORDER, Time = split_signal(_vals, _order, _indt, _t)
	NT = len(ORDER)
    
	da_dft = _xrft.fft(_Theta0.transpose(), dim='x', true_phase=True, true_amplitude=True)
	da_dft = da_dft.rename({'freq_x':'k'})
	Kn = _copy.deepcopy(da_dft['k'].values)
    
	da_y = _xrft.fft(_Theta0, dim='y', true_phase=True, true_amplitude=True)
	da_y = da_y.rename({'freq_y':'l'})
	Ln = _copy.deepcopy(da_y['l'].values)
    
	even_coeffs, odd_coeffs, phi_new, phi_old = coeff_project(da_dft, yt)

	acoords = {'r':range(len(even_coeffs))}
	bcoords = {'r':range(len(odd_coeffs)-1)}
	afacs = _np.ones(_np.shape(range(len(even_coeffs))))
	afacs[0] = 2

	bfacs = _np.ones(_np.shape(range(len(odd_coeffs) - 1)))

	afacs = _xr.DataArray(afacs, coords=acoords, dims='r')
	bfacs = _xr.DataArray(bfacs, coords=bcoords, dims='r')

	# initialize 
	ii = 0
	t1 = _t[IND[ii][0][0]:IND[ii][0][-1]]
	dsA = _DAS[ORDER[ii][0]]
	dsB = _DBS[ORDER[ii][0]]
	alpha0 = _ALPHA0[ORDER[ii][0]]
	nPe = abs(_vals[ORDER[ii][0]]) * _Pe
	d0 = evolve_ds_serial_off(dsA, dsB, Kn, alpha0, nPe, even_coeffs, afacs, odd_coeffs, bfacs, _x, _y, t1)
	dstep = d0['Theta'].isel(time=-1)
	da_dft = _xrft.fft(dstep.transpose(), dim='x', true_phase=True, true_amplitude=True)
	da_dft = da_dft.rename({'freq_x':'k'})
	even_coeffs, odd_coeffs, phi_new, phi_old = coeff_project(da_dft, yt)

	for jj in range(1, len(IND[ii])): # iterates over elements
		dsA = _DAS[ORDER[ii][jj]]
		dsB = _DBS[ORDER[ii][jj]]
		alpha0 = _ALPHA0[ORDER[ii][jj]]
		nPe = abs(_vals[ORDER[ii][jj]]) * _Pe
		t0 = _t[IND[ii][jj-1][-1]-1]
		t1 = _t[IND[ii][jj][0]:IND[ii][jj][-1]] #?

		# evolve
		d1 = evolve_ds_serial_off(dsA, dsB, Kn, alpha0, nPe, even_coeffs, afacs, odd_coeffs, bfacs, _x, _y, t1, t0)
		dstep = d1['Theta'].isel(time=-1)
		if jj < len(IND[ii])-1:
			da_dft = _xrft.fft(dstep.transpose(), dim='x', true_phase=True, true_amplitude=True)
			da_dft = da_dft.rename({'freq_x':'k'})
			even_coeffs, odd_coeffs, phi_new, phi_old = coeff_project(da_dft, yt)

		d0 = d0.combine_first(d1)


	for ii in range(1, len(IND)):
		if ii % 2 == 0:  # if even number.
			_DA = _DAS
			_DB = _DBS
			_wvn = Kn
			_dim = 'x'
			_dimf = 'y'
			_wa = 'k'
			_rename = {'freq_x':_wa}
			_coor = yt
			_tp = True
		else:  # rotated
			_DA = _DAS_rot
			_DB = _DBS_rot
			_wvn = Ln
			_dim = 'y'
			_dimf = 'x'
			_wa = 'l'
			_rename = {'freq_y':_wa}
			_coor = xt
			_tp = False

		for jj in range(len(IND[ii])): # iterates over elements
			if jj==0:
				t0 = _t[IND[ii-1][-1][-1]-1]  # last from previous iter
				if _tp:
					dstep = dstep.transpose()
				da_dft = _xrft.fft(dstep, dim=_dim, true_phase=True, true_amplitude=True)
				da_dft = da_dft.rename(_rename)
				even_coeffs, odd_coeffs, phi_new, phi_old = coeff_project(da_dft, _coor, dim=_dimf)
			else:
				t0 = _t[IND[ii][jj-1][-1]-1]
			t1 = _t[IND[ii][jj][0]:IND[ii][jj][-1]] # ?
			alpha0 = _ALPHA0[ORDER[ii][jj]]
			nPe = abs(_vals[ORDER[ii][jj]]) * _Pe
			dsA = _DA[ORDER[ii][jj]]
			dsB = _DB[ORDER[ii][jj]]
			# evolve
			d1 = evolve_ds_serial_off(dsA, dsB, _wvn, alpha0, nPe, even_coeffs, afacs, odd_coeffs, bfacs, _x, _y, t1, t0, _dim = _wa)
			dstep = d1['Theta'].isel(time=-1)
			if jj < len(IND[ii]) - 1:
				if _tp:
					dstep = dstep.transpose()
				da_dft = _xrft.fft(dstep, dim=_dim, true_phase=True, true_amplitude=True)
				da_dft = da_dft.rename(_rename)
				even_coeffs, odd_coeffs, phi_new, phi_old = coeff_project(da_dft, _coor, dim = _dimf)

			d0 = d0.combine_first(d1)

	return d0


def evolve_forcing_modal(_da_xrft, _dAs, _K, _Ubar, _Pe, _delta, _Q0, _X, _Y, _t, _tf=0):
	"""Evolves the solution to the advection diffusion eqn for a steady shear flow in the presence of 
    external forcing Q(x,y). The shear flow is defined solely by a cosine Fourier series and so
    is the forcing. The forcing has the form
    	Q(k, y) = cos(y)
	
	Parameters
	----------

		_da_xrft: xarray.dataarray.
			Contains the Fourier coefficients in x, and has dimension `k`. Output of xrft.fft(da).
		_dAs: xarray.dataset.
			Contains eigenvalues, eigenvectors and eigenfunctions associateed with the operator.
		_K: numpy.array (1D like).
			array with all along-strong wavenumbers. Determined by the discretization of the domain.
			Is calculated as output from xrft.fft(_da).
		_Ubar: float.
			Mean velocity calculated as the numerical average of U(y, t_0). at t=t_0.
		_Pe: float.
			Peclet number.
		_Q0: float.
			amplitude of forcing scaled by diffusive timescale t_d.
		_X, _Y: 1d-array like (numpy)
			Span the domain. Non-dimensional.
		_time: 1d array-like (numpy)
			Time array.
		_tf: float.
			Initial time. Default is zero, but can vary in case shear flow is time-dependent.

	
	Returns:

		_ds: xarray.dataset
			Contains Theta the analytical solution. 
	"""
	coords = {"time": _t, "y": 2 * _Y, "x": _X}
	Temp = _xr.DataArray(_np.nan, coords=coords, dims=["time", 'y', 'x'])
	ds = _xr.Dataset({'Theta_p': Temp, 'Theta_h': Temp, 'Theta': Temp})
	exp_arg = (1j)*_Ubar*(2* _np.pi*_K)*_Pe + (2* _np.pi*_K)**2
	exp2 = _dAs['a_2n'] + 4*(1j)*(2*_np.pi*_K)*_Pe * _Ubar + 4*(2* _np.pi*_K)**2 + 4*(1j) * _delta
	ndAs_p =  4*_Q0 * _dAs['A_2r'].isel(r=1) / exp2  # this defines a single mode.
	ndAs_h =  -ndAs_p
	for i in range(len(_t)):
		PHI2n_h = _xr.dot(ndAs_h, _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*_t[i]), dims='n')
		PHI2n_p = _xr.dot(ndAs_p, _dAs['phi_2n'] * _np.exp((1j)* _delta * _t[i]), dims='n')
		T0 = _xrft.ifft(_da_xrft * PHI2n_h, dim='k', true_phase=True, true_amplitude=True).real
		T0 = T0.rename({'freq_k':'x'}).transpose('y', 'x')
		Tp = _xrft.ifft(_da_xrft * PHI2n_p, dim='k', true_phase=True, true_amplitude=True).real
		Tp = Tp.rename({'freq_k':'x'}).transpose('y', 'x')
		ds['Theta_h'].data[i, :, :] = T0.data
		ds['Theta_p'].data[i, :, :] = Tp.data
		ds['Theta'].data[i, :, :] = (Tp + T0).data
	return ds



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



def phase_generator(nft):
	"""generates a phase shift that coincides with the change in sign of a (discretized) signal.
	By default, the initial phase is zero. Note, there must be a zero crossing, or phase is constant.
	Input:
		nft - numpy.array. Real, float elements. Must be output from re_sample. 
	"""
	phase = 0 * nft
	if _np.isin(nft, 0).any():
		ll = _np.where(nft == 0)[0]
		# check for repeated zeros
		d = ll[1:]-ll[:-1]  
		ld = _np.where(d>1)[0]
		if len(ld) > 2:
			l = []
			for i in range(len(ld)):
				_rd = _np.random.random_sample() # positive rand number <1
				_rd = _rd * _np.sign(_np.cos(2*_np.pi*_rd))  # assign a rand sign
				l.append(ll[ld[i]])
				if i ==0:
					phase[:l[i]] = 0
				else:
					phase[l[i-1]:l[i]] = _rd
			_rd = _np.random.random_sample()
			phase[ll[-1]:] =  _rd * _np.sign(_np.cos(2*_np.pi*_rd)) 
		else:
			_rd = _np.random.random_sample() # positive rand number <1
			_rd = _rd * _np.sign(_np.cos(2*_np.pi*_rd))  # assign a rand sign
			phase[ld[0]:] = _rd 

	else:
		phase[:] = _np.random.random_sample()

	return phase


def split_signal(_vals, _order, _indt, _t, n=2):
	"""
	Takes a (discretized) time-varying amplitude of the shear flow, identifies
	the sign reversals and returns a collection of lists to evaluate eigenfunctions
	in a manner that is ordered according to the renewing flow problem.

	Parameters:
		_vals: list.
			Discretized velocity values, monotonically increasing. At a minimum
			must unclude [-1, 0, 1]. 
		_order: list.
			Ordered list of indexes of variable `_vals` for each time step. For example, 
			if _vals=[-1, 0, 1], order=[2, 1, 0, 1, 2]. Implies Vel starts at 1,
			decreases to -1 and ends at 1.
		_indt: list
			Map between time-variable (with len = nt) and _order.
		_t: 1d np.array.
			Discretized time.
		n: int
			n= 1 or n=2 (default is 2). On a time-periodic (amp) signal, n=1 implies
			the shear flow rotates as soon as the amplitude goes to zero. n=2 allows
			the shear flow to reverse direction before rotating. 
	
	Returns:

		_IND, _ORDER, _T
	"""

	_n0 = _np.where(_np.array(_vals) == 0)[0][0]
	lll = _np.where(_np.array(_order) == _n0)[0][1::n] # skips the first zero.
	_IND = [_indt[:lll[0]+1]]
	_T = [_t[:_indt[:lll[0]+1][-1][-1]]]
	_ORDER = [_order[:lll[0]+1]]
	for i in range(1, len(lll)):
		_T.append(_t[_indt[:lll[i-1]+1][-1][-1]: _indt[:lll[i]+1][-1][-1]])
		_IND.append(_indt[lll[i-1]+1:lll[i]+1])
		_ORDER.append(_order[lll[i-1]+1:lll[i]+1])
	_T.append(_t[_indt[lll[i]][0]+1:])
	_IND.append(_indt[lll[i]+1:])
	_ORDER.append(_order[lll[i]+1:])

	return _IND, _ORDER, _T








