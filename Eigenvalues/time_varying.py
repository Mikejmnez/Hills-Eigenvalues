"""
defines functions that allow time-evolution
"""

import numpy as _np
import copy
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
	if phi_old > 2*_np.pi:
		phi_old = phi_old - 2*_np.pi
	elif phi_old < 0:
		phi_old = phi_old + 2*_np.pi

	fac = _np.pi * frac   # divides the Fourier coefficients (xrft scales them).

	L = len(_y)  # number of wavenumbers
	if (L - 1) % 2 == 0:  # number should be odd.
		nL = int((L - 1) / 2)

	da_phi = _xr.DataArray(_phi, dims=_phi.dims, coords=_phi.coords)
	da_dft_phi = _xrft.fft(da_phi, dim=dim, true_phase=True, true_amplitude=True)
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


def evolve_ds_modal_uniform(_dAs, _K, _alpha0, _Pe, _X, _Y, _time, _tf=0):
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


def evolve_ds_modal(_dAs, _K, _alpha0, _Pe, _gauss_alps, _facs, _X, _Y, _time, _tf=0):
	"""Constructs the modal solution to the IVP that is localized across the jet."""

	coords = {"t": _time, "y": 2 * _Y[:, 0], "x": _X[0, :]}
	Temp = _xr.DataArray(_np.nan, coords=coords, dims=["t", 'y', 'x'])
	ds = _xr.Dataset({'Theta': Temp})
	Nr = len(_dAs.n) # length of truncated array
	ndAs = complement_dot(_facs * _gauss_alps, _dAs)  # has final size in n (sum in p)
	for i in range(len(_time)):
		exp_arg =  (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		PHI2n = _xr.dot(ndAs, _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*(_time[i]-_tf)), dims='n')
		PHI2n = PHI2n.sel(k=_K).expand_dims({'x':_X[0, :]}).transpose('y', 'x')
		T0 = (PHI2n * _np.exp((2*_np.pi* _K * _X) * (1j))).real
		ds['Theta'].data[i, :, :] = T0.data
	return ds, PHI2n.isel(x=0).drop_vars({'x'})  # return the eigenfunction sum


def evolve_ds_modal_off(_dAs, _dBs, _K, _alpha0, _Pe, _a_alps, _afacs, _b_alps, _bfacs, _X, _Y, _t, _tf=0):
	"""Constructs the modal solution to the IVP that is localized across the jet,
	with arbitrary location in y"""

	coords = {"time": copy.deepcopy(_t), "y": 2 * _Y[:, 0], "x": _X[0, :]}
	Temp = _xr.DataArray(_np.nan, coords=coords, dims=["time", 'y', 'x'])
	ds = _xr.Dataset({'Theta': Temp})
	_ndAs = _xr.dot(_afacs * _a_alps, _dAs['A_2r'], dims='r')
	_ndBs = _xr.dot(_bfacs * _b_alps, _dBs['B_2r'], dims='r')
	for i in range(len(_time)):    	
		arg_e = 0.25*_dAs['a_2n'] + (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		arg_o = 0.25*_dBs['b_2n'] + (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		_PHI2n_e = _xr.dot(_ndAs, _dAs['phi_2n'] * _np.exp(- arg_e*(_t[i] - _tf)), dims='n')
		_PHI2n_o = _xr.dot(_ndBs, _dBs['phi_2n'] * _np.exp(- arg_o*(_t[i] - _tf)), dims='n')
		_PHI2n = _PHI2n_e + _PHI2n_o
		_PHI2n = _PHI2n.sel(k=_K).expand_dims({'x':_X[0, :]}).transpose('y', 'x')
		T0 = (_PHI2n * _np.exp((2*_np.pi* _K * _X) * (1j))).real
		ds['Theta'].data[i, :, :] = T0.data
	return ds, PHI2n.isel(x=0).drop_vars({'x'})  # return the eigenfunction sum



def evolve_ds(_dAs, _da_xrft, _K, _alpha0, _Pe, _gauss_alps, _facs, _x, _y, _time, _tf=0):
    """Constructs the solution to the IVP"""
    ## Initialize the array.
    coords = {"t": _time, "y": 2 * _y, "x": _x}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["t", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    Nr = len(_dAs.n) # length of truncated array
    ndAs = complement_dot(_facs*_gauss_alps, _dAs)  # has final size in n (sum in p)
    for i in range(len(_time)):
        exp_arg =  (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
        # if Nr < len(_facs):  # Is this necessary?
        #     PHI2n = _xr.dot(ndAs.isel(n=slice(Nr)), _dAs['phi_2n'].isel(n=slice(Nr)) * _np.exp(-(0.25*_dAs['a_2n'].isel(n=slice(Nr)) + exp_arg)*(_time[i]-_tf)), dims='n')
        #     PHI2n = PHI2n + _xr.dot(ndAs[Nr:], _dAs['phi_2n'][Nr:] * _np.exp(-(0.25*_dAs['a_2n'][Nr:] + exp_arg)*(_time[i]-_tf)), dims='n')
        # else:
        PHI2n = _xr.dot(ndAs, _dAs['phi_2n'] * _np.exp(-(0.25*_dAs['a_2n'] + exp_arg)*(_time[i] - _tf)), dims='n')
        T0 = _xrft.ifft(_da_xrft * PHI2n, dim='k', true_phase=True, true_amplitude=True).real # Signal in direct space
        nT0 = T0.rename({'freq_k':'x'}).transpose('y', 'x')
        ds['Theta'].data[i, :, :] = nT0.data
    return ds, PHI2n



def evolve_ds_off(_dAs, _dBs, _da_xrft, _K, _alpha0, _Pe, _a_alps, _afacs, _b_alps, _bfacs, _x, _y, _t, _tf=0):
    """Constructs the solution to the IVP"""
    ## Initialize the array
    coords = {"time": _t, "y": 2 * _y, "x": _x}
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
    	T0 = _xrft.ifft(_da_xrft * _PHI2n, dim='k', true_phase=True, true_amplitude=True).real
    	nT0 = T0.rename({'freq_k':'x'}).transpose('y', 'x')
    	ds['Theta'].data[i, :, :] = nT0.data

    return ds, _PHI2n



def evolve_ds_modal_time(_DAS, _indt, _order, _vals, _K0, _ALPHA0, _Pe, _gauss_alps, _facs, _X, _Y, _time):
	"""
	Evolve an initial condition defined in Fourier space by its y-F. coefficients, and the along-flow wavenumber k.

	Input:
		_DAS : List. each element contains a reference to the spectrum of the advection diffusion operator.
		_indt: index in time.
		_order:
		_vals:

	"""
	DS = []
	ncoeffs = copy.deepcopy(_gauss_alps)
	for i in range(len(_indt)):
		if i == 0:
			tf = 0
		else:
			tf =_time[_indt[i - 1][1] - 1]
		ds, phi = evolve_ds_modal_gaussian(_DAS[_order[i]], _K0, _ALPHA0[_order[i]], abs(_vals[_order[i]])*_Pe, ncoeffs, _facs, _X, _Y, _time[_indt[i][0]:_indt[i][1]], tf)
		DS.append(ds)
		ncoeffs, odd_coeffs, phi_new, phi_old  = coeff_project(phi, _Y[:, 0])
	
	for i in range(len(DS)):
		if i ==0:
			ds_f = DS[i]
		else:
			ds_f = ds_f.combine_first(DS[i])
	
	return ds_f


def evolve_ds_time(_DAS, _indt, _order, _vals, _Kn, _ALPHA0, _Pe, _da_dft, _gauss_alps, _facs, _x, _y, _time):
	"""
	evolves a localized initial condition defined by its 2d Fourier coefficients.
	"""
	DS = []
	ncoeffs = copy.deepcopy(_gauss_alps)
	for i in range(len(_indt)):
		if i == 0:
			tf = 0
		else:
			tf =_time[_indt[i - 1][1] - 1]
		ds, Phi2n = evolve_ds(_DAS[_order[i]], _da_dft, _Kn, _ALPHA0[_order[i]], abs(_vals[_order[i]])*_Pe, ncoeffs, _facs, _x, _y, _time[_indt[i][0]:_indt[i][1]],  tf)
		DS.append(ds)
		ncoeffs, odd_coeffs, phi_new, phi_old  = coeff_project(Phi2n, _y)
    
	for i in range(len(DS)):
		if i ==0:
			ds_f = DS[i]
		else:
			ds_f = ds_f.combine_first(DS[i])
	return ds_f


def evolve_off_ds_time(_DAS, _DBS, _indt, _order, _vals, _Kn, _ALPHA0, _Pe, _da_dft, _even_alps, e_facs, _odd_alps, o_facs, _x, _y, _time, _shift=0):
	"""evolves a localized initial condition defined by its 2d Fourier coefficients."""
	DS = []
	PHI_NEW = []
	PHI_OLD = []
	ecoeffs = copy.deepcopy(_even_alps)
	ocoeffs = copy.deepcopy(_odd_alps)
	if _shift == 0:
		_shift = [0 for i in range(len(_time))]
	for i in range(len(_indt)):
		phi_new = _shift[_indt[i][0]]  # only sample first - they are all the same
		if i == 0:
			tf = 0
			phi_old = _np.pi
			ndAs = _xr.dot(e_facs * ecoeffs, _DAS[_order[i]]['A_2r'])
			ndBs = _xr.dot(o_facs * ocoeffs, _DBS[_order[i]]['B_2r'])
			PHI2n_e = _xr.dot(ndAs, _DAS[_order[i]]['phi_2n'], dims='n')
			PHI2n_o = _xr.dot(ndBs, _DBS[_order[i]]['phi_2n'], dims='n')
			Phi2n = PHI2n_e + PHI2n_o
		else:
			tf =_time[_indt[i - 1][1] - 1]
		ecoeffs, ocoeffs, phi_new, phi_old  = coeff_project(Phi2n, _y, phi_old=phi_old, phi_new=phi_new)
		ds, Phi2n = evolve_ds_off(_DAS[_order[i]], _DBS[_order[i]], _da_dft, _Kn, _ALPHA0[_order[i]], abs(_vals[_order[i]])*_Pe, ecoeffs, e_facs, ocoeffs, o_facs,  _x, _y, _time[_indt[i][0]:_indt[i][1]], tf)
		DS.append(copy.deepcopy(ds))
		PHI_NEW.append(phi_new)
		PHI_OLD.append(phi_old)
	for i in range(len(DS)):
		if i == 0:
			ds_f = DS[i]
		else:
			jump = abs(PHI_OLD[i] - PHI_OLD[0])
			dsign = int(_np.sign(PHI_OLD[i] - PHI_OLD[0]))
			diff = abs(2*_y - jump)
			ii = dsign * _np.where(diff == _np.min(diff))[0][0]
			ds_f = ds_f.combine_first(DS[i].roll(y=ii, roll_coords=False))
	return ds_f, PHI_NEW, PHI_OLD


def evolve_ds_rot(_dAs, _da_xrft, _L, _alpha0, _Pe, _alps, _facs, _x, _y, _time,  _tf=0):
    """Constructs the solution to the IVP. Shear flow aligned with y"""
    
    coords = {"time": _time, "y": _y, "x": 2*_x}
    Temp = _xr.DataArray(_np.nan, coords=coords, dims=["time", 'y', 'x'])
    ds = _xr.Dataset({'Theta': Temp})
    _ndAs = _xr.dot(_facs * _alps, _dAs['A_2r'], dims='r')
    for i in range(len(_time)):
        arg = 0.25*_dAs['a_2n'] + (1j)*_alpha0*(2*_np.pi*_L)*_Pe + (2*_np.pi*_L)**2
        _PHI2n = _xr.dot(_ndAs, _dAs['phi_2n'] * _np.exp(- arg*(_time[i] - _tf)), dims='n')
        T0 = _xrft.ifft(_da_xrft * _PHI2n, dim='l', true_phase=True, true_amplitude=True).real
        nT0 = T0.rename({'freq_l':'y'})
        ds['Theta'].data[i, :, :] = nT0.data
    return ds, _PHI2n


def evolve_ds_off_rot(_dAs, _dBs, _da_xrft, _L, _alpha0, _Pe, _a_alps, _afacs, _b_alps, _bfacs, _x, _y, _t, _tf=0):
    """Constructs the solution to the IVP"""
    ## Initialize the array
    coords = {"time": _t, "y": _y, "x": 2 * _x}
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

def evolve_ds_rot_time(_DAS, _indt, _order, _vals, _Ln, _ALPHA0, _Pe, _da_dft, _gauss_alps, _facs, _x, _y, _time):
	"""
	evolves a localized initial condition defined by its 2d Fourier coefficients.
	"""
	DS = []
	ncoeffs = copy.deepcopy(_gauss_alps)
	for i in range(len(_indt)):
		if i == 0:
			tf = 0
		else:
			tf =_time[_indt[i - 1][1] - 1]
		ds, Phi2n = evolve_ds_rot(_DAS[_order[i]], _da_dft, _Ln, _ALPHA0[_order[i]], abs(_vals[_order[i]])*_Pe, ncoeffs, _facs, _x, _y, _time[_indt[i][0]:_indt[i][1]], tf)
		DS.append(ds)
		ncoeffs, odd_coeffs, phi_new, phi_old  = coeff_project(Phi2n, _y, dim='x')
    
	for i in range(len(DS)):
		if i ==0:
			ds_f = DS[i]
		else:
			ds_f = ds_f.combine_first(DS[i])

	return ds_f


def evolve_ds_serial(_dAs, _Kn, _alpha0, _Pe, _gauss_alps, _facs, _X, _Y, _time, _tf=0, _dim='k'):
	"""Constructs the modal solution to the IVP that is localized across the jet."""
	coords = {"t": _time, "y": _Y[:, 0], _dim: _Kn, 'x': _X[0, :]}
	Temp = _xr.DataArray(_np.nan, coords=coords, dims=["t", 'y', 'x', _dim])
	DS = []
	if _dim == 'k':
		phi_arg = {'x':_X[0, :]}
	elif _dim == 'l':
		phi_arg = {'y':_Y[:, 0]}
	for kk in range(len(_Kn)):
		_K = _Kn[kk]
		k_args = {_dim: _K}
		ds = _xr.Dataset({'Theta': Temp})
		if _dim in _gauss_alps.dims:
			ndAs = _xr.dot(_facs * _gauss_alps.sel(**k_args), _dAs['A_2r'].sel(**k_args))
		else:
			ndAs = _xr.dot(_facs * _gauss_alps, _dAs['A_2r'].sel(**k_args))
		exp_arg_e =  0.25*_dAs['a_2n'].sel(**k_args) + (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		exp_arg_o =  0.25*_dBs['a_2n'].sel(**k_args) + (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		for i in range(len(_time)):
			PHI2n_e = _xr.dot(ndAs, _dAs['phi_2n'].sel(**k_args) * _np.exp(-exp_arg_e*(_time[i]-_tf)), dims='n')
			PHI2n_o = _xr.dot(ndBs, _dBs['phi_2n'].sel(**k_args) * _np.exp(-exp_arg_b*(_time[i]-_tf)), dims='n')
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
	coords = {"time": _time, "y": _Y[:, 0], 'x': _X[0, :]}
	da_final = _xr.DataArray(da.real, coords=coords, dims=['time', 'y', 'x'])
	ds_final = _xr.Dataset({'Theta': da_final})
	return ds_final


def evolve_ds_serial_off(_dAs, _dBs, _Kn, _alpha0, _Pe, _a_alps, _afacs, _b_alps, _bfacs, _X, _Y, _time, _tf=0, _dim='k'):
	"""Constructs the modal solution to the IVP that is localized across the jet."""
	coords = {"t": _time, "y": _Y[:, 0], _dim: _Kn, 'x': _X[0, :]}
	Temp = _xr.DataArray(_np.nan, coords=coords, dims=["t", 'y', 'x', _dim])
	DS = []
	if _dim == 'k':
		phi_arg = {'x':_X[0, :]}
	elif _dim == 'l':
		phi_arg = {'y':_Y[:, 0]}
	for kk in range(len(_Kn)):
		_K = _Kn[kk]
		k_args = {_dim: _K}
		ds = _xr.Dataset({'Theta': Temp})
		if _dim in _a_alps.dims:
			ndAs = _xr.dot(_afacs * _a_alps.sel(**k_args), _dAs['A_2r'].sel(**k_args))
			ndBs = _xr.dot(_bfacs * _b_alps.sel(**k_args), _dBs['A_2r'].sel(**k_args))
		else:
			ndAs = _xr.dot(_afacs * _a_alps, _dAs['A_2r'].sel(**k_args))
			ndBs = _xr.dot(_bfacs * _b_alps, _dAs['A_2r'].sel(**k_args))

		exp_arg =  0.25*_dAs['a_2n'].sel(**k_args) + (1j)*_alpha0*(2*_np.pi*_K)*_Pe + (2*_np.pi*_K)**2
		for i in range(len(_time)):
			PHI2n = _xr.dot(ndAs, _dAs['phi_2n'].sel(**k_args) * _np.exp(-exp_arg*(_time[i]-_tf)), dims='n')
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
	coords = {"time": _time, "y": _Y[:, 0], 'x': _X[0, :]}
	da_final = _xr.DataArray(da.real, coords=coords, dims=['time', 'y', 'x'])
	ds_final = _xr.Dataset({'Theta': da_final})
	return ds_final


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









