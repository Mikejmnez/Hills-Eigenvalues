"""
Module that evolves an initial condition contains class of stirring velocity fields.
"""

import numpy as _np
from Hills_eigenfunctions import (
	complement_dot,
	_A_coefficients)
from time_varying import _coeff_project, evolve_ds_off, evolve_ds

from Hills_eigenfunctions import eigenfunctions as _eigfns
import copy as _copy
import xarray as _xr
import xrft as _xrft

_xrda_type = _xr.core.dataarray.DataArray

class planarflows:
	"""
	defines several classes of two dimensional flow fields
	"""
	def __init__(self)


	@classmethod
	def shear_flow(
		cls,
		U = None,
		BCs = None,
		IC = None,
		kappa = None,
		Amp=None,
		Phase=None,
		tau=None,
		Q=None,
		delta=None,
		t0=None,
		tf=None,
		dt=None,
		)
	"""
	Calculates the (analytical) solution to the advection diffusion equation in the
	case the advection is, at any give time, done by a plane parallel shear flow.

	Parameters:
	-----------
		U: 1d array-like. xarray.dataarray.
			Defines the shear flow as U(y). Only the cross-stream domain is defined.
		BCs: int
			Cross-stream boundary conditions to be satisfied by analytical solution.
			Periodic or Neumann.
		IC: None, or 2d xarray.dataarray.
			Initial condition. The domain must be consistent with that in U(y). When
			`None`, 
		kappa: float, list
			Background diffusivity. When float, only a single tracer is evolved. It 
			a list of N (different) values, then solve for N tracers.
		Amp: None, int or 1d-array like.
			Time dependency of the shear flow's amplitude. Sign can change. If `None`
			then default is unit value 1.
		Phase: None, int or 1d-array like.
			Phase of shear flow. If None, then default is zero.
		tau: `None`, or float.
			If `None`, the plane parallel shear flow is always unidirectional. Its
			amplitude and phase can still vary. If `float`, 1/tau defines the
			frequency at which the shear flow rotates. The variable `amp` determines
			the sign of the velocity field.
		Q: None, or 2d xarray.dataarray.
			Defines the spatial forcing, separable along its independent coordinates. 
			When `None`, there is no forcing. It can also be a collection of forcing
			functions, each separable along its spatial coordinates. The domain must
			match that of the initial condition when both given, as well as that of 
			the cross-stream domain.
		delta: None, float or 1d array-like.
			Frequency of forcing. When `None` there is no forcing (even when Q is 
			given). If 1d array-like, len(delta) == number of forcing functions.
		t0: float.
			Initial time. If `None`, then default is zero.
		tf: float.
			Final time. If `None`, then default is 1 diffusive timescale unit.
		dt: float.
			time discretization. Must allow sufficient time resolution to resolve
			time-dependence of shear flow.
	"""

	y = U.y.data # extract coordinate - 0 to 2\pi
	x = U.x.data
	even_coeffs, odd_coeffs, *a = _coeff_project(U, y)

	# evolve in time. Only steady shear flow for now.
	t = _np.arange(t0, tf + dt, dt)

	# need to assert that U is defined by an even Fourier series. Otherwise, 
	# need to extend periodically so that it has an even Fourier series.

	if _np.max(abs(odd_coeffs)) > 1e-7:  # I can make this smaller
		Uold = _copy.deepcopy(U)
		U = _xr.DataArray(Uold, dims=('y',), coords={'y': 2*y})
		# make periodic extension
		U[U.y > 2*_np.pi] = 1 - U[U.y > 2*_np.pi]

		# velocity should only have even Fourier series
		# max amplitude of shear flow is retained
		# but domain is effectively halved. 

		even_coeffs, *a = _coeff_project(U, y)  # Fourier coeffs for shear flow

		# effective Peclet number due to domain halving
		Pe = 2 * Pe 
		odd_flow = True # flag that will restore the dimensional value of Pe
	else:
		odd_flow = False


	# truncate velocity Fourier series.
	# so that len(alphas_m) < len(y).
	#  this needs to be a function with proper testing
	maxUsum = max(abs(even_coeffs.real).cumsum().data)
	lll = _np.where(abs(even_coeffs.real).cumsum().data < maxUsum * 0.99)[0]
	alphas_m = even_coeffs.real.data[:lll[-1]]
	Km = range(1, len(alphas_m))


	#  initial condition
	if type(IC) == _xrda_type:  # one ic is given
		fx = IC['f(x)'] # along-stream component of ic.
		gy = IC['g(y)'] # cross-stream component of ic.

		Ck = _xrft.fft(fx, true_phase=True, true_amplitude=True)
		Ck = Ck.rename({'freq_x':'k'}) # label k the wavenumber 

		Kn = Ck['k'].values # wavenumber vals

		da_y = _xrft.fft(gy, true_phase=True, true_amplitude=True)
		da_y = da_y.rename({'freq_y':'l'})

		even_coeffs, odd_coeffs, *a = _coeff_project(da_y, y / 2)

		acoords = {'r':range(len(even_coeffs.data))}
		bcoords = {'r':range(len(odd_coeffs.data)-1)}

		a_alps_y = _xr.DataArray(even_coeffs.data, coords=acoords, dims='r')
		b_alps_y = _xr.DataArray(odd_coeffs.data[1:], coords=bcoords, dims='r')


		# check in i.c. is off centered.
		# if not, only even eigenfunctions are needed
		if _np.max(abs(odd_coeffs)) > 1e-10:
			odd_eigs = True
		else:
			odd_eigs = False

	if type(Q) == _xrda_type: # only one forcing function
		Qx = Q['Qx'] # x-separable forcing fn
		Qy = Q['Qy'] # y-separable forcing fn

		Qx_k = _xrft.fft(Qx, true_phase=True, true_amplitude=True)
		Qx_k = Qx_k.rename({'freq_x':'k'}) # label k the wavenumber 

		Kn = Qx_k['k'].values # wavenumber vals. These should be the same to i.c.


	afacs = np.ones(np.shape(range(len(even_coeffs))))
	afacs[0] = 2
	bfacs = np.ones(np.shape(range(len(odd_coeffs)-1)))



	if type(tau) == float:
			rot_eigs = True
		else:
			rot_eigs = False		


	# Construct eigenfunctions
	args = {
		'K': Kn,
		'Pe': Pe,
		"N": 40,
		'_betas_m': betas_m,
		'Kj': Km,
		'symmetry': 'even',
		"opt": True,
		"reflect": True,
	}

	ds_As = _A_coefficients(**args)
	ds_As = _eigfns.phi_even(Kn, Pe, y / 2, 50, betas_m, Km, dAs = ds_As)


	evolve_args = {
		"_dAs": ds_As,
		"_da_xrft": Ck,
		"_K": Kn,
		"_alpha0": alphas_m[0],
		"_Pe": Pe,
		"_gauss_alps": a_alps_y,
		"_facs": afacs,
		"_x": x,
		"_y": y,
		"_time": t,
	}

	if odd_eigs:
		args = {
		"K": Kn,
		"Pe": Pe,
		'N': 40,
		'_betas_m': betas_m,
		'Kj': Km,
		'symmetry': 'odd',
		"opt": True,
		"reflect": True,
		}

		ds_Bs = _A_coefficients(**args)
		ds_Bs = _eigfns.phi_odd(Kn, Pe, yt, 40, betas_m, Km, dBs = ds_Bs)


		evolve_args = {
		"_dAs": ds_As,
		"_dBs": ds_Bs,
		"_da_xrft": Ck,
		"_K": Kn,
		"_alpha0": alphas_m[0],
		"_Pe": Pe,
		"_a_alps": a_alps_y,
		"_afacs": afacs,
		"_b_alps": b_alps_y,
		"_bfacs": bfacs,
		"_x": x,
		"_y": y,
		"_time": t,
	}
	

		time_evolve = evolve_ds_off
	else:
		time_evolve = evolve_ds


	ds = time_evolve(**args)

	return ds


