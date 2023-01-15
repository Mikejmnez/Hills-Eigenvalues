"""
Module that evolves an initial condition contains class of stirring velocity fields.
"""

import numpy as _np
from Hills_eigenfunctions import (
	complement_dot,
	A_coefficients)
from time_varying import _coeff_project
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
		t0=None,
		tf=None,
		Amp=None,
		Phase=None,
		tau=None,
		Q=None,
		delta=None,
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
		t0: float.
			Initial time. If `None`, then default is zero.
		tf: float.
			Final time. If `None`, then default is 1 diffusive timescale unit.
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
	"""

	y = U.y.data. # extract coordinate - 0 to 2\pi
	even_coeffs, odd_coeffs, *a = _coeff_project(U, y)

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

		alphas_m, *a = _coeff_project(U, y)  # Fourier coeffs for shear flow

		# effective Peclet number due to domain halving
		Pe = 2 * Pe 
		odd_flow = True # flag that will restore the dimensional value of Pe
	else:
		odd_flow = False

	#  initial condition
	if type(IC) == _xrda_type:  # one ic is given
		fx = IC['f(x)'] # along-stream component of ic.
		g = IC['g(y)'] # cross-stream component of ic.

		Ck = _xrft.fft(fx, true_phase=True, true_amplitude=True)
		Ck = Ck.rename({'freq_x':'k'}) # label k the wavenumber 

		Kn = Ck['k'].values # wavenumber vals

		da_y = _xrft.fft(g, true_phase=True, true_amplitude=True)
		da_y = da_y.rename({'freq_y':'l'})

		even_coeffs, odd_coeffs, *a = _coeff_project(da_y, y / 2)

		acoords = {'r':range(len(even_coeffs.data))}
		bcoords = {'r':range(len(odd_coeffs.data)-1)}

		a_alps_y = _xr.DataArray(even_coeffs.data, coords=acoords, dims='r')
		b_alps_y = _xr.DataArray(odd_coeffs.data[1:], coords=bcoords, dims='r')





	pass



