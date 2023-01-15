"""
Module that evolves an initial condition contains class of stirring velocity fields.
"""

import numpy as _np
from Hills_eigenfunctions import (
	complement_dot,
	A_coefficients,
	)
from Hills_eigenfunctions import eigenfunctions as _eigfns
import copy as _copy
import xarray as _xr
import xrft as _xrft

class planarflows:
	"""
	defines several classes of two dimensional flow fields
	"""
	def __init__(self)


	@classmethod
	def shear_flow(
		cls,
		U = None,
		bcs = None,
		ic = None,
		kappa = None,
		t0=None,
		tf=None,
		amp=None,
		phase=None,
		rotation=None,
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
		bcs: int
			Cross-stream boundary conditions to be satisfied by analytical solution.
			Periodic or Neumann.
		ic: None, or 2d xarray.dataarray.
			Initial condition. The domain must be consistent with that in U(y). When
			`None`, 
		kappa: float, list
			Background diffusivity. When float, only a single tracer is evolved. It 
			a list of N (different) values, then solve for N tracers.
		t0: float.
			Initial time. If `None`, then default is zero.
		tf: float.
			Final time. If `None`, then default is 1 diffusive timescale unit.
		amp: None, int or 1d-array like.
			Time dependency of the shear flow's amplitude. Sign can change. If `None`
			then default is unit value 1.
		phase: None, int or 1d-array like.
			Phase of shear flow. If None, then default is zero.
		rotation: None, or 1d-array.
			If None, plane parallel shear flow is always unidirectional. Its amplitude
			and phase can vary. The renewing (renovating) flow can is defined by an
			array of ones and zeros. A zero implies no rotation, and a one implies a
			rotation of the velocity field by 90 degrees. The variable `amp` determines
			the sign of the velocity field.
		Q: None, or 2d xarray.dataarray.
			Defines the spatial forcing, separable along its independent coordinates. 
			When `None`, there is no forcing. It can also be a collection of forcing
			functions, each separable along its spatial coordinates. The domain must
			match that of the initial condition when both given, as well as that of 
			the cross-stream domain.
		delta: None, float or 1d array-like.
			Frequency of forcing. When `None` there is no forcing (even when Q is 
			given). If 1d array-like, len(delta) ==. number of forcing functions.
	"""

	
	
	pass



