"""
Contains useful information about exceptional points of some shear flows of interest
"""
import numpy as _np


def cosine_shear(n=1):
	"""
	returns the location of EPs associated with a shear flow of multiplicity n.
	"""

	if n==1:
		# EPs up to q = 4000 (a_2n = )
		qs_a = [2.938438, 32.942942, 95.613113, 190.950950, 318.959459, 479.636636,
			  	672.983483, 898.998498, 1157.684684, 1449.039039, 1773.061561, 2129.755255,
			  	2519.69, 2941.22, 3395.896, 3883.9679
		]
		# 
		# odd eigenvalues too (begining with n=14,15)
		qs_b = [1024.5409, 1299.4, 1947.59759, 2370.17,

		]

	else:
		qs = 0

	return qs


def norm_min(fn, ki=0, kf=-1):
	"""
	Finds the location of minima from the array of (row-)squared sums.
	Parameters:
	-----------
		fn: xarray.DataArray.
			must have `r` and `k` dimensions.
		ki: float
			minimum wavenumber to evaluate
		kf: float
			maximum wavenumber to evaluate. The rannge slice(ki, kf) must be within the dataarray.
	
	Returns:
	--------
		n0: int
			location with the minimum (in the sum) of fn in k-index.
	"""
	if ki == 0 and kf == -1:
		_normr = abs(fn.isel(k=slice(ki, kf)).real).sum(dim='r')
		_normi = abs(fn.isel(k=slice(ki, kf)).imag).sum(dim='r')
	else:
		_normr = abs(fn.sel(k=slice(ki, kf)).real).sum(dim='r')
		_normi = abs(fn.sel(k=slice(ki, kf)).imag).sum(dim='r')

	_norm = _normr + _normi
	n0 = _np.where(_norm.data == _np.min(_norm.data))[0][0]

	return n0







