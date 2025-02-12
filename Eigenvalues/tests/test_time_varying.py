# test the functions inside test_varying.py
import numpy as _np
import xarray as _xr
import pytest
import xrft as _xrft


# this is temporal. 
import sys
sys.path.append('/Users/miguelangeljimenezurias/Hills-Eigenvalues/Eigenvalues/')
#

from time_varying import (
	loc_vals,
	indt_intervals,
	re_sample,
	coeff_project,
)

from utils import _find_nearest


t = _np.linspace(0, 1, 100)
sine_func = _np.sin((2*_np.pi)*t)

@pytest.mark.parametrize(
	"ft, nt, expected",
	[
		# (t, 0, ),
		(sine_func, 0, 0.486),
		(sine_func, 1, 0.249),
		(sine_func, 2, 0.123),
		(sine_func, 5, 0.015),
		# ([0, 0.5, 1], 0, ),
	]
)
def test_re_sample_convergence(ft, nt, expected):
	""" tests convergenve of approximation.
	"""
	nft, *a = re_sample(ft, nt)  # iterate through nt to increase the approx
	delf = _np.linalg.norm(ft - nft, _np.inf)
	assert _np.round(delf, 3) == expected


@pytest.mark.parametrize(
	"ft, nt, expected",
	[
		(sine_func, 0, [-1.0, 0, 1.0]),
		(sine_func, 1, [-1, -0.5, 0, 0.5, 1.0]),
		(sine_func, 2, [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]),
		(sine_func, 5, list(_np.arange(-1, 1.01, 1/(2**5)))),
	]
)

def test_re_sample(ft, nt, expected):
	""" tests coefficients in vals. this approximate a sin fn.
	"""
	nft, vals, ivals = re_sample(ft, nt)  # iterate through nt to increase the approx
	assert vals == expected
	assert max(ivals) + 1 == len(vals)


@pytest.mark.parametrize(
	"ft, nt",
	[
		(sine_func, 0),
		(sine_func, 1),
		(sine_func, 2),
		(sine_func, 5),
	]
)
def test_loc_vals(ft, nt):
	""" tests items so that there are no nans in ivals. This array is initialized as all nans.
	"""
	nft, vals, ivals = re_sample(ft, nt)  # iterate through nt to increase the approx
	ivals = loc_vals(nft, vals)  # this is a double calculation. But calls directly loc_vals.
	assert _np.isnan(ivals).all() == False


@pytest.mark.parametrize(
	"ft, nt",
	[
		(sine_func, 0),
		(_np.cos(4*_np.pi*_np.linspace(0, 1, 10)), 0),
		(_np.cos(4*_np.pi*_np.linspace(0, 1, 50)), 0),
		(_np.cos(4*_np.pi*_np.linspace(0, 1, 9)), 0),
		(sine_func, 1),
		(sine_func, 2),
		(sine_func, 5),
	]
)
def test_indt_intervals(ft, nt):
	""" test that output is consistent.
	"""
	nft, vals, ivals = re_sample(ft, nt)
	indt = indt_intervals(ivals)
	assert indt[-1][-1] == len(nft)
	for i in range(len(indt)-1):
		assert indt[i][-1] == indt[i + 1][0] # so data is continuosly sampled

	for i in range(len(indt)):
		sample = ivals[indt[0][0]:indt[0][1]][0]
		nlist = [abs(kk - sample) for kk in ivals[indt[0][0]:indt[0][1]]]
		assert 0 == max(nlist)  # assert all elements are the same



y = _np.linspace(0, (_np.pi * 2), 1001, endpoint=False)
x = _np.linspace(-_np.pi, _np.pi, 1001, endpoint=False)

X, Y = _np.meshgrid(x, y)


Ld1 = 1/4


@pytest.mark.parametrize(
	"phi_new, phi_old, _y, values",
	[
		(0, 0, y, _np.exp(-0.5*((y - 0)**2)/(0.5*Ld1**2))),
		(0.25*_np.pi, 0.5*_np.pi, y, _np.exp(-0.5*((y - 0.25*_np.pi)**2)/(0.5*Ld1**2))),
		(0.5*_np.pi, 1.5*_np.pi, y, _np.exp(-0.5*((y - _np.pi)**2)/(0.5*Ld1**2))),
		(0.75*_np.pi, _np.pi, Y, _np.exp(-0.5*((y - 0.25*_np.pi)**2/(0.5*Ld1**2)))),
	]
)
def test_coeff_project(phi_new, phi_old, _y, values):

	Phi = _np.exp(-0.5*((_y - phi_old)**2)/(0.5*Ld1**2))

	if len(_np.shape(_y)) == 2:
		dims = ('y', 'x')
		coords = {'y': _y[:, 0], 'x': x}
		ny = _y[:, 0]
		rargs = {'x':0}
	else:
		dims = ('y',)
		coords = {'y': _y}
		ny = _y
		rargs = {}


	da_phi = _xr.DataArray(Phi, dims=dims, coords=coords)

	if len(_np.shape(da_phi))==2:
		da_phi = da_phi.transpose()

	args = {'_phi': da_phi,
			'phi_new': phi_new,
			'phi_old': phi_old,
			'_y': ny}


	even_coeffs_n, odd_coeffs_n, phi_new, phi_old = coeff_project(**args)


	Nr = len(even_coeffs_n.isel(rargs).data)

	gaussian_y_e = sum([even_coeffs_n.isel({**rargs, **{'r':n}}).data * _np.cos(n * ny) for n in range(Nr)])
	gaussian_y_o = sum([odd_coeffs_n.isel({**rargs, **{'r':n-1}}).data * _np.sin(n * ny) for n in range(1, Nr)])

	gaussian_y = gaussian_y_e + gaussian_y_o
	assert _np.max(abs(values - gaussian_y)) < 1e-4



