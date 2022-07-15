# test the functions inside test_varying.py
import numpy as _np
import xarray as _xr
import pytest

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


t = _np.linspace(0, 1, 100)

@pytest.mark.parametrize(
	"ft, nt, expected",
	[
		# (t, 0, ),
		(_np.sin((2*_np.pi)*t), 0, 0.486),
		(_np.sin((2*_np.pi)*t), 1, 0.249),
		(_np.sin((2*_np.pi)*t), 2, 0.123),
		(_np.sin((2*_np.pi)*t), 5, 0.015),
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
		(_np.sin((2*_np.pi)*t), 0, [-1.0, 0, 1.0]),
		(_np.sin((2*_np.pi)*t), 1, [-1, -0.5, 0, 0.5, 1.0]),
		(_np.sin((2*_np.pi)*t), 2, [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]),
		(_np.sin((2*_np.pi)*t), 5, list(_np.arange(-1, 1.01, 1/(2**5)))),
	]
)

def test_re_sample(ft, nt, expected):
	""" tests coefficients in vals. this approximate a sin fn.
	"""
	nft, vals, ivals = re_sample(ft, nt)  # iterate through nt to increase the approx
	assert vals == expected
	assert max(ivals) + 1 == len(vals)





