"""
Module that evolves an initial condition contains class of stirring velocity fields.
"""

import numpy as _np
from Hills_eigenfunctions import (
	complement_dot,
	spectra_list,
	A_coefficients)
from time_varying import (
	coeff_project,
	evolve_ds_off,
	evolve_ds,
	re_sample,
	indt_intervals,
	get_order,
	phase_generator,
	evolve_off_ds_time
)
from Hills_eigenfunctions import eigenfunctions as _eigfns
import copy as _copy
import xarray as _xr
import xrft as _xrft

_xrda_type = _xr.core.dataarray.DataArray
_xrds_type = _xr.core.dataset.Dataset


class planarflows:
	# """
	# defines several classes of two dimensional flow fields
	# """
	__slots__ = (
		"U",
		"V",
		"BCs",
		"IC",
		"kappa",
		"Amp",
		"Phase",
		"tau",
		"Q",
		"delta",
		"t0",
		"tf",
		"dt",
	)

	@classmethod
	def shear_flow(
		cls,
		U = None,
		V = None,
		x=None,
		y=None,
		BCs = None,
		IC = None,
		Pe = None,
		kappa = None,
		Amp=None,
		Phase=None,
		tau=None,
		Q=None,
		delta=None,
		t0=None,
		tf=None,
		dt=None,
		):
		"""
		Calculates the (analytical) solution to the advection diffusion equation in
		the case the advection is, at any give time, done by a plane parallel shear
		flow. Time dependence can be any prescribe function, but currently only
		time-periodic function is taken. Only one component of velocity is needed, 
		and if both are given by default only U is taken.

		Parameters:
		-----------
			U: 1d array-like. xarray.dataarray.
				Defines the pure shear flow as U(y). Only the cross-stream domain is defined.
			V: 1d array-like. xarray.dataarray.
				Defines the pure shear flow as V(x). Only the cross-stream domain is defined.
			x: 1d array-like.
				Defines the domain in the along-stream direction.
			y: 1d array-like.
				Defines the domain in the cross-stream direction.
			BCs: int
				Cross-stream boundary conditions to be satisfied by analytical solution.
				Periodic or Neumann.
			IC: None, or 2d xarray.dataarray.
				Initial condition. The domain must be consistent with that in U(y). When
				`None`, 
			Pe: float.
				Peclet number.
			kappa: float, list
				Background diffusivity. When float, only a single tracer is evolved. It 
				a list of N (different) values, then solve for N tracers.
			Amp: None, int or 1d-array like.
				Time dependency of the shear flow's amplitude. Sign can change. If `None`
				then default is unit value 1.
			Phase: None, True.
				Phase of shear flow. If `None`, then default is zero. If `True`, generates
				random phase shift.
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
		# initialize some flags
		steady_flow = True
		odd_flow = False
		time_osc = False
		phase_shift = False
		odd_eigs = False  # 
		renew_eigs = False
		ic_flag = True
		shear = True  # False only when flow is constant

		Utypes = [_np.ndarray, _xrda_type, float, int]

		if U == None and V == None:
			print('No shear flow provided')
			U = _xr.DataArray(0, coords={'y':y}, dims=['y'])

		if U != None:
			if V != None:
				print('only the x-component of velocity will be considered')
				V = None
			if type(U) in Utypes:
				if type(U) != _xrda_type:
					if type(U) in [int, float]:
						shear = False
					U = _xr.DataArray(_copy.deepcopy(U), coords={'y':y}, dims=['y'])
			else:
				raise TypeError("Only float, numpy or xarray types are supported")
			# x-velocity componen non-zero
			even_coeffs, odd_coeffs, *a = coeff_project(U, y)
		elif U == None:
			if type(V) in Utypes:
				if type(V) != _xrda_type:
					if type(V) in [int, float]:
						shear = False
					V = _xr.DataArray(_copy.deepcopy(V), coords={'x':x}, dims=['x'])
			else:
				raise TypeError("Only float, numpy or xarray types are supported")


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

			even_coeffs, *a = coeff_project(U, y)  # Fourier coeffs for shear flow

			# effective Peclet number due to domain halving
			Pe = 2 * Pe 
			odd_flow = True # flag that will restore the dimensional value of Pe


		# truncate velocity Fourier series.
		# so that len(alphas_m) < len(y).
		#  this needs to be a function with proper testing
		if shear:
			maxUsum = max(abs(even_coeffs.real).cumsum().data)
			lll = _np.where(abs(even_coeffs.real).cumsum().data < maxUsum * 0.99)[0]
			alphas_m = even_coeffs.real.data[:lll[-1]]
			Km = range(1, len(alphas_m))
		else:
			alphas_m = 0 * _np.arange(10)
			Km = range(1, 10)
			if type(U) != None:
				alphas_m[0] = U.mean().values
			else:
				alphas_m[0] = V.mean().values

		if type(Amp) == _np.ndarray:  # time-varying amplitude
			# if shear: # only if shear is present
			time_osc = True
			steady_flow = False
			
			nft, vals, ivals = re_sample(Amp, nt=2)
			indt = indt_intervals(ivals)
			order = get_order(nft, indt, vals)

			if phase_shift:
				shifts = phase_generator(nft)

		#  initial condition
		if type(IC) == _xrds_type:  # one ic is given
			fx = IC['f(x)'].isel(y=0) # along-stream component of ic.
			gy = IC['g(y)'].isel(x=0) # cross-stream component of ic.

			Ck = _xrft.fft(fx, true_phase=True, true_amplitude=True)
			Ck = Ck.rename({'freq_x':'k'}) # label k the wavenumber 

			Kn = Ck['k'].values # wavenumber vals

			even_coeffs, odd_coeffs, *a = coeff_project(gy, y)

			acoords = {'r':range(len(even_coeffs.data))}
			bcoords = {'r':range(len(odd_coeffs.data)-1)}

			a_alps_y = _xr.DataArray(even_coeffs.data, coords=acoords, dims='r')
			b_alps_y = _xr.DataArray(odd_coeffs.data[:-1], coords=bcoords, dims='r')

			afacs = _np.ones(_np.shape(range(len(even_coeffs))))
			afacs[0] = 2
			bfacs = _np.ones(_np.shape(range(len(odd_coeffs)-1)))

			afacs = _xr.DataArray(afacs, coords=acoords, dims='r')
			bfacs = _xr.DataArray(bfacs, coords=bcoords, dims='r')

			# check in i.c. is off centered.
			# if not, only even eigenfunctions are needed
			if _np.max(abs(odd_coeffs)) > 1e-10:
				odd_eigs = True
		else:
			# No initial condition given
			ic_flag = False


		if type(Q) == _xrda_type: # only one forcing function
			Qx = Q['Qx'] # x-separable forcing fn
			Qy = Q['Qy'] # y-separable forcing fn

			Qx_k = _xrft.fft(Qx, true_phase=True, true_amplitude=True)
			Qx_k = Qx_k.rename({'freq_x':'k'}) # label k the wavenumber 

			Kn = Qx_k['k'].values # wavenumber vals. These should be the same to i.c.

			steady_flow = False


		if type(tau) == float:  # if renewing flows
			renew_eigs = True
			steady_flow = False


		# define universal parameters

		args = {
			'_K': Kn,
			'_Pe': Pe,
			"_N": 40,
			'_betas_m': alphas_m[1:],
			'_Km': Km,
		}

		# ========================
		# Construct eigenfunctions
		# ========================

		# Depending on a series of flag or parameters

		if steady_flow:

			args = {**args, '_y': y/2}
			ds_As = _eigfns.phi_even(**{**args, **{"opt": True,"reflect": True}})


			if odd_eigs:
				ds_Bs = _eigfns.phi_odd(**{**args, **{"opt": True,"reflect": True}})

			eargs = {**args, **{"_x": x, "_alpha0": alphas_m[0], "_t": t}}

			if ic_flag:
				nargs = {
					"_dAs": ds_As,
					"_da_xrft": Ck,
					"_a_alps": a_alps_y,
					"_afacs": afacs,
				}
				eargs = {**eargs, **nargs}
				if odd_eigs:
					add_args={
						"_dBs": ds_Bs,
						"_b_alps": b_alps_y,
						"_bfacs": bfacs,
					}
					eargs = {**eargs, **add_args}

	
					time_evolve = evolve_ds_off
				else:
					time_evolve = evolve_ds

		if time_osc:

			nargs = {
				'_vals': vals,
				'_alpha0': alphas_m[0],
				'_y': y/2,
			}

			DAS, DBS, ALPHA0, vals = spectra_list(**{**args, **nargs})

			eargs = {**args, **{"_x": x, '_y': y/2, "_t": t}}

			if ic_flag:
				nargs = {
					'_DAS': DAS, 
					'_DBS': DBS, 
					'_indt': indt, 
					'_order': order,
					'_vals': vals,  
					'_ALPHA0': ALPHA0, 
					'_da_dft': Ck,
					'_even_alps': a_alps_y,
					'e_facs': afacs,
					'_odd_alps': b_alps_y,
					'o_facs': bfacs
				}
				eargs = {**eargs, **nargs}
			if phase_shift:
				eargs = {**eargs, '_shift': shifts}

			time_evolve = evolve_off_ds_time

		for key in ['_N', '_betas_m', '_Km']:
			if key in eargs.keys():
				eargs.pop(key)

		ds, *a = time_evolve(**eargs)

		# U_f = alphas_m[0]+sum([alphas_m[n] * _np.cos(y*n) for n in Km])
		# ds['U'] = U_f

		return ds


