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
	evolve_ds_rot,
	evolve_ds_off_rot,
	evolve_ds,
	renewing_evolve,
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

		if type(U) in Utypes:
			if type(V) in Utypes:
				print('only the x-component of velocity will be considered')
				V = None
			if type(U) != _xrda_type:
				if type(U) in [int, float]:
					shear = False
				U = _xr.DataArray(_copy.deepcopy(U), coords={'y':y}, dims=['y'])
			even_coeffs, odd_coeffs, *a = coeff_project(U, y)
			if type(tau) == float:
				# rotating shear flow. only flows with even F. series
				Urot = _xr.DataArray(_copy.deepcopy(U).data, coords={'x':x}, dims=['x'])
				xeven_coeffs, *a = coeff_project(Urot, x, dim='x')
				xalphas_m = xeven_coeffs.real.data[:40]
				xKm = range(1, len(xalphas_m))

		if type(V) in Utypes:
			if type(V) != _xrda_type:
				if type(V) in [int, float]:
					shear = False
				V = _xr.DataArray(_copy.deepcopy(V), coords={'x':x}, dims=['x'])
			even_coeffs, odd_coeffs, *a = coeff_project(V, x, dim='x')
		if type(U) not in Utypes and type(V) not in Utypes:
			if U == None and V == None:
				U = _xr.DataArray(0, coords={'y':y}, dims=['y'])
				even_coeffs, odd_coeffs, *a = coeff_project(U, y)
				shear = False
			else:
				raise TypeError("Only float, numpy or xarray types are supported")
			# x-velocity componen non-zero


		# evolve in time. Only steady shear flow for now.
		t = _np.arange(t0, tf + dt, dt)
		X, Y = _np.meshgrid(x, y)
		Ucoords = {'x':x, 'y': y, 'time': t}

		# ===========================
		# TODO:
		# Need to assert that U is defined by an even Fourier series. Otherwise, 
		# extend periodically so that it has an even Fourier series.
		# Finish this, and extend to V vector. 

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
			Pe = 2 * Pe   # def a mutiplicative factor? 
			# =======================
			# TODO:
			# redefine the domain in cross-stream direction to twice its length,
			# but retain spatial resolution.
			# =======================
			odd_flow = True # flag that will restore the dimensional value of Pe


		# truncate velocity Fourier series.
		# so that len(alphas_m) < len(y).
		#  this needs to be a function with proper testing
		if shear:
			maxUsum = max(abs(even_coeffs.real).cumsum().data)
			lll = _np.where(abs(even_coeffs.real).cumsum().data < maxUsum * 0.999)[0][-1]
			if lll < 40:
				lll = 40 # set min range of Fourier coeffs
			alphas_m = even_coeffs.real.data[:lll]
			Km = range(1, len(alphas_m))
		else:
			alphas_m = 0 * _np.arange(10)
			Km = range(1, 10)
			if type(U) == _xrda_type:
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

			if type(U) in Utypes:
				Ck = _xrft.fft(fx, true_phase=True, true_amplitude=True)
				Ck = Ck.rename({'freq_x':'k'}) # label k the wavenumber 
				Kn = Ck['k'].values # wavenumber vals
				phi = gy  # cross-stream component of i.c.
				_axis, _dim = y, 'y'  # pair of cross-stream coord and label
				_uaxis, _udim = x, 'x'  # pair of along-stream coord and label

				if type(tau) == float:  # if renewing flows
					renew_eigs = True
					steady_flow = False

					phiy = _xrft.fft(gy, true_phase=True, true_amplitude=True)
					phiy = phiy.rename({'freq_y':'l'}) # label l the wavenumber 
					Ln = phiy['l'].values # wavenumber vals
			else:
				Ck = _xrft.fft(gy, true_phase=True, true_amplitude=True)
				Ck = Ck.rename({'freq_y':'l'}) # label k the wavenumber 
				Kn = Ck['l'].values # wavenumber vals
				phi = fx  # cross-stream component of i.c.
				_axis, _dim = x, 'x'  # pair of cross-stream coord and label 
				_uaxis, _udim = y, 'y'  # pair of along-stream coord and label

			even_coeffs, odd_coeffs, *a = coeff_project(phi, _axis, dim=_dim)

			acoords = {'r':range(len(even_coeffs.data))}
			bcoords = {'r':range(len(odd_coeffs.data)-1)}

			a_alps = _xr.DataArray(even_coeffs.data, coords=acoords, dims='r')
			b_alps = _xr.DataArray(odd_coeffs.data[:-1], coords=bcoords, dims='r')

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
			# steady_flow = False


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

			U_f = alphas_m[0] + sum([alphas_m[n] * _np.cos(Y*n) for n in Km])
			da = _xr.DataArray(U_f, coords=Ucoords, dims=['time', 'y', 'x'])


			if odd_eigs:
				ds_Bs = _eigfns.phi_odd(**{**args, **{'_y': _axis / 2, "opt": True,"reflect": True}})
				if _udim == 'y':
					ds_Bs = ds_Bs.rename_dims({'k':'l', 'y':'x'}).rename_vars({'k':'l', 'y':'x'})

			ds_As = _eigfns.phi_even(**{**args, **{'_y': _axis / 2, "opt": True,"reflect": True}})

			if _udim == 'y':
				ds_As = ds_As.rename_dims({'k':'l', 'y':'x'}).rename_vars({'k':'l', 'y':'x'})

				args.pop('_K')
				args = {**args, '_L': Kn}

			eargs = {**args, **{'_y': _axis, "_x": _uaxis, "_alpha0": alphas_m[0], "_t": t}}

			if ic_flag:
				nargs = {
					"_dAs": ds_As,
					"_da_xrft": Ck,
					"_a_alps": a_alps,
					"_afacs": afacs,
				}
				eargs = {**eargs, **nargs}
				if odd_eigs:
					add_args={
						"_dBs": ds_Bs,
						"_b_alps": b_alps,
						"_bfacs": bfacs,
					}
					eargs = {**eargs, **add_args}

					if _udim == 'x':
						time_evolve = evolve_ds_off
					elif _udim == 'y':
						time_evolve = evolve_ds_off_rot
				else:
					if _udim == 'x':
						time_evolve = evolve_ds
					elif _udim == 'y':
						time_evolve = evolve_ds_rot

		if time_osc:

			iargs = {
				'_vals': vals,
				'_alpha0': alphas_m[0],
				'_y': y / 2,
			}

			DAS, DBS, ALPHA0, vals = spectra_list(**{**args, **iargs})

			eargs = {**args, **{"_x": x, '_y': y, "_t": t}}  # _axis has to be factored by 2

			if ic_flag:
				nargs = {
					'_DAS': DAS, 
					'_DBS': DBS, 
					'_indt': indt, 
					'_order': order,
					'_vals': vals,  
					'_ALPHA0': ALPHA0, 
					'_da_dft': Ck,
					'_even_alps': a_alps,
					'e_facs': afacs,
					'_odd_alps': b_alps,
					'o_facs': bfacs
				}
				eargs = {**eargs, **nargs}
			if phase_shift:
				eargs = {**eargs, '_shift': shifts}

			time_evolve = evolve_off_ds_time

		if renew_eigs:
			# renewing shear flow, for now there is no phase shift in here
			# TODO: add phase shift
			ds_As = _eigfns.phi_even(**{**args, **{'_y': y / 2, "opt": True,"reflect": True}})
			ds_Bs = _eigfns.phi_odd(**{**args, **{'_y': y / 2, "opt": True,"reflect": True}})

			args.pop('_K')
			args.pop('_betas_m')
			args.pop('_Km')
			args = {**args, '_K': Ln, '_betas_m': xalphas_m[1:], '_Km':xKm}

			ds_As_rot = _eigfns.phi_even(**{**args, **{'_y':x / 2, "opt": True,"reflect": True}})
			ds_As_rot = ds_As_rot.rename_dims({'k':'l', 'y':'x'}).rename_vars({'k':'l', 'y':'x'})

			ds_Bs_rot = _eigfns.phi_odd(**{**args, **{'_y':x / 2, "opt": True,"reflect": True}})
			ds_Bs_rot = ds_Bs_rot.rename_dims({'k':'l', 'y':'x'}).rename_vars({'k':'l', 'y':'x'})

			args.pop('_K')

			eargs = {
				'_dAs': ds_As,
				'_dBs': ds_Bs,
				'_dAs_rot': ds_As_rot,
				'_dBs_rot': ds_Bs_rot,
				'_alpha0': alphas_m[0],
				'_Theta0': IC['Theta0'],
				'_tau': tau,
				"_x": x,
				'_y': y,
				"_t": t
			}

			eargs = {**eargs, **args} 


			time_evolve = renewing_evolve



		for key in ['_N', '_betas_m', '_Km']:
			if key in eargs.keys():
				eargs.pop(key)


		ds, *a = time_evolve(**eargs)

		return ds


