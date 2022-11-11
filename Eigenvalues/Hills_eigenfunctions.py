"""
defines a class of eigenfunctions for Hill's equation
"""

import numpy as _np
import copy
from eig_system import matrix_system, eig_pairs
import copy as _copy
import xarray as _xr


class eigenfunctions:

    def __init__(self, q, K, Pe, y, N, coeffs, Kj, symmetry, case, opt):
        self._q = q
        self._K = K
        self._Pe = Pe
        self._y = y
        self._N = N
        self._coeffs = coeffs
        self._Kj = Kj
        self._symmetry = symmetry
        self._case = case
        self._opt = opt


    @classmethod
    def phi_even(
        cls,
        K,
        Pe,
        y,
        N,
        coeffs,
        Kj,
        case='None',
        opt=False,
        dAs=None,
    ):
        """Even eigenfunctions that solve Hill's equation associated with
        the case where Neumann BC and coeffs are associated with a purely
        cosine Fourier series. Simplest case is that of Mathieu's ce_2n
        associates with coeffs = 1. when K =1, otherwise coeffs =0. Th
        """
        if dAs is None:
            dAs = A_coefficients(K, Pe, N, coeffs, Kj, 'even', opt)
        # initialize a dataarray with right dimensions
        N = len(dAs.n)  # update the size of the array
        cos_coords = {'r':range(N), 'y':y}
        dcos = _xr.DataArray((1j) * _np.nan, coords=cos_coords, dims=['r', 'y'])

        for r in range(N):  # populate with the base
            dcos.sel(r = r)[:] =  _np.cos(2 * r * y)

        dphi = _xr.dot(dcos, dAs['A_2r'], dims='r').transpose('n', 'k', 'y')  # dataarray

        dAs['phi_2n'] = dphi

        return dAs


    @classmethod
    def phi_odd(
        cls,
        K,
        Pe,
        y,
        N,
        coeffs,
        Kj,
        case='None',
        opt=False,
        dBs=None,
    ):
        """Even eigenfunctions that solve Hill's equation associated with
        the case where Neumann BC and coeffs are associated with a purely
        cosine Fourier series. Simplest case is that of Mathieu's ce_2n
        associates with coeffs = 1. when K =1, otherwise coeffs =0. Th
        """
        if dBs is None:
            dBs = A_coefficients(K, Pe, N, coeffs, Kj, 'odd', opt)
        # initialize a dataarray with right dimensions
        _range = dBs.n.data  # update the size of the array
        sin_coords = {'r':_range, 'y':y}
        dsin = _xr.DataArray((1j) * _np.nan, coords=sin_coords, dims=['r', 'y'])

        for r in _range:  # populate with the base
            dsin.sel(r = r)[:] =  _np.sin(2 * (r + 1) * y)

        dphi = _xr.dot(dsin, dBs['B_2r'], dims='r').transpose('n', 'k', 'y')  # dataarray

        dBs['phi_2n'] = dphi

        return dBs


    @classmethod
    def phi_even_old(
        cls,
        q,
        x,
        N,
        coeffs,
        K,
        symmetry='None',
        case='None',
        As=None,
        Ncut=0,
    ):
        """Even eigenfunctions that solve Hill's equation associated with
        the case where Neumann BC and coeffs are associated with a purely
        cosine Fourier series. Simplest case is that of Mathieu's ce_2n
        associates with coeffs = 1. when K =1, otherwise coeffs =0. Th
        """
        if As is None:
            As = A_coefficients_old(q, N, coeffs, K, symmetry)
        vals = {}
        if Ncut != 0:
            N = Ncut
        LEN = range(N)
        for n in LEN:
            terms = [_np.cos((2*k)*x)*(As['A'+str(2*n)][0, k]) for k in LEN]
            vals.update({'phi' + str(2 * n): + _np.sum(terms, axis=0)})
            vals.update({'phi' + str(2 * n):
                         vals['phi' + str(2 * n)][_np.newaxis, :]})
            vals.update({'a' + str(2 * n): As['a' + str(2 * n)]})
        for i in range(1, len(q)):
            for n in LEN:
                terms = [_np.cos((2*k)*x)*(As['A'+str(2*n)][i, k]) for k in LEN]
                phi = _np.sum(terms, axis=0)
                phi = phi[_np.newaxis, :]
                phi = _np.append(vals['phi' + str(2 * n)], phi, axis=0)
                vals.update({'phi' + str(2 * n): phi})
        return vals



    @classmethod
    def phi_odd_old(
        cls,
        q,
        x,
        N,
        coeffs,
        K,
        symmetry='Odd',
        case='None',
        Bs=None,
        Ncut=0,
    ):
        """Odd eigenfunctions that solve Hill's equation associated with
        the case where Dirichlet BCs and coeffs are associated with a pure
        sine Fourier series. Simplest case is that of Mathieu's se_2n+2
        associates with coeffs = 1. when K = 1, otherwise coeffs =0. Th
        """
        if Bs is None:
            Bs = A_coefficients_old(q, N, coeffs, K, symmetry)
        vals = {}
        if Ncut != 0:
            N = Ncut
        LEN = range(N-1)
        for n in LEN:
            terms = [_np.sin(2*(k+1)*x)*(Bs['B'+str(2*(n+1))][0, k]) for k in LEN]
            vals.update({'phi' + str(2 * (n + 1)): + _np.sum(terms, axis=0)})
            vals.update({'phi' + str(2 * (n + 1)):
                         vals['phi' + str(2 * (n + 1))][_np.newaxis, :]})
            vals.update({'b' + str(2 * (n + 1)): Bs['b' + str(2 * (n + 1))]})
        for i in range(1, len(q)):
            for n in LEN:
                terms = [_np.sin(2*(k+1)*x)*(Bs['B'+str(2*(n+1))][i, k]) for k in LEN]
                phi = _np.sum(terms, axis=0)
                phi = phi[_np.newaxis, :]
                phi = _np.append(vals['phi' + str(2 * (n + 1))], phi, axis=0)
                vals.update({'phi' + str(2 * (n + 1)): phi})
        return vals


def A_coefficients(K, Pe, N, coeffs, Kj, symmetry='even', opt=False, reflect=True):
    """ Returns the (sorted) eigenvalues and orthonormal eigenvectors of
    Hill's equation.

    Input:
        K: 1d-array, all along-jet (x) wavenumbers.
        Pe: Peclet number.
        N: size of (square) Matrix
        coeffs: 1d-array, containing Fourier coefficients associated with
            periodic coefficient in Hill's equation.
        Kj: range(1, M, d) defining the jet. M is highest harmonic in coeffs. d is either 1 or
            two. if d=1, Fourier sum is : cos(y)+cos(2*y)+... if d=2, the sum
            : cos(y)+cos(3*y)+cos(5*y)... 
        symmetry: `even` (default). Implies solution has even periodic solutions. If `odd`, 
            then periodic coeff has a sine Fourier series.
        opt: `False` (default): If `True`, optimizes the calculation of the eigenvalues. This is achieved
            by truncating the matrix differently at each value of q (while making matrix is diagonally dominant)
            and thus performing eigenvalue calculation on a submatrix, supplementing the rest with theoretical 
            limiting values a_2n -> 4n^2 and A_{2n}^{(2n)} --> 1, zero otherwise.

    Output:
        xarray.Dataset: 'a_{2n}(q)' (dims: n, k) and 'A^{2n}_{2r}(q)' with dims
            (n, k, r).
    """
    coeffs = _np.array(coeffs)
    q = (1j) * (2 * 2 * _np.pi * K * Pe)  # canonical parameter - note the scaling consistent with xrft
    if symmetry=='even':
        _eigv = 'A_2r'
        _eigs = 'a_2n'
        _r0 = 0
    elif symmetry=='odd':
        _eigv = 'B_2r'
        _eigs = 'b_2n'
        _r0 = 1
    if opt:  # perform this calculation
        lll = _np.where(K >= 0)[0]
        K = K[lll]
        q = q[lll]
        R = _np.round(_np.sqrt(35 * (q.imag) * abs(coeffs[0]) / 4))
        Rmax = int(_np.max(R))
        if Rmax < 75:  # case where q is very small and Rmax is less than (arbitrary) minimum
            Rmax = 75  # always set the minimum size of matrix
        N = Rmax + 5
        ll = _np.where(R < Rmax)[0]
        R[ll] = Rmax # set minimum value
    
    As_ds = phi_array(N, K, symmetry)  # initializes with the theoretical values in the limit q=0 (k=0).

    for k in range(len(q)):
        if opt:
            Nr = int(R[k])  # must be integer
        else:
            Nr = N  # matrix size constant for all q
        if abs(q[k].imag) > 0:
            ak, Ak = eig_pairs(matrix_system(q[k], Nr + 5, coeffs, Kj, symmetry), symmetry)
            for n in range(Nr - _r0):
                if opt is True and (Nr + 5) < Rmax:
                    As_ds[_eigv].isel(k=k, n=n, r=slice(Nr + 5 - _r0)).data[:] = Anorm(Ak[:, n], symmetry)
                else:
                    As_ds[_eigv].isel(k=k, n=n, r=slice(Nr - _r0)).data[:] = Anorm(Ak[:-5, n], symmetry)
            As_ds[_eigs].isel(k=k, n=slice(Nr - _r0)).data[:] = ak[:-5]

    if reflect:  # Using symmetry, complete for k<0 values. For now, only for \{A_2r, a_2n\} pairs
        As_dsc = reflect_dataset(As_ds, k=True, Pe=False, symmetry=symmetry)
        As_ds = As_dsc.combine_first(As_ds)  # combine along k values.


    return As_ds



def A_coefficients_old(q, N, coeffs, K, symmetry='even', case='None'):
    """ Returns the (sorted) eigenvalues and orthonormal eigenvectors of
    Hill's equation.

    Input:
        q: 1d-array, all elements must be imaginary and in ascending order.
        N: size of (square) Matrix
        coeffs: 1d-array, containing Fourier coefficients associated with
            periodic coefficient in Hill's equation.
        K: range(1, M, d). M is highest harmonic in coeffs. d is either 1 or
            two. if d=1, Fourier sum is : cos(y)+cos(2*y)+... if d=2, the sum
            : cos(y)+cos(3*y)+cos(5*y)...
        cosine: True (default). This has to do with Fourier approx to periodic
            coefficient in HIlls equation. If False, then periodic coeff has a
            sine Fourier series.

    Output:
        dict: 'a_{2n}(q)' (2d-array, key: 'a2n') and 'A^{2n}_{2r}(q)' (key:
            'A2n', 3D array).
    """
    cases = ['linear', 'linear2', 'linear3', 'linear4', 'linear5','asine',
             'gaussian', 'gaussian2', 'gaussian3', 'quad', 'cuartic',
             'sixth', 'tenth', 'square3', 'square4', 'square5']
    vals = {}
    if len(q)>1:
        if q.imag.any() == 0:
            raise Warning("q must be imaginary")
    else:
        if q.imag == 0:
            raise Warning("q must be imaginary")
    if symmetry == 'odd':
        n0 = 1
        _eigv = 'B'
        _eigs = 'b'
    else:
        n0 = 0
        _eigv = 'A'
        _eigs = 'a'
    for n in range(N - n0):
        am, Am = eig_pairs(matrix_system(q[0], N, coeffs, K, symmetry), symmetry)
        a = [am[n]]  # makes a list of the nth eigenvalue
        As = Anorm(Am[:, n], symmetry)
        As = As[_np.newaxis, :]
        for k in range(1, len(q)):
            an, An = eig_pairs(matrix_system(q[k], N, coeffs, K, symmetry), symmetry)
            a.append(an[n])
            nA = Anorm(An[:, n], symmetry)
            nAs = nA[_np.newaxis, :]
            As = _np.append(As, nAs, axis=0)
        if case not in cases:
            As = Fcoeffs(As, n, q, case)
        vals.update({_eigs + str(2 * n): _np.array(a)})
        vals.update({_eigv + str(2 * n): As})
    if case in cases:
        if case == 'linear':
            vals = reorder_linear(vals, q)
        elif case == 'linear2':
            vals = reorder_linear2(vals, q)
        elif case == 'linear3':
            vals = reorder_linear3(vals, q)
        elif case == 'linear4':
            vals = reorder_linear4(vals, q)
        elif case=='linear5':
            vals = reorder_linear5(vals, q)
        elif case == 'square3':
            vals = reorder_sqr3(vals, q)
            for n in range(N):
                As = copy.deepcopy(vals[_eigv + str(2 * n)])
                As = Fcoeffs(As, n, q, case)
                vals.update({_eigv + str(2 * n): As})
        elif case == 'square4':
            vals = reorder_sqr4(vals, q)
        elif case == 'square5':
            vals = reorder_sqr5(vals, q)
        elif case == 'asine':
            vals = reorder_asine(vals, q)
        elif case == 'quad':
            vals = reorder_quad(vals, q)
        elif case == 'cuartic':
            vals = reorder_cuartic(vals, q)
        elif case == 'sixth':
            vals = reorder_sixth(vals, q)
        elif case == 'tenth':
            vals = reorder_tenth(vals, q)
        elif case == 'gaussian':  # narrow gaussian
            vals = reorder_gauss(vals, q)
            for n in range(N):
                As = copy.deepcopy(vals['A' + str(2 * n)])
                As = Fcoeffs(As, n, q, case)
                vals.update({_eigv + str(2 * n): As})
        elif case == 'gaussian2':
            vals = reorder_gauss2(vals, q)
        elif case == 'gaussian3':  # wide gaussian
            vals = reorder_gauss3(vals, q)
    return vals


def Fcoeffs(As, n=0, q=0.00001 * (1j), case='None'):
    """ Returns the Fourier coefficient of the Mathieu functions for given
    parameter q. Makes sure the coefficients are continuous (in q). Numerical
    routines for estimating eigenvectors might converge in different signs
    for the eigenvectors for different (neighboring) values of q. In case where
    q is purely imaginary, eigenvectors need to be rotated, so that these
    satisfy certain relations across branch points.
        Input:
            As: 2d array. Eigenvector shape(As)=Nq, N, as a function of q and
                containing N entries, each associated with a Fourier
                coefficient.
            n: int. Eigenvalue index. If n=0, eigenvalue is a0. n=1, eigenvalue
            is a2.
            q: array, real or imag. Default is q=0.00001j, imaginary.
        Output:
            Corrected Eigenvector with same shape as original
    """
    # Estimate limiting value for small q (pos or neg) and correct.
    if case == 'linear':
        As = linCoeffs(As, n, q)
    elif case == 'linear0':
        As = lin0Coeffs(As, n, q)
    elif case == 'linear2':
        As = linCoeffs2(As, n, q)
    elif case == 'nlinear':
        As = linnCoeffs(As, n, q)
    elif case == 'square3':
        As = sqrCoeffs(As, n, q)
    elif case == 'step':  # sine flow, associated with odd symmetry
        As = stepCoeffs(As, n, q)
    elif case == 'gaussian':
        As = gaussCoeffs(As, n, q)
    elif case == 'gaussian3':
        As = gauss3Coeffs(As, n, q)
    elif case == 'cosine':
        As = cCoeffs(As, n, q)
    elif case == 'cosine_shifted':
        nAs = cCoeffs(As, n, q)
        As = copy.deepcopy(nAs)
        for r in range(len(As[0, :])):
            As[:, r] = ((-1)**n) * _np.cos(r * _np.pi) * nAs[:, r]
    return As


def cCoeffs(A, n, q):
    '''Correct the behavior of the Fourier coefficients as a function of
    parameter (purely imaginary). The Fourier coefficients are complex.
    This is the case of a cosine-jet (Mathieu equation)
    Input:
        A: nd-array. Fourier coefficients (eigenvector) with real and imaginary
            components.
        n: int, index of the eigenvector -> n associated with ce_{2n}
        q: complex, value of the parameter. For now assumed to span values
            before the second branch point q<16i.
    Output:
        A: nd-array. Corrected Fourier coefficient.
    '''
    qs = [2.938438, 32.942942,
          95.613113, 190.950950,
          318.959459, 479.636636,
          672.983483, 898.998498,
          1157.684684, 1449.039039,
          1773.061561, 2129.755255]
    N = len(A[0, :])
    if n < 2 and q[0].imag < qs[0]:
        if q.imag[-1] > qs[0]:
            ll = _np.where(q.imag <= qs[0])[0]
            if n == 0:
                for k in range(N):
                    if k % 2 ==0:
                        A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                    else:
                        A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
    if n in [2, 3] and q[0].imag < qs[1]:
        if q.imag[-1] > qs[1]:
            ll = _np.where(q.imag <= qs[1])[0]
            if n == 2:
                for k in range(N):
                    if k % 2 ==0:
                        A[ll[-1] + 1:, k] = -A[ll[-1] + 1:, k]
                    else:
                        A[ll[-1] + 1:, k] = -A[ll[-1] + 1:, k]
                m0 = _np.where(A[ll[-1]+1:, 0].real > 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real<0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
            if n == 3:
                m0 = _np.where(A[ll[-1]+1:, 0].real > 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real<0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [4, 5] and q[0].imag < qs[2]:
        if q.imag[-1] > qs[2]:
            ll = _np.where(q.imag <= qs[2])[0]
            if n == 4:
                for k in range(N):
                    if k % 2 ==0:
                        A[ll[-1] + 1:, k] = - A[ll[-1] + 1:, k]
                    else:
                        A[ll[-1] + 1:, k] = - A[ll[-1] + 1:, k]
                m0 = _np.where(A[ll[-1]+1:, 0].real < 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
            if n == 5:
                for k in range(N):
                    if k % 2 ==0:
                        A[ll[-1]+1:, k] = - A[ll[-1]+1:, k]
                    else:
                        A[ll[-1]+1:, k] = - A[ll[-1]+1:, k]
                m0 = _np.where(A[ll[-1]+1:, 0].real < 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
    if n in [6, 7] and q[0].imag < qs[3]:
        if q.imag[-1] > qs[3]:
            ll = _np.where(q.imag <= qs[3])[0]
            if n == 6:
                for k in range(N):
                    if k % 2 ==0:
                        NNN = 1
                    # else:
                    #     A[:, k] = -A[:, k]
                m0 = _np.where(A[ll[-1]+1:, 0].real > 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
            if n == 7:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                    else:
                        A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
    if n in [8, 9] and q[0].imag < qs[4]:
        if q[-1].imag > qs[4]:
            ll = _np.where(q.imag <= qs[4])[0]
            if n == 8:
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
            if n == 9:
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
    if n in [10, 11] and q[0].imag < qs[5]:
        if q[-1].imag > qs[5]:
            ll = _np.where(q.imag <= qs[5])[0]
            if n == 10:
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
            if n == 11:
                for k in range(N):
                    if k % 2==0:
                        A[ll[-1]+1:, k]= -A[ll[-1]+1:, k]
                    else:
                        A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
    if n in [12, 13] and q[0].imag < qs[6]:
        if q[-1].imag > qs[6]:
            ll = _np.where(q.imag <= qs[6])[0]
            if n == 12:
                for k in range(N):
                    if k % 2 ==0:
                        A[:ll[-1]+1, k].real = -A[:ll[-1]+1, k].real
                    else:
                        A[:ll[-1]+1, k].imag = -A[:ll[-1]+1, k].imag
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
            if n == 13:
                # for k in range(N):
                    # if k % 2 == 0:
                    #     NNN = 1
                    # else:
                    #     A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
    if n in [14, 15] and q[0].imag < qs[7]:
        if q[-1].imag > qs[7]:
            ll = _np.where(q.imag <= qs[7])[0]
            if n == 14:
                m0 = _np.where(A[ll[-1] + 1:, 0].real >0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
            if n == 15:
                m0 = _np.where(A[ll[-1] + 1:, 0].real >0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
    if n in [16, 17] and q[0].imag < qs[8]:
        if q[-1].imag > qs[8]:
            ll = _np.where(q.imag <= qs[8])[0]
            if n == 16:
                m0 = _np.where(A[ll[-1] + 1:, 0].real <0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
            if n == 17:
                m0 = _np.where(A[ll[-1] + 1:, 0].real <0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
    if n in [18, 19] and q[0].imag < qs[9]:
        if q[-1].imag > qs[9]:
            ll = _np.where(q.imag <= qs[9])[0]
            if n == 18:
                m0 = _np.where(A[ll[-1] + 1:, 0].real >0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
            if n == 19:
                m0 = _np.where(A[ll[-1] + 1:, 0].real >0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
    if n in [20, 21] and q[0].imag < qs[10]:
        if q[-1].imag > qs[10]:
            ll = _np.where(q.imag <= qs[10])[0]
            if n == 20:
                m0 = _np.where(A[ll[-1] + 1:, 0].real <0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
            if n == 21:
                for k in range(N):
                    if k % 2 ==0:
                        # NNN = 1
                        A[:ll[-1]+1, k].imag = -A[:ll[-1]+1, k].imag
                    else:
                        A[:ll[-1]+1, k].real = -A[:ll[-1]+1, k].real
                m0 = _np.where(A[ll[-1] + 1:, 0].real <0)[0]  # never changes sign
                A[m0+ll[-1]+1, :] = -A[m0+ll[-1]+1, :]
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
    # if q.imag[-1] >= qs[-1]:
    #     raise ValueError("Not yet implemented for values of Mathieu`s"
    #                      "canonical parameter q>95i")
    return A


def linCoeffs(A, n, q):
    '''Correct the behavior of the Fourier coefficients as a function of
    parameter (purely imaginary). The Fourier coefficients are complex.
    This is the case of a linear (triangular)-jet.
    Input:
        A: nd-array. Fourier coefficients (eigenvector) with real and imaginary
            components.
        n: int, index of the eigenvector -> n associated with ce_{2n}
        q: complex, value of the parameter. For now assumed to span values
            before the second branch point q<16i.
    Output:
        A: nd-array. Corrected Fourier coefficient.
    '''
    qs = [3.561561, 44.290290,
          131.600600, 264.419419,
          443.489489, 668.252752,
          939.133133]
    N = len(A[0, :])
    if n < 2 and q[0].imag < qs[0]:
        if q.imag[-1] > qs[0]:
            ll = _np.where(q.imag <= qs[0])[0]
            if n == 0:
                for k in range(N):
                    if k % 2 == 0:
                        NNNNN = 1
                        # A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[:, k].imag = -A[:, k].imag
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 1:
                for k in range(N):
                    if k % 2 == 0:
                        A[:, k].imag = -A[:, k].imag
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            #         else:
            #             A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                mm = _np.where(A[:, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [2, 3] and q[0].imag < qs[1]:
        if q.imag[-1] > qs[1]:
            ll = _np.where(q.imag <= qs[1])[0]
            if n == 2:
                for k in range(N):
                    if k % 2 == 0:  # even
                        NNNNN=1
                    else:
                        A[:, k] = -A[:, k]
            if n == 3:
                for k in range(N):
                    if k % 2 == 0:
                        A[:, k] = -A[:, k]
                # after EP
                mm = _np.where(A[:, 0].real > 0)[0]  # should be <0
                A[mm, :] = -A[mm, :]
                # before EP
                mm = _np.where(A[:ll[-1] + 1, n].real <0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [4, 5] and q[0].imag < qs[2]:
        if q.imag[-1] > qs[2]:
            ll = _np.where(q.imag <= qs[2])[0]
            if n == 4:
                for k in range(N):
                    if k % 2 == 0:
                        NNN = 1
                    else:
                        A[:, k] = - A[:, k]
            if n == 5:
                for k in range(N):
                    if k % 2 == 0:
                        A[:, k] = -A[:, k]

                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [6, 7] and q[0].imag < qs[3]:
        if q.imag[-1] > qs[3]:
            ll = _np.where(q.imag <= qs[3])[0]
            if n == 6:
                for k in range(N):
                    if k % 2 == 0:
                        NNN = 1
                    else:
                        A[:, k] = -A[:, k]
            if n == 7:
                for k in range(N):
                    if k % 2 == 0:
                        A[:, k]= -A[:, k]
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [8, 9] and q[0].imag < qs[4]:
        if q.imag[-1] > qs[4]:
            ll = _np.where(q.imag <= qs[4])[0]
            if n == 8:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                for k in range(N):
                    if k % 2 == 0:
                        NNN = 1
                    else:
                        A[:, k] = -A[:, k]
            if n == 9:
                for k in range(N):
                    if k % 2 == 0:
                        A[:, k] = -A[:, k]
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [10, 11] and q[0].imag < qs[5]:
        if q.imag[-1] > qs[5]:
            ll = _np.where(q.imag <= qs[5])[0]
            if n == 10:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                for k in range(N):
                    if k % 2 == 0:
                        NNN = 1
                    else:
                        A[:, k] = -A[:, k]
            if n == 11:
                for k in range(N):
                    if k % 2 == 0:
                        A[:, k] = -A[:, k]
                    # else:
                        # A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [12, 13] and q[0].imag < qs[6]:
        if q.imag[-1] > qs[6]:
            ll = _np.where(q.imag <= qs[6])[0]
            if n == 12:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                for k in range(N):
                    if k % 2 == 0:
                        NNN = 1
                    else:
                        A[:, k] = -A[:, k]
            if n == 13:
                for k in range(N):
                    if k % 2 == 0:
                        A[:, k] = -A[:, k]
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    return A


def linCoeffs2(A, n, q):
    '''Correct the behavior of the Fourier coefficients as a function of
    parameter (purely imaginary). The Fourier coefficients are complex.
    This is the case of a linear (triangular)-jet.
    Input:
        A: nd-array. Fourier coefficients (eigenvector) with real and imaginary
            components.
        n: int, index of the eigenvector -> n associated with ce_{2n}
        q: complex, value of the parameter. For now assumed to span values
            before the second branch point q<16i.
    Output:
        A: nd-array. Corrected Fourier coefficient.
    '''
    qs = [3.561561, 44.290290,
          131.600600, 264.419419,
          443.489489, 668.252752,
          939.133133]
    N = len(A[0, :])
    if n < 2 and q[0].imag < qs[0]:
        if q.imag[-1] > qs[0]:
            ll = _np.where(q.imag <= qs[0])[0]
            if n == 0:
                for k in range(N):
                    if k % 2 ==0:
                        NNN = 1
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 1:
                for k in range(N):
                    if k % 2 == 0:
                        A[:, k].imag = -A[:, k].imag
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                    else:
                        A[:, k].imag = -A[:, k].imag
                        # A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                mm = _np.where(A[:ll[-1]+1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [2, 3] and q[0].imag < qs[1]:
        if q.imag[-1] > qs[1]:
            ll = _np.where(q.imag <= qs[1])[0]
            if n == 2:
                for k in range(N):
                    if k % 2 == 0:  # even
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 3:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                # after EP
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                # before EP
                mm = _np.where(A[:ll[-1] + 1, n].real <0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [4, 5] and q[0].imag < qs[2]:
        if q.imag[-1] > qs[2]:
            ll = _np.where(q.imag <= qs[2])[0]
            if n == 4:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real= -A[ll[-1] + 1:, k].real
            if n == 5:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real= -A[ll[-1] + 1:, k].real

                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [6, 7] and q[0].imag < qs[3]:
        if q.imag[-1] > qs[3]:
            ll = _np.where(q.imag <= qs[3])[0]
            if n == 6:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 7:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [8, 9] and q[0].imag < qs[4]:
        if q.imag[-1] > qs[4]:
            ll = _np.where(q.imag <= qs[4])[0]
            if n == 8:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                for k in range(N):
                    if k % 2 == 0:
                       A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 9:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [10, 11] and q[0].imag < qs[5]:
        if q.imag[-1] > qs[5]:
            ll = _np.where(q.imag <= qs[5])[0]
            if n == 10:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                # m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                # A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 11:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [12, 13] and q[0].imag < qs[6]:
        if q.imag[-1] > qs[6]:
            ll = _np.where(q.imag <= qs[6])[0]
            if n == 12:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
            if n == 13:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    return A


def linnCoeffs(A, n, q):
    '''Correct the behavior of the Fourier coefficients as a function of
    parameter (purely imaginary). The Fourier coefficients are complex.
    This is the case of a linear (triangular)-jet.
    Input:
        A: nd-array. Fourier coefficients (eigenvector) with real and imaginary
            components.
        n: int, index of the eigenvector -> n associated with ce_{2n}
        q: complex, value of the parameter. For now assumed to span values
            before the second branch point q<16i.
    Output:
        A: nd-array. Corrected Fourier coefficient.
    '''
    qs = [3.561561, 44.290290,
          131.600600, 264.419419,
          443.489489, 668.252752,
          939.133133]
    N = len(A[0, :])
    if n < 2 and q[0].imag < qs[0]:
        if q.imag[-1] > qs[0]:
            ll = _np.where(q.imag <= qs[0])[0]
            if n == 0:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                    else:
                        A[ll[-1] + 1:, k] = -A[ll[-1] + 1:, k]
            if n == 1:
                # for k in range(N):
                #     if k % 2 == 0:
                #         A[:, k].imag = -A[:, k].imag
                mm = _np.where(A[:, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [2, 3] and q[0].imag < qs[1]:
        if q.imag[-1] > qs[1]:
            ll = _np.where(q.imag <= qs[1])[0]
            if n == 2:
                for k in range(N):
                    if k % 2 == 0:  # even
                        A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                    else:
                        A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
            mm = _np.where(A[:, 0].real > 0)[0]  # should be <0
            A[mm, :] = -A[mm, :]
            if n == 3:
                for k in range(N):
                    if k % 2 == 0:
                        A[:, k] = -A[:, k]
                    else:
                        A[:, k] = -A[:, k]
                # after EP
                mm = _np.where(A[:, 0].real > 0)[0]  # should be <0
                A[mm, :] = -A[mm, :]
                # before EP
                mm = _np.where(A[:ll[-1] + 1, n].real <0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [4, 5] and q[0].imag < qs[2]:
        if q.imag[-1] > qs[2]:
            ll = _np.where(q.imag <= qs[2])[0]
            if n == 4:
                mm = _np.where(A[:, 0].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
            if n == 5:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                    else:
                        A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [6, 7] and q[0].imag < qs[3]:
        if q.imag[-1] > qs[3]:
            ll = _np.where(q.imag <= qs[3])[0]
            if n == 6:
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
            if n == 7:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [8, 9] and q[0].imag < qs[4]:
        if q.imag[-1] > qs[4]:
            ll = _np.where(q.imag <= qs[4])[0]
            if n == 8:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
            if n == 9:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [10, 11] and q[0].imag < qs[5]:
        if q.imag[-1] > qs[5]:
            ll = _np.where(q.imag <= qs[5])[0]
            if n == 10:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
            if n == 11:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [12, 13] and q[0].imag < qs[6]:
        if q.imag[-1] > qs[6]:
            ll = _np.where(q.imag <= qs[6])[0]
            if n == 12:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                # for k in range(N):
                #     if k % 2 == 0:
                #         NNN = 1
                #     else:
                #         A[:, k] = -A[:, k]
            if n == 13:
                # for k in range(N):
                #     if k % 2 == 0:
                #         A[:, k] = -A[:, k]
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    return A


def lin0Coeffs(A, n, q):
    '''Correct the behavior of the Fourier coefficients as a function of
    parameter (purely imaginary). The Fourier coefficients are complex.
    This is the case of a linear (triangular)-jet.
    Input:
        A: nd-array. Fourier coefficients (eigenvector) with real and imaginary
            components.
        n: int, index of the eigenvector -> n associated with ce_{2n}
        q: complex, value of the parameter. For now assumed to span values
            before the second branch point q<16i.
    Output:
        A: nd-array. Corrected Fourier coefficient.
    '''
    qs = [3.561561, 44.290290,
          131.600600, 264.419419,
          443.489489, 668.252752,
          939.133133]
    N = len(A[0, :])
    if n < 2 and q[0].imag < qs[0]:
        if q.imag[-1] > qs[0]:
            ll = _np.where(q.imag <= qs[0])[0]
            if n == 1:
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0+ll[-1] + 1, :]
                # before EP
                mm = _np.where(A[:ll[-1] + 1, n].real <0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [2, 3] and q[0].imag < qs[1]:
        if q.imag[-1] > qs[1]:
            ll = _np.where(q.imag <= qs[1])[0]
            if n == 3:
                mm = _np.where(A[:, 0].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                # before EP
                mm = _np.where(A[:ll[-1] + 1, n].real <0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
    if n in [4, 5] and q[0].imag < qs[2]:
        if q.imag[-1] > qs[2]:
            ll = _np.where(q.imag <= qs[2])[0]
            # if n == 4:
                # mm = _np.where(A[:, 0].real < 0)[0]  # should be >0
                # A[mm, :] = -A[mm, :]
                # mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                # A[mm, :] = -A[mm, :]
            if n == 5:
                # for k in range(N):
                #     if k % 2 == 0:
                #         A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                #     else:
                #         A[ll[-1]+1:, k] = -A[ll[-1]+1:, k]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                # mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                # A[mm, :] = -A[mm, :]
    if n in [6, 7] and q[0].imag < qs[3]:
        if q.imag[-1] > qs[3]:
            ll = _np.where(q.imag <= qs[3])[0]
            if n == 6:
                for k in range(N):
                    if k % 2 == 0:
                        NNN = 1
                        # A[:ll[-1]+1, k].imag = -A[:ll[-1]+1, k].imag
                    else:
                        A[:ll[-1]+1, k].real = -A[:ll[-1]+1, k].real
            #     m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
            #     A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
            #     mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
            #     A[mm, :] = -A[mm, :]
            if n == 7:
                # for k in range(N):
                #     if k % 2 == 0:
                #         A[:ll[-1]+1, k] = -A[:ll[-1]+1, k]
                #     else:
                #         A[:ll[-1]+1, k].real = -A[:ll[-1]+1, k].real
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [8, 9] and q[0].imag < qs[4]:
        if q.imag[-1] > qs[4]:
            ll = _np.where(q.imag <= qs[4])[0]
            if n == 8:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                # m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                # A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
            if n == 9:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [10, 11] and q[0].imag < qs[5]:
        if q.imag[-1] > qs[5]:
            ll = _np.where(q.imag <= qs[5])[0]
            if n == 10:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
            #     m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
            #     A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
            if n == 11:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    if n in [12, 13] and q[0].imag < qs[6]:
        if q.imag[-1] > qs[6]:
            ll = _np.where(q.imag <= qs[6])[0]
            if n == 12:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
            #     m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
            #     A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
            if n == 13:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    return A



def sqrCoeffs(A, n, q):
    '''Correct the behavior of the Fourier coefficients as a function of
    parameter (purely imaginary). The Fourier coefficients are complex.
    This is the case of a square-jet.
    Input:
        A: nd-array. Fourier coefficients (eigenvector) with real and imaginary
            components.
        n: int, index of the eigenvector -> n associated with ce_{2n}
        q: complex, value of the parameter. For now assumed to span values
            before the second branch point q<16i.
    Output:
        A: nd-array. Corrected Fourier coefficient.
    '''
    qs = [2.296796, 19.198698,
          44.409409, 75.029029,
          109.620620, 147.331331]
    N = len(A[0, :])
    # if n < 2 and q[0].imag < qs[0]:
    #     if q.imag[-1] > qs[0]:
    #         ll = _np.where(q.imag <= qs[0])[0]
    #         if n == 0:
    #             for k in range(N):
    #                 if k % 2 == 0:
    #                     A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    #                 else:
    #                     A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
    #         if n == 1:
    #             mm = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
    #             A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
    #             for k in range(N):
    #                 if k % 2 == 0:
    #                     A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
    #                 else:
    #                     A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    if n == 2:
        if q.imag[-1] > 20:
            ll = _np.where(q.imag <= qs[1])[0]
            # if n == 2:
                # mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                # A[mm, :] = -A[mm, :]
            m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
            A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                # for k in range(N):
                #     if k % 2 == 0:
                #         A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                #     else:
                #         A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            # if n == 3:
            #     # mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
            #     # A[mm, :] = -A[mm, :]
            #     m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be >0
            #     A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
            #     for k in range(N):
            #         if k % 2 == 0:
            #             A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            #         else:
            #             A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    # if n in [4, 5] and q[0].imag < qs[2]:
    #     if q.imag[-1] > qs[2]:
    #         ll = _np.where(q.imag <= qs[2])[0]
    #         if n == 4:
    #             for k in range(N):
    #                 if k % 2 == 0:
    #                     A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    #                 else:
    #                     A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
    #         if n == 5:
    #             for k in range(N):
    #                 if k % 2 == 0:
    #                     A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
    #                 else:
    #                     A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    #             mm = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
    #             A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
    # if n in [6, 7] and q[0].imag < qs[3]:
    #     if q.imag[-1] > qs[3]:
    #         ll = _np.where(q.imag <= qs[3])[0]
    #         if n == 6:
    #             for k in range(N):
    #                 if k % 2 == 0:
    #                     A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    #                 else:
    #                     A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
    #         if n == 7:
    #             for k in range(N):
    #                 if k % 2 == 0:
    #                     A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
    #                 else:
    #                     A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    #             mm = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
    #             A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
    # if n in [8, 9] and q[0].imag < qs[4]:
    #     if q.imag[-1] > qs[4]:
    #         ll = _np.where(q.imag <= qs[4])[0]
    #         if n == 8:
    #             for k in range(N):
    #                 if k % 2 == 0:
    #                     A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    #                 else:
    #                     A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
    #         if n == 9:
    #             for k in range(N):
    #                 if k % 2 == 0:
    #                     A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
    #                 else:
    #                     A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    #         mm = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
    #         A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
    # if n in [10, 11] and q[0].imag < qs[5]:
    #     if q.imag[-1] > qs[5]:
    #         ll = _np.where(q.imag <= qs[5])[0]
    #         if n == 10:
    #             for k in range(N):
    #                 if k % 2 == 0:
    #                     A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    #                 else:
    #                     A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
    #         if n == 11:
    #             for k in range(N):
    #                 if k % 2 == 0:
    #                     A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
    #                 else:
    #                     A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    return A


def stepCoeffs(B, n, q):
    '''Correct the behavior of the Fourier coefficients as a function of
    parameter (purely imaginary). The Fourier coefficients are complex.
    This is the case of a step-jet, associated with an odd-eigenfunction
    Input:
        B: nd-array. Fourier coefficients (eigenvector) with real and imaginary
            components.
        n: int, index of the eigenvector -> n associated with se_{2n}
        q: complex, value of the parameter. For now assumed to span values
            before the second branch point q<16i.
    Output:
        A: nd-array. Corrected Fourier coefficient.
    '''
    qs = [5.525525, 33.860860,
          78.102602, 145.605605,
          228.899899]
    N = len(B[0, :])
    if n < 2 and q[0].imag < qs[0]:
        if q.imag[-1] > qs[0]:
            ll = _np.where(q.imag <= qs[0])[0]
            if n == 0:
                for k in range(N):
                    if k % 2 == 0:
                        B[ll[-1] + 1:, k].real = -B[ll[-1] + 1:, k].real
                    else:
                        B[ll[-1] + 1:, k].imag = -B[ll[-1] + 1:, k].imag
                mm = _np.where(B[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                B[mm + ll[-1] + 1, :] = -B[mm + ll[-1] + 1, :]
            if n == 1:
                for k in range(N):
                    if k % 2 == 0:
                        B[ll[-1] + 1:, k].real = -B[ll[-1] + 1:, k].real
                    else:
                        B[ll[-1] + 1:, k].imag = -B[ll[-1] + 1:, k].imag
    if n in [2, 3] and q[0].imag < qs[1]:
        if q.imag[-1] > qs[1]:
            ll = _np.where(q.imag <= qs[1])[0]
            if n == 2:
                for k in range(N):
                    if k % 2 == 0:
                        B[ll[-1] + 1:, k].real = -B[ll[-1] + 1:, k].real
                        # B[ll[-1] + 1, k] = -B[ll[-1] + 1, k]
                    else:
                        B[ll[-1] + 1:, k].imag = -B[ll[-1] + 1:, k].imag
                    B[ll[-1] + 1, k] = -B[ll[-1] + 1, k]
                mm = _np.where(B[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                B[mm + ll[-1] + 1, :] = -B[mm + ll[-1] + 1, :]
            elif n == 3:
                for k in range(N):
                    if k % 2 == 0:
                        B[ll[-1] + 1:, k].real = -B[ll[-1] + 1:, k].real
                    else:
                        B[ll[-1] + 1:, k].imag = -B[ll[-1] + 1:, k].imag
    if n in [4, 5] and q[0].imag < qs[2]:
        if q.imag[-1] > qs[2]:
            ll = _np.where(q.imag <= qs[2])[0]
            if n == 4:
                nn = _np.where(B[:ll[-1], n] < 0)[0]  # should be > 0.
                B[nn, :] = - B[nn, :]
                for k in range(N):
                    if k % 2 == 0:
                        B[ll[-1] + 1:, k].imag = -B[ll[-1] + 1:, k].imag
                    else:
                        B[ll[-1] + 1:, k].real = -B[ll[-1] + 1:, k].real
                mm = _np.where(B[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                B[mm + ll[-1] + 1, :] = -B[mm + ll[-1] + 1, :]
            if n == 5:
                nn = _np.where(B[:ll[-1], n] < 0)[0]  # should be > 0.
                B[nn, :] = - B[nn, :]
                for k in range(N):
                    if k % 2 == 0:
                        B[ll[-1] + 1:, k].real = -B[ll[-1] + 1:, k].real
                    else:
                        B[ll[-1] + 1:, k].imag = -B[ll[-1] + 1:, k].imag
    if n in [6, 7] and q[0].imag < qs[3]:
        if q.imag[-1] > qs[3]:
            ll = _np.where(q.imag <= qs[3])[0]
            if n == 6:
                nn = _np.where(B[:ll[-1], n] < 0)[0]  # should be > 0.
                B[nn, :] = - B[nn, :]
                for k in range(N):
                    if k % 2 == 0:
                        B[ll[-1] + 1:, k].imag = -B[ll[-1] + 1:, k].imag
                    else:
                        B[ll[-1] + 1:, k].real = -B[ll[-1] + 1:, k].real
                mm = _np.where(B[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                B[mm + ll[-1] + 1, :] = -B[mm + ll[-1] + 1, :]
            elif n == 7:
                nn = _np.where(B[:ll[-1], n] < 0)[0]  # should be > 0.
                B[nn, :] = - B[nn, :]
                B[ll[-1], :] = -B[ll[-1], :]
                for k in range(N):
                    if k % 2 == 0:
                        B[ll[-1] + 1:, k].real = -B[ll[-1] + 1:, k].real
                    else:
                        B[ll[-1] + 1:, k].imag = -B[ll[-1] + 1:, k].imag
    if n in [8, 9] and q[0].imag < qs[4]:
        if q.imag[-1] > qs[4]:
            ll = _np.where(q.imag <= qs[4])[0]
            if n == 8:
                nn = _np.where(B[:ll[-1], n] < 0)[0]  # should be > 0.
                B[nn, :] = - B[nn, :]
                for k in range(N):
                    if k % 2 == 0:
                        B[ll[-1] + 1:, k].imag = -B[ll[-1] + 1:, k].imag
                    else:
                        B[ll[-1] + 1:, k].real = -B[ll[-1] + 1:, k].real
                mm = _np.where(B[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                B[mm + ll[-1] + 1, :] = -B[mm + ll[-1] + 1, :]
            elif n == 9:
                nn = _np.where(B[:ll[-1], n] < 0)[0]  # should be > 0.
                B[nn, :] = - B[nn, :]
                B[ll[-1], :] = -B[ll[-1], :]
                for k in range(N):
                    if k % 2 == 0:
                        B[ll[-1] + 1:, k].real = -B[ll[-1] + 1:, k].real
                    else:
                        B[ll[-1] + 1:, k].imag = -B[ll[-1] + 1:, k].imag
                mm = _np.where(B[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                B[mm + ll[-1] + 1, :] = -B[mm + ll[-1] + 1, :]
    return B


def gauss3Coeffs(As, n, q):
    '''Correct the behavior of the Fourier coefficients as a function of
    parameter (purely imaginary). The Fourier coefficients are complex.
    This is the case of a narrow gaussian-jet, associated with an even
    eigenfunction. In this case there are not Exceptional Points, hence no
    symmetry of the Fourier coefficients.

    In this case, the only requirement is that the q-dependence of As is
    continuous (smooth), i.e. there are not (sign) jumps in q over what
    looks like EPs.
    '''
    # Define crossing of Eigenvalues.
    qs = [36.060060, 70.293793,
          114.818818, 175.898898,
          260.515515, 379.465965,
          553.160660, 788.364864]
    if n == 1:
        mm = _np.where(As[:, 0].real > 0)[0]  # sign jumps
        As[mm, :] = - As[mm, :]
    return As


def gaussCoeffs(As, n, q):
    '''Correct behavior of Fourier coefficients as a function of q-parameter. This case is
    associated with the narrow gaussian jet (Ld = 1 / 2.9).
    ''' 
    if n == 1:
        mm = _np.where(As[:, 0].real > 0)[0]  # sign jumps
        As[mm, :] = - As[mm, :]

    # if n ==2:
    #     if q.imag[-1] > 20:
    #         ll = _np.where(q.imag <= 20)[0]
    #         # mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
    #         # A[mm, :] = -A[mm, :]
    #         m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be <0
    #         A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
    return As


def Anorm(A, symmetry='even'):
    """ Normalization of eigenvectors in accordance to Mathieu functions.
    Default is for that associated with ce_{2n}(q, z).
    Input:
        A: 1d-array. len(A) = N, N being the number of Fourier coefficients.
        type: str, default `ce2n`. Normalization for other functions is
        different.
        case: str, 'None' (default), or 'Mathieu'.
    Output:
        A: 1d-array. Normalized eigenvector.
    """
    A0star = _copy.deepcopy(A[0])
    Astar = _copy.deepcopy(A[1:])
    if symmetry == "even":
        fac = 2
    else:
        fac = 1
    norm = _np.sqrt((fac * (A[0] * A0star)) + _np.sum(A[1:] * Astar))
    A = _copy.deepcopy(A) / norm
    return A


def reorder_linear(Avals, Q):
    """Changes the ordering of pairs of eigenvalues and eigenvectors that are stored within a
    dictionary. This is associated with triangular jet (Ld=1.5).
    """
    qs = [544.031, 589.201]
    pair = ['16', '18', '20']  # pair whose eigvals cross.
    Adict = copy.deepcopy(Avals)
    l0 = _np.where(Q.imag >= qs[0])[0][0]
    l1 = _np.where(Q.imag >= qs[1])[0][0]

    A16 = copy.deepcopy(Avals['A16'][l0:l1, :])
    A18 = copy.deepcopy(Avals['A18'][l0:l1, :])
    A20 = copy.deepcopy(Avals['A20'][l0:l1, :])

    a16 = copy.deepcopy(Avals['a16'][l0:l1])
    a18 = copy.deepcopy(Avals['a18'][l0:l1])
    a20 = copy.deepcopy(Avals['a20'][l0:l1])

    Adict['A16'][l0:l1, :] = A18
    Adict['A18'][l0:l1, :] = A20
    Adict['A20'][l0:l1, :] = A16

    Adict['a16'][l0:l1] = a18
    Adict['a18'][l0:l1] = a20
    Adict['a20'][l0:l1] = a16
    return Adict


def reorder_linear2(Avals, Q):
    """Changes the ordering of the eigenvectors and eigenvalues that are stored
    within a dictionary, whenever the value of canonical parameter q lies
    between an interval. This represents the case of a middle gaussian jet.
    """
    # first mode, asymptotes to 2n=2
    qs = [117.2, 734.15, 2281.5, 5152, 9769.6]
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A2 = copy.deepcopy(Adict1['A2'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :])
        a2 = copy.deepcopy(Adict1['a2'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 2))][M[m] + 1:])
        Adict1['A2'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :] = A2
        Adict1['a2'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 2))][M[m] + 1:] = a2


    # 2nd mode, asymptotes to 2n=6
    qs = [86.725, 314.4, 764.9, 1506.5, 2613.9, 4158.4, 6180.3, 8768]
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A6 = copy.deepcopy(Adict2['A6'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict2['A' + str(2 * (m + 4))][M[m] + 1:, :])
        a6 = copy.deepcopy(Adict2['a6'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict2['a' + str(2 * (m + 4))][M[m] + 1:])
        Adict2['A6'][M[m] + 1:, :] = Am
        Adict2['A' + str(2 * (m + 4))][M[m] + 1:, :] = A6
        Adict2['a6'][M[m] + 1:] = am
        Adict2['a' + str(2 * (m + 4))][M[m] + 1:] = a6

    # pair of eigs
    qs = [181.2]
    pair = [['10', '12']]  # pair whose eigvals cross.
    Adict3 = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict3['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict3['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict3['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict3['a' + pair[m][1]][M[m] + 1:])

        Adict3['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict3['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict3['a' + pair[m][0]][M[m] + 1:] = a2
        Adict3['a' + pair[m][1]][M[m] + 1:] = a0


    # pair of eigs
    qs = [395.153]
    pair = [['14', '16']]  # pair whose eigvals cross.
    Adict4 = copy.deepcopy(Adict3)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict4['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict4['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict4['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict4['a' + pair[m][1]][M[m] + 1:])

        Adict4['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict4['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict4['a' + pair[m][0]][M[m] + 1:] = a2
        Adict4['a' + pair[m][1]][M[m] + 1:] = a0


    # pair of eigs
    qs = [411.8]
    pair = [['10', '16']]  # pair whose eigvals cross.
    Adict5 = copy.deepcopy(Adict4)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict5['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict5['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict5['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict5['a' + pair[m][1]][M[m] + 1:])

        Adict5['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict5['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict5['a' + pair[m][0]][M[m] + 1:] = a2
        Adict5['a' + pair[m][1]][M[m] + 1:] = a0

    # pair of eigs
    qs = [567.1]
    pair = [['18', '20']]  # pair whose eigvals cross.
    Adict6 = copy.deepcopy(Adict5)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict6['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict6['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict6['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict6['a' + pair[m][1]][M[m] + 1:])

        Adict6['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict6['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict6['a' + pair[m][0]][M[m] + 1:] = a2
        Adict6['a' + pair[m][1]][M[m] + 1:] = a0

    # pair of eigs
    qs = [793.8]
    pair = [['14', '20']]  # pair whose eigvals cross.
    Adict7 = copy.deepcopy(Adict6)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict7['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict7['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict7['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict7['a' + pair[m][1]][M[m] + 1:])

        Adict7['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict7['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict7['a' + pair[m][0]][M[m] + 1:] = a2
        Adict7['a' + pair[m][1]][M[m] + 1:] = a0

# pair of eigs
    qs = [830.8]
    pair = [['10', '20']]  # pair whose eigvals cross.
    Adict8 = copy.deepcopy(Adict7)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict8['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict8['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict8['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict8['a' + pair[m][1]][M[m] + 1:])

        Adict8['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict8['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict8['a' + pair[m][0]][M[m] + 1:] = a2
        Adict8['a' + pair[m][1]][M[m] + 1:] = a0


# pair of eigs
    qs = [936.9]
    pair = [['22', '24']]  # pair whose eigvals cross.
    Adict9 = copy.deepcopy(Adict8)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict9['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict9['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict9['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict9['a' + pair[m][1]][M[m] + 1:])

        Adict9['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict9['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict9['a' + pair[m][0]][M[m] + 1:] = a2
        Adict9['a' + pair[m][1]][M[m] + 1:] = a0


# pair of eigs
    qs = [954.2,]
    pair = [['18', '24']]  # pair whose eigvals cross.
    Adict10 = copy.deepcopy(Adict9)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict10['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict10['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict10['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict10['a' + pair[m][1]][M[m] + 1:])

        Adict10['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict10['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict10['a' + pair[m][0]][M[m] + 1:] = a2
        Adict10['a' + pair[m][1]][M[m] + 1:] = a0

# pair of eigs
    qs = [1456.3]
    pair = [['10', '14']]  # pair whose eigvals cross.
    Adict11 = copy.deepcopy(Adict10)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict11['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict11['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict11['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict11['a' + pair[m][1]][M[m] + 1:])

        Adict11['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict11['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict11['a' + pair[m][0]][M[m] + 1:] = a2
        Adict11['a' + pair[m][1]][M[m] + 1:] = a0

    qs = [1393.0125]
    pair = [['14', '24']]  # pair whose eigvals cross.
    Adict12 = copy.deepcopy(Adict11)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict12['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict12['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict12['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict12['a' + pair[m][1]][M[m] + 1:])

        Adict12['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict12['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict12['a' + pair[m][0]][M[m] + 1:] = a2
        Adict12['a' + pair[m][1]][M[m] + 1:] = a0

# pair of eigs
    qs = [2334.4]
    pair = [['10', '14']]  # pair whose eigvals cross.
    Adict13 = copy.deepcopy(Adict12)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict13['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict13['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict13['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict13['a' + pair[m][1]][M[m] + 1:])

        Adict13['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict13['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict13['a' + pair[m][0]][M[m] + 1:] = a2
        Adict13['a' + pair[m][1]][M[m] + 1:] = a0

# pair of eigs
    qs = [2233.175]
    pair = [['14', '18']]  # pair whose eigvals cross.
    Adict14 = copy.deepcopy(Adict13)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict14['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict14['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict14['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict14['a' + pair[m][1]][M[m] + 1:])

        Adict14['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict14['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict14['a' + pair[m][0]][M[m] + 1:] = a2
        Adict14['a' + pair[m][1]][M[m] + 1:] = a0

# pair of eigs
    qs = [3516.1]
    pair = [['10', '14']]  # pair whose eigvals cross.
    Adict15 = copy.deepcopy(Adict14)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict15['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict15['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict15['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict15['a' + pair[m][1]][M[m] + 1:])

        Adict15['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict15['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict15['a' + pair[m][0]][M[m] + 1:] = a2
        Adict15['a' + pair[m][1]][M[m] + 1:] = a0

# pair of eigs
    qs = [3363.8]
    pair = [['14', '22']]  # pair whose eigvals cross.
    Adict16 = copy.deepcopy(Adict15)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict16['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict16['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict16['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict16['a' + pair[m][1]][M[m] + 1:])

        Adict16['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict16['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict16['a' + pair[m][0]][M[m] + 1:] = a2
        Adict16['a' + pair[m][1]][M[m] + 1:] = a0

# pair of eigs
    qs = [5116.5]
    pair = [['10', '14']]  # pair whose eigvals cross.
    Adict17 = copy.deepcopy(Adict16)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict17['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict17['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict17['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict17['a' + pair[m][1]][M[m] + 1:])

        Adict17['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict17['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict17['a' + pair[m][0]][M[m] + 1:] = a2
        Adict17['a' + pair[m][1]][M[m] + 1:] = a0


# pair of eigs
    qs = [4729.4]
    pair = [['14', '26']]  # pair whose eigvals cross.
    Adict18 = copy.deepcopy(Adict17)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict18['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict18['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict18['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict18['a' + pair[m][1]][M[m] + 1:])

        Adict18['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict18['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict18['a' + pair[m][0]][M[m] + 1:] = a2
        Adict18['a' + pair[m][1]][M[m] + 1:] = a0

    return Adict18


def reorder_linear3(Avals, Q):
    """Changes the ordering of the eigenvectors and eigenvalues that are stored
    within a dictionary, whenever the value of canonical parameter q lies
    between an interval. This represents the case of a narrow gaussian jet.
    """
    # first mode, asymptotes to 2n=2
    qs = [18, 66.468, 205.3, 465.5, 882.1, 1487.6, 2327.8, 
          3435.4, 4855., 6643.2, 8864.6, 11584.]
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A2 = copy.deepcopy(Adict1['A2'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :])
        a2 = copy.deepcopy(Adict1['a2'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 2))][M[m] + 1:])
        Adict1['A2'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :] = A2
        Adict1['a2'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 2))][M[m] + 1:] = a2


    # 2nd mode, asymptotes to 2n=16
    qs =  [363.7, 576.7, 805.35, 1090.2, 1455., 1888.525, 2395.1, 
           2976.4, 3648.1]  
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A16 = copy.deepcopy(Adict2['A16'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict2['A' + str(2 * (m + 9))][M[m] + 1:, :])
        a16 = copy.deepcopy(Adict2['a16'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict2['a' + str(2 * (m + 9))][M[m] + 1:])
        Adict2['A16'][M[m] + 1:, :] = Am
        Adict2['A' + str(2 * (m + 9))][M[m] + 1:, :] = A16
        Adict2['a16'][M[m] + 1:] = am
        Adict2['a' + str(2 * (m + 9))][M[m] + 1:] = a16

    # 3rd mode, asymptotes to 2n=26
    qs = [861.2] 
    Adict3 = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A26 = copy.deepcopy(Adict3['A26'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict3['A' + str(2 * (m + 14))][M[m] + 1:, :])
        a26 = copy.deepcopy(Adict3['a26'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict3['a' + str(2 * (m + 14))][M[m] + 1:])
        Adict3['A26'][M[m] + 1:, :] = Am
        Adict3['A' + str(2 * (m + 14))][M[m] + 1:, :] = A26
        Adict3['a26'][M[m] + 1:] = am
        Adict3['a' + str(2 * (m + 14))][M[m] + 1:] = a26

    return Adict3


def reorder_linear4(Avals, Q):
    """Changes the ordering of pairs of eigenvalues and eigenvectors that are stored within a
    dictionary. This is associated with triangular jet (Ld=1.5).
    """
    # pair of eigs
    qs = [13.05]
    pair = [['0', '2']]  # pair whose eigvals cross.
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Avals['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Avals['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Avals['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Avals['a' + pair[m][1]][M[m] + 1:])

        Adict1['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict1['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict1['a' + pair[m][0]][M[m] + 1:] = a2
        Adict1['a' + pair[m][1]][M[m] + 1:] = a0

    # next pair of eigs (EP at 14.5 between the two eigs)
    qs = [43]  #
    pair = [['0','2', '4']]  # pair whose eigvals cross.
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict1['A' + pair[m][0]][M[m] + 1:, :])
        A1 = copy.deepcopy(Adict1['A' + pair[m][1]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict1['A' + pair[m][2]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict1['a' + pair[m][0]][M[m] + 1:])
        a1 = copy.deepcopy(Adict1['a' + pair[m][1]][M[m] + 1:])
        a2 = copy.deepcopy(Adict1['a' + pair[m][2]][M[m] + 1:])

        Adict2['A' + pair[m][1]][M[m] + 1:, :] = A2
        Adict2['A' + pair[m][2]][M[m] + 1:, :] = A1
        Adict2['a' + pair[m][1]][M[m] + 1:] = a2
        Adict2['a' + pair[m][2]][M[m] + 1:] = a1

        A0 = copy.deepcopy(Adict2['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict2['A' + pair[m][2]][M[m] + 1:, :])

        a0 = copy.deepcopy(Adict2['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict2['a' + pair[m][2]][M[m] + 1:])

        Adict2['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict2['A' + pair[m][2]][M[m] + 1:, :] = A0
        Adict2['a' + pair[m][0]][M[m] + 1:] = a2
        Adict2['a' + pair[m][2]][M[m] + 1:] = a0

    # pair of eigs
    qs = [81.6]
    pair = [['2', '4']]  # pair whose eigvals cross.
    Adict3 = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict2['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict2['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict2['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict2['a' + pair[m][1]][M[m] + 1:])

        Adict3['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict3['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict3['a' + pair[m][0]][M[m] + 1:] = a2
        Adict3['a' + pair[m][1]][M[m] + 1:] = a0


    # pair of eigs
    qs = [178.2]
    pair = [['8', '10']]  # pair whose eigvals cross.
    Adict4 = copy.deepcopy(Adict3)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict3['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict3['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict3['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict3['a' + pair[m][1]][M[m] + 1:])

        Adict4['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict4['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict4['a' + pair[m][0]][M[m] + 1:] = a2
        Adict4['a' + pair[m][1]][M[m] + 1:] = a0

    # pair of eigs
    qs = [264.6]
    pair = [['8','10', '12']]  # pair whose eigvals cross.
    Adict5 = copy.deepcopy(Adict4)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A1 = copy.deepcopy(Adict4['A' + pair[m][1]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict4['A' + pair[m][2]][M[m] + 1:, :])
        a1 = copy.deepcopy(Adict4['a' + pair[m][1]][M[m] + 1:])
        a2 = copy.deepcopy(Adict4['a' + pair[m][2]][M[m] + 1:])

        Adict5['A' + pair[m][1]][M[m] + 1:, :] = A2
        Adict5['A' + pair[m][2]][M[m] + 1:, :] = A1
        Adict5['a' + pair[m][1]][M[m] + 1:] = a2
        Adict5['a' + pair[m][2]][M[m] + 1:] = a1

        A0 = copy.deepcopy(Adict5['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict5['A' + pair[m][2]][M[m] + 1:, :])

        a0 = copy.deepcopy(Adict5['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict5['a' + pair[m][2]][M[m] + 1:])

        Adict5['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict5['A' + pair[m][2]][M[m] + 1:, :] = A0
        Adict5['a' + pair[m][0]][M[m] + 1:] = a2
        Adict5['a' + pair[m][2]][M[m] + 1:] = a0

    # pair of eigs
    qs = [528.6125]
    pair = [['16', '18']]  # pair whose eigvals cross.
    Adict6 = copy.deepcopy(Adict5)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict5['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict5['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict5['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict5['a' + pair[m][1]][M[m] + 1:])

        Adict6['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict6['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict6['a' + pair[m][0]][M[m] + 1:] = a2
        Adict6['a' + pair[m][1]][M[m] + 1:] = a0

    # last pair of eigs
    qs = [671.5]
    pair = [['16','18', '20']]  # pair whose eigvals cross.
    Adict7 = copy.deepcopy(Adict6)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A1 = copy.deepcopy(Adict6['A' + pair[m][1]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict6['A' + pair[m][2]][M[m] + 1:, :])
        a1 = copy.deepcopy(Adict6['a' + pair[m][1]][M[m] + 1:])
        a2 = copy.deepcopy(Adict6['a' + pair[m][2]][M[m] + 1:])

        Adict7['A' + pair[m][1]][M[m] + 1:, :] = A2
        Adict7['A' + pair[m][2]][M[m] + 1:, :] = A1
        Adict7['a' + pair[m][1]][M[m] + 1:] = a2
        Adict7['a' + pair[m][2]][M[m] + 1:] = a1

        A0 = copy.deepcopy(Adict7['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict7['A' + pair[m][2]][M[m] + 1:, :])

        a0 = copy.deepcopy(Adict7['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict7['a' + pair[m][2]][M[m] + 1:])

        Adict7['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict7['A' + pair[m][2]][M[m] + 1:, :] = A0
        Adict7['a' + pair[m][0]][M[m] + 1:] = a2
        Adict7['a' + pair[m][2]][M[m] + 1:] = a0

    return Adict7



def reorder_gauss(Avals, Q):
    """Changes the ordering of the eigenvectors and eigenvalues that are stored
    within a dictionary, whenever the value of canonical parameter q lies
    between an interval. This represents the case of a narrow gaussian jet.
    """
    # first mode, asymptotes to n=2
    qs = [17.15, 55.3, 162, 434.85, 988.1, 
          1991.1, 3655.9, 6240.7, 10070.]

    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(abs(Q.imag) <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(M) - 1):
        A2 = copy.deepcopy(Adict1['A2'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :])
        a2 = copy.deepcopy(Adict1['a2'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 2))][M[m] + 1:])
        Adict1['A2'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :] = A2
        Adict1['a2'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 2))][M[m] + 1:] = a2

    # second mode, asymptotes to 2n=18
    qs = [348.15, 470.9, 635.75, 866.8, 1164.5, 1538.3, 2008.3, 2589.6]
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(abs(Q.imag) <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A18 = copy.deepcopy(Adict2['A18'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict2['A' + str(2 * (m + 10))][M[m] + 1:, :])
        a18 = copy.deepcopy(Adict2['a18'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict2['a' + str(2 * (m + 10))][M[m] + 1:])
        Adict2['A18'][M[m] + 1:, :] = Am
        Adict2['A' + str(2 * (m + 10))][M[m] + 1:, :] = A18
        Adict2['a18'][M[m] + 1:] = am
        Adict2['a' + str(2 * (m + 10))][M[m] + 1:] = a18

    # third mode, asymptotes to 2n=32
    qs = [972.375, 1158.9, 1376.65, 1645., 1975.]

    Adict = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(abs(Q.imag) <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A32 = copy.deepcopy(Adict['A32'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict['A' + str(2 * (m + 17))][M[m] + 1:, :])
        a32 = copy.deepcopy(Adict['a32'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict['a' + str(2 * (m + 17))][M[m] + 1:])
        Adict['A32'][M[m] + 1:, :] = Am
        Adict['A' + str(2 * (m + 17))][M[m] + 1:, :] = A32
        Adict['a32'][M[m] + 1:] = am
        Adict['a' + str(2 * (m + 17))][M[m] + 1:] = a32

    return Adict


def reorder_gauss2(Avals, Q):
    """Changes the ordering of the eigenvectors and eigenvalues that are stored
    within a dictionary, whenever the value of canonical parameter q lies
    between an interval. This represents the case of a intermediate (Ld=0.5) gaussian jet.
    """
    # first mode, asymptotes to 2n=2
    qs = [42.8, 628, 3852.7, ]
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A2 = copy.deepcopy(Adict1['A2'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :])
        a2 = copy.deepcopy(Adict1['a2'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 2))][M[m] + 1:])
        Adict1['A2'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :] = A2
        Adict1['a2'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 2))][M[m] + 1:] = a2

    # second mode, asymptotes to 2n=10
    qs = [167.15, 393.5, 848.15, 1690.7, 3141.5, 5493.6, 9126.9]

    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A10 = copy.deepcopy(Adict2['A10'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict2['A' + str(2 * (m + 6))][M[m] + 1:, :])
        a10 = copy.deepcopy(Adict2['a10'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict2['a' + str(2 * (m + 6))][M[m] + 1:])
        Adict2['A10'][M[m] + 1:, :] = Am
        Adict2['A' + str(2 * (m + 6))][M[m] + 1:, :] = A10
        Adict2['a10'][M[m] + 1:] = am
        Adict2['a' + str(2 * (m + 6))][M[m] + 1:] = a10

    # third mode, asymptotes to 2n=16
    qs = [280.3, 439.2, 721.7, 1152.2, 1794.9, 2727.3, 4043.1]
    Adict3 = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A16 = copy.deepcopy(Adict3['A16'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict3['A' + str(2 * (m + 9))][M[m] + 1:, :])
        a16 = copy.deepcopy(Adict3['a16'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict3['a' + str(2 * (m + 9))][M[m] + 1:])
        Adict3['A16'][M[m] + 1:, :] = Am
        Adict3['A' + str(2 * (m + 9))][M[m] + 1:, :] = A16
        Adict3['a16'][M[m] + 1:] = am
        Adict3['a' + str(2 * (m + 9))][M[m] + 1:] = a16

    # third mode, asymptotes to 2n=24
    qs = [611.5, 852.7, 1200.6, 1675.9, 2314.8]
    Adict4 = copy.deepcopy(Adict3)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A24 = copy.deepcopy(Adict4['A24'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict4['A' + str(2 * (m + 13))][M[m] + 1:, :])
        a24 = copy.deepcopy(Adict4['a24'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict4['a' + str(2 * (m + 13))][M[m] + 1:])
        Adict4['A24'][M[m] + 1:, :] = Am
        Adict4['A' + str(2 * (m + 13))][M[m] + 1:, :] = A24
        Adict4['a24'][M[m] + 1:] = am
        Adict4['a' + str(2 * (m + 13))][M[m] + 1:] = a24

    return Adict4


def reorder_gauss3(Avals, Q):
    """Changes the ordering of pairs of eigenvalues and eigenvectors that are stored within a
    dictionary. This is associated with wide gaussian jet (Ld=1.5).
    """
    qs = [189.028528, 498.264264, 777.692692, 765.746746, 999.832832, 3124]
    pair = [['10', '12'], ['6', '8'], ['20', '22'], ['24', '26'], ['16', '18'],
            ['10', '14']]  # pair whose eigvals cross.
    Adict = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict['a' + pair[m][1]][M[m] + 1:])

        Adict['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict['a' + pair[m][0]][M[m] + 1:] = a2
        Adict['a' + pair[m][1]][M[m] + 1:] = a0
    return Adict


def reorder_quad(Avals, Q):
    """ changes the ordering"""

    # # first mode, asymptotes to 2n=2
    qs = [1650.3]
    Adict = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A2 = copy.deepcopy(Adict['A2'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict['A' + str(2 * (m + 2))][M[m] + 1:, :])
        a2 = copy.deepcopy(Adict['a2'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict['a' + str(2 * (m + 2))][M[m] + 1:])
        Adict['A2'][M[m] + 1:, :] = Am
        Adict['A' + str(2 * (m + 2))][M[m] + 1:, :] = A2
        Adict['a2'][M[m] + 1:] = am
        Adict['a' + str(2 * (m + 2))][M[m] + 1:] = a2


    # # first mode, asymptotes to 2n=6
    qs = [121.4, 703.1]
    Adict0 = copy.deepcopy(Adict)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A6 = copy.deepcopy(Adict0['A6'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict0['A' + str(2 * (m + 4))][M[m] + 1:, :])
        a6 = copy.deepcopy(Adict0['a6'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict0['a' + str(2 * (m + 4))][M[m] + 1:])
        Adict0['A6'][M[m] + 1:, :] = Am
        Adict0['A' + str(2 * (m + 4))][M[m] + 1:, :] = A6
        Adict0['a6'][M[m] + 1:] = am
        Adict0['a' + str(2 * (m + 4))][M[m] + 1:] = a6


    # 2n mode, asymptotes to 2n=12
    qs = [496.7] # [119, 687.875]
    Adict1 = copy.deepcopy(Adict0)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A12 = copy.deepcopy(Adict1['A12'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 7))][M[m] + 1:, :])
        a12 = copy.deepcopy(Adict1['a12'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 7))][M[m] + 1:])
        Adict1['A12'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 7))][M[m] + 1:, :] = A12
        Adict1['a12'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 7))][M[m] + 1:] = a12


    # 2n mode, asymptotes to 2n=12
    qs = [] #[ 488, 715.875, 757.125]
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A12 = copy.deepcopy(Adict2['A12'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict2['A' + str(2 * (m + 7))][M[m] + 1:, :])
        a12 = copy.deepcopy(Adict2['a12'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict2['a' + str(2 * (m + 7))][M[m] + 1:])
        Adict2['A12'][M[m] + 1:, :] = Am
        Adict2['A' + str(2 * (m + 7))][M[m] + 1:, :] = A12
        Adict2['a12'][M[m] + 1:] = am
        Adict2['a' + str(2 * (m + 7))][M[m] + 1:] = a12

    # 3rd mode, asymptotes to 2n=16
    qs = [535.3] #[560.625, 622.375]
    Adict3 = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A16 = copy.deepcopy(Adict3['A16'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict3['A' + str(2 * (m + 9))][M[m] + 1:, :])
        a16 = copy.deepcopy(Adict3['a16'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict3['a' + str(2 * (m + 9))][M[m] + 1:])
        Adict3['A16'][M[m] + 1:, :] = Am
        Adict3['A' + str(2 * (m + 9))][M[m] + 1:, :] = A16
        Adict3['a16'][M[m] + 1:] = am
        Adict3['a' + str(2 * (m + 9))][M[m] + 1:] = a16
    # ## reorders between a specific pair of modes after the value qs
    # qs = [] # [3 811.5]
    # pair = [['12', '16']]
    # Adictn = copy.deepcopy(Adict4)
    # M = []
    # for k in range(len(qs)):
    #     M.append(_np.where(Q.imag <= qs[k])[0][-1])
    # M.append(len(Q))

    # for m in range(len(qs)):
    #     A0 = copy.deepcopy(Adictn['A' + pair[m][0]][M[m] + 1:, :])
    #     A2 = copy.deepcopy(Adictn['A' + pair[m][1]][M[m] + 1:, :])
    #     a0 = copy.deepcopy(Adictn['a' + pair[m][0]][M[m] + 1:])
    #     a2 = copy.deepcopy(Adictn['a' + pair[m][1]][M[m] + 1:])

    #     Adictn['A' + pair[m][0]][M[m] + 1:, :] = A2
    #     Adictn['A' + pair[m][1]][M[m] + 1:, :] = A0
    #     Adictn['a' + pair[m][0]][M[m] + 1:] = a2
    #     Adictn['a' + pair[m][1]][M[m] + 1:] = a0

    return Adict3


def reorder_cuartic(Avals, Q):
    """Changes the ordering of the eigenvectors and eigenvalues that are stored
    within a dictionary, whenever the value of canonical parameter q lies
    between an interval. This represents the case of a jet with cuartic polynomial dep.
    """
    # first mode, asymptotes to 2n=2
    qs = [46.2, 360.2, 1390.3, 3833.5, 8776.3]
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A2 = copy.deepcopy(Adict1['A2'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :])
        a2 = copy.deepcopy(Adict1['a2'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 2))][M[m] + 1:])
        Adict1['A2'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :] = A2
        Adict1['a2'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 2))][M[m] + 1:] = a2

    # 2nd mode, 2n=8
    qs = [138.5, 261.6, 501.6, 874.6, 1431, 2226.76, 3365, 3865.4]
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A8 = copy.deepcopy(Adict2['A8'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict2['A' + str(2 * (m + 5))][M[m] + 1:, :])
        a8 = copy.deepcopy(Adict2['a8'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict2['a' + str(2 * (m + 5))][M[m] + 1:])
        Adict2['A8'][M[m] + 1:, :] = Am
        Adict2['A' + str(2 * (m + 5))][M[m] + 1:, :] = A8
        Adict2['a8'][M[m] + 1:] = am
        Adict2['a' + str(2 * (m + 5))][M[m] + 1:] = a8

    # 3nd mode, 2n=14
    qs = [336.41, 484.3, 740.9]
    Adict3 = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A14 = copy.deepcopy(Adict3['A14'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict3['A' + str(2 * (m + 8))][M[m] + 1:, :])
        a14 = copy.deepcopy(Adict3['a14'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict3['a' + str(2 * (m + 8))][M[m] + 1:])
        Adict3['A14'][M[m] + 1:, :] = Am
        Adict3['A' + str(2 * (m + 8))][M[m] + 1:, :] = A14
        Adict3['a14'][M[m] + 1:] = am
        Adict3['a' + str(2 * (m + 8))][M[m] + 1:] = a14

    # 4nd mode, 2n=22
    qs = [818.]
    Adict4 = copy.deepcopy(Adict3)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A22 = copy.deepcopy(Adict4['A22'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict4['A' + str(2 * (m + 12))][M[m] + 1:, :])
        a22 = copy.deepcopy(Adict4['a22'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict4['a' + str(2 * (m + 12))][M[m] + 1:])
        Adict4['A22'][M[m] + 1:, :] = Am
        Adict4['A' + str(2 * (m + 12))][M[m] + 1:, :] = A22
        Adict4['a22'][M[m] + 1:] = am
        Adict4['a' + str(2 * (m + 12))][M[m] + 1:] = a22


    # ## reorders between a specific pair of modes after the value qs
    # qs = [740.875]
    # pair = [['16', '20']]
    # Adict4 = copy.deepcopy(Adict3)
    # M = []
    # for k in range(len(qs)):
    #     M.append(_np.where(Q.imag <= qs[k])[0][-1])
    # M.append(len(Q))

    # for m in range(len(qs)):
    #     A0 = copy.deepcopy(Adict4['A' + pair[m][0]][M[m] + 1:, :])
    #     A2 = copy.deepcopy(Adict4['A' + pair[m][1]][M[m] + 1:, :])
    #     a0 = copy.deepcopy(Adict4['a' + pair[m][0]][M[m] + 1:])
    #     a2 = copy.deepcopy(Adict4['a' + pair[m][1]][M[m] + 1:])

    #     Adict4['A' + pair[m][0]][M[m] + 1:, :] = A2
    #     Adict4['A' + pair[m][1]][M[m] + 1:, :] = A0
    #     Adict4['a' + pair[m][0]][M[m] + 1:] = a2
    #     Adict4['a' + pair[m][1]][M[m] + 1:] = a0

    return Adict4


def reorder_sixth(Avals, Q):
    """Changes the ordering of the eigenvectors and eigenvalues that are stored
    within a dictionary, whenever the value of canonical parameter q lies
    between an interval. This represents the case of a jet with cuartic polynomial dep.
    """
    # first mode, asymptotes to 2n=2
    qs = [29.1, 130.1, 404.4, 965, 1948., 3516.2, 5910, 9511.3]
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A2 = copy.deepcopy(Adict1['A2'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :])
        a2 = copy.deepcopy(Adict1['a2'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 2))][M[m] + 1:])
        Adict1['A2'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :] = A2
        Adict1['a2'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 2))][M[m] + 1:] = a2

    # 2n mode, asymptotes to 2n=10
    qs = [199.08,  301.125, 459.5, 691.5, 992.125, 1376.5, 1865.1, 2432.7]
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A10 = copy.deepcopy(Adict2['A10'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict2['A' + str(2 * (m + 6))][M[m] + 1:, :])
        a10 = copy.deepcopy(Adict2['a10'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict2['a' + str(2 * (m + 6))][M[m] + 1:])
        Adict2['A10'][M[m] + 1:, :] = Am
        Adict2['A' + str(2 * (m + 6))][M[m] + 1:, :] = A10
        Adict2['a10'][M[m] + 1:] = am
        Adict2['a' + str(2 * (m + 6))][M[m] + 1:] = a10


    # 3n mode, asymptotes to 2n=18
    qs = [555.125, 710.4, 895., 1194.5, 1489.7, 1844.1]
    Adict3 = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A18 = copy.deepcopy(Adict3['A18'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict3['A' + str(2 * (m + 10))][M[m] + 1:, :])
        a18 = copy.deepcopy(Adict3['a18'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict3['a' + str(2 * (m + 10))][M[m] + 1:])
        Adict3['A18'][M[m] + 1:, :] = Am
        Adict3['A' + str(2 * (m + 10))][M[m] + 1:, :] = A18
        Adict3['a18'][M[m] + 1:] = am
        Adict3['a' + str(2 * (m + 10))][M[m] + 1:] = a18

    return Adict3


def reorder_tenth(Avals, Q):
    """Changes the ordering of the eigenvectors and eigenvalues that are stored
    within a dictionary, whenever the value of canonical parameter q lies
    between an interval. This represents the case of a jet with cuartic polynomial dep.
    """
    # first mode, asymptotes to 2n=2
    qs = [23.9, 71.125, 167.4, 362.9, 661.875,
          1109.2, 1734.7, 2580.4, 3707.3, 5215.8, 
          7097.6, 8818.5, 10786.7]
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A2 = copy.deepcopy(Adict1['A2'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :])
        a2 = copy.deepcopy(Adict1['a2'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 2))][M[m] + 1:])
        Adict1['A2'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 2))][M[m] + 1:, :] = A2
        Adict1['a2'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 2))][M[m] + 1:] = a2

    # 2nd mode, asymptotes to 2n=14
    qs = [368.8, 484.4, 620.25, 782.65, 1019.8, 1296., 1578.95, 
          1939.4, 2190.7, 2371.8]
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A14 = copy.deepcopy(Adict2['A14'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict2['A' + str(2 * (m + 8))][M[m] + 1:, :])
        a14 = copy.deepcopy(Adict2['a14'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict2['a' + str(2 * (m + 8))][M[m] + 1:])
        Adict2['A14'][M[m] + 1:, :] = Am
        Adict2['A' + str(2 * (m + 8))][M[m] + 1:, :] = A14
        Adict2['a14'][M[m] + 1:] = am
        Adict2['a' + str(2 * (m + 8))][M[m] + 1:] = a14

    return Adict2


def reorder_sqr3(Avals, Q):
    """Changes the ordering of the eigenvectors and eigenvalues that are stored
    within a dictionary, whenever the value of canonical parameter q lies
    between an interval. This represents the case of a jet with cuartic polynomial dep.
    """
    # first mode, asymptotes to 2n=4
    qs = [38.2, 137.5, 2723.7]
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A4 = copy.deepcopy(Adict1['A4'][M[m] + 1:, :])  # anomalous mode
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 3))][M[m] + 1:, :])
        a4 = copy.deepcopy(Adict1['a4'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 3))][M[m] + 1:])
        Adict1['A4'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 3))][M[m] + 1:, :] = A4
        Adict1['a4'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 3))][M[m] + 1:] = a4


    # 2n mode, asymptotes to 2n=24
    qs = [237.35, 666.4]
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A24 = copy.deepcopy(Adict2['A24'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict2['A' + str(2 * (m + 13))][M[m] + 1:, :])
        a24 = copy.deepcopy(Adict2['a24'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict2['a' + str(2 * (m + 13))][M[m] + 1:])
        Adict2['A24'][M[m] + 1:, :] = Am
        Adict2['A' + str(2 * (m + 13))][M[m] + 1:, :] = A24
        Adict2['a24'][M[m] + 1:] = am
        Adict2['a' + str(2 * (m + 13))][M[m] + 1:] = a24

    # 3rd mode, asymptotes to 2n=46

    qs = [] #[931.]
    Adict3 = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A46 = copy.deepcopy(Adict3['A46'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict3['A' + str(2 * (m + 24))][M[m] + 1:, :])
        a46 = copy.deepcopy(Adict3['a46'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict3['a' + str(2 * (m + 24))][M[m] + 1:])
        Adict3['A46'][M[m] + 1:, :] = Am
        Adict3['A' + str(2 * (m + 24))][M[m] + 1:, :] = A46
        Adict3['a46'][M[m] + 1:] = am
        Adict3['a' + str(2 * (m + 24))][M[m] + 1:] = a46

    return Adict3


def reorder_sqr4(Avals, Q):
    """Changes the ordering of pairs of eigenvalues and eigenvectors that are stored within a
    dictionary. This is associated with triangular jet (Ld=1.5).
    """
    # pair of eigs
    qs = [8.2]
    pair = [['0', '2']]  # pair whose eigvals cross.
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Avals['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Avals['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Avals['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Avals['a' + pair[m][1]][M[m] + 1:])

        Adict1['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict1['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict1['a' + pair[m][0]][M[m] + 1:] = a2
        Adict1['a' + pair[m][1]][M[m] + 1:] = a0

    # next pair of eigs (EP at 14.5 between the two eigs)
    qs = [29.4]  #
    pair = [['0','2', '4']]  # pair whose eigvals cross.
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict1['A' + pair[m][0]][M[m] + 1:, :])
        A1 = copy.deepcopy(Adict1['A' + pair[m][1]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict1['A' + pair[m][2]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict1['a' + pair[m][0]][M[m] + 1:])
        a1 = copy.deepcopy(Adict1['a' + pair[m][1]][M[m] + 1:])
        a2 = copy.deepcopy(Adict1['a' + pair[m][2]][M[m] + 1:])

        Adict2['A' + pair[m][1]][M[m] + 1:, :] = A2
        Adict2['A' + pair[m][2]][M[m] + 1:, :] = A1
        Adict2['a' + pair[m][1]][M[m] + 1:] = a2
        Adict2['a' + pair[m][2]][M[m] + 1:] = a1

        A0 = copy.deepcopy(Adict2['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict2['A' + pair[m][2]][M[m] + 1:, :])

        a0 = copy.deepcopy(Adict2['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict2['a' + pair[m][2]][M[m] + 1:])

        Adict2['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict2['A' + pair[m][2]][M[m] + 1:, :] = A0
        Adict2['a' + pair[m][0]][M[m] + 1:] = a2
        Adict2['a' + pair[m][2]][M[m] + 1:] = a0

    qs = [53.8875]
    pair = [['8', '10']]  # pair whose eigvals cross.
    Adict3 = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict2['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict2['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict2['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict2['a' + pair[m][1]][M[m] + 1:])

        Adict3['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict3['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict3['a' + pair[m][0]][M[m] + 1:] = a2
        Adict3['a' + pair[m][1]][M[m] + 1:] = a0


    qs = [113.645]
    pair = [['16', '18']]  # pair whose eigvals cross.
    Adict4 = copy.deepcopy(Adict3)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict3['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict3['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict3['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict3['a' + pair[m][1]][M[m] + 1:])

        Adict4['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict4['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict4['a' + pair[m][0]][M[m] + 1:] = a2
        Adict4['a' + pair[m][1]][M[m] + 1:] = a0

    qs = [180.813]
    pair = [['24', '26']]  # pair whose eigvals cross.
    Adict5 = copy.deepcopy(Adict4)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict4['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict4['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict4['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict4['a' + pair[m][1]][M[m] + 1:])

        Adict5['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict5['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict5['a' + pair[m][0]][M[m] + 1:] = a2
        Adict5['a' + pair[m][1]][M[m] + 1:] = a0

    return Adict5


def reorder_linear5(Avals, Q):
    """Changes the ordering of pairs of eigenvalues and eigenvectors that are stored within a
    dictionary. This is associated with triangular jet (Ld=1.5).
    """
    # pair of eigs
    qs = [29.8]
    pair = [['0', '4']]  # pair whose eigvals cross.
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Avals['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Avals['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Avals['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Avals['a' + pair[m][1]][M[m] + 1:])

        Adict1['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict1['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict1['a' + pair[m][0]][M[m] + 1:] = a2
        Adict1['a' + pair[m][1]][M[m] + 1:] = a0

    # pair of eigs
    qs = [29.9]
    pair = [['2', '4']]  # pair whose eigvals cross.
    Adict2 = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict1['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict1['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict1['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict1['a' + pair[m][1]][M[m] + 1:])

        Adict2['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict2['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict2['a' + pair[m][0]][M[m] + 1:] = a2
        Adict2['a' + pair[m][1]][M[m] + 1:] = a0

    # pair of eigs
    qs = [398.189]
    pair = [['12', '16']]  # pair whose eigvals cross.
    Adict3 = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adict2['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adict2['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adict2['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adict2['a' + pair[m][1]][M[m] + 1:])

        Adict3['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict3['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict3['a' + pair[m][0]][M[m] + 1:] = a2
        Adict3['a' + pair[m][1]][M[m] + 1:] = a0

    return Adict3


def reorder_sqr5(Avals, Q):
    """Changes the ordering of pairs of eigenvalues and eigenvectors that are stored within a
    dictionary. This is associated with triangular jet (Ld=1.5).
    """
    # pair of eigs
    qs = [18.973]
    pair = [['0', '4']]  # pair whose eigvals cross.
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Avals['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Avals['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Avals['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Avals['a' + pair[m][1]][M[m] + 1:])

        Adict1['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adict1['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adict1['a' + pair[m][0]][M[m] + 1:] = a2
        Adict1['a' + pair[m][1]][M[m] + 1:] = a0

    return Adict1


def phi_array(N, K, symmetry='even', y = 0, coeffs=True, eigs=True, phis=False):
    """creates an xarray with eigenvalues and eigenfunctions associated with the limit q=0.
    We use these values as the complement to the truncated matrix calculation to optimize code.
    
    Input:
        N: total number of cosine modes required to, say, accurately approximate a Gaussian.
        y: 
    """
    if symmetry=='even':
        _eigv = 'A_2r'
        _eigs = 'a_2n'
        _r0 = 0
    elif symmetry=='odd':
        _eigv = 'B_2r'
        _eigs = 'b_2n'
        _r0 = 1
    if coeffs:
        A_coords =  {"n": range(N-_r0), "k":K, 'r':range(N-_r0)} 
        A_2r = _xr.DataArray((1j)*0, coords=A_coords, dims=['n', 'k', 'r'])
    if eigs:
        n_coords =  {"n": range(N-_r0), "k":K, }
        a_2n = _xr.DataArray((1j)*0, coords=n_coords, dims=['n', 'k'])
    if phis:
        phi_coords = {"n": range(N-_r0), "y": y}
        phi_2n = _xr.DataArray((1j)*0, coords=phi_coords, dims=['n', 'y'])
    for n in range(N-_r0):
        if coeffs:
            A_2r.isel(n=n, r=n).data[:] = _np.ones(_np.shape(K))
        if eigs:
            if symmetry == 'even':
                _vals = (2 * n)**2
            elif symmetry == 'odd':
                _vals = (2 * (n+1))**2
            a_2n.isel(n=n).data[:] = (_vals * _np.ones(_np.shape(K)))
        if phis:
            if symmetry == 'even':
                phi_2n.isel(n=n).data[:] = _np.cos(2*n*y)
            elif symmetry == "odd":
                phi_2n.isel(n=n).data[:] = _np.sin(2*(n+1)*y)
    data_vars = {}
    if coeffs:
        data_vars = {**data_vars, **{_eigv: A_2r}}
    if eigs:
        data_vars = {**data_vars, **{_eigs: a_2n}}
    if phis:
        data_vars = {**data_vars, **{'phi_2n': phi_2n}}
    eig_fns =  _xr.Dataset(data_vars)
    if symmetry == 'even':
        eig_fns['A_2r'].isel(n=0, r=0).data[:] = (1/_np.sqrt(2)) * _np.ones(_np.shape(K))
    return eig_fns


def complement_dot(_gauss_alps, _ds_As):
    """complement the dot product when calculating solutions. This is the case where initial condition 
    is localized in y.
    Input
        _gauss_alps : (1+delta_p0)gamma_p in theory.

    """
    if 'A_2r' in _ds_As.data_vars:
        _eigv = 'A_2r'
    elif 'B_2r' in _ds_As.data_vars:
        _eigv = 'B_2r'
    _Nr = len(_ds_As.r)  # this is the truncated array. 
    _Nt = len(_gauss_alps)  # total length - set at the initial condition.
    _K = _ds_As['k'].data
    _ndAs = _xr.dot(_gauss_alps.isel(r=slice(_Nr)), _ds_As[_eigv], dims='r').compute()
    if _Nr < _Nt:
        _coeffs_alps = copy.deepcopy(_gauss_alps[_Nr:])
        _coeffs_alps = _coeffs_alps.rename({'r':'n'})
        if 'k' not in _coeffs_alps.dims:
            _coeffs_alps = _coeffs_alps.expand_dims({'k':_K})
        _coeffs_alps = _ndAs.combine_first(_coeffs_alps)
    else:
        _coeffs_alps = _ndAs
    return _coeffs_alps

def ragged_sum(_datasets, _gauss_alps, alpha0, Pe, tk):
    """Computed the double sum (in p and n) necessary for calculating the (averaged) analytical solution. 
    Returns a dataarray with only one dimension: k spanning both positive and negative values.
    There is a factor of `2 * np.pi` when calculating the exponential term. This is because 
    it is used with xrft which normalizes the wavenumber differently-hence the need to always
    multiply K by a factor of 2 * np.pi. 
    Input:
        datasets: list of datasets spanning a subset of wavenumbers k>0. The dimensions in p and r
    """

    if 'A_2r' in _ds_As.data_vars:
        _eigv = 'A_2r'
        _eigs = 'a_2n'
    elif 'B_2r' in _ds_As.data_vars:
        _eigv = 'B_2r' 
        _eigs = 'b_2n'

    PHI2n = []
    for i in range(len(_datasets)):
        Ki = _datasets[i].k.data
        NR = len(_datasets[i].n)
        exp_arg = (1j) * alpha0*(2 *_np.pi *Ki)*Pe + (2 * _np.pi *Ki)**2
        ndAs = complement_dot(_gauss_alps, _datasets[i]).compute()  # has final size in n (sum in p)
        A_0 = _datasets[i][_eigv].isel(r=0, n=slice(NR))
        a_2n = _datasets[i][_eigs].isel(n=slice(NR))
        phi2n = _xr.dot(ndAs.isel(n=slice(NR)), A_0 * _np.exp(-(0.25*a_2n+ exp_arg)*tk), dims='n').compute()
        if i==0:
            PHI2n = phi2n
        else:
            PHI2n = PHI2n.combine_first(phi2n)  # upto here, k>0 is considered.
    
    # using symmetry arguments, expand into k<0
    phi_2n = _xr.ones_like(PHI2n)
    phi_2n['k'] = -phi_2n['k'][::-1]
    phi_2n.data = PHI2n.conjugate()[::-1]
    
    PHI = phi_2n.combine_first(PHI2n)

    return PHI


def reflect_dataset(ds, k=True, Pe=False, symmetry='even'):
    """ Reflects a dataset along the q-axis. This reflection can be due to a change in sign of the 
    velocity field (or change in Pe), or a change in sign of wavenumber k>=0. This function skips
    the eigenvalue calculation. 
    Input:
        ds: dataset.
        k: bool. `True` by default. Means the dataset must be complemented for values with k<0. This
            means the dataset must be complemented and updated before returning.
        Pe: bool. `False` by default. When `True`, the complete dataset must be reflected for all values
            of `k`. The returned dataset
    """
    if symmetry is 'even':
        _eigs = 'a_2n'
        _eigv = 'A_2r'
    else:
        _eigs = 'b_2n'
        _eigv = 'B_2r'
    if k:
        As_dsc = _xr.ones_like(ds)
        As_dsc['nk'] =  -As_dsc['k'].data[::-1]
        As_dsc = As_dsc.drop_dims('k').rename({'nk':'k'})
        As_dsc[_eigs] = _xr.ones_like(ds[_eigs])
        As_dsc[_eigv] = _xr.ones_like(ds[_eigv])
        As_dsc[_eigs].data = ds.conjugate()[_eigs][:, ::-1]
        As_dsc[_eigv].data = ds.conjugate()[_eigv][:, ::-1, :]

        nds = As_dsc.combine_first(ds)  # combine along k values.
    elif Pe:
        nds = _xr.ones_like(ds.drop_vars('phi_2n'))
        nds[_eigs].data = ds.conjugate()[_eigs][:]
        nds[_eigv].data = ds.conjugate()[_eigv][:]

    return nds



def ds_spectralist(_Kn, _vals, _Pe, _N, _betas_m, _Km, _y, both=True):
    """Creates a list of datasets in which each element contains the spectra of the governing operator.
    """
    mDAS = []
    nDAS = []
    mDBS = []
    nDAS = []
    nDBS = []
    mALPHA0 = []
    nALPHA0 = []
    nval = []
    mval = []
    ll = _np.where(_np.array(_vals)==0)[0][0]

    for i in _vals[ll:]:
        ds_As = A_coefficients(_Kn, abs(i*_Pe), _N, _np.sign(i)*_np.array(_betas_m), _Km, symmetry='even', opt=True, reflect=True)
        ds_As = phi_even(_Kn, abs(i*_Pe), _y, _N, _np.sign(i)*_np.array(_betas_m), _Km, dAs = ds_As)
        mDAS.append(copy.deepcopy(ds_As))
        if both:
            ds_Bs = A_coefficients(_Kn, abs(i*_Pe), _N, _np.sign(i)*_np.array(_betas_m), _Km, symmetry='odd', opt=True, reflect=True)
            ds_Bs = phi_odd(_Kn, abs(i*_Pe), _y, _N, _np.sign(i)*_np.array(_betas_m), _Km, dBs = ds_Bs)
            mDBS.append(copy.deepcopy(ds_Bs))

        mALPHA0.append(i*alpha0_gauss2) 
        mval.append(i)
        if i > 0:
            nds_As = reflect_dataset(ds_As, k=False, Pe=True, symmetry = 'even')
            nds_As = phi_even(Kn, abs(i*Pe), y, _N, -_np.sign(i)*_np.array(_betas_m), _Km, dAs = nds_As)
            nDAS.append(copy.deepcopy(nds_As))
            if both:
                nds_Bs = reflect_dataset(ds_Bs, k=False, Pe=True, symmetry = 'odd')
                nds_Bs = phi_odd(Kn, abs(i*Pe), y, _N, -_np.sign(i)*_np.array(_betas_m), _Km, dBs = nds_Bs)
                nDBS.append(copy.deepcopy(nds_Bs))
                nds_Bs = 0
            nALPHA0.append(-i*alpha0_gauss2)
            nds_As = 0
            nval.append(-i)
        ds_As = 0
        ds_Bs = 0

    DAS = nDAS[::-1] + mDAS
    if both:
        DBS = nDBS[::-1] + mDBS
    ALPHA0 = nALPHA0[::-1] + mALPHA0
    vals = nval[::-1] + mval
    return DAS, DBS, ALPHA0, vals



