"""
defines a class of eigenfunctions for Hill's equation
"""

import numpy as _np
import copy
from eig_system import matrix_system, eig_pairs
import copy as _copy


class eigenfunctions:

    def __init__(self, q, N, coeffs, K, symmetry, case):
        self._q = q
        self._N = N
        self._coeffs = coeffs
        self._K = K
        self._symmetry = symmetry
        self._case = case


    @classmethod
    def phi_even(
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
            As = A_coefficients(q, N, coeffs, K, symmetry)
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
    def phi_odd(
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
            Bs = A_coefficients(q, N, coeffs, K, symmetry)
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


def A_coefficients(q, N, coeffs, K, symmetry='even', case='None'):
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
    vals = {}
    if len(q)>1:
        if q.imag.any() == 0:
            raise Warning("q must be imaginary")
    else:
        if q.imag == 0:
            raise Warning("q must be imaginary")
    for n in range(N):
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
        if case not in ['gaussian', 'gaussian3']:
            As = Fcoeffs(As, n, q, case)
        if symmetry in ['None', 'even']:
            vals.update({'a' + str(2 * n): _np.array(a)})
            vals.update({'A' + str(2 * n): As})
        elif symmetry is 'odd':
            vals.update({'b' + str(2 * (n + 1)): _np.array(a)})
            vals.update({'B' + str(2 * (n + 1)): As})
    if case in ['linear', 'asine', 'gaussian', 'gaussian2', 'gaussian3', 'quad']:
        if case == 'linear':
            vals = reorder_linear(vals, q)
        elif case == 'asine':
            vals = reorder_asine(vals, q)
        elif case == 'quad':
            vals = reorder_quad(vals, q)
        elif case == 'gaussian':  # narrow gaussian
            vals = reorder_gauss(vals, q)
        elif case == 'gaussian2':
            vals = reorder_gauss2(vals, q)
        elif case == 'gaussian3':  # wide gaussian
            vals = reorder_gauss3(vals, q)
        for n in range(N):
            As = copy.deepcopy(vals['A' + str(2 * n)])
            # As = Fcoeffs(As, n, q, case)
            vals.update({'A' + str(2 * n): As})
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
    if case is 'linear':
        As = linCoeffs(As, n, q)
    elif case is 'linear0':
        As = lin0Coeffs(As, n, q)
    elif case is 'linear2':
        As = linCoeffs2(As, n, q)
    elif case is 'nlinear':
        As = linnCoeffs(As, n, q)
    elif case is 'square':
        As = sqrCoeffs(As, n, q)
    elif case is 'step':  # sine flow, associated with odd symmetry
        As = stepCoeffs(As, n, q)
    elif case is 'gaussian':
        As = gaussCoeffs(As, n, q)
    elif case is 'gaussian3':
        As = gauss3Coeffs(As, n, q)
    elif case is 'cosine':
        As = cCoeffs(As, n, q)
    elif case is 'cosine_shifted':
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
    if n in [2, 3] and q[0].imag < qs[1]:
        if q.imag[-1] > qs[1]:
            ll = _np.where(q.imag <= qs[1])[0]
            if n == 2:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
                # for k in range(N):
                #     if k % 2 == 0:
                #         A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                #     else:
                #         A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 3:
                mm = _np.where(A[:ll[-1] + 1, n].real < 0)[0]  # should be >0
                A[mm, :] = -A[mm, :]
                m0 = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[m0 + ll[-1] + 1, :] = -A[m0 + ll[-1] + 1, :]
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


def gaussCoeffs(As, n, q):
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


def gauss3Coeffs(As, n, q):
    '''Correct behavior of Fourier coefficients as a function of q-parameter. This case is
    associated with the wide gaussian jet (Ld=1.5). Only some eigenvalue pairs cross.
    ''' 
    qs = [206.551551, 304.801801, 805.732732, 939.888888]
    pair = [['10', '12'], ['6', '8'], ['20', '22'], ['16', '18']]  # pair whose eigvals cross.
    if n == 1:
        lll = _np.where(As[:, 0].real > 0)[0]  # choosing (trial and error) to be neg
        As[lll, :] = -As[lll, :]
    elif n == 3:
        lll = _np.where(As[:, 0].real < 0)[0]  # choosing (trial and error) to be neg
        As[lll, :] = -As[lll, :]
    elif n == 5:
        lll = _np.where(As[:, 0].real > 0)[0]
        As[lll, :] = -As[lll, :]
    elif n == 8:
        lll = _np.where(As[:, 0].real > 0)[0]  # pattern suggest otherwise, but graph this
        As[lll, :] = -As[lll, :]
    elif n == 10:
        lll = _np.where(As[:, 0].real < 0)[0]  # pattern suggest otherwise, but graph this
        As[lll, :] = -As[lll, :]
    elif n == 12:
        lll = _np.where(As[:, 0].real > 0)[0]  # pattern suggest otherwise, but graph this
        As[lll, :] = -As[lll, :]
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


def reorder_gauss(Avals, Q):
    """Changes the ordering of the eigenvectors and eigenvalues that are stored
    within a dictionary, whenever the value of canonical parameter q lies
    between an interval. This represents the case of a narrow gaussian jet.
    """
    # first mode, asymptotes to n=2
    qs = [46.161161, 102.157657,
          217.651651, 446.634634,
          844.616616, 1482, 2447, 3846]
    Adict1 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A4 = copy.deepcopy(Adict1['A4'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 3))][M[m] + 1:, :])
        a4 = copy.deepcopy(Adict1['a4'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 3))][M[m] + 1:])
        Adict1['A4'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 3))][M[m] + 1:, :] = A4
        Adict1['a4'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 3))][M[m] + 1:] = a4

    # second mode, asymptotes to n=24
    qs = [577.603103, 718.255255,
          888.313313, 1096.65, 1356.4,
          1675.75, 2049.875, 2486.5]
    Adict = copy.deepcopy(Adict1)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A24 = copy.deepcopy(Adict['A24'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict['A' + str(2 * (m + 13))][M[m] + 1:, :])
        a24 = copy.deepcopy(Adict['a24'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict['a' + str(2 * (m + 13))][M[m] + 1:])
        Adict['A24'][M[m] + 1:, :] = Am
        Adict['A' + str(2 * (m + 13))][M[m] + 1:, :] = A24
        Adict['a24'][M[m] + 1:] = am
        Adict['a' + str(2 * (m + 13))][M[m] + 1:] = a24
    return Adict


def reorder_gauss2(Avals, Q):
    """Changes the ordering of the eigenvectors and eigenvalues that are stored
    within a dictionary, whenever the value of canonical parameter q lies
    between an interval. This represents the case of a intermediate (Ld=0.5) gaussian jet.
    """
    # first mode, asymptotes to 2n=2
    qs = [20.236736, 108.712712,
          471.690690, 1470.5, 3680.3]
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

    # second mode, asymptotes to 2n=14
    qs = [255.095095, 410.962962,
          659.524524, 1020.25, 1532.75,
          2240, 3191.5]
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

    # third mode, asymptotes to 2n=24
    qs = [610.876876, 797.193193, 1064.625,
          1394.75, 1814.125, 2336.625,
          2981, 3767.625]
    Adict = copy.deepcopy(Adict2)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A24 = copy.deepcopy(Adict['A24'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict['A' + str(2 * (m + 13))][M[m] + 1:, :])
        a24 = copy.deepcopy(Adict['a24'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict['a' + str(2 * (m + 13))][M[m] + 1:])
        Adict['A24'][M[m] + 1:, :] = Am
        Adict['A' + str(2 * (m + 13))][M[m] + 1:, :] = A24
        Adict['a24'][M[m] + 1:] = am
        Adict['a' + str(2 * (m + 13))][M[m] + 1:] = a24
    return Adict


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


def reorder_asine(Avals, Q):
    """ changes the ordering"""
    qs = [216.46969697]
    pair = [['10', '12']]
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

    # first mode, asymptotes to 2n=2
    qs = [1647.4]
    Adict0 = copy.deepcopy(Avals)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A2 = copy.deepcopy(Adict0['A2'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict0['A' + str(2 * (m + 2))][M[m] + 1:, :])
        a2 = copy.deepcopy(Adict0['a2'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict0['a' + str(2 * (m + 2))][M[m] + 1:])
        Adict0['A2'][M[m] + 1:, :] = Am
        Adict0['A' + str(2 * (m + 2))][M[m] + 1:, :] = A2
        Adict0['a2'][M[m] + 1:] = am
        Adict0['a' + str(2 * (m + 2))][M[m] + 1:] = a2


    # 2n mode, asymptotes to 2n=6
    qs = [119, 687.875]
    Adict1 = copy.deepcopy(Adict0)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A6 = copy.deepcopy(Adict1['A6'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict1['A' + str(2 * (m + 4))][M[m] + 1:, :])
        a6 = copy.deepcopy(Adict1['a6'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict1['a' + str(2 * (m + 4))][M[m] + 1:])
        Adict1['A6'][M[m] + 1:, :] = Am
        Adict1['A' + str(2 * (m + 4))][M[m] + 1:, :] = A6
        Adict1['a6'][M[m] + 1:] = am
        Adict1['a' + str(2 * (m + 4))][M[m] + 1:] = a6


    # 2n mode, asymptotes to 2n=12
    qs = [488, 715.875, 757.125]
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
    qs = [560.625, 622.375]
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

    # 4rd mode, asymptotes to 2n=18
    qs = [651.375]
    Adict4 = copy.deepcopy(Adict3)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))
    for m in range(len(M) - 1):
        A18 = copy.deepcopy(Adict4['A18'][M[m] + 1:, :])  # anomalous mode  $ should be 4
        Am = copy.deepcopy(Adict4['A' + str(2 * (m + 10))][M[m] + 1:, :])
        a18 = copy.deepcopy(Adict4['a18'][M[m] + 1:])  # anomalous eigenvalue
        am = copy.deepcopy(Adict4['a' + str(2 * (m + 10))][M[m] + 1:])
        Adict4['A18'][M[m] + 1:, :] = Am
        Adict4['A' + str(2 * (m + 10))][M[m] + 1:, :] = A18
        Adict4['a18'][M[m] + 1:] = am
        Adict4['a' + str(2 * (m + 10))][M[m] + 1:] = a18

    ## reorders between a specific pair of modes after the value qs
    qs = [811.5]
    pair = [['12', '16']]
    Adictn = copy.deepcopy(Adict4)
    M = []
    for k in range(len(qs)):
        M.append(_np.where(Q.imag <= qs[k])[0][-1])
    M.append(len(Q))

    for m in range(len(qs)):
        A0 = copy.deepcopy(Adictn['A' + pair[m][0]][M[m] + 1:, :])
        A2 = copy.deepcopy(Adictn['A' + pair[m][1]][M[m] + 1:, :])
        a0 = copy.deepcopy(Adictn['a' + pair[m][0]][M[m] + 1:])
        a2 = copy.deepcopy(Adictn['a' + pair[m][1]][M[m] + 1:])

        Adictn['A' + pair[m][0]][M[m] + 1:, :] = A2
        Adictn['A' + pair[m][1]][M[m] + 1:, :] = A0
        Adictn['a' + pair[m][0]][M[m] + 1:] = a2
        Adictn['a' + pair[m][1]][M[m] + 1:] = a0

    return Adictn















