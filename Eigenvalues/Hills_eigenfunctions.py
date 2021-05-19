"""
defines a class of eigenfunctions for Hill's equation
"""

import numpy as _np
from eig_system import matrix_system, eig_pairs


def A_coefficients(q, N, coeffs, K, symmetry='None', case='None'):
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
    if q.imag.any() == 0:
        raise Warning("q must be imaginary")
    for n in range(N):
        a, A = eig_pairs(matrix_system(q[0], N, coeffs, K, symmetry, case))
        a = [a[n]]  # makes a list of the nth eigenvalue
        As = Anorm(A[:, n], case)
        As = As[_np.newaxis, :]
        for k in range(1, len(q)):
            an, A = eig_pairs(matrix_system(q[k], N, coeffs, K, symmetry, case))
            a.append(an[n])
            nA = Anorm(A[:, n], case=case)
            nAs = nA[_np.newaxis, :]
            As = _np.append(As, nAs, axis=0)
        As = Fcoeffs(As, n, q, case)
        if symmetry in ['None', 'even']:
            vals.update({'a' + str(2 * n): _np.array(a)})
            vals.update({'A' + str(2 * n): As})
        elif symmetry is 'odd':
            vals.update({'b' + str(2 * (n + 1)): _np.array(a)})
            vals.update({'B' + str(2 * (n + 1)): As})
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
    if case is "Mathieu":  # cosine jet
        As = cCoeffs(As, n, q)
    if case is 'linear':
        As = linCoeffs(As, n, q)
    if case is 'square':
        As = sqrCoeffs(As, n, q)
    if case is 'step':  # sine flow, associated with odd symmetry
        As = stepCoeffs(As, n, q)
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
    qs = [1.466466, 16.466466,
          47.797797, 95.4654654,
          159.469469, 239.809809,
          336.468468, 452.972972,
          578.813813, 724.434434,
          885.195195]
    N = len(A[0, :])
    if n < 2 and q[0].imag < qs[0]:
        if q.imag[-1] > qs[0]:
            ll = _np.where(q.imag <= qs[0])[0]
            if n == 0:
                for k in range(N):
                    A[ll[-1] + 1:, k] = -A[ll[-1] + 1:, k]
    if n in [2, 3] and q[0].imag < qs[1]:
        if q.imag[-1] > qs[1]:
            ll = _np.where(q.imag <= qs[1])[0]
            if n == 2:
                for k in range(N):
                    A[ll[-1] + 1:, k] = -A[ll[-1] + 1:, k]
                mm = _np.where(A[:, 0].real > 0)[0]  # never changes sign
                A[mm, :] = -A[mm, :]
    if n in [4, 5] and q[0].imag < qs[2]:
        if q.imag[-1] > qs[2]:
            ll = _np.where(q.imag <= qs[2])[0]
            if n == 4:
                for k in range(N):
                    A[ll[-1] + 1:, k] = - A[ll[-1] + 1:, k]
                mm = _np.where(A[:, 0].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
    if n in [6, 7] and q[0].imag < qs[3]:
        if q.imag[-1] > qs[3]:
            ll = _np.where(q.imag <= qs[3])[0]
            if n == 6:
                mm = _np.where(A[:, 0].real > 0)[0]  # always negative
                A[mm, :] = -A[mm, :]
            if n == 7:
                for k in range(N):
                    As = A[ll[-1] - 1, k]
                    for m in _np.arange(ll[-1], ll[-1] + 2):
                        if k % 2 == 0:
                            if _np.sign(A[m, k].imag) != _np.sign(As.imag):
                                A[m, k] = -A[m, k]
                        else:
                            if _np.sign(A[m, k].real) != _np.sign(As.real):
                                A[m, k] = -A[m, k]
    if n in [8, 9] and q[0].imag < qs[4]:
        if q[-1].imag > qs[4]:
            ll = _np.where(q.imag <= qs[4])[0]
            if n == 8:
                mm = _np.where(A[:, 0].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
            if n == 9:
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
    if n in [10, 11] and q[0].imag < qs[5]:
        if q[-1].imag > qs[5]:
            ll = _np.where(q.imag <= qs[5])[0]
            if n == 10:
                mm = _np.where(A[:, 0].real > 0)[0]  # always negative
                A[mm, :] = -A[mm, :]
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
            if n == 11:
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
    if n in [12, 13] and q[0].imag < qs[6]:
        if q[-1].imag > qs[6]:
            ll = _np.where(q.imag <= qs[6])[0]
            if n == 12:
                mm = _np.where(A[:, 0].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
            if n == 13:
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
    if n in [14, 15] and q[0].imag < qs[7]:
        if q[-1].imag > qs[7]:
            ll = _np.where(q.imag <= qs[7])[0]
            if n == 14:
                mm = _np.where(A[:, 0].real > 0)[0]  # always negative
                A[mm, :] = -A[mm, :]
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
            if n == 15:
                mm = _np.where(A[:, 0].real > 0)[0]  # always negative
                A[mm, :] = -A[mm, :]
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
    if n in [16, 17] and q[0].imag < qs[8]:
        if q[-1].imag > qs[8]:
            ll = _np.where(q.imag <= qs[8])[0]
            if n == 16:
                mm = _np.where(A[:, 0].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
            if n == 17:
                mm = _np.where(A[:, 0].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
    if n in [18, 19] and q[0].imag < qs[9]:
        if q[-1].imag > qs[9]:
            ll = _np.where(q.imag <= qs[9])[0]
            if n == 18:
                mm = _np.where(A[:, 0].real > 0)[0]  # always negative
                A[mm, :] = -A[mm, :]
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
            if n == 19:
                mm = _np.where(A[:, 0].real > 0)[0]  # always negative
                A[mm, :] = -A[mm, :]
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
    if n in [20, 21] and q[0].imag < qs[10]:
        if q[-1].imag > qs[10]:
            ll = _np.where(q.imag <= qs[10])[0]
            if n == 20:
                mm = _np.where(A[:, 0].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
            if n == 21:
                mm = _np.where(A[:, 0].real < 0)[0]  # always positive
                A[mm, :] = -A[mm, :]
                for k in range(N):
                    As = abs(A[:ll[-1] + 1, k])  # before EP
                    sign = (1j)**(n - k)  # before EP
                    A[:ll[-1] + 1, k] = sign * As  # Before EP
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
    qs = [2.171671, 22.152152,
          60.626626, 118.621621]
    N = len(A[0, :])
    if n < 2 and q[0].imag < qs[0]:
        if q.imag[-1] > qs[0]:
            ll = _np.where(q.imag <= qs[0])[0]
            if n == 0:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 1:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                    else:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                mm = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
    if n in [2, 3] and q[0].imag < qs[1]:
        if q.imag[-1] > qs[1]:
            ll = _np.where(q.imag <= qs[1])[0]
            if n == 2:
                A[ll[-1], :] = -A[ll[-1], :]
                for k in range(N):
                    if k % 2 == 0:  # even
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 3:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                    else:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                mm = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
    if n in [4, 5] and q[0].imag < qs[2]:
        if q.imag[-1] > qs[2]:
            ll = _np.where(q.imag <= qs[2])[0]
            if n == 4:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 5:
                mm = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
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
                A[ll[-1] - 1:ll[-1] + 1, :] = -A[ll[-1] - 1:ll[-1] + 1, :]
                mm = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
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
    qs = [3.386386, 23.207207,
          52.086086, 86.723723,
          125.530530, 167.591591]
    N = len(A[0, :])
    if n < 2 and q[0].imag < qs[0]:
        if q.imag[-1] > qs[0]:
            ll = _np.where(q.imag <= qs[0])[0]
            if n == 0:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 1:
                mm = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                    else:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
    if n in [2, 3] and q[0].imag < qs[1]:
        if q.imag[-1] > qs[1]:
            ll = _np.where(q.imag <= qs[1])[0]
            if n == 2:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 3:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                    else:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                mm = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
    if n in [4, 5] and q[0].imag < qs[2]:
        if q.imag[-1] > qs[2]:
            ll = _np.where(q.imag <= qs[2])[0]
            if n == 4:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 5:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                    else:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                mm = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
                A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
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
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                    else:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                mm = _np.where(A[ll[-1] + 1:, 0].real > 0)[0]  # should be <0
                A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
    if n in [8, 9] and q[0].imag < qs[4]:
        if q.imag[-1] > qs[4]:
            ll = _np.where(q.imag <= qs[4])[0]
            if n == 8:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 9:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                    else:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
            mm = _np.where(A[ll[-1] + 1:, 0].real < 0)[0]  # should be >0
            A[mm + ll[-1] + 1, :] = -A[mm + ll[-1] + 1, :]
    if n in [10, 11] and q[0].imag < qs[5]:
        if q.imag[-1] > qs[5]:
            ll = _np.where(q.imag <= qs[5])[0]
            if n == 10:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
                    else:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
            if n == 11:
                for k in range(N):
                    if k % 2 == 0:
                        A[ll[-1] + 1:, k].real = -A[ll[-1] + 1:, k].real
                    else:
                        A[ll[-1] + 1:, k].imag = -A[ll[-1] + 1:, k].imag
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
            elif n == 7:
                nn = _np.where(B[:ll[-1], n] < 0)[0]  # should be > 0.
                B[nn, :] = - B[nn, :]
    return B


def Anorm(A, case='None'):
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
    if case == "Mathieu":
        A0star = A[0]
        Astar = A[1:]
        norm = _np.sqrt((2 * (A[0] * A0star)) + _np.sum(A[1:] * Astar))
    else:
        norm = _np.sqrt(_np.sum(A ** 2))
    A = A / norm
    return A
