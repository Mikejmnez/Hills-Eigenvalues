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
        # As = Anorm(A[:, n], type, period)
        As = A[_np.newaxis, :]
        for k in range(1, len(q)):
            an, nA = eig_pairs(matrix_system(q[k], N, coeffs, K, symmetry, case))
            a.append(an[n])
            # nA = Anorm(A[:, n], type, period)
            nAs = nA[_np.newaxis, :]
            As = _np.append(As, nAs, axis=0)
        # As = Fcoeffs(As, n, q, flag=imag)
        if symmetry is 'None':
            vals.update({'a' + str(2 * n): _np.array(a)})
        elif symmetry is 'even':
            vals.update({'a' + str(2 * n): _np.array(a)})
        elif symmetry is 'odd':
            vals.update({'b' + str(2 * (n + 1)): _np.array(a)})
        vals.update({'A' + str(2 * n): As})
    return vals
