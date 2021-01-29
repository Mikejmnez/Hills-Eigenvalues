from numpy import linalg as LA
import numpy as _np


def matrix_system(q, N, alphas, K, cosine=True):
    ''' Creates a Floquet-Fourier-Hill matrix of order NxN. The size N is
    determined (for now) outside, but it should be larger than the order of
    approximation of the Fourier series.

    Input:
        q: 1d-array. Canonical parameter, purely imaginary.
        N: int. Size of (square) matrix, and thus determines the order of the
            highest harmonic in the trigonometric series that defines each
            eigen-function (must be a higher harmonic than that used in
            approximating the periodic coefficient in ODE).
        alphas: 1d array.  Fourier coefficients (convergent, thus decreasing
            in order)
        cosine: `True` (default), `False`. Type of Fourier approximation of
            the periodic coefficient. `True` implies it is a cosine Fourier
            series.
        K: range(1, M, d). Represents the ordering of Fourier coefficients
            alphas when writing the Fourier sum. if d=1, then a sum of the form
            cos(2*y) + cos(4*y) + .... If d=2 sum is: cos(2*y) + cos(6*y).
    Output:
        A: nd-array. Square matrix with off-diagonal terms that are purely imag
            and a purely reakl diagonal term that increases with the size of A.
    '''
    # make sure q is purely imaginary, N is an integer.
    M = len(alphas)
    if M > N:  # this shouldn't happen, but if it does just increase N.
        N = M
    diag = [4 * (k**2) for k in range(N)]  # diagonal of A.
    A = _np.diag(diag, 0)
    for k in K:
        a = q * K[k] * _np.ones(N - k)  # defines the off-diagonal term
        A = A + _np.diag(a, k) + _np.diag(a, -k)  # adds off-diagonal arrays
    return A

