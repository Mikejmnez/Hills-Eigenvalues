from numpy import linalg as LA
import numpy as _np


def matrix_system(q, N, alphas, K, cosine=True):
    ''' Creates a matrix of order NxN. The size N is
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
    for k in range(len(K)):
        a = q * alphas[k] * _np.ones(N - K[k])  # defines the off-diagonal term
        A = A + _np.diag(a, K[k]) + _np.diag(a, -K[k])  # adds off-diagonals
    return A


def FFH_matrix(q, N, gammas, K):
    """ Creates a matrix of order (2N+1)x(2N+1) following Floquet-Fourier-Hill
    method.
    Input:
        q: 1d-numpy array. Cannonical parameter, purely imaginary.
        N: Minimum size of matrix (2N+1).
        gammas: 1d numpy array. Monotonically decreasing (complex) truncated
            Fourier coefficients a_m -ib_m. Length(gammas) = M.
        K: range(1, M), ordering of Fourier coefficients begining with m=1.
    Output:
        A: (2N+1)x(2N+1) matrix. Main diagonal is purely real, off diagonals
            purely imaginary.
    """
    M = len(gammas)  # Truncated Fourier coefficients (decreasing)
    if M > N:
        N = M
    diag = [4 * (k**2) for k in range(N + 1)]
    diag = diag[::-1] + diag[1:]
    A = _np.diag(diag, 0)  # initialize matrix
    for k in range(len(K)):
        a = 2 * q * gammas[k] * _np.ones((2 * N) + 1 - K[k])
        ac = 2 * q * _np.conjugate(gammas[k]) * _np.ones((2 * N) + 1 - K[k])
        A = A + _np.diag(ac, K[k]) + _np.diag(a, -K[k])
    return A


def eig_pairs(A):
    """ Calculates the characteristic value (eigenvalue) and the Fourier
    coefficients of Matrix A (particular Hills equation). Both eigenvals and
    Eigenvectors are sorted in ascending order.

    Input:
        A: Matrix, output from matrix_system.

    Output:
        w: sorted eigenvalues.
    """
    w, V = LA.eig(A)  # calculates the eigenvalue and eigenvector
    ord_w, V = order_check(w, V)
    return ord_w


def order_check(a, v):
    """ Check the ordering of the eigenvalue array, from smaller to larger. If
    true, return a unchanged. Ordering also matters if a is complex. If a is
    complex, ordering again is first set according to real(a). If two
    eigenvalues are complex conjugates, then ordering is in accordance to the
    sign of complex(a). Negative sign is first.
    """
    if a.imag.any() == 0:
        ordered_a = a
        nv = v
    else:
        Ind = _np.argsort(_np.round(a, 1))  # sorting through 1 decimal
        ordered_a = a[Ind]
        nv = 0 * _np.copy(v)
        for k in range(len(Ind)):
            nv[:, k] = v[:, Ind[k]]
    return ordered_a, nv



