from scipy import linalg as LA
import math
from scipy.sparse.linalg import eigs as eigs_sp
import numpy as _np
import copy as _copy


def matrix_system(q, N, coeffs, K, symmetry='even'):
    ''' Constructs a matrix system whose eigenvalues and eigenvectors solve
    Hill's equation.

    Input:
        q: 1d-array. Canonical parameter, purely imaginary.
        N: int. Size of (square) matrix, and thus determines the order of the
            highest harmonic in the trigonometric series that defines each
            eigen-function (must be a higher harmonic than that used in
            approximating the periodic coefficient in ODE).
        coeffs: 1d array.  Fourier coefficients (convergent, thus decreasing
            in order)
        K: range(1, M, d). Ordering of Fourier coefficients.
        symmetry: `None` (default), `even`, 'odd'. If 'None', the matrix is
            constructed in according to Fourier-Floquet-Hill theory. 'even'
            implies the system is built using cosine modes that solve Neumann's
            boundary condition. If 'odd', the system is built using sine-modes
            only, that satisfy Dirichlet BC.
        case: 'None' (default) or 'Mathieu'. If 'Mathieu', a factor of sqrt(2)
            is introduced in the matrix construction, in accordance to
            cannonical normalizations of Mathieu's eigenvalue matrix.

    Output:
        A: (N X N) (if a 'even' or 'odd' symmetry) or (2N+1)X(2N+1) matrix.
    '''
    M = len(coeffs)
    if M > N:  # e.g. q -> 0 for which we don't need a large matrix. Only need greavest coeffs
        _coeffs = coeffs[:N]
        _K = K[:N]  # take into account only the gravest modes (len(K)=len(coeffs))
    else:
        _coeffs = coeffs
        _K = K
    if symmetry not in ['None', 'even', 'odd']:
        raise Warning("symmetry argument not recognized. Acceptable options"
                      "are: `None`, `even` and `odd`.")
    if symmetry == 'None':
        A = FFH_matrix(q, N, _coeffs, _K)
    elif symmetry == 'even':
        A = even_matrix(q, N, _coeffs, _K)
    elif symmetry == 'odd':
        A = odd_matrix(q, N, _coeffs, _K)
    return A


def even_matrix(q, N, alphas, K):
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
        case: `None` (default), 'mathieu'. If mathieu, the matrix is modified
            by introducing a factor of sqrt(2) on the first element of the
            first off-diagonal row and column in accordance to Mathieu's
            eigenvalue matrix
    Output:
        A: nd-array. Square matrix with off-diagonal terms that are purely imag
            and a purely real diagonal term that increases with the size of A.
    '''
    # make sure q is purely imaginary, N is an integer.
    diag = [4 * (k**2) for k in range(N)] # diagonal of A.
    A = _np.diag(diag, 0)
    nA = _np.zeros(_np.shape(A))*1j
    for k in range(len(K)):
        a = q * alphas[k] * _np.ones(N - int(K[k]))  # defines the off-diagonal term
        A = A + _np.diag(a, int(K[k])) + _np.diag(a, -int(K[k]))  # adds off-diagonals
    for n in range(1, len(K)):
        nA[1: len(K) + (1 - n), n] = _copy.deepcopy(1j*(q.imag * alphas[n:]))
    A = A + nA
    A[0, 1:] = _copy.deepcopy(A[0, 1:]) * _np.sqrt(2)
    A[1:, 0] = _copy.deepcopy(A[1:, 0]) * _np.sqrt(2)
    return A


def odd_matrix(q, N, alphas, K):
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
            sin(2*y) + sin(4*y) + .... If d=2 sum is: sin(2*y) + sin(6*y).
    Output:
        A: nd-array. Square matrix with off-diagonal terms that are purely imag
            and a purely reakl diagonal term that increases with the size of A.
    '''
    # make sure q is purely imaginary, N is an integer.
    diag = [4 * (k**2) for k in range(N)] # diagonal of A.
    A = _np.diag(diag, 0)
    nA = _np.zeros(_np.shape(A))*1j
    for k in range(len(K)):
        a = q * alphas[k] * _np.ones(N - int(K[k]))  # defines the off-diagonal term
        A = A + _np.diag(a, int(K[k])) + _np.diag(a, -int(K[k]))  # adds off-diagonals
    for n in range(1, len(K)):
        nA[1: len(K) + (1 - n), n] = -_copy.deepcopy(1j*(q.imag * alphas[n:]))
    odd_A = A[1:, 1:] + nA[1:, 1:]
    return odd_A


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


def eig_pairs(A, symmetry='even', sparse=False, Ne=10):
    """ Computes and orders in ascending order the eigenvalues and eigenvector of
    matrices resulting from Hills equation. Eigenvectors are normalized according to
    (non-definite) norm.

        2[A_{0}]^2 + \sum_{r=1}^{\infty}[A_{2r}]^2 = 1,

    if eigenvectors correspond to an even eig system. Or

        \sum_{r=1}^{\infty}[B_{2r+2}]^2 = 1,
    if eigenvectors correspond to an odd- eigenvalue system.

    Parameters:
    ------------
        A: np.ndarray.
            Square matrix, mostly sparse, output from matrix_system.
        symmetry: str.
            'even' (default) or 'odd'.
        sparse: bool
            when `False` (default), all eigenvalues and eigenvectors are computed
            using scipy.linalg.eigs. When `True`, the lowest eigenvalues are `Ne`
            computed using `scipy.sparse.linalg.eigs`.
        Ne: int.
            10 (default) is only used when sparse is `True`. When sparse is 1True`,
            Ne must be smaller than number of columns of A.

    Output:
        w: sorted eigenvalues.
    """
    w, V = LA.eig(A)  # calculates the eigenvalue and eigenvector

    if sparse:
        N = len(A)
        if Ne - N > 0:
            ke = Ne 
        else:
            ke = math.ceil(N / 3)
        if ke < 2:
            ke = 2  # minimum eig to be calc
        wsp, Vsp = eigs_sp(A, k=Ne, which='SR')
        w[:Ne] = wsp
        Vsp[:, :Ne] = Vsp

    if symmetry =='even':
        V[0, :] = _copy.deepcopy(V[0, :]) / _np.sqrt(2)
    ord_w, V = order_check(w, V)
    return ord_w, V


def order_check(a, v, symmetry = 'even'):
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
        Ind = _np.argsort(_np.round(a, 4))  # sorting through 1 decimal
        ordered_a = a[Ind]
        nv = 0 * _np.copy(v)
        for k in range(len(Ind)):
            nv[:, k] = v[:, Ind[k]]
    return ordered_a, nv
