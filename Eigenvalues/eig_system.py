from numpy import linalg as LA
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
        _coeffs = _np.array(coeffs[:N])
        _K = K[:N]  # take into account only the gravest modes (len(K)=len(coeffs))
        print(_coeffs)
    else:
        _coeffs = _np.array(coeffs)
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


def even_matrix(q, N, alphas, K, symmetry='even'):
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
    if symmetry == 'even':
        A[0, 1:] = _copy.deepcopy(A[0, 1:]) * _np.sqrt(2)
        A[1:, 0] = _copy.deepcopy(A[1:, 0]) * _np.sqrt(2)
    return A


def odd_matrix(q, N, betas, K):
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
    M = len(betas)
    if M > N:  # this shouldn't happen, but if it does just increase N.
        N = M
    diag = [4 * (k**2) for k in range(1, N + 1)]  # diagonal of A.
    A = _np.diag(diag, 0)
    for k in range(len(K)):
        a = q * betas[k] * _np.ones(N - K[k])  # defines the off-diagonal term
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


def eig_pairs(A, symmetry='even'):
    """ Calculates the characteristic value (eigenvalue) and the Fourier
    coefficients of Matrix A (particular Hills equation). Both eigenvals and
    Eigenvectors are sorted in ascending order.

    Input:
        A: Matrix, output from matrix_system.

    Output:
        w: sorted eigenvalues.
    """
    w, V = LA.eig(A)  # calculates the eigenvalue and eigenvector
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
        Ind = _np.argsort(_np.round(a, 1))  # sorting through 1 decimal
        ordered_a = a[Ind]
        nv = 0 * _np.copy(v)
        for k in range(len(Ind)):
            nv[:, k] = v[:, Ind[k]]
    if symmetry == 'None':
        inv = _np.arange(len(Ind))
        mv = 0 * _np.copy(nv)
        ordered_a = ordered_a[inv]
        # for k in range(Ind):
    return ordered_a, nv
