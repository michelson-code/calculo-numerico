# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:19:55 2024
Update on Ago 22 15:50:35 2025

@author: 7michelson
"""

import sys
import numpy as np
from scipy.linalg import dft
# importar Warnigs para evitar mensagem de erro ao 
# ignorar operações com matrizes complexas!
import warnings
try:
    from numpy import ComplexWarning
except ImportError:
    from numpy.exceptions import ComplexWarning

# Operacoes com vetores

### Produto escalar-vetor
def scalar_vec_real(a,x,check_input=True):
    '''
    Compute the product of a scalar a and vector x, where
    a is real and x is in R^N.

    The code uses a simple "for" to iterate on the array.

    input
    -----------------
    a: scalar
        Real number

    x: 1D array
       Vector with N elements.

    returns
    ------------------
    y: 1D array
       Vector with N elements equal the product between a and x.

    '''
    if check_input is True:
        assert isinstance(a, (float, int)), 'a must be a scalar'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'

    result = np.zeros_like(x)
    for i in range(x.size):
        result[i] = a*x[i]

    return result

def scalar_vec_complex(a, x, check_input=True):
    '''
    Compute the dot product of a is a complex number and x
    is a complex vector.

    Parameters
    ----------
    a : scalar
        Complex number.

    x : array 1D
        Complex vector with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Product of a and x.
    '''
    if check_input is True:
        assert isinstance(a, (complex, float, int)), 'a may be complex or scalar'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'

    # Code here

    result_real = scalar_vec_real(a.real, x.real, check_input=False)
    result_real -= scalar_vec_real(a.imag, x.imag, check_input=False)
    result_imag = scalar_vec_real(a.real, x.imag, check_input=False)
    result_imag += scalar_vec_real(a.imag, x.real, check_input=False)

    result = result_real + 1j*result_imag

    return result

### Dot product
def dot_real(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    # Check input here
    
    if check_input:
        assert len(x) == len(y), 'Numero de elementos em x é diferente de numero de elementos em y'
        assert isinstance(x, np.ndarray), 'x deve ser um numpy array, ex: numpy.array([])'
        assert isinstance(y, np.ndarray), 'y deve ser um numpy array, ex: numpy.array([])'
        assert x.ndim == 1, 'x deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        assert y.ndim == 1, 'y deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
    else:
        pass

    # Code here    
    N = len(x) # lembrar que o N de x e y deve ser igual, fazer o acert
    result = 0
    
    for i in range(0, N):
        result += x[i]*y[i]


    return result


def dot_complex(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of C^N.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    # Check input here

    if check_input:
        assert len(x) == len(y), 'Numero de elementos em x é diferente de numero de elementos em y'
        assert isinstance(x, np.ndarray), 'x deve ser um numpy array, ex: numpy.array([])'
        assert isinstance(y, np.ndarray), 'y deve ser um numpy array, ex: numpy.array([])'
        assert x.ndim == 1, 'x deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        assert y.ndim == 1, 'y deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
    else:
        pass
    # Complete here

    c_R  = dot_real(x.real, y.real)
    c_R -= dot_real(x.imag, y.imag)
    c_I  = dot_real(x.real, y.imag)
    c_I += dot_real(x.imag, y.real)
    result = c_R + 1j*c_I
    return result

# Outer product
def outer_real_simple(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    # Check input here

    if check_input:
        # verifica se são vetores 1D
        assert x.ndim == 1 and y.ndim ==1, "x and y must be 1-dimensional arrays"
        # Verifica se x e y são arrays numpy
        assert isinstance(x, np.ndarray) or  isinstance(y, np.ndarray), TypeError("x and y must be numpy arrays")
        # Verifica se os elementos são reais (ignorando partes imaginárias)
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            x = x.real
            y = y.real
    else:
        x.real
        y.real
    # Complete here
    N = len(x)
    M = len(y)
    result = np.zeros(shape=(N, M))

    for i in range(0, N):
        for j in range(0, M):
            result[i,j] = x[i]*y[j]

    return result


def outer_real_row(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code use a single for to compute the rows of 
    the resultant matrix as a scalar-vector product.

    This code uses the function 'scalar_vec_real'.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    # Check input here
    if check_input:
        # verifica se são vetores 1D
        assert x.ndim == 1 and y.ndim ==1, "x and y must be 1-dimensional arrays"
        # Verifica se x e y são arrays numpy
        assert isinstance(x, np.ndarray) or  isinstance(y, np.ndarray), TypeError("x and y must be numpy arrays")
        # Verifica se os elementos são reais (ignorando partes imaginárias)
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            x = x.real
            y = y.real
    else:
        x.real
        y.real
    # Complete here

    N = len(x)
    M = len(y)
    result = np.zeros(shape=(N, M))
    for i in range(0, N):
        result[i,:] = x[i]*y[:]
    return result


def outer_real_column(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code use a single for to compute the columns of 
    the resultant matrix as a scalar-vector product.

    This code uses the function 'scalar_vec_real'.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    # Check input here
    if check_input:
        # verifica se são vetores 1D
        assert x.ndim == 1 and y.ndim ==1, "x and y must be 1-dimensional arrays"
        # Verifica se x e y são arrays numpy
        assert isinstance(x, np.ndarray) or  isinstance(y, np.ndarray), TypeError("x and y must be numpy arrays")
        # Verifica se os elementos são reais (ignorando partes imaginárias)
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            x = x.real
            y = y.real
    else:
        x.real
        y.real
    # Complete here

    N = len(x)
    M = len(y)
    result = np.zeros(shape=(N, M))
    for j in range(0, M):    
        result[:, j] = x[:]*y[j]


    return result


def outer_complex(x, y, check_input=True, function='simple'):
    '''
    Compute the outer product of x and y, where x and y are complex vectors.

    Parameters
    ----------
    x, y : 1D arrays
        Complex vectors.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Defines the outer_real function to be used. The possible
        values are 'simple', 'row' and 'column'. 

    Returns
    -------
    result : 2D array
        Outer product of x and y.
    '''
    x = np.asarray(x, dtype=complex)
    y = np.asarray(y, dtype=complex)

    if check_input:
        # Verifica se function é string
        if not isinstance(function, str):
            raise TypeError("function parameter must be a string")

        if function  not in ['simple', 'row', 'column']:
            raise ValueError(f"function must be one of simple, row, column! Got {function}.")

    else:
        
        function = 'row'

    outer_real = {
        'simple' : outer_real_simple,
        'row' : outer_real_row,
        'column' : outer_real_column
    }
    A_real = outer_real[function](x.real, y.real)
    A_real -= outer_real[function](x.imag, y.imag)
    A_imag = outer_real[function](x.real, y.imag)
    A_imag += outer_real[function](x.imag, y.real)

    # use the syntax outer_real[function] to specify the
    # the outer_real_* function.
    result = A_real + 1j*A_imag

    return result

# Hadamard product
def hadamard_real(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.

    The code uses a simple doubly nested loop to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    with warnings.catch_warnings():
            warnings.simplefilter("ignore", ComplexWarning)
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)

    if check_input:
        # Check if x and y are numpy arrays
        assert  isinstance(x, np.ndarray) or  isinstance(y, np.ndarray), "Inputs x and y must be numpy arrays."
        # Check if x and y have the same shape
        assert x.shape == y.shape,  ValueError("Inputs x and y must have the same shape.")

    else:
        pass
    # Check if x and y are real
        

    
    if x.ndim == 1:  # Vector
        N = x.shape[0]
        result = np.zeros(N)
        for i in range(0, N):
            result[i] = x[i] * y[i]
    elif x.ndim == 2:  # Matrix
        N = x.shape[0]
        M = x.shape[1]
        result = np.zeros(shape=(N, M))
        for i in range(0, N):
            for j in range(0, M):
                result[i, j] = x[i, j] * y[i, j]
    else:
        raise ValueError("Inputs x and y must be 1D or 2D arrays.")

    return result


def hadamard_complex(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be complex vectors or matrices having the same shape.

    Parameters
    ----------
    x, y : arrays
        Complex vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''

    if check_input:
        assert np.shape(x) == np.shape(y), "Inputs x and y must have the same shape."

    C_R  = hadamard_real(x.real, y.real)
    C_R -= hadamard_real(x.imag, y.imag)
    C_I  = hadamard_real(x.real, y.imag)
    C_I += hadamard_real(x.imag, y.real)
    result = C_R + 1j*C_I

    return result

## Operations with matrix

# Matrix-vector product

def matvec_real_simple(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code uses a simple doubly nested "for" to iterate on the arrays.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''

    N, M = np.shape(A)
    A = A.real
    x = x.real
    
    if check_input:
        assert A.ndim==2, 'A must be array 2D.'
        assert x.ndim ==1, 'x must be array 1D'
        assert M == x.shape[0], f"Matrix columns ({M}) must match vector length ({x.shape[0]})"
        assert isinstance(A, np.ndarray) and isinstance(x, np.ndarray), TypeError("Both A and x must be numpy arrays")
    else:
        pass
    
    y = np.zeros(N)
    for i in range(0, N):
        for j in range(0, M):
            y[i] += A[i,j]*x[j]
    result = y
    return result


def matvec_real_dot(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code replaces a for by a dot product.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    N, M = np.shape(A)
    A = A.real
    x = x.real
    
    if check_input:
        assert A.ndim==2, 'A must be array 2D.'
        assert x.ndim ==1, 'x must be array 1D'
        assert M == x.shape[0], f"Matrix columns ({M}) must match vector length ({x.shape[0]})"
        assert isinstance(A, np.ndarray) and isinstance(x, np.ndarray), TypeError("Both A and x must be numpy arrays")
    else:
        pass
    result = np.zeros(N)
    for i in range(0, N):
        result[i] = dot_real(A[i,:], x[:])
    

    return result


def matvec_real_columns(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code replaces a for by a scalar-vector product.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    N, M = np.shape(A)
    A = A.real
    x = x.real
    
    if check_input:
        assert A.ndim==2, 'A must be array 2D.'
        assert x.ndim ==1, 'x must be array 1D'
        assert M == x.shape[0], f"Matrix columns ({M}) must match vector length ({x.shape[0]})"
        assert isinstance(A, np.ndarray) and isinstance(x, np.ndarray), TypeError("Both A and x must be numpy arrays")
    else:
        pass
    result = np.zeros(N)
    for j in range(0, M):
        result[:] += scalar_vec_real(x[j], A[:,j])

    return result


def matvec_complex(A, x, check_input=True, function='dot'):
    '''
    Compute the matrix-vector product of an NxM matrix A and
    a Mx1 vector x.

    Parameters
    ----------
    A : array 2D
        NxM matrix.

    x : array 1D
        Mx1 vector.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Defines the matvec_real function to be used. The possible
        values are 'simple', 'dot' and 'columns'.

    Returns
    -------
    result : array 1D
        Product of A and x.
        Linear sistem = y = Ax
    '''
    x = np.asarray(x, dtype=complex)
    A = np.asarray(A, dtype=complex)

    if check_input:
        # Verifica se function é string
        if not isinstance(function, str):
            raise TypeError("function parameter must be a string")

        if function  not in ['simple', 'dot', 'columns']:
            raise ValueError(f"function must be one of; simple, dot, columns! Got {function}.")

    else:
        function = 'dot'

    matvec_real = {
        'simple' : matvec_real_simple,
        'dot' : matvec_real_dot,
        'columns' : matvec_real_columns
    }


    # use the syntax matvec_real[function] to specify the
    C_real = matvec_real[function](A.real, x.real)
    C_real -= matvec_real[function](A.imag, x.imag)
    C_imag = matvec_real[function](A.real, x.imag)
    C_imag += matvec_real[function](A.imag, x.real)
    # the matvec_real_* function.

    result = C_real + 1j*C_imag
    return result

# matrix-matrix product

def matmat_real_simple(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxL and B in R^LxM. The imaginary parts are ignored.

    The code uses a simple triply nested "for" to iterate on the arrays.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''

    # With:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        B = np.asanyarray(B, dtype=float)

    N, L = A.shape
    P, M = B.shape

    if check_input:
        # Testar se numeros de colunas em A{L} é igual ao numeros de linha em B{P}!
        assert L == P, ValueError(f"Columns of A ({L}) must match rows of B ({P}) for matrix multiplication")
        assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), TypeError("Inputs must be NumPy arrays")
        assert A.ndim  == 2 and B.ndim==2, "Both A and B must be 2D arrays."
    result = np.zeros(shape=(N, M))
    for i in range(0, N):
        for j in range(0,M):
            for k in range(0, L):
                result[i, j] += A[i, k]*B[k, j]
    
    return result


def matmat_real_dot(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces one "for" by a dot product.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        B = np.asanyarray(B, dtype=float)

    N, L = A.shape
    P, M = B.shape

    if check_input:
        # Testar se numeros de colunas em A{L} é igual ao numeros de linha em B{P}!
        assert L == P, ValueError(f"Columns of A ({L}) must match rows of B ({P}) for matrix multiplication")
        assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), TypeError("Inputs must be NumPy arrays")
        assert A.ndim  == 2 and B.ndim==2, "Both A and B must be 2D arrays."
    result = np.zeros(shape=(N, M))

    for i in range(0, N):
        for j in range(0, M):
            result[i,:] = dot_real(A[i, :], B[:, j])

    return result


def matmat_real_rows(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "fors" by a matrix-vector product defining
    a row of the resultant matrix.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        B = np.asanyarray(B, dtype=float)

    N, L = A.shape
    P, M = B.shape

    if check_input:
        # Testar se numeros de colunas em A{L} é igual ao numeros de linha em B{P}!
        assert L == P, ValueError(f"Columns of A ({L}) must match rows of B ({P}) for matrix multiplication")
        assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), TypeError("Inputs must be NumPy arrays")
        assert A.ndim  == 2 and B.ndim==2, "Both A and B must be 2D arrays."
    result = np.zeros(shape=(N, M))

    for i in range(0, N):
        result[i, :] = matvec_real_dot(B[:, :].T , A[i, :])

    return result


def matmat_real_columns(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "fors" by a matrix-vector product defining
    a column of the resultant matrix.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        B = np.asanyarray(B, dtype=float)

    N, L = A.shape
    P, M = B.shape

    if check_input:
        # Testar se numeros de colunas em A{L} é igual ao numeros de linha em B{P}!
        assert L == P, ValueError(f"Columns of A ({L}) must match rows of B ({P}) for matrix multiplication")
        assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), TypeError("Inputs must be NumPy arrays")
        assert A.ndim  == 2 and B.ndim==2, "Both A and B must be 2D arrays."
    result = np.zeros(shape=(N, M))

    for j in range(0, M):
        result[:, j] = matvec_real_dot(A[:,:], B[:, j])

    return result


def matmat_real_outer(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "fors" by an outer product.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        B = np.asanyarray(B, dtype=float)

    N, L = A.shape
    P, M = B.shape

    if check_input:
        # Testar se numeros de colunas em A{L} é igual ao numeros de linha em B{P}!
        assert L == P, ValueError(f"Columns of A ({L}) must match rows of B ({P}) for matrix multiplication")
        assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), TypeError("Inputs must be NumPy arrays")
        assert A.ndim  == 2 and B.ndim==2, "Both A and B must be 2D arrays."
    result = np.zeros(shape=(N, M))

    for  k in range(0, L):
        result[:,:] += outer_real_row(A[:,k], B[k,:])

    return result


def matmat_complex(A, B, check_input=True, function='simple'):
    '''
    Compute the matrix-matrix product of A and B, where
    A in C^NxM and B in C^MxP.

    Parameters
    ----------
    A, B : 2D arrays
        Complex matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Defines the matmat_real function to be used. The possible
        values are 'simple', 'dot', 'rows', 'columns' or 'outer'.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''

    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)

    if check_input:
        # Verifica se function é string
        if not isinstance(function, str):
            raise TypeError("function parameter must be a string")

        if function  not in ['simple', 'dot', 'rows', 'columns', 'outer']:
            raise ValueError(f"function must be one of; simple, dot, rows, columns, outer! Got {function}.")

    else:
        function = 'simple'

    matmat_real = {
        'simple' : matmat_real_simple,
        'dot' : matmat_real_dot,
        'rows' : matmat_real_rows,
        'columns' : matmat_real_columns,
        'outer' : matmat_real_outer
    }

    # use the syntax matmat_real[function] to specify the
    # the matmat_real_* function.
    C_real = matmat_real[function](A.real, B.real)
    C_real -= matmat_real[function](A.imag, B.imag)
    C_imag = matmat_real[function](A.real, B.imag)
    C_imag += matmat_real[function](A.imag, B.real)
    result = C_real +1j*C_imag

    return result

## Triangular matrices
def matvec_triu_prod3(U, x, check_input=True):
    '''
    Compute the product of an upper triangular matrix U 
    and a vector x. All elements are real numbers.
    
    Each element of the resultant vector is obtained by 
    computing a dot product.

    Parameters
    ----------
    U : numpy array 2d
        Upper triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''

    # create your code here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        U = np.asanyarray(U, dtype=float)
        x = np.asanyarray(x, dtype=float)

    N, M = U.shape
    L = x.shape[0]

    if check_input:
        assert N==M, "U must be squere."
        assert isinstance(U, np.ndarray) and U.ndim == 2, "U must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == L, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(U, np.triu(U)), "U must be upper triangular" 
    result = np.zeros(L)

    for i in range(0, N):
        result[i] = dot_real(U[i, i:], x[i:])
    
    return result

def matvec_triu_prod5(U, x, check_input=True):
    '''
    Compute the product of an upper triangular matrix U 
    and a vector x. All elements are real numbers.
    
    The elements of the resultant vector are obtained by 
    computing successive scalar vector products.

    Parameters
    ----------
    U : numpy array 2d
        Upper triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''

    # create your code here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        U = np.asanyarray(U, dtype=float)
        x = np.asanyarray(x, dtype=float)

    N, M = U.shape
    L = x.shape[0]

    if check_input:
        assert N==M, "U must be squere."
        assert isinstance(U, np.ndarray) and U.ndim == 2, "U must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == L, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(U, np.triu(U)), "U must be upper triangular" 
    result = np.zeros_like(x)
    # obs, ao usar :j é preciso somar 1 para acessar o ultimo elemnto
    #   pois o laço é de N-1, e indexado no inicio em 0
    for j in range(0, N):
        result[:j+1] += scalar_vec_real(a=x[j], x=U[:j+1,j])
    
    return result

def matvec_tril_prod8(L, x, check_input=True):
    '''
    Compute the product of an lower triangular matrix L 
    and a vector x. All elements are real numbers.
    
    Each element of the resultant vector is obtained by 
    computing a dot product.

    Parameters
    ----------
    L : numpy array 2d
        Lower triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''

    # create your code here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        L = np.asanyarray(L, dtype=float)
        x = np.asanyarray(x, dtype=float)

    N, M = L.shape
    P = x.shape[0]

    if check_input:
        assert N==M, "L must be squere"
        assert isinstance(L, np.ndarray) and L.ndim == 2, "L must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == P, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(L, np.tril(L)), "L must be lower triangular" 
    result = np.zeros(P)
    # obs, ao usar :j é preciso somar 1 para acessar o ultimo elemnto
    #   pois o laço é de N-1, e indexado no inicio em 0
    for i in range(0, N):
        result[i] = dot_real(L[i,:i+1], x[:i+1])
    
    return result

def matvec_tril_prod10(L, x, check_input=True):
    '''
    Compute the product of an lower triangular matrix L 
    and a vector x. All elements are real numbers.
    
    The elements of the resultant vector are obtained by 
    computing successive scalar vector products.

    Parameters
    ----------
    L : numpy array 2d
        Lower triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        L = np.asanyarray(L, dtype=float)
        x = np.asanyarray(x, dtype=float)

    N, M = L.shape
    P = x.shape[0]

    if check_input:
        assert N==M, "L must be squere."
        assert isinstance(L, np.ndarray) and L.ndim == 2, "L must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == P, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(L, np.tril(L)), "L must be lower triangular" 
    result = np.zeros(P)
    # create your code here
    for j in range(0, N):
        result[j:] += scalar_vec_real(a=x[j], x=L[j:, j])
    return result
 

def triu_system(A, x, check_input=True):
    '''
    Solve the linear system Ax = y for x by using back substitution.

    The elements of x are computed by using a 'dot' within a single for.

    Parameters
    ----------
    A : numpy array 2d
        Upper triangular matrix.
    y : numpy array 1d
        Independent vector of the linear system.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Solution x of the linear system.
        Ay = x
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        x = np.asanyarray(x, dtype=float)
    N, M = A.shape
    n = x.shape[0]

    if check_input:
        assert N==M, "U must be squere."
        assert isinstance(A, np.ndarray) and A.ndim == 2, "U must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == n, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(A, np.triu(A)), "U must be upper triangular" 

    # create your code here

    result = np.zeros(n)

    for i in range(N-1, -1, -1):
        result[i] = x[i]  - dot_real(A[i, i+1:], result[i+1:])
        result[i] /= A[i, i]
    
    return result



def tril_system(A, x, check_input=True):
    '''
    Solve the linear system Ax = y for x by using forward substitution.

    The elements of x are computed by using a 'dot' within a single for.

    Parameters
    ----------
    A : numpy array 2d
        Lower triangular matrix.
    y : numpy array 1d
        Independent vector of the linear system.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Solution x of the linear system.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        x = np.asanyarray(x, dtype=float)
    N, M = A.shape
    n = x.shape[0]

    if check_input:
        assert N==M, "U must be squere."
        assert isinstance(A, np.ndarray) and A.ndim == 2, "U must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == n, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(A, np.tril(A)), "L must be lower triangular" 

    # create your code here
    result = np.zeros(n)

    for i in range(0, N):
        result[i] = x[i]
        for j in range(0, i):
            result[i] -= A[i, j]*result[j]
        result[i]/=A[i,i]
    
    return result

def dim(obj):
    '''
    Essa função recebe uma lsita e retorna a dimenssão dela. 
    '''
    if isinstance(obj, (list, np.ndarray)):
        if isinstance(obj, np.ndarray):
            return obj.ndim
        elif isinstance(obj, list):
            return 1 + max(dim(item) for item in obj) if obj else 1
    else:
        return TypeError('A estrutura de dados não é uma lista ou um numpy array.')


def vec_norm(x, p, check_input=True):
    '''
    x deve ser um vetor. 
    Enquanto p diz a ordem da normalização do vetor.
    p = 0,1,2.
    '''
    if check_input == True:
        assert p == 0 or p==1 or p ==2, 'A ordem só deve ser 0, 1 ou  2.'
        assert np.ndim(x) == 1, 'O vetor deve ter uma dimenção!'
    else:
        pass    
    result = 0
    if p == 0:
        for i in range(0, len(x)):
            x[i] = abs(x[i].real)
        result = np.max(x)
    elif p==1:
        for i in range(0, len(x)):
            result += abs(x[i].real)
    elif p == 2:
        for i in range(0,len(x)):
            result += x[i].real*x[i].real
        result = result**(1/2)

    return float(result)

# OPERAÇÕES COM MATRIZES - VETORES

def mat_norm(A, norm_type='fro', check_input=True):
    """
    Calcula a norma de uma matriz de acordo com o tipo especificado.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz 2D (NxM) para a qual será calculada a norma.
    
    norm_type : str
        Tipo de norma a ser calculada. Pode ser:
        - 'fro' : Norma de Frobenius (padrão)
        - '1'   : 1-norma (máxima soma de colunas)
        - '2'   : 2-norma (norma espectral)
        - 'inf' : Infinito-norma (máxima soma de linhas)
        
    check_input : bool, opcional
        Se True, verifica a validade da entrada, assegurando que A é uma matriz (não um vetor).
        O padrão é True.
    
    Retorna:
    --------
    norm : float
        O valor da norma calculada.
    
    Exceções:
    ---------
    ValueError:
        Lança erro se A não for uma matriz 2D ou se um tipo de norma inválido for especificado.
    """
    
    # Verificação da entrada
    if check_input:
        # Verifica se a entrada é uma matriz (2D)
        if not isinstance(A, np.ndarray):
            raise ValueError("A deve ser um array numpy.")
        if A.ndim != 2:
            raise ValueError("A deve ser uma matriz 2D. Vetores não são permitidos.")
        assert norm_type in ['fro', '1', '2', 'inf'], "A norma p deve ser 'fro', 1, 2, ou 'inf'."
    
    AtA = matmat_real_dot(A.T, A)
    if norm_type == 'fro':
        # Norma de Frobenius usando a raiz quadrada do traço de A^T A
        return np.sqrt(np.trace(AtA))
    elif norm_type == '1':
        # 1-norma (máxima soma de colunas)
        return np.max(np.sum(np.abs(A), axis=0))
    elif norm_type == '2':
        # 2-norma (norma espectral)
        # Calcula os autovalores de A^T A (matriz de Gram)
        eigenvalues = np.linalg.eigvals(AtA)
        # A 2-norma é a raiz quadrada do maior autovalor
        norm_2 = np.sqrt(np.max(eigenvalues))
        return norm_2
    elif norm_type == 'inf':
        # Infinito-norma (máxima soma de linhas)
        return np.max(np.sum(np.abs(A), axis=1))
    
def mat_sma(data, window, check_input=True):
    '''
    Calculate the moving average filter by using the matrix-vector product.

    Parameters
    ----------
    data : numpy array 1d
        Vector containing the data.
    window : positive integer
        Positive integer defining the number of elements forming the window.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector containing the filtered data.
    '''
    
    if check_input:
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("data must be a 1-dimensional numpy array")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")
        if window >= len(data):
            raise ValueError("window size must be smaller than data size")
        if window % 2 == 0:
            raise ValueError("window size must be odd")
    
    N = len(data)
    ws = window
    i0 = ws // 2
    #matrix caracteristica da media móvel
    A = np.array(
        np.hstack(
            (
                (1./ws) * np.ones(ws), 
                np.zeros(N - ws + 1)
            )
        )
    )
    
    A = np.resize(A, (N - 2 * i0, N))
    A = np.vstack((np.zeros(N), A, np.zeros(N)))
    
    result = matvec_dot(A, data)
    
    return result

def deriv1d(data, spacing, check_input=True):
    '''
    Calculate the first derivative by using the matrix-vector product.

    Parameters
    ----------
    data : numpy array 1d
        Vector containing the data.
    spacing : positive scalar
        Positive scalar defining the constant data spacing.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector containing the computed derivative.
    '''
    
    if check_input:
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("data must be a 1-dimensional numpy array")
        if not isinstance(spacing, (int, float)) or spacing <= 0:
            raise ValueError("spacing must be a positive scalar")
    
    N = len(data)
    ws = 3  # window size
    i0 = ws // 2
    h = spacing
    
    # Step 1: Create the initial matrix with -1, 0, 1 and zeros
    A = np.array(
        np.hstack(
            (np.array([-1, 0, 1]), np.zeros(N - ws + 1))
        )
    )
    
    # Step 2: Resize the matrix
    A = np.resize(A, (N - 2 * i0, N))
    
    # Step 3: Add rows of zeros at the top and bottom
    D = np.vstack((np.zeros(N), A, np.zeros(N)))
    
    # Step 4: Divide by 2h
    D = D / (2 * h)
    
    # Step 5: Multiply by the data vector
    result = matvec_dot(D, data)
    result = np.delete(result, [0, -1]) #elimina o primeiro e o último elemento
    
    return result




### Resulução sistema ax = d

#Matriz de Permutação
def permut(C, k):
        '''Função para encontrar o pivô e realizar a permutação de linhas'''
        # Encontra o índice do maior elemento em módulo na coluna k a partir da linha k
        max_index = np.argmax(abs(C[k:, k])) + k
        if max_index != k:
            # Realiza a troca de linhas, se necessário
            C[[k, max_index]] = C[[max_index, k]]
        return max_index, C

#Eliminação de Gauss 
  
def Gauss_elim(A, x, check_input=True, Mat_L = False):
    """
    Executa a eliminação Gaussiana em uma matriz quadrada A e vetor x para resolver o sistema linear Ax = b.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz quadrada 2D (NxN) representando os coeficientes do sistema linear.
        
    x : np.ndarray
        Vetor 1D (N) representando o vetor do lado direito do sistema Ax = b.
        
    check_input : bool, opcional
        Se True, verifica se as entradas são válidas, incluindo a verificação de que A é uma matriz 2D 
        e x é um vetor 1D, além de garantir que ambos contenham apenas números reais. 
        O padrão é True.
    Mat_L : bool, opcional
        Se True, Retrona a matriz L (triangular inferior).
        O padrão é False.
    Retornos:
    ---------
    Uper : np.ndarray
        Matriz 2D (NxN) resultante da eliminação de Gauss, onde Uper é a matriz triangular superior.
    
    b : np.ndarray
        Vetor 1D (N) modificado correspondente ao vetor do lado direito após a eliminação Gaussiana.

    L : np.ndarray
        Matriz 2D (NxN) reultante do processo da eliminação de Gauss, onde L é uma matriz triangular inferior
    
    Exceções:
    ---------
    AssertionError
        Se `check_input` for True e `A` não for uma matriz 2D, ou `x` não for um vetor 1D, ou se contiverem
        valores não reais.
    
    Exemplo de uso:
    --------------
    A = np.array([[2, -1, 1],
                  [3, 3, 9],
                  [3, 3, 5]], dtype=float)
    
    x = np.array([8, 0, -6], dtype=float)
    
    Uper, b = Gaussian_elimination(A, x)
    print("Matriz escalonada (Uper):", Uper)
    print("Vetor modificado (b):", b)
    """
    N = A.shape[0]  # Número de linhas
    Uper = np.copy(A)
    b = np.copy(x)
    I = np.identity(N)
    L = np.identity(N) # L inicia como identidade

    if check_input == True:
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(x, np.ndarray) and b.ndim == 1, 'x deve ser uma vetor 1D'
        # Verifica se A e b contêm apenas números reais
        assert np.isrealobj(A), 'A deve conter apenas números reais'
        assert np.isrealobj(x), 'x deve conter apenas números reais'
    else:
        pass
    
    for k in range(N - 1):
        # Criação do vetor u_k
        u_k = np.zeros(N)
        u_k[k] = 1
        #C = np.zeros(A.shape)
        # Criação do vetor t_k
        t_k = np.zeros(N)
      
        for i in range(k + 1, N):
            t_k[i] = Uper[i, k] / Uper[k, k]
        # Cria a matriz de permutação
    
        # Atualização da matriz Uper e do vetor b usando matmat_real_dot e matvec_dot
        M = outer_real(t_k, u_k)
        Uper = matmat_real_dot(I - M, Uper)
        b = matvec_dot(I - outer_real(t_k, u_k), b)

        # Acumula as operações de eliminação para construir a matriz L
        L = matmat_real_dot(L, I + M)
    
    if Mat_L == False:
        return Uper, b
    else:
        return Uper, b, L

# Ax = B resolução de sistema, usando matriz triangulçar superior e inferior
def triangular_superior(a, d, check_input=True):
    '''
    Resolva o sistema linear Ax = y para x usando a substituição reversa.

    Os elementos de x são calculados usando um 'ponto' dentro de um único for.

    Parâmetros
    ----------
    A : numpy array 2d
        Matriz triangular superior.
    y : matriz numpy 1d
        Vetor independente do sistema linear.
    check_input : booleano
        Se True, verifique se a entrada é válida. O padrão é True.

    Retorna
    -------
    Resultado: numpy array 1d
        Solução x do sistema linear.
    '''

    N = len(d)

    if check_input == True:
        # Verifica se A é uma matriz 2D
        if not (isinstance(a, np.ndarray) and a.ndim == 2):
            raise ValueError('A deve ser uma matriz 2D (NxN)')
        # Verifica se d é um vetor 1D
        if not (isinstance(d, np.ndarray) and d.ndim == 1):
            raise ValueError('d deve ser um vetor 1D')
        # Verifica se A é uma matriz quadrada
        if a.shape[0] != a.shape[1]:
            raise ValueError('A deve ser uma matriz quadrada (NxN)')
        # Verifica se o comprimento de d corresponde ao número de linhas de A
        if a.shape[0] != d.size:
            raise ValueError('O comprimento de d deve ser igual ao número de linhas de A')
        # Verifica se A é triangular superior
        if not np.allclose(a, np.triu(a)):
            raise ValueError('A deve ser uma matriz triangular superior')
        # Verifica se não há zeros na diagonal principal de A
        if np.any(np.diag(a) == 0):
            raise ValueError('Nenhum elemento da diagonal principal de A deve ser zero')
    else:
        pass
    
    p = np.zeros(N)
    for i in range(N-1, -1, -1):
        soma = 0
        p[i] = d[i]
        for j in range(i+1, N):
            soma += a[i, j]*p[j]
        p[i] = (d[i]- soma)/a[i,i]
    return p

def triangular_inferior(a, d, check_input=True):
    '''
    Resolva o sistema linear Ax = y para x usando a substituição direta.
    Os elementos de x são calculados usando um 'ponto' dentro de um único for.

    Parâmetros
    ----------
    A : numpy array 2d
        Matriz triangular inferior.
    y : matriz numpy 1d
        Vetor independente do sistema linear.
    check_input : booleano
        Se True, verifique se a entrada é válida. O padrão é True.

    Retorna
    -------
    Resultado: numpy array 1d
        Solução x do sistema linear.
    '''

    if check_input == True:
        # Verifica se A é uma matriz 2D
        if not (isinstance(a, np.ndarray) and a.ndim == 2):
            raise ValueError('A deve ser uma matriz 2D (NxN)')
        
        # Verifica se d é um vetor 1D
        if not (isinstance(d, np.ndarray) and d.ndim == 1):
            raise ValueError('d deve ser um vetor 1D')
        
        # Verifica se A é uma matriz quadrada
        if a.shape[0] != a.shape[1]:
            raise ValueError('A deve ser uma matriz quadrada (NxN)')
        
        # Verifica se o comprimento de d corresponde ao número de linhas de A
        if a.shape[0] != d.size:
            raise ValueError('O comprimento de d deve ser igual ao número de linhas de A')
        
        # Verifica se A é triangular inferior
        if not np.allclose(a, np.tril(a)):
            raise ValueError('A deve ser uma matriz triangular inferior')
        
        # Verifica se não há zeros na diagonal principal de A
        if np.any(np.diag(a) == 0):
            raise ValueError('Nenhum elemento da diagonal principal de A deve ser zero')
    else:
        pass

    N = len(d)
    p = np.zeros(N)
    for i in range(0, N):
        soma = 0
        p[i] = d[i]
        for j in range(0, i):
            soma += a[i, j]*p[j]
        p[i] = (d[i] - soma )/a[i,i]
    return p
#Função de minimos quadrados
def minimos_quadrados(A, d, check_input=True, inc = False):
    """
    Calcula a solução do sistema Ax = d utilizando o método dos mínimos quadrados.
    
    Seja o sistema Ax = d, onde A é a matriz de sensitividade e d é o vetor de dados observados.
    A solução por mínimos quadrados é obtida resolvendo o sistema linear aproximado que minimiza
    o erro quadrático entre os dados observados e os dados preditos.
    
    A solução dos mínimos quadrados, x_min, é dada por:
    
    x_min = (A.T @ A)^(-1) @ A.T @ d
    
    Onde:
    - A.T é a transposta de A.
    - (A.T @ A)^(-1) é a inversa da matriz de Gram.
    - d é o vetor de dados observados.

    Parâmetros:
    -----------
    A : np.ndarray
        Matriz 2D (NxM) representando a matriz de sensitividade dos dados.
    
    d : np.ndarray
        Vetor 1D (N) representando os dados observados.
    
    Retornos:
    ---------
    x_min : np.ndarray
        Vetor 1D (M) que representa a solução dos mínimos quadrados, minimizando o erro quadrático.
    
    Exemplo de uso:
    --------------
    A = np.array([[1, 2], [3, 4], [5, 6]])
    d = np.array([7, 8, 9])
    
    x_min = minimos_quadrados(A, d)
    print("Solução dos mínimos quadrados:", x_min)
    """

    if check_input == True:
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        assert isinstance(d, np.ndarray) and d.ndim == 1, 'd deve ser um vetor 1D'
    
        # Verifica se A e d têm dimensões compatíveis
        assert A.shape[0] == d.size, 'O número de linhas de A deve ser igual ao tamanho de d'
    else:
        pass
        
    y = matvec_real_simple(A.T, d)
    G = matmat_real_simple(A.T, A)
    G_inv = np.linalg.inv(G)
    x_min = matvec_real_simple(G_inv, y)

    if inc:
        # Calcula a matriz de covariância dos parâmetros
        incerteza = covariancia_parametros(A, d=d)
        return x_min, incerteza

    return x_min

def minimos_quadrados_ponderado(A, d, w, inc = False, check_input=True):
    """
    Estima os valores absolutos de gravidade nos nós de uma rede sintética, 
    usando o método dos mínimos quadrados ponderados, e calcula as incertezas associadas.

    Parâmetros:
    A (np.ndarray): Matriz de coeficientes que relaciona as observações com os nós da rede.
    d (np.ndarray): Vetor de observações de gravidade.
    W (np.ndarray): Matriz de pesos das observações.
    sigma_d (float): Desvio padrão das observações.

    Retorna:
    tuple: Um tuplo contendo:
        - p_hat (np.ndarray): Vetor dos valores estimados de gravidade nos nós.
        - uncertainties (np.ndarray): Vetor das incertezas associadas às estimativas.

    Lança:
    AssertionError: Se as entradas forem inválidas.
    """
    # Verificar as entradas
    W = np.diag(w) # matriz diagonal de w (pesos de ponderamento)
    if check_input == True:
        assert isinstance(A, np.ndarray) and A.ndim == 2, "A deve ser uma matriz 2D (np.ndarray)"
        assert isinstance(d, np.ndarray) and d.ndim == 1, "d deve ser um vetor 1D (np.ndarray)"
        assert isinstance(W, np.ndarray) and W.ndim == 2, "W deve ser uma matriz 2D (np.ndarray)"
        assert A.shape[0] == d.shape[0], "A e d devem ter o mesmo número de linhas"
        assert W.shape == (A.shape[0], A.shape[0]), "W deve ser uma matriz quadrada com o mesmo número de linhas que A"
    else:
        pass
        
    L = matmat_real_simple(matmat_real_dot(A.T, W) , A) # At *W* A
    wd = matvec_dot(W, d) # W *d
    t = matvec_dot(A.T, wd) # (A.t * W *d)
    L_inv = np.linalg.inv(L)
    p_hat = matvec_dot(L_inv, t)

    if inc == True:
        N = len(w) 
        W_half = np.diag([1/np.sqrt(w[0])]*5 + [1/np.sqrt(w[N-1])]*2)
        sigma_d = np.sqrt(variancia(d)) # Desvio padrão dos dados observados
        assert isinstance(sigma_d, (float, int)) and sigma_d > 0, "sigma_d deve ser um número positivo"
        covariance_matrix = L_inv @ A.T @ W_half @ np.linalg.inv(W) @ W_half @ A @ L_inv
        std = np.sqrt(np.diag(covariance_matrix))
        return p_hat, covariance_matrix, std
    else:
        return p_hat

def residuo(A, x_min, d, check_input=True):
    """
    Calcula o resíduo r = d - Ax_min e a norma do resíduo (R2).

    Parâmetros
    ----------
    A : np.ndarray
        Matriz 2D (NxM) representando a matriz de sensitividade dos dados.
    x_min : np.ndarray
        Vetor 1D (M) representando a solução dos mínimos quadrados.
    d : np.ndarray
        Vetor 1D (N) representando os dados observados.
    check_input : bool, opcional
        Se True, verifica se as entradas são válidas. O padrão é True.

    Retornos
    --------
    residuo : np.ndarray
        Vetor 1D (N) representando o resíduo r = d - Ax_min.
    R2 : float
        Norma do resíduo, calculada como ||d - Ax_min||.

    Exceções
    ---------
    AssertionError
        Se `check_input` for True e `A` não for uma matriz 2D, `x_min` e `d` não forem vetores 1D,
        ou se `A`, `x_min`, e `d` tiverem dimensões incompatíveis.

    Exemplo de uso
    --------------
    A = np.array([[1, 2], [3, 4], [5, 6]])
    x_min = np.array([0.5, 1.5])
    d = np.array([7, 8, 9])
    
    residuo_valor, R2 = residuo(A, x_min, d)
    print("Resíduo:", residuo_valor)
    print("Norma do resíduo (R2):", R2)
    """
    
    if check_input == True:
        # Verifica se A é uma matriz 2D
        assert isinstance(A, np.ndarray) and A.ndim == 2, 'A deve ser uma matriz 2D'
        # Verifica se x_min e d são vetores 1D
        assert isinstance(x_min, np.ndarray) and x_min.ndim == 1, 'x_min deve ser um vetor 1D'
        assert isinstance(d, np.ndarray) and d.ndim == 1, 'd deve ser um vetor 1D'
        # Verifica se A, x_min e d têm dimensões compatíveis
        assert A.shape[1] == x_min.size, 'O número de colunas de A deve ser igual ao tamanho de x_min'
        assert A.shape[0] == d.size, 'O número de linhas de A deve ser igual ao tamanho de d'
    else:
        pass

    d_pred = matvec_real_simple(A, x_min)
    residuo = d - d_pred
    R2 = vec_norm(residuo, 2)
    return residuo, R2

# Decomposição LU 
def lu_decomp(A, f, check_input=True):
    """
    Realiza a decomposição LU da matriz A para o sistema linear Ax = f.
    
    A decomposição LU decompõe a matriz A em duas matrizes:
        - L: matriz triangular inferior
        - U: matriz triangular superior
    
    Parâmetros:
    -----------
    A : numpy.ndarray
        Matriz quadrada de coeficientes (n x n) do sistema linear.
    f : numpy.ndarray
        Vetor de termos independentes do sistema (n x 1).
    check_input : bool, opcional
        Se True, verifica as dimensões de entrada de A e f. O padrão é True.
    
    Retorna:
    --------
    L : numpy.ndarray
        Matriz triangular inferior resultante da decomposição LU.
    U : numpy.ndarray
        Matriz triangular superior resultante da decomposição LU.
    """
    if check_input:
        # Verificações de entrada usando assert
        assert isinstance(A, np.ndarray), "A matriz A deve ser um numpy.ndarray."
        assert A.ndim == 2, "A matriz A deve ser 2D."
        assert A.shape[0] == A.shape[1], "A matriz A deve ser quadrada."
        assert isinstance(f, np.ndarray), "O vetor f deve ser um numpy.ndarray."
        assert f.ndim == 1, "O vetor f deve ser 1D."
        assert A.shape[0] == f.shape[0], "O número de linhas de A deve ser igual ao tamanho de f."
    
    # Executa a eliminação de Gauss para obter U, L e b modificados
    U, b, L = Gauss_elim(A, f, Mat_L=True)
    
    return [L, U]

def lu_decomp_pivoting(A, retornaLU = False, check_input=True):
    '''
    Computa a decomposição LU para uma matriz A aplicando pivotamento parcial.
    
    Parâmetros
    ----------
    A : numpy ndarray 2D
        Matriz quadrada do sistema linear.
    retornaLU : booleano
        Se True, decompõe a Matriz C em L U.
        Padrão é False, Retornando somente Matriz C.
    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.
    
    Retornos
    -------
    Se retornaLU == Falso:
        P : lista de inteiros
            Lista contendo as permutações.
        C : numpy array
            Matriz composta  da L (elementos abaixo da diagonal + identidade),
            e U (elementos acima da diagonal incluindo a diagonal).
    Se retornaLU == True:
        C : numpy array 2D
            Matriz Composta de L e U.
        L : numpy array 2D
            Matriz triangular inferior com elementos de L.
        U : numpy array 2D
            Matriz triangular superior com elementos de U.
        '''
    
    N = A.shape[0]
    if check_input:
        assert A.ndim == 2, 'A deve ser uma matriz'
        assert A.shape[1] == N, 'A deve ser quadrada'

    # Cria a matriz C como uma cópia de A
    C = A.copy()

    # Lista inicial de permutações
    P = list(range(N))

            
    # Decomposição LU com pivotamento parcial
    for k in range(N - 1):
        # Etapa de permutação
        p, C = permut(C, k)
        
        # Atualiza a lista de permutações
        P[k], P[p] = P[p], P[k]
        
        # Verifica se o pivô é diferente de zero
        assert C[k, k] != 0., 'pivô nulo!'
        
        # Calcula os multiplicadores de Gauss e armazena na parte inferior de C
        C[k+1:, k] = C[k+1:, k] / C[k, k]
        
        # Zera os elementos na k-ésima coluna
        C[k+1:, k+1:] = C[k+1:, k+1:] - np.outer(C[k+1:, k], C[k, k+1:])
    
    if retornaLU == True:
        # Separando L e U de C
        L = np.tril(C, -1) + np.eye(N)  # (-1) pega o Elementos abaixo da diagonal de C + diagonal de 1s
        U = np.triu(C)  # Elementos acima da diagonal de C (incluindo a diagonal)
        return P, L, U
    else:
        return P, C

def lu_solve(C, y, check_input=True):
    """
    Resolve o sistema linear LUx = y, onde C contém as matrizes L e U 
    resultantes da decomposição LU.

    Parâmetros:
    -----------
    C : list
        Lista contendo duas matrizes [L, U], onde:
        - L é a matriz triangular inferior (decomposição LU).
        - U é a matriz triangular superior (decomposição LU).
    
    y : np.ndarray
        Vetor independente do sistema linear LUx = y.
    
    check_input : bool, opcional
        Se True, verifica se as entradas são válidas. O padrão é True.

    Retorna:
    --------
    x : np.ndarray
        Vetor solução do sistema linear LUx = y.
    """
    # Verificação de entrada se o check_input estiver habilitado
    if check_input ==True:
        assert isinstance(C, list) and len(C) == 2, "C deve ser uma lista contendo [L, U]."
        assert isinstance(y, np.ndarray), "y deve ser um vetor (np.ndarray)."
        # Extrair L e U da lista C
        L, U = C
        assert L.shape[0] == L.shape[1] == U.shape[0] == U.shape[1], "Matrizes L e U devem ser quadradas."
        assert y.shape[0] == L.shape[0], "Dimensão de y deve ser compatível com L e U."
    else:
        # Extrair L e U da lista C
        L, U = C
    
    # Resolver o sistema Ly = y com substituição direta
    y_sol = triangular_inferior(L, y)
    # Resolver o sistema Ux = y_sol com substituição reversa
    x = triangular_superior(U, y_sol)
    return x

def lu_solve_pivoting(P, C, y, check_input=True):
    '''
    Resolve o sistema linear Ax = y utilizando a decomposição LU da matriz A 
    com pivotamento parcial.
    
    Parâmetros
    ----------
    P : lista de inteiros
        Lista contendo todas as permutações definidas para computar a decomposição LU 
        com pivotamento parcial (saída da função 'lu_decomp_pivoting').
    C : numpy ndarray 2D
        Matriz quadrada contendo os elementos de L abaixo da diagonal principal e 
        os elementos de U na parte superior (incluindo os elementos da diagonal principal).
        (Saída da função 'lu_decomp_pivoting').
    y : numpy ndarray 1D
        Vetor independente do sistema linear.
    check_input : booleano
        Se True, verifica se a entrada é válida. O padrão é True.
    
    Retornos
    -------
    x : numpy ndarray 1D
        Solução do sistema linear Ax=y.
    '''
    N = C.shape[0]
    
    if check_input:
        assert C.ndim == 2, 'C deve ser uma matriz'
        assert C.shape[1] == N, 'C deve ser quadrada'
        assert isinstance(P, list), 'P deve ser uma lista'
        assert len(P) == N, 'P deve ter N elementos'
        assert y.ndim == 1, 'y deve ser um vetor'
        assert y.size == N, 'O número de colunas de C deve ser igual ao tamanho de y'
    
    # Aplicar as permutações no vetor y
    y_permuted = y[P]

    # Separar L e U da matriz C
    L = np.tril(C, -1) + np.eye(N)  # Matriz L com diagonal de 1s
    U = np.triu(C)  # Matriz U

    # Passo 1: Resolver Lz = y_permuted (substituição direta)
    w = triangular_inferior(L, y_permuted)

    # Passo 2: Resolver Ux = z (substituição retroativa)
    x = triangular_superior(U, w)
    return x

# Decomposição de Cholesky

def cho_decomp(A, check_input=True):
    '''
    Compute the Cholesky decomposition of a symmetric and 
    positive definite matrix A. Matrix A is not modified.
    
    Parameters
    ----------
    A : numpy narray 2d
        Full square matrix of the linear system.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    G : numpy array 2d
        Lower triangular matrix representing the Cholesky factor of matrix A.
    '''
    N = A.shape[0]
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
        assert np.all(A.T == A), 'A must be symmetric'
    
    G = np.zeros((N, N))  # Initialize G as an NxN matrix of zeros
    for j in range(N):
        # Compute the diagonal element of G
        G[j, j] = A[j, j] - dot_real(G[j, :j], G[j, :j])
        if G[j, j] <= 0:
            raise ValueError("A is not positive definite")
        G[j, j] = np.sqrt(G[j, j])
        
        # Compute the off-diagonal elements of G
        if j < N - 1:
            G[j+1:, j] = (A[j+1:, j] - matvec_dot(G[j+1:, :j], G[j, :j])) / G[j, j]
    
    return G

def cho_decomp_overwrite(A, check_input=True):
    '''
    Compute the Cholesky decomposition of a symmetric and 
    positive definite matrix A. The lower triangle of A, including its main
    diagonal, is overwritten by its Cholesky factor.
    
    Parameters
    ----------
    A : numpy narray 2d
        Full square matrix of the linear system.
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    A : numpy array 2d
        Modified matrix A with its lower triangle, including its main diagonal, overwritten 
        by its corresponding Cholesky factor.
    '''
    N = A.shape[0]
    if check_input is True:
        assert A.ndim == 2, 'A must be a matrix'
        assert A.shape[1] == N, 'A must be square'
        assert np.all(A.T == A), 'A must be symmetric'
    
    for j in range(N):
        # Atualizar o elemento diagonal
        A[j, j] = np.sqrt(A[j, j] - np.dot(A[j, :j], A[j, :j]))
        
        if A[j, j] <= 0:
            raise ValueError("A matriz não é definida positiva")
        
        # Atualizar os elementos abaixo da diagonal
        for i in range(j+1, N):
            A[i, j] = (A[i, j] - np.dot(A[i, :j], A[j, :j])) / A[j, j]
        
        # Zerando a parte superior (opcional, mas pode ajudar na depuração)
        A[j, j+1:] = 0.0
    
    return A

def cho_inverse(G, check_input=True):
    '''
    Compute the inverse of a symmetric and positive definite matrix A 
    by using its Cholesky factor.
    
    Parameters
    ----------
    G : numpy narray 2d
        Cholesky factor of matrix A (output of function 'cho_decomp' or 'cho_decomp_overwrite').
    check_input : boolean
        If True, verify if the input is valid. Default is True.
    Returns
    -------
    Ainv : numpy array 2d
        Inverse of A.
    '''
    N = G.shape[0]
    if check_input is True:
        assert G.ndim == 2, 'G must be a matrix'
        assert G.shape[1] == N, 'G must be square'
    
    # Inicializar Ainv como uma matriz identidade
    Ainv = np.zeros((N, N))
    
    # Resolver para cada coluna de Ainv
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1  # vetor unitário
        # Resolver G @ z = e_i
        z = np.linalg.solve(G, e_i)
        # Resolver G.T @ Ainv[:, i] = z
        Ainv[:, i] = np.linalg.solve(G.T, z)
    
    return Ainv

# Transformada discrta de fourier
def DFT_matrix(N):
    """Cria a matriz de transformada discreta de Fourier (DFT)."""
    W = np.exp(-2j * np.pi / N)
    F = np.array([[W ** (n * k) for k in range(N)] for n in range(N)])
    return F

def DFT(x):
    """Calcula a Transformada Discreta de Fourier (DFT) de um vetor x."""
    N = len(x)
    F = DFT_matrix(N)
    return matvec_complex(F, x)

def IDFT_matrix(N):
    """Cria a matriz de transformada discreta de Fourier inversa (IDFT)."""
    W = np.exp(2j * np.pi / N)
    F_inv = np.array([[W ** (n * k) for k in range(N)] for n in range(N)])
    return F_inv

def IDFT(X):
    """Calcula a Transformada Discreta de Fourier inversa (IDFT) de um vetor X."""
    N = len(X)
    F_inv = IDFT_matrix(N)
    return np.dot(F_inv, X) / N


# Extra 
def mat_covariancia(H, v):
    N = v.shape[0]
    var = variancia(v)
    # Defina a matriz de covariância Σ_v (diagonal para elementos não correlacionados)
    sigma_v = var  # Variância de v
    Sigma_v = sigma_v ** 2 * np.eye(N)  # Matriz de covariância diagonal

    #Matriz de Covariancia
    Sigma_t = Sigma_v @ matmat_real_dot(H, H.T)
    return Sigma_t

def covariancia_parametros(A, d):
    """
    Calcula a matriz de covariância dos parâmetros estimados.
    
    Parameters:
    A (numpy.ndarray): Matriz de design.
    Sigma_d (numpy.ndarray): Matriz de covariância dos dados.
    
    Returns:
    numpy.ndarray: Matriz de covariância dos parâmetros estimados.
    """
    N = d.shape[0]
    var = variancia(dados=d)
    # Defina a matriz de covariância Σ_v (diagonal para elementos não correlacionados)
    sigma_d = var  # Variância de v
    Sigma_d = sigma_d ** 2 * np.eye(N)  # Matriz de covariância diagonal
    # Calcula a matriz de covariância dos parâmetros
    return np.linalg.inv(A.T @ np.linalg.inv(Sigma_d) @ A)

def variancia(dados):
    N = len(dados)
    media = np.mean(dados)
    soma = 0
    for i in range(0, N):
        soma = soma + (dados[i] - media)**2
    var = soma/(N-1)
    return var

def coeficiente_determinacao(y_true, y_pred):
    """
    Calcula o coeficiente de determinação R².
    
    Parâmetros:
    -----------
    y_true : array-like
        Valores observados (reais).
    y_pred : array-like
        Valores previstos pelo modelo.
    
    Retorna:
    --------
    r2 : float
        O coeficiente de determinação R².
    """
    # Converte para numpy arrays se não forem
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcula a soma dos quadrados dos resíduos (SSR)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calcula a soma total dos quadrados (SST)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Calcula o coeficiente de determinação (R²)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2

def erro_padrao_e_intervalo(param_cov_matrix, alfa=0.05):
    """
    Calcula os erros padrão dos parâmetros e seus intervalos de confiança.
    
    Parâmetros:
    -----------
    param_cov_matrix : array-like
        Matriz de covariância dos parâmetros estimados.
    alfa : float, opcional
        Nível de significância para o intervalo de confiança (padrão: 0.05 para 95% de confiança).
        
    Retorna:
    --------
    erros_padrao : np.ndarray
        Erros padrão dos parâmetros.
    intervalos_conf : np.ndarray
        Intervalos de confiança para os parâmetros (inferior, superior).
    """
    # Número de parâmetros
    num_params = param_cov_matrix.shape[0]
    
    # Erros padrão (raiz quadrada da variância, que está na diagonal da matriz de covariância)
    erros_padrao = np.sqrt(np.diag(param_cov_matrix))
    
    # Valor crítico para a distribuição normal (para o intervalo de confiança)
    z = 1.96  # Aproximadamente para 95% de confiança
    
    # Calcula os intervalos de confiança
    intervalos_conf = np.zeros((num_params, 2))
    for i in range(num_params):
        intervalo = z * erros_padrao[i]
        intervalos_conf[i, 0] = -intervalo
        intervalos_conf[i, 1] = intervalo
    
    return erros_padrao, intervalos_conf

# SQ line
def straight_line_matrix(x):
    """
    Cria a matriz de design para ajuste de linha reta.
    
    Parameters:
    x (numpy.ndarray): Vetor de pontos x.
    
    Returns:
    numpy.ndarray: Matriz de design para linha reta.
    """
    # Adiciona uma coluna de 1s para o termo de interceptação
    return np.vstack([x, np.ones_like(x)]).T

def straight_line(x, y):
    """
    Ajusta uma linha reta aos dados x e y e retorna os parâmetros da linha.
    
    Parameters:
    x (numpy.ndarray): Vetor de pontos x.
    y (numpy.ndarray): Vetor de pontos y.
    
    Returns:
    numpy.ndarray: Vetores de parâmetros [inclinação, interceptação].
    """
    A = straight_line_matrix(x)
    # Resolve o sistema de equações normais para obter os parâmetros
    return np.linalg.lstsq(A, y, rcond=None)[0]


#Somente para os residuos 
class Estatisticas:
    def __init__(self, residuo):
        """
        Inicializa o objeto com o vetor de resíduos.
        
        :param residuo: Lista ou vetor contendo os resíduos.
        """
        self.residuo = residuo
        self.media = self.calc_media()
        self.variancia = self.calc_variancia()
        self.desvio_padrao = self.calc_desvio_padrao()

    def calc_media(self):
        """Calcula a média dos resíduos."""
        return np.mean(self.residuo)

    def calc_variancia(self):
        """Calcula a variância dos resíduos."""
        return np.var(self.residuo, ddof=1)  # ddof=1 para variância amostral

    def calc_desvio_padrao(self):
        """Calcula o desvio padrão dos resíduos."""
        return np.std(self.residuo, ddof=1)  # ddof=1 para desvio padrão amostral

