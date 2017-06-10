#from funkyyak import grad, numpy_wrapper as np
from autograd import grad
import autograd.numpy as np

import numpy.random as npr
from numpy.linalg import norm
from scipy.linalg import eig_banded

def compute_spectrum(hess_vec_prod, size, method="exact", max_steps=10):
    """Returns an array containing sorted eigenvalues and eigenvectors.
       hess_vec_prod takes a vector only"""
    # The slow, N^3 way.
    if method == "exact":
        hessian = matrix_from_mvp(hess_vec_prod, size)
        return np.linalg.eigvals(hessian)
    elif method == "approx":
        a, b = lanczos_iteration(hess_vec_prod, size, max_steps)
        banded = np.concatenate((b[None,:], a[None,:]), axis=0)
        eigvals, eigvects = eig_banded(banded)
        return subset_to_full(eigvals, eigvects[0, :]**2, size)
    else:
        raise ValueError("Unrecognized method {0}".format(method))


def hvp_eig(hess_vec_prod, size):
    """Returns an array containing sorted eigenvalues and eigenvectors.
       hess_vec_prod takes a vector only, the slow, N^3 way."""
    hessian = matrix_from_mvp(hess_vec_prod, size)
    return np.linalg.eig(hessian)

def subset_to_full(X, weights, N):
    weights = weights / np.sum(weights) * (N - 1)
    full_eigs = []
    cum = 0
    i = 0
    for x, w in zip(X, weights):
        cum += w
        while i < cum + 1e-6:
            full_eigs.append(x)
            i += 1
    return np.array(full_eigs)

def lanczos_iteration(hess_vec_prod, size, max_steps, orthogonalize=True):
    """Computes a similar tridiagional matrix, using
       Algorithm 10.1.1 from Matrix Computations by Golub + van Loan"""
    npr.seed(0)
    q = np.zeros(size)
    all_q = []
    r = npr.randn(size)
    r = r / norm(r)
    a = np.zeros(max_steps)   # Diagonals.
    b = np.zeros(max_steps)   # Off-diagonals.
    b[0] = 1
    for k in xrange(max_steps):
        new_q = r / b[k]
        if orthogonalize:
            new_q = orthogonalize_vector(new_q, all_q)
            all_q.append(new_q)
        hvp = hess_vec_prod(new_q)
        a[k] = np.dot(new_q, hvp)
        if k == max_steps -1 : break
        r = hvp - a[k]*new_q - b[k]*q
        b[k+1] = norm(r)
        q = new_q
    return a, b

def orthogonalize_vector(v, basis_vects):
    for e in basis_vects:
        v = v - np.dot(v, e) * e
    v = v / np.linalg.norm(v)
    return v

def matrix_from_mvp(matrix_vector_product, size):
    m = np.zeros((size, size))
    directions = np.eye(size)
    for ix, d in enumerate(directions):   # Take a step in an axis-aligned direction.
        m[ix, :] = matrix_vector_product(d)
    return m

def inverse_Hessian_from_hvp(x, hess_vec_prod):
    # The slow, N^3 way.
    hessian = matrix_from_mvp(x, hess_vec_prod)
    return np.linalg.inv(hessian)

def hvp(f_grad):
    """Builds the exact Hessian-vector product.
       Because the Hessian is symmetric, there are two ways to take the Hessian-vector product:
       1. As the change in gradient as we move in a direction, and
       2. As the change in the directional gradient as we move the input."""
    def grad_prod(x, d):
        return np.dot(f_grad(x), d)
    return grad(grad_prod)  # Has args (x, d)

def sliced_hvp(f_grad):
    """This version returns a function that also takes in a slice
       for the direction vector."""
    f_hvp = hvp(f_grad)
    def partial_hvp(x, d, subset):
        expanded_dir = np.zeros(len(x))
        expanded_dir[subset] = d
        return f_hvp(x, expanded_dir)[subset]
    return partial_hvp
