import autograd.numpy as np


def norm_orth(u0, u, gs):
    if gs: u0 = u0 - np.dot(u, np.dot(u.T, u0))
    return u0 / np.linalg.norm(u0)


def scnaks(grad, hvp, parms, r=2, num_iters=100, gs=True,
            step_size=1, eps=2**-8, callback=None):
    v, hv = np.zeros((parms.size, r+1)), np.zeros((parms.size, r+1))
    for curr_iter in range(num_iters):
        if callback: callback(parms, curr_iter)
        g = grad(parms, curr_iter)
        v[:, 0] = g/np.linalg.norm(g)
        hv[:, 0] = hvp(parms, curr_iter, v[:, 0])
        for j in range(r):
            v[:, j+1] = norm_orth(hv[:, j], v[:, 0:(j+1)], gs)
            hv[:, j+1] = hvp(parms, curr_iter, v[:, j+1])
        vhv, vg = v.T.dot(hv), v.T.dot(g)
        d, e = np.linalg.eigh(vhv)
        vhv_inv = np.dot(1/(np.abs(d)+eps*np.max(np.abs(d)))*e, e.T)
        parms -= step_size * v.dot(vhv_inv.dot(vg))
    return parms


def scnaksm(grad, hvp, parms, r=2, num_iters=100, gs=True,
            step_size=1, b1=0.9, b2=0.999, eps=2**-8, callback=None):
    v, hv = np.zeros((parms.size, r+1)), np.zeros((parms.size, r+1))
    m, n = np.zeros(parms.size), np.zeros(parms.size)
    for curr_iter in range(num_iters):
        if callback: callback(parms, curr_iter)
        g = grad(parms, curr_iter)
        v[:, 0] = g/np.linalg.norm(g)
        hv[:, 0] = hvp(parms, curr_iter, v[:, 0])
        for j in range(r):
            v[:, j+1] = norm_orth(hv[:, j], v[:, 0:(j+1)], gs)
            hv[:, j+1] = hvp(parms, curr_iter, v[:, j+1])
        vhv, vg = v.T.dot(hv), v.T.dot(g)
        d, e = np.linalg.eigh(vhv)
        vhv_abs_inv = np.dot(1/(np.abs(d)+eps*np.max(np.abs(d)))*e, e.T)
        step = v.dot(vhv_abs_inv.dot(vg))
        m = (1 - b1) * step      + b1 * m  # First  moment estimate.
        n = (1 - b2) * (step**2) + b2 * n  # Second moment estimate.
        mhat = m / (1 - b1**(curr_iter + 1))    # Bias correction.
        vhat = n / (1 - b2**(curr_iter + 1))
        parms -= step_size*mhat/(np.sqrt(vhat) + eps)
    return parms
