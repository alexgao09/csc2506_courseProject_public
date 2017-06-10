import numpy as np
import numpy.random as npr
from scipy.optimize import fmin_cg

from numerics import inverse_Hessian_from_hvp

def gd_with_momentum(gradfun, num_weights, callback=None, num_epochs=100, learn_rate=0.1,
                      momentum=0.9, param_scale=0.1):
    """Stochastic gradient descent with momentum."""
    npr.seed(0)
    w = npr.randn(num_weights) * param_scale
    cur_dir = np.zeros(num_weights)

    for epoch in xrange(num_epochs):
        cur_grad = gradfun(w)
        cur_dir = momentum * cur_dir - (1.0 - momentum) * cur_grad
        w += learn_rate * cur_dir
        if callback: callback(epoch, w)
    return w


def sgd_with_momentum(gradfunc, num_training_examples, num_weights, callback=None,
                      batch_size=100, num_epochs=100, learn_rate=0.1,
                      mass=0.9, param_scale=0.1):
    """Stochastic gradient descent with momentum."""
    weights = npr.randn(num_weights) * param_scale   # Initialize with random weights.
    velocity = np.zeros(num_weights)
    batches = batch_idx_generator(batch_size, num_training_examples)
    for epoch in xrange(num_epochs):
        for batch in batches:
            cur_grad = gradfunc(weights, batch)
            velocity = mass * velocity - (1.0 - mass) * cur_grad
            weights += learn_rate * velocity
        if callback: callback(epoch, weights)
    return weights

def rms_prop(grad, N_x, N_w, callback=None,
             batch_size=100, num_epochs=100, learn_rate=0.1,
             param_scale=0.1, gamma=0.9):
    """Root mean squared prop: See Adagrad paper for details."""
    npr.seed(0)
    w = npr.randn(N_w) * param_scale
    avg_sq_grad = np.ones(N_w)
    batches = batch_idx_generator(batch_size, N_x)
    for epoch in xrange(num_epochs):
        for batch in batches:
            cur_grad = grad(w, batch)
            avg_sq_grad = avg_sq_grad * gamma + cur_grad**2 * (1 - gamma)
            w -= learn_rate * cur_grad/np.sqrt(avg_sq_grad)
        if callback: callback(epoch, w)
    return w


global epoch

def cg(objfun, gradfun, num_weights, callback=None, num_epochs=100, param_scale=0.1):
    """Conjugate gradients."""
    init_x = npr.randn(num_weights) * param_scale   # Initialize with random weights.

    global epoch
    epoch = 0
    def wrapped_callback(x):
        global epoch
        callback(epoch, x)
        epoch += 1

    return fmin_cg(objfun, init_x, fprime=gradfun, maxiter=num_epochs, callback=wrapped_callback)

def endless_batch_generator():
    pass


def cg_minibatches(objfun, gradfun, num_training_examples, num_weights, callback=None,
                      batch_size=200, num_epochs=100, param_scale=0.1):
    """Conjugate gradients with minibatches."""
    weights = npr.randn(num_weights) * param_scale   # Initialize with random weights.
    batches = batch_idx_generator(batch_size, num_training_examples)
    for epoch in xrange(num_epochs):
        for batch in batches:
            cur_grad = gradfunc(weights, batch)
            velocity = mass * velocity - (1.0 - mass) * cur_grad
            weights += learn_rate * velocity
        if callback: callback(epoch, weights)
    return weights




def Newton(grad, hess_vec_product, num_weights, callback=None, num_epochs=100, step_size=1, init_scale=0.1):
    """Vanilla Newton's method."""
    x = npr.randn(num_weights) * init_scale

    for epoch in xrange(num_epochs):
        invHessian = inverse_Hessian_from_hvp(x, hess_vec_product)
        cur_grad = grad(x)
        x += -np.dot(invHessian, cur_grad) * step_size
        if callback: callback(epoch, x)
    return x

def SaddleFree():
    pass

def make_batcher(input_data, batch_size):
    batch_idxs = batch_idx_generator(batch_size, len(input_data.values()[0]))
    data_batches = [{k : v[idxs] for k, v in input_data.iteritems()}
                    for idxs in batch_idxs]
    def batcher():
        for data_batch in data_batches:
            for node, value in data_batch.iteritems():
                node.value = value
            yield

    return batcher

def batch_idx_generator(batch_size, total_size):
    start = 0
    end = batch_size
    batches = []
    while True:
        if start >= total_size:
            break
        batches.append(slice(start, end))
        start += batch_size
        end += batch_size

    return batches
