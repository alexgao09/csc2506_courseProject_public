#from funkyyak import grad, numpy_wrapper as np
from autograd import grad
import autograd.numpy as np

from functools import partial

from util import mnist
from numerics import hvp, sliced_hvp

class WeightsParser(object):
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

def make_batches(N_total, N_batch):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + N_batch))
        start += N_batch
    return batches

def logsumexp(X, axis):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

def make_nn_funs(layer_sizes, L2_reg):
    parser = WeightsParser()
    for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        parser.add_weights('Layer ' + str(i + 1) + ' Weights', shape)
        parser.add_weights('Layer ' + str(i + 1)+ ' Biases', (1, shape[1]))

    def predictions(W_vect, X):
        cur_units = X
        for i in range(len(layer_sizes) - 1):
            cur_W = parser.get(W_vect, 'Layer ' + str(i + 1) + ' Weights')
            cur_B = parser.get(W_vect, 'Layer ' + str(i + 1) + ' Biases')
            cur_units = np.tanh(np.dot(cur_units, cur_W) + cur_B)
        return cur_units - logsumexp(cur_units, axis=1)

    def loss(W_vect, X, T, idxs):
        """Include a slice in case we want to do SGD."""
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predictions(W_vect, X[idxs]) * T[idxs])
        return (-log_prior - log_lik)/T.shape[0]

    return parser, loss

def build_nn_objective(num_hidden=5, num_data=1000):
    """Builds a neural net, creates weights and data from that net,
    then defines the objective as the training error."""

    # Load and process MNIST data (borrowing from Kayak)
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

    edge_pixels_removed = 4  # on each edge.
    subsample_pixels = 5     # in both directions.
    images, labels = mnist()
    images = images[:, edge_pixels_removed:-edge_pixels_removed,
                       edge_pixels_removed:-edge_pixels_removed]   # Remove edges
    images = images[:num_data,::subsample_pixels,::subsample_pixels]   # Subsample data
    images = partial_flatten(images) / 255.0   # After this, train_images is N by (x * y)
    labels = one_hot(labels, 10)[:num_data,:]  #TODO: Randomize order?

    # Build the network.
    layer_sizes = [images.shape[1]] + num_hidden + [10]
    L2_reg = 0
    parser, loss, = make_nn_funs(layer_sizes, L2_reg)

    # Build functions to interrogate the objective at a particular set of parameters.
    def objective(x, idxs=slice(0,num_data)):
        return loss(x, X=images, T=labels, idxs=idxs)
    obj_grad = grad(objective)
    obj_hvp = sliced_hvp(obj_grad)
    weights_subsets = {k:v[0] for k, v in parser.idxs_and_shapes.iteritems()}
    return parser.N, objective, obj_grad, obj_hvp, weights_subsets


def make_logistic_funs(in_size, out_size, L2_reg):
    parser = WeightsParser()
    parser.add_weights('weights', (in_size, out_size))
    parser.add_weights('biases', (1, out_size))

    def predictions(W_vect, X):
        cur_W = parser.get(W_vect, 'weights')
        cur_B = parser.get(W_vect, 'biases')
        cur_units = np.dot(X, cur_W) + cur_B
        return cur_units - logsumexp(cur_units, axis=1)

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T)
        return (-log_prior - log_lik)/T.shape[0]

    return parser, loss


def build_logistic_objective():
    """Builds a neural net, creates weights and data from that net,
    then defines the objective as the training error."""

    # Load and process MNIST data (borrowing from Kayak)
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

    subsample_pixels = 5
    subsample_data = 10
    images, labels = mnist()
    images = images[::subsample_data,::subsample_pixels,::subsample_pixels]       # Subsample data
    images = partial_flatten(images) / 255.0   # After this, train_images is N by (x * y)
    labels = one_hot(labels, 10)[::subsample_data,:]

    # Build the network.
    L2_reg = 0
    parser, loss, = make_logistic_funs(images.shape[1], 10, L2_reg)

    # Build functions to interrogate the objective at a particular set of parameters.
    objective = partial( loss, X=images, T=labels)
    obj_grad = grad(objective)
    obj_hvp = sliced_hvp(obj_grad)
    weights_subsets = {k:v[0] for k, v in parser.idxs_and_shapes.iteritems()}
    return parser.N, objective, obj_grad, obj_hvp, weights_subsets

def branin(x):
    x0 = x[0]*15
    x1 = (x[1]*15)-5
    b = np.square(x1 - (5.1/(4*np.square(np.pi)))*np.square(x0)
                + (5/np.pi)*x0 - 6) + 10*(1-(1./(8*np.pi)))*np.cos(x0) + 10
    return b + np.sum((x*np.arange(x.size))**2)

def build_branin_objective(D=100):
    obj_grad = grad(branin)
    obj_hvp = sliced_hvp(obj_grad)
    return D, branin, obj_grad, obj_hvp, {}



