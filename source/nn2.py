"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd import hessian_vector_product
from autograd.util import flatten
from autograd.util import flatten_func

from autograd.optimizers import adam, sgd
from data import load_mnist
from SCNaKS import *
import time
import matplotlib.pyplot as plt


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),  # weight matrix
             scale * rs.randn(n))  # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)


def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)


def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    return log_prior + log_lik


def accuracy(params, inputs, targets):
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


def scnaks_accuracy(flattened_params, unflattener, inputs, targets):
    target_class = np.argmax(targets, axis=1)
    unflattened_params = unflattener(flattened_params)
    predicted_class = np.argmax(neural_net_predict(unflattened_params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


if __name__ == '__main__':
    # Model parameters
    layer_sizes = [784, 200, 100, 10]
    L2_reg = 1.0

    # Training parameters
    param_scale = 0.1
    batch_size = 4096
    num_epochs = 3
    step_size = 0.001
    R = 3  # number of vectors in subspace
    ss = 1  # the step size for scnaks

    print("Loading training data...")
    N, train_images, train_labels, test_images, test_labels = load_mnist()

    init_params = init_random_params(param_scale, layer_sizes)

    num_batches = int(np.ceil(len(train_images) / batch_size))


    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx + 1) * batch_size)


    # Define training objective
    def objective(params, iter):
        idx = batch_indices(iter)
        # print("idx ",idx)
        return -log_posterior(params, train_images[idx], train_labels[idx], L2_reg)


    print("Training now.")

    flattened_objective, unflatten, flat_init_params = flatten_func(objective, init_params)

    # Initialize global variables that contain training/testing info.
    scnaks_epoch_list = []
    scnaks_training_acc_list = []
    scnaks_testing_acc_list = []
    scnaks_time_list = []
    epoch_list = []
    training_acc_list = []
    testing_acc_list = []
    time_list = []
    sgd_epoch_list = []
    sgd_training_acc_list = []
    sgd_testing_acc_list = []
    sgd_time_list = []

    start = time.time()


    def print_scnaks_perf(flattened_params, iter):
        if iter % num_batches == 0:
            end = time.time()
            temp_time = (end - start)
            train_acc = scnaks_accuracy(flattened_params, unflatten, train_images, train_labels)
            test_acc = scnaks_accuracy(flattened_params, unflatten, test_images, test_labels)
            print("{:15}|{:20}|{:20}|{:20}".format(iter // num_batches, train_acc, test_acc, temp_time))
            scnaks_epoch_list.append(iter // num_batches)
            scnaks_training_acc_list.append(train_acc)
            scnaks_testing_acc_list.append(test_acc)
            scnaks_time_list.append(temp_time)


    objective_grad = grad(flattened_objective)
    objective_hvp = hessian_vector_product(flattened_objective)
    print("SCNAKS Epoch   |    Train accuracy  |       Test accuracy|   Elapsed Time (seconds)  ")
    scnaks(objective_grad, objective_hvp, flat_init_params, callback=print_scnaks_perf, r=R, gs=1,
           num_iters=num_epochs * num_batches, step_size=ss, eps=2 ** -16)

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    start = time.time()


    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            end = time.time()
            temp_time = (end - start)
            train_acc = accuracy(params, train_images, train_labels)
            test_acc = accuracy(params, test_images, test_labels)
            print("{:15}|{:20}|{:20}|{:20}".format(iter // num_batches, train_acc, test_acc, temp_time))
            epoch_list.append(iter // num_batches)
            training_acc_list.append(train_acc)
            testing_acc_list.append(test_acc)
            time_list.append(temp_time)
