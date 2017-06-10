# Simple example of computing Eigenspectrum of a Hessian
# of a neural network training objective function

import numpy as np
from functools import partial

import matplotlib.pyplot as plt

from optimization_routines import gd_with_momentum, Newton, sgd_with_momentum, rms_prop, cg, cg_minibatches
from objectives import build_nn_objective, build_branin_objective, build_logistic_objective
from numerics import compute_spectrum
from plotting_utils import plot_along_eigendirections, eigenscape_array, ribbon_array

num_data = 1000
num_epochs = 50
learn_rate = 0.01
init_param_scale = 0.1
momentum = 0.98
plot_eigenscapes=True
eigenscape_subsampling=10
plot_eigendirections=False
num_eigenvalues = 20

#num_weights, objective, gradfun, hess_vec_prod, subset_dict = build_branin_objective(10)
num_weights, objective, gradfun, hess_vec_prod, subset_dict = build_nn_objective(num_hidden=[18], num_data=num_data)
#num_weights, objective, gradfun, hess_vec_prod, subset_dict = build_logistic_objective()

print "Number of parameters:", num_weights
if subset_dict is None: subset_dict = {}
subset_dict['All Weights'] = slice(0,num_weights)    # Examine all the weights together by default.

# Initialize all the result-holding matrices.
sampled_epochs = range(num_epochs)[0:num_epochs:eigenscape_subsampling]  # Which epochs to look at.
loss_array = np.zeros((num_epochs, 1))
spectrum_array = {}
grad_size_array = {}
for subset_name, subset_ix in subset_dict.iteritems():
    spectrum_array[subset_name] = np.zeros((len(sampled_epochs), subset_ix.stop - subset_ix.start))
    grad_size_array[subset_name] = np.zeros((num_epochs, 1))

bottom_line = 1.3

def callback(epoch, x):
    """Save statistics about the local properties of the function being optimized."""
    loss_array[epoch] = objective(x, slice(0,num_data))
    for subset_name, subset_ix in subset_dict.iteritems():
        if plot_eigenscapes and epoch in sampled_epochs:
            subset_hvp = partial(hess_vec_prod, x, subset=subset_ix)
            eigenvalues = compute_spectrum(subset_hvp, subset_ix.stop - subset_ix.start,
                                           method='approx', max_steps=num_eigenvalues)
            spectrum_array[subset_name][sampled_epochs.index(epoch), :] = sorted(np.real(eigenvalues))
        grad_size_array[subset_name][epoch] = np.sqrt(np.sum(gradfun(x)**2))
    print "Epoch:", epoch, "Objective:", objective(x)
    if plot_eigendirections and epoch % 10 == 0:
        plot_along_eigendirections(num_weights, x, objective, hess_vec_prod)

# Optimize the objective function.
#final_x = rms_prop(gradfun, num_data, num_weights, callback=callback,
#                            num_epochs=num_epochs, learn_rate=learn_rate,
#                            momentum=momentum, param_scale=init_param_scale)

final_x = cg(objective, gradfun, num_weights, callback=callback,
                            num_epochs=num_epochs, param_scale=init_param_scale)

if plot_eigenscapes:
    print "\nMaking plots..."
    last_ax = eigenscape_array(spectrum_array, subset_dict, sampled_epochs)
    last_ax.set_xlabel('Epoch')
    plt.show()

    ribbon_array(spectrum_array, subset_dict, sampled_epochs)
    last_ax = last_ax.set_xlabel('Epoch')
    plt.show()

# Plot learning curves.
fig = plt.figure()
ax = plt.subplot(2,1,1)
ax.plot(loss_array)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')

ax = plt.subplot(2,1,2)
ax.plot(grad_size_array['All Weights'])
ax.set_xlabel('Iteration')
ax.set_ylabel('Gradient magnitude')
plt.show()

print "Done"