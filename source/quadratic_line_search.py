"""A simple example of quadratic line search (i.e r=0)."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.convenience_wrappers import hessian_vector_product as hvp
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x0 = np.array([4.3, 3.9])
path_=[x0]

def ellipse(x): # x is a np array of length 2 
    # this is a rosenbrock function where a=1, b=100
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def qls(f,x0,tol=1e-12): # performs quadratic line search
    g=grad(f) # gradient object
    h=hvp(f) # hvp object
    x_old=x0
    alpha_new=np.sum(g(x_old)**2)/np.sum(g(x_old)*h(x_old,g(x_old)))
    x_new=x_old-alpha_new*g(x_old)
    err=err=np.linalg.norm(x_new-x_old)/np.linalg.norm(x_old)
    i=1
    while (err>tol):
        i+=1
        x_old=x_new
        alpha_new=np.sum(g(x_old)**2)/np.sum(g(x_old)*h(x_old,g(x_old)))
        x_new=x_old-alpha_new*g(x_old)
        print(i, x_new, err)
        err=np.linalg.norm(x_new-x_old)/np.linalg.norm(x_old)
        path_.append(x_new)
    return(x_new)


def contour_plot(f, path, x1_min=-4.5, x1_max=4.5, x2_min=-4.5, x2_max=4.5, h=0.01):
    x1, x2=np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    mesh=np.c_[x1.ravel(), x2.ravel()] # creates our 2d mesh
    z=np.array([f(mesh[i]) for i in xrange(mesh.shape[0])]) # applies function to mesh
    z=z.reshape(x1.shape)
    plt.contourf(x1, x2, z, levels=550*np.array(range(120)), cmap=cm.RdBu, alpha=0.5)
    plt.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, color='k', width=0.0015, headwidth=3)
    plt.plot(1,1, 'r*', markersize=12)
    plt.show()


qls(f=ellipse,x0=x0,tol=1e-8)
path = np.array(path_).T
print(path.shape)
contour_plot(f=ellipse, path=path)