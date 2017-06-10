import numpy as np
from contextlib import contextmanager
from time import time
from functools import partial
import gzip, struct, array

def slicedict(d, ixs):
    return {k : v[ixs] for k, v in d.iteritems()}

class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        return self.cache.setdefault(args, self.func(*args))

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

def normalize_array(A):
    mean, std = np.mean(A), np.std(A)
    A_normed = (A - mean) / std
    def restore_function(X):
        return X * std + mean

    return A_normed, restore_function

@contextmanager
def tictoc(name=""):
    print "--- Start clock {0} ---".format(name)
    t1 = time()
    yield
    dt = time() - t1
    print "--- Stop clock {0} : {1} seconds elapsed ---".format(name, dt)


def mnist():
    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    #train_images = parse_images('data/train-images-idx3-ubyte.gz')
    #train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images  = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels  = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    #return train_images, train_labels, test_images, test_labels
    return test_images, test_labels

