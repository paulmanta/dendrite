import numpy
from app.utility import *


def uniform_integer_matrix(low=-10, high=10, shape=None):
    shape = numpy.random.randint(10, 20, 2) if shape is None else shape
    return numpy.random.randint(low, high, shape)


def uniform_float_matrix(low=-10.0, high=10.0, shape=None):
    shape = numpy.random.randint(10, 20, 2) if shape is None else shape
    return numpy.random.uniform(low, high, shape)


def random_samples(**kwargs):
    if 'nsamples' in kwargs and ('xsamples' in kwargs or 'tsamples' in kwargs):
        raise KeyError('Incorrect sample size keyword arguments combination.')
    if 'network' in kwargs and ('xsize' in kwargs or 'tsize' in kwargs):
        raise KeyError('Incorrect feature size keyword arguments combination.')

    if 'network' in kwargs:
        network = kwargs.pop('network')
        xsize = network.architecture[0]
        tsize = network.architecture[-1]
        return random_samples(xsize=xsize, tsize=tsize, **kwargs)

    nsamples = kwargs.get('nsamples', numpy.random.randint(1, 50))
    xsamples = kwargs.get('xsamples', nsamples)
    tsamples = kwargs.get('tsamples', nsamples)
    xsize = kwargs.get('xsize', numpy.random.randint(1, 10))
    tsize = kwargs.get('tsize', numpy.random.randint(1, 10))

    inputs = uniform_float_matrix(shape=(xsize, xsamples))
    targets = uniform_float_matrix(shape=(tsize, tsamples))

    return Samples(inputs, targets)
