import numpy
import sys


def identity(x, gradient=False):
    return numpy.ones(x.shape) if gradient else x


def sigmoid(x, gradient=False):
    eps = sys.float_info.epsilon
    y = numpy.clip(1 / (1 + numpy.exp(-x)), eps, 1 - eps)
    return numpy.multiply(y, 1 - y) if gradient else y


def mean_absolute_error(y, t, gradient=False):
    if gradient:
        return ((y - t) / numpy.abs(y - t)) / y.shape[1]
    return numpy.sum(numpy.abs(t - y)) / y.shape[1]


def mean_squared_error(y, t, gradient=False):
    if gradient:
        return (y - t) / y.shape[1]
    return numpy.sum(numpy.square(y - t)) / (2 * y.shape[1])


def mean_logistic_error(y, t, gradient=False):
    from numpy import divide, multiply
    from numpy import log, sum

    if gradient:
        return divide(y - t, multiply(y, 1 - y)) / y.shape[1]

    errors = multiply(t, -log(y)) + multiply(1 - t, -log(1 - y))
    return sum(errors) / y.shape[1]
