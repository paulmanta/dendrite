import numpy
from nose.tools import *
from app.functions import *
from tests.unit.commons import *


def test_identity_return():
    x = uniform_float_matrix()
    y = identity(x)
    assert_true(numpy.array_equal(x, y))


def test_identity_gradient():
    verify_activation_gradient(identity)


def test_sigmoid_range():
    x = uniform_float_matrix()
    y = sigmoid(x)
    assert_true(numpy.all(0 < y) and numpy.all(y < 1))


def test_sigmoid_gradient():
    verify_activation_gradient(sigmoid)


def test_mean_squared_error_zero():
    verify_zero_error(mean_squared_error)


def test_mean_squared_error_gradient():
    verify_error_gradient(mean_squared_error)


def test_mean_absolute_error_zero():
    verify_zero_error(mean_absolute_error)


def test_mean_absolute_error_gradient():
    verify_error_gradient(mean_absolute_error)


def test_mean_logistic_error_gradient():
    y = uniform_float_matrix(1e-4, 1.0, (2, 2))
    t = uniform_integer_matrix(0, 2, y.shape)
    verify_error_gradient(mean_logistic_error, y, t, dy=1e-10, tol=1e-3)


def verify_activation_gradient(func, x=None, dx=1e-4, tol=1e-4):
    x = uniform_float_matrix() if x is None else x

    analytic = func(x, gradient=True)
    numeric = compute_numerical_activation_gradient(func, x, dx)

    assert_true(numpy.allclose(analytic, numeric, rtol=0, atol=tol))


def compute_numerical_activation_gradient(f, x, epsilon):
    gradient = numpy.zeros(x.shape)
    dx = numpy.zeros(x.shape)

    for i in range(x.size):
        dx.flat[i] = epsilon
        gradient += f(x + dx) - f(x)
        dx.flat[i] = 0

    gradient /= epsilon
    return gradient


def verify_error_gradient(func, y=None, t=None, dy=1e-4, tol=1e-4):
    y = uniform_float_matrix() if y is None else y
    t = uniform_float_matrix(shape=y.shape) if t is None else t

    analytic = func(y, t, gradient=True)
    numeric = compute_numerical_error_gradient(func, y, t, dy)

    assert_true(numpy.allclose(analytic, numeric, rtol=0, atol=tol))


def verify_zero_error(func, t=None):
    t = uniform_float_matrix() if not t else t
    e = mean_squared_error(t, t)
    assert_true(numpy.all(e == 0))


def compute_numerical_error_gradient(f, y, t, epsilon):
    gradient = numpy.zeros(y.shape)
    dy = numpy.zeros(y.shape)

    for i in range(y.size):
        dy.flat[i] = epsilon
        gradient.flat[i] += f(y + dy, t) - f(y, t)
        dy.flat[i] = 0

    gradient /= epsilon
    return gradient
