import numpy
from nose.tools import *
from app.functions import *
from app.neural import *
from app.utility import *
from tests.unit.commons import *


def test_backprop_weight_gradient():
    network = random_neural_net(biased=False)
    samples = random_samples(network=network)
    error = mean_squared_error

    analytic, _ = backprop(network, samples, error)
    numeric = numeric_backprop_gradient(network, samples, error, 'weights')

    assert_true(numpy.allclose(analytic.raveled, numeric, rtol=0, atol=1e-4))


def test_backprop_bias_gradient():
    network = random_neural_net(biased=True)
    samples = random_samples(network=network)
    error = mean_squared_error

    _, analytic = backprop(network, samples, error)
    numeric = numeric_backprop_gradient(network, samples, error, 'biases')

    assert_true(numpy.allclose(analytic.raveled, numeric, rtol=0, atol=1e-4))


def numeric_backprop_gradient(network, samples, error, attribute):
    raveled = getattr(network, attribute).raveled
    gradient = numpy.zeros(raveled.shape)
    delta = 1e-8

    for i in range(raveled.size):
        raveled[i] += delta
        gradient.flat[i] = error(network(samples.inputs), samples.targets)
        raveled[i] -= delta
        gradient.flat[i] -= error(network(samples.inputs), samples.targets)

    gradient /= delta
    return gradient


def random_neural_net(**kwargs):
    architecture = kwargs.get('architecture', numpy.random.randint(1, 10, 4))
    biased = kwargs.get('biased', numpy.random.choice([True, False]))
    activations = tuple([sigmoid] * (len(architecture) - 1))
    config = NeuralNetConfig(architecture, activations, biased)

    weight_shapes, bias_shapes = config.shapes

    num_weights = sum(x * y for x, y in weight_shapes)
    raveled_weights = numpy.random.uniform(-5.0, 5.0, num_weights)
    weights = RaveledMatrixList(weight_shapes, raveled_weights)

    if biased:
        num_biases = sum(x * y for x, y in bias_shapes)
        raveled_biases = numpy.random.uniform(-5.0, 5.0, num_biases)
        biases = RaveledMatrixList(bias_shapes, raveled_biases)

    return NeuralNet(config, weights, biases if biased else None)
