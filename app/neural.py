import numpy
from app.functions import *
from app.utility import *


class NeuralNetConfig():
    """
    Configuration options for building a :class:`.NeuralNet`.

    This is used together with :class:`.BackpropTrainer` to create and
    train a :class:`.NeuralNet`. Instances of this class are immutable.

    """

    def __init__(self, architecture, activations, biased):
        """
        Arguments:
          architecture (int tuple): Tuple to specify the number of layers and
            number of neurons per each layer. The network will have as many
            layers as there are elements in the tuple, and each layer will have
            as many neurons as specified by the corresponding tuple element.

          activations (function tuple): Activation functions to be used for all
            layers except the input layer. The activation functions must be one
            fewer than the number of layers.

          biased (bool): Flag to indicate whether the network should use bias
            units.

        Raises:
          ValueError: Incorrect number of activation functions.

        """

        if len(architecture) != len(activations) + 1:
            raise ValueError('Incorrect number of activation functions.')

        self._architecture = architecture
        self._activations = activations
        self._biased = biased

    @property
    def layer_count(self):
        """ (int) Number of layers """
        return len(self.architecture)

    @property
    def architecture(self):
        """ (int tuple) Number of neurons for each layer """
        return self._architecture

    @property
    def activations(self):
        """ (function tuple) Activation functions """
        return self._activations

    @property
    def biased(self):
        """ (bool) Indicates whether the network should use bias units. """
        return self._biased

    @property
    def shapes(self):
        """
        (tuple) A tuple of two lists, `ws` and `bs` of shapes for the weight
        and bias arrays. If the network does not have bias unit, `bs` will be
        an empty list.

        """
        ws = zip(self.architecture[1:], self.architecture)
        bs = [(x, 1) for x in self.architecture[1:]] if self.biased else []
        return tuple(ws), tuple(bs)


class NeuralNet():
    """
    Feed-forward neural network.

    This class can be used to define feed-forward neural networks with any
    number of layers and any number of neurons per layer, and the activation
    functions can be set individually for each layer. All these configurations
    are done through :class:`.NeuralNetConfig`.

    Upon creation, the weights (and biases) used by the network also need to
    be specified. Creating networks is typically done not directly but through
    :class:`.BackpropTrainer`. This is also the only way to train a network,
    since the `NeuralNet` class only offers methods for evaluating the network;
    the training algorithms are implemented only in `BackpropTrainer`.

    """

    def __init__(self, config, weights, biases=None):
        """
        Arguments:
          config (:class:`.NeuralNetConfig`): This network's configuration,
            which specifies the architecture, the activation functions, etc.

          weights (:class:`app.utility.RaveledMatrixList`): List of weight
            matrices to be used by the network.

          biases (:class:`app.utility.RaveledMatrixList`, optional): List of
            bias vectors. Must be specified iff `config.biased` is `True`.

        Raises:
          ValueError: Incorrect number of weight matrices, or incorrect number
            of bias vectors, or disagreement between the `biases` argument and
            `config.biased`.

        """
        if config.layer_count != len(weights) + 1:
            raise ValueError('Incorrect number of weight matrices.')

        if config.biased and (biases is None):
            raise ValueError('Required bias vectors are not specified.')

        if (not config.biased) and (biases is not None):
            raise ValueError('Superfluous bias vectors are specified.')

        if config.biased and (len(weights) != len(biases)):
            raise ValueError('Incorrect number of bias vectors.')

        self._config = config
        self._weights = weights
        self._biases = biases

    @staticmethod
    def uniform(config, span):
        """
        Creates a neural network with uniformly initialized weights in the
        range `-span`, `span`. Biases are initialized in the same way if the
        network is biased.

        Arguments:
          config (:class:`.NeuralNetConfig`): This network's configuration,
            which specifies the architecture, the activation functions, etc.

          span (float): A zero or positive number that specifies the size of
            the range in which to initialize weights and biases.

        Returns:
          (:class:`.NeuralNet`): The uniformly initialized neural network.

        Raises:
          ValueError: The `span` argument is negative.

        """
        if span < 0.0:
            raise ValueError('The span should be a zero or positive number.')

        ws, bs = config.shapes

        nw = sum(x * y for x, y in ws)
        nb = sum(x * y for x, y in bs) if config.biased else None

        w = numpy.random.uniform(-span, span, nw)
        b = numpy.random.uniform(-span, span, nb) if config.biased else None

        w = RaveledMatrixList(ws, w)
        b = RaveledMatrixList(bs, b) if config.biased else None

        return NeuralNet(config, w, b)

    def __call__(self, x):
        """
        Evaluates the network on the given input matrix.

        See Also:
          :func:`~app.neural.NeuralNet.activate`

        Arguments:
          x (numpy.array): Input samples of size `I`-by-`N`, where `I` is the
            number of input neurons and `N` is the number of samples.

        Returns:
          (numpy.array): The network output, an array of size `O`-by-`N`, where
            `O` is the number of output neurons, and `N` is the same number of
            samples as for the input matrix.

        """
        return self.activate(x, detailed=False)

    def activate(self, x, detailed=True):
        """
        Activates the network and (optionally) returns the inputs and outputs
        of each network layer, not only of the output layer.

        Arguments:
          x (numpy.array): Input samples of size `I`-by-`N`, where `I` is the
            number of input neurons, as specified by its architecture, and `N`
            is the number of samples.

          detailed (bool, optional): `True` if the inputs and outputs of all
            layers should be returned. If `False`, the output will be the same
            as for :func:`~app.neural.NeuralNet.__call__`.

        Returns:
          (tuple or numpy.array): If `detailed` is `True`, it returns a tuple
            of two lists, `(z, a)`, where `z` is the list of layer inputs, and
            `a` is the list of layer outputs. The first element in each list
            is the `x` argument, which represents both the input and the output
            of the first layer. If `detailed` is `False`, only the output of
            the last layer is returned.

        """

        w = self.weights
        b = self.biases

        z = [None] * self.layer_count  # Layer inputs
        a = [None] * self.layer_count  # Layer activations
        z[0] = a[0] = x

        for i in range(1, self.layer_count):
            z[i] = numpy.dot(w[i - 1], a[i - 1]) + (b[i - 1] if b else 0)
            a[i] = self.activations[i - 1](z[i])

        return (z, a) if detailed else a[-1]

    @property
    def layer_count(self):
        """ (int) Number of layer """
        return self.config.layer_count

    @property
    def activations(self):
        """
        (tuple) Activation functions used for each layer, except the input
        layer: the input layer does not have an activation function.

        """
        return self.config.activations

    @property
    def architecture(self):
        """ (tuple) Number of neurons for each layer """
        return self.config.architecture

    @property
    def config(self):
        """ (:class:`.NeuralNetConfig`) Configuration of this network """
        return self._config

    @property
    def weights(self):
        """ (:class:`app.utility.RaveledMatrixList`) Weight matrices """
        return self._weights

    @property
    def biases(self):
        """
        (:class:`app.utility.RaveledMatrixList`) Bias vectors or `None` if
        this network does not use bias units.

        """
        return self._biases

    @property
    def biased(self):
        """ (bool) `True` if this network uses bias units """
        assert self.config.biased == (self.biases is not None)
        return self.config.biased


class BackpropTrainer():
    """
    Trains a :class:`.NeuralNet` using the backpropagation algorithm.

    The trainer can be configured with an error function to measure the
    network's performance, and with the speed factor for the gradient descent
    algorithm. The speed factor is constant at the moment.

    Note:
      The trainer can only use gradient descent at the moment. In the future
      this might change to allow other optimisation algorithms to be used as
      well.

    """

    def __init__(self, error=mean_squared_error, alpha=0.2):
        """
        Arguments:
          error (function, optional): Error function used for measuring network
            performance. Defaults to :func:`~app.functions.mean_squared_error`.

          alpha (float, optional): The speed factor used for the gradient
            descent algorithm. Defaults to `0.2`.

        .. (This Sphinx comment prevents a Sphinx parser bug.)

        """
        self.error = error
        self.alpha = alpha

    def train(self, config, datasets, hook=None):
        """
        Creates and trains a network on the given data.

        The trainer uses backpropagation with gradient descent and iterates
        until the validation error starts increasing. There are currently no
        early stops implemented. An optional hook function can be specified to
        monitor the progress.

        Arguments:
          config (:class:`.NeuralNetConfig`): Configuration of the network to
            be trained.

          datasets (:class:`app.utility.Datasets`): The train, validation, and
            test sets to be used during training.

          hook (function, optional): Function with three arguments: the first
            is the current iteration number; the second is the current training
            set error; the third and last is the current validation set error.
            The hook will be called each iteration.

        Returns:
          (tuple): A tuple with two elements, the first of which is the
            trained network. The second is another tuple, of three elements:
            the training error, the validation error, and the test error.

        """
        network = NeuralNet.uniform(config, 1.0)
        errors = self._gradient_descent(network, datasets, hook)
        return network, errors

    def _error(self, network, samples):
        return self.error(network(samples.inputs), samples.targets)

    def _gradient_descent(self, network, datasets, hook=None):
        best_train_error = self._error(network, datasets.train)
        best_validation_error = self._error(network, datasets.validation)

        iteration = 0

        while True:
            dedw, dedb = backprop(network, datasets.train, self.error)
            network.weights.raveled[:] -= self.alpha * dedw.raveled
            if network.biased:
                network.biases.raveled[:] -= self.alpha * dedb.raveled

            train_error = self._error(network, datasets.train)
            validation_error = self._error(network, datasets.validation)

            if hook:
                hook(iteration, train_error, validation_error)

            if validation_error > best_validation_error:
                network.weights.raveled[:] += self.alpha * dedw.raveled
                if network.biased:
                    network.biases.raveled[:] += self.alpha * dedb.raveled
                break

            best_train_error = train_error
            best_validation_error = validation_error

            iteration += 1

        test_error = self._error(network, datasets.test)
        return best_train_error, best_validation_error, test_error


def backprop(network, samples, error):
    """
    Backpropagation algorithm for calculating error gradient with respect to
    weights and biases.

    Arguments:
      network (:class:`.NeuralNet`): The network to run backprop on.
      samples (:class:`app.utility.Samples`): Data to run backprop on.
      error (function): Function that measures network performance.

    Returns:
      (tuple): A tuple of two :class:`~app.utility.RaveledMatrixList` objects:
        the first is the list of weight gradients and the second is the list of
        bias gradients. If the network does not have biases, the second element
        in the tuple is `None`.

    """
    x = samples.inputs
    t = samples.targets

    z, a = network.activate(x)
    y = a[-1]  # Network output

    w = network.weights
    b = network.biases

    deda = [None] * len(a)  # Error gradient wrt. layer activations
    deda[-1] = error(y, t, gradient=True)

    ws = w.shapes
    bs = b.shapes if b else None

    dedw = RaveledMatrixList(ws)                 # Error gradient wrt. weights
    dedb = RaveledMatrixList(bs) if b else None  # Error gradient wrt. biases

    assert (len(dedw) == len(deda) - 1) and \
           (not b or len(dedw) == len(dedb))

    for i in reversed(range(len(dedw))):
        dadz = network.activations[i](z[i + 1], gradient=True)
        dedz = numpy.multiply(deda[i + 1], dadz)

        dedw[i] = numpy.dot(dedz, numpy.transpose(a[i]))
        deda[i] = numpy.dot(numpy.transpose(w[i]), dedz)

        if b:
            dedb[i] = numpy.sum(dedz, 1).reshape(dedz.shape[0], 1)

    return (dedw, dedb)
