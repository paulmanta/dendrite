# Dendrite

Dendrite is a Python mini-library that implements neural networks. It gives the
user flexibility to build a network with any architecture (number and size of
neuron layers) and the flexibility to specify a separate activation function
for each layer.

Training uses the backpropagation and gradient descent algorithms and also
correctly evaluates the performance of the network by using separate training,
validation, and test sets. The performance of the network is measured with
the error function the user provides.

A small selection of popular activation and error functions are made available,
and users are able to define their own functions and plug them into the network
configuration or the trainer without any other changes to the code.

## Setup

Make sure you have `python3` and `pip3` available and then install all the
necessary third-party dependencies by running:

    pip3 install -r requirements.txt

## `make` rules

  * `make doc` — Run [Sphinx](http://www.sphinx-doc.org/) to generate the API
    documentation in HTML format. See the *Troubleshooting* section if you are
    using macOS and are unable to generate the documentation.

  * `make pep8` — Check the coding style of the sources in the `app` and `tests`
    directories against the [PEP8](https://www.python.org/dev/peps/pep-0008/)
    guidelines.

  * `make clean` — Remove all temporary and generated files and directories,
    such as `.pyc` files, `__pycache__` directories, and generated API
    documentation.

  * `make unit` — Run all unit tests in the `tests/unit` directory.

  * `make test-iris` and `make test-autovit` — Run the two available integration
    tests. See the *Integration tests* section for details.

## Integration tests

### Iris test

This is a classification test running on the [Iris flower data set]
(https://en.wikipedia.org/wiki/Iris_flower_data_set). The data is divided
equally into training, validation, and test sets. There are four independent
variables (input neurons) for each of the attributes in the data set and
three dependent variables (output neurons) for each of the possible classes
of flowers.

The test will train a three layered neural network with 4, 10, and 3
neurons respectively, plus bias neurons. Layers are activated using the
[sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function. The error
is measured using `functions.mean_logistic_error`.

The test will report the classification results at the end. The performance
of the network will vary with each run of the test due to the weights being
randomly initialized, but typical classification errors can be seen below.
The percentages represent the proportion of misclassified entries for each of
the training, validation and test data sets.

    Train:      1.05%
    Validation: 3.33%
    Test:       6.67%

### Autovit test

Regression test running on public data collected from www.autovit.ro about
used Volkswagen Golf vehicles available for resale. The test attempts to build
a model for predicting the price of an old Volkswagen Golf based on attributes
such as year of manufacturing, kilometrage, number of accidents, and whether
the car is damaged.

The network has three layers with 5 input neurons, 12 hidden neurons, and
only one output neuron (the price of the car). The hidden layer is activated
using `functions.sigmoid` and the output layer is simply `functions.identity`.
The identity function is used for the output layer to allow the output
value (the price) to be unbounded. The error is measured using
`functions.mean_absolute_error`.

## Troubleshooting (`make doc` on macOS)

Running the `make doc` command on macOS may cause Sphinx to throw an exception:
`ValueError: unknown locale: UTF-8`. The problem can be fixed by adding the
following two lines in your `~/.bash_profile` file:

    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8

After saving the file you run the following in your terminal to apply the
changes:

    source ~/.bash_profile
