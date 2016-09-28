import csv
import os.path
import numpy
from app.functions import *
from app.neural import *
from app.utility import *
from tests.integration.commons import *


def iris_dataset_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'dataset', 'iris.data')


def load_dataset(path):
    xs = list()
    ts = list()

    with open(path) as csvfile:
        fieldnames = ['x0', 'x1', 'x2', 'x3', 't']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)

        for row in reader:
            x = [float(row[key]) for key in fieldnames[:-1]]
            xs.append(x)

            t = row[fieldnames[-1]]
            ts.append(t)

    return [xs, ts]


def sparse(size, index):
    vector = [0.0] * size
    vector[index] = 1.0
    return vector


def normalize_data(data):
    t_set = set(data[1])
    t_encodings = {w: sparse(len(t_set), i) for i, w in enumerate(t_set)}
    data[1] = [t_encodings[t] for t in data[1]]
    return data


def make_network_config(x_features, t_features):
    architecture = (x_features, 10, t_features)
    activations = (sigmoid, sigmoid)
    biased = True
    return NeuralNetConfig(architecture, activations, biased)


def make_trainer():
    return BackpropTrainer(error=mean_logistic_error, alpha=0.2)


def error_rate(network, samples):
    x, t = samples.inputs, samples.targets
    y = numpy.round(network(x))
    return 100.0 * numpy.sum(numpy.any(y != t, 0)) / y.shape[1]


def main():
    path = iris_dataset_path()
    data = load_dataset(path)
    data = normalize_data(data)
    data = shuffle_data(data)

    x = numpy.transpose(data[0])
    t = numpy.transpose(data[1])
    datasets = Datasets.partition(x, t, (6, 6, 3))

    netconf = make_network_config(x.shape[0], t.shape[0])
    trainer = make_trainer()

    print_datasets_info('Iris Flower Dataset', datasets)
    print()
    print_network_config_info(netconf)
    print()
    print_trainer_info(trainer)

    print()
    print('Training...')

    hook = CountingHook()
    network, errors = trainer.train(netconf, datasets, hook=hook)

    print()
    print_info('Statistics', [
        ('Iterations',        hook.iterations),
        ('Train error',       errors[0]),
        ('Validation error:', errors[1]),
        ('Test error',        errors[2])
    ])

    print()
    print_info('Error rates', [
        ('Train',      show_percent(error_rate(network, datasets.train))),
        ('Validation', show_percent(error_rate(network, datasets.validation))),
        ('Test',       show_percent(error_rate(network, datasets.test)))
    ])

    return 0


if __name__ == '__main__':
    exit(main())
