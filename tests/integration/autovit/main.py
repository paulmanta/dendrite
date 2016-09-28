import csv
import os.path
from app.functions import *
from app.neural import *
from app.utility import *
from tests.integration.commons import *


def autovit_dataset_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'volkswagen-golf.csv')


def load_dataset(path):
    xs = list()
    ts = list()

    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x = [None] * (len(row) - 1)
            x[0] = 2015.0 - float(row['year'])
            x[1] = float(row['kilometrage'] == 'True')
            x[2] = float(row['first_owner'] == 'True')
            x[3] = float(row['damaged'] == 'True')
            x[4] = float(row['accidents'] == 'True')

            xs.append(x)
            ts.append([float(row['price'])])

    return [xs, ts]


def normalize_data(data):
    x, t = data
    x = numpy.array(x)
    t = numpy.array(t)

    xbaseline = numpy.max(x, 0).reshape(1, x.shape[1])
    xbaseline[xbaseline == 0.0] = 1.0
    x /= xbaseline

    tbaseline = numpy.max(t, 0).reshape(1, t.shape[1])
    tbaseline[tbaseline == 0.0] = 1.0
    t /= tbaseline

    return [x, t]


def make_network_config(x_features, t_features):
    architecture = (x_features, int(x_features * 2.5), t_features)
    activations = (sigmoid, identity)
    biased = True
    return NeuralNetConfig(architecture, activations, biased)


def make_trainer():
    return BackpropTrainer(error=mean_absolute_error, alpha=0.005)


def main():
    path = autovit_dataset_path()
    data = load_dataset(path)
    data = shuffle_data(data)
    data = normalize_data(data)

    x = numpy.transpose(data[0])
    t = numpy.transpose(data[1])
    datasets = Datasets.partition(x, t, (6, 2, 2))

    netconf = make_network_config(x.shape[0], t.shape[0])
    trainer = make_trainer()

    print_datasets_info('Autovit Volkswagen Golf', datasets)
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

    return 0


if __name__ == '__main__':
    exit(main())
