import numpy


class CountingHook():
    def __init__(self):
        self.iterations = 0

    def __call__(self, it, train_error, validation_error):
        self.iterations += 1


def shuffle_data(data):
    zipped = list(zip(*data))
    numpy.random.shuffle(zipped)
    return list(zip(*zipped))


def show_names_tuple(tup):
    return show_tuple(x.__name__ for x in tup)


def show_tuple(tup):
    return '(' + ', '.join(str(x) for x in tup) + ')'


def show_boolean(b):
    return 'yes' if b else 'no'


def show_percent(p):
    return '%.2f%%' % p


def print_info(title, info):
    print(title + ':')
    alignment = max(len(key) for key, _ in info)

    for key, value in info:
        print('  ' + (key + ':').ljust(alignment + 1), value)


def print_datasets_info(name, datasets):
    print_info('Datasets', [
        ('Name',                  name),
        ('Train samples',         len(datasets.train)),
        ('Validation samples',    len(datasets.validation)),
        ('Test samples',          len(datasets.test)),
        ('Independent variables', datasets.train.inputs.shape[0]),
        ('Dependent variables',   datasets.train.targets.shape[0])
    ])


def print_network_config_info(netconf):
    print_info('Neural network', [
        ('Architecture', show_tuple(netconf.architecture)),
        ('Activations',  show_names_tuple(netconf.activations)),
        ('Biased',       show_boolean(netconf.biased))
    ])


def print_trainer_info(trainer):
    print_info('Trainer', [
        ('Error function', trainer.error.__name__),
        ('Alpha (GD)',     trainer.alpha)
    ])
