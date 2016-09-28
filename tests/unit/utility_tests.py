import numpy
from nose.tools import *
from app.utility import *
from tests.unit.commons import *


@raises(ValueError)
def test_raveled_wrong_shapes():
    array = uniform_float_matrix()
    r, c = array.shape
    _ = RaveledMatrixList(shapes=[(r + 1, c)], raveled=array.ravel())


def test_raveled_correct_shapes():
    array = uniform_float_matrix()
    sections = numpy.random.randint(1, array.shape[0])
    splits = numpy.array_split(array, sections)
    shapes = [split.shape for split in splits]

    try:
        _ = RaveledMatrixList(shapes, array.ravel())
        assert_true(True)
    except ValueError:
        assert_true(False)


def test_raveled_length():
    count = numpy.random.randint(1, 10)
    arrays = [uniform_float_matrix() for i in range(count)]
    shapes = [a.shape for a in arrays]
    raveled = numpy.concatenate([a.flat for a in arrays])
    raveled_matrix_list = RaveledMatrixList(shapes, raveled)
    assert_equal(count, len(raveled_matrix_list))


def test_raveled_indexing():
    count = numpy.random.randint(1, 10)
    arrays = [uniform_float_matrix() for i in range(count)]
    shapes = [a.shape for a in arrays]
    raveled = numpy.concatenate([a.flat for a in arrays])
    raveled_matrix_list = RaveledMatrixList(shapes, raveled)

    for i in range(count):
        assert_true(numpy.array_equal(arrays[i], raveled_matrix_list[i]))


def test_raveled_change_views():
    count = numpy.random.randint(1, 10)
    arrays = [uniform_float_matrix() for i in range(count)]
    shapes = [a.shape for a in arrays]
    raveled = numpy.concatenate([a.flat for a in arrays])
    raveled_matrix_list = RaveledMatrixList(shapes, raveled)

    for i in range(len(raveled_matrix_list)):
        raveled_matrix_list[i] *= 0.0

    assert_true(numpy.all(raveled_matrix_list.raveled == 0.0))


def test_raveled_same_vector():
    count = numpy.random.randint(1, 10)
    arrays = [uniform_float_matrix() for i in range(count)]
    shapes = [a.shape for a in arrays]
    raveled = numpy.concatenate([a.flat for a in arrays])
    raveled_matrix_list = RaveledMatrixList(shapes, raveled)
    assert_true(numpy.all(raveled == raveled_matrix_list.raveled))


def test_raveled_default_vector_zero():
    count = numpy.random.randint(1, 10)
    shapes = [numpy.random.randint(2, 10, 2) for i in range(count)]
    raveled_matrix_list = RaveledMatrixList(shapes)
    assert_true(numpy.all(raveled_matrix_list.raveled == 0.0))


@raises(ValueError)
def test_samples_wrong_sizes():
    # Use different sample counts to raise error
    input_sample_count = numpy.random.randint(1, 50)
    target_sample_count = input_sample_count + 1
    random_samples(xsamples=input_sample_count, tsamples=target_sample_count)


def test_samples_correct_sizes():
    try:
        random_samples()
        assert_true(True)
    except ValueError:
        assert_true(False)


def test_samples_length():
    sample_count = numpy.random.randint(1, 50)
    samples = random_samples(nsamples=sample_count)
    assert_equal(len(samples), sample_count)


@raises(ValueError)
def test_samples_immutable_inputs():
    samples = random_samples()
    samples.inputs[0] = 1


@raises(ValueError)
def test_samples_immutable_targets():
    samples = random_samples()
    samples.targets[0] = 1


@raises(ValueError)
def test_dataset_negative_ratios():
    ratios = [1, 1, 1]
    index = numpy.random.choice(range(len(ratios)))
    ratios[index] = -1
    _ = Datasets.partition(None, None, tuple(ratios))


@raises(ValueError)
def test_dataset_zero_ratios():
    ratios = [1, 1, 1]
    index = numpy.random.choice(range(len(ratios)))
    ratios[index] = 0
    _ = Datasets.partition(None, None, tuple(ratios))


def test_datasets_all_elements():
    x = uniform_float_matrix()
    t = uniform_float_matrix(shape=x.shape)

    datasets = Datasets.partition(x, t, tuple(numpy.random.randint(1, 10, 3)))
    samples = [datasets.train, datasets.validation, datasets.test]

    datasets_sample_count = sum(len(sample) for sample in samples)
    initial_sample_count = x.shape[1]

    assert_equal(initial_sample_count, datasets_sample_count)
