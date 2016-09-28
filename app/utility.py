import numpy


class RaveledMatrixList():
    """
    Stores a list of matrices as one long vector.

    The matrices can still be accessed individually (in their original shape)
    through views. Any changes done to one of the matrices will be reflected in
    the raveled vector and vice versa.

    """

    def __init__(self, shapes, raveled=None):
        """
        Arguments:
          shapes (list): The shapes of the matrices as a list of 2-tuples. Each
            2-tuple is of the form `(r, c)`, with `r` being the number of rows
            and `c` the number of columns.

          raveled (numpy.array, optional): One-dimensional vector of all the
            initial matrix values; the class takes ownership of the array. If
            not specified, the matrices will be zero-initialised.

        Raises:
          ValueError: The length of the `raveled` vector does not match the
            number of elements expected by the `shapes` list.

        """

        if raveled is None:
            raveled = numpy.zeros(sum(x * y for x, y in shapes))
        elif sum(x * y for x, y in shapes) != raveled.size:
            raise ValueError('The raveled vector and shapes do not match.')

        self._raveled = raveled
        self._shapes = shapes
        self._views = [None] * len(shapes)

        for i in range(len(self._views)):
            start = sum(x * y for x, y in shapes[0:i])
            end = start + shapes[i][0] * shapes[i][1]
            self._views[i] = raveled[start:end].reshape(shapes[i])

    @property
    def raveled(self):
        return self._raveled

    @property
    def shapes(self):
        return self._shapes

    def __getitem__(self, key):
        return self._views[key]

    def __setitem__(self, key, value):
        self._views[key][:] = value

    def __len__(self):
        return len(self._views)


class Samples():
    """
    Stores input and target samples to be used during training.

    The class offers a convenient way for related input and target arrays to
    be grouped together, while also ensuring that the number of samples in each
    is the same (i.e. both arrays have the same number of columns). The two
    arrays will be made immutable by setting `inputs.flags.writeable` and
    `targets.flags.writeable` to `False`.

    """

    def __init__(self, inputs, targets):
        """
        Arguments:
          inputs (numpy.array): Array of input samples of size `I`-by-`N`,
            where `I` is the number of input features and `N` is the number
            of samples.

          targets (numpy.array): Array of target samples of size `T`-by-`N`,
            where `T` is the number of output features and `N` is the same
            number of samples as for `inputs`.

        Raises:
          ValueError: The `inputs` and `targets` arrays have different number
            of samples (different number of columns).

        """

        if inputs.shape[1] != targets.shape[1]:
            raise ValueError('Matrices have different number of samples.')

        inputs.flags.writeable = False
        targets.flags.writeable = False

        self._inputs = inputs
        self._targets = targets

    def __len__(self):
        """
        Returns:
          (int): The number of samples

        """
        assert self.inputs.shape[1] == self.targets.shape[1]
        return self.inputs.shape[1]

    @property
    def inputs(self):
        """ (numpy.array) The input samples array passed at construction """
        return self._inputs

    @property
    def targets(self):
        """ (numpy.array) The target samples array passed at construction """
        return self._targets


class Datasets():
    """
    Holds train, validation and test sets to be used during training. These
    sets, as well as the values they contain, are immutable.

    """

    def __init__(self, train, validation, test):
        """
        Arguments:
          train (:class:`.Samples`): Data used during training.
          validation (:class:`.Samples`): Data used for cross-validation.
          test (:class:`.Samples`): Data used for the final test phase.

        Raises:
          ValueError: Samples sets cannot be `None` or empty.

        """
        if None in (train, validation, test):
            raise ValueError('Sample sets cannot be `None`.')

        if 0 in (len(train), len(validation), len(test)):
            raise ValueError('Sample sets cannot be empty.')

        self._train = train
        self._validation = validation
        self._test = test

    @staticmethod
    def partition(inputs, targets, ratios):
        """
        Partitions the given data into separate data sets, according to the
        indicated ratios. The arrays will be made immutable by setting their
        `flags.writeable` properties to `False`.

        Arguments:
          inputs (numpy.array): Array of input samples of size `I`-by-`N`,
            where `I` is the number of input features and `N` is the number
            of samples.

          targets (numpy.array): Array of target samples of size `T`-by-`N`,
            where `T` is the number of output features and `N` is the same
            number of samples as for `inputs`.

          ratios (float or int tuple): Tuple of three elements, `(A, B, C)`,
            where `A`, `B`, and `C` are the relative ratios of the train,
            validation, and test samples, respectively. None can be 0.

        Returns:
          (:class:`.Datasets`): The `Datasets` instance, with data partitioned
            as indicated.

        Raises:
          ValueError: The tuple does not have three elements, some of the ratio
            values are 0 or negative, or the `inputs` and `targets` arrays have
            different number of samples.

        """

        if len(ratios) != 3:
            raise ValueError('The ratios argument must be a 3-tuple.')

        if any(r <= 0.0 for r in ratios):
            raise ValueError('Ratios must be positive.')

        if inputs.shape[1] != targets.shape[1]:
            raise ValueError('Different number of input and train samples.')

        inputs.flags.writeable = False
        targets.flags.writeable = False

        sizes = numpy.array(ratios)
        sizes = sizes * inputs.shape[1] / numpy.sum(sizes)
        sizes = sizes.astype(int)
        sizes[0] += inputs.shape[1] - numpy.sum(sizes)

        assert all(size > 0.0 for size in sizes)
        assert sum(sizes) == inputs.shape[1]

        x = inputs
        t = targets

        train, x, t = Datasets._take(x, t, sizes[0])
        validation, x, t = Datasets._take(x, t, sizes[1])
        test, _, _ = Datasets._take(x, t, sizes[2])

        return Datasets(train, validation, test)

    @property
    def train(self):
        """ (:class:`.Samples`) Train set (immutable and non-empty) """
        return self._train

    @property
    def validation(self):
        """ (:class:`.Samples`) Validation set (immutable and non-empty) """
        return self._validation

    @property
    def test(self):
        """ (:class:`.Samples`) Test set (immutable and non-empty) """
        return self._test

    @staticmethod
    def _take(x, t, size):
        # This if-statement is not strictly necessary since numpy.hsplit
        # already handles this corner case, but it also throws a FutureWarning
        # that is not relevant to this code and only distracts.
        if size == x.shape[1]:
            assert x.shape[1] == t.shape[1]
            return Samples(x, t), None, None

        x, remaining_x = numpy.hsplit(x, [size])
        t, remaining_t = numpy.hsplit(t, [size])
        return Samples(x, t), remaining_x, remaining_t
