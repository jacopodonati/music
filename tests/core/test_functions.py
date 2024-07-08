""" Tests for music.core.functions """
import numpy as np
from music.core import functions


def test_normalize_mono():
    """ Test normalization of unidimensional sonic vector. """

    # Test case: sonic vector filled with 0s.
    sonic_vector_zero = np.zeros(100)
    assert np.array_equal(
        functions.normalize_mono(sonic_vector_zero),
        sonic_vector_zero
        )

    # Test case: sonic vector filled with positive values
    # and bias cancellation.
    sonic_vector_positive = np.array([1, 2, 3, 4, 5])
    normalized_vector_positive = functions.normalize_mono(
        sonic_vector_positive,
        remove_bias=True
        )
    assert np.all(normalized_vector_positive >= -1) and \
        np.all(normalized_vector_positive <= 1)
    assert np.isclose(normalized_vector_positive.mean(), 0)

    # Test case: sonic vector filled with negative values
    # and bias cancellation.
    sonic_vector_negative = np.array([-5, -4, -3, -2, -1])
    normalized_vector_negative = functions.normalize_mono(
        sonic_vector_negative,
        remove_bias=True
        )
    assert np.all(normalized_vector_negative >= -1) and \
        np.all(normalized_vector_negative <= 1)
    assert np.isclose(normalized_vector_negative.mean(), 0)

    # Test case: sonic vector filled with positive values
    # and no bias cancellation.
    sonic_vector_positive_no_bias = np.array([1, 2, 3, 4, 5])
    normalized_vector_positive_no_bias = functions.normalize_mono(
        sonic_vector_positive_no_bias,
        remove_bias=False
        )
    assert np.all(normalized_vector_positive_no_bias >= -1) and \
        np.all(normalized_vector_positive_no_bias <= 1)

    # Test case: sonic vector filled with negative values
    # and no bias cancellation.
    sonic_vector_negative_no_bias = np.array([-5, -4, -3, -2, -1])
    normalized_vector_negative_no_bias = functions.normalize_mono(
        sonic_vector_negative_no_bias,
        remove_bias=False
        )
    assert np.all(normalized_vector_negative_no_bias >= -1) and \
        np.all(normalized_vector_negative_no_bias <= 1)

    # Test case: sonic vector filled with random values
    sonic_vector_random = np.random.randn(100)
    normalized_vector_random = functions.normalize_mono(sonic_vector_random)
    assert np.all(normalized_vector_random >= -1) and \
        np.all(normalized_vector_random <= 1)


def test_normalize_stereo():
    """ Test normalization of bidimensional sonic vector. """

    # Test case: stereo sonic vector filled with 0s.
    sonic_vector_zero = np.zeros((2, 100))
    assert np.array_equal(
        functions.normalize_stereo(sonic_vector_zero),
        sonic_vector_zero
        )

    # Test case: stereo sonic vector filled with positive values
    # and bias cancellation.
    sonic_vector_positive = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    normalized_vector_positive = functions.normalize_stereo(
        sonic_vector_positive,
        remove_bias=True
        )
    assert np.all(normalized_vector_positive >= -1) and \
        np.all(normalized_vector_positive <= 1)
    assert np.isclose(normalized_vector_positive.mean(axis=1), 0).all()

    # Test case: stereo sonic vector filled with negative values
    # and bias cancellation.
    sonic_vector_negative = np.array([
        [-5, -4, -3, -2, -1],
        [-1, -2, -3, -4, -5]
        ])
    normalized_vector_negative = functions.normalize_stereo(
        sonic_vector_negative,
        remove_bias=True
        )
    assert np.all(normalized_vector_negative >= -1) and \
        np.all(normalized_vector_negative <= 1)
    assert np.isclose(normalized_vector_negative.mean(axis=1), 0).all()

    # Test case: stereo sonic vector filled with positive values
    # and no bias cancellation.
    sonic_vector_positive_no_bias = np.array([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1]
        ])
    normalized_vector_positive_no_bias = functions.normalize_stereo(
        sonic_vector_positive_no_bias,
        remove_bias=False
        )
    assert np.all(normalized_vector_positive_no_bias >= -1) and \
        np.all(normalized_vector_positive_no_bias <= 1)

    # Test case: stereo sonic vector filled with negative values
    # and no bias cancellation.
    sonic_vector_negative_no_bias = np.array([
        [-5, -4, -3, -2, -1],
        [-1, -2, -3, -4, -5]
        ])
    normalized_vector_negative_no_bias = functions.normalize_stereo(
        sonic_vector_negative_no_bias,
        remove_bias=False
        )
    assert np.all(normalized_vector_negative_no_bias >= -1) and \
        np.all(normalized_vector_negative_no_bias <= 1)

    # Test case: stereo sonic vector filled with random values
    sonic_vector_random = np.random.randn(2, 100)
    normalized_vector_random = functions.normalize_stereo(sonic_vector_random)
    assert np.all(normalized_vector_random >= -1) and \
        np.all(normalized_vector_random <= 1)
