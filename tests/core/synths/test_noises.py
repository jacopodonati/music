""" Tests for music.core.synths.noises """
import numpy as np
import pytest
from scipy.signal import welch
from scipy.stats import kstest, shapiro, anderson
from scipy.stats import chi2_contingency, linregress
from music.core.synths import noises


def statistical_test_for_white_noise(data):
    """
    Perform a statistical test to check if the given data
    represents white noise.

    Parameters
    ----------
    data : array_like
        The data to be tested.

    Returns
    -------
    p_value : float
        The p-value of the statistical test.
    """
    _, bins = np.histogram(data, bins='auto')
    observed, _ = np.histogram(data, bins=bins)
    expected = np.ones_like(observed) * np.mean(observed)
    _, p_value, _, _ = chi2_contingency([observed, expected])
    return p_value


def calculate_slope_of_log_log_plot(data):
    """
    Calculate the slope of the log-log plot of the given data.

    Parameters
    ----------
    data : array_like
        The data to be analyzed.

    Returns
    -------
    slope : float
        The slope of the log-log plot.
    """
    freqs, psd = welch(data, nperseg=len(data))

    nonzero_indices = psd > 0.001 * np.max(psd)  # Adjust threshold as needed
    freqs = freqs[nonzero_indices]
    psd = psd[nonzero_indices]

    log_freqs = np.log10(freqs)
    log_psd = np.log10(psd)
    slope, _, _, _, _ = linregress(log_freqs, log_psd)
    return slope


def test_silence():
    """ Test silence(). """

    # Test case: Generating silence with default duration and sample rate
    silence_default = noises.silence()
    assert len(silence_default) == 44100
    assert np.all(silence_default == 0)

    # Test case: Generating silence with custom duration and sample rate
    duration = 2
    sample_rate = 22050
    silence_custom = noises.silence(duration=duration, sample_rate=sample_rate)
    assert len(silence_custom) == duration * sample_rate
    assert np.all(silence_custom == 0)


@pytest.mark.parametrize("noise_type", ["brown", "pink", "white", "blue",
                                        "violet", "black"])
def test_noise_distribution(noise_type):
    noise_vector = noises.noise(noise_type=noise_type)

    dist = 'norm'
    # exponent = 0
    if (noise_type != 'white'):
        dist = 'expon'
        # if (noise_type == 'black'):
        #     exponent = -3
        # elif (noise_type == 'brown'):
        #     exponent = -2
        # elif (noise_type == 'pink'):
        #     exponent = -1
        # elif (noise_type == 'blue'):
        #     exponent = 1
        # elif (noise_type == 'violet'):
        #     exponent = 2

    result = anderson(noise_vector, dist=dist)
    p_value = np.exp(-result.statistic)
    assert p_value > 0.05


@pytest.mark.parametrize("noise_type", ["brown", "pink", "white", "blue",
                                        "violet", "black"])
def test_noise(noise_type):
    """ Test synth of different kinds of noises. """

    # Test case: noise distribution
    noise_vector = noises.noise(noise_type, duration=2, sample_rate=44100)
    print(noise_vector[0:4])

    if noise_type == "white":
        # White noise should be normally distributed
        stat, p = kstest(noise_vector, 'norm')
        assert p > 0.05  # Fail if p-value is less than 0.05
    elif noise_type == "pink":
        # Pink noise should be exponentially distributed
        stat, p = kstest(noise_vector, 'expon')
        assert p > 0.05  # Fail if p-value is less than 0.05
    elif noise_type == "brown":
        # Brown noise should be normally distributed with a mean of 0 and a
        # variance of 1
        stat, p = kstest(noise_vector, 'norm', args=(0, 1))
        assert p > 0.05  # Fail if p-value is less than 0.05
    elif noise_type == "blue":
        # Blue noise should be uniformly distributed
        stat, p = kstest(noise_vector, 'uniform')
        assert p > 0.05  # Fail if p-value is less than 0.05
    elif noise_type == "violet":
        # Violet noise should be normally distributed with a mean of 0 and a
        # variance of 1
        stat, p = kstest(noise_vector, 'norm', args=(0, 1))
        assert p > 0.05  # Fail if p-value is less than 0.05
    elif noise_type == "black":
        # Black noise should be exponentially distributed
        stat, p = kstest(noise_vector, 'expon')
        assert p > 0.05  # Fail if p-value is less than 0.05


@pytest.mark.parametrize("noise_type", ["brown", "pink", "white", "blue",
                                        "violet", "black"])
def test_noise_power_spectrum(noise_type):
    # Test case: power spectrum
    noise_vector = noises.noise(noise_type, duration=2, sample_rate=44100)

    # Calculate power spectrum using FFT
    freq = np.fft.fft(len(noise_vector))
    power_spectrum = np.abs(np.fft.fft(noise_vector))**2
    print(np.mean(power_spectrum), power_spectrum[100])
    # Test power spectrum shape
    if noise_type == "white":
        # White noise should have a flat power spectrum.  Fail if power
        # spectrum is not flat.
        assert np.allclose(power_spectrum, np.mean(power_spectrum), atol=1e-2)
    elif noise_type == "pink":
        # Pink noise should have a power spectrum that decreases by
        # 3 dB/octave.  Fail if slope is not close to -3
        slope, _ = np.polyfit(np.log10(freq), np.log10(power_spectrum), 1)
        assert np.allclose(slope, -3, atol=1e-2)
    elif noise_type == "brown":
        # Brown noise should have a power spectrum that decreases by
        # 6 dB/octave.  Fail if slope is not close to -6
        slope, _ = np.polyfit(np.log10(freq), np.log10(power_spectrum), 1)
        assert np.allclose(slope, -6, atol=1e-2)
    elif noise_type == "blue":
        # Blue noise should have a power spectrum that increases by
        # 3 dB/octave.  Fail if slope is not close to 3.
        slope, _ = np.polyfit(np.log10(freq), np.log10(power_spectrum), 1)
        assert np.allclose(slope, 3, atol=1e-2)
    elif noise_type == "violet":
        # Violet noise should have a power spectrum that increases by
        # 6 dB/octave.  Fail if slope is not close to 6.
        slope, _ = np.polyfit(np.log10(freq), np.log10(power_spectrum), 1)
        assert np.allclose(slope, 6, atol=1e-2)
    elif noise_type == "black":
        # Black noise should have a power spectrum that decreases by
        # 12 dB/octave.  Fail if slope is not close to -12.
        slope, _ = np.polyfit(np.log10(freq), np.log10(power_spectrum), 1)
        assert np.allclose(slope, -12, atol=1e-2)

    # # Test case: Generating white noise with default parameters
    # white_noise_default = noises.noise(noise_type="white")
    # assert len(white_noise_default) == 88200
    # assert np.all(np.abs(white_noise_default) <= 1)
    # _, p_value_white = kstest(white_noise_default, 'uniform')
    # assert p_value_white > 0.05

    # # Test case: Generating pink noise with custom parameters
    # pink_noise_custom = noises.noise(
    #     noise_type="pink",
    #     duration=3,
    #     min_freq=20,
    #     max_freq=10000,
    #     sample_rate=22050
    #     )
    # assert len(pink_noise_custom) == 66150
    # assert np.all(np.abs(pink_noise_custom) <= 1)
    # _, p_value_pink = kstest(pink_noise_custom, 'norm', args=(0, 1))
    # assert p_value_pink > 0.05

    # # Test case: Generating brown noise with custom parameters
    # brown_noise_custom = noises.noise(
    #     noise_type="brown",
    #     duration=5,
    #     min_freq=10,
    #     max_freq=5000,
    #     sample_rate=44100
    #     )
    # assert len(brown_noise_custom) == 220500
    # assert np.all(np.abs(brown_noise_custom) <= 1)
    # _, p_value_brown = kstest(brown_noise_custom, 'uniform')
    # assert p_value_brown > 0.05

    # # Test case: Generating blue noise with custom parameters
    # blue_noise_custom = noises.noise(
    #     noise_type="blue",
    #     duration=2,
    #     min_freq=50,
    #     max_freq=15000,
    #     sample_rate=44100
    #     )
    # assert len(blue_noise_custom) == 88200
    # assert np.all(np.abs(blue_noise_custom) <= 1)
    # _, p_value_blue = kstest(blue_noise_custom, 'uniform')
    # assert p_value_blue > 0.05

    # # Test case: Generating violet noise with custom parameters
    # violet_noise_custom = noises.noise(
    #     noise_type="violet",
    #     duration=4,
    #     min_freq=30,
    #     max_freq=20000,
    #     sample_rate=22050)
    # assert len(violet_noise_custom) == 88200
    # assert np.all(np.abs(violet_noise_custom) <= 1)
    # _, p_value_violet = kstest(violet_noise_custom, 'norm', args=(0, 1))
    # assert p_value_violet > 0.05

    # # Test case: Generating black noise with custom parameters
    # black_noise_custom = noises.noise(
    #     noise_type="black",
    #     duration=6,
    #     min_freq=5,
    #     max_freq=10000,
    #     sample_rate=44100
    #     )
    # assert len(black_noise_custom) == 264600
    # assert np.all(np.abs(black_noise_custom) <= 1)
    # _, p_value_black = kstest(black_noise_custom, 'uniform')
    # assert p_value_black > 0.05

    # # Test case: Generating noise with a numeric value for noise_type
    # numeric_noise_custom = noises.noise(
    #     noise_type=4.5,
    #     duration=2,
    #     min_freq=100,
    #     max_freq=20000,
    #     sample_rate=44100
    #     )
    # assert len(numeric_noise_custom) == 88200
    # assert np.all(np.abs(numeric_noise_custom) <= 1)
    # _, p_value_numeric = kstest(numeric_noise_custom, 'uniform')
    # assert p_value_numeric > 0.05


def test_noise_features():
    """ Test features of every noise. """

    # Test case: Generating white noise and checking PSD constancy
    white_noise_default = noises.noise(noise_type="white")
    p_value_white = statistical_test_for_white_noise(white_noise_default)
    assert p_value_white > 0.05

    # Test case: Generating pink noise and checking power decrease rate
    pink_noise_custom = noises.noise(
        noise_type="pink",
        duration=3,
        min_freq=20,
        max_freq=10000,
        sample_rate=22050
        )
    slope_pink = calculate_slope_of_log_log_plot(pink_noise_custom)
    assert slope_pink < -0.25 and slope_pink > -0.35

    # Test case: Generating brown noise and checking power decrease rate
    brown_noise_custom = noises.noise(
        noise_type="brown",
        duration=5,
        min_freq=10,
        max_freq=5000,
        sample_rate=44100
        )
    slope_brown = calculate_slope_of_log_log_plot(brown_noise_custom)
    assert slope_brown < -0.55 and slope_brown > -0.65

    # Test case: Generating blue noise and checking power increase rate
    blue_noise_custom = noises.noise(
        noise_type="blue",
        duration=2,
        min_freq=50,
        max_freq=15000,
        sample_rate=44100
        )
    slope_blue = calculate_slope_of_log_log_plot(blue_noise_custom)
    assert slope_blue > 0.25 and slope_blue < 0.35

    # Test case: Generating violet noise and checking power increase rate
    violet_noise_custom = noises.noise(
        noise_type="violet",
        duration=4,
        min_freq=30,
        max_freq=20000,
        sample_rate=22050
        )
    slope_violet = calculate_slope_of_log_log_plot(violet_noise_custom)
    assert slope_violet > 0.55 and slope_violet < 0.65

    # Test case: Generating black noise and checking power decrease rate
    black_noise_custom = noises.noise(
        noise_type="black",
        duration=6,
        min_freq=5,
        max_freq=10000,
        sample_rate=44100
        )
    slope_black = calculate_slope_of_log_log_plot(black_noise_custom)
    assert slope_black < -1.05


def test_gaussian_noise():
    """ Test synth of gaussian noise. """

    # Test case: Generating Gaussian noise with default parameters
    noise_default = noises.gaussian_noise()
    assert len(noise_default) == 88200
    _, p_value_default = shapiro(noise_default)
    assert p_value_default > 0.05

    # Test case: Generating Gaussian noise with custom mean
    # and standard deviation
    mean = 2
    std = 0.8
    noise_custom = noises.gaussian_noise(mean=mean, std=std)
    assert np.mean(noise_custom) == mean
    assert np.std(noise_custom) == std
    _, p_value_custom = shapiro(noise_custom)
    assert p_value_custom > 0.05

    # Test case: Generating Gaussian noise with custom duration and sample rate
    duration = 5
    sample_rate = 22050
    noise_custom_params = noises.gaussian_noise(
        duration=duration,
        sample_rate=sample_rate
        )
    assert len(noise_custom_params) == duration * sample_rate
    assert np.all(np.isclose(
        np.fft.fftfreq(len(noise_custom_params)),
        np.arange(len(noise_custom_params)) /
        (len(noise_custom_params) /
         sample_rate)))
    _, p_value_params = shapiro(noise_custom_params)
    assert p_value_params > 0.05
