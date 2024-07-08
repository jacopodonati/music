""" Tests for music.core.synths.envelopes """
import numpy as np
import pytest
from music.utils import WAVEFORM_SINE
from music.core.synths import envelopes


def test_am():
    """ Tests for the creation of an AM envelope. """

    # Test case: valid parameters
    # envelope = envelopes.am(duration=2, fm=50, max_amplitude=0.4,
    #                         waveform_table=WAVEFORM_SINE, sample_rate=44100)
    # assert isinstance(envelope, np.ndarray)
    # assert len(envelope) > 0
    # assert np.all(envelope >= 0) and np.all(envelope <= 1)

    # Test case: max amplitude value
    envelope_max_amplitude = envelopes.am(duration=2, fm=50, max_amplitude=1,
                                          waveform_table=WAVEFORM_SINE,
                                          sample_rate=44100)
    assert np.all(envelope_max_amplitude == 1)

    # Test case: invalid parameters
    with pytest.raises(ValueError):
        envelopes.am(duration=-1, fm=50, max_amplitude=0.4,
                     waveform_table=WAVEFORM_SINE, sample_rate=44100)


def test_tremolo():
    """ Tests for the creation of a tremolo envelope. """

    # Test case: valid parameters
    envelope = envelopes.tremolo(duration=2, tremolo_freq=2, max_db_dev=10,
                                 alpha=1, waveform_table=WAVEFORM_SINE,
                                 sample_rate=44100)
    assert isinstance(envelope, np.ndarray)
    assert len(envelope) > 0
    assert np.all(envelope >= 0)

    # Test case: max dB deviation
    envelope_max_db_dev = envelopes.tremolo(duration=2, tremolo_freq=2,
                                            max_db_dev=0, alpha=1,
                                            waveform_table=WAVEFORM_SINE,
                                            sample_rate=44100)
    assert np.all(envelope_max_db_dev == 1)

    # Test case: invalid parameters
    with pytest.raises(ValueError):
        envelopes.tremolo(duration=-1, tremolo_freq=2, max_db_dev=10, alpha=1,
                          waveform_table=WAVEFORM_SINE, sample_rate=44100)


def test_tremolos():
    """ Tests for the creation of multiple tremolos. """

    # Test case: valid parameters
    sonic_vector = np.ones(44100)  # Esempio di vettore sonico
    envelope = envelopes.tremolos(durations=[[2, 3]], tremolo_freqs=[[2, 6]],
                                  max_db_devs=[[10, 20]], alpha=[[1, 1]],
                                  waveform_tables=[[WAVEFORM_SINE,
                                                    WAVEFORM_SINE]],
                                  sonic_vector=sonic_vector, sample_rate=44100)
    assert isinstance(envelope, np.ndarray)
    assert len(envelope) == 44100
    assert np.all(envelope >= 0)

    # Test case: invalid parameters
    with pytest.raises(ValueError):
        envelopes.tremolos(durations=[[-2, 3]], tremolo_freqs=[[2, 6]],
                           max_db_devs=[[10, 20]], alpha=[[1, 1]],
                           waveform_tables=[[WAVEFORM_SINE, WAVEFORM_SINE]],
                           sonic_vector=sonic_vector, sample_rate=44100)
