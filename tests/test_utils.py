""" Test for music.utils """
import pytest
import numpy as np
from music import utils, core


@pytest.fixture(name="data")
def data_fixture():
    """ Sample data for the tests. """
    data = {
        "hz_value": 440.0,
        "midi_value": 69.0,
        "midi_interval": 12.0,
        "start_freq": 220.0,
        "semitones": [0, 7, 7, 4, 7, 0],
        "vector": np.array([1, 2, 3, 4]),
        "matrix": np.array([[1, 2, 3], [4, 5, 6]]),
        "sonic_vector1": np.array([0.5, 0.7, -0.3]),
        "sonic_vector2": np.array([-0.2, 0.9, 0.1]),
        "duration": 0.5,
        "sample_rate": 44100
    }
    return data


def test_horizontal_stack(data):
    """ Test horizontal stack. """
    result = utils.horizontal_stack(data["vector"],
                                    data["matrix"])
    assert result.shape == (2, 7)


def test_db_to_amp():
    """ Test conversion from dB to amplitude. """
    result = utils.db_to_amp(6)
    assert np.isclose(result, 2.0, atol=0.005)


def test_amp_to_db():
    """ Test conversion from amplitude to dB. """
    result = utils.amp_to_db(2.0)
    assert np.isclose(result, 6.0, atol=0.025)


def test_hz_to_midi(data):
    """ Test conversion of Hz value to MIDI tone. """
    result = utils.hz_to_midi(data["hz_value"])
    assert np.isclose(result, 69.0)


def test_midi_to_hz(data):
    """ Test conversion of MIDI tone to Hz value. """
    result = utils.midi_to_hz(data["midi_value"])
    assert np.isclose(result, 440.0)


def test_midi_to_hz_interval(data):
    """ Test conversion of MIDI interval to Hz interval. """
    result = utils.midi_to_hz_interval(data["midi_interval"])
    assert np.isclose(result, 2.0)


def test_pitch_to_freq(data):
    """ Test creation of list of frequencies from a starting frequency
        and a list of semitones.
    """
    result = utils.pitch_to_freq(data["start_freq"],
                                 data["semitones"])
    assert len(result) == len(data["semitones"])


def test_mix(data):
    """ Test the mix of two sonic vectors. """
    result = utils.mix(data["sonic_vector1"],
                       data["sonic_vector2"])
    assert result.shape == data["sonic_vector1"].shape


def test_mix_stereo(data):
    """ Test the mix of two bidimensional sonic vectors. """
    result = utils.mix_stereo(data["sonic_vector1"],
                              data["sonic_vector2"])
    assert result.shape == (2, data["sonic_vector1"].shape[0])


def test_resolve_stereo(data):
    """ Test the resolution of stereo args. """
    # Dizionario degli argomenti
    args = {'freq': [440, 443]}
    
    # Argomenti stereo da risolvere
    stereo_vars = ['freq']
    
    # Chiamata alla funzione da testare
    result = utils.resolve_stereo(core.synths.note, args, stereo_vars)
    
    # Risultato atteso
    expected = np.array([[4, 6], [5, 6]])
    
    # Verifica che il risultato sia corretto
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"


def test_convert_to_stereo(data):
    """ Test the conversion of a mono sonic vector into a stereo one. """
    result = utils.convert_to_stereo(data["vector"])
    assert result.shape == (2, data["vector"].shape[0])


def test_mix_with_offset(data):
    """ Test mix with offset. """
    result = utils.mix_with_offset(data["sonic_vector1"],
                                   data["sonic_vector2"],
                                   data["duration"],
                                   sample_rate=data["sample_rate"])
    assert result.shape == (max(len(data["sonic_vector1"]),
                                len(data["sonic_vector2"])),)


def test_mix_with_offset_(data):
    """ Test mix with offset. """
    result = utils.mix_with_offset_(data["sonic_vector1"],
                                    data["sonic_vector2"],
                                    data["duration"])
    assert result.shape == (max(len(data["sonic_vector1"]),
                                len(data["sonic_vector2"])),)


def test_pan_transitions(data):
    """ Test pan transitions. """
    result = utils.pan_transitions(sample_rate=data["sample_rate"],
                                   sonic_vector=[data["sonic_vector1"],
                                                 data["sonic_vector2"]]
                                   )
    assert result.shape == (2, max(len(data["sonic_vector1"]),
                                   len(data["sonic_vector2"])))


def test_mix2(data):
    """ Test a mix of sonic vectors. """
    result = utils.mix2([data["sonic_vector1"],
                         data["sonic_vector2"]])
    assert result.shape == (max(len(data["sonic_vector1"]),
                                len(data["sonic_vector2"])),)


# def test_profile():
#     """ Test della funzione profile, se necessario. """
#     pass


def test_rhythm_to_durations(data):
    """ Test the calculation of rythmic patterns durations. """
    result = utils.rhythm_to_durations([4, 2, 2, 4, 1, 1, 1, 1, 2, 2, 4],
                                       duration=data["duration"])
    assert len(result) == 11
