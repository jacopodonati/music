""" Tests for music.core.synths.notes """
import numpy as np
import pytest
from scipy.fft import fft
from music.utils import WAVEFORM_SAWTOOTH, WAVEFORM_SINE
from music.core.synths import notes


def test_note():
    """ Test the synth of a single clean note. """

    # Test case: Check length
    duration = 2
    sample_rate = 44100
    generated_note = notes.note(duration=duration, sample_rate=sample_rate)
    expected_length = duration * sample_rate
    assert len(generated_note) == expected_length

    # Test case: Check frequency of the custom note
    freq = 443
    duration = 2
    waveform_table = WAVEFORM_SAWTOOTH
    custom_note = notes.note(
        freq=freq,
        duration=duration,
        waveform_table=waveform_table
        )
    fft_result = fft(custom_note)
    freqs = np.fft.fftfreq(len(fft_result), 1/44100)
    dominant_freq = freqs[np.argmax(np.abs(fft_result))]
    assert np.isclose(dominant_freq, freq, rtol=0.01)


def test_note_with_doppler():
    """ Test suite for the synth of a note with Doppler effect. """

    # Test case: Check stereo effect
    doppler_note = notes.note_with_doppler(stereo=True)
    assert doppler_note.shape[0] == 2

    # Test case: Check length
    duration = 2
    sample_rate = 44100
    generated_note = notes.note_with_doppler(duration=duration,
                                             sample_rate=sample_rate)
    expected_length = duration * sample_rate
    assert len(generated_note[0]) == len(generated_note[1])
    assert len(generated_note[0]) == expected_length

    # Test case: Check the localization effect of the note with Doppler.
    note_left = notes.note_with_doppler(x=[-10, 0])
    note_right = notes.note_with_doppler(x=[0, 10])
    assert np.mean(note_left[0, :]) > np.mean(note_right[0, :])

    # Test case: Check the intensity of the note with Doppler.
    note_slow = notes.note_with_doppler(zeta=0.1)
    note_fast = notes.note_with_doppler(zeta=0.5)
    assert np.max(np.abs(note_fast)) > np.max(np.abs(note_slow))


def test_note_with_fm():
    """ Test for the synth of a note with FM. """

    # Test case: Check length
    duration = 2
    sample_rate = 44100
    generated_note = notes.note_with_fm(duration=duration,
                                        sample_rate=sample_rate)
    expected_length = duration * sample_rate
    assert len(generated_note) == expected_length

    # Test case: check max deviation of FM
    freq = 440
    fm = 100
    max_fm_deviation = 50
    waveform_table = WAVEFORM_SINE

    fm_note = notes.note_with_fm(freq=freq, duration=duration, fm=fm,
                                 max_fm_deviation=max_fm_deviation,
                                 waveform_table=waveform_table,
                                 sample_rate=sample_rate)
    fft_result = fft(fm_note)
    freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    dominant_freq = freqs[np.argmax(np.abs(fft_result))]
    expected_max_deviation = freq + max_fm_deviation

    assert np.isclose(dominant_freq, freq, rtol=0.01)
    assert np.any(np.isclose(freqs, expected_max_deviation, rtol=0.01))


def test_note_with_phase():
    """ Test for the synth of a note with phase """

    # Test case: Check length
    freq = 443
    duration = 2
    sample_rate = 44100
    generated_note = notes.note_with_phase(freq=freq, duration=duration,
                                           sample_rate=sample_rate)
    expected_length = duration * sample_rate
    assert len(generated_note) == expected_length

    # Test case: Check phase
    fft_result = fft(generated_note)
    phase_spectrum = np.angle(fft_result)
    dominant_phase = np.unwrap(phase_spectrum)[np.argmax(np.abs(fft_result))]
    assert np.isclose(dominant_phase, freq, atol=0.01)


def test_note_with_glissando():
    """ Test synth of a note with glissando """

    # Test case: Check length
    duration = 2
    sample_rate = 44100
    generated_note = notes.note_with_glissando(duration=duration,
                                               sample_rate=sample_rate)
    expected_length = duration * sample_rate
    assert len(generated_note) == expected_length

    # Test case: Check linear transition
    # TODO: check atol
    start_freq = 220
    end_freq = 440
    generated_note = notes.note_with_glissando(start_freq=start_freq,
                                               end_freq=end_freq,
                                               duration=duration,
                                               sample_rate=sample_rate)
    expected_frequency = np.linspace(start_freq, end_freq,
                                     duration * sample_rate)
    assert np.allclose(generated_note, expected_frequency, atol=1e-3)

    # Test case: Check exponential transition
    # TODO: check atol
    alpha = 0.5
    generated_note = notes.note_with_glissando(start_freq=start_freq,
                                               end_freq=end_freq,
                                               duration=duration, alpha=alpha,
                                               sample_rate=sample_rate)
    expected_frequency = start_freq * \
        (end_freq / start_freq) ** ((np.arange(duration * sample_rate) /
                                    (duration * sample_rate - 1)) ** alpha)
    assert np.allclose(generated_note, expected_frequency, atol=1e-3)


def test_note_with_glissando_vibrato():
    """ Test synth of a note with both glissando and vibrato """

    # Test case: Check length.
    duration = 2
    sample_rate = 44100
    default_note = notes.note_with_glissando_vibrato(duration=duration,
                                                     sample_rate=sample_rate)
    expected_length = duration * sample_rate
    assert len(default_note) == expected_length

    # Test case: Check dominant frequency
    note = notes.note_with_glissando_vibrato(start_freq=220, end_freq=440,
                                             duration=2, vibrato_freq=4,
                                             max_glissando_dev=2)
    dominant_freq = np.argmax(np.abs(np.fft.fft(note)))
    assert pytest.approx(dominant_freq, abs=10) == 440

    # Test case: Check vibrato
    note = notes.note_with_glissando_vibrato(vibrato_freq=4,
                                             max_glissando_dev=2)
    dominant_freq = np.argmax(np.abs(np.fft.fft(note)))
    assert pytest.approx(dominant_freq, abs=10) != 220


def test_note_with_two_vibratos():
    """ Test synth of a note with two vibrato """

    # Test case: Check length.
    duration = 2
    sample_rate = 44100
    default_note = notes.note_with_two_vibratos(duration=duration,
                                                sample_rate=sample_rate)
    expected_length = duration * sample_rate
    assert len(default_note) == expected_length

    # Test case: Check if the two vibratos cancel each other out.
    freq = 220
    duration = 4
    first_vibrato = 440
    second_vibrato = first_vibrato
    nu1 = 1
    nu2 = nu1 * -1
    alpha = 0
    cancelled_note = \
        notes.note_with_two_vibratos(freq=220, duration=4,
                                     vibrato_freq=first_vibrato,
                                     secondary_vibrato_freq=second_vibrato,
                                     nu1=nu1, nu2=nu2, alphav1=alpha,
                                     alphav2=alpha)
    fft_result = fft(cancelled_note)
    freqs = np.fft.fftfreq(len(fft_result), 1/44100)
    dominant_freq = freqs[np.argmax(np.abs(fft_result))]
    assert np.isclose(dominant_freq, freq, rtol=0.01)


def test_note_with_two_vibratos_glissando():
    """ Test synth of a note with two vibratos and a glissando """

    # Test case: Check length.
    duration = 2
    sample_rate = 44100
    default_note = \
        notes.note_with_two_vibratos_glissando(duration=duration,
                                               sample_rate=sample_rate)
    expected_length = duration * sample_rate
    assert len(default_note) == expected_length

    # TODO: add more test cases


def test_note_with_vibrato():
    """ Test synth of a note with a vibrato """

    # Test case: Check length.
    duration = 2
    sample_rate = 44100
    default_note = notes.note_with_vibrato(duration=duration,
                                           sample_rate=sample_rate)
    expected_length = duration * sample_rate
    assert len(default_note) == expected_length

    # TODO: add more test cases


def test_note_with_vibrato_seq_localization():
    """ Test synth of a sound with arbitrary meta-vibratos, transitions of
        frequency and localization.
    """

    # TODO: add more test cases


def test_note_with_vibratos_glissandos():
    """ Test synth of a note with multiple vibratos and glissandos. """

    # TODO: add more test cases


def test_trill():
    """ Test synth of a trill """

    # Test case: Check length.
    duration = 2
    sample_rate = 44100
    default_note = \
        notes.trill(duration=duration, sample_rate=sample_rate)
    expected_length = duration * sample_rate
    assert len(default_note) == expected_length

    # TODO: add more test cases
