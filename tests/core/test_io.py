""" Tests for music.core.io """
from unittest.mock import mock_open, patch
import numpy as np
from music.core import io


def test_read_wav():
    """ Tests the reading of PCM from a mock WAV file. """

    # Dati PCM casuali per il file WAV fittizio
    pcm_data = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)

    # Costruzione del contenuto del file WAV fittizio
    wav_content = b'RIFF' + \
                  b'\x00\x00\x00\x00' + \
                  b'WAVE' + \
                  b'fmt ' + \
                  b'\x10\x00\x00\x00' + \
                  b'\x01\x00' + \
                  b'\x01\x00' + \
                  b'\x80\x3E\x00\x00' + \
                  b'\x80\x3E\x00\x00' + \
                  b'\x02\x00\x10\x00' + \
                  b'data' + \
                  len(pcm_data).to_bytes(4, 'little') + \
                  pcm_data.tobytes()

    # Impostazione del mock_open per simulare la lettura del file WAV
    with patch("builtins.open", mock_open(read_data=wav_content)) as mock_file:
        # Chiamata alla funzione read_wav, che legge il file WAV fittizio
        audio_data = io.read_wav('mock_file.wav')
        mock_file.assert_called_once_with('mock_file.wav', 'rb')

    # Verifica che i dati audio letti corrispondano ai dati audio attesi
    assert np.array_equal(audio_data, pcm_data)


def test_write_wav_mono():
    """ Tests the writing of mono PCM data into a WAV file"""

    sonic_vector = np.random.uniform(-1, 1, 44100)
    mock_open_func = mock_open()

    with patch('builtins.open', mock_open_func), \
            patch('music.core.io.write_wav_mono'):
        io.write_wav_mono(sonic_vector, filename="fake_file.wav")
        audio_data = io.read_wav("fake_file.wav")
    assert np.array_equal(audio_data, sonic_vector)


def test_write_wav_stereo():
    """Test the writing of a stereo WAV file."""

    sonic_vector_left = np.random.uniform(-1, 1, 44100)
    sonic_vector_right = np.random.uniform(-1, 1, 44100)
    sonic_vector_stereo = np.vstack((sonic_vector_left, sonic_vector_right))

    mock_wav_file = "stereo_mock_file.wav"
    mock_open_func = mock_open()

    with patch('builtins.open', mock_open_func), \
            patch('music.core.io.write_wav_stereo'):
        io.write_wav_stereo(sonic_vector_stereo, filename=mock_wav_file)
        audio_data = io.read_wav(mock_wav_file)

    expected_pcm_left = audio_data[:, 0]
    expected_pcm_right = audio_data[:, 1]

    assert np.array_equal(sonic_vector_left, expected_pcm_left)
    assert np.array_equal(sonic_vector_right, expected_pcm_right)
