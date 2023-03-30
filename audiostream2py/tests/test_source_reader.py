"""Test PyAudioSourceReader"""
import time
from contextlib import contextmanager
from unittest.mock import patch

from audiostream2py import PyAudioSourceReader, PaStatusFlags, AudioSegment


@contextmanager
def mock_pyaudio_context():
    class MockPyAudio:
        @classmethod
        def return_true(cls, *_a, **_kw):
            return True

        def __getattr__(self, item):
            return self.return_true

    yield MockPyAudio()


def test_stream_callback():
    """Test normal data reads and input overflow"""

    input_device_index = 999  # fake device index
    width = 2
    channels = 1
    frames_per_buffer = 10
    rate = 10

    with patch('time.time') as mock_time, patch(
        'audiostream2py.PyAudioSourceReader._pyaudio'
    ) as mock_source_pyaudio:
        mock_source_pyaudio.return_value = mock_pyaudio_context()
        mock_time.side_effect = list(range(1, 100))
        mock_pcm_bytes = b''.join(
            i.to_bytes(width, 'little') * channels for i in range(frames_per_buffer)
        )
        source = PyAudioSourceReader(
            input_device_index=input_device_index,
            width=width,
            channels=channels,
            frames_per_buffer=frames_per_buffer,
            rate=rate,
        )
        # simulate normal
        for i in range(1, 10):
            source._stream_callback(
                mock_pcm_bytes, frames_per_buffer, None, PaStatusFlags.paNoError
            )
            assert len(source.data) == i
            audio_segment: AudioSegment = source.data[-1]
            assert audio_segment.waveform == mock_pcm_bytes
            assert audio_segment.frame_count == frames_per_buffer
            assert audio_segment.start_date == (i - 1) * 1e6
            assert audio_segment.end_date == i * 1e6

        data_count = len(source.data)
        last_audio_segment: AudioSegment = source.data[-1]
        # simulate input overflow
        input_overflow_time = 10
        for _ in range(input_overflow_time):
            time.time()  # increment mock_time
            source._stream_callback(
                mock_pcm_bytes, frames_per_buffer, None, PaStatusFlags.paInputOverflow
            )
            assert len(source.data) == data_count
            audio_segment: AudioSegment = source.data[-1]
            assert audio_segment == last_audio_segment

        # simulate first normal after input overflow
        source._stream_callback(
            mock_pcm_bytes, frames_per_buffer, None, PaStatusFlags.paNoError
        )
        assert len(source.data) == data_count + 2
        error_filler: AudioSegment = source.data[-2]
        final_audio_segment: AudioSegment = source.data[-1]

        assert error_filler.waveform == (
            PyAudioSourceReader.FILL_VALUE.to_bytes(width, 'little')
            * channels
            * frames_per_buffer
            * input_overflow_time
        )

        assert error_filler.frame_count == frames_per_buffer * input_overflow_time
        assert error_filler.start_date == last_audio_segment.end_date
        assert error_filler.end_date == final_audio_segment.start_date
        assert error_filler.start_date != error_filler.end_date

        assert final_audio_segment.waveform == mock_pcm_bytes
        assert final_audio_segment.frame_count == frames_per_buffer
        assert (
            final_audio_segment.start_date
            == last_audio_segment.start_date + (input_overflow_time + 1) * 1e6
        )
        assert (
            final_audio_segment.end_date
            == last_audio_segment.start_date + (input_overflow_time + 2) * 1e6
        )
