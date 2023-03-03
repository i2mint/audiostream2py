"""Tests for AudioSegment"""
import math

from audiostream2py import AudioSegment, PaStatusFlags
from recode import decode_pcm_bytes


def mock_audio_data(rate: int, width: int, channels: int, frames_per_buffer: int):
    def mock_pcm_bytes():
        """Value of sample is the frame index"""
        return b''.join(
            i.to_bytes(width, 'little') * channels for i in range(frames_per_buffer)
        )

    return AudioSegment(
        start_date=0,
        end_date=1e6 * frames_per_buffer / rate,
        waveform=mock_pcm_bytes(),
        frame_count=frames_per_buffer,
        status_flags=PaStatusFlags.paNoError,
    )


def test_process_to_find_first_sample_of_session():
    def first_sample_of_session(bt_i, t_start, tt_i, n):
        """Process defined in timestamping discussion"""
        m = math.ceil((t_start - bt_i) / (tt_i - bt_i) * n)
        t_m = bt_i + m / n * (tt_i - bt_i)
        return m, t_m

    ad = mock_audio_data(44100, 2, 1, 4096)
    t_start = 0.05 * 1e6
    s_start, s_time = first_sample_of_session(
        ad.start_date, t_start, ad.end_date, ad.frame_count
    )
    session = ad[t_start:]
    session_start = decode_pcm_bytes(session.waveform, 2, 1)[0]
    assert s_start == session_start, 'Wrong Start Sample'
    assert s_time == session.start_date, 'Wrong Start Time'


def test_process_to_find_last_sample_of_session():
    def last_sample_of_session(bt_i, t_stop, tt_i, n):
        """Process defined in timestamping discussion"""
        m = math.floor((t_stop - bt_i) / (tt_i - bt_i) * n)
        t_m1 = bt_i + (m + 1) / n * (tt_i - bt_i)
        return m, t_m1

    ad = mock_audio_data(44100, 2, 1, 4096)
    t_stop = 0.05 * 1e6
    s_stop, s_time = last_sample_of_session(
        ad.start_date, t_stop, ad.end_date, ad.frame_count
    )
    session = ad[:t_stop]
    session_stop = decode_pcm_bytes(session.waveform, 2, 1)[-1]
    assert s_stop == session_stop, 'Wrong End Sample'
    assert s_time == session.end_date, 'Wrong End Time'
