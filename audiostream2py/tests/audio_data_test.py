"""Tests for AudioSegment"""
import math

from audiostream2py import AudioSegment, PaStatusFlags
import pytest
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


def _create_test_audiosegment() -> AudioSegment:
    start_date = 10
    end_date = 20
    waveform = bytes(range(15))
    frame_count = 5
    status_flags = PaStatusFlags.paNoError
    return AudioSegment(start_date, end_date, waveform, frame_count, status_flags)


def test_properties() -> None:
    audioseg = _create_test_audiosegment()
    _assert_equality(audioseg.bt, 10)
    _assert_equality(audioseg.tt, 20)
    _assert_equality(audioseg.frame_size, 3)
    _assert_equality(audioseg.duration, 10)
    _assert_equality(audioseg.frame_period, 2)
    _assert_equality(audioseg.frame_rate, 0.5)


def test_comparisons() -> None:
    audioseg = _create_test_audiosegment()
    assert 9 < audioseg < 11
    assert (
        AudioSegment(9, 10, b'\xff', 1, None)
        < audioseg
        < AudioSegment(11, 12, b'\xff', 1, None)
    )
    assert audioseg == AudioSegment(10, 20, b'\xff', 1, None)


def test_empty() -> None:
    assert AudioSegment.empty().is_empty()


def test_get_slice() -> None:
    audioseg = _create_test_audiosegment()
    _assert_is_empty(audioseg[7:9])
    _assert_is_empty(audioseg[7:10])
    _assert_audiosegment_values(audioseg[7:10.2], 10, 12, bytes(range(0, 3)), 1)
    _assert_audiosegment_values(audioseg[10:10.2], 10, 12, bytes(range(0, 3)), 1)
    _assert_is_empty(audioseg[10.1:10.2])
    _assert_audiosegment_values(audioseg[10.1:14], 12, 14, bytes(range(3, 6)), 1)
    _assert_is_empty(audioseg[14:14])
    _assert_audiosegment_values(audioseg[14:18], 14, 18, bytes(range(6, 12)), 2)
    _assert_audiosegment_values(audioseg[14:20], 14, 20, bytes(range(6, 15)), 3)
    _assert_audiosegment_values(audioseg[14:20.2], 14, 20, bytes(range(6, 15)), 3)
    _assert_is_empty(audioseg[18.1:20.2])
    _assert_is_empty(audioseg[20:20.2])
    _assert_is_empty(audioseg[20.1:20.2])


def test_get_frame() -> None:
    audioseg = _create_test_audiosegment()
    _assert_audiosegment_values(audioseg[10], 10, 12, bytes(range(0, 3)), 1)
    _assert_audiosegment_values(audioseg[10.1], 10, 12, bytes(range(0, 3)), 1)
    _assert_audiosegment_values(audioseg[12], 12, 14, bytes(range(3, 6)), 1)
    _assert_audiosegment_values(audioseg[18], 18, 20, bytes(range(12, 15)), 1)
    _assert_audiosegment_values(audioseg[18.1], 18, 20, bytes(range(12, 15)), 1)


def test_get_ts_of_frame_index() -> None:
    audioseg = _create_test_audiosegment()
    _assert_equality(audioseg.get_ts_of_frame_index(0), (10, 12))
    _assert_equality(audioseg.get_ts_of_frame_index(1), (12, 14))
    _assert_equality(audioseg.get_ts_of_frame_index(4), (18, 20))
    _assert_equality(audioseg.get_ts_of_frame_index(-1), (18, 20))
    _assert_equality(audioseg.get_ts_of_frame_index(-5), (10, 12))


def test_concatenate() -> None:
    audioseg1 = AudioSegment(-1.5, 0, bytes(range(0, 15)), 3, PaStatusFlags.paNoError)
    audioseg2 = AudioSegment(
        0, 3, bytes(range(15, 45)), 6, PaStatusFlags.paInputOverflow
    )
    audioseg3 = AudioSegment(
        3, 4.5, bytes(range(45, 60)), 3, PaStatusFlags.paInputUnderflow
    )
    empty = AudioSegment.empty()
    concat = AudioSegment.concatenate(
        [audioseg1, audioseg3, empty, audioseg2, audioseg1]
    )
    _assert_audiosegment_values(concat, -1.5, 4.5, bytes(range(0, 60)), 12)
    assert concat.status_flags == 3


def test_error_handling() -> None:
    normal_kw = dict(
        start_date=10,
        end_date=20,
        waveform=bytes(range(15)),
        frame_count=5,
        status_flags=PaStatusFlags.paNoError,
    )
    normal_seg = AudioSegment(**normal_kw)
    empty_seg = AudioSegment.empty()

    def copy_and_update(original_dict=normal_kw, **updates):
        copied_dict = original_dict.copy()
        copied_dict.update(updates)
        return copied_dict

    # Test start_date and end_date
    AudioSegment(**copy_and_update(start_date=1, end_date=2))
    with pytest.raises(ValueError):
        AudioSegment(**copy_and_update(start_date=20, end_date=10))
    with pytest.raises(ValueError):
        AudioSegment(**copy_and_update(start_date=20, end_date=20))
    # Test waveform and frame_count
    AudioSegment(**copy_and_update(waveform=b'\xff', frame_count=1))
    AudioSegment(**copy_and_update(waveform=b'\xff' * 6, frame_count=3))
    with pytest.raises(ValueError):
        AudioSegment(**copy_and_update(waveform=b''))
    with pytest.raises(ValueError):
        AudioSegment(**copy_and_update(frame_count=0))
    with pytest.raises(ValueError):
        AudioSegment(**copy_and_update(frame_count=-1))
    with pytest.raises(ValueError):
        AudioSegment(**copy_and_update(waveform=b'\xff', frame_count=2))
    # Test frame index type
    normal_seg.get_ts_of_frame_index(1)
    with pytest.raises(TypeError):
        normal_seg.get_ts_of_frame_index(1.0)
    normal_seg.get_ts_of_frame_index(normal_seg.frame_count - 1)
    normal_seg.get_ts_of_frame_index(-normal_seg.frame_count)
    with pytest.raises(IndexError):
        normal_seg.get_ts_of_frame_index(normal_seg.frame_count)
    with pytest.raises(IndexError):
        normal_seg.get_ts_of_frame_index(-normal_seg.frame_count - 1)
    # Test slice type
    normal_seg[2:4]
    normal_seg[normal_seg.bt : normal_seg.tt]
    with pytest.raises(TypeError):
        normal_seg[2:4:1]
    with pytest.raises(IndexError):
        normal_seg[normal_seg.tt + 1]
    with pytest.raises(IndexError):
        normal_seg[normal_seg.bt - 1]
    # Test equality with empty
    normal_seg == normal_seg
    normal_seg < normal_seg
    normal_seg > normal_seg
    with pytest.raises(TypeError):
        empty_seg == empty_seg
    with pytest.raises(TypeError):
        normal_seg == empty_seg
    with pytest.raises(TypeError):
        empty_seg == normal_seg
    with pytest.raises(TypeError):
        normal_seg > empty_seg
    with pytest.raises(TypeError):
        empty_seg > normal_seg
    with pytest.raises(TypeError):
        normal_seg < empty_seg
    with pytest.raises(TypeError):
        empty_seg < normal_seg
    assert normal_seg[normal_seg.bt - 20 : normal_seg.tt + 20] == normal_seg
    assert normal_seg[normal_seg.bt + 1 : normal_seg.tt - 1] > normal_seg


def _assert_equality(item1, item2) -> None:
    assert item1 == item2, f'Expected {item2}; Got {item1}'


def _assert_audiosegment_values(
    audioseg: AudioSegment, bt, tt, bytes_, frame_count
) -> None:
    _assert_equality(audioseg.start_date, bt)
    _assert_equality(audioseg.end_date, tt)
    _assert_equality(audioseg.waveform, bytes_)
    _assert_equality(audioseg.frame_count, frame_count)


def _assert_is_empty(audioseg: AudioSegment) -> None:
    assert audioseg.is_empty(), f'Expected empty; Got {audioseg}'
