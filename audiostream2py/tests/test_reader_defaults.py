"""AudioBufferReader takes stream2py's ``blocking=`` / ``timeout=`` constructor defaults (#12)."""

import threading


from audiostream2py.reader import AudioBufferReader
from stream2py.utility.locked_sorted_deque import RWLockSortedDeque


def _reader(**kwargs):
    buffer = RWLockSortedDeque(key=lambda seg: seg.start_date)
    return AudioBufferReader(
        buffer=buffer, source_reader_info={}, stop_event=threading.Event(), **kwargs
    )


def test_defaults_are_non_blocking_without_a_timeout():
    reader = _reader()
    assert reader.blocking is False and reader.timeout is None


def test_blocking_and_timeout_are_forwarded_to_the_base_reader():
    reader = _reader(blocking=True, timeout=0.5)
    assert reader.blocking is True and reader.timeout == 0.5


def test_the_class_is_an_audio_reader_through_the_source_buffer_factory():
    from audiostream2py.audio import PyAudioSourceReader

    assert PyAudioSourceReader.buffer_reader_class is AudioBufferReader
