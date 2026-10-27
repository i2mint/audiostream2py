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


def test_mk_reader_forwards_blocking_and_timeout_and_read_blocks_within_timeout():
    """End to end through stream2py's buffer factory, without PyAudio."""
    import time

    from stream2py import StreamBuffer
    from stream2py.source_reader import SourceReader

    class _Empty(SourceReader):
        buffer_reader_class = AudioBufferReader
        key = staticmethod(lambda seg: seg)

        def open(self):
            pass

        def close(self):
            pass

        def read(self):
            return None

    with StreamBuffer(_Empty()) as sb:
        reader = sb.mk_reader(blocking=True, timeout=0.2)
        assert isinstance(reader, AudioBufferReader)
        assert reader.blocking is True and reader.timeout == 0.2
        t0 = time.monotonic()
        assert reader.read() is None
        assert 0.15 <= time.monotonic() - t0 < 2
