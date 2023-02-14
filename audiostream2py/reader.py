"""BufferReader specific to AudioData"""

from typing import List

from stream2py import BufferReader
from creek.infinite_sequence import (
    OverlapsPastError,
    OverlapsFutureError,
)

from audiostream2py.data import AudioData


class AudioBufferReader(BufferReader):
    def __getitem__(self, item) -> AudioData:
        if isinstance(item, slice):
            return self.slice(item)
        with self._buffer.reader_lock() as reader:
            return reader.find_le(item)

    def slice(self, s: slice) -> AudioData:
        """Slice starting from start_date and stopping at end_date. Trims first and last AudioData
        based on timestamps. Then joins all the AudioData objects.

        :param s: slice object
        :return: concatenated AudioData
        """
        with self._buffer.reader_lock() as reader:
            if s.start:
                head: AudioData = reader.head(peek=True)
                if s.start < head.start_date:
                    raise OverlapsPastError(
                        f'You asked for {s}, but the buffer starts from: {head.start_date}'
                    )
                start = reader.key(reader.find_le(s.start))
            else:
                start = None

            if s.stop:
                tail: AudioData = reader.tail(peek=True)
                if s.stop > tail.end_date:
                    raise OverlapsFutureError(
                        f'You asked for {s}, but the buffer stops at: {tail.end_date}'
                    )
                elif tail.start_date < s.stop < tail.end_date:
                    stop = reader.key(tail)
                else:
                    stop = reader.key(reader.find_le(s.stop))
            else:
                stop = None

            items: List[AudioData] = reader.range(start, stop, s.step)
        if s.start:
            items[0] = items[0][s.start :]
        if s.stop:
            items[-1] = items[-1][: s.stop]
        return AudioData.concatenate(items)
