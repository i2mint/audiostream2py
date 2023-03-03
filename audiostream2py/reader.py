"""BufferReader specific to AudioSegment"""

from typing import List

from stream2py import BufferReader

from audiostream2py.data import AudioSegment


class OverlapsPastError(IndexError):
    pass


class OverlapsFutureError(IndexError):
    pass


class AudioBufferReader(BufferReader):
    def __getitem__(self, item) -> AudioSegment:
        if isinstance(item, slice):
            return self.get_slice(item)
        with self._buffer.reader_lock() as reader:
            return reader.find_le(item)

    def get_slice(self, s: slice) -> AudioSegment:
        """Slice starting from start_date and stopping at end_date. Trims first and last
        AudioSegment based on timestamps. Then joins all the AudioSegment objects.

        :param s: slice object
        :return: concatenated AudioSegment
        """
        with self._buffer.reader_lock() as reader:
            if s.start:
                head: AudioSegment = reader[0]
                if s.start < head.start_date:
                    raise OverlapsPastError(
                        f'You asked for {s}, but the buffer starts from: {head.start_date}'
                    )
                start = reader.key(reader.find_le(s.start))
            else:
                start = None

            if s.stop:
                tail: AudioSegment = reader[-1]
                if s.stop > tail.end_date:
                    raise OverlapsFutureError(
                        f'You asked for {s}, but the buffer stops at: {tail.end_date}'
                    )
                if tail.start_date < s.stop < tail.end_date:
                    stop = reader.key(tail)
                else:
                    stop = reader.key(reader.find_le(s.stop))
            else:
                stop = None

            items: List[AudioSegment] = reader.range(start, stop, s.step)
        if s.start:
            items[0] = items[0][s.start :]
        if s.stop:
            items[-1] = items[-1][: s.stop]
        return AudioSegment.concatenate(items)
