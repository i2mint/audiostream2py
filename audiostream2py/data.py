"""AudioSegment dataclass for read values from PyAudioSourceReader.  Provides methods to concatenate
multiple AudioSegment objs and slicing an individual AudioSegment obj while maintaining the correct
timestamps and frame counts.
"""

import math
from functools import cached_property
from typing import Union, Sequence, Tuple, Literal, Optional
from dataclasses import dataclass

from audiostream2py.enum import PaStatusFlags


@dataclass(frozen=True)
class AudioSegment:
    start_date: Union[int, float]
    end_date: Union[int, float]
    waveform: bytes
    frame_count: int
    status_flags: PaStatusFlags

    def __post_init__(self):
        if not self.is_empty():
            if self.end_date <= self.start_date:
                raise ValueError('end_date must be higher than start_date')
            if len(self.waveform) == 0:
                raise ValueError(
                    'waveform cannot be empty. Use AudioSegment.empty() to create empty segment'
                )
            if self.frame_count <= 0:
                raise ValueError('frame_count must be positive')
            if len(self.waveform) % self.frame_count != 0:
                raise ValueError('frame_count must be a divisor of len(waveform)')

    @classmethod
    def empty(cls):
        """Clean way of creating empty segments"""
        return cls(None, None, b'', 0, PaStatusFlags.paNoError)

    def is_empty(self) -> bool:
        """Clean way of testing if a segment is empty"""
        cond1 = self.start_date is None
        cond2 = self.end_date is None
        cond3 = len(self.waveform) == 0
        cond4 = self.frame_count == 0
        if cond1 and cond2 and cond3 and cond4:
            return True
        return False

    @property
    def bt(self) -> Optional[Union[int, float]]:
        """Legacy naming: bottom time"""
        return self.start_date

    @property
    def tt(self) -> Optional[Union[int, float]]:
        """Legacy naming: top time"""
        return self.end_date

    @cached_property
    def frame_size(self) -> Optional[int]:
        """Byte count per frame"""
        if self.is_empty():
            return None
        return int(len(self.waveform) / self.frame_count)

    @property
    def duration(self) -> Optional[Union[int, float]]:
        """AudioSegment duration, in the same unit as start and end date"""
        if self.is_empty():
            return None
        return self.end_date - self.start_date

    @property
    def frame_period(self) -> Optional[float]:
        """frame period, in the same unit as start and end date"""
        if self.is_empty():
            return None
        return self.duration / self.frame_count

    @property
    def frame_rate(self) -> Optional[float]:
        """frame rate, in the same unit as start and end date, inversed"""
        if self.is_empty():
            return None
        return 1 / self.frame_period

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}('
            f'start_date={self.start_date}, '
            f'end_date={self.end_date}, '
            f'frame_count={self.frame_count}, '
            f'status_flags={self.status_flags}'
            ')'
        )

    def __lt__(self, other: Union['AudioSegment', int, float]) -> bool:
        """Less than, comparing start_date

        :param other: AudioSegment | int | float
        :return: bool
        """
        self._test_if_not_empty_for_comparison(other)
        other_val = other.start_date if isinstance(other, AudioSegment) else other
        return self.start_date < other_val

    def __gt__(self, other: Union['AudioSegment', int, float]) -> bool:
        """Greater than, comparing start_date

        :param other: AudioSegment | int | float
        :return: bool
        """
        self._test_if_not_empty_for_comparison(other)
        other_val = other.start_date if isinstance(other, AudioSegment) else other
        return self.start_date > other_val

    def __eq__(self, other: 'AudioSegment'):
        """Equal to, comparing start_date and end_date

        :param other: AudioSegment | int | float
        :return: bool
        """
        self._test_if_not_empty_for_comparison(other)
        return self.start_date == other.start_date and self.end_date == other.end_date

    def __getitem__(self, ts: Union[slice, int, float]) -> 'AudioSegment':
        """Slice waveform and return AudioSegment with updated timestamps and data.

        :param ts: timestamp in the form of slice or scalar
        :return: Sliced AudioSegment
        """
        if self.is_empty():
            raise TypeError("can't get items of empty AudioSegment")
        if isinstance(ts, slice):
            return self._get_slice(ts)
        return self._get_frame(ts)

    def __add__(self, other: 'AudioSegment') -> 'AudioSegment':
        """Concatenates two AudioSegments. They must be contiguous."""
        return AudioSegment.concatenate([self, other])

    def __iadd__(self, other: 'AudioSegment') -> 'AudioSegment':
        return self + other

    @staticmethod
    def concatenate(audio_segments: Sequence['AudioSegment']):
        """Join a sequence of AudioSegment. AudioSegments must be contiguous. 
        If all are empty, returns an empty AudioSegment.

        :param audio_datas: list of AudioSegment
        :return: The concatenated AudioSegment.
        """
        audio_segments = {
            audioseg for audioseg in audio_segments if not audioseg.is_empty()
        }
        if not audio_segments:
            return AudioSegment.empty()
        audio_segments = sorted(audio_segments)
        bts, tts = zip(*[(audioseg.bt, audioseg.tt) for audioseg in audio_segments])
        assert all(
            tts[i] == bts[i + 1] for i in range(len(audio_segments) - 1)
        ), 'Audio segments must be contiguous'
        assert (
            len({audioseg.frame_size for audioseg in audio_segments}) == 1
        ), 'Audio segments must all have the same frame_size'
        start_date, end_date = bts[0], tts[-1]
        waveform = b''.join(audioseg.waveform for audioseg in audio_segments)
        frame_count = sum(audioseg.frame_count for audioseg in audio_segments)
        status_flags = PaStatusFlags.paNoError
        for audioseg in audio_segments:
            status_flags |= PaStatusFlags(audioseg.status_flags)

        return AudioSegment(start_date, end_date, waveform, frame_count, status_flags)

    def get_ts_of_frame_index(self, frame_idx: int) -> Tuple[float, float]:
        """get bt and tt of frame at index frame_idx

        :param frame_idx: frame index
        :return: bt, tt
        """
        if not isinstance(frame_idx, int):
            raise TypeError('Index must be an integer')
        if not -self.frame_count <= frame_idx < self.frame_count:
            raise IndexError(
                'Index out of range. '
                f'Must take values in [-{self.frame_count}, {self.frame_count}), '
                f'Got {frame_idx}.'
            )
        frame_idx = frame_idx % self.frame_count
        bt = self.start_date + frame_idx * self.frame_period
        tt = self.start_date + (frame_idx + 1) * self.frame_period
        return bt, tt

    def _get_slice(self, s: slice) -> 'AudioSegment':
        """Slice waveform and return AudioSegment with updated timestamps and data.

        Further info on methodology found in timestamping data stream discussion:
        https://miro.com/app/board/uXjVPsuJtdM=/?share_link_id=872583858418

        Note: slice 'step' is unlikely used in the traditional sense which is how it is implemented.
        A future improvement could use 'step' as a channel selector by filtering out bytes of other
        channels.
        Charlie's EDIT: changed the function so that slice steps are no longer authorized. The
        current class implementation does not support multi-channel audio segments.

        Implemented so that a series of contiguous slices, whatever their ranges and values, will 
        generate a series of AudioSegments that can be concatenated using the concatenate method, 
        i.e. a series of AudioSegments that are either contiguous with no duplicates, or empty.

        If audioseg is an AudioSegment object, then, for any increasing values of t0, t1, ..., tn:
        audioseg[t0:t1] + audioseg[t1:t2] + ... + audioseg[tn_2:tn_1] + audioseg[tn_1:tn] is equal 
        to audioseg[t0:tn]

        More details on the behavior of this method in its corresponding test function in 
        audiostream2py.test.audio_data_test

        NB: If the slice is smaller than a frame period, it is possible this method returns an empty
        AudioSegment, even if the slice is within the AudioSegment time range. This will happen if 
        the slice is located on a single and same frame. This choice has been made to meet the 
        non-duplication and reconcatenation requirements mentionned above.        
        
        :param s: slice object with start and stop by timestamp
        :return: Sliced AudioSegment
        """
        if s.step is not None:
            raise TypeError('Slice steps are not supported.')

        if s.start is not None:
            start_frame = self._nearest_frame_index(ts=s.start, rounding_type='ceil')
        else:
            start_frame = 0

        if s.stop is not None:
            end_frame = self._nearest_frame_index(ts=s.stop, rounding_type='floor')
            if end_frame is not None:
                if (
                    s.stop < self.end_date
                    and (s.stop - self.start_date) % self.frame_period == 0
                ):
                    # Does not include frame i if s.stop == bt of frame i
                    end_frame -= 1
        else:
            end_frame = self.frame_count - 1

        if None in (start_frame, end_frame) or start_frame > end_frame:
            # Happens in invalid slices (descending or out of time range)
            return self.empty()

        start_date, _ = self.get_ts_of_frame_index(start_frame)
        _, end_date = self.get_ts_of_frame_index(end_frame)

        start = start_frame * self.frame_size
        stop = (end_frame + 1) * self.frame_size
        waveform = self.waveform[start:stop]
        frame_count = (end_frame + 1) - start_frame
        return AudioSegment(
            start_date, end_date, waveform, frame_count, self.status_flags
        )

    def _get_frame(self, ts: Union[int, float]) -> 'AudioSegment':
        """Get frame at timestamp.

        :param ts: timestamp microseconds
        :return: Sliced AudioSegment
        """
        if not self.start_date <= ts < self.end_date:
            raise IndexError(
                'Timestamp out of range. '
                f'Must take values in ({self.start_date}, {self.end_date}); '
                f'Got {ts}.'
            )
        frame_idx = self._nearest_frame_index(ts=ts, rounding_type='floor')
        start_date, end_date = self.get_ts_of_frame_index(frame_idx)
        start = frame_idx * self.frame_size
        stop = (frame_idx + 1) * self.frame_size
        waveform = self.waveform[start:stop]
        frame_count = 1

        return AudioSegment(
            start_date, end_date, waveform, frame_count, self.status_flags
        )

    def _nearest_frame_index(
        self, ts, rounding_type: Literal['floor', 'ceil'] = 'ceil'
    ) -> int:
        """Calculate frame index coming just after ('ceil') or just before ('floor') given timestamp

        'ceil' for slice().start, 'floor' for slice().stop
        'floor' for indices

        Further info on methodology found in timestamping data stream discussion:
        https://miro.com/app/board/uXjVPsuJtdM=/?share_link_id=872583858418

        :param ts: timestamp in the same unit as start and end date
        :param rounding_type: 'ceil' or 'floor'
        :return: frame_index or None if ts and rounding_method does not lead to an actual frame
        """
        if rounding_type not in ('floor', 'ceil'):
            raise ValueError(
                f'{type(self).__name__}.nearest_frame_index: rounding_type must be '
                f'literal "floor" or "ceil" but got "{rounding_type}"'
            )
        rounder = getattr(math, rounding_type)
        frame_idx = rounder((ts - self.start_date) / self.frame_period)
        if frame_idx < 0 and rounding_type == 'floor':
            return None
        if frame_idx > self.frame_count - 1 and rounding_type == 'ceil':
            return None
        return min(max(frame_idx, 0), self.frame_count - 1)

    def _test_if_not_empty_for_comparison(self, other) -> None:
        if self.is_empty():
            raise TypeError("Can't compare empty audio segments")
        if isinstance(other, AudioSegment):
            if other.is_empty():
                raise TypeError("Can't compare empty audio segments")
