"""AudioData dataclass for read values from PyAudioSourceReader.  Provides methods to concatenate
multiple AudioData objs and slicing an individual AudioData obj while maintaining the correct
timestamps and frame counts.
"""

import math
from functools import cached_property
from typing import Union, Sequence, Tuple
from dataclasses import dataclass

from audiostream2py import PaStatusFlags


@dataclass(frozen=True)
class AudioData:
    start_date: Union[int, float]
    end_date: Union[int, float]
    waveform: bytes
    frame_count: int
    status_flags: PaStatusFlags

    @property
    def bt(self):
        """Legacy naming: bottom time"""
        return self.start_date

    @property
    def tt(self):
        """Legacy naming: top time"""
        return self.end_date

    @staticmethod
    def concatenate(audio_datas: Sequence['AudioData']):
        """Join a sequence of AudioData.

        :param audio_datas: list of AudioData
        :return: The concatenated AudioData.
        """
        start_date = audio_datas[0].start_date
        end_date = audio_datas[-1].end_date
        waveform = b''.join([ad.waveform for ad in audio_datas])
        frame_count = sum([ad.frame_count for ad in audio_datas])
        status_flags = PaStatusFlags.paNoError
        for ad in audio_datas:
            status_flags |= PaStatusFlags(ad.status_flags)

        return AudioData(start_date, end_date, waveform, frame_count, status_flags)

    def __lt__(self, other: Union['AudioData', int, float]) -> bool:
        """Less than comparing start_date

        :param other: AudioData | int | float
        :return: bool
        """
        other_val = other.start_date if isinstance(other, AudioData) else other
        return self.start_date < other_val

    def __gt__(self, other: Union['AudioData', int, float]) -> bool:
        """Greater than comparing start_date

        :param other: AudioData | int | float
        :return: bool
        """
        other_val = other.start_date if isinstance(other, AudioData) else other
        return self.start_date > other_val

    def __getitem__(self, index) -> 'AudioData':
        """Slice waveform and return AudioData with updated timestamps and data.

        Further info on methodology found in timestamping data stream discussion:
        https://miro.com/app/board/uXjVPsuJtdM=/?share_link_id=872583858418

        Note: slice 'step' is unlikely used in the traditional sense which is how it is implemented.
        A future improvement could use 'step' as a channel selector by filtering out bytes of other
        channels.

        :param index: timestamp or slice
        :return: Sliced AudioData
        """
        if isinstance(index, slice):
            if index.start is not None:
                start_sample, start_date = self.nearest_sample_index_and_time(
                    timestamp=index.start, rounding_type='ceil'
                )
            else:
                start_sample, start_date = None, self.start_date

            if index.stop is not None:
                end_sample, end_date = self.nearest_sample_index_and_time(
                    timestamp=index.stop, rounding_type='floor'
                )
            else:
                end_sample, end_date = None, self.end_date

            start = start_sample * self.frame_size if start_sample is not None else None
            stop = end_sample * self.frame_size if end_sample is not None else None
            step = index.step * self.frame_size if index.step is not None else None
            waveform = self.waveform[start:stop:step]
            frame_count = (
                end_sample if end_sample is not None else self.frame_count
            ) - start_sample

        else:
            sample_idx, start_date = self.nearest_sample_index_and_time(
                timestamp=index, rounding_type='floor'
            )
            _, end_date = self.nearest_sample_index_and_time(
                timestamp=index, rounding_type='ceil'
            )

            waveform = self.waveform[sample_idx : sample_idx + self.frame_size]
            frame_count = 1

        return AudioData(start_date, end_date, waveform, frame_count, self.status_flags)

    @cached_property
    def frame_size(self) -> int:
        """Byte count per frame"""
        return int(len(self.waveform) / self.frame_count)

    def nearest_sample_index_and_time(
        self, timestamp, rounding_type='ceil'
    ) -> Tuple[int, Union[int, float]]:
        """Calculate sample location and round up or down to the nearest whole sample.

        'ceil' for session start and 'floor' for session end

        Further info on methodology found in timestamping data stream discussion:
        https://miro.com/app/board/uXjVPsuJtdM=/?share_link_id=872583858418

        :param timestamp: time of sample in the same unit as start and end date
        :param rounding_type: 'ceil' or 'floor'
        :return: sample_index, sample_timestamp
        """
        rounder = getattr(math, rounding_type)
        sample_idx = int(
            rounder(
                (timestamp - self.start_date)
                / (self.end_date - self.start_date)
                * self.frame_count
            )
        )
        sample_time = self.start_date + (
            sample_idx + (0 if rounding_type == 'ceil' else 1)
        ) / self.frame_count * (self.end_date - self.start_date)
        return sample_idx, sample_time

    def __repr__(self):
        return f'{type(self).__name__}({self.start_date}, {self.end_date})'

    def __eq__(self, other: 'AudioData'):
        return self.start_date == other.start_date and self.end_date == other.end_date
