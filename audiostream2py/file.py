"""WavFileSourceReader to read a wav file and produce read output similar to PyAudioSourceReader."""

import wave
from functools import cached_property
from pathlib import Path
from typing import Union

from stream2py import SourceReader
from stream2py.utility.typing_hints import ComparableType

from audiostream2py import AudioSegment, PaStatusFlags
from audiostream2py.reader import AudioBufferReader


class WavFileSourceReader(SourceReader):
    buffer_reader_class = AudioBufferReader  # BufferReader specific to AudioSegment

    def __init__(
        self, file: Union[str, bytes, Path], *, frames_per_buffer=1024, start_date=0
    ):
        super().__init__()
        self.file = file if not isinstance(file, Path) else str(file)
        self.frames_per_buffer = frames_per_buffer
        self.start_date = start_date
        self._fp: wave.Wave_read = wave.open(self.file, 'rb')

        self._index = 0
        self._next_bt = self.start_date

    @property
    def info(self) -> dict:
        return {
            'channels': self._fp.getnchannels(),
            'width': self._fp.getsampwidth(),
            'rate': self._fp.getframerate(),
            'n_frames': self._fp.getnframes(),
            'frames_per_buffer': self.frames_per_buffer,
            'bt': self.start_date,
        }

    def open(self) -> None:
        self._fp.rewind()
        self._index = 0
        self._next_bt = self.start_date

    def close(self) -> None:
        pass

    @cached_property
    def sr(self) -> int:
        return self._fp.getframerate()

    @cached_property
    def bytes_per_frame(self) -> int:
        width = self._fp.getsampwidth()
        n_channels = self._fp.getnchannels()
        return width * n_channels

    def read(self) -> AudioSegment:
        data = self._fp.readframes(self.frames_per_buffer)
        if len(data) == 0:
            return None

        n_frames = int(len(data) / self.bytes_per_frame)
        bt = self._next_bt
        tt = bt + n_frames * 1e6 / self.sr
        self._next_bt = tt
        return AudioSegment(
            start_date=bt,
            end_date=tt,
            waveform=data,
            frame_count=n_frames,
            status_flags=PaStatusFlags.paNoError,
        )

    def key(self, data: AudioSegment) -> ComparableType:
        """AudioSegment is a ComparableType

        :param data: {'bt': timestamp, 'wf': waveform, 'status_flags': status_flags}
        :return: AudioSegment
        """
        return data.start_date

    def __del__(self):
        self._fp.close()
