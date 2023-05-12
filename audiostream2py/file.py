"""WavFileSourceReader to read a wav file and produce read output similar to PyAudioSourceReader."""

import wave
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Union

from stream2py import SourceReader
from stream2py.utility.typing_hints import ComparableType

from audiostream2py import AudioSegment, PaStatusFlags
from audiostream2py.reader import AudioBufferReader


def resolve_file(file: Union[str, bytes, Path, BytesIO]) -> Union[str, BytesIO]:
    """Resolve file into format excepted by wave.open()

    :param file: file path or bytes
    :return: str or BytesIO
    """
    if isinstance(file, bytes):
        return BytesIO(file)
    if isinstance(file, Path):
        return str(file)
    return file


def file_to_audio_segment(
    file: Union[str, bytes, Path, BytesIO], start_date=0
) -> AudioSegment:
    """Read wav file into an AudioSegment

    :param file: file path or bytes
    :param start_date: timestamp in microseconds
    :return:
    """
    file = resolve_file(file)
    with wave.open(file, 'rb') as wave_file:
        framerate = wave_file.getframerate()
        nframes = wave_file.getnframes()
        end_date = start_date + nframes * 1e6 / framerate
        audio_data = wave_file.readframes(nframes)

        return AudioSegment(
            start_date=start_date,
            end_date=end_date,
            waveform=audio_data,
            frame_count=nframes,
            status_flags=PaStatusFlags.paNoError,
        )


class WavFileSourceReader(SourceReader):
    buffer_reader_class = AudioBufferReader  # BufferReader specific to AudioSegment

    def __init__(
        self,
        file: Union[str, bytes, Path, BytesIO],
        *,
        frames_per_buffer=1024,
        start_date=0
    ):
        super().__init__()
        self.file = resolve_file(file)
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
