"""
PyAudio Source
--------------
This source requires PyAudio.
`Find PyAudio installation instructions here <https://people.csail.mit.edu/hubert/pyaudio/>`_.
::
    pip install PyAudio

.. autoclass:: audiostream2py.audio.PyAudioSourceReader()
    :members:
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: audiostream2py.audio.PaStatusFlags
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


PyAudio Mixins
--------------
Example of how to use mixins to manipulate behavior to suit your needs.

.. autoclass:: audiostream2py.audio.FillErrorWithZeroesMixin
    :members:

.. autoclass:: audiostream2py.audio.DictDataMixin
    :members:

.. autoclass:: audiostream2py.audio.RaiseRuntimeErrorOnStatusFlagMixin
    :members:

Example PyAudio Mixin Usage
---------------------------
Mixins must be subclassed in order of precedence from left to right.
Meaning the base class at the right and mixins with overloading methods to the left.

.. autoclass:: audiostream2py.audio.PyAudioSourceReaderRaiseOnError
    :members:
    :show-inheritance:

.. autoclass:: audiostream2py.audio.PyAudioSourceReaderRaiseOnError
    :members:
    :show-inheritance:

.. autoclass:: audiostream2py.audio.PyAudioSourceReaderWithZeroedErrorsAndDictData
    :members:
    :show-inheritance:
"""

__all__ = [
    'PyAudioSourceReader',
    'PaStatusFlags',
    'PaCallbackReturnCodes',
    'get_input_device_index',
]

from collections import deque
from contextlib import suppress, contextmanager
from enum import IntFlag
import math
import operator
from typing import Generator, List, Callable, Union
import re

import pyaudio

from stream2py import SourceReader
from stream2py.utility.typing_hints import ComparableType

_ITEMGETTER_0 = operator.itemgetter(0)


def list_recording_device_index_names():
    """List (index, name) of available recording devices. Returns a list of the form:
    [(2, 'MacBook Pro Microphone'), (4, 'BluetoothHeadset'), (6, 'ZoomAudioDevice')]
    """
    return sorted(
        (d['index'], d['name'])
        for d in PyAudioSourceReader.list_device_info()
        if d['maxInputChannels'] > 0
    )


def _list_device_info():
    return PyAudioSourceReader.list_device_info()


def _match_device_info(filt: Callable):
    return filter(filt, _list_device_info())


def match_device_info(filt: Callable, assert_unique=True):
    """Find a device ('s info) through a function of its info (dict)"""
    it = _match_device_info(filt)
    info = next(it, None)
    if info is None:
        raise RuntimeError('No device infos matched your filter')
    if assert_unique:
        if next(it, None) is not None:
            raise RuntimeError('More than one device info dict matched your filter')
    return info


def is_recording_device(device_info: dict):
    """Says if a device (info) is a recording device"""
    return device_info.get('maxInputChannels', 0) > 0


def get_recording_device_info_by_name(name_pattern, assert_unique=True):
    """Find a recording device by matching name pattern and return info dict"""
    name_pattern = re.compile(name_pattern)
    filt = lambda device_info: name_pattern.search(
        device_info['name']
    ) and is_recording_device(device_info)
    return match_device_info(filt, assert_unique)


def get_input_device_index(input_device=None, input_device_index=None, verbose=True):
    info = None
    # You can't specify both input_device_index and input_device!
    if input_device is not None and input_device_index is not None:
        raise RuntimeError(
            f"You can't have specify both input_device and input_device_index!: "
            f'{input_device=} and {input_device_index=}'
        )
    # If no information on the device is given, ask pyaudio to find a default
    elif input_device is None and input_device_index is None:
        info = PyAudioSourceReader.get_default_input_device_info()
        if verbose:
            import json

            print(
                f"Will use {info['name']} (index={info['index']}) as an input device",
                end='\r',
            )
            print(f"It's info:\n{json.dumps(info, indent=2)}", end='\r')
    else:
        input_device = input_device or input_device_index
        if isinstance(input_device, int):
            info = {'index': input_device}
        elif isinstance(input_device, Callable):
            info = match_device_info(filt=input_device)
        else:  # should be string or re.compile instance
            info = get_recording_device_info_by_name(name_pattern=input_device)
    return info['index']


# Note: Keeping around for (approx) back-compatibility
def find_a_default_input_device_index(verbose=True):
    from warnings import warn

    warn('Deprecating find_a_default_input_device_index. Use get_input_device_index')
    return get_input_device_index(verbose=verbose)


class PaStatusFlags(IntFlag):
    """Enum to check status_flag for error codes

    >>> from audiostream2py.audio import PaStatusFlags
    >>> PaStatusFlags(0)
    <PaStatusFlags.paNoError: 0>
    >>> PaStatusFlags(2)
    <PaStatusFlags.paInputOverflow: 2>
    >>> PaStatusFlags(3)
    <PaStatusFlags.paInputOverflow|paInputUnderflow: 3>
    >>> PaStatusFlags.paInputOverflow in PaStatusFlags(3)  # Check if contains certain error
    True
    >>> PaStatusFlags.paNoError == PaStatusFlags(3)  # Check for no error
    False
    """

    paNoError = pyaudio.paNoError
    paInputUnderflow = pyaudio.paInputUnderflow
    paInputOverflow = pyaudio.paInputOverflow
    paOutputUnderflow = pyaudio.paOutputUnderflow
    paOutputOverflow = pyaudio.paOutputOverflow
    paPrimingOutput = pyaudio.paPrimingOutput


class PaCallbackReturnCodes(IntFlag):
    """Enum of valid _stream_callback return codes.
    Only used by PyAudioSourceReader._stream_callback"""

    paContinue = pyaudio.paContinue
    paComplete = pyaudio.paComplete
    paAbort = pyaudio.paAbort


class PyAudioSourceReader(SourceReader):
    _pyaudio_instance = None

    def __init__(
        self,
        *,
        rate=44100,
        width=2,
        unsigned=True,
        channels=1,
        input_device_index=None,
        frames_per_buffer=1024,
        input_device=None,
        verbose=True,
    ):
        """

        :param rate: Specifies the desired sample rate (in Hz)
        :param width: Sample width in bytes (1, 2, 3, or 4)
        :param unsigned: For 1 byte width, specifies signed or unsigned format.
            Ignored if byte width is not 1.
        :param channels: The desired number of input channels.
            Ignored if input_device is not specified (or None).
        :param input_device: Specification of what input device to use.
        :param input_device_index: Legacy specification of input Device to use.
            Has to be the integer index, unlike input_device which can be a name too.
            Unspecified (or None) uses default device.
        :param frames_per_buffer: Specifies the number of frames per buffer.
        :param verbose: Permission to print stuff when we feel like it?

        """

        self._init_kwargs = {
            k: v for k, v in locals().items() if k not in ('self', '__class__')
        }
        input_device_index = get_input_device_index(
            input_device=input_device,
            input_device_index=input_device_index,
            verbose=verbose,
        )
        with self._pyaudio() as pa:
            input_format = pa.get_format_from_width(width, unsigned)
            pa.is_format_supported(
                rate=rate,
                input_device=input_device_index,
                input_channels=channels,
                input_format=input_format,
                output_device=None,
                output_channels=None,
                output_format=None,
            )

        self._pyaudio_open_params = {
            'rate': rate,
            'channels': channels,
            'format': input_format,
            'input': True,
            'output': False,
            'input_device_index': input_device_index,
            'output_device_index': None,
            'frames_per_buffer': frames_per_buffer,
            'start': True,
            'input_host_api_specific_stream_info': None,
            'output_host_api_specific_stream_info': None,
            'stream_callback': self._stream_callback,
        }

        self._fp = None
        self._open_time = None
        self.data = deque()
        self._init_vars()

    def _init_vars(self):
        if self._fp:
            self.close()
        self._fp = None
        if self.data:
            self.data.clear()

        self.buffer_start = None
        self.buffer_end = None

    def __repr__(self):
        quote_strings = lambda x: f"'{x}'" if isinstance(x, str) else x
        args_string = ', '.join(
            f'{k}={quote_strings(v)}' for k, v in self._init_kwargs.items()
        )
        return f'{type(self).__name__}({args_string})'

    @classmethod
    def get_default_input_device_info(cls):
        with cls._pyaudio() as p:
            return p.get_default_input_device_info()

    @property
    def sleep_time_on_read_none_s(self) -> float:
        """One third of the expected rate frames_per_buffer will be filled"""
        seconds_per_read = (
            self._init_kwargs['frames_per_buffer'] / self._init_kwargs['rate']
        )
        one_third_read_rate = seconds_per_read / 3
        return one_third_read_rate

    @property
    def sr(self) -> Union[int, float]:
        return self._init_kwargs['rate']

    @property
    def info(self) -> dict:
        """
        Provides a dict with init kwargs, bt, and device info

        >>> from audiostream2py.audio import PyAudioSourceReader
        >>> from pprint import pprint
        >>> source = PyAudioSourceReader(
        ... rate=44100, width=2, channels=1,
        ... input_device_index=7, frames_per_buffer=4096)  # doctest: +SKIP
        >>> source.open()  # doctest: +SKIP
        >>> pprint(source.info) # doctest: +SKIP
        {'bt': 1582851038965183,
         'channels': 1,
         'device_info': {'defaultHighInputLatency': 0.021333333333333333,
                         'defaultHighOutputLatency': 0.021333333333333333,
                         'defaultLowInputLatency': 0.021333333333333333,
                         'defaultLowOutputLatency': 0.021333333333333333,
                         'defaultSampleRate': 48000.0,
                         'hostApi': 0,
                         'index': 7,
                         'maxInputChannels': 128,
                         'maxOutputChannels': 128,
                         'name': 'sysdefault',
                         'structVersion': 2},
         'frames_per_buffer': 4096,
         'input_device_index': 7,
         'rate': 44100,
         'unsigned': True,
         'width': 2}
        >>> source.close()  # doctest: +SKIP

        :return: dict
        """
        _info = {'start_date': self._open_time}
        _info.update(**self._init_kwargs)
        with suppress(Exception):
            _info.update(
                device_info=self._pyaudio_instance.get_device_info_by_index(
                    self._init_kwargs['input_device_index']
                )
            )
        return _info

    def key(self, data) -> ComparableType:
        """
        :param data: (timestamp, waveform, frame_count, time_info, status_flags)
        :return: timestamp
        """
        return _ITEMGETTER_0(data)

    def data_to_append(self, timestamp, waveform, frame_count, time_info, status_flags):
        """Can to be overloaded to change the shape of read outputs by altering the return value.
        The key function must also be overloaded to return timestamp from the new shape.

        :param timestamp: The time data is received. The end time of the buffer.
        :param waveform: recorded input data
        :param frame_count: number of frames, sample count
        :param time_info: dict,
            see http://portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html
        :param status_flags: PaStatusFlags
        :return: (timestamp, waveform, frame_count, time_info, status_flags)
        """
        return timestamp, waveform, frame_count, time_info, status_flags

    def open(self):
        """Will first close if already open and clear any data before starting the audio stream"""
        self._init_vars()
        self._init_pyaudio()
        self._open_time = self.get_timestamp()
        self._fp = self._pyaudio_instance.open(**self._pyaudio_open_params)

    def close(self):
        """Stop audio stream"""
        with suppress(Exception):
            self._fp.stop_stream()
        with suppress(Exception):
            self._fp.close()
        self._terminate_pyaudio()

    def read(self):
        """Returns one data item as structured by PyAudioSourceReader.data_to_append

        :return: (timestamp, waveform, frame_count, time_info, status_flags)
        """
        if len(self.data):
            return self.data.popleft()
        return None

    def _stream_callback(self, in_data, frame_count, time_info, status_flags):
        """Calculates timestamp based on open() bt and frames read.
        If there is an error conveyed by status_flags, the frame count is reset to 0 and starting
        timestamp is shifted from open() bt by time_info to approximate actual time in case of
        sample loss.
        See _stream_callback in https://people.csail.mit.edu/hubert/pyaudio/docs/#class-stream

        :param in_data: recorded input data, waveform
        :param frame_count: number of frames, sample count
        :param time_info: dictionary
        :param status_flags: PaStatusFlags
        :return: None, PaCallbackReturnCodes.paContinue
        """
        buffer_end_timestamp = self.get_timestamp()
        if self.buffer_end is None:
            t_len = frame_count * self.timestamp_seconds_to_unit_conversion / self.sr
            self.buffer_start = buffer_end_timestamp - t_len
        else:
            self.buffer_start = self.buffer_end
        self.buffer_end = buffer_end_timestamp

        self.data.append(
            self.data_to_append(
                self.buffer_end, in_data, frame_count, time_info, status_flags
            )
        )
        return None, PaCallbackReturnCodes.paContinue

    @classmethod
    def _init_pyaudio(cls) -> pyaudio.PyAudio:
        cls._terminate_pyaudio()
        cls._pyaudio_instance = pyaudio.PyAudio()
        return cls._pyaudio_instance

    @classmethod
    def _terminate_pyaudio(cls):
        with suppress(Exception):
            if cls._pyaudio_instance is not None and isinstance(
                cls._pyaudio_instance, (pyaudio.PyAudio,)
            ):
                cls._pyaudio_instance.terminate()
                cls._pyaudio_instance = None

    @classmethod
    @contextmanager
    def _pyaudio(cls) -> Generator[pyaudio.PyAudio, None, None]:
        try:
            yield cls._init_pyaudio()
        finally:
            cls._terminate_pyaudio()

    @classmethod
    def list_device_info(cls) -> List[dict]:
        """
        .. todo::
            * filter for only devices with input channels

        :return: list
        """
        with cls._pyaudio() as pa:
            return [
                pa.get_device_info_by_index(idx) for idx in range(pa.get_device_count())
            ]

    @staticmethod
    def audio_buffer_size_seconds_to_maxlen(
        buffer_size_seconds, rate, frames_per_buffer
    ) -> int:
        """Calculate maxlen for StreamBuffer to keep a minimum of buffer_size_seconds of data on
        buffer

        :param buffer_size_seconds: desired length of StreamBuffer in seconds
        :param rate: sample rate
        :param frames_per_buffer: number of frames per buffer
        :return: maxlen for StreamBuffer
        """
        seconds_per_read = frames_per_buffer / rate
        return math.ceil(buffer_size_seconds / seconds_per_read)

    @classmethod
    def list_recording_devices(cls):
        """List names of available recording devices"""
        return sorted(
            d['name'] for d in cls.list_device_info() if d['maxInputChannels'] > 0
        )

    @classmethod
    def info_of_input_device_index(cls, input_device_index):
        return next(
            (
                dev
                for dev in cls.list_device_info()
                if dev['index'] == input_device_index
            ),
            None,
        )
