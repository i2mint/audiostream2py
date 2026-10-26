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
    "PyAudioSourceReader",
    "PaStatusFlags",
    "PaCallbackReturnCodes",
    "get_input_device_index",
    "AudioSegment",
]

from collections import deque
from contextlib import suppress, contextmanager
import logging
import math
import time
from typing import List, Union
from collections.abc import Generator, Callable
import re

import pyaudio
from stream2py import SourceReader
from stream2py.utility.typing_hints import ComparableType

from audiostream2py.reader import AudioBufferReader
from audiostream2py.data import AudioSegment
from audiostream2py.enum import PaCallbackReturnCodes, PaStatusFlags


def list_recording_device_index_names():
    """List (index, name) of available recording devices. Returns a list of the form:
    [(2, 'MacBook Pro Microphone'), (4, 'BluetoothHeadset'), (6, 'ZoomAudioDevice')]
    """
    return sorted(
        (d["index"], d["name"])
        for d in PyAudioSourceReader.list_device_info()
        if d["maxInputChannels"] > 0
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
        raise RuntimeError("No device infos matched your filter")
    if assert_unique:
        if next(it, None) is not None:
            raise RuntimeError("More than one device info dict matched your filter")
    return info


def is_recording_device(device_info: dict):
    """Says if a device (info) is a recording device"""
    return device_info.get("maxInputChannels", 0) > 0


def get_recording_device_info_by_name(name_pattern, assert_unique=True):
    """Find a recording device by matching name pattern and return info dict"""
    name_pattern = re.compile(name_pattern)

    def filt(device_info):
        return name_pattern.search(device_info["name"]) and is_recording_device(
            device_info
        )

    return match_device_info(filt, assert_unique)


def get_input_device_index(input_device=None, input_device_index=None, verbose=True):
    info = None
    # You can't specify both input_device_index and input_device!
    if input_device is not None and input_device_index is not None:
        raise RuntimeError(
            f"You can't have specify both input_device and input_device_index!: "
            f"{input_device=} and {input_device_index=}"
        )
    # If no information on the device is given, ask pyaudio to find a default
    if input_device is None and input_device_index is None:
        info = PyAudioSourceReader.get_default_input_device_info()
        if verbose:
            import json

            print(
                f"Will use {info['name']} (index={info['index']}) as an input device",
                end="\r",
            )
            print(f"It's info:\n{json.dumps(info, indent=2)}", end="\r")
    else:
        input_device = input_device or input_device_index
        if isinstance(input_device, int):
            info = {"index": input_device}
        elif isinstance(input_device, Callable):
            info = match_device_info(filt=input_device)
        else:  # should be string or re.compile instance
            info = get_recording_device_info_by_name(name_pattern=input_device)
    return info["index"]


_log = logging.getLogger(__name__)

#: How many times releasing a PortAudio resource (stop/close a stream, terminate PyAudio)
#: is attempted before the failure is raised.
RELEASE_ATTEMPTS = 3

#: Seconds to wait between two attempts at releasing a PortAudio resource.
RELEASE_RETRY_DELAY_S = 0.05


def _release(action: Callable, *, attempts: int, retry_delay_s: float):
    """Call ``action()``, retrying on failure; raise the last error when attempts run out.

    Releasing PortAudio resources must not fail silently: a stream or instance that was
    not released leaves the device busy, and the next ``open`` then fails far from the cause.

    >>> calls = []
    >>> def flaky():
    ...     calls.append(1)
    ...     if len(calls) < 3:
    ...         raise OSError('busy')
    >>> _release(flaky, attempts=3, retry_delay_s=0)
    >>> len(calls)
    3
    >>> _release(lambda: 1 / 0, attempts=2, retry_delay_s=0)
    Traceback (most recent call last):
      ...
    ZeroDivisionError: division by zero
    """
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:
            if attempt == attempts:
                raise
            _log.warning(
                "%s failed (attempt %d/%d): %r",
                getattr(action, "__qualname__", action),
                attempt,
                attempts,
                exc,
            )
            time.sleep(retry_delay_s)


class BasePyAudioSourceReader(SourceReader):
    _pyaudio_instance = None  # set per reader by ``open``; see ``_init_pyaudio``
    #: Attempts at releasing the stream / PyAudio instance on ``close`` before raising.
    release_attempts = RELEASE_ATTEMPTS
    #: Seconds between two release attempts.
    release_retry_delay_s = RELEASE_RETRY_DELAY_S

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
        ts_refresh_period=1000,
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
        :param ts_refresh_period: period, in seconds, with which a buffer will be timestamped with
        the host-system time, instead of using the frame rate and frame count values. This limits
        the drift between buffer timestamps and host-system time due to frame rate innaccuracies.
        Crucial if we need to synchronize data from different sources.
        """
        super().__init__()
        self._init_kwargs = {
            k: v for k, v in locals().items() if k not in ("self", "__class__")
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
            "rate": rate,
            "channels": channels,
            "format": input_format,
            "input": True,
            "output": False,
            "input_device_index": input_device_index,
            "output_device_index": None,
            "frames_per_buffer": frames_per_buffer,
            "start": True,
            "input_host_api_specific_stream_info": None,
            "output_host_api_specific_stream_info": None,
            "stream_callback": self._stream_callback,
        }

        self._fp = None
        self._open_time = None
        self.data = deque()
        self.buffer_start = None
        self.buffer_end = None
        self.last_ts_refresh = None
        self.ts_refresh_period = (
            ts_refresh_period * self.timestamp_seconds_to_unit_conversion
        )

    def _init_vars(self):
        if self._fp:
            self.close()
        self._fp = None
        if self.data:
            self.data.clear()

        self.buffer_start = None
        self.buffer_end = None
        self.last_ts_refresh = None

    def __repr__(self):
        def quote_strings(x):
            return f"'{x}'" if isinstance(x, str) else x

        args_string = ", ".join(
            f"{k}={quote_strings(v)}" for k, v in self._init_kwargs.items()
        )
        return f"{type(self).__name__}({args_string})"

    @classmethod
    def get_default_input_device_info(cls):
        with cls._pyaudio() as p:
            return p.get_default_input_device_info()

    @property
    def sleep_time_on_read_none_s(self) -> float:
        """One third of the expected rate frames_per_buffer will be filled"""
        seconds_per_read = (
            self._init_kwargs["frames_per_buffer"] / self._init_kwargs["rate"]
        )
        one_third_read_rate = seconds_per_read / 3
        return one_third_read_rate

    @property
    def sr(self) -> int | float:
        return self._init_kwargs["rate"]

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
        _info = {"start_date": self._open_time}
        _info.update(**self._init_kwargs)
        with suppress(Exception):
            _info.update(
                device_info=self._pyaudio_instance.get_device_info_by_index(
                    self._init_kwargs["input_device_index"]
                )
            )
        return _info

    def key(self, data) -> ComparableType:
        """
        :param data: (start_date, end_date, waveform, frame_count, time_info, status_flags)
        :return: timestamp
        """
        return data[1]

    def data_to_append(
        self, start_date, end_date, waveform, frame_count, time_info, status_flags
    ):
        """Overloaded data_to_append method change the shape of read outputs of PyAudioSourceReader.
        The key method must also be overloaded to return timestamp from the new shape.

        :param start_date: bt timestamp
        :param end_date: tt timestamp
        :param waveform: recorded input frames in bytes
        :param frame_count: number of frames, sample count or waveform
        :param time_info: http://portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html
        :param status_flags: PaStatusFlags
        :return: (start_date, end_data, waveform, frame_count, time_info, status_flags)
        """
        return start_date, end_date, waveform, frame_count, time_info, status_flags

    def open(self):
        """Will first close if already open and clear any data before starting the audio stream"""
        self._init_vars()
        self._init_pyaudio()
        self._open_time = self.get_timestamp()
        try:
            self._fp = self._pyaudio_instance.open(**self._pyaudio_open_params)
        except BaseException:  # ``__exit__`` never runs when ``open`` fails
            self._terminate_pyaudio()
            raise

    def close(self):
        """Stop and release the audio stream, then terminate PyAudio.

        Each step is retried (``release_attempts``) and a failure that persists is raised,
        never swallowed, because an unreleased PortAudio resource breaks the next ``open``.
        PyAudio is terminated even when the stream could not be closed; the stream is kept
        in ``_fp`` in that case, so a later ``close`` retries it. Closing twice is a no-op.
        """
        retry = dict(
            attempts=self.release_attempts, retry_delay_s=self.release_retry_delay_s
        )
        try:
            if self._fp is not None:
                try:
                    _release(self._fp.stop_stream, **retry)
                except Exception as exc:
                    # Closing a stream aborts it, so only a failed close is fatal.
                    _log.warning("stop_stream failed, closing anyway: %r", exc)
                _release(self._fp.close, **retry)
                self._fp = None
        finally:
            self._terminate_pyaudio()

    def read(self):
        """Returns one data item as structured by PyAudioSourceReader.data_to_append

        :return: (timestamp, waveform, frame_count, time_info, status_flags)
        """
        if len(self.data):
            return self.data.popleft()
        return None

    def _stream_callback(self, in_data, frame_count, time_info, status_flags):
        """See _stream_callback in https://people.csail.mit.edu/hubert/pyaudio/docs/#class-stream

        :param in_data: recorded input data, waveform
        :param frame_count: number of frames, sample count
        :param time_info: dictionary
        :param status_flags: PaStatusFlags
        :return: None, PaCallbackReturnCodes.paContinue
        """
        _additional_status_flags = self._set_buffer_start_end(frame_count)
        status_flags = PaStatusFlags(status_flags) | _additional_status_flags
        self.data.append(
            self.data_to_append(
                self.buffer_start,
                self.buffer_end,
                in_data,
                frame_count,
                time_info,
                status_flags,
            )
        )
        return None, PaCallbackReturnCodes.paContinue

    def _set_buffer_start_end(self, frame_count):
        """Set buffer timestamps when data is sent to callback.
        :param frame_count: number of frames contained in buffer
        :return: status flag
        """
        status_flag = PaStatusFlags.paNoError

        # Buffer duration (time unit)
        buffer_dur = frame_count * self.timestamp_seconds_to_unit_conversion / self.sr
        # Host-system time
        ts = self.get_timestamp()

        # Timestamping buffer end...
        if (
            self.buffer_end is None
            or self.last_ts_refresh is None
            or ts - self.last_ts_refresh > self.ts_refresh_period
        ):
            # with host-system time, because a refresh is needed or there is no info about a
            # previous buffer
            buffer_end_timestamp = ts
            self.last_ts_refresh = ts
        else:
            # based on timestamp of previous buffer otherwise
            buffer_end_timestamp = self.buffer_end + buffer_dur

        # Timestamping buffer start...
        if self.buffer_end is None:
            # by inference if there is no info about a previous buffer
            self.buffer_start = buffer_end_timestamp - buffer_dur
        else:
            # based on timestamp of previous buffer otherwise.
            # But first, checking if new buffer timestamp is posterior enough to previous buffer
            # timestamp. If not, we readjust and set last_ts_refresh to None to force timestamp
            # refresh on the next buffer.
            _adjustment = (
                self.buffer_end
                + frame_count / self.timestamp_seconds_to_unit_conversion
            )
            if buffer_end_timestamp < _adjustment:
                buffer_end_timestamp = _adjustment
                self.last_ts_refresh = None
                status_flag = PaStatusFlags.hostTimeSync

            self.buffer_start = self.buffer_end

        # Updating buffer end timestamp
        self.buffer_end = buffer_end_timestamp
        return status_flag

    def _init_pyaudio(self) -> pyaudio.PyAudio:
        """Give this reader its own PyAudio instance (closing a previous one of its own).

        The instance is per reader: ``PyAudio.terminate`` closes every stream the
        instance owns, so sharing one across readers (or with device queries) would
        close an open stream under its reader.
        """
        self._terminate_pyaudio()
        self._pyaudio_instance = pyaudio.PyAudio()
        return self._pyaudio_instance

    @classmethod
    def _terminate(cls, instance):
        """Terminate a PyAudio instance with the class's release policy (single place)."""
        _release(
            instance.terminate,
            attempts=cls.release_attempts,
            retry_delay_s=cls.release_retry_delay_s,
        )

    def _terminate_pyaudio(self):
        """Terminate this reader's PyAudio instance (retrying); raise if it cannot be.

        The instance reference is only dropped once ``terminate`` has succeeded.
        """
        instance = self._pyaudio_instance
        pyaudio_type = pyaudio.PyAudio  # may be a mock in a caller's tests
        if isinstance(pyaudio_type, type) and isinstance(instance, pyaudio_type):
            self._terminate(instance)
            self._pyaudio_instance = None

    @classmethod
    @contextmanager
    def _pyaudio(cls) -> Generator[pyaudio.PyAudio, None, None]:
        """A short-lived PyAudio instance for device queries.

        Independent of every reader's instance, so querying devices (or constructing
        another reader) never closes a stream that is open elsewhere.
        """
        instance = pyaudio.PyAudio()
        try:
            yield instance
        finally:
            cls._terminate(instance)

    @classmethod
    def list_device_info(cls) -> list[dict]:
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
            d["name"] for d in cls.list_device_info() if d["maxInputChannels"] > 0
        )

    @classmethod
    def info_of_input_device_index(cls, input_device_index):
        return next(
            (
                dev
                for dev in cls.list_device_info()
                if dev["index"] == input_device_index
            ),
            None,
        )


class FillErrorWithOnesMixin:
    """Mixin to handle all pyaudio's error status flags by filling the error time gap with zeros"""

    FILL_VALUE = 1
    _first_error_timestamp = None
    _error_status_flag = None

    def _stream_callback(self, in_data, frame_count, time_info, status_flags):
        """On paInputOverflow status flag error code, create wf bytes of value one for the entire
        duration of the error to replace garbled data in addition to the base behavior.

        :param in_data: recorded input data, waveform
        :param frame_count: number of frames, sample count
        :param time_info: dictionary
        :param status_flags: PaStatusFlags
        :return: None, PaCallbackReturnCodes.paContinue
        """

        if PaStatusFlags.paInputOverflow in PaStatusFlags(status_flags):
            self.buffer_start = None
            if self._first_error_timestamp is None:
                # track when errors started
                self._first_error_timestamp = self.buffer_end
                # recalculate next timestamps in _set_buffer_start_end
                self.buffer_end = None
                # track what errors occurred
                self._error_status_flag = PaStatusFlags(status_flags)
            else:
                # use OR to mark any new error status flags
                self._error_status_flag |= PaStatusFlags(status_flags)
        else:
            _additional_status_flags = self._set_buffer_start_end(frame_count)
            status_flags = PaStatusFlags(status_flags) | _additional_status_flags

            if self._first_error_timestamp is not None:
                # first ok status after there was an error status
                fill_data, fill_count = self._fill_time_interval_with_ones(
                    self._first_error_timestamp, self.buffer_start
                )
                self.data.append(
                    self.data_to_append(
                        self._first_error_timestamp,
                        self.buffer_start,
                        fill_data,
                        fill_count,
                        {},
                        self._error_status_flag,
                    )
                )
                self._first_error_timestamp = None
                self._error_status_flag = None

            self.data.append(
                self.data_to_append(
                    self.buffer_start,
                    self.buffer_end,
                    in_data,
                    frame_count,
                    time_info,
                    status_flags,
                )
            )
        return None, PaCallbackReturnCodes.paContinue

    def _fill_time_interval_with_ones(self, first_error_status_ts, first_ok_status_ts):
        """Create wf bytes of value zero for the entire duration of the error to replace garbled
        data

        :param first_error_status_ts: Error bt
        :param first_ok_status_ts: Error tt
        :return: in_data, frame_count
        """

        single_fill_sample = (
            self.FILL_VALUE.to_bytes(self._init_kwargs["width"], "little")
            * self._init_kwargs["channels"]
        )
        samples_per_time_unit = self.sr / self.timestamp_seconds_to_unit_conversion

        delta_time = first_ok_status_ts - first_error_status_ts
        sample_count = int(delta_time * samples_per_time_unit)
        wf_bytes = single_fill_sample * sample_count
        return wf_bytes, sample_count


class DictDataMixin:
    """Mixin to reduce data to a dict with bt, tt, wf, and status_flag. Removing typically discarded
    information.
    """

    def data_to_append(
        self, start_date, end_data, waveform, frame_count, time_info, status_flags
    ):  # pylint: disable=W0613
        """Simplify data only

        :param timestamp: start time of waveform
        :param waveform: recorded input data
        :param frame_count: discarded
        :param time_info: discarded
        :param status_flags: PaStatusFlags error codes
        :return: {'bt': timestamp, 'wf': waveform, 'status_flags': status_flags}
        """
        return {
            "bt": start_date,
            "tt": end_data,
            "wf": waveform,
            "status_flags": status_flags,
        }

    def key(self, data) -> ComparableType:
        """
        :param data: {'bt': timestamp, 'wf': waveform, 'status_flags': status_flags}
        :return: data['tt']
        """
        return data["tt"]


class AudioSegmentMixin:
    """Mixin to put data into AudioSegment"""

    buffer_reader_class = AudioBufferReader  # BufferReader specific to AudioSegment

    def data_to_append(
        self,
        start_date,
        end_data,
        waveform,
        frame_count,
        time_info,  # pylint: disable=W0613
        status_flags,
    ) -> AudioSegment:
        """Puts data into AudioSegment

        :param start_date: starting timestamp
        :param end_data: ending timestamp
        :param waveform: recorded input data
        :param frame_count: frame count of current buffer
        :param time_info: discarded
        :param status_flags: PaStatusFlags error codes
        :return: {'bt': timestamp, 'wf': waveform, 'status_flags': status_flags}
        """
        return AudioSegment(start_date, end_data, waveform, frame_count, status_flags)

    def key(self, data: AudioSegment) -> ComparableType:
        """AudioSegment is a ComparableType

        :param data: {'bt': timestamp, 'wf': waveform, 'status_flags': status_flags}
        :return: AudioSegment
        """
        return data.start_date


class PyAudioSourceReader(
    AudioSegmentMixin, FillErrorWithOnesMixin, BasePyAudioSourceReader
):
    """PyAudioSourceReader changed to handle errors and serve data in an easy-to-read dataclass."""
