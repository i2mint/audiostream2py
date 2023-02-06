from stream2py.utility.typing_hints import ComparableType

from audiostream2py import PaStatusFlags, PaCallbackReturnCodes, PyAudioSourceReader


class FillErrorWithZeroesMixin:
    """Mixin to handle all pyaudio's error status flags by filling the error time gap with zeros"""

    _first_error_timestamp = None
    _error_status_flag = None

    def _stream_callback(self, in_data, frame_count, time_info, status_flags):
        """On status flag error code, reate wf bytes of value zero for the entire duration of the
        error to replace garbled data in addtion to the base behavior.

        :param in_data: recorded input data, waveform
        :param frame_count: number of frames, sample count
        :param time_info: dictionary
        :param status_flags: PaStatusFlags
        :return: None, PaCallbackReturnCodes.paContinue
        """
        if not self._first_time_info:
            self._first_time_info = time_info

        if self.frame_index == 0:
            # set start time based on audio time_info difference
            _time_info_diff_s = (
                time_info['input_buffer_adc_time']
                - self._first_time_info['input_buffer_adc_time']
            )
            _timestamp_diff = (
                _time_info_diff_s * self.timestamp_seconds_to_unit_conversion
            )
            self._start_time = self.bt + _timestamp_diff

        _frame_time_s = self.frame_index / self._init_kwargs['rate']
        timestamp = int(
            self._start_time + _frame_time_s * self.timestamp_seconds_to_unit_conversion
        )

        if PaStatusFlags(status_flags) != PaStatusFlags.paNoError:
            # reset frame index and thus self._start_time on any error status
            self.frame_index = 0
            if self._first_error_timestamp is None:
                self._first_error_timestamp = timestamp  # track when errors started
                self._error_status_flag = PaStatusFlags(
                    status_flags
                )  # track what errors occured
            else:
                self._error_status_flag |= PaStatusFlags(
                    status_flags
                )  # use OR to mark any new error status flags
        else:
            if (
                self._first_error_timestamp is not None
            ):  # first ok status after there was an error status
                (zeroed_data, zeroed_count,) = self._fill_time_interval_with_zeroes(
                    self._first_error_timestamp, timestamp
                )
                self.data.append(
                    self.data_to_append(
                        self._first_error_timestamp,
                        zeroed_data,
                        zeroed_count,
                        time_info,
                        self._error_status_flag,
                    )
                )
                self._first_error_timestamp = None
                self._error_status_flag = None

            self.data.append(
                self.data_to_append(
                    timestamp, in_data, frame_count, time_info, status_flags
                )
            )
            self.frame_index += frame_count
        return None, PaCallbackReturnCodes.paContinue

    def _fill_time_interval_with_zeroes(
        self, first_error_status_ts, first_ok_status_ts
    ):
        """Create wf bytes of value zero for the entire duration of the error to replace garbled
        data

        :param first_error_status_ts: Error bt
        :param first_ok_status_ts: Error tt
        :return: in_data, frame_count
        """

        single_zero_sample = (
            b'\x00' * self._init_kwargs['channels'] * self._init_kwargs['width']
        )  # interleaved zeros
        samples_per_time_unit = (
            self._init_kwargs['rate'] / self.timestamp_seconds_to_unit_conversion
        )

        delta_time = first_ok_status_ts - first_error_status_ts
        sample_count = int(delta_time * samples_per_time_unit)
        wf_bytes = single_zero_sample * sample_count
        return wf_bytes, sample_count


class FillErrorWithOnesMixin:
    """Mixin to handle all pyaudio's error status flags by filling the error time gap with zeros"""

    _first_error_timestamp = None
    _error_status_flag = None

    def _stream_callback(self, in_data, frame_count, time_info, status_flags):
        """On status flag error code, reate wf bytes of value zero for the entire duration of the
        error to replace garbled data in addtion to the base behavior.

        :param in_data: recorded input data, waveform
        :param frame_count: number of frames, sample count
        :param time_info: dictionary
        :param status_flags: PaStatusFlags
        :return: None, PaCallbackReturnCodes.paContinue
        """
        if not self._first_time_info:
            self._first_time_info = time_info

        if self.frame_index == 0:
            # set start time based on audio time_info difference
            _time_info_diff_s = (
                time_info['input_buffer_adc_time']
                - self._first_time_info['input_buffer_adc_time']
            )
            _timestamp_diff = (
                _time_info_diff_s * self.timestamp_seconds_to_unit_conversion
            )
            self._start_time = self.bt + _timestamp_diff

        _frame_time_s = self.frame_index / self._init_kwargs['rate']
        timestamp = int(
            self._start_time + _frame_time_s * self.timestamp_seconds_to_unit_conversion
        )

        if PaStatusFlags(status_flags) != PaStatusFlags.paNoError:
            # reset frame index and thus self._start_time on any error status
            self.frame_index = 0
            if self._first_error_timestamp is None:
                self._first_error_timestamp = timestamp  # track when errors started
                self._error_status_flag = PaStatusFlags(
                    status_flags
                )  # track what errors occured
            else:
                self._error_status_flag |= PaStatusFlags(
                    status_flags
                )  # use OR to mark any new error status flags
        else:
            if (
                self._first_error_timestamp is not None
            ):  # first ok status after there was an error status
                zeroed_data, zeroed_count = self._fill_time_interval_with_ones(
                    self._first_error_timestamp, timestamp
                )
                self.data.append(
                    self.data_to_append(
                        self._first_error_timestamp,
                        zeroed_data,
                        zeroed_count,
                        time_info,
                        self._error_status_flag,
                    )
                )
                self._first_error_timestamp = None
                self._error_status_flag = None

            self.data.append(
                self.data_to_append(
                    timestamp, in_data, frame_count, time_info, status_flags
                )
            )
            self.frame_index += frame_count
        return None, PaCallbackReturnCodes.paContinue

    def _fill_time_interval_with_ones(
        self, first_error_status_ts, first_ok_status_ts, fill_value=1
    ):
        """Create wf bytes of value zero for the entire duration of the error to replace garbled
        data

        :param first_error_status_ts: Error bt
        :param first_ok_status_ts: Error tt
        :param fill_value: default is 1
        :return: in_data, frame_count
        """

        single_fill_sample = (
            fill_value.to_bytes(self._init_kwargs['width'], 'little')
            * self._init_kwargs['channels']
        )
        samples_per_time_unit = (
            self._init_kwargs['rate'] / self.timestamp_seconds_to_unit_conversion
        )

        delta_time = first_ok_status_ts - first_error_status_ts
        sample_count = int(delta_time * samples_per_time_unit)
        wf_bytes = single_fill_sample * sample_count
        return wf_bytes, sample_count


class DictDataMixin:
    """Mixin to reduce data to a dict with bt, wf, and status_flag. Removing typically discarded
    information.
    """

    def data_to_append(
        self, timestamp, waveform, frame_count, time_info, status_flags
    ):  # pylint: disable=W0613
        """Simplify data only

        :param timestamp: start time of waveform
        :param waveform: recorded input data
        :param frame_count: discarded data
        :param time_info: discarded data
        :param status_flags: PaStatusFlags error codes
        :return: {'bt': timestamp, 'wf': waveform, 'status_flags': status_flags}
        """
        return {'bt': timestamp, 'wf': waveform, 'status_flags': status_flags}

    def key(self, data) -> ComparableType:
        """
        :param data: {'bt': timestamp, 'wf': waveform, 'status_flags': status_flags}
        :return: data['bt']
        """
        return data['bt']


class RaiseRuntimeErrorOnStatusFlagMixin:
    """Mixin to raise RuntimeError when status_flag is not PaStatusFlags.paNoError (0)"""

    def data_to_append(self, timestamp, waveform, frame_count, time_info, status_flags):
        if PaStatusFlags.paNoError != PaStatusFlags(status_flags):
            raise RuntimeError(PaStatusFlags(status_flags))

        return super().data_to_append(
            timestamp, waveform, frame_count, time_info, status_flags
        )


class PyAudioSourceReaderRaiseOnError(
    RaiseRuntimeErrorOnStatusFlagMixin, PyAudioSourceReader
):
    """PyAudioSourceReader changed to handle errors by raising RuntimeError"""


class PyAudioSourceReaderWithZeroedErrorsAndDictData(
    DictDataMixin, FillErrorWithZeroesMixin, PyAudioSourceReader
):
    """PyAudioSourceReader changed to handle errors and serve data in an easy to read dict."""


class PyAudioSourceReaderWithOnedErrorsAndDictData(
    DictDataMixin, FillErrorWithOnesMixin, PyAudioSourceReader
):
    """PyAudioSourceReader changed to handle errors and serve data in an easy to read dict."""
