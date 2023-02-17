"""Test run PyAudioSourceReader and print info"""
import time

from audiostream2py import PyAudioSourceReader, PaStatusFlags
from audiostream2py.audio import BasePyAudioSourceReader


def _test_run_PyAudioSourceReader(
    readerClass=BasePyAudioSourceReader,
    timestamp_getter=lambda data: data[0],
    in_data_getter=lambda data: data[1],
    status_flags_getter=lambda data: data[4],
    input_device_index=None,
):
    """Run PyAudioSourceReader and print timestamp, byte length, status code, status flag

    :param readerClass: PyAudioSourceReader class or subclass
    :param timestamp_getter: get timestamp from data
    :param in_data_getter: get wf bytes from data
    :param status_flags_getter: get status_flags from data
    """
    from pprint import pprint

    if input_device_index is None:
        print('list_recording_devices')
        pprint(readerClass.list_recording_devices())

    source = readerClass(
        rate=44100,
        width=2,
        channels=1,
        input_device_index=input_device_index,
        frames_per_buffer=4096,
    )
    pprint(source.info)
    try:
        source.open()
        pprint(source.info)
        i = 0
        while i < 600:
            data = source.read()
            if data:
                i += 1
                timestamp = timestamp_getter(data)
                in_data = in_data_getter(data)
                status_flags = status_flags_getter(data)
                print(
                    timestamp,
                    len(in_data),
                    int(status_flags),
                    PaStatusFlags(status_flags),
                )

    finally:
        source.close()
        pprint(source.info)


def _test_run_PyAudioSourceReader_with_StreamBuffer(
    readerClass=PyAudioSourceReader,
    timestamp_getter=lambda data: data.bt,
    in_data_getter=lambda data: data.waveform,
    status_flags_getter=lambda data: data.status_flags,
    input_device_index=None,
    rate=44100,
    maxlen_seconds=10,
    frames_per_buffer=44100,
):
    """Run PyAudioSourceReader and print timestamp, byte length, status code, status flag

    :param readerClass: PyAudioSourceReader class or subclass
    :param timestamp_getter: get timestamp from data
    :param in_data_getter: get wf bytes from data
    :param status_flags_getter: get status_flags from data
    """
    from pprint import pprint

    if input_device_index is None:
        print('list_recording_devices')
        pprint(readerClass.list_recording_devices())

    source = readerClass(
        rate=rate,
        width=2,
        channels=1,
        input_device_index=input_device_index,
        frames_per_buffer=frames_per_buffer,
    )
    pprint(source.info)
    with source.stream_buffer(
        maxlen=source.audio_buffer_size_seconds_to_maxlen(
            maxlen_seconds, rate, frames_per_buffer
        ),
    ) as sb:
        reader = sb.mk_reader()
        time.sleep(6)
        start = reader.head(peek=True).start_date + 0.5 * 1e5
        stop = reader.tail(peek=True).end_date - 0.5 * 1e5
        print(reader[start:stop])
        print(reader[start : start + 0.1 * 1e5])


if __name__ == '__main__':
    # _test_run_PyAudioSourceReader(input_device_index='NexiGo N930AF FHD Webcam Audio')
    # _test_run_PyAudioSourceReader(PyAudioSourceReaderRaiseOnError)

    # _test_run_PyAudioSourceReader(
    #     PyAudioSourceReader,
    #     timestamp_getter=lambda data: data.bt,
    #     in_data_getter=lambda data: data.waveform,
    #     status_flags_getter=lambda data: data.status_flags,
    # )

    _test_run_PyAudioSourceReader_with_StreamBuffer()
