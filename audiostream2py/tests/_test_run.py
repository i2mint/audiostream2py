"""Test run PyAudioSourceReader and print info"""
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


if __name__ == '__main__':
    # _test_run_PyAudioSourceReader(input_device_index='NexiGo N930AF FHD Webcam Audio')
    # _test_run_PyAudioSourceReader(PyAudioSourceReaderRaiseOnError)

    from audiostream2py import PyAudioSourceReader

    _test_run_PyAudioSourceReader(
        PyAudioSourceReader,
        timestamp_getter=lambda data: data['bt'],
        in_data_getter=lambda data: data['wf'],
        status_flags_getter=lambda data: data['status_flags'],
    )
