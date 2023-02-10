"""PyAudioSourceReader Status Flag Enums"""

from enum import IntFlag

import pyaudio


class PaStatusFlags(IntFlag):
    """Enum to check status_flag for error codes.

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

    paNoError = pyaudio.paNoError  # 0
    paInputUnderflow = pyaudio.paInputUnderflow  # 1
    paInputOverflow = pyaudio.paInputOverflow  # 2
    paOutputUnderflow = pyaudio.paOutputUnderflow  # 4
    paOutputOverflow = pyaudio.paOutputOverflow  # 8
    paPrimingOutput = pyaudio.paPrimingOutput  # 16
    hostTimeSync = 32  # Timestamp adjusted to avoid overlap due to time sync with NTP


class PaCallbackReturnCodes(IntFlag):
    """Enum of valid _stream_callback return codes.
    Only used by PyAudioSourceReader._stream_callback"""

    paContinue = pyaudio.paContinue
    paComplete = pyaudio.paComplete
    paAbort = pyaudio.paAbort
