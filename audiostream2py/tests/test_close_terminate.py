"""Closing / terminating a PyAudioSourceReader must retry, then raise -- never suppress.

These tests use fakes (and the repo-root ``conftest.py`` stand-in for ``pyaudio`` when it
is not installed), so they need no audio hardware or system libraries.
"""

import pytest

import pyaudio
from audiostream2py import audio
from audiostream2py.audio import BasePyAudioSourceReader


class Flaky:
    """A callable that fails ``failures`` times (``None``: forever), then succeeds."""

    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def __call__(self):
        self.calls += 1
        if self.failures is None or self.calls <= self.failures:
            raise OSError("device busy")


class FakeStream:
    def __init__(self, stop_failures=0, close_failures=0):
        self.stop_stream = Flaky(stop_failures)
        self.close = Flaky(close_failures)


class FakePyAudio(pyaudio.PyAudio):
    def __init__(self, terminate_failures=0):
        self.terminate = Flaky(terminate_failures)


@pytest.fixture
def reader(monkeypatch):
    """A reader built without touching PortAudio, with instant retries."""
    monkeypatch.setattr(audio.time, "sleep", lambda _s: None)
    monkeypatch.setattr(BasePyAudioSourceReader, "_pyaudio_instance", None)
    obj = object.__new__(BasePyAudioSourceReader)
    obj._fp = None
    return obj


def test_close_retries_a_transient_failure(reader):
    reader._fp = stream = FakeStream(stop_failures=1, close_failures=2)
    BasePyAudioSourceReader._pyaudio_instance = pa = FakePyAudio(terminate_failures=1)
    reader.close()
    assert (stream.stop_stream.calls, stream.close.calls) == (2, 3)
    assert pa.terminate.calls == 2
    assert reader._fp is None and BasePyAudioSourceReader._pyaudio_instance is None


def test_close_raises_when_the_stream_cannot_be_closed_but_still_terminates(reader):
    reader._fp = stream = FakeStream(close_failures=None)
    BasePyAudioSourceReader._pyaudio_instance = pa = FakePyAudio()
    with pytest.raises(OSError, match="device busy"):
        reader.close()
    assert stream.close.calls == BasePyAudioSourceReader.release_attempts
    assert pa.terminate.calls == 1  # PortAudio is released regardless
    assert reader._fp is stream  # kept, so a later close() can retry


def test_close_tolerates_a_stream_that_cannot_be_stopped_but_closes(reader):
    reader._fp = stream = FakeStream(stop_failures=None)
    reader.close()
    assert stream.close.calls == 1 and reader._fp is None


def test_terminate_failure_is_raised_and_instance_kept(reader):
    BasePyAudioSourceReader._pyaudio_instance = pa = FakePyAudio(terminate_failures=None)
    with pytest.raises(OSError, match="device busy"):
        reader.close()
    assert pa.terminate.calls == BasePyAudioSourceReader.release_attempts
    assert BasePyAudioSourceReader._pyaudio_instance is pa


def test_close_is_idempotent(reader):
    reader.close()
    reader.close()  # nothing open, nothing to terminate, no error
