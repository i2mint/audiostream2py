"""Test setup: stand in for ``pyaudio`` when it (or PortAudio) is not installed.

``audiostream2py`` imports ``pyaudio`` at import time, and building it needs the PortAudio
system library. The tests here use mocks and need no audio hardware, so on a machine
without ``pyaudio`` a minimal stand-in module is registered instead. When the real
``pyaudio`` is importable it is used untouched.
"""

import sys
import types

try:
    import pyaudio  # noqa: F401
except ImportError:
    _stand_in = types.ModuleType("pyaudio")
    _stand_in.PyAudio = type("PyAudio", (), {})
    # The status-flag and callback-return constants audiostream2py.enum reads.
    for _name, _value in dict(
        paNoError=0,
        paInputUnderflow=1,
        paInputOverflow=2,
        paOutputUnderflow=4,
        paOutputOverflow=8,
        paPrimingOutput=16,
        paContinue=0,
        paComplete=1,
        paAbort=2,
    ).items():
        setattr(_stand_in, _name, _value)
    sys.modules["pyaudio"] = _stand_in
