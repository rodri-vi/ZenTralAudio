"""Consultas mínimas a CoreAudio (macOS) sin dependencias externas.

PortAudio solo enumera los dispositivos al inicializarse y no notifica cambios.
Para seguir el dispositivo de salida predeterminado del sistema (p. ej. al
conectar unos AirPods) se lee su identificador directamente de CoreAudio.
En otras plataformas las funciones devuelven ``None``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from functools import lru_cache


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


class _PropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


_SYSTEM_OBJECT = 1
_DEFAULT_OUTPUT_DEVICE = _fourcc("dOut")
_SCOPE_GLOBAL = _fourcc("glob")
_ELEMENT_MAIN = 0


@lru_cache(maxsize=1)
def _core_audio() -> ctypes.CDLL | None:
    if sys.platform != "darwin":
        return None
    path = ctypes.util.find_library("CoreAudio")
    if not path:
        return None
    try:
        lib = ctypes.cdll.LoadLibrary(path)
        lib.AudioObjectGetPropertyData.restype = ctypes.c_int32
        lib.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_PropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        return lib
    except OSError:
        return None


def default_output_device_id() -> int | None:
    """Identificador CoreAudio del dispositivo de salida predeterminado actual."""
    lib = _core_audio()
    if lib is None:
        return None
    address = _PropertyAddress(_DEFAULT_OUTPUT_DEVICE, _SCOPE_GLOBAL, _ELEMENT_MAIN)
    device = ctypes.c_uint32(0)
    size = ctypes.c_uint32(ctypes.sizeof(device))
    status = lib.AudioObjectGetPropertyData(
        _SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(device)
    )
    return int(device.value) if status == 0 and device.value else None
