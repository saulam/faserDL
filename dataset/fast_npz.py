"""
Description: a fast drop-in for np.load(path, allow_pickle=True) on the event
.npz files.

np.load parses the zip directory with the pure-Python zipfile module and each
.npy header with ast.literal_eval: ~0.17 ms per array, ~3.4 ms per event for
the 19 arrays of stage 2, even from memory.  load_npz reads the file once,
parses the zip directory and the plain .npy headers directly, and decodes an
array only when it is accessed (as np.load does).  The arrays are identical:
the same dtype, shape and bytes, writeable, as np.load's views.  Each member's
CRC-32 is checked, as zipfile does.  Anything else (zip64, encryption, data
descriptors, other compressions, object arrays, other .npy headers) falls back
to numpy for that file or member.  FASER_FAST_NPZ=0 uses np.load throughout.
"""

import ast
import io
import math
import os
import re
import struct
import zlib
from collections.abc import Mapping

import numpy as np

ENABLED = os.environ.get("FASER_FAST_NPZ", "1") != "0"

_EOCD = struct.Struct("<IHHHHIIH")                  # end of central directory record
_CENTRAL = struct.Struct("<IHHHHHHIIIHHHHHII")      # central directory file header
_LOCAL = struct.Struct("<IHHHHHIIIHH")              # local file header
_NPY_HEADER = re.compile(
    rb"\{'descr': (?P<descr>'[^']*'|\[[^\]]*\]), 'fortran_order': (?P<fortran>True|False), "
    rb"'shape': \((?P<shape>[0-9, ]*)\), \} *\n")
_DTYPES = {}


class _Unsupported(Exception):
    pass


def _dtype(descr):
    dtype = _DTYPES.get(descr)
    if dtype is None:
        dtype = _DTYPES[descr] = np.lib.format.descr_to_dtype(ast.literal_eval(descr.decode("latin1")))
    return dtype


def _read_npy(raw):
    """A .npy file (writeable buffer) as np.lib.format.read_array would return it."""
    if raw[:6] != b"\x93NUMPY" or raw[6] != 1:
        return None
    length = int.from_bytes(raw[8:10], "little")
    match = _NPY_HEADER.fullmatch(bytes(raw[10:10 + length]))
    if match is None:
        return None
    dtype = _dtype(match["descr"])
    if dtype.hasobject:
        return None
    shape = tuple(int(size) for size in match["shape"].split(b",") if size.strip())
    array = np.frombuffer(raw, dtype=dtype, count=math.prod(shape), offset=10 + length)
    if match["fortran"] == b"True":
        return array.reshape(shape[::-1]).transpose()
    return array.reshape(shape)


class FastNpz(Mapping):
    """The arrays of one .npz file, decoded on access (like np.lib.npyio.NpzFile)."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as handle:
            self._buffer = bytearray(handle.read())
        buffer = self._buffer
        end = buffer.rfind(b"PK\x05\x06", max(0, len(buffer) - 65557))
        if end < 0:
            raise _Unsupported("no end of central directory")
        _, disk, _, _, count, _, offset, _ = _EOCD.unpack_from(buffer, end)
        if disk != 0 or count == 0xFFFF or offset == 0xFFFFFFFF:
            raise _Unsupported("multi-disk or zip64")
        self._members = {}
        for _ in range(count):
            (signature, _, _, flags, method, _, _, crc, compressed, size, name_length, extra_length,
             comment_length, _, _, _, local) = _CENTRAL.unpack_from(buffer, offset)
            if signature != 0x02014B50 or flags & 0x1 or method not in (0, 8) or compressed == 0xFFFFFFFF:
                raise _Unsupported("unsupported member")
            name = bytes(buffer[offset + 46:offset + 46 + name_length]).decode(
                "utf-8" if flags & 0x800 else "cp437")
            self._members[name] = (local, method, compressed, size, crc)
            offset += 46 + name_length + extra_length + comment_length
        self.files = [name[:-4] if name.endswith(".npy") else name for name in self._members]

    def _raw(self, name):
        local, method, compressed, size, crc = self._members[name]
        signature, _, _, _, _, _, _, _, _, name_length, extra_length = _LOCAL.unpack_from(self._buffer, local)
        if signature != 0x04034B50:
            raise _Unsupported("bad local header")
        start = local + _LOCAL.size + name_length + extra_length
        data = memoryview(self._buffer)[start:start + compressed]
        if method == 8:
            data = bytearray(zlib.decompress(data, -15, size))
        if len(data) != size or zlib.crc32(data) != crc:
            raise OSError(f"Bad CRC-32 or size for {name} in {self.path}")
        return data

    def __getitem__(self, key):
        name = key if key in self._members else f"{key}.npy"
        if name not in self._members:
            raise KeyError(f"{key} is not a file in the archive")
        raw = self._raw(name)
        array = _read_npy(raw) if name.endswith(".npy") else None
        if array is None:   # an unusual header or dtype: numpy reads the member
            array = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=True)
        return array

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __contains__(self, key):
        return key in self._members or f"{key}.npy" in self._members


def load_npz(path):
    """np.load(path, allow_pickle=True) for an event file, ~5x faster (see above)."""
    if ENABLED:
        try:
            return FastNpz(path)
        except _Unsupported:
            pass
    return np.load(path, allow_pickle=True)
