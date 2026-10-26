"""Read TensorFlow (V2) checkpoints without TensorFlow, e.g. the ``variables/`` of a Keras SavedModel.

A checkpoint is ``<prefix>.index``, an SSTable (LevelDB table format) mapping each variable name to a
``BundleEntryProto`` (dtype, shape, shard, offset, size), and ``<prefix>.data-00000-of-00001``, the raw
little-endian tensor bytes. This reader handles what Keras writes for small models: one data shard,
uncompressed blocks, numeric dtypes; anything else raises an error.
"""

import struct
from pathlib import Path

import numpy as np

_SSTABLE_MAGIC = 0xDB4775248B80FB57
# TensorFlow DataType enum values -> numpy dtypes.
_DTYPES = {1: np.float32, 2: np.float64, 3: np.int32, 4: np.uint8, 6: np.int8, 9: np.int64, 10: np.bool_, 19: np.float16}


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _block_entries(block: bytes):
    """Key/value pairs of one SSTable block (keys are prefix-compressed against the previous one)."""
    n_restarts = struct.unpack("<I", block[-4:])[0]
    end = len(block) - 4 - 4 * n_restarts
    i, key = 0, b""
    while i < end:
        shared, i = _varint(block, i)
        unshared, i = _varint(block, i)
        value_length, i = _varint(block, i)
        key = key[:shared] + block[i : i + unshared]
        i += unshared
        yield key, block[i : i + value_length]
        i += value_length


def _read_block(data: bytes, handle: bytes) -> bytes:
    offset, i = _varint(handle, 0)
    size, _ = _varint(handle, i)
    if data[offset + size] != 0:  # the byte after each block gives its compression
        raise ValueError("Compressed checkpoint index blocks are not supported")
    return data[offset : offset + size]


def read_index(path) -> dict[bytes, bytes]:
    """All entries of a checkpoint ``.index`` file."""
    data = Path(path).read_bytes()
    footer = data[-48:]
    if struct.unpack("<Q", footer[-8:])[0] != _SSTABLE_MAGIC:
        raise ValueError(f"{path} is not a TensorFlow checkpoint index")
    i = 0
    for _ in range(2):  # skip the metaindex block handle
        _, i = _varint(footer, i)
    index_block = _read_block(data, footer[i:])
    entries = {}
    for _, handle in _block_entries(index_block):
        entries.update(_block_entries(_read_block(data, handle)))
    return entries


def _fields(buf: bytes):
    """(field number, wire type, value) of a protobuf message; length-delimited values as bytes."""
    i = 0
    while i < len(buf):
        tag, i = _varint(buf, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 2:
            n, i = _varint(buf, i)
            value, i = buf[i : i + n], i + n
        elif wire == 5:
            value, i = buf[i : i + 4], i + 4
        elif wire == 1:
            value, i = buf[i : i + 8], i + 8
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield field, wire, value


def _bundle_entry(buf: bytes) -> dict:
    """The parts of a ``BundleEntryProto`` needed to read a tensor."""
    entry = {"dtype": 0, "shape": [], "shard_id": 0, "offset": 0, "size": 0, "slices": False}
    for field, _, value in _fields(buf):
        if field == 1:
            entry["dtype"] = value
        elif field == 2:  # TensorShapeProto: repeated Dim dim = 2 {int64 size = 1}
            for dim_field, _, dim in _fields(value):
                if dim_field == 2:
                    entry["shape"].append(next((v for f, _, v in _fields(dim) if f == 1), 0))
        elif field == 3:
            entry["shard_id"] = value
        elif field == 4:
            entry["offset"] = value
        elif field == 5:
            entry["size"] = value
        elif field == 7:
            entry["slices"] = True
    return entry


def read_checkpoint(prefix) -> dict[str, np.ndarray]:
    """Numeric variables of a checkpoint, by name (e.g. ``"layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE"``).

    `prefix` is the path without extension, e.g. ``<saved model>/variables/variables``.
    """
    entries = read_index(f"{prefix}.index")
    shards = sorted(Path(prefix).parent.glob(f"{Path(prefix).name}.data-*"))
    if len(shards) != 1:
        raise ValueError(f"Expected one data shard for {prefix}, found {len(shards)}")
    data = shards[0].read_bytes()
    tensors = {}
    for key, value in entries.items():
        if not key:  # the bundle header
            continue
        entry = _bundle_entry(value)
        if entry["dtype"] not in _DTYPES:  # e.g. strings (the object graph)
            continue
        if entry["slices"] or entry["shard_id"]:
            raise ValueError(f"Partitioned variable {key.decode()} is not supported")
        array = np.frombuffer(data[entry["offset"] : entry["offset"] + entry["size"]], dtype=_DTYPES[entry["dtype"]])
        tensors[key.decode()] = array.reshape(entry["shape"]).copy()
    return tensors
