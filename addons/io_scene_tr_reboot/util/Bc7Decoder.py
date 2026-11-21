import numpy as np
from numpy.typing import NDArray

# Pure numpy BC7 decoder. Blender's DDS loader can't decode BC7 (DXGI formats 97-99),
# so these textures are decoded here and handed to Blender as raw pixels.

_PARTITIONS_2 = [
    "0011001100110011", "0001000100010001", "0111011101110111", "0001001100110111",
    "0000000100010011", "0011011101111111", "0001001101111111", "0000000100110111",
    "0000000000010011", "0011011111111111", "0000000101111111", "0000000000010111",
    "0001011111111111", "0000000011111111", "0000111111111111", "0000000000001111",
    "0000100011101111", "0111000100000000", "0000000010001110", "0111001100010000",
    "0011000100000000", "0000100011001110", "0000000010001100", "0111001100110001",
    "0011000100010000", "0000100010001100", "0110011001100110", "0011011001101100",
    "0001011111101000", "0000111111110000", "0111000110001110", "0011100110011100",
    "0101010101010101", "0000111100001111", "0101101001011010", "0011001111001100",
    "0011110000111100", "0101010110101010", "0110100101101001", "0101101010100101",
    "0111001111001110", "0001001111001000", "0011001001001100", "0011101111011100",
    "0110100110010110", "0011110011000011", "0110011010011001", "0000011001100000",
    "0100111001000000", "0010011100100000", "0000001001110010", "0000010011100100",
    "0110110010010011", "0011011011001001", "0110001110011100", "0011100111000110",
    "0110110011001001", "0110001100111001", "0111111010000001", "0001100011100111",
    "0000111100110011", "0011001111110000", "0010001011101110", "0100010001110111",
]

_PARTITIONS_3 = [
    "0011001102212222", "0001001122112221", "0000200122112211", "0222002200110111",
    "0000000011221122", "0011001100220022", "0022002211111111", "0011001122112211",
    "0000000011112222", "0000111111112222", "0000111122222222", "0012001200120012",
    "0112011201120112", "0122012201220122", "0011011211221222", "0011200122002220",
    "0001001101121122", "0111001120012200", "0000112211221122", "0022002200221111",
    "0111011102220222", "0001000122212221", "0000001101220122", "0000110022102210",
    "0122012200110000", "0012001211222222", "0110122112210110", "0000011012211221",
    "0022110211020022", "0110011020022222", "0011012201220011", "0000200022112221",
    "0000000211221222", "0222002200120011", "0011001200220222", "0120012001200120",
    "0000111122220000", "0120120120120120", "0120201212010120", "0011220011220011",
    "0011112222000011", "0101010122222222", "0000000021212121", "0022112200221122",
    "0022001100220011", "0220122102201221", "0101222222220101", "0000212121212121",
    "0101010101012222", "0222011102220111", "0002111200021112", "0000211221122112",
    "0222011101110222", "0002111211120002", "0110011001102222", "0000000021122112",
    "0110011022222222", "0022001100110022", "0022112211220022", "0000000000002112",
    "0002000100020001", "0222122202221222", "0101222222222222", "0111201122012220",
]

_ANCHOR_2_2 = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15,  2,  8,  2,  2,  8,  8, 15,  2,  8,  2,  2,  8,  8,  2,  2,
    15, 15,  6,  8,  2,  8, 15, 15,  2,  8,  2,  2,  2, 15, 15,  6,
     6,  2,  6,  8, 15, 15,  2,  2, 15, 15, 15, 15, 15,  2,  2, 15,
]

_ANCHOR_3_2 = [
     3,  3, 15, 15,  8,  3, 15, 15,  8,  8,  6,  6,  6,  5,  3,  3,
     3,  3,  8, 15,  3,  3,  6, 10,  5,  8,  8,  6,  8,  5, 15, 15,
     8, 15,  3,  5,  6, 10,  8, 15, 15,  3, 15,  5, 15, 15, 15, 15,
     3, 15,  5,  5,  5,  8,  5, 10,  5, 10,  8, 13, 15, 12,  3,  3,
]

_ANCHOR_3_3 = [
    15,  8,  8,  3, 15, 15,  3,  8, 15, 15, 15, 15, 15, 15, 15,  8,
    15,  8, 15,  3, 15,  8, 15,  8,  3, 15,  6, 10, 15, 15, 10,  8,
    15,  3, 15, 10, 10,  8,  9, 10,  6, 15,  8, 15,  3,  6,  6,  8,
    15,  3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  3, 15, 15,  8,
]

_WEIGHTS = {
    2: np.array([0, 21, 43, 64], dtype = np.int32),
    3: np.array([0, 9, 18, 27, 37, 46, 55, 64], dtype = np.int32),
    4: np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], dtype = np.int32),
}

# Subset index of every pixel, per subset count and partition: [num_subsets][partition][pixel]
_SUBSETS = {
    1: np.zeros((64, 16), dtype = np.int32),
    2: np.array([[int(c) for c in p] for p in _PARTITIONS_2], dtype = np.int32),
    3: np.array([[int(c) for c in p] for p in _PARTITIONS_3], dtype = np.int32),
}

# Whether a pixel is an anchor (stored with one index bit less): [num_subsets][partition][pixel]
def _make_anchors() -> dict[int, NDArray[np.bool_]]:
    anchors = {n: np.zeros((64, 16), dtype = np.bool_) for n in (1, 2, 3)}
    for n in (1, 2, 3):
        anchors[n][:, 0] = True

    for p in range(64):
        anchors[2][p, _ANCHOR_2_2[p]] = True
        anchors[3][p, _ANCHOR_3_2[p]] = True
        anchors[3][p, _ANCHOR_3_3[p]] = True

    return anchors

_ANCHORS = _make_anchors()

class _Mode:
    def __init__(self, ns: int, pb: int, rb: int, isb: int, cb: int, ab: int, epb: int, spb: int, ib: int, ib2: int) -> None:
        self.ns = ns
        self.pb = pb
        self.rb = rb
        self.isb = isb
        self.cb = cb
        self.ab = ab
        self.epb = epb
        self.spb = spb
        self.ib = ib
        self.ib2 = ib2

_MODES = [
    _Mode(3, 4, 0, 0, 4, 0, 1, 0, 3, 0),
    _Mode(2, 6, 0, 0, 6, 0, 0, 1, 3, 0),
    _Mode(3, 6, 0, 0, 5, 0, 0, 0, 2, 0),
    _Mode(2, 6, 0, 0, 7, 0, 1, 0, 2, 0),
    _Mode(1, 0, 2, 1, 5, 6, 0, 0, 2, 3),
    _Mode(1, 0, 2, 0, 7, 8, 0, 0, 2, 2),
    _Mode(1, 0, 0, 0, 7, 7, 1, 0, 4, 0),
    _Mode(2, 6, 0, 0, 5, 5, 1, 0, 2, 0),
]

class _BitReader:
    def __init__(self, bits: NDArray[np.uint8], start: int) -> None:
        self.bits = bits
        self.rows = np.arange(bits.shape[0])
        self.pos = np.full(bits.shape[0], start, dtype = np.int32)

    def read(self, count: int | NDArray[np.int32]) -> NDArray[np.int32]:
        counts = np.broadcast_to(np.asarray(count, dtype = np.int32), self.pos.shape)
        max_count = int(counts.max(initial = 0))
        value = np.zeros(self.pos.shape, dtype = np.int32)
        for i in range(max_count):
            bit = self.bits[self.rows, np.minimum(self.pos + i, 127)].astype(np.int32)
            value |= np.where(i < counts, bit << i, 0)

        self.pos += counts
        return value

def _unquantize(value: NDArray[np.int32], precision: int) -> NDArray[np.int32]:
    value = value << (8 - precision)
    return value | (value >> precision)

def _decode_mode(bits: NDArray[np.uint8], mode_idx: int) -> NDArray[np.uint8]:
    mode = _MODES[mode_idx]
    num_blocks = bits.shape[0]
    reader = _BitReader(bits, mode_idx + 1)

    partition = reader.read(mode.pb) if mode.pb else np.zeros(num_blocks, dtype = np.int32)
    rotation  = reader.read(mode.rb) if mode.rb else np.zeros(num_blocks, dtype = np.int32)
    index_sel = reader.read(mode.isb) if mode.isb else np.zeros(num_blocks, dtype = np.int32)

    num_endpoints = mode.ns * 2
    endpoints = np.zeros((num_blocks, num_endpoints, 4), dtype = np.int32)
    for channel in range(3):
        for e in range(num_endpoints):
            endpoints[:, e, channel] = reader.read(mode.cb)

    if mode.ab:
        for e in range(num_endpoints):
            endpoints[:, e, 3] = reader.read(mode.ab)

    color_precision = mode.cb
    alpha_precision = mode.ab
    if mode.epb or mode.spb:
        pbits = np.zeros((num_blocks, num_endpoints), dtype = np.int32)
        if mode.epb:
            for e in range(num_endpoints):
                pbits[:, e] = reader.read(1)
        else:
            for s in range(mode.ns):
                pbit = reader.read(1)
                pbits[:, s * 2] = pbit
                pbits[:, s * 2 + 1] = pbit

        endpoints[:, :, :3] = (endpoints[:, :, :3] << 1) | pbits[:, :, None]
        color_precision += 1
        if mode.ab:
            endpoints[:, :, 3] = (endpoints[:, :, 3] << 1) | pbits
            alpha_precision += 1

    endpoints[:, :, :3] = _unquantize(endpoints[:, :, :3], color_precision)
    if mode.ab:
        endpoints[:, :, 3] = _unquantize(endpoints[:, :, 3], alpha_precision)
    else:
        endpoints[:, :, 3] = 255

    anchors = _ANCHORS[mode.ns][partition]         # (blocks, 16)
    subsets = _SUBSETS[mode.ns][partition]         # (blocks, 16)

    indices = np.zeros((num_blocks, 16), dtype = np.int32)
    for i in range(16):
        indices[:, i] = reader.read(np.where(anchors[:, i], mode.ib - 1, mode.ib))

    indices2: NDArray[np.int32] | None = None
    if mode.ib2:
        indices2 = np.zeros((num_blocks, 16), dtype = np.int32)
        for i in range(16):
            indices2[:, i] = reader.read(mode.ib2 - 1 if i == 0 else mode.ib2)

    rows = np.arange(num_blocks)[:, None]
    e0 = endpoints[rows, subsets * 2]              # (blocks, 16, 4)
    e1 = endpoints[rows, subsets * 2 + 1]

    if indices2 is None:
        weights = _WEIGHTS[mode.ib][indices]
        color_weights = weights
        alpha_weights = weights
    else:
        primary_weights = _WEIGHTS[mode.ib][indices]
        secondary_weights = _WEIGHTS[mode.ib2][indices2]
        swap = (index_sel == 1)[:, None]
        color_weights = np.where(swap, secondary_weights, primary_weights)
        alpha_weights = np.where(swap, primary_weights, secondary_weights)

    pixels = np.empty((num_blocks, 16, 4), dtype = np.int32)
    pixels[:, :, :3] = ((64 - color_weights[:, :, None]) * e0[:, :, :3] + color_weights[:, :, None] * e1[:, :, :3] + 32) >> 6
    pixels[:, :, 3]  = ((64 - alpha_weights) * e0[:, :, 3] + alpha_weights * e1[:, :, 3] + 32) >> 6

    for rot in (1, 2, 3):
        mask = rotation == rot
        if mask.any():
            channel = rot - 1
            alpha = pixels[mask, :, 3].copy()
            pixels[mask, :, 3] = pixels[mask, :, channel]
            pixels[mask, :, channel] = alpha

    return pixels.astype(np.uint8)

def decode_bc7(data: bytes | memoryview, width: int, height: int) -> NDArray[np.uint8]:
    """Decodes a BC7 surface into an array of shape (height, width, 4), top row first."""
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    num_blocks = blocks_x * blocks_y

    raw = np.frombuffer(data, dtype = np.uint8, count = num_blocks * 16).reshape(num_blocks, 16)
    pixels = np.zeros((num_blocks, 16, 4), dtype = np.uint8)

    chunk_size = 1 << 16
    for chunk_start in range(0, num_blocks, chunk_size):
        chunk = raw[chunk_start:chunk_start + chunk_size]
        bits = np.unpackbits(chunk, axis = 1, bitorder = "little")

        # The mode is the position of the lowest set bit in the first byte; mode 8 (no bit set) is reserved and decodes to zeros
        first_byte = chunk[:, 0].astype(np.int32)
        lowest_bit = first_byte & -first_byte
        modes = np.where(first_byte == 0, 8, np.log2(np.maximum(lowest_bit, 1)).astype(np.int32))

        for mode_idx in range(8):
            mask = modes == mode_idx
            if mask.any():
                pixels[chunk_start + np.nonzero(mask)[0]] = _decode_mode(bits[mask], mode_idx)

    image = pixels.reshape(blocks_y, blocks_x, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(blocks_y * 4, blocks_x * 4, 4)
    return image[:height, :width]
