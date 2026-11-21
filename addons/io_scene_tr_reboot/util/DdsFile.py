from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple
from io_scene_tr_reboot.util.BinaryReader import BinaryReader
from io_scene_tr_reboot.util.IoHelper import IoHelper

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

class DdsConstants:
    DDS_FOURCC = 0x00000004
    DDS_RGB = 0x00000040

class DdsType(IntEnum):
    OTHER  = 0
    RGB    = 1
    NORMAL = 2

class DdsInfo(NamedTuple):
    type: DdsType
    width: int
    height: int
    dxgi_format: int | None

class DdsFile:
    HEADER_SIZE = 0x80
    DX10_HEADER_SIZE = 0x14
    BC7_FORMATS = (97, 98, 99)      # DXGI_FORMAT_BC7_TYPELESS, _UNORM, _UNORM_SRGB

    @staticmethod
    def get_info(file_path: str) -> DdsInfo:
        with IoHelper.open_read(file_path) as file:
            header = file.read(DdsFile.HEADER_SIZE + DdsFile.DX10_HEADER_SIZE)

        reader = BinaryReader(header)
        height = reader.read_uint32_at(0xC)
        width = reader.read_uint32_at(0x10)
        dxgi_format: int | None = None
        if bytes(reader.read_bytes_at(0x54, 4)) == b"DX10":
            dxgi_format = reader.read_int32_at(DdsFile.HEADER_SIZE)

        return DdsInfo(DdsFile.get_type(file_path), width, height, dxgi_format)

    @staticmethod
    def is_bc7(info: DdsInfo) -> bool:
        return info.dxgi_format in DdsFile.BC7_FORMATS

    @staticmethod
    def read_bc7_pixels(file_path: str, info: DdsInfo) -> "NDArray[np.uint8]":
        """Returns the top mip level as an array of shape (height, width, 4), top row first."""
        from io_scene_tr_reboot.util.Bc7Decoder import decode_bc7
        with IoHelper.open_read(file_path) as file:
            data = file.read()

        return decode_bc7(memoryview(data)[DdsFile.HEADER_SIZE + DdsFile.DX10_HEADER_SIZE:], info.width, info.height)

    @staticmethod
    def get_type(file_path: str) -> DdsType:
        header: bytes
        with IoHelper.open_read(file_path) as file:
            header = file.read(0x94)

        reader = BinaryReader(header)
        if bytes(reader.read_bytes(4)) != b"DDS ":
            raise Exception("Invalid DDS magic")

        # Skip to DDS_PIXELFORMAT structure
        reader.skip(0x48)

        pixel_format_flags = reader.read_uint32_at(4)
        if pixel_format_flags & DdsConstants.DDS_RGB:
            r_bitmask = reader.read_uint32_at(0x10)
            g_bitmask = reader.read_uint32_at(0x14)
            b_bitmask = reader.read_uint32_at(0x18)
            if r_bitmask > 0 and g_bitmask > 0:
                if b_bitmask > 0:
                    return DdsType.RGB
                else:
                    return DdsType.NORMAL
            else:
                return DdsType.OTHER
        elif pixel_format_flags & DdsConstants.DDS_FOURCC:
            fourcc = bytes(reader.read_bytes_at(8, 4))
            match fourcc:
                case b"DXT1" | b"DXT2" | b"DXT3" | b"DXT4" | b"DXT5" | b"BC4U" | b"BC4S" | b"ATI2" | b"BC5S":
                    return DdsType.RGB

                case b"DX10":
                    # Skip to DDS_HEADER_DXT10 structure
                    reader.skip(0x20 + 0x14)
                    dxgi_format = reader.read_int32_at(0)
                    if 1  <= dxgi_format <= 14 or \
                       23 <= dxgi_format <= 32 or \
                       70 <= dxgi_format <= 78 or \
                       87 <= dxgi_format <= 93 or \
                       97 <= dxgi_format <= 99 or \
                       dxgi_format == 115:
                        return DdsType.RGB
                    elif 15 <= dxgi_format <= 18 or \
                         33 <= dxgi_format <= 38 or \
                         48 <= dxgi_format <= 52 or \
                         82 <= dxgi_format <= 84:
                        return DdsType.NORMAL
                    else:
                        return DdsType.OTHER

                case _:
                    return DdsType.OTHER
        else:
            return DdsType.OTHER
