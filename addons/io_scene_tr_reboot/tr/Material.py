from abc import abstractmethod
from typing import Protocol
from io_scene_tr_reboot.tr.ResourceKey import ResourceKey
from io_scene_tr_reboot.tr.ResourceReader import ResourceReader
from io_scene_tr_reboot.tr.ResourceReference import ResourceReference

class IMaterialPass(Protocol):
    ps_textures_ref: ResourceReference | None
    num_ps_textures: int

    vs_textures_ref: ResourceReference | None
    num_vs_textures: int

class Material:
    # Texture category used for alpha-tested (cutout) textures, e.g. hair cards, eyebrows and eyelashes.
    # Their alpha channel is the opacity, and their color is the material's base color.
    TEXTURE_CATEGORY_ALPHA_TEST = 1

    pixel_shader_textures: list[ResourceKey]
    vertex_shader_textures: list[ResourceKey]
    alpha_test_textures: list[ResourceKey]
    pass_indices: list[int]

    def __init__(self) -> None:
        self.pixel_shader_textures = []
        self.vertex_shader_textures = []
        self.alpha_test_textures = []
        self.pass_indices = []

    def read(self, reader: ResourceReader) -> None:
        reader.skip(self.magic_offset)
        magic = bytes(reader.read_bytes(4))
        if magic != b"DtaM":
            raise Exception("Invalid magic in material resource")

        for pass_idx, pass_ref in enumerate(reader.read_ref_list(self.num_passes)):
            if pass_ref is None:
                continue

            self.pass_indices.append(pass_idx)

            reader.seek(pass_ref)
            material_pass = self.read_pass(reader)
            if material_pass.ps_textures_ref is not None:
                self.read_pass_textures(reader, material_pass.ps_textures_ref, material_pass.num_ps_textures, self.pixel_shader_textures, self.alpha_test_textures)

            if material_pass.vs_textures_ref is not None:
                self.read_pass_textures(reader, material_pass.vs_textures_ref, material_pass.num_vs_textures, self.vertex_shader_textures)

    def read_pass_textures(
        self,
        reader: ResourceReader,
        ref: ResourceReference,
        count: int,
        textures: list[ResourceKey],
        alpha_test_textures: list[ResourceKey] | None = None
    ) -> None:
        reader.seek(ref)
        for _ in range(count):
            texture_ref = reader.read_ref()
            extra_info = reader.read_bytes(self.pass_texture_extra_info_size)
            if texture_ref is None:
                continue

            if texture_ref not in textures:
                textures.append(texture_ref)

            if alpha_test_textures is not None and \
               self.get_texture_category(extra_info) == Material.TEXTURE_CATEGORY_ALPHA_TEST and \
               texture_ref not in alpha_test_textures:
                alpha_test_textures.append(texture_ref)

    @property
    def has_texture_categories(self) -> bool:
        return False

    @property
    def is_translucent(self) -> bool | None:
        """Whether the material is drawn in the forward (blended) passes instead of the opaque ones; None if unknown."""
        return None

    def get_texture_category(self, extra_info: memoryview) -> int | None:
        return None

    @property
    @abstractmethod
    def magic_offset(self) -> int: ...

    @property
    @abstractmethod
    def num_passes(self) -> int: ...

    @property
    @abstractmethod
    def pass_texture_extra_info_size(self) -> int: ...

    @abstractmethod
    def read_pass(self, reader: ResourceReader) -> IMaterialPass: ...
