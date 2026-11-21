from io_scene_tr_reboot.tr.rise.RiseMaterial import RiseMaterial

class ShadowMaterial(RiseMaterial):
    @property
    def num_passes(self) -> int:
        return 9

    @property
    def has_texture_categories(self) -> bool:
        return True

    @property
    def is_translucent(self) -> bool | None:
        # Opaque materials have the deferred G-buffer passes (2 and 3); translucent ones (corneas, eyelashes,
        # hair cards, eye shadow overlays...) only have the forward passes
        return 2 not in self.pass_indices

    def get_texture_category(self, extra_info: memoryview) -> int | None:
        # float mipLodBias; ushort category:3, type:5, class:3, slot:5; ubyte filter; ubyte padding
        return int.from_bytes(extra_info[4:6], "little") & 7
