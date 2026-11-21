from typing import NamedTuple, cast
import bpy
import numpy as np
import os
import re
from io_scene_tr_reboot.BlenderHelper import BlenderHelper
from io_scene_tr_reboot.BlenderNaming import BlenderNaming
from io_scene_tr_reboot.tr.Collection import Collection
from io_scene_tr_reboot.tr.Enumerations import ResourceType
from io_scene_tr_reboot.tr.Material import Material
from io_scene_tr_reboot.tr.ResourceKey import ResourceKey
from io_scene_tr_reboot.util.DdsFile import DdsFile, DdsInfo, DdsType
from io_scene_tr_reboot.util.Enumerable import Enumerable
from io_scene_tr_reboot.util.SlotsBase import SlotsBase

class _TextureNodeSet(NamedTuple):
    diffuse: bpy.types.ShaderNodeTexImage | None
    normal:  bpy.types.ShaderNodeTexImage | None
    diffuse_alpha_is_opacity: bool
    default_opacity: float

class MaterialImporter(SlotsBase):
    def import_from_collection(self, tr_collection: Collection) -> None:
        for material_resource in tr_collection.get_resources(ResourceType.MATERIAL):
            self.import_material(tr_collection, material_resource)

    def import_material(self, tr_collection: Collection, material_resource: ResourceKey) -> bpy.types.Material | None:
        material_name = BlenderNaming.make_material_name(tr_collection.get_resource_name(material_resource), material_resource.id)

        bl_material = bpy.data.materials.get(material_name)
        if bl_material is not None:
            return bl_material

        tr_material = tr_collection.get_material(material_resource)
        if tr_material is None:
            return None

        bl_material = BlenderHelper.create_empty_material(material_name)
        if bl_material.node_tree is None:
            raise Exception()

        bl_texture_nodes = self.__add_texture_nodes(bl_material, tr_collection, tr_material)

        bl_transparent_node = bl_material.node_tree.nodes.new("ShaderNodeBsdfTransparent")
        bl_transparent_node.location = (300, 0)

        bl_diffuse_shader_node = bl_material.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
        bl_diffuse_shader_node.location = (300, -100)

        if bl_texture_nodes.diffuse is not None:
            bl_material.node_tree.links.new(bl_texture_nodes.diffuse.outputs[0], bl_diffuse_shader_node.inputs[0])

        bl_transparent_node.width = max(bl_transparent_node.width, bl_diffuse_shader_node.width)
        bl_diffuse_shader_node.width = bl_transparent_node.width

        bl_normal_node: bpy.types.Node | None = None
        if bl_texture_nodes.normal is not None:
            bl_separate_node = bl_material.node_tree.nodes.new("ShaderNodeSeparateColor")
            bl_separate_node.location = (300, -250)
            bl_separate_node.hide = True
            bl_material.node_tree.links.new(bl_texture_nodes.normal.outputs[0], bl_separate_node.inputs[0])

            bl_combine_node = bl_material.node_tree.nodes.new("ShaderNodeCombineColor")
            bl_combine_node.location = (300, -300)
            bl_combine_node.hide = True
            bl_material.node_tree.links.new(bl_separate_node.outputs[0], bl_combine_node.inputs[0])
            bl_material.node_tree.links.new(bl_separate_node.outputs[1], bl_combine_node.inputs[1])
            cast(bpy.types.NodeSocketFloat, bl_combine_node.inputs[2]).default_value = 1.0

            bl_normal_node = bl_material.node_tree.nodes.new("ShaderNodeNormalMap")
            bl_normal_node.location = (300, -350)
            cast(bpy.types.NodeSocketFloat, bl_normal_node.inputs[0]).default_value = 0.5
            bl_material.node_tree.links.new(bl_combine_node.outputs[0], bl_normal_node.inputs[1])
            bl_material.node_tree.links.new(bl_normal_node.outputs[0], bl_diffuse_shader_node.inputs[2])

        bl_mix_node = bl_material.node_tree.nodes.new("ShaderNodeMixShader")
        bl_mix_node.location = (500, 0)
        bl_material.node_tree.links.new(bl_transparent_node.outputs[0], bl_mix_node.inputs[1])
        bl_material.node_tree.links.new(bl_diffuse_shader_node.outputs[0], bl_mix_node.inputs[2])
        cast(bpy.types.NodeSocketFloat, bl_mix_node.inputs[0]).default_value = bl_texture_nodes.default_opacity
        if bl_texture_nodes.diffuse is not None and bl_texture_nodes.diffuse_alpha_is_opacity:
            bl_material.node_tree.links.new(bl_texture_nodes.diffuse.outputs[1], bl_mix_node.inputs[0])

        bl_output_node = bl_material.node_tree.nodes.new("ShaderNodeOutputMaterial")
        bl_output_node.location = (700, 0)
        bl_material.node_tree.links.new(bl_mix_node.outputs[0], bl_output_node.inputs[0])

        return bl_material

    def __add_texture_nodes(self, bl_material: bpy.types.Material, tr_collection: Collection, tr_material: Material) -> _TextureNodeSet:
        bl_texture_nodes: list[bpy.types.ShaderNodeTexImage] = []

        texture_resources: list[ResourceKey] = list(tr_material.pixel_shader_textures)

        if bl_material.node_tree is None:
            raise Exception()

        texture_file_paths: dict[ResourceKey, str] = {}
        for texture_resource in texture_resources:
            file_path = tr_collection.get_resource_file_path(texture_resource)
            if file_path is not None:
                texture_file_paths[texture_resource] = file_path

        def get_file_path_sort_key(file_path: str) -> str:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            if re.fullmatch(r"\d+", file_name):
                file_name = f"{file_name:0>10}"

            return file_name

        # If the material has an alpha-tested texture (eyelashes, eyebrows, hair cards...), that texture
        # holds both the base color and the opacity. Otherwise, the largest color texture is taken as the
        # base color. For an opaque material, that texture's alpha channel is something else (e.g. an iris mask)
        # and must not be used as opacity. Games without texture categories keep using the alpha as opacity.
        alpha_test_file_path = Enumerable(tr_material.alpha_test_textures).select(lambda r: texture_file_paths.get(r)).of_type(str).first_or_none()
        is_translucent = tr_material.is_translucent
        diffuse_alpha_is_opacity = alpha_test_file_path is not None or \
                                   not tr_material.has_texture_categories or \
                                   is_translucent is not False

        diffuse_texture_idx = -1
        diffuse_texture_area = -1

        normal_texture_idx = -1
        normal_texture_area = -1

        for file_path in sorted(set(texture_file_paths.values()), key = get_file_path_sort_key):
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            dds_info = DdsFile.get_info(file_path)
            texture_type = dds_info.type
            if re.search(r"_n\.\d+$", file_name):
                texture_type = DdsType.NORMAL

            texture_area = dds_info.width * dds_info.height
            if file_path == alpha_test_file_path:
                diffuse_texture_idx = len(bl_texture_nodes)
                diffuse_texture_area = 1 << 62
            elif texture_type == DdsType.RGB and texture_area > diffuse_texture_area:
                diffuse_texture_idx = len(bl_texture_nodes)
                diffuse_texture_area = texture_area
            elif texture_type == DdsType.NORMAL and texture_area > normal_texture_area:
                normal_texture_idx = len(bl_texture_nodes)
                normal_texture_area = texture_area

            bl_image = self.__load_image(file_path, dds_info)

            bl_texture_node = cast(bpy.types.ShaderNodeTexImage, bl_material.node_tree.nodes.new("ShaderNodeTexImage"))
            bl_texture_node.image = bl_image
            bl_texture_node.location = (0, len(bl_texture_nodes) * -40)
            bl_texture_node.hide = True
            bl_texture_nodes.append(bl_texture_node)

        # Opacity when the diffuse texture's alpha isn't used: opaque materials are solid; a translucent
        # material without any color texture (e.g. a cornea, which only has a normal map) is fully clear;
        # if translucency is unknown, keep the old half-transparent look.
        default_opacity: float
        if is_translucent is None:
            default_opacity = 0.5
        elif is_translucent and diffuse_texture_idx < 0:
            default_opacity = 0.0
        else:
            default_opacity = 1.0

        return _TextureNodeSet(
            diffuse_texture_idx >= 0 and bl_texture_nodes[diffuse_texture_idx] or None,
            normal_texture_idx  >= 0 and bl_texture_nodes[normal_texture_idx]  or None,
            diffuse_alpha_is_opacity,
            default_opacity
        )

    def __load_image(self, file_path: str, dds_info: DdsInfo) -> bpy.types.Image:
        image_name = os.path.basename(file_path)
        bl_image = bpy.data.images.get(image_name)
        if bl_image is not None:
            return bl_image

        if not DdsFile.is_bc7(dds_info):
            return bpy.data.images.load(file_path)

        # Blender can't decode BC7, so decode it ourselves and pack the result into the .blend
        pixels = DdsFile.read_bc7_pixels(file_path, dds_info)
        bl_image = bpy.data.images.new(image_name, dds_info.width, dds_info.height, alpha = True)
        bl_image.pixels.foreach_set((pixels[::-1].astype(np.float32) / 255.0).ravel())
        bl_image.pack()
        return bl_image
