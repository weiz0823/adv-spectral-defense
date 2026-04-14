import os.path as osp
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesUV
from torch import Tensor

from adv_person.utils.torch_utils import gen_uniform_grid
from adv_person.utils.typing import _pair, _size_2_t

from ..patch import ThinPlateSpline
from . import mesh_utils as MU
from . import pytorch3d_modify as p3dmd


@torch.no_grad()
def compute_map_mask(textures: TexturesUV, shape: Optional[_size_2_t] = None):
    x = textures.verts_uvs_padded()[0, :, 0]
    y = textures.verts_uvs_padded()[0, :, 1]
    if shape is None:
        shape = textures._maps_padded.shape[1:3]
    else:
        shape = _pair(shape)
    m = np.zeros(shape)
    faces = textures.faces_uvs_padded()[0]

    xt = x * (m.shape[1] - 1)
    yt = (1 - y) * (m.shape[0] - 1)
    xt = xt.to(torch.int32).tolist()
    yt = yt.to(torch.int32).tolist()
    tri_coords = [np.array([[xt[i], yt[i]] for i in f]) for f in faces]
    cv2.drawContours(m, tri_coords, -1, 1.0, cv2.FILLED)

    mask = torch.from_numpy(m).to(textures._maps_padded)
    return mask


class ClothesModel(object):
    """Camouflage texture generation not included in this model."""

    def __init__(
        self,
        obj_file: str,
        fig_size: _size_2_t,
        num_colors=4,
        resolution=4,
        device: Optional[torch.device] = None,
        loc_info: dict = {},
    ) -> None:
        self.obj_file = obj_file
        self.fig_size = _pair(fig_size)
        self.num_colors = num_colors
        self.resolution = resolution
        self.device = device
        self.loc_info = loc_info

        self.mesh = load_objs_as_meshes([obj_file], device=device)
        textures: TexturesUV = self.mesh.textures
        self.faces = textures.faces_uvs_padded()
        self.verts_uv = textures.verts_uvs_padded()
        self.faces_uvs = textures.faces_uvs_list()[0]

    def initialize_tps(
        self, loc_file: str, tps_range: tuple[float, float] = (0.1, 0.8727)
    ):
        """Intialize 2D TPS.

        Args:
            tps_range: (r, theta) in polar coordinate, theta in radius (default value is about 50 degrees).
        """
        locations_ori: Tensor = torch.load(loc_file, map_location="cpu")
        self.info = MU.get_map_kernel(locations_ori.to(self.device), self.faces_uvs)
        target_ctrl_points = (
            p3dmd.get_points(self.loc_info, wrap=False).squeeze(0).cpu()
        )
        # Dummy shape
        self.tps = ThinPlateSpline(
            0, target_ctrl_points, target_coord=locations_ori
        ).to(self.device)
        self.tps_range = tps_range

    def tps_gen_locations(self, batch_size=1):
        tps_r, tps_theta = self.tps_range
        source_ctrl_points = p3dmd.get_points(
            self.loc_info, tps_theta, tps_r, bs=batch_size, random=True
        )
        return self.tps.get_source_coordinate(source_ctrl_points.to(self.device))

    def update_texture(self, texture: Tensor):
        """Update texture.

        Args:
            texture: NCHW texture tensor.
        """
        texture = texture.permute(0, 2, 3, 1)  # Maps need NHWC format
        self.mesh.textures = TexturesUV(
            maps=texture, faces_uvs=self.faces, verts_uvs=self.verts_uv
        )

    def compute_map_mask(self, shape: Optional[_size_2_t] = None):
        return compute_map_mask(self.mesh.textures, shape)

    def get_masked_map(self, shape: Optional[_size_2_t] = None):
        textures: TexturesUV = self.mesh.textures
        maps = textures._maps_padded.permute(0, 3, 1, 2)  # NCHW
        if shape is not None:
            shape = _pair(shape)
            maps = F.interpolate(maps, size=shape, mode="nearest")
        mask = compute_map_mask(textures, shape)
        return maps * mask + (1 - mask)


class PersonModel(object):
    def __init__(
        self,
        data_dir: str,
        num_colors=4,
        resolution=4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.data_dir = osp.expanduser(data_dir)
        self.num_colors = num_colors
        self.resolution = resolution
        self.device = device

        obj_filename_man = osp.join(self.data_dir, "Archive/Man_join/man.obj")
        obj_filename_tshirt = osp.join(self.data_dir, "Archive/tshirt_join/tshirt.obj")
        obj_filename_trouser = osp.join(
            self.data_dir, "Archive/trouser_join/trouser.obj"
        )
        selected_tshirt = torch.cat(
            [torch.arange(27), torch.arange(28, 31), torch.arange(32, 43)]
        )
        tshirt_locations_info = {
            "nparts": 3,
            "centers": [[7.5, 0], [-7.5, 0], [0, 0]],
            "Rs": [1.5, 1.5, 15.0],
            "ntfs": [6, 6, 8],
            "ntws": [6, 6, 8],
            "radius_fixed": [[1.0], [1.0], [0.5]],
            "radius_wrap": [[0.5], [0.5], [1.0]],
            "signs": [-1, -1, 1],
            "selected": selected_tshirt,
        }
        trouser_locations_info = {
            "nparts": 2,
            "centers": [[3.43, 0], [-3.43, 0]],
            "Rs": [3.3] * 2,
            "ntfs": [20] * 2,
            "ntws": [12] * 2,
            "radius_fixed": [[1.2]] * 2,
            "radius_wrap": [[0.4]] * 2,
            "signs": [1, 1],
            "selected": None,
        }

        self.mesh_man = load_objs_as_meshes([obj_filename_man], device=device)
        self.tshirt = ClothesModel(
            obj_filename_tshirt,
            (340, 864),
            device=device,
            loc_info=tshirt_locations_info,
        )
        self.trouser = ClothesModel(
            obj_filename_trouser,
            (484, 700),
            device=device,
            loc_info=trouser_locations_info,
        )
        self.clothes = [self.tshirt, self.trouser]
        self.mesh = MU.join_meshes([self.mesh_man, self.tshirt.mesh, self.trouser.mesh])

        self.tshirt.initialize_tps(
            osp.join(self.data_dir, "Archive/tshirt_join/projections/part_all_2p5.pt")
        )
        self.trouser.initialize_tps(
            osp.join(
                self.data_dir, "Archive/trouser_join/projections/part_all_off3p4.pt"
            )
        )
        self.initialize_tps3d()

    def initialize_tps3d(self, tps3d_range=0.15):
        pmin = (-0.28170400857925415, -0.7323740124702454, -0.15313300490379333)
        pmax = (0.28170400857925415, 0.5564370155334473, 0.0938199982047081)
        pnum = (5, 8, 5)
        max_range = (torch.tensor(pmax) - torch.tensor(pmin)) / torch.tensor(pnum)
        self.max_range = (max_range * tps3d_range).tolist()
        target_ctrl_points = gen_uniform_grid(pmin, pmax, pnum, flatten=True)
        self.tps3d = ThinPlateSpline(
            0,
            target_ctrl_points,
            max_range=self.max_range,
            target_coord=self.mesh.verts_packed().cpu(),
        ).to(self.device)

    def render(self, cameras, lights, batch_size=1, use_tps2d=True, use_tps3d=True):
        """Render the model to an image.

        Returns:
            NCHW Tensor: RGBA images.
        """
        if use_tps2d:
            locations_tshirt = self.tshirt.tps_gen_locations(batch_size)
            locations_trouser = self.trouser.tps_gen_locations(batch_size)
        else:
            locations_tshirt = locations_trouser = None
        if use_tps3d:
            source_coord = self.tps3d.get_source_coordinate(
                self.tps3d.random_source_ctrl_points(batch_size)
            ).view(-1, 3)
        else:
            source_coord = None

        # Render images
        images_predicted: Tensor = p3dmd.view_mesh_wrapped(
            [self.mesh_man, self.tshirt.mesh, self.trouser.mesh],
            [None, locations_tshirt, locations_trouser],
            [None, self.tshirt.info, self.trouser.info],
            source_coord,
            cameras=cameras,
            lights=lights,
            image_size=800,
            fov=60,
            max_faces_per_bin=30000,
            faces_per_pixel=3,
        )
        # We need NCHW
        return images_predicted.permute(0, 3, 1, 2)
