# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, Literal

import imageio
import numpy as np
import open3d as o3d
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras import camera_utils
from nerfstudio.data.dataparsers.base_dataparser import (
	DataParser, 
	DataParserConfig, 
	DataparserOutputs
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json



def flip_y(mat):
	r11, r12, r13 = mat[0, :3]
	r21, r22, r23 = mat[1, :3]
	r31, r32, r33 = mat[2, :3]
	t1, t2, t3 = mat[:3, 3]

	return np.array([
		r11, -r12, r13, t1,
		-r21, r22, -r23, -t2,
		r31, -r32, r33, t3,
		0, 0, 0, 1
	]).reshape(4, 4)

def flip_z(mat):
	r11, r12, r13 = mat[0, :3]
	r21, r22, r23 = mat[1, :3]
	r31, r32, r33 = mat[2, :3]
	t1, t2, t3 = mat[:3, 3]

	return np.array([
		r11, r12, -r13, t1,
		r21, r22, -r23, t2,
		-r31, -r32, r33, -t3,
		0, 0, 0, 1
	]).reshape(4, 4)

@dataclass
class BiGSBlenderConfig(DataParserConfig):
	"""BiGS dataset parser config"""

	_target: Type = field(default_factory=lambda: BiGSBlender)
	"""target class to instantiate"""
	data: Path = Path("data")
	"""Directory specifying location of data."""
	scale_factor: float = 1.0
	"""How much to scale the camera origins by."""
	alpha_color: Optional[str] = "white"
	"""alpha color of background, when set to None, InputDataset that consumes DataparserOutputs will not attempt 
	to blend with alpha_colors using image's alpha channel data. Thus rgba image will be directly used in training. """
	ply_path: Optional[Path] = None
	"""Path to PLY file to load 3D points from, defined relative to the dataset directory. This is helpful for
	Gaussian splatting and generally unused otherwise. If `None`, points are initialized randomly."""
	mask_type: Literal["com", "obj"] = "com"

	auto_scale_poses: bool = False
	""" scale the poses with in [0, 1], the same as nerfstudio-dataparser
	"""

	exr_gamma_coeff: float = 2.2
	"""gamma correction using the formula: gamma(x) = x^(1/gamma_coeff)"""


@dataclass
class BiGSBlender(DataParser):
	"""
	support exr files, and gamma mapping
	"""

	config: BiGSBlenderConfig

	def __init__(self, config: BiGSBlenderConfig):
		super().__init__(config=config)
		self.data: Path = config.data
		self.alpha_color = config.alpha_color
		if self.alpha_color is not None:
			self.alpha_color_tensor = get_color(self.alpha_color)
		else:
			self.alpha_color_tensor = None

	def _generate_dataparser_outputs(self, split="train"):
		meta = load_from_json(self.data / f"transforms_{split}.json")
		image_filenames = []
		mask_filenames = []
		poses = []

		fx = []
		fy = []
		cx = []
		cy = []
		height = []
		width = []
		distort = []

		camera_positions = []

		for frame in meta["frames"]:
			img_fname = self.data / Path(frame["file_path"].replace("./", ""))
			image_filenames.append(img_fname)

			mask_fname = self.data / Path(frame[f"{self.config.mask_type}_mask_path"].replace("./", ""))
			mask_filenames.append(mask_fname)

			# poses.append(
			# 	flip_z(flip_y(np.array(frame["transform_matrix"])))
			# )
			poses.append(np.array(frame["transform_matrix"]))
			camera_positions.append(np.array(frame["transform_matrix"])[:3, 3])
			# print("cam_pos: ", np.array(frame["transform_matrix"])[:3, 3])
			# print("cam_dir: ", np.array(frame["transform_matrix"])[:3, 2])


			fx.append(float(frame["fl_x"] if "fl_x" in frame else meta["fl_x"]))
			fy.append(float(frame["fl_y"] if "fl_y" in frame else meta["fl_y"]))
			cx.append(float(frame["cx"] if "cx" in frame else meta["cx"]))
			cy.append(float(frame["cy"] if "cy" in frame else meta["cy"]))
			height.append(int(frame["h"] if "h" in frame else meta["h"]))
			width.append(int(frame["w"] if "w" in frame else meta["w"]))
			distort.append(
				camera_utils.get_distortion_params(
					k1=float(frame["k1"]) if "k1" in frame else 0.0,
					k2=float(frame["k2"]) if "k2" in frame else 0.0,
					k3=float(frame["k3"]) if "k3" in frame else 0.0,
					k4=float(frame["k4"]) if "k4" in frame else 0.0,
					p1=float(frame["p1"]) if "p1" in frame else 0.0,
					p2=float(frame["p2"]) if "p2" in frame else 0.0,
				)
			)

		# if "camera_angle_x" in meta:
		# 	camera_angle_x = float(meta["camera_angle_x"])
		# 	default_fx = default_fy = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

		# 	default_cx = image_width / 2.0
		# 	default_cy = image_height / 2.0
		# else:
		# 	default_fx = float(meta["fl_x"])
		# 	default_fy = float(meta["fl_y"])
		# 	default_cx = float(meta["cx"])
		# 	default_cy = float(meta["cy"])

		poses = torch.tensor(np.array(poses).astype(np.float32))

		scene_box_aabb = torch.tensor(meta["light_aabb"], dtype=torch.float32) if "light_aabb" in meta else torch.tensor([[-10, -10, -10], [10, 10, 10]], dtype=torch.float32)

		scene_box = SceneBox(aabb=scene_box_aabb)

		fx = torch.tensor(fx, dtype=torch.float32)
		fy = torch.tensor(fy, dtype=torch.float32)
		cx = torch.tensor(cx, dtype=torch.float32)
		cy = torch.tensor(cy, dtype=torch.float32)
		height = torch.tensor(height, dtype=torch.int32)
		width = torch.tensor(width, dtype=torch.int32)
		distortion_params = torch.stack(distort, dim=0)

		cameras = Cameras(
			fx=fx,
			fy=fy,
			cx=cx,
			cy=cy,
			distortion_params=distortion_params,
			height=height,
			width=width,
			camera_to_worlds=poses[:, :3, :4],
		)

		scale_factor = 1.0
		if self.config.auto_scale_poses:
			scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
		scale_factor *= self.config.scale_factor

		poses[:, :3, 3] *= scale_factor


		metadata = {
			"exr_gamma_coeff": self.config.exr_gamma_coeff,
		}

		if self.config.ply_path is not None:
			metadata.update(self._load_3D_points(self.config.data / self.config.ply_path))
		
		assert len(image_filenames) == len(mask_filenames) == len(cameras.camera_to_worlds), (
			f"Number of images: {len(image_filenames)}, "
			f"Number of masks: {len(mask_filenames)}, "
			f"Number of cameras: {len(cameras.camera_to_worlds)}")

		dataparser_outputs = DataparserOutputs(
			image_filenames=image_filenames,
			mask_filenames=mask_filenames,
			cameras=cameras,
			alpha_color=self.alpha_color_tensor,
			scene_box=scene_box,
			dataparser_scale=scale_factor,
			metadata=metadata,
		)

		return dataparser_outputs

	def _load_3D_points(self, ply_file_path: Path):
		pcd = o3d.io.read_point_cloud(str(ply_file_path))

		points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32) * self.config.scale_factor)
		points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

		out = {
			"points3D_xyz": points3D,
			"points3D_rgb": points3D_rgb,
		}
		return out