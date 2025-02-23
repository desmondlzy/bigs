import torch
import numpy as np
from pathlib import Path
import json
import functools
import imageio.v2 as imageio
from torchvision.io import read_image
from torchvision.transforms.v2 import (
	Compose,
	Resize,
	ToDtype,
)

from relight.read_exr import read_exr


class BiGSDataset(torch.utils.data.Dataset):
	root: Path
	split: str
	camera_ids: list[int]
	img_suffix: str
	scale_factor: float
	light_intensity_factor: float
	shared_attributes = ["camera_angle_x", "fl_x", "fl_y", "cx", "cy"]

	def __init__(self, 
			root: str,
			split: str,
			camera_ids: list[int] | None,
			img_suffix: str = ".png",
			background: str | torch.Tensor = "black",
			scale_factor: float = 1.0,
			light_intensity_factor: float = 1.0,
			load_masks: bool = False):

		self.split = split
		self.root = Path(root)
		self.img_suffix = img_suffix.lstrip(".")
		self.load_masks = load_masks
		self.scale_factor = scale_factor
		self.light_intensity_factor = light_intensity_factor
		if isinstance(background, str):
			if background == "black":
				self.background = torch.tensor([0, 0, 0], dtype=torch.float32).reshape(3, 1, 1)
			elif background == "white":
				self.background = torch.tensor([1, 1, 1], dtype=torch.float32).reshape(3, 1, 1)
			else:
				raise ValueError(f"Invalid background: {background}")
		elif isinstance(background, torch.Tensor):
			self.background = background.reshape(3, 1, 1)

		with open(self.root / f"transforms_{split}.json", "r") as f:
			tfm = json.load(f)

		self.camera_ids = list(camera_ids) if camera_ids else list(range(len(tfm["frames"])))

		self.camera_intrinsics = {}
		for attr in self.shared_attributes:
			if attr in tfm:
				self.camera_intrinsics[attr] = tfm[attr]

		self.camera_frames = [f for i, f in enumerate(tfm["frames"]) 
						if i in self.camera_ids]

		self.camera_matrices = [
			torch.tensor(camera_frame["transform_matrix"], dtype=torch.float32)
			for camera_frame in self.camera_frames
		]


	def __len__(self):
		return len(self.camera_frames)


	def __getitem__(self, idx):
		transform_matrix = self.camera_matrices[idx]
		camera_frame = self.camera_frames[idx]

		img_path = self.root / camera_frame["file_path"]
		com_mask_path = self.root / camera_frame["com_mask_path"]
		obj_mask_path = self.root / camera_frame["obj_mask_path"]

		if img_path.suffix == ".exr":
			rgba = torch.tensor(read_exr(str(img_path))).permute(2, 0, 1)
		else:
			rgba = read_image(str(img_path))
		
		rgb = rgba[:3, ...]
		alpha = rgba[3:, ...]

		if torch.numel(alpha) > 0:
			img = alpha * rgb + (1 - alpha) * self.background
		else:
			img = rgb
		
		# convert img to float32
		if img.dtype == torch.uint8:
			img = img.to(torch.float32) / 255.0
		
		# scale the transform_matrix by self.scale_factor
		transform_matrix = transform_matrix.clone()  # super important!!!!!!
		transform_matrix[:3, 3] *= self.scale_factor

		res = {
			**camera_frame,
			"img": img,
			"img_path": str(img_path),
			"transform_matrix": transform_matrix,
			"image_width": img.shape[-1],
			"image_height": img.shape[-2],
		}

		if self.load_masks:
			res.update({
				# "com_mask": read_image(str(com_mask_path)) > 0,
				"obj_mask": read_image(str(obj_mask_path)) > 0,
			})

		for attr in self.camera_intrinsics:
			res[attr] = self.camera_intrinsics[attr]

		for attr in self.shared_attributes:
			if attr in camera_frame:
				res[attr] = camera_frame[attr]
	

		if "light_direction" in camera_frame:
			light_dir = torch.tensor(camera_frame["light_direction"], dtype=torch.float32)
			res["light_dir"] = light_dir

		if "light_axis_angle" in camera_frame:
			light_axis_angle = torch.tensor(camera_frame["light_axis_angle"], dtype=torch.float32)
			res["light_axis_angle"] = light_axis_angle
		
		if "light" in camera_frame:
			res["light"] = {
				"position": torch.tensor(camera_frame["light"]["position"], dtype=torch.float32) * self.scale_factor,
				"intensity": torch.tensor(camera_frame["light"]["intensity"], dtype=torch.float32) * self.light_intensity_factor,
			}

		return res



class BiGSOlatDataset(torch.utils.data.Dataset):
	subsets: list[BiGSDataset]

	def __init__(self, subsets):
		self.subsets = [s for s in subsets]
	

	def __len__(self):
		return sum(len(s) for s in self.subsets)
	
	def __getitem__(self, idx):
		if isinstance(idx, torch.Tensor) and idx.ndim == 0:
			# tensor doesn't work with cache, so convert it back to int
			idx = idx.item()

		return self.cached_getitem(idx)

	
	@functools.lru_cache(maxsize=512)
	def cached_getitem(self, idx):
		offset = 0

		for s in self.subsets:
			idx_offseted = idx - offset
			if idx_offseted < len(s):
				return s[idx_offseted]
			else:
				offset += len(s)

		raise IndexError(f"idx >= len(self): {idx = }, {len(self) = }")


def get_bigs_olat_dataset(root, light_ids, camera_ids, load_masks=False, *args, **kwargs):
	root = Path(root)
	subset_names = []
	for i in light_ids:
		if isinstance(i, int):
			subset_names.append(f"olat_{i}")
		elif isinstance(i, str):
			if (root / i).exists():
				subset_names.append(i)
			else:
				raise ValueError(f"{root / i} does not exist")

	subset_paths = [root / name for name in subset_names]

	assert root.exists()
	for p in subset_paths:
		assert p.exists(), f"{p} does not exist"
	
	subsets = [
		BiGSDataset(
			p, 
			"train", 
			camera_ids, 
			img_suffix=".png",
			load_masks=load_masks,
			*args,
			**kwargs,
		)
		for p in subset_paths
	]

	return BiGSOlatDataset(subsets)
