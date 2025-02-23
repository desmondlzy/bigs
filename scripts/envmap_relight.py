import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from relight.make_video import make_video
from relight.dict_to_camera import dict_to_camera
from relight.gamma_correction import gamma_correction
from relight.read_exr import read_exr
from relight.bigs import BiGS, init_bigs_from_state_dict
from relight.render_bigs_with_point_light import render_bigs_with_point_light
from relight.bigs_olat_dataset import get_bigs_olat_dataset
from relight.render_splats import render_splats
from relight.upper_to_symmetric import upper_to_symmetric


@dataclass
class EnvmapRelightArgs:
	dataset_root: Path
	checkpoint_path: Path
	output_path: Path
	envmap: Path = Path(__file__).parent.parent / "data/envmaps/gear-store.exr"


identifier_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

from IPython import get_ipython
in_ipython = get_ipython() is not None
if in_ipython:
	args = EnvmapRelightArgs(
		dataset_root = Path(__file__).parent.parent / 'data/bigs/dragon',
		output_path = Path(__file__).parent / "output" / Path(__file__).stem / "test_point_relight",
		checkpoint_path = Path(__file__).parent / "output" / "train_cli" / "model_20000.pth",
	)
else:
	import tyro
	args = tyro.cli(EnvmapRelightArgs)

args.output_path.mkdir(exist_ok=True, parents=True)

def envmap_lookup(envmap, dirs=None, thetaphis=None, verbose=False):
	if dirs is not None:
		dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True)
		thetas = torch.atan2(dirs[:, 1], dirs[:, 0]) + np.pi # range [0, 2pi]
		phis = torch.acos(dirs[:, 2])  # range [0, pi]
	elif thetaphis is not None:
		thetas = thetaphis[:, 0]
		phis = thetaphis[:, 1]
	

	polars = torch.stack([ thetas / (torch.pi * 2), phis / torch.pi,], dim=1) 
	polars = polars.reshape(1, 1, -1, 2)

	# change polars to [-1, 1] range 
	coordinates = polars * 2 - 1

	# use polars as coordinate and grid_sample to interpolate
	values = torch.nn.functional.grid_sample(
		envmap.permute(2, 0, 1).unsqueeze(0), 
		coordinates.reshape(1, 1, -1, 2), mode="bilinear", align_corners=True)
	
	values = values.squeeze(0).squeeze(1).permute(1, 0)

	if verbose:
		print(f"{thetas[0] = }")
		print(f"{phis[0] = }")
		print(f"{polars[0] = }")
		print(f"{values.shape = }")

	return values, (thetas, phis)


checkpoint = torch.load(args.checkpoint_path)
bigs = init_bigs_from_state_dict(checkpoint["bigs"])
background = torch.tensor([0.0, 0.0, 0.0])
test_light_ids = [i for i in range(41, 99)]
train_light_ids = [i for i in range(1, 41)]
light_intensity_factor = 1.0
scale_factor = 1.0
flip_with = torch.tensor([-1.0, 1.0, -1.0], device="cuda")  # synthetic data generated using mitsuba

train_dataset = get_bigs_olat_dataset(
	root=args.dataset_root,
	light_ids=train_light_ids,
	camera_ids=[0],	
	load_masks=True,
	background=background,
	light_intensity_factor=light_intensity_factor,
	scale_factor=scale_factor,
)
test_dataset = get_bigs_olat_dataset(
	root=args.dataset_root,
	light_ids=test_light_ids,
	camera_ids=None,
	load_masks=True,
	background=background,
	light_intensity_factor=light_intensity_factor,
	scale_factor=scale_factor,
)

training_light_positions = torch.row_stack([
	datapoint["light"]["position"] for datapoint in train_dataset
]).cuda() * torch.tensor([[-1.0, 1.0, -1.0]]).cuda()

# # half sphere
training_light_solidangles = torch.full(
	(training_light_positions.shape[0], ), 
	2 * np.pi / training_light_positions.shape[0], device=bigs.means.device, dtype=bigs.means.dtype).cuda()

envmap_dim = (32, 64)
envmap_original = read_exr(args.envmap)[..., :3]
envmap_name = args.envmap.stem
print("original size: ", envmap_original.shape)
envmap_original_dim = envmap_original.shape
envmap = cv2.resize(envmap_original, (envmap_dim[1], envmap_dim[0]))
print(f"envmap info: {envmap.shape = }, {envmap.dtype = }, {envmap.max() = }")

# # resize to env
gauss_kernel_x = cv2.getGaussianKernel(5, 1)
gauss_kernel_y = cv2.getGaussianKernel(5, 1).T
gauss_kernel = gauss_kernel_x @ gauss_kernel_y
assert 0.99 < gauss_kernel.sum() < 1.01
print(f"before filter2D: {envmap.shape = }, {envmap.dtype = }, {envmap.max() = }")
envmap_padded_x = np.pad(envmap, ((150, 150), (0, 0), (0, 0)), mode="edge")
envmap_padded = np.pad(envmap_padded_x, ((0, 0), (150, 150), (0, 0)), mode="wrap")
print(f"after padding: {envmap_padded.shape = }")
envmap = cv2.filter2D(envmap_padded, -1, gauss_kernel_x, borderType=cv2.BORDER_DEFAULT)
envmap = cv2.filter2D(envmap, -1, gauss_kernel_y, borderType=cv2.BORDER_DEFAULT)[150:-150, 150:-150]
# envmap = np.stack([
# 	convolve2d(envmap[:, :, c], gauss_kernel, boundary="wrap")
# 	for c in range(3)
# ]) 
print(f"after filter2D: {envmap.shape = }, {envmap.dtype = }, {envmap.max() = }")

envmap = torch.flip(torch.tensor(envmap, dtype=torch.float32, device="cuda"), dims=(0, 1))
print(f"envmap info: {envmap.shape = }, {envmap.dtype = }, {envmap.max() = }")
# envmap = torch.roll(envmap, 16, dims=1)

video_frame_dir = args.output_path / f"frames_{envmap_name}"
video_frame_dir.mkdir(exist_ok=True, parents=True)
video_path = args.output_path / f"video_{envmap_name}.mp4"

for i in tqdm(range(envmap.shape[1])):
	light_results = []
	rolled_envmap = torch.roll(envmap, i, dims=1)
	envmap_vals, envmap_coords = envmap_lookup(
		rolled_envmap, training_light_positions)
	assert envmap_vals.shape == (training_light_solidangles.shape[0], 3), f"{envmap_vals.shape = }"

	for index, datapoint in enumerate(train_dataset):
		light = datapoint["light"]
		light_pos = light["position"].cuda() * flip_with
		camera = dict_to_camera(datapoint)

		results = render_bigs_with_point_light(
			camera, 
			bigs, 
			light_pos=light_pos, 
			light_intensity=envmap_vals[index].cuda(), 
			inverse_square_falloff=False,
			direct_light_shs=bigs.direct_light_shs, 
			phase_shs=upper_to_symmetric(bigs.phase_shs_upper),
			albedos=torch.sigmoid(bigs.albedos),
			indirect_light_shs=bigs.indirect_light_shs,
			background=background.cuda(),
		)
		
		light_results.append(results)

	out_sh_sum = sum(result["out_colors"] * training_light_solidangles[i] for i, result in enumerate(light_results))
	out_sh_image = render_splats(camera, bigs, color_override=out_sh_sum, background=background)["rgb"]
	f = plt.figure(figsize=(20, 10))
	plt.subplot(1, 2, 1)
	out_image_gamma = gamma_correction(out_sh_image.clamp(0.0, 1.0)).detach().cpu().numpy()
	plt.imshow(out_image_gamma)
	plt.axis("off")
	plt.subplot(1, 2, 2)
	plt.imshow(gamma_correction(torch.flip(rolled_envmap, dims=(0, 1)).clamp(0.0, 1.0)).cpu().numpy())
	plt.axis("off")
	# plt.show()
	plt.title(f"envmap_{i:04d}")
	plt.savefig(video_frame_dir / f"envmap_render_{i:04d}.png")
	plt.close(f)

	# save the render image
	plt.imsave(video_frame_dir / f"render_{i:04d}.png", out_image_gamma)

	shift_on_original = int(i / envmap_dim[1] * envmap_original_dim[1])

	# sample the original envmap so that every pixel in the original envmap is shifted to the right by shift_on_original
	shifted_original = gamma_correction(np.clip(np.roll(envmap_original, shift_on_original, axis=1), 0.0, 1.0))
	plt.imsave(video_frame_dir / f"rotating_envmap_{i:04d}.png", shifted_original)


make_video(
	video_path,
	[[*sorted(video_frame_dir.glob(f"envmap_render_*.png")), *sorted(video_frame_dir.glob(f"envmap_render_*.png"))]],
	titles=[f"envmap {envmap_name}"],
	fps=15,
)