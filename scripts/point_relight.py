#%%
import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import datetime
from pathlib import Path
from dataclasses import dataclass
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from relight.bigs_olat_dataset import get_bigs_olat_dataset
from relight.gamma_correction import gamma_correction
from relight.dict_to_camera import dict_to_camera
from relight.render_bigs_with_point_light import render_bigs_with_point_light
from relight.upper_to_symmetric import upper_to_symmetric
from relight.bigs import BiGS, init_bigs_from_state_dict
from relight.ssim import ssim
from relight.plot_results import plot_results
from relight.make_video import make_video
from load_pretrained_and_set_args import load_pretrained_and_set_args



#%%
"""
evaluate on test set
"""

@dataclass
class PointRelightArgs:
	output_path: Path
	dataset_root: Path = Path(__file__).parent.parent / "data/bigs/dragon"
	checkpoint_path: Path = Path("")
	use_pretrained: str | None = None


identifier_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()

from IPython import get_ipython
in_ipython = get_ipython() is not None
if in_ipython:
	args = PointRelightArgs(
		dataset_root = Path(__file__).parent.parent / 'data/bigs/dragon',
		output_path = Path(__file__).parent / "output" / Path(__file__).stem / "test_point_relight",
		checkpoint_path = Path(__file__).parent / "output" / "train_cli" / "model_20000.pth",
	)
else:
	import tyro
	args = tyro.cli(PointRelightArgs)

if args.use_pretrained:
	load_pretrained_and_set_args(args)

args.output_path.mkdir(exist_ok=True, parents=True)


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

# load the model
checkpoint = torch.load(args.checkpoint_path)
bigs = init_bigs_from_state_dict(checkpoint["bigs"])
it = checkpoint["it"]

active_sh_degree = int(bigs.direct_light_shs.shape[-2] ** 0.5)

out_img_dir = args.output_path	/ "images"
out_img_dir.mkdir(exist_ok=True)

for nm, dataset in [("train", train_dataset), ("test", test_dataset)]:
	video_path = args.output_path / f"{identifier_str}_{nm}view.mp4"
	if video_path.exists():
		print(f"video for {nm} already exists at {video_path}, skip")
		continue
	else:
		print(f"video for {nm} does not exist, start rendering")

	psnr_items = []
	loss_items = []
	lpip_items = []
	ssim_items = []
	nonneg_items = []
	unitnorm_items = []

	for index, datapoint in tqdm(enumerate(dataset), total=len(dataset)):
		light = datapoint["light"]
		light_pos = light["position"].cuda() * flip_with
		camera = dict_to_camera(datapoint)

		results = render_bigs_with_point_light(
			camera, 
			bigs, 
			light_pos=light_pos, 
			light_intensity=light["intensity"].cuda(), 
			inverse_square_falloff=True,
			direct_light_shs=bigs.direct_light_shs, 
			phase_shs=upper_to_symmetric(bigs.phase_shs_upper),
			albedos=torch.sigmoid(bigs.albedos),
			indirect_light_shs=bigs.indirect_light_shs,
			background=background.cuda(),
			active_sh_deg=active_sh_degree,
			compute_out_sh=False
		)

		render = gamma_correction(results["image"].clamp(min=0.0, max=1.0))
		ground_truth = gamma_correction(datapoint["img"].permute(1, 2, 0).cuda().clamp(min=0.0, max=1.0))

		mse = ((render - ground_truth) ** 2).mean()
		assert mse >= 0, f"{mse = }"

		psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse))
		psnr_items += [psnr_val.detach().cpu().numpy()]

		ssim_items += [ssim(ground_truth, render).detach().cpu().numpy()]
		lpip_items += [lpips(
			ground_truth.permute(2, 0, 1).unsqueeze(0), 
			render.permute(2, 0, 1).unsqueeze(0)).detach().cpu().numpy()]

		# penalize negative values in scatters

		f = plt.figure(figsize=(20, 5))
		plot_results(results, camera, bigs, datapoint, mute_clip=True)
		plt.savefig(out_img_dir / f"{nm}view_{index:06d}.png")
		plt.close(f)

	psnr_val = np.mean(psnr_items)
	ssim_val = np.mean(ssim_items)
	lpip_val = np.mean(lpip_items)

	print("split:", nm)
	print("psnr:", psnr_val)
	print("ssim:", ssim_val)
	print("lpip:", lpip_val)

	# write metrics to json
	metrics = {
		"psnr": float(psnr_val),
		"ssim": float(ssim_val),
		"lpip": float(lpip_val),
	}

	with open(args.output_path / f"{identifier_str}_{nm}_metrics.json", "w") as f:
		json.dump(metrics, f)

	# write video (need to install ffmpeg and ffmpeg-python, otherwise codec can't be found)
	fps = 12 if nm == "test" else 4
	make_video(
		args.output_path / f"{identifier_str}_it{it}_{nm}view.mp4",
		[sorted(out_img_dir.glob(f"{nm}view_*.png"))],
		titles=[f"it={it} PSNR={psnr_val:.2f} SSIM={ssim_val:.2f} LPIP={lpip_val:.2f} {identifier_str}"],
		fps=fps,
		use_tqdm=False,
	)

print("point light relight videos done!")
