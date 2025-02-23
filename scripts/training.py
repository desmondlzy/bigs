#%%
import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pathlib import Path
from dataclasses import dataclass
import json

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from relight.ssim import l1_loss, ssim
from relight.bigs import BiGS, init_bigs_from_splats
from relight.gamma_correction import gamma_correction 
from relight.bigs_olat_dataset import (
	get_bigs_olat_dataset)
from relight.get_sh_encoder import get_sh_encoder
from relight.dict_to_camera import dict_to_camera as make_camera
from relight.upper_to_symmetric import upper_to_symmetric
from relight.render_splats import render_splats
from relight.render_bigs_with_point_light import render_bigs_with_point_light
from relight.plot_results import plot_results

from datetime import datetime
print(f"current time: {datetime.now().isoformat()}")

#%%
from nsextension.masksplat.mask_splat import MaskSplats, MaskSplats
from nerfstudio.utils.eval_utils import eval_setup

from IPython import get_ipython
in_ipython = get_ipython() is not None

@dataclass
class TrainingArgs:
	dataset_root: Path
	ns_gaussian_config: Path
	output_path: Path

	phase_channels: int = 3
	direct_light_nonneg_coeff: float = 1e-1
	scatter_nonneg_coeff: float = 1e-1
	unitnorm_sh_coeff: float = 1e-1
	inverse_square_falloff: bool = True
	phase_lr: float = 1e-4
	opacity_lr: float = 1e-3
	albedo_lr: float = 1e-3
	n_iters: int = 100000
	indirect_light_lr: float = 1e-3
	extra_regularization: bool = True
	sh_degree: int = 5
	autoload_checkpoint = False


if in_ipython:
	args = TrainingArgs(
		dataset_root = Path(__file__).parent.parent / 'data/bigs/dragon',
		ns_gaussian_config = Path('outputs/dragon/relightable-splat/2025-02-18_085749/config.yml'),
		output_path = Path(__file__).parent / "output" / Path(__file__).stem / "test_refactoring",
	)
else:
	import tyro
	args = tyro.cli(TrainingArgs)

dataset_name = args.dataset_root.stem

background = torch.tensor([0.0, 0.0, 0.0])

args.output_path.mkdir(exist_ok=True, parents=True)

sh_encoder = get_sh_encoder(args.sh_degree ** 2)

identifier_str = datetime.now().isoformat()

print(f"Using output path: {args.output_path}, dataset root: {args.dataset_root}")

def load_splats_model(nerfstudio_config_path):
	os.chdir(Path(__file__).parent.parent)  # the checkpoint path in the nerfstudio config is relative to where ns-train is called
	config_path = Path(nerfstudio_config_path)  # gaussian-splat + alpha = 1
	ns_config, pipeline, checkpoint_path, step = eval_setup(config_path)

	splats: MaskSplats = pipeline.model

	return pipeline, splats

#%%
pipeline, pretrained_ns_splats = load_splats_model(Path(__file__).parent.parent / args.ns_gaussian_config)

test_light_ids = [i for i in range(41, 99)]
train_light_ids = [i for i in range(1, 41)]
light_intensity_factor = 1.0
scale_factor = 1.0
flip_with = torch.tensor([-1.0, 1.0, -1.0], device="cuda")  # synthetic data generated using mitsuba

#%%
train_dataset = get_bigs_olat_dataset(
	root=args.dataset_root,
	light_ids=train_light_ids, 
	# light_ids=[i for i in range(10, 11)], 
	# camera_ids=[0],
	background=background,
	camera_ids=None,
	load_masks=True,
	light_intensity_factor=light_intensity_factor,
	scale_factor=scale_factor,
)

#%%
active_sh_degree = 1
max_sh_degree = args.sh_degree

bigs = init_bigs_from_splats(splats=pretrained_ns_splats, sh_degree=args.sh_degree, phase_channels=3)
basis_dim = bigs.direct_light_shs.shape[1]


def compute_loss(results, datapoint):
	render = gamma_correction(results["image"].clamp(min=0.00000001, max=1.0))
	ground_truth = gamma_correction(datapoint["img"].permute(1, 2, 0).cuda().clamp(0.0, 1.0))

	assert torch.isfinite(render).all(), "render has non-finite values"

	l1_loss_term = l1_loss(ground_truth, render)
	ssim_loss_term = 1 - ssim(ground_truth, render)
	loss_dict = {
		"photometric": 0.8 * l1_loss_term,
		"ssim": 0.2 * ssim_loss_term,
	}

	loss_dict_without_coeff = loss_dict.copy()

	if args.extra_regularization:
		n = results["phase_shs"].shape[0]
		random_out_dirs = torch.randn((n, 3), device=results["phase_shs"].device, dtype=results["phase_shs"].dtype)
		random_in_dirs = torch.randn_like(random_out_dirs)

		random_out_dirs /= torch.linalg.norm(random_out_dirs, dim=1, keepdim=True)
		random_in_dirs /= torch.linalg.norm(random_in_dirs, dim=1, keepdim=True)

		in_dir_vals = sh_encoder((random_in_dirs + 1) / 2)
		out_dir_vals = sh_encoder((random_out_dirs + 1) / 2)

		shs_sym = results["phase_shs"]
		out_sh = torch.einsum("ijkc,ij->ikc", shs_sym, in_dir_vals)
		out_vals = torch.einsum("ikc,ik->ic", out_sh, out_dir_vals)


	# penalize negative values in scatters
	phase = results["scatters"] 
	direct_light = results["direct_light"]

	loss_dict_without_coeff["scatter_nonneg"] = (
		  torch.relu(-phase).mean() 
	)

	loss_dict_without_coeff["direct_light_nonneg"] = (
		torch.relu(-direct_light).mean()
	)

	if args.extra_regularization:
		loss_dict_without_coeff["scatter_nonneg"] += torch.relu(-out_vals).mean()

	loss_dict["direct_light_nonneg"] = args.direct_light_nonneg_coeff * loss_dict_without_coeff["direct_light_nonneg"]
	loss_dict["scatter_nonneg"] = args.scatter_nonneg_coeff * loss_dict_without_coeff["scatter_nonneg"]

	
	scatter_out_sh = results["scatter_out_sh"]
	scatter_out_integral = (scatter_out_sh[:, 0, :]) * np.sqrt(4 * np.pi)
	assert scatter_out_integral.shape in ((bigs.n, 3), (bigs.n, 1)), f"{scatter_out_integral.shape = }"

	loss_dict_without_coeff["unitnorm"] = (
		((torch.relu(scatter_out_integral - 1) + torch.relu(-scatter_out_integral)) ** 2).mean()
	)

	if args.extra_regularization:
		out_sh_integral = out_sh[:, 0, :] * np.sqrt(4 * np.pi)
		loss_dict_without_coeff["unitnorm"] += ((torch.relu(out_sh_integral - 1) + torch.relu(-out_sh_integral)) ** 2).mean()
		

	loss_dict["unitnorm"] = args.unitnorm_sh_coeff * loss_dict_without_coeff["unitnorm"]
		
	opacity_render = render_splats(camera, bigs, color_override=torch.sigmoid(bigs.opacities), background=torch.tensor([0.0, 0.0, 0.0], device="cuda"))["rgb"]
	outside_mask = torch.logical_not(datapoint["obj_mask"].cuda()).squeeze(0).unsqueeze(-1)
	outside_mass = opacity_render * outside_mask
	loss_dict_without_coeff["outside"] = outside_mass.sum()

	loss_dict["outside"] = loss_dict_without_coeff["outside"]


	return loss_dict, loss_dict_without_coeff


#%%
torch.random.manual_seed(0)
n_iters = 100000 if in_ipython else args.n_iters
n_iters += 1
iter_indirect_start = n_iters - 30000
save_iters = [i for i in range(0, n_iters, 5000)]
if save_iters[-1] != n_iters - 1:
	save_iters.append(n_iters - 1)

render_iters = [100000]

print("save iters", save_iters)
print("render iters", render_iters)


random_indices = torch.randint(0, len(train_dataset), (n_iters,))
# optimizer = torch.optim.Adam([direct_light_net], lr=1e-3)
optimizer = torch.optim.Adam([
	{"params": bigs.direct_light_shs, "lr": 1e-3},
	{"params": bigs.phase_shs_upper, "lr": args.phase_lr},

	{"params": bigs.albedos, "lr": args.albedo_lr},
	{"params": bigs.opacities, "lr": args.opacity_lr},
	{"params": bigs.means, "lr": 1e-5},
	{"params": bigs.quats, "lr": 1e-5},
	{"params": bigs.scales, "lr": 1e-5},
])

indirect_light_optimizer = torch.optim.Adam([
	{"params": bigs.indirect_light_shs, "lr": args.indirect_light_lr},
])

#%%
all_checkpoints = list(args.output_path.glob("model_*.pth"))
if args.autoload_checkpoint and len(all_checkpoints) > 0:
	latest_checkpoint = max(all_checkpoints, key=lambda x: int(x.stem.split("_")[-1]))

	print(f"found {len(all_checkpoints)} checkpoint files, loading {latest_checkpoint.stem}")

	checkpoint_iter = int(latest_checkpoint.stem.split("_")[-1])
	checkpoint = torch.load(latest_checkpoint)

	bigs.load_state_dict(checkpoint["bigs"])
	optimizer.load_state_dict(checkpoint["optimizer"])
	indirect_light_optimizer.load_state_dict(checkpoint["indirect_light_optimizer"])
	checkpoint_iter = checkpoint["it"]

	print("Loaded checkpoint", latest_checkpoint)
	print("Starting from iteration: ", checkpoint_iter)
else:
	checkpoint_iter = 0
	print("starting new training session")


print("total iterations: ", n_iters)

for it in tqdm(range(checkpoint_iter, n_iters), initial=checkpoint_iter, total=n_iters):
	datapoint = train_dataset[random_indices[it]]
	camera = make_camera(datapoint)

	ground_truth = gamma_correction(datapoint["img"].permute(1, 2, 0).cuda())
	# ground_truth = datapoint["img"][:3, :, :].permute(1, 2, 0).cuda() 
	light = datapoint["light"]
	light_pos = light["position"].cuda() * torch.tensor([-1.0, 1.0, -1.0], device="cuda")

	if it % 2000 == 0 and it != 0 and active_sh_degree < max_sh_degree:
		active_sh_degree = min(active_sh_degree + 1, max_sh_degree)
		print("increase active_sh_degree to", active_sh_degree, flush=True)

	results = render_bigs_with_point_light(
		camera, 
		bigs, 
		light_pos=light_pos.cuda(), 
		light_intensity=light["intensity"].cuda(), 
		inverse_square_falloff=args.inverse_square_falloff,
		direct_light_shs=bigs.direct_light_shs, 
		phase_shs=upper_to_symmetric(bigs.phase_shs_upper),
		albedos=torch.sigmoid(bigs.albedos),
		indirect_light_shs=bigs.indirect_light_shs,
		background=background.cuda(),
		active_sh_deg=active_sh_degree,
		compute_out_sh=False)
	
	render = gamma_correction(results["image"].clamp(min=0.000001))

	assert torch.isfinite(render).all(), "render has non-finite values"

	indirect_light_optimizer.zero_grad()
	optimizer.zero_grad()

	loss_dict, loss_dict_without_coeff = compute_loss(results, datapoint)
	loss = sum(loss_dict.values())
	loss.backward()

	optimizer.step()
	if it >= iter_indirect_start:
		indirect_light_optimizer.step()


	if it % 1000 == 0:
		with torch.no_grad():
			f = plt.figure(figsize=(30, 10))
			plot_results(results, camera, bigs, datapoint, mute_clip=True, background=background)
			plt.tight_layout()
			if in_ipython:
				plt.show()
			else:
				train_img_dir = args.output_path / f"train_img"
				train_img_dir.mkdir(exist_ok=True)
				plt.savefig(str(train_img_dir / f"img_{it:06d}.png"))

			plt.close(f)
		
		loss_string = "\n".join([f"{k}: {v.item():.2e}" for k, v in loss_dict_without_coeff.items()])
		
		print(f"[{datetime.now().isoformat()}] iter: {it}, loss_dict: {loss_string}", flush=True)	
	
	if it in save_iters: 
		# check if all tensors are finite
		assert torch.all(torch.isfinite(bigs.direct_light_shs)), "direct_light_shs has non-finite values"
		assert torch.all(torch.isfinite(bigs.phase_shs_upper)), "phase_shs_upper has non-finite values"
		assert torch.all(torch.isfinite(bigs.albedos)), "_albedos has non-finite values"

		if not in_ipython:
			print(f"[{datetime.now().isoformat()}] iter: {it}, save model, all tensors are finite", flush=True)

			loss_string = "\n".join([f"{k}: {v.item():.2e}" for k, v in loss_dict_without_coeff.items()])
			print(f"[{datetime.now().isoformat()}] iter: {it}, loss_dict: {loss_string}", flush=True)	

		torch.save({
			"bigs": bigs.state_dict(),
			"optimizer": optimizer.state_dict(),
			"indirect_light_optimizer": indirect_light_optimizer.state_dict(),
			"it": it,
		}, args.output_path / f"model_{it:06d}.pth")

		print(f"{it = }: save checkpoint to {args.output_path / f'model_{it:06d}.pth'}", flush=True)

print("training done")
