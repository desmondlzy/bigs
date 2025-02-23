from pathlib import Path

import matplotlib.pyplot as plt
import torch
from nerfstudio.cameras.cameras import Cameras

from relight.bigs import BiGS
from relight.gamma_correction import gamma_correction
from relight.render_splats import render_splats

def plot_results(results, camera: Cameras, bigs: BiGS, datapoint=None, mute_clip=False, background=None):
	rows = 2
	cols = 0
	show_keys = ["image", "albedos", "scatters", "direct_light", "incident_light", "indirect_light"]

	background = torch.tensor([0.0, 0.0, 0.0]).cuda() if background is None else background

	output_images = {}

	for key in show_keys:
		if key in results:
			cols += 1

	if datapoint is not None:
		cols += 2

	index = 1

	if datapoint is not None:
		ground_truth = gamma_correction(datapoint["img"].permute(1, 2, 0).clamp(0.0, 1.0)).cuda()
		output_images["gt"] = ground_truth
		plt.subplot(rows, cols, index)	
		plt.title(f"gt ({Path(datapoint['img_path']).name})")
		if mute_clip:
			plt.imshow(torch.clamp(ground_truth, 0.0, 1.0).detach().cpu().numpy())
		else:
			plt.imshow(ground_truth.detach().cpu().numpy())
		plt.axis("off")

		index += 1

	if "image" in results:
		plt.subplot(rows, cols, index)
		render = gamma_correction(results["image"].clamp(0.0, 1.0))
		output_images["render"] = render
		render_min, render_max = float(results["image"].min()), float(results["image"].max())
		title_string = f"render ({render_min:.1f}, {render_max:.1f})"

		if datapoint is not None:
			render_tonemapped = gamma_correction(results["image"].clamp(0.0, 1.0))
			gt_tonemapped = gamma_correction(datapoint["img"].permute(1, 2, 0).cuda().clamp(0.0, 1.0))
			mse = (render_tonemapped - gt_tonemapped).pow(2).mean()
			psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
			title_string += f" psnr={psnr:.2f}"

		plt.title(title_string)
		if mute_clip:
			plt.imshow(torch.clamp(render, 0.0, 1.0).detach().cpu().numpy())
		else:
			plt.imshow(render.detach().cpu().numpy())
		plt.axis("off")
		index += 1

		if datapoint is not None:
			abs_err = torch.abs(ground_truth - render)
			plt.subplot(rows, cols, index)
			plt.title(f"abs_err ({abs_err.min():.2f}, {abs_err.max():.2f}), {abs_err.mean():.2f}")
			abs_err = abs_err / abs_err.max()
			plt.imshow(abs_err.detach().cpu().numpy())
			plt.axis("off")
			index += 1


	for key in ["albedos", "direct_light", "incident_light", "scatters", "indirect_light"]:
		if key in results:
			vals = results[key]
			vals_min, vals_max = float(vals.min()), float(vals.max())

			if vals.shape == (bigs.n, 3):
				vals = vals.clamp(min=0.0)

			vals_render = render_splats(camera, bigs, color_override=vals, background=background.cuda())["rgb"]
			vals_mean = float(vals.mean())
			plt.subplot(rows, cols, index)
			title = f"{key} ({vals_min:.2f}, {vals_max:.2f}, {vals_mean:.2f})"
			if key == "incident_light":
				title += f":{tuple(bigs.direct_light_shs.shape[1:])}"
			if key == "scatters":
				title += f":{tuple(bigs.phase_shs_upper.shape[1:])}"
				
			plt.title(title.replace(" ", ""))

			vals_render = gamma_correction(torch.clamp(vals_render, 0.0, 1.0))

			output_images[key] = vals_render
			
			plt.imshow(vals_render.clamp(min=0.0).detach().cpu().numpy())
			plt.axis("off")

			# render the second row
			if key == "albedos" and "incident_light" in results:
				incident_lights = results["incident_light"]
				diffuse = vals * incident_lights.mean()
				diffuse_render = render_splats(camera, bigs, color_override=diffuse, background=background.cuda())["rgb"]
				diffuse_render = gamma_correction(diffuse_render.clamp(0.0, 1.0))

				plt.subplot(rows, cols, index + cols)
				plt.title(f"diffuse")
				plt.imshow(diffuse_render.detach().cpu().numpy())
				plt.axis("off")

				output_images["diffuse"] = diffuse_render
			
			if key == "scatters" and "incident_light" in results:
				incident_lights = results["incident_light"]
				directional = vals.clamp(min=0.0) * incident_lights.clamp(min=0.0)
				directional_render = render_splats(camera, bigs, color_override=directional, background=background.cuda())["rgb"]
				directional_render = gamma_correction(directional_render.clamp(0.0, 1.0))

				plt.subplot(rows, cols, index + cols)
				plt.title(f"directional")
				plt.imshow(directional_render.detach().cpu().numpy())
				plt.axis("off")

				output_images["directional"] = directional_render
			
			index += 1

	plt.tight_layout()

	return output_images
