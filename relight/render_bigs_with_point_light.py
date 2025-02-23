import torch
from nerfstudio.cameras.cameras import Cameras

from relight.eval_spherical_harmonics import eval_spherical_harmonics
from relight.bigs import BiGS
from relight.render_splats import render_splats
from relight.constants import C0
from relight.double_shs_single_eval import double_shs_single_eval

def render_bigs_with_point_light(
		camera: Cameras, 
		splats: BiGS, 
		light_intensity: torch.Tensor | float,
		light_pos: torch.Tensor, 
		direct_light_shs: torch.Tensor, 
		phase_shs: torch.Tensor, 
		albedos: torch.Tensor,
		indirect_light_shs: torch.Tensor,
		inverse_square_falloff: bool,
		compute_out_sh: bool = False,
		active_sh_deg: int = None,
		background=None):


	assert len(phase_shs.shape) == 4, f"{phase_shs.shape = }"
	assert albedos.shape == (splats.n, 3), f"{albedos.shape = }"

	light_vectors = splats.means - light_pos
	light_distances = torch.linalg.norm(light_vectors, dim=1, keepdim=True)
	light_dirs = light_vectors / light_distances

	direct_light_sh_deg = int((direct_light_shs.shape[1] ** 0.5))
	direct_active_sh_deg = active_sh_deg if active_sh_deg is not None else direct_light_sh_deg
	direct_light = eval_spherical_harmonics(direct_light_sh_deg , direct_active_sh_deg, light_dirs, direct_light_shs) + 0.5

	if inverse_square_falloff:
		incident_light = direct_light * light_intensity / light_distances ** 2
	else:
		incident_light = direct_light * light_intensity
	
	c2w = camera.camera_to_worlds.to(splats.means.device, torch.float32)
	view_dirs_unnormalized = splats.means - c2w[:3, 3]
	view_dirs = view_dirs_unnormalized / torch.linalg.norm(view_dirs_unnormalized, dim=1, keepdim=True)

	assert view_dirs.shape == (splats.n, 3), f"{view_dirs.shape = }"
	assert view_dirs.device == splats.means.device

	basis_dim = phase_shs.shape[1] 

	assert phase_shs.shape == (splats.n, basis_dim, basis_dim, 3) or phase_shs.shape == (splats.n, basis_dim, basis_dim, 1), f"{phase_shs.shape = }"


	scatter_out_sh = double_shs_single_eval(phase_shs, light_dirs, active_sh_deg=active_sh_deg)

	assert scatter_out_sh.shape == (splats.n, basis_dim, 3) or scatter_out_sh.shape == (splats.n, basis_dim, 1), f"{scatter_out_sh.shape = }"

	phase_sh_deg = int((scatter_out_sh.shape[1] ** 0.5)) 
	phase_sh_active_deg = active_sh_deg if active_sh_deg is not None else phase_sh_deg
	phase = eval_spherical_harmonics(phase_sh_deg, phase_sh_active_deg, view_dirs, scatter_out_sh)
	assert phase.shape == (splats.n, phase_shs.shape[-1]), f"{phase.shape = }"

	indirect_light_sh_deg = int((indirect_light_shs.shape[1] ** 0.5))
	indirect_light_active_sh_deg = active_sh_deg if active_sh_deg is not None else indirect_light_sh_deg
	indirect_light = eval_spherical_harmonics(indirect_light_sh_deg, indirect_light_active_sh_deg, light_dirs, indirect_light_shs) * light_intensity

	if inverse_square_falloff:
		indirect_light = indirect_light / light_distances ** 2


	res = {
		"albedos": albedos,
		"scatters": phase,
		"scatter_out_sh": scatter_out_sh,
		"direct_light": direct_light,
		"incident_light": incident_light,
		"indirect_light": indirect_light,
		"phase_shs": phase_shs,
	}

	if compute_out_sh:
		out_sh = scatter_out_sh.clone() 
		out_sh[:, 0, :] += albedos / C0
		out_sh *= incident_light.unsqueeze(1).clamp(min=0.0)
		out_sh[:, 0, :] += indirect_light.clamp(min=0.0) / C0

		out_image = render_splats(camera, splats, color_override=out_sh, background=background)["rgb"]

		res["out_sh"] = out_sh
		res["image"] = out_image

	else:
		colors = incident_light.clamp(min=0.0) * (phase + albedos).clamp(min=0.0) + indirect_light.clamp(min=0.0)
		out_image = render_splats(camera, splats, color_override=colors, background=background)["rgb"]

		res["image"] = out_image
		res["out_colors"] = colors

	return res