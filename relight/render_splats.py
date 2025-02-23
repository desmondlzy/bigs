import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from relight.eval_spherical_harmonics import eval_spherical_harmonics
from relight.bigs import BiGS

def render_splats(camera: Cameras, gaussians: BiGS | SplatfactoModel, opacity_override=None, color_override=None, background=None):
	"""
	a wrapper around gsplat.rasterization() for some common parameters
	can rasterize gaussians and splatfacto
	"""
	background = torch.ones((3, ), device=gaussians.means.device, dtype=gaussians.means.dtype) if background is None else background.cuda()

	c2w = camera.camera_to_worlds.to(gaussians.means.device, torch.float32)

	R = c2w[:3, :3]
	t = c2w[:3, 3:4]

	w2c = torch.eye(4, device=gaussians.means.device, dtype=torch.float32)
	R_edit = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=gaussians.means.device, dtype=torch.float32))
	R = R @ R_edit
	R_inv = R.T
	t_inv = -R_inv @ t
	w2c[:3, :3] = R_inv
	w2c[:3, 3:4] = t_inv

	W, H = int(camera.width.item()), int(camera.image_height.item())

	n_gaus = gaussians.means.shape[0]
	opacities_sigmoid = torch.sigmoid(gaussians.opacities) if opacity_override is None else torch.ones((n_gaus, 1), device=gaussians.means.device, dtype=gaussians.means.dtype) * opacity_override
	opacities_sigmoid = opacities_sigmoid.flatten()

	viewdirs = gaussians.means - c2w[:3, 3]
	if color_override is None:
		rgbs = spherical_harmonics(3, viewdirs, gaussians.shs)  # input unnormalized viewdirs
		rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
	else:
		if color_override.shape == gaussians.means.shape:
			rgbs = color_override
		elif color_override.shape[1] in [9, 16, 25, 36, 49, 64]:
			sh_deg = int(color_override.shape[1] ** 0.5)
			rgbs = eval_spherical_harmonics(sh_deg, sh_deg, viewdirs, color_override)
		elif torch.numel(color_override) * 3 == torch.numel(gaussians.means):
			c = color_override.view(-1)
			rgbs = torch.column_stack([c, c, c])
		else:
			raise ValueError(f"color_override has incompatible shape: {color_override.shape}, means shape: {gaussians.means.shape}")

	out_image, alpha, _ = rasterization(
		gaussians.means,
		gaussians.quats / torch.norm(gaussians.quats, dim=1, keepdim=True),
		torch.exp(gaussians.scales),
		opacities_sigmoid,
		rgbs,
		viewmats=w2c.unsqueeze(0),
		Ks=camera.get_intrinsics_matrices().unsqueeze(0).cuda(),
		width=W,
		height=H,
	)


	return {"rgb": out_image[0], "alpha": alpha[0]}
