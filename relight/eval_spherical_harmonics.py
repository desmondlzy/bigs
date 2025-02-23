import torch
from relight.c import _relight_impl

def eval_spherical_harmonics(degree: int, degree_to_use: int, dirs: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
	return SphericalHarmonics.apply(degree, degree_to_use, dirs, coeffs)

class SphericalHarmonics(torch.autograd.Function):
	@staticmethod
	def forward(
		ctx, 
		degree: int,
		degree_to_use: int,
		dirs: torch.Tensor,
		coeffs: torch.Tensor,
	):
		assert 1 <= degree <= 8
		assert 1 <= degree_to_use <= degree

		# save degree and degree_to_use for backward pass
		ctx.degree = degree
		ctx.degree_to_use = degree_to_use
		ctx.save_for_backward(dirs)

		num_points = dirs.shape[0]
		num_channels = coeffs.shape[2]

		assert coeffs.shape[0] == num_points
		assert coeffs.shape[1] == degree ** 2 
		assert coeffs.ndim == 3

		ret = _relight_impl.eval_sh_forward(degree, degree_to_use, dirs, coeffs)

		assert ret.shape[0] == num_points
		assert ret.shape[1] == num_channels

		return ret

	@staticmethod
	def backward(ctx, dLdy: torch.Tensor):
		dirs = ctx.saved_tensors[0]

		num_points = dirs.shape[0]

		dL_dcoeffs = _relight_impl.eval_sh_backward(ctx.degree, ctx.degree_to_use, dirs, dLdy)

		assert dL_dcoeffs.shape[0] == num_points
		assert dL_dcoeffs.shape[1] == ctx.degree ** 2

		return None, None, None, dL_dcoeffs
	