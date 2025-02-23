from functools import cache
import torch
import tinycudann as tcnn

from relight.envmap_lookup import envmap_lookup


@cache
def get_sh_encoder(degree):
	_sh_encoder = tcnn.Encoding(
		n_input_dims=3,
		encoding_config={
			"otype": "SphericalHarmonics",
			"degree": degree,
		},
		dtype=torch.float32,
	)

	def sh_encoder(dirs):
		"""
		assuming dirs are unit vectors, of shape (N, 3)
		"""
		return _sh_encoder((dirs + 1) / 2)

	return sh_encoder


def envmap_sh_encode(envmap, degree):
	assert degree <= 7
	basis_num = degree ** 2

	n_samples = 10000
	dirs = torch.randn(n_samples, 3, device=envmap.device)
	dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True)

	sample_weights = 4 * torch.pi / n_samples  # solid angles as weights
	envmap_values, (lats, longs) = envmap_lookup(envmap, dirs=dirs)

	sh_encoder = get_sh_encoder(degree)

	dirs_sh_vals = sh_encoder(dirs)
	assert dirs_sh_vals.shape == (n_samples, basis_num)

	projection = torch.zeros((basis_num, 3), device="cuda")

	for s in range(n_samples):
		basis_val = dirs_sh_vals[s].reshape(-1, 1)

		projection += basis_val * envmap_values[s].reshape(1, -1) * sample_weights
	
	assert projection.shape == (basis_num, 3)

	return projection
