from tinycudann import Encoding
import torch

def get_sh_encoder(basis_dim):
	return Encoding(
		n_input_dims=3,
		encoding_config={
			"otype": "SphericalHarmonics",
			"degree": int(basis_dim ** 0.5),
		},
		dtype=torch.float32,
	)
