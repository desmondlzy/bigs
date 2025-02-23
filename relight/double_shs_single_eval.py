from relight.eval_spherical_harmonics import eval_spherical_harmonics

def double_shs_single_eval(double_shs, dirs, active_sh_deg=None):
	n = double_shs.shape[0]
	basis_dim = double_shs.shape[1]
	channels = double_shs.shape[-1]
	deg = int(basis_dim ** 0.5)
	active_sh_deg = deg if active_sh_deg is None else active_sh_deg

	smashed_shs = double_shs.view(n, basis_dim, basis_dim * channels)
	single_shs = eval_spherical_harmonics(deg, active_sh_deg, dirs, smashed_shs).reshape(n, -1, 3)
	return single_shs
