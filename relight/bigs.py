from dataclasses import dataclass
import torch

from nerfstudio.models.splatfacto import SplatfactoModel, SH2RGB

from relight.inverse_sigmoid import inverse_sigmoid_safe as inverse_sigmoid

@dataclass
class BiGS(torch.nn.Module):
	means: torch.nn.Parameter
	quats: torch.nn.Parameter
	scales: torch.nn.Parameter
	opacities: torch.nn.Parameter
	albedos: torch.nn.Parameter
	direct_light_shs: torch.nn.Parameter
	indirect_light_shs: torch.nn.Parameter
	phase_shs_upper: torch.nn.Parameter


	# https://discuss.pytorch.org/t/how-to-use-dataclass-with-pytorch/53444/9
	def __new__(cls, *args, **kwargs):
		inst = super().__new__(cls)
		torch.nn.Module.__init__(inst)
		return inst

	@property
	def n(self):
		return self.means.shape[0]


def init_bigs_from_splats(splats: SplatfactoModel, sh_degree: int, phase_channels: int) -> BiGS:
	"""
	initializing the fields from a splat model, given sh_degree for lighting components, and number of channels used for phase
	"""

	N = splats.means.shape[0]
	device = splats.means.device

	max_sh_degree = sh_degree
	basis_dim = max_sh_degree ** 2

	direct_light_shs = torch.zeros((N, basis_dim, 1), device=device, dtype=splats.means.dtype)

	indirect_light_shs = torch.zeros((N, basis_dim, 3), device=device, dtype=splats.means.dtype)

	phase_shs_upper = torch.zeros((
		N, 
		(basis_dim + 1) * basis_dim // 2, 
		phase_channels), device=device, dtype=splats.means.dtype)

	bigs = BiGS(
		means=torch.nn.Parameter(splats.means.detach().clone()),
		quats=torch.nn.Parameter(splats.quats.detach().clone()),
		scales=torch.nn.Parameter(splats.scales.detach().clone()),
		opacities=torch.nn.Parameter(splats.opacities.detach().clone()),
		albedos=torch.nn.Parameter(inverse_sigmoid(SH2RGB(splats.features_dc))),
		direct_light_shs=torch.nn.Parameter(direct_light_shs),
		indirect_light_shs=torch.nn.Parameter(indirect_light_shs),
		phase_shs_upper=torch.nn.Parameter(phase_shs_upper),
	)

	return bigs


def init_bigs_from_state_dict(state_dict: dict) -> BiGS:
	"""
	initializing the fields from a state dict
	"""

	bigs = BiGS(
		means=torch.nn.Parameter(state_dict["means"]),
		quats=torch.nn.Parameter(state_dict["quats"]),
		scales=torch.nn.Parameter(state_dict["scales"]),
		opacities=torch.nn.Parameter(state_dict["opacities"]),
		albedos=torch.nn.Parameter(state_dict["albedos"]),
		direct_light_shs=torch.nn.Parameter(state_dict["direct_light_shs"]),
		indirect_light_shs=torch.nn.Parameter(state_dict["indirect_light_shs"]),
		phase_shs_upper=torch.nn.Parameter(state_dict["phase_shs_upper"]),
	)

	return bigs
