import torch

def envmap_lookup(envmap, dirs=None, latlongs=None):
	"""
	envmap: torch.Tensor, shape (H, W, 3)
	dirs: torch.Tensor, shape (N, 3)
	latlongs: torch.Tensor, shape (N, 2)

	returns:
		values: torch.Tensor, shape (N, 3)
		lats: torch.Tensor, shape (N,) range [0, 2pi]
		longs: torch.Tensor, shape (N, ) range [0, pi]
	"""
	if dirs is not None:
		dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True)
		lats = torch.atan2(dirs[:, 1], dirs[:, 0]) # range [-pi, pi]
		lats = torch.where(lats < 0, lats + 2 * torch.pi, lats) # range [0, 2pi]
		longs = torch.acos(dirs[:, 2])  # range [0, pi]
	elif latlongs is not None:
		lats = latlongs[:, 0]
		longs = latlongs[:, 1]
	
	polars = torch.stack([ lats / (torch.pi * 2), longs / torch.pi,], dim=1) 
	polars = polars.reshape(1, 1, -1, 2)

	# change polars to [-1, 1] range 
	coordinates = polars * 2 - 1

	# use polars as coordinate and grid_sample to interpolate
	values = torch.nn.functional.grid_sample(
		envmap.permute(2, 0, 1).unsqueeze(0), 
		coordinates.reshape(1, 1, -1, 2), mode="bilinear", align_corners=True)
	
	values = values.squeeze(0).squeeze(1).permute(1, 0)

	return values, (lats, longs)