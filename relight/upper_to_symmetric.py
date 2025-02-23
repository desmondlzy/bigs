import torch

# https://stackoverflow.com/questions/68027886/elegant-way-to-get-a-symmetric-torch-tensor-over-diagonal
def upper_to_symmetric(upper):
	"""
	convert a n-by-n upper triangle matrix to symmetric
	upper: (8n + 1, C) -> (n, n, C)
	"""
	n = int((8 * upper.shape[1] + 1) ** 0.5 / 2 - 0.5)
	indices = torch.triu_indices(n, n)
	symmetric = torch.zeros((upper.shape[0], n, n, 3), device=upper.device, dtype=upper.dtype)
	symmetric[:, indices[0], indices[1]] = upper
	symmetric[:, indices[1], indices[0]] = upper

	return symmetric
