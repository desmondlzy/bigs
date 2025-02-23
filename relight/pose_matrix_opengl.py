import torch

def pose_matrix_look_at_opengl(eye, look_at, up=None, mat_height=4):
	"""
	given camera position (eye), and look_dir,
	return the 4x4 pose matrix (camera to world) transform
	*Use OpenGL/Blender Convention
	"""
	look_dir = look_at / torch.norm(look_at)

	if up != None:
		assert up.shape == (3, ), "up must be a 3D vector"
		assert torch.linalg.norm(up) > 0, "up must be a non-zero vector"
		assert torch.dot(look_dir, up) == 0, "look_dir and up must be orthogonal"
	else:
		# arbitrary select an up vector orthogonal to look_dir
		up = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0], device=eye.device, dtype=eye.dtype), look_dir) if look_dir[1] != 1 else torch.tensor([1.0, 0.0, 0.0], device=eye.device, dtype=eye.dtype)

	y = up
	z = -look_dir
	right = torch.linalg.cross(y, z)  # right-handed coordinate system

	# the camera to world matrix
	c2w = torch.eye(4, device=eye.device, dtype=eye.dtype)

	c2w[:3, 0] = right  # x axis
	c2w[:3, 1] = up  # y axis
	c2w[:3, 2] = -look_dir  # z axis 
	c2w[:3, 3] = eye

	return c2w[:mat_height, :]
