import numpy as np

def flip_z(mat):
	r11, r12, r13 = mat[0, :3]
	r21, r22, r23 = mat[1, :3]
	r31, r32, r33 = mat[2, :3]
	t1, t2, t3 = mat[:3, 3]

	return np.array([
		r11, r12, -r13, t1,
		r21, r22, -r23, t2,
		-r31, -r32, r33, -t3,
		0, 0, 0, 1
	]).reshape(4, 4)
