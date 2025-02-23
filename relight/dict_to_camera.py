from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.camera_utils import get_distortion_params

def dict_to_camera(datapoint):
	camera_to_world = datapoint["transform_matrix"][:3].unsqueeze(0)

	distort = get_distortion_params(
		k1=float(datapoint["k1"]) if "k1" in datapoint else 0.0,
		k2=float(datapoint["k2"]) if "k2" in datapoint else 0.0,
		k3=float(datapoint["k3"]) if "k3" in datapoint else 0.0,
		k4=float(datapoint["k4"]) if "k4" in datapoint else 0.0,
		p1=float(datapoint["p1"]) if "p1" in datapoint else 0.0,
		p2=float(datapoint["p2"]) if "p2" in datapoint else 0.0,
	)


	camera = Cameras(
		camera_to_worlds=camera_to_world.float(),
		fx=datapoint["fl_x"],
		fy=datapoint["fl_y"],
		cx=datapoint["cx"],
		cy=datapoint["cy"],
		width=datapoint["img"].shape[-1],
		height=datapoint["img"].shape[-2],
		distortion_params=distort,
	)[0]

	return camera
