import os; os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

def read_exr(image_path):
	img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
	if img.shape[2] == 4:
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
	else:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	return img

