from PIL import Image, ImageDraw, ImageFont, ImageFile
import imageio
import numpy as np


def make_video(save_path, stream_paths, titles=None, rows=None, cols=None, fps=None, use_tqdm=True):
	num_streams = len(stream_paths)
	num_frames = len(stream_paths[0])

	if not use_tqdm:
		def tqdm(iterable, *args, **kwargs):
			return iterable
	else:
		from tqdm import tqdm

	# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
	ImageFile.LOAD_TRUNCATED_IMAGES = True

	titles = titles or ["" for _ in range(num_streams)]

	fps = fps or num_frames // 10  # default to 10 seconds

	assert all([len(paths) == num_frames for paths in stream_paths]), f"{[len(paths) for paths in stream_paths]}"

	rows = rows or int(np.ceil(np.sqrt(num_streams)))
	cols = cols or num_streams // rows + int(num_streams % rows > 0)

	assert rows * cols >= num_streams, f"not enough space for {num_streams} streams with {rows} rows and {cols} cols"

	print(f"using {num_streams} streams with {num_frames} frames each, fps={fps}")
	print(f"saving to {save_path}")
	print(f"layout: rows={rows}, cols={cols}")

	# read first frames of each stream
	first_frames = [imageio.v2.imread(paths[0]) for paths in stream_paths]
	heights = [frame.shape[0] for frame in first_frames]
	widths = [frame.shape[1] for frame in first_frames]
	max_image_height, max_image_width = max(heights), max(widths)
	title_height = int(0.1 * max_image_height)
	max_height = max_image_height + title_height
	total_height, total_width = rows * (max_height + title_height), cols * max_image_width

	writer = imageio.get_writer(save_path, fps=fps)

	for paths in tqdm(zip(*stream_paths), total=num_frames, desc="Writing video"):
		images = [imageio.v2.imread(path)[:, :, :3] for path in paths]

		im = np.zeros((total_height, total_width, 3), dtype=np.uint8)
		for i, image in enumerate(images):
			row, col = divmod(i, cols)

			h_start = row * (max_height)
			h_title = h_start + title_height
			h_end = h_title + max_image_height
			c_start = col * max_image_width
			c_end = c_start + max_image_width

			# put text on the upper left corner of the image
			text = titles[i]
			pil_image = Image.fromarray(im[h_start: h_title, c_start: c_end])
			font = ImageFont.load_default(size=int(title_height * 0.7))
			draw = ImageDraw.Draw(pil_image)
			draw.text((10, 10), text, (255, 255, 255), font=font)

			im[h_start: h_title, c_start: c_end] = np.array(pil_image)
			im[h_title: h_title + image.shape[0], c_start: c_start + image.shape[1]] = image


		writer.append_data(im)
	
	writer.close()
	print("write to: ", save_path)
