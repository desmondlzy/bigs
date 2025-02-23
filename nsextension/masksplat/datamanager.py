"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union
import numpy as np
from PIL import Image

import os; os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import torch

from nerfstudio.data.datamanagers.full_images_datamanager import (
    # FullImageDatamanager as FullImageLightDatamanager,
    # FullImageDatamanagerConfig as FullImageLightDatamanagerConfig,
    FullImageDatamanager, FullImageDatamanagerConfig,
)

from nerfstudio.data.datasets.base_dataset import InputDataset


class EXRDataset(InputDataset):
    def get_numpy_image(self, image_idx: int):
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        if image_filename.suffix in (".exr", ".png"):
            image = cv2.imread(str(image_filename), cv2.IMREAD_UNCHANGED)
            assert image is not None, f"Image at {image_filename} could not be read."
            assert len(image.shape) in (3, 4), f"Image shape of {image.shape} is incorrect."

            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) 
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image.dtype == np.uint8:
                image = image.astype(np.float32)
                image[:, :, :3] /= 255.0
            
            gamma_coeff = self._dataparser_outputs.metadata["exr_gamma_coeff"]
            if gamma_coeff > 0:
                image[:, :, :3] = image[:, :, :3].clip(min=1e-6) ** (1 / gamma_coeff)

            if self.scale_factor != 1.0:
                width, height = image.shape[1], image.shape[0]
                newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
                image = cv2.resize(image, newsize, interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Unsupported image format: {image_filename.suffix}")


        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.float32, f"Image dtype of {image.dtype} is incorrect."
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is incorrect."
        return image

    def get_image_float32(self, image_idx: int):
        """Returns a 3 channel image in float32 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """

        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32"))
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])

        return image

    def get_image_uint8(self, image_idx: int):
        """Returns a 3 channel image in uint8 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        raise NotImplementedError("EXR values have to in float32, might beyond 1.0 (255)")


@dataclass
class FullImageLightDatamanagerConfig(FullImageDatamanagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: FullImageLightDatamanager)

    cache_images_type = "float32"



class FullImageLightDatamanager(FullImageDatamanager[EXRDataset]):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: FullImageLightDatamanagerConfig
