"""
"""
from dataclasses import dataclass, field
from typing import Dict, Type, Tuple

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig
from torch._tensor import Tensor  # for custom Model

import torch


@dataclass
class MaskSplatConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: MaskSplat)

    use_scale_regularization: bool = True

    opacity_regularization_weights: float = 0.0
    alpha_regularization_weights: float = 0.01




class MaskSplat(SplatfactoModel):
    """A modified Splatfacto methods, with addtional parameters for removing floaters using mask.
    """

    config: MaskSplatConfig

    def populate_modules(self):
        super().populate_modules()


    def get_outputs(self, camera: Cameras) -> Dict[str, Tensor]:
        outputs = super().get_outputs(camera)
        return outputs


    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if "mask" in batch:
            mask = batch["mask"]
            mask = self._downscale_if_required(batch["mask"]) 

            # _downscale_if_required cast the mask to float, so cast back
            if mask.dtype == torch.float32:
                mask = torch.isclose(mask, torch.tensor(1.0, device=mask.device))

            # we only wanna enforce the transparent part has no gaussian coverage
            pred_alpha = outputs["accumulation"]
            gt_alpha = torch.where(mask.to(outputs["accumulation"].device), pred_alpha, 0.0)

            alpha_diff = torch.abs(pred_alpha - gt_alpha)
            alpha_reg = self.config.alpha_regularization_weights * alpha_diff.mean()

            opacity_reg = self.config.opacity_regularization_weights * (torch.abs(torch.sigmoid(self.opacities) - 0.9).mean())

            return {
                **loss_dict, 
                "alpha_reg": alpha_reg,
                "opacity_reg": opacity_reg,
            }

        return loss_dict


    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        mask = batch["mask"]
        masked_img = torch.concat([self.get_gt_img(batch["image"]), mask], dim=2)
        gt_rgb = self.composite_with_background(masked_img, outputs["background"])
        predicted_rgb = outputs["rgb"]

        gt_rgb.clamp_(0.0, 1.0)
        predicted_rgb.clamp_(0.0, 1.0)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
