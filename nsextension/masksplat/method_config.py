"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

def get_method_config():
    from nerfstudio.plugins.types import MethodSpecification
    from nerfstudio.configs.base_config import ViewerConfig
    from nerfstudio.engine.optimizers import AdamOptimizerConfig
    from nerfstudio.engine.schedulers import (
        ExponentialDecaySchedulerConfig,
    )
    from nerfstudio.engine.trainer import TrainerConfig
    from masksplat.mask_splat import MaskSplatConfig
    from masksplat.dataparser import BiGSBlenderConfig
    from masksplat.datamanager import (
        FullImageLightDatamanagerConfig,
    )
    from masksplat.pipeline import (
        MaskSplatPipelineConfig,
    )

    method_config = MethodSpecification(
        config=TrainerConfig(
            method_name="mask-splat", 
            steps_per_eval_image=100,
            steps_per_eval_batch=0,
            steps_per_save=2000,
            steps_per_eval_all_images=1000,
            max_num_iterations=30000,
            mixed_precision=False,
            pipeline=MaskSplatPipelineConfig(
                datamanager=FullImageLightDatamanagerConfig(
                    dataparser=BiGSBlenderConfig(),
                ),
                model=MaskSplatConfig(),
            ),
            optimizers={
                "means": {
                    "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                    "scheduler": ExponentialDecaySchedulerConfig(
                        lr_final=1.6e-6,
                        max_steps=30000,
                    ),
                },
                "features_dc": {
                    "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                    "scheduler": None,
                },
                "features_rest": {
                    "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                    "scheduler": None,
                },
                "opacities": {
                    "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                    "scheduler": None,
                },
                "scales": {
                    "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                    "scheduler": None,
                },
                "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
                "camera_opt": {
                    "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                    "scheduler": ExponentialDecaySchedulerConfig(
                        lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                    ),
                },
            },
            viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
            vis="viewer",
        ),
        description="Gaussian Splatting with additional mask loss for removing floaters",
    )

    return method_config

method_config = get_method_config()
