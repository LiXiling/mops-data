from dataclasses import dataclass, field

import torch
from mani_skill.render.shaders import ShaderConfig

RT_RGB_ONLY_CONFIG = ShaderConfig(
    shader_pack="rt",
    texture_names={"Color": ["rgb"]},
    shader_pack_config={
        "ray_tracing_samples_per_pixel": 32,
        "ray_tracing_path_depth": 16,
        "ray_tracing_denoiser": "optix",
    },
    texture_transforms={
        "Color": lambda data: {"rgb": (data[..., :3] * 255).to(torch.uint8)},
    },
)
