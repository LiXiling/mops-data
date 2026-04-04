import itertools
from typing import Dict, List

import numpy as np


def generate_base_variations(viewpoints, lightings) -> List[Dict]:
    """Return the Cartesian product of all viewpoints × lighting types.

    Args:
        viewpoints: List of viewpoint dicts (each has ``azimuth`` and ``elevation``
            keys in degrees).
        lightings: List of lighting type strings (e.g. ``["studio", "natural"]``).

    Returns:
        List of ``{"viewpoint": ..., "lighting": ...}`` dicts, length
        ``len(viewpoints) * len(lightings)``.
    """
    return [
        {"viewpoint": vp, "lighting": lt}
        for vp, lt in itertools.product(viewpoints, lightings)
    ]


def sample_variations_for_asset(config, n_images: int, base_variations) -> List[Dict]:
    """Sample *n_images* rendering variations for a single asset.

    Draws with replacement from *base_variations*, then independently jitters
    each sample: ±10° azimuth, ±5° elevation, and freshly-sampled lighting.

    Args:
        config: Dataset config providing lighting ranges and types.
        n_images: Number of variations to sample.
        base_variations: Output of :func:`generate_base_variations`.

    Returns:
        List of variation dicts ready to unpack as ``env_kwargs``.
    """
    num_base = len(base_variations)
    indices = np.random.choice(num_base, size=n_images, replace=n_images > num_base)

    variations = []
    for i in indices:
        var = base_variations[i].copy()
        var["viewpoint"] = var["viewpoint"].copy()
        # Add small random perturbations to each sampled viewpoint
        var["viewpoint"]["azimuth"] += np.random.uniform(-10, 10)
        var["viewpoint"]["elevation"] += np.random.uniform(-5, 5)
        var["lighting"] = sample_random_lighting(config)
        variations.append(var)

    return variations


def sample_random_lighting(config) -> Dict:
    """Sample a random lighting config within the ranges set in *config*.

    Returns:
        Dict with keys ``type``, ``temperature`` (Kelvin), ``intensity``.
    """
    return {
        "type": np.random.choice(config.lighting_types),
        "temperature": np.random.uniform(*config.light_temp_range),
        "intensity": np.random.uniform(*config.light_intensity_range),
    }
