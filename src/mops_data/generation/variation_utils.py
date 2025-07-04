import itertools
from typing import Dict, List

import numpy as np


def generate_base_variations(viewpoints, lightings) -> List[Dict]:
    """Generate base variation combinations from viewpoints and lighting types."""
    return [
        {"viewpoint": vp, "lighting": lt}
        for vp, lt in itertools.product(viewpoints, lightings)
    ]


def sample_variations_for_asset(config, n_images: int, base_variations) -> List[Dict]:
    """Sample n variations for a single asset with random lighting."""
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
    """Sample random lighting parameters."""
    return {
        "type": np.random.choice(config.lighting_types),
        "temperature": np.random.uniform(*config.light_temp_range),
        "intensity": np.random.uniform(*config.light_intensity_range),
    }
