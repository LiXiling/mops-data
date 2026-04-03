# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MOPS-Data is a dataset generation framework for creating photoreal synthetic datasets for computer vision tasks in robotic manipulation. It renders PartNet-Mobility objects in ManiSkill3/SAPIEN simulations and outputs HDF5 datasets with multi-modal observations (RGB, depth, segmentation masks, surface normals).

**Requires Python 3.10** (ManiSkill3 constraint).

## Commands

### Setup
```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

### Dataset Generation
```bash
# Debug mode (fast, small images, few samples)
python scripts/generate_single_object.py --debug
python scripts/generate_kitchen.py --debug
python scripts/generate_clutter.py --debug

# Full production run
python scripts/generate_single_object.py
python scripts/generate_kitchen.py
python scripts/generate_clutter.py
```

### Linting
```bash
ruff check .
ruff format .
pre-commit run --all-files
```

## Architecture

### Generation Pipeline
Each dataset type follows the same pattern: `Config dataclass` → `Pipeline` → `SubprocessRenderer` → `HDF5Writer`.

- **`src/mops_data/generation/base_config.py`**: `BaseDatasetConfig` dataclass with the asset blacklist (33 PartNet IDs known to cause crashes). Subclasses: `SingleObjectDatasetConfig`, `KitchenDatasetConfig`, `ClutterDatasetConfig`.
- **`src/mops_data/generation/base_pipeline.py`**: Abstract `BaseDatasetPipeline` — filters assets, generates viewpoint×lighting variation plans.
- **`src/mops_data/generation/subprocess_renderer.py`**: Spawns fresh subprocesses per render batch to force GPU memory cleanup via OS (prevents OptiX/CUDA OOM accumulation). Key functions: `render_in_subprocess()`, `render_batch_parallel()`.
- **`src/mops_data/generation/hdf_writer.py`**: `HDF5Writer` context manager. Writes images and all mask types (semantic, instance, part, affordance, depth, normal, bbox) into a structured HDF5 file.
- **`src/mops_data/generation/variation_utils.py`**: Generates the Cartesian product of viewpoints × lighting conditions, then samples with stochastic jitter (±10° azimuth, ±5° elevation).

### Simulation Environments (ManiSkill3 / Gymnasium)
Custom environments in `src/mops_data/envs/dataset_envs/` registered via `@register_env`:
- **`SingleObjectRenderEnv-v1`**: Single PartNet object at origin with configurable pose/lighting.
- **`KitchenRenderEnv-v1`**: RoboCasa kitchen scene with objects on counter fixtures.
- **`ClutterRenderEnv-v1`**: Multiple objects scattered on a tabletop, top-down camera.

Base class `DatasetRenderEnv` (`base_rendering_env.py`) handles Kelvin→RGB conversion, lighting setup, and observation extraction.

### Asset Management
- **`AnnotationHandler`** (`anno_handler.py`): Singleton that loads embedded JSON resources (`class_affordances.json`, `partnet-mobility_affordances.json`) and builds a dataframe of all PartNet-Mobility objects with class/affordance metadata.
- **`PartNetMobilityLoader`** (`partnet_mobility_loader.py`): Parses URDF files, extracts semantic link annotations, and creates SAPIEN articulations.
- **`ObjectAnnotationRegistry`** (`object_annotation_registry.py`): Caches loaded objects and maps segmentation IDs to class/part labels.

### Observation Augmentation
- **`AffordObsAugmentor`** (`src/mops_data/render/afford_obs_augmentor.py`): Post-processes raw SAPIEN segmentation into semantic/instance/affordance/part masks and `is_partnet` flags.
- **`RT_RGB_ONLY_CONFIG`** (`shader_config.py`): OptiX ray-tracing config — 8 SPP, depth 8, OptiX denoiser, outputs uint8 RGB.

### Data Paths
`data/` contains symlinks:
- `data/partnet_mobility/` → `/mnt/data/partnet_mobility`
- `data/mops_data/` → `/mnt/data/mops-data`
- `data/robocasa_dataset/` → `~/.maniskill/data/scene_datasets/robocasa_dataset`

### HDF5 Output Structure
```
.h5
├── images/           # RGB (gzip-6)
├── masks/
│   ├── semantic/     # Per-class segmentation
│   ├── instance/
│   ├── part/
│   ├── affordance/
│   ├── depth/
│   ├── normal/
│   ├── is_partnet/
│   └── bbox/         # [x, y, w, h, class_id]
├── labels/
│   ├── class_names
│   └── class_labels
└── metadata/
    ├── splits        # Train/test binary flags
    ├── image_info    # Per-image JSON metadata
    └── attrs: {total_images, creation_date, version, num_classes}
```
