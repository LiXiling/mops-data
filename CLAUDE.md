# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MOPS-Data is a dataset generation framework for creating photoreal synthetic datasets for computer vision tasks in robotic manipulation. It renders PartNet-Mobility objects in ManiSkill3/SAPIEN simulations and outputs Hugging Face Parquet datasets with multi-modal observations (RGB, depth, segmentation masks, surface normals).

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
Each dataset type follows the same pattern: `Config dataclass` → `Pipeline` → `SubprocessRenderer` → `ParquetWriter`.

- **`src/mops_data/generation/base_config.py`**: `BaseDatasetConfig` dataclass with the asset blacklist (33 PartNet IDs known to cause crashes). Subclasses: `SingleObjectDatasetConfig`, `KitchenDatasetConfig`, `ClutterDatasetConfig`.
- **`src/mops_data/generation/base_pipeline.py`**: Abstract `BaseDatasetPipeline` — filters assets, generates viewpoint×lighting variation plans.
- **`src/mops_data/generation/subprocess_renderer.py`**: Spawns fresh subprocesses per render batch to force GPU memory cleanup via OS (prevents OptiX/CUDA OOM accumulation). Key functions: `render_in_subprocess()`, `render_batch_parallel()`.
- **`src/mops_data/generation/parquet_writer.py`**: `ParquetWriter` context manager. Writes images and all mask types into sharded Parquet files (Hugging Face `datasets` format). RGB stored as `Image()`; all other arrays as lossless `.npy` bytes.
- **`src/mops_data/generation/hdf_writer.py`**: `HDF5Writer` context manager (legacy). Writes into a single HDF5 file with gzip compression.
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

### Parquet Output Structure
```
dataset_dir/
├── train/
│   ├── 00000.parquet
│   └── ...
├── test/
│   ├── 00000.parquet
│   └── ...
└── dataset_info.json

Each Parquet row:
  image_id      (string)   — e.g. "image_000042"
  image         (Image)    — RGB, PNG-encoded (datasets.Image feature)
  asset_id      (string)
  render_params (string)   — JSON
  class_name    (string)   — present when class_names provided
  class_idx     (int32)    — present when class_names provided
  semantic      (binary)   — npy bytes, np.load(io.BytesIO(val))
  instance      (binary)
  part          (binary)
  affordance    (binary)
  depth         (binary)
  normal        (binary)
  is_partnet    (binary)
  bbox          (string)   — JSON, [x, y, w, h, class_id]
```

Load with: `datasets.load_dataset("parquet", data_dir="dataset_dir")`
