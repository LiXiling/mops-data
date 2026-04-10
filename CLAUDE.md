# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MOPS-Data is a dataset generation framework for creating photoreal synthetic datasets for computer vision tasks in robotic manipulation. It renders PartNet-Mobility objects in ManiSkill3/SAPIEN simulations and outputs datasets with multi-modal observations (RGB, depth, segmentation masks, surface normals). Default output format is WebDataset (sharded TARs) for Hugging Face Hub publishing; HDF5 is available via `--format hdf5`.

**Requires Python 3.10** (ManiSkill3 constraint).

## Commands

### Setup
```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

### Venv location
`.venv`

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

### Formatting
```bash
ruff format .
```

## Architecture

### Generation Pipeline
Each dataset type follows the same pattern: `Config dataclass` → `Pipeline` → `SubprocessRenderer` → `Writer`. The writer is selected by `BaseDatasetConfig.output_format` (`OutputFormat.WEBDATASET` default, `OutputFormat.HDF5` for legacy). The `_open_writer()` context manager on `BaseDatasetPipeline` handles the dispatch.

- **`src/mops_data/generation/base_config.py`**: `BaseDatasetConfig` dataclass with the asset blacklist (33 PartNet IDs known to cause crashes) and `output_format` field. Subclasses: `SingleObjectDatasetConfig`, `KitchenDatasetConfig`, `ClutterDatasetConfig`.
- **`src/mops_data/generation/base_pipeline.py`**: Abstract `BaseDatasetPipeline` — filters assets, generates viewpoint×lighting variation plans, provides `_open_writer()` factory.
- **`src/mops_data/generation/subprocess_renderer.py`**: Spawns fresh subprocesses per render batch to force GPU memory cleanup via OS (prevents OptiX/CUDA OOM accumulation). Key functions: `render_in_subprocess()`, `render_batch_parallel()`.
- **`src/mops_data/generation/webdataset_writer.py`**: `WebDatasetWriter` context manager (default). Writes sharded TAR archives in WebDataset format for HF Hub. Integer masks as lossless PNG; multi-channel/float arrays as compressed `.npz`.
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

### WebDataset Output Structure (default)
```
dataset_dir/
├── train/
│   ├── 00000.tar
│   └── ...
├── test/
│   ├── 00000.tar
│   └── ...
└── dataset_info.json

Each sample in a TAR shard (files sharing a key prefix):
  {id}.png               — RGB image (PNG)
  {id}.semantic.png      — semantic mask, lossless grayscale PNG (~40 classes)
  {id}.instance.png      — instance mask, grayscale PNG (uint8 or uint16)
  {id}.part.png          — part mask, grayscale PNG (uint8 or uint16)
  {id}.is_partnet.png    — binary mask, grayscale PNG
  {id}.affordance.npz    — (H,W,56) int multi-hot, compressed numpy
  {id}.depth.npz         — (H,W,1) float32, compressed numpy
  {id}.normal.npz        — (H,W,3) float32, compressed numpy
  {id}.bbox.json         — bounding boxes [x, y, w, h, class_id]
  {id}.json              — metadata (image_id, asset_id, render_params, class)
```

Load with: `datasets.load_dataset("webdataset", data_dir="dataset_dir")`
Decode arrays: `np.load(io.BytesIO(row["depth.npz"]))["data"]`

### Observation Data Characteristics
- **Affordance** masks are extremely sparse: shape `(H,W,56)` multi-hot int, but 99%+ of values are zero (most pixels are background, and objects only have ~5-10 active affordances out of 56). The WebDataset writer uses `.npz` (numpy deflate) which achieves ~100x compression on these sparse arrays.
- **Depth** is `(H,W,1)` float32 with smooth gradients — compresses ~2-3x with deflate.
- **Normal** is `(H,W,3)` float32 with high spatial frequency — compresses poorly (~20%).
- **Integer masks** (semantic, instance, part, is_partnet) are single-channel with low cardinality — PNG achieves excellent lossless compression.
