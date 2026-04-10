# MOPS-data 

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=globe)](https://intuitive-robots.github.io/mops/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader)](https://intuitive-robots.github.io/mops/static/li2025mops-preprint.pdf)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![ManiSkill3](https://img.shields.io/badge/Built%20on-ManiSkill3-green?style=for-the-badge)](https://github.com/haosulab/ManiSkill)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **⚠️ Alpha Release Notice**  
> This is an alpha release for [MOPS](https://intuitive-robots.github.io/mops/) data generation. The public API might still change, and some bugs might be present.

## 🚀 Quick Start

### Prerequisites
This project builds on `ManiSkill3`, a simulation framework built on `SAPIEN`, which requires **Python 3.10**.  
We recommend [uv](https://docs.astral.sh/uv/) for environment and dependency management.

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a Python 3.10 virtual environment and install all dependencies
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

### Asset Setup

1. **Download RoboCasa assets** (required for kitchen dataset)
   ```bash
   python -m mani_skill.utils.download_asset RoboCasa
   ```

2. **Download PartNet Mobility assets** (required for all datasets)  
   Get the assets from [SAPIEN UCSD](https://sapien.ucsd.edu/browse) and place them under `data/partnet_mobility/`.

3. **Expected directory structure**
   ```
   mops-data/
   ├── data/
   │   ├── partnet_mobility/   # PartNet-Mobility assets
   │   └── mops_data/          # generated datasets (output)
   ├── scripts/                # dataset generation entry points
   ├── demos/                  # visualisation & exploration scripts
   ├── src/
   │   └── mops_data/          # core library
   └── ...
   ```

## 🗂️ Data Generation

Three dataset variants are provided. Each has a **debug mode** (small, fast) and a **full mode** (production quality).

### Single-Object Dataset

Isolated PartNet-Mobility objects rendered from multiple viewpoints with varied lighting.

```bash
# Debug run (quick sanity check)
python scripts/generate_single_object.py --debug

# Full generation
python scripts/generate_single_object.py

# Custom output path
python scripts/generate_single_object.py --output data/mops_data/my_single_obj
```

### Kitchen (Affordance) Dataset

Objects placed inside RoboCasa kitchen environments, rendered from table-level and overhead viewpoints.

```bash
python scripts/generate_kitchen.py --debug   # debug run
python scripts/generate_kitchen.py           # full generation
```

### Tabletop Clutter Dataset

Cluttered tabletop scenes with multiple objects rendered from top-down viewpoints.

```bash
python scripts/generate_clutter.py --debug   # debug run
python scripts/generate_clutter.py           # full generation
```

### Output Format

All scripts default to **WebDataset** (sharded TAR archives), which is the recommended format for publishing on [Hugging Face Hub](https://huggingface.co/docs/datasets/en/image_dataset#webdataset). To generate in legacy HDF5 format instead:

```bash
python scripts/generate_kitchen.py --debug --format hdf5
```

To convert an existing HDF5 dataset to WebDataset:

```bash
python scripts/convert_hdf5_to_webdataset.py data/mops_data/dataset.h5 [-o output_dir]
```

### Custom Configuration

Each script accepts `--output <path>` to override the output directory and `--format <webdataset|hdf5>` to select the output format.  
For deeper customisation, edit the corresponding config class in `src/mops_data/generation/`:

| Dataset        | Config class                  | Module                                           |
| -------------- | ----------------------------- | ------------------------------------------------ |
| Single-object  | `SingleObjectDatasetConfig`   | `mops_data.generation.single_object_dataset`     |
| Kitchen        | `KitchenDatasetConfig`        | `mops_data.generation.kitchen_dataset`           |
| Clutter        | `ClutterDatasetConfig`        | `mops_data.generation.clutter_dataset`           |

Key parameters shared by all configs (`BaseDatasetConfig`):

| Parameter                    | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| `output_path`                | Path to the output dataset directory                 |
| `image_size`                 | `(width, height)` in pixels                         |
| `target_train_images_per_set`| Training images per object/scene set                 |
| `target_test_images_per_set` | Test images per object/scene set                     |
| `min_assets_per_class`       | Minimum assets required to include a class           |
| `light_temp_range`           | Kelvin range for light colour temperature variation  |
| `light_intensity_range`      | Range for light intensity variation                  |
| `obs_mode`                   | Observation channels (`rgb+depth+segmentation+normal`) |

## 🛠️ Development Setup

For code development, install the dev dependencies and configure pre-commit hooks:

```bash
uv pip install -e ".[dev]"
pre-commit install
pre-commit run
```

## 📚 Documentation

This project builds on [ManiSkill3](https://github.com/haosulab/ManiSkill), so much of its [official documentation](https://maniskill.readthedocs.io/en/latest/) is applicable to MOPS-data.

### Notable Changes & Extensions

- Custom environments accept an optional `np_rng` keyword argument (a NumPy `Generator` object) for reproducible scene sampling.
- `AffordanceKitchenEnv-v1` supports an optional `preroll` argument to deterministically recreate a specific kitchen setup.

## 📖 Code Overview

| Component                       | Description                                               |
| ------------------------------- | --------------------------------------------------------- |
| **`scripts/`**                  | High-level dataset generation entry points                |
| **`demos/`**                    | Visualisation and exploration scripts                     |
| **`src/mops_data`**             | Core library                                              |
| **`mops_data.generation`**      | Dataset generation pipelines and configuration            |
| **`mops_data.asset_manager`**   | PartNet-Mobility annotation loading and asset handling    |
| **`mops_data.envs`**            | Custom ManiSkill3 environments for rendering              |
| **`mops_data.render`**          | Observation augmentation and shader configuration         |
| **`xr_teleop`**                 | WebXR-based VR teleoperation controller. Experimental     |

## 📊 Features

- 🎨 **Photoreal Simulation**: High-quality visual rendering for computer vision
- 🤖 **Robotic Manipulation**: Specialised environments for manipulation tasks
- 🏠 **Kitchen Environments**: Realistic household scenarios built on RoboCasa
- 📦 **Object Diversity**: Support for cluttered tabletop and single-object scenarios
- 🔧 **Extensible**: Modular pipeline and config design for custom datasets

## 📝 Citation

If you use MOPS-data in your research, please cite:

```bibtex
@article{li2026mops,
  title={Multi-Objective Photoreal Simulation (MOPS) Dataset for Computer Vision in Robotic Manipulation},
  author={
    Maximilian Xiling Li and
    Paul Mattes and
    Nils Blank and
    Rudolf Lioutikov
  },
  year={2026}
}
```

## 🤝 Contributing

We welcome contributions! Please see the development setup above and feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>🔗 Links</strong><br>
  <a href="https://intuitive-robots.github.io/mops/">Project Page</a> •
  <a href="https://intuitive-robots.github.io/mops/static/li2025mops-preprint.pdf">Paper</a> •
  <a href="#citation">Citation</a>
</div>