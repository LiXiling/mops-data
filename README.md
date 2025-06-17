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

### Installation

```bash
# Create and activate conda environment
conda create --name mops python=3.10
conda activate mops

# Install ManiSkill3 and dependencies
pip install mani_skill
pip install torch torchvision torchaudio
pip install -e .
```

### Asset Setup

1. **Download RoboCasa assets**
   ```bash
   python -m mani_skill.utils.download_asset RoboCasa
   ```

2. **Download PartNet Mobility Assets**  
   Get the assets from [SAPIEN UCSD](https://sapien.ucsd.edu/browse)

3. **Organize directory structure**
   ```
   mops-data/
   ├── demos/
   ├── data/
   │   └── partnet_mobility/
   ├── src/
   └── ...
   ```

## 🛠️ Development Setup

For code development, install and configure pre-commit hooks:

```bash
pip install black isort pre-commit
pre-commit install
pre-commit run
```

## 📚 Documentation

This project builds on [ManiSkill3](https://github.com/haosulab/ManiSkill), so much of its [official documentation](https://maniskill.readthedocs.io/en/latest/) is applicable to MOPS-data.

### Notable Changes & Extensions

Our CustomEnvs now accept an optional keyword argument `np_rng`, which expects a Numpy Random Number Generator Object.

The `AffordanceKitchenEnv-v1` also supports an optional argument `preroll`, which deterministically creates a kitchen setup.

For MOPS-specific functionality, refer to the code documentation in the `src/` directory and example scripts in `demos/`.

## 📖 Usage Guide

| Component                     | Description                                              |
| ----------------------------- | -------------------------------------------------------- |
| **`demos/`**                  | Executable high-level scripts for quick image generation |
| **`src.mops_data`**           | Core source code                                         |
| **`mops_data.asset_manager`** | Observation augmentation using annotation resources      |
| **`mops_data.envs`**          | Custom environments for rendering objects and scenes     |
| **`xr_teleop`**               | WebXR-based VR teleoperation controller                  |

### 🥽 VR Integration
The `xr_teleop` module provides rudimentary VR teleoperation through WebXR. Serve the webpage and access it with a VR headset to send control commands. 

> 🔜 **Coming Soon**: [IRIS](https://intuitive-robots.github.io/iris-project-page/) support!

## 📊 Features

- 🎨 **Photoreal Simulation**: High-quality visual rendering for computer vision
- 🤖 **Robotic Manipulation**: Specialized environments for manipulation tasks
- 🏠 **Kitchen Environments**: Realistic household scenarios
- 📦 **Object Diversity**: Support for cluttered tabletop scenarios
- 🔧 **Extensible**: Modular design for custom environments

## 📝 Citation

If you use MOPS-data in your research, please cite:

```bibtex
@article{li2025mops,
  title={Multi-Objective Photoreal Simulation (MOPS) Dataset for Computer Vision in Robotic Manipulation},
  author={
    Maximilian Xiling Li and
    Paul Mattes and
    Nils Blank and
    Korbinian Franz Rudolf and
    Paul Werker L\"odige and
    Rudolf Lioutikov
  },
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our development setup above and feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>🔗 Links</strong><br>
  <a href="https://intuitive-robots.github.io/mops/">Project Page</a> •
  <a href="https://intuitive-robots.github.io/mops/static/li2025mops-preprint.pdf">Paper</a> •
  <a href="#citation">Citation</a>
</div>