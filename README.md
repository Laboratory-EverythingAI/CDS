<div align="center">


### **Cross-Dimensional Synchronization**
#### *3D Simulation × 2D Segmentation · Real-Time Intravascular Guidewire Navigation*

<br/>

[![Paper](https://img.shields.io/badge/📄_Paper-Preprint_2024-4A90D9?style=flat-square)](https://arxiv.org)
[![Dataset](https://img.shields.io/badge/🗄️_Dataset-Open_Source-27AE60?style=flat-square)](https://drive.google.com/file/d/1Nb73pFPN9yH_AFU8CnvOktmB0lTH3eAS/view?usp=sharing)
[![Code](https://img.shields.io/badge/💻_Code-In_Progress-E67E22?style=flat-square)](#release-status)
[![License](https://img.shields.io/badge/📜_License-Academic_Use-8E44AD?style=flat-square)](#license)

<br/>

---

</div>

## 📌 Overview

**CDS** is a unified framework for real-time guidewire state estimation that tightly couples **3D differentiable physics simulation** with **2D fluoroscopic image segmentation** through a learned bidirectional synchronization mechanism. Unlike conventional approaches that treat 3D reconstruction and 2D segmentation as separate problems, CDS introduces a cross-dimensional attention module enabling continuous, mutually corrective information exchange between the two domains.

> 💡 **Key Insight:** Physics simulation encodes material constraints and resolves 2D ambiguities; 2D observations continuously correct simulation drift. Neither domain alone suffices — CDS makes them cooperate.

<br/>

**Highlights at a glance:**

| Metric | Value |
|--------|-------|
| 2D Segmentation DICE | **0.928 ± 0.019** |
| 3D Tip Position Error | **1.94 ± 0.48 mm** |
| Inference Speed | **35.2 ± 1.3 FPS** |
| Computational Overhead vs. Sequential | **−22%** |

<br/>

---



## 🏗️ Framework Architecture

The CDS framework consists of four tightly integrated modules:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CDS Framework                            │
│                                                                 │
│   ┌─────────────────┐        ┌──────────────────────────────┐  │
│   │  3D Domain      │◄──────►│  2D Domain                   │  │
│   │                 │        │                              │  │
│   │ physics_        │        │ segmentation_                │  │
│   │ simulator       │        │ network                      │  │
│   │ (1 kHz)         │        │ (30 Hz)                      │  │
│   └────────┬────────┘        └───────────┬──────────────────┘  │
│            │                             │                      │
│            ▼                             ▼                      │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │           cross_dim_attention                            │  │
│   │     Bipartite Graph · Geometric Prior · Uncertainty      │  │
│   └──────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│            ┌─────────────────┴──────────────┐                  │
│            ▼                                ▼                  │
│   ┌─────────────────┐             ┌──────────────────┐         │
│   │ state_          │             │ neural_          │         │
│   │ synchronization │             │ renderer         │         │
│   │ (Kalman + VAE)  │             │ (NeRF + Uncert.) │         │
│   └─────────────────┘             └──────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

<br/>

---

## 📁 Repository Structure

```
CDS/
│
├── 🧠 networks/                    # Core neural network modules
│   ├── physics_simulator           # Differentiable Cosserat rod dynamics
│   │                               #   └─ Implicit time integration (1 kHz)
│   │                               #   └─ Contact & collision handling
│   │                               #   └─ Analytical sensitivity computation
│   ├── segmentation_network        # Geometry-aware 2D segmentation
│   │                               #   └─ Multi-scale steerable filters
│   │                               #   └─ Topological feature learning
│   ├── cross_dim_attention         # Cross-dimensional attention mechanism
│   │                               #   └─ Heterogeneous bipartite graph
│   │                               #   └─ Uncertainty-weighted correspondence
│   ├── state_synchronization       # Bidirectional state sync module
│   │                               #   └─ Variational cross-dim encoder
│   │                               #   └─ Kalman smoothing + manifold learning
│   └── neural_renderer             # Stochastic neural rendering engine
│                                   #   └─ Differentiable volume rendering
│                                   #   └─ Uncertainty quantification (NeRF)
│
├── 🔬 experiments/                 # Training & evaluation entry points
│   ├── cds_guidewire_catheter_sync # Guidewire–catheter synchronization
│   ├── cds_physics_baseline_eval   # Physics baseline comparison
│   ├── cds_no_rendering_ablation   # Ablation: w/o neural renderer
│   └── cds_single_guidewire_seg    # Single guidewire segmentation (main)
│
├── 🫀 case/                        # Cardiovascular scenario configurations
│                                   #   Coronary · Peripheral · Carotid
│                                   #   Renal · Aortic
│
├── 🖥️  device/                     # Hardware & C-arm geometry configuration
│
├── 🔧 scripts/                     # Data preprocessing & pipeline utilities
│
├── 📊 tools/                       # Evaluation metrics & benchmark helpers
│                                   #   └─ DICE, Hausdorff, CDC score, FPS
│
├── 🎨 visualisation/               # 2D/3D output rendering & animation tools
│
├── run.sh                          # Quick-start entry point
└── setup.py                        # Package installation
```

<br/>

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 1.13
- CUDA-capable GPU (≥ 8 GB VRAM recommended)

### Installation

```bash
git clone https://github.com/XXXXXXXXXXXX/CDS.git
cd CDS
pip install -e .
```

### Run Demo

```bash
bash run.sh
```

This will launch a single-guidewire segmentation demo on the provided sample data using pre-configured C-arm geometry and default physics parameters.

### Quick Experiment

```bash
# Single guidewire segmentation (main experiment)
python experiments/cds_single_guidewire_seg.py --config case/coronary_simple.yaml

# Ablation: remove neural renderer
python experiments/cds_no_rendering_ablation.py --config case/coronary_simple.yaml
```

<br/>

---

## 🗄️ Dataset

We provide a demonstration dataset comprising **paired synthetic fluoroscopic sequences with full 3D ground-truth guidewire annotations**, covering all five cardiovascular scenario types evaluated in the paper.

| Scenario | Sequences | Avg. Frames | 3D GT |
|----------|-----------|-------------|-------|
| Coronary Artery | ✅ | 120 | ✅ |
| Peripheral Vascular | ✅ | 120 | ✅ |
| Carotid Artery | ✅ | 120 | ✅ |
| Renal Artery | ✅ | 120 | ✅ |
| Complex Aortic | ✅ | 120 | ✅ |

> To the best of our knowledge, this represents one of the first publicly available **paired 2D–3D guidewire datasets** with physics-based ground truth, and we hope it will serve as a benchmark resource for the community.

🔗 **[Download Demo Dataset](https://drive.google.com/file/d/1Nb73pFPN9yH_AFU8CnvOktmB0lTH3eAS/view?usp=sharing)**

<br/>

---

## 📦 Release Status

| Component | Status | Notes |
|-----------|--------|-------|
| Demo dataset | ✅ **Available** | All 5 cardiovascular scenarios |
| `networks/physics_simulator` | 🔄 In progress | Core Cosserat rod implementation |
| `networks/segmentation_network` | 🔄 In progress | Steerable filter backbone |
| `networks/cross_dim_attention` | 🔄 In progress | Bipartite graph attention |
| `networks/state_synchronization` | 🔄 In progress | Kalman + VAE sync module |
| `networks/neural_renderer` | 🔄 In progress | Differentiable NeRF renderer |
| Full training pipeline | 🔜 Coming soon | Multi-task curriculum training |
| Pretrained checkpoints | 🔜 Coming soon | RTX 4090 trained weights |

> ⭐ **Please watch and star this repository** to receive updates on new releases.

<br/>

---



## 📜 License

This repository is released for **academic research use only**. Commercial use is not permitted without prior written consent from the authors.

<br/>

<div align="center">

---


</div>
