# TA-STGCN: Temporal Attention-Guided Spatio-Temporal Graph Convolutional Network

Official modular PyTorch implementation of **TA-STGCN** (Temporal Attention-Guided Spatio-Temporal Graph Convolutional Network) for city-scale multi-horizon traffic volume forecasting under motorcycle-dominant mixed traffic networks.

---

## Architecture Overview

**TA-STGCN** combines spatial Chebyshev Spectral Graph Convolutions with Model-Level Multi-Head Temporal Self-Attention:
1. **Spatial Chebyshev Spectral Graph Convolutions ($K=3$):** Captures localized spatial graph topology across physical road networks using normalized Chebyshev Laplacians ($\tilde{L}$).
2. **Temporal Gated Convolutions (GLU 1D Causal Conv):** Extracts short-term temporal dynamics across historical observation steps.
3. **Multi-Head Temporal Self-Attention:** Observes the full historical sequence ($T_{\text{in}} = 24$ timesteps / 120 minutes), dynamically reweighting critical temporal context steps and mitigating perception noise propagation.
4. **Parameter Economy:** Achieves state-of-the-art noise robustness while utilizing only **~453K parameters** (a 70.4% reduction over heavy baselines such as ASTGCN).

```text
Input X (B, T_in=24, N=608, C_in=5)
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ STGCN BLOCK 1                                           │
│  ├─ Temporal Conv (GLU 1D Causal Conv)                  │
│  ├─ Spatial Graph Conv (Chebyshev K=3) + ReLU           │
│  ├─ Temporal Conv (GLU 1D Causal Conv)                  │
│  └─ Residual Add + LayerNorm + Dropout                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ TEMPORAL SELF-ATTENTION (Middle Placement)              │
│  └─ Multi-Head Attention + Residual + LayerNorm + FFN   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ STGCN BLOCK 2                                           │
│  ├─ Temporal Conv (GLU 1D Causal Conv)                  │
│  ├─ Spatial Graph Conv (Chebyshev K=3) + ReLU           │
│  ├─ Temporal Conv (GLU 1D Causal Conv)                  │
│  └─ Residual Add + LayerNorm + Dropout                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ TIME-DIMENSION REDUCTION CONV 1D (T_in -> 1)            │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
Output Y (B, Horizon=6, N=608, C_out=2)  [Cars & Motorcycles]
```

---

## Repository Structure

```text
ta_stgcn/
├── README.md                  # Comprehensive Documentation & Usage Guide
├── requirements.txt           # Python dependency requirements
├── config.yaml                # Default YAML configuration file
├── train.py                   # Main CLI script for model training
├── eval.py                    # Standalone checkpoint evaluation script
├── data/
│   ├── __init__.py
│   └── dataset.py             # Dataset loader, Z-score scaling, time features
├── models/
│   ├── __init__.py
│   ├── cheb_conv.py           # Chebyshev Spectral Graph Convolution (ChebConvLayer)
│   ├── temporal_conv.py       # Temporal Gated Convolution (GLU 1D Causal Conv)
│   ├── temporal_attention.py  # Multi-Head Temporal Self-Attention module
│   ├── stgcn_block.py         # Spatio-Temporal Convolutional Block (STGCNBlock)
│   └── ta_stgcn.py            # Complete TA-STGCN Model assembly
└── utils/
    ├── __init__.py
    ├── graph_utils.py         # Distance graph loading & Scaled Laplacian computation
    ├── metrics.py             # MAE, RMSE, MAPE, WAPE metrics (overall & per-horizon)
    ├── loss.py                # Huber Loss with optional smoothness regularization
    ├── ema.py                 # Exponential Moving Average (ModelEMA)
    └── logger.py              # Dual console & file logger (TeeLogger)
```

---

## ⚡ Quick Start & Installation

### 1. Requirements
Ensure Python 3.9+ and PyTorch 2.0+ are installed. Install required packages:

```bash
pip install -r requirements.txt
```

---

## Data Preparation & Input Format

Before starting training, ensure your data files are placed in the root directory (or specify their custom paths via args):

1. **Traffic Volume Time-Series CSV (`count_7_7_merg_sort_fix_fill.csv`):**
   - Must contain columns: `Timestamp`, `STT` (Node ID), `Car Count`, `Bike Count`.
   - Aggregated at 5-minute timestep resolution across all nodes.

2. **Graph Adjacency Matrix Excel (`Graph_fix_py_3.xlsx`):**
   - Excel spreadsheet where sheet 0 contains a square distance matrix ($N \times N$) between spatial node IDs.

---

## 🏋️ Training the TA-STGCN Model

Run the training pipeline using the default parameters defined in `config.yaml`:

```bash
python train.py
```

### Custom Training Arguments

You can override default configuration options directly via CLI flags:

```bash
python train.py \
    --csv_path "count_7_7_merg_sort_fix_fill.csv" \
    --adj_path "Graph_fix_py_3.xlsx" \
    --epochs 500 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --seed 42 \
    --device cuda \
    --save_dir checkpoints
```

### Key Training Options

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--config` | `config.yaml` | Path to YAML configuration file |
| `--csv_path` | `count_7_7_merg_sort_fix_fill.csv` | Traffic count dataset path |
| `--adj_path` | `Graph_fix_py_3.xlsx` | Graph distance matrix path |
| `--epochs` | `500` | Maximum training epochs |
| `--batch_size` | `64` | Training batch size |
| `--learning_rate` | `0.0005` | Initial learning rate for AdamW |
| `--seed` | `42` | Random seed for exact reproducibility |
| `--device` | `cuda` | Target compute device (`cuda` or `cpu`) |
| `--save_dir` | `checkpoints` | Checkpoint output folder |

---

## 📈 Evaluating Saved Checkpoints

To evaluate a trained checkpoint on the test set and print detailed metric breakdowns:

```bash
python eval.py --checkpoint checkpoints/ta_stgcn_best.pth
```

### Sample Evaluation Output

```text
==================================================
 🏆 TA-STGCN PERFORMANCE EVALUATION METRICS 🏆
==================================================
  Overall MAE  : 3.2401
  Overall RMSE : 5.8124
  Overall MAPE : 8.12%
  Overall WAPE : 7.45%
--------------------------------------------------
  Horizon Breakdown:
   ├─ t+1 ( 5m): MAE=2.7120 | RMSE=4.8912 | WAPE=6.23%
   ├─ t+2 (10m): MAE=3.0145 | RMSE=5.3410 | WAPE=6.91%
   ├─ t+3 (15m): MAE=3.2104 | RMSE=5.7128 | WAPE=7.38%
   ├─ t+4 (20m): MAE=3.3890 | RMSE=6.0125 | WAPE=7.79%
   ├─ t+5 (25m): MAE=3.5120 | RMSE=6.2410 | WAPE=8.07%
   ├─ t+6 (30m): MAE=3.6030 | RMSE=6.4512 | WAPE=8.28%
==================================================
```

---

## Configuration File (`config.yaml`)

Hyperparameters can be updated directly inside `config.yaml`:

```yaml
data:
  csv_path: "count_7_7_merg_sort_fix_fill.csv"
  adj_path: "Graph_fix_py_3.xlsx"
  time_step_minutes: 5
  history_minutes: 120
  horizon: 6

model:
  cheb_K: 3
  num_blocks: 2
  block_hidden: 80
  dropout: 0.25
  use_temporal_attention: true
  attn_num_heads: 4
  attn_dropout: 0.1
  attn_position: "middle"

training:
  seed: 42
  batch_size: 64
  epochs: 500
  learning_rate: 0.0005
  patience: 60
  grad_clip_norm: 5.0
  use_lr_scheduler: true
  use_ema: true
  ema_decay: 0.995
  loss_delta: 1.0
```

---

## License & Acknowledgments

This codebase is part of the urban traffic forecasting research benchmark. All rights reserved.
