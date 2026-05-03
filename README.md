# Deep-Learning-Based-Next-Day-Wildfire-Spread-Forecasting-Using-Multisource-Geospatial-Data
# Deep Learning Based Next Day Wildfire Spread Forecasting

> B.Tech Major Project — Dayananda Sagar University, Bengaluru (2025–26)  
> Department of CSE (AI & ML)

## Team
- Monish V (ENG22AM0035)
- R Madan Kumar (ENG22AM0043)
- Rohan K S (ENG22AM0049)
- Sai Nikhil R (ENG22AM0052)

---

## Overview
This project applies deep learning-based image segmentation to predict next-day wildfire spread using the [Next Day Wildfire Spread (NDWS)](https://arxiv.org/abs/2112.02447) dataset. Three encoder-decoder architectures were trained and compared on 20,097 geospatial samples with 12 multisource input features.

---

## Models
| Model | IoU | Recall | F1 |
|---|---|---|---|
| U-Net Baseline | 0.4840 | 0.2769 | 0.3381 |
| Attention U2-Net ⭐ | **0.6164** | 0.4377 | **0.4265** |
| UNet3+ | 0.6076 | **0.4627** | 0.4107 |

**Best model: Attention U2-Net** — 27.4% IoU improvement over baseline.

---

## Dataset
- **Source:** [NDWS Dataset](https://arxiv.org/abs/2112.02447) (Huot et al., 2022)
- **Format:** TFRecord (54 files)
- **Samples:** 9,600 train / 4,301 val / 6,196 test
- **Input:** 64×64 grid, 12 geospatial feature channels
- **Output:** Binary next-day fire spread mask

> ⚠️ Dataset is NOT included in this repo. Download from the official source and place in your Google Drive.

---

## Notebooks
| Notebook | Description |
|---|---|
| `01_unet_baseline.ipynb` | U-Net Baseline — BCE+Dice loss, 10 epochs |
| `02_attention_u2net.ipynb` | Attention U2-Net — Dice-BCE loss, 9 epochs |
| `03_unet3plus.ipynb` | UNet3+ — Focal Tversky+BCE loss, 17 epochs |

All notebooks are designed to run on **Google Colab (T4 GPU)** with dataset mounted from Google Drive.

---

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Run on Colab
1. Mount your Google Drive with the NDWS TFRecord files
2. Open any notebook in Google Colab
3. Update the dataset path variable at the top of the notebook
4. Run all cells

---

## Key Results
- **Attention U2-Net** achieves best overall IoU (0.6164) and F1 (0.4265)
- **UNet3+** achieves best Recall (0.4627) — preferred for early warning systems
- **U-Net Baseline** used as performance reference

---

## Tech Stack
- Python 3.11, TensorFlow 2.x, Keras
- `tf.data` API for TFRecord pipeline
- Google Colab + NVIDIA T4 GPU
- NumPy, Matplotlib

---

## Supervisor
Dr. Vegi Fernando A — Associate Professor, CSE (AI & ML), DSU
