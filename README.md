# DCASE 2026 Task 2: Unsupervised Anomalous Sound Detection

A PyTorch-based pipeline for unsupervised anomalous sound detection in industrial machinery, developed for the DCASE 2026 Challenge (Task 2). This repository implements a patch-based contrastive learning architecture with contextual global attention, specifically optimized for practical efficiency and deployment on constrained hardware.

By focusing on smarter architectural design rather than raw computing power, this system efficiently processes audio data to identify transient and steady-state anomalies across diverse machine types and domain shifts.

---

## 🚀 Key Features

- **Patch-Based Representation Learning:** Extracts 32×32 patches from Log-Mel spectrograms, allowing the network to capture fine-grained, localized kinematic audio features without overwhelming memory limits.

- **Contextual Attention Pooling:** Utilizes a custom attention mechanism (`attention_pooling.py`) to dynamically weigh the importance of individual temporal and frequency patches.

- **Domain Generalization:** Integrates CORAL, MMD, and Domain Adversarial (Gradient Reversal Layer) losses to maintain robust performance across source and target domains.

- **Multi-Objective Contrastive Training:** Employs an NT-Xent loss formulation coupled with dynamic confidence weighting and center-pull penalties to create a highly separable latent space for normal sounds.

- **Non-Parametric Anomaly Scoring:** Evaluates embeddings using Mahalanobis and Cosine distances, ensuring accurate anomaly detection without relying on artificial geometric constraints.

---

## 📁 Repository Structure

| File | Description |
|--------|-------------|
| `convert_rgb.py` | Pre-processing script to convert raw `.wav` files into 224×224 RGB Log-Mel spectrograms using the near-field microphone channel. |
| `astra_attn_patch_dataset.py` | Custom PyTorch Dataset class handling multi-scale patch extraction, domain labeling (source vs. target), and dynamic augmentations. |
| `patch_attn_model.py` | Core architecture featuring a ResNet34 encoder, temporal attention networks, and a domain discriminator. |
| `attention_pooling.py` | Attention-based pooling module with attribute-bias gating to fuse local patch embeddings into a global representation. |
| `train_1.py` | Main training loop implementing the multi-objective contrastive loss, learning rate scheduling, and EMA center tracking. |
| `evaluation4.py` | Evaluation pipeline for embedding extraction and anomaly scoring. |
| `pauc.py` | Utility script for detailed pAUC analysis and reporting. |

---

## ⚙️ Installation & Requirements

Ensure Python 3.8 or later is installed.

### Clone the Repository

```bash
git clone https://github.com/yourusername/dcase2026-task2-anomaly.git
cd dcase2026-task2-anomaly
```

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install librosa soundfile pandas numpy matplotlib pillow scikit-learn tqdm
```

---

## 🛠️ Usage Pipeline

### 1. Data Preparation

Place the downloaded DCASE 2026 dataset in the project root directory. The script expects the standard DCASE folder structure containing:

- `train/`
- `test/`
- `supplemental/`

for each machine type.

Generate RGB Log-Mel spectrograms:

```bash
python convert_rgb.py
```

---

### 2. Training

Start model training:

```bash
python train_1.py
```

Checkpoints are saved automatically during training.

> **Note:** Adjust `BATCH_SIZE`, `EPOCHS`, and `ROOT_DIR` in the configuration section of `train_1.py` according to available hardware resources.

---

### 3. Evaluation

Run the evaluation pipeline:

```bash
python evaluation4.py
```

For detailed pAUC analysis:

```bash
python pauc.py
```

---

## 📊 Evaluation Metrics

The system follows the official DCASE Task 2 evaluation protocol.

### 1. AUC (Area Under the ROC Curve)

Computed independently for:

- Source Domain
- Target Domain

### 2. pAUC (Partial AUC)

Computed under a constrained false positive rate:

```text
max_fpr = 0.1
```

This metric emphasizes detection performance in low false-alarm industrial environments.

### 3. Harmonic Mean (Ω)

The final challenge metric aggregates source-domain and target-domain AUC/pAUC scores across all machine types using a harmonic mean formulation.

---

## 📝 Implementation Details

The system is implemented in PyTorch and explicitly optimized for constrained hardware, with training conducted on an NVIDIA RTX 5060 Laptop GPU.

To maintain computational efficiency, input Log-Mel spectrograms are partitioned into 32×32 patches with a stride of 16, yielding a maximum of 32 patches per sample.

Training is performed jointly across all machine types for **150 epochs** using the Adam optimizer with:

- Initial learning rate: **2 × 10⁻⁴**
- Batch size: **96**

Learning rate adaptation is handled through a ReduceLROnPlateau scheduler configured with:

- Patience: **10 epochs**
- Decay factor: **0.5**

For contrastive representation learning, the NT-Xent loss temperature is fixed at:

```math
\tau = 0.05
```

Center representations used for confidence weighting are maintained through an Exponential Moving Average (EMA) with momentum:

```math
\gamma = 0.9
```

Following an **8-epoch warmup phase**, the optimization objective is augmented with:

- Boundary penalty:
  
  ```math
  \beta_{boundary}=0.02
  ```

- Center-pull penalty:
  
  ```math
  \beta_{center}=0.05
  ```

- Dynamic confidence weighting:
  
  ```math
  w_{hard}\rightarrow1.25
  ```

  ```math
  w_{soft}\rightarrow0.9
  ```

These additions progressively refine the latent representation space and improve anomaly separability while preserving domain robustness.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@misc{dcase2026task2,
  title={Patch-Based Contextual Attention for Unsupervised Anomalous Sound Detection},
  author={Your Name},
  year={2026},
  note={DCASE 2026 Challenge Task 2 Submission}
}
```
