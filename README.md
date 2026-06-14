```python
content = """# DCASE 2026 Task 2: Unsupervised Anomalous Sound Detection

A PyTorch-based pipeline for unsupervised anomalous sound detection in industrial machinery, developed for the DCASE 2026 Challenge (Task 2). This repository implements a patch-based contrastive learning architecture with contextual global attention, specifically optimized for practical efficiency and deployment on constrained hardware.

By focusing on smarter architectural design rather than raw computing power, this system efficiently processes audio data to identify transient and steady-state anomalies across diverse machine types and domain shifts.

---

## 🚀 Key Features

* **Patch-Based Representation Learning:** Extracts 32x32 patches from Log-Mel spectrograms, allowing the network to capture fine-grained, localized kinematic audio features without overwhelming memory limits.
* **Contextual Attention Pooling:** Utilizes a custom attention mechanism (`attention_pooling.py`) to dynamically weigh the importance of individual temporal and frequency patches.
* **Domain Generalization:** Integrates CORAL, MMD, and Domain Adversarial (Gradient Reversal Layer) losses to maintain robust performance across source and target domains.
* **Multi-Objective Contrastive Training:** Employs an NT-Xent loss formulation coupled with dynamic confidence weighting and center-pull penalties to create a highly separable latent space for normal sounds.
* **Non-Parametric Anomaly Scoring:** Evaluates embeddings using Mahalanobis and Cosine distances, ensuring accurate anomaly detection without relying on artificial geometric constraints.

---

## 📁 Repository Structure

* **`convert_rgb.py`** - Pre-processing script to convert raw `.wav` files into 224x224 RGB Log-Mel spectrograms using the near-field microphone channel.
* **`astra_attn_patch_dataset.py`** - Custom PyTorch Dataset class handling multi-scale patch extraction, domain labeling (source vs. target), and dynamic augmentations.
* **`patch_attn_model.py`** - The core architecture featuring a ResNet34 encoder, temporal attention networks, and a domain discriminator.
* **`attention_pooling.py`** - Attention-based pooling module with attribute-bias gating to fuse local patch embeddings into a global representation.
* **`train_1.py`** - The main training loop implementing the multi-objective contrastive loss, learning rate scheduling, and moving average (EMA) center tracking.
* **`evaluation4.py` & `pauc.py`** - Evaluation scripts to extract embeddings, fit covariance models (Ledoit-Wolf/Empirical), and compute final AUC, pAUC, and F1 scores.

---

## ⚙️ Installation & Requirements

Ensure you have Python 3.8+ installed. The primary dependencies are PyTorch, Librosa, and Scikit-Learn.


```

```text
File successfully created.

```bash
# Clone the repository
git clone [https://github.com/yourusername/dcase2026-task2-anomaly.git](https://github.com/yourusername/dcase2026-task2-anomaly.git)
cd dcase2026-task2-anomaly

# Install required packages
pip install torch torchvision torchaudio
pip install librosa soundfile pandas numpy matplotlib pillow scikit-learn tqdm

```

---

## 🛠️ Usage Pipeline

### 1. Data Preparation

Place the downloaded DCASE 2026 dataset in your root directory. The script expects the standard `train`, `test`, and `supplemental` folder structures for each machine type.

Run the spectrogram conversion to generate the `RGB_NEAR` datasets:

```bash
python convert_rgb.py

```

### 2. Training

Initiate the training sequence. The script will automatically loop through the defined machine types and apply the contrastive learning protocol. Checkpoints are saved periodically.

```bash
python train_1.py

```

*Note: Adjust `BATCH_SIZE`, `EPOCHS`, and `ROOT_DIR` in the config section of `train_1.py` according to your specific hardware constraints.*

### 3. Evaluation

Once training is complete, run the evaluation script to calculate the source and target metrics, including the final harmonic mean (Omega) of the AUC and pAUC scores.

```bash
python evaluation4.py

```

To run specific pAUC analytics:

```bash
python pauc.py

```

---

## 📊 Evaluation Metrics

The system is evaluated based on the official DCASE guidelines:

1. **AUC (Area Under the ROC Curve):** Calculated independently for both source and target domains.
2. **pAUC (Partial AUC):** Calculated over a constrained false positive rate (max_fpr = 0.1) to prioritize precision in high-stakes industrial environments.
3. **Harmonic Mean ($\Omega$):** The final aggregated metric across all machine types.

---

## 📝 Implementation Details (Paper Extract)

\subsection{Implementation Details}

The system is implemented in PyTorch and explicitly optimized for constrained hardware, with training conducted on an NVIDIA RTX 5060 Laptop GPU. To maintain computational efficiency, input Log-Mel spectrograms are partitioned into 32x32 patches with a stride of 16 (maximum 32 patches per sample). The network is trained jointly across all machine types for 40 epochs using the Adam optimizer (initial learning rate $2 \times 10^{-4}$, batch size 96) and regulated by a ReduceLROnPlateau scheduler (patience 10, decay factor 0.5). For the contrastive learning formulation, the NT-Xent loss temperature is fixed at $\tau=0.05$, with center representations for confidence weighting maintained via an Exponential Moving Average (momentum $\gamma=0.9$). Following an 8-epoch warmup phase, the objective is augmented with boundary ($\beta_{\text{boundary}}=0.02$) and center-pull ($\beta_{\text{center}}=0.05$) penalties, alongside dynamic confidence weighting ($w_{\text{hard}} \to 1.25$, $w_{\text{soft}} \to 0.9$), to progressively refine the representation space.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
"""

with open("dcase2026_documentation.md", "w", encoding="utf-8") as f:
f.write(content)
print("File successfully created.")

```


I have combined the `README.md` repository structure along with the `Implementation Details` paragraph into a single downloadable Markdown file. You can directly copy the text from it or save it straight to your GitHub repository! Let me know if you need any adjustments to the layout or content.

```
