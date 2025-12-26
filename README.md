# WaveletResfusion: Physics-Guided Wavelet Residual Diffusion for Heavy Rain Restoration

This repository provides the **Core Implementation** of **WaveletResfusion**, as described in our manuscript. We have curated this codebase to highlight the essential algorithmic innovations: the **Heteroscedastic Forward Diffusion** and the **Hierarchical Wavelet Architecture**.

---

## 📂 Core Code Navigation

To facilitate a quick and efficient review process, we provide the implementation of the core modules within the original file structure:

| Module                          | Key Implementation | Corresponding Manuscript Section |
|:--------------------------------| :--- | :--- |
| **`model/waveletresfusion.py`** | Heteroscedastic Forward Diffusion & Multi-domain Loss | **Section 3.2 & 3.4** |
| **`model/UNet_wave.py`**        | Hierarchical Wavelet Block (HWB) & Cross-Scale Fusion (CSF) | **Section 3.3** |
| **`utils/dwt.py`**              | Differentiable DWT/IWT Subband Factorization | **Section 3.1** |

---

## 🚀 Key Technical Highlights

### 1. Heteroscedastic Forward Diffusion (Physics-Guided)
Unlike standard DDPMs that use spatially homogeneous noise, our framework introduces a **physics-guided heteroscedastic noise weighting** mechanism ($w_{sb}$).
- **Mechanism**: We modulate noise levels according to the energy statistics of wavelet subbands, specifically targeting the directional rain streaks in high-frequency components while preserving low-frequency structural priors.
- **Code Reference**: See `get_dynamic_weights()` and the subband noise synthesis logic in `training_step()` of `resfusion_restore.py`.

### 2. Hierarchical Wavelet Architecture (HWB & CSF)
The backbone is designed to process heavy rain in a "divide-and-conquer" manner across different frequency domains:
- **Hierarchical Wavelet Block (HWB)**: Decomposes features into subbands and applies orientation-specific convolutions to capture the physical geometry of rain streaks.
- **Cross-Scale Fusion (CSF)**: Aligns and fuses features across different scales of the WaveUnet to ensure texture consistency.
- **Code Reference**: See `HierarchicalWaveBlock` and `CrossScaleFusion` classes in `WaveUnet.py`.



### 3. Multi-Domain Objective Function
To address the "spectral blurring" and structural distortion caused by heavy rain, we implement a joint loss function:
1. **Subband Loss**: Adaptive L1 regularization across wavelet subbands.
2. **Gradient Loss**: Enforces structural sharpness using Sobel-based consistency (see `compute_gradient_loss`).
3. **Frequency Loss**: Regularizes spectral statistics via FFT to ensure global texture stability (see `compute_frequency_loss`).

---

## 🛠 Implementation Details

The framework is built upon **PyTorch Lightning** for transparency and reproducibility.

- **Efficient Inference**: The `generate()` function implements our **Accelerated 5-Step Sampling** strategy. By leveraging the residual prior at the **Acceleration Point ($T_{acc}$)**, we significantly reduce inference time while maintaining high-fidelity restoration.
- **Differentiable DWT**: The `utils/dwt.py` provides custom layers that allow the wavelet transform to be fully integrated into the end-to-end backpropagation process.

---

> **Note**: This repository contains the core algorithmic logic and framework structure to support the peer-review process. The complete codebase, including data preprocessing pipelines, full training scripts, and pre-trained checkpoints, will be released upon formal acceptance of the paper.