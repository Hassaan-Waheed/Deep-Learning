\# Autonomous Driving: SOTA Semantic Segmentation with U-Net \& Focal Loss



\## 📌 Project Overview

This repository contains an industry-standard implementation of \*\*Semantic Segmentation\*\* for autonomous vehicle perception using the \*\*CamVid\*\* dataset. To move beyond baseline performance, this project employs \*\*Focal Loss\*\* and \*\*OneCycleLR Scheduling\*\* to specifically address extreme class imbalance and enhance safety-critical object detection (e.g., pedestrians).







\## 🚀 Key Features \& SOTA Techniques

\* \*\*Architecture:\*\* Symmetric \*\*U-Net\*\* with skip connections for high-fidelity spatial reconstruction.

\* \*\*Loss Function:\*\* Custom \*\*Focal Loss ($\\gamma=2$)\*\* implementation to perform "Online Hard Example Mining," forcing the model to focus on sparse, difficult classes.

\* \*\*Optimization:\*\* \*\*OneCycleLR Scheduler\*\* for faster convergence and supercharged learning during the mid-training phase.

\* \*\*Augmentation:\*\* Professional pipeline using \*\*Albumentations\*\* (Affine transforms, Brightness/Contrast, Normalization).

\* \*\*Efficiency:\*\* Integrated \*\*Mixed Precision (FP16)\*\* training for 2x faster throughput on NVIDIA T4 GPUs.



\## 📊 Performance Metrics (IEEE Standard)

The model was evaluated using a multi-metric approach to ensure both global accuracy and local object integrity.



| Metric | Score | Industry Significance |

| :--- | :--- | :--- |

| \*\*Global Pixel Accuracy\*\* | \*\*83.94%\*\* | Demonstrates excellent global scene parsing. |

| \*\*Mean IoU (mIoU)\*\* | \*\*0.4548\*\* | Competitive performance across 32 distinct urban classes. |

| \*\*Frequency Weighted IoU\*\* | \*\*0.7404\*\* | High reliability on dominant road/building features. |

| \*\*Mean F1 (Dice Score)\*\* | \*\*0.5637\*\* | Balanced Precision/Recall for spatial masks. |

| \*\*Pedestrian IoU\*\* | \*\*0.7223\*\* | \*\*Elite Performance:\*\* Superior hazard detection. |







\## 🛠️ Tech Stack

\* \*\*Framework:\*\* PyTorch (v2.x)

\* \*\*Data Augmentation:\*\* Albumentations

\* \*\*Visualization:\*\* Matplotlib / OpenCV

\* \*\*Hardware:\*\* NVIDIA T4 GPU (CUDA)



\## 📂 Project Structure

```text

├── model.py            # U-Net Architecture

├── data\_setup.py       # SOTA Albumentations Data Pipeline

├── train.py            # Focal Loss + OneCycleLR Training Engine

├── evaluate.py         # Multi-metric Quantitative Analysis (mIoU, FWIoU, Acc, F1)

└── unet\_camvid\_elite.pth # Final Trained Weights


\## Technical Reflection
While standard Cross-Entropy often overlooks minority classes, the Focal Loss strategy implemented here effectively "muted" the majority class noise (Road/Sky). This resulted in a Pedestrian IoU of 0.72, significantly higher than the mean, proving that the model is optimized for Safety-Critical decision making in autonomous driving scenarios. Unlike simple class weighting, Focal Loss dynamically handles "hard examples," leading to more stable gradients and superior boundary definition.
