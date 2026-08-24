# AIDER: Attention-Guided Deep Embeddings for Fine-Grained Insect Recognition

AIDER (**Attention-guided Insect Deep Embedding Recognition**) is a research framework for fine-grained insect recognition in cluttered field imagery. It studies how attention-guided representation refinement and downstream decision modeling affect recognition under matched experimental conditions.

This repository accompanies the manuscript:

> **Attention-Guided Deep Embeddings for Fine-Grained Insect Recognition**  
> Quoc-Anh Nguyen, Thanh-Nghi Doan, and Huu-Hoa Nguyen

The repository provides the source code, experiment configurations, data-partition metadata, and supporting materials associated with the study.

---

## Overview

Fine-grained insect recognition is challenging because closely related categories may differ only in localized morphological cues such as wing patterns, antennae, thorax markings, body contours, or texture. In field imagery, these cues can be weakened by background clutter, pose variation, scale changes, occlusion, and illumination differences.

AIDER examines two complementary parts of this problem. The first is **representation refinement**, where attention modules are integrated into pretrained visual backbones. The second is **decision modeling**, where the learned representation is classified either by the jointly trained Softmax head or by an independently fitted classifier after the encoder has been selected and frozen.

Three representation settings are considered:

- **Base**: the native backbone without an added attention module
- **ECA**: channel-only refinement using Efficient Channel Attention
- **CBAM**: channel-spatial refinement using the Convolutional Block Attention Module

The final feature map is globally pooled to obtain an image-level deep embedding. In the frozen-embedding route, this representation is standardized using training-set statistics and passed to an independent downstream classifier. This route changes the decision rule applied to the fixed representation without modifying the learned embedding.

---

## Experimental Setting

### Backbones

The study covers multiple CNN and Transformer backbone families.

| Backbone | `timm` identifier | IP102 input | Embedding dimension | Attention settings |
|---|---|---:|---:|---|
| MobileNetV3-Large | `mobilenetv3_large_100.ra_in1k` | 224 × 224 | 1280 | Base, CBAM |
| EfficientNet-B5 | `tf_efficientnet_b5.ra_in1k` | 456 × 456 | 2048 | Base, CBAM |
| EfficientNetV2-S | `tf_efficientnetv2_s.in1k` | 384 × 384 | 1280 | Base, ECA, CBAM |
| EfficientNetV2-M | `tf_efficientnetv2_m.in1k` | 384 × 384 | 1280 | Base, ECA, CBAM |
| ConvNeXt-S | `convnext_small.fb_in1k` | 224 × 224 | 768 | Base, CBAM |
| ConvNeXt-B | `convnext_base.fb_in1k` | 224 × 224 | 1024 | Base, CBAM |
| Swin-S | `swin_small_patch4_window7_224.ms_in1k` | 224 × 224 | 768 | Base, CBAM |

ECA uses the adaptive channel-kernel rule with \(\gamma=2\) and \(b=1\). CBAM uses reduction ratio \(r=16\) and a 7 × 7 spatial convolution. Attention preserves the native output dimension of each backbone.

### Datasets

IP102 is the primary benchmark. Xie24 and D0 provide secondary within-dataset evaluations.

| Dataset | Classes | Images | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| IP102 | 102 | 75,222 | 45,095 | 7,508 | 22,619 |
| Xie24 | 24 | 1,600 original images | 1,120 | 160 | 320 |
| D0 | 40 | 4,508 | 3,156 | 451 | 901 |

The official IP102 classification split is retained. Xie24 is partitioned at the original-image level before augmentation, while D0 uses duplicate-aware class-stratified partitioning. Exact and perceptual duplicate screening is applied as part of the integrity audit. No cross-partition exact or visually confirmed near-duplicate pairs were identified under the reported procedure.

The datasets are third-party research datasets and are not redistributed in this repository.

### Training and model selection

Neural models are trained for 100 epochs with AdamW and ImageNet-pretrained `timm` weights. The validation-best checkpoint is selected by macro F1-score. The main IP102 experiments use ten matched seeds, `202601` through `202610`.

The primary analysis compares AIDER-CBAM Softmax with the matched backbone-only Softmax model under ordinary cross-entropy without class weighting. Complementary experiments separate representation choice, downstream decision model, class weighting, and explicit representation regularization.

For frozen embeddings, non-Softmax hyperparameters are tuned independently for each encoder run using stratified five-fold cross-validation on the training embeddings only. The validation set is used for post-tuning classifier and route selection, and the test set is reserved for final evaluation.

---

## Primary Results on IP102

Across the six primary matched backbone comparisons, AIDER-CBAM improves accuracy by **1.87 to 3.92 percentage points** and macro F1-score by **2.24 to 4.70 points** over the corresponding backbone-only models.

| Backbone | Base Accuracy | CBAM Accuracy | Δ Accuracy | Base Macro F1 | CBAM Macro F1 | Δ Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV3-Large | 70.48 ± 0.34 | 72.85 ± 0.28 | +2.37 | 61.81 ± 0.52 | 64.75 ± 0.43 | +2.94 |
| EfficientNet-B5 | 74.63 ± 0.27 | 76.50 ± 0.22 | +1.87 | 65.81 ± 0.41 | 68.05 ± 0.34 | +2.24 |
| EfficientNetV2-M | 72.73 ± 0.31 | 76.65 ± 0.25 | +3.92 | 64.95 ± 0.46 | 69.65 ± 0.37 | +4.70 |
| ConvNeXt-S | 76.43 ± 0.22 | 78.85 ± 0.18 | +2.42 | 68.61 ± 0.33 | 71.68 ± 0.28 | +3.07 |
| ConvNeXt-B | 76.93 ± 0.20 | 80.13 ± 0.17 | +3.20 | 69.21 ± 0.30 | 73.15 ± 0.26 | +3.94 |
| Swin-S | 76.68 ± 0.23 | 79.43 ± 0.19 | +2.75 | 69.03 ± 0.34 | 72.28 ± 0.29 | +3.25 |

Values are mean ± standard deviation over ten matched runs. All six primary paired comparisons have positive 95% confidence intervals for both accuracy and macro F1-score and remain significant after Holm correction. A paired class-stratified bootstrap with 10,000 resamples provides complementary uncertainty estimates over the fixed test set.

The strongest observed IP102 configuration combines AIDER-CBAM with ConvNeXt-B and an inverse-frequency-weighted frozen Linear SVM, reaching **81.16% accuracy** and **74.25% macro F1-score**. The controlled analyses treat this as a combination of representation refinement, downstream decision modeling, and class-imbalance handling rather than as a single isolated effect.

---

## Controlled Analysis of the Learned Representation

A factorial analysis with EfficientNetV2-M evaluates Base, ECA, and CBAM representations with Softmax, Logistic Regression, and Linear SVM under unweighted and inverse-frequency-weighted conditions. Across matched settings, representation refinement produces the largest gain, while the frozen Linear SVM and class weighting provide smaller complementary improvements.

For example, under unweighted Softmax, macro F1-score increases from **64.95% for Base** to **69.65% for CBAM**. Applying an unweighted frozen Linear SVM to the same CBAM representation reaches **70.09%**, and inverse-frequency weighting raises the frozen-SVM result to **70.73%**. A parameter-matched non-attention residual control reaches **65.68% macro F1-score**, compared with **69.65%** for full CBAM, indicating that the observed gain is not explained by added trainable capacity alone in this controlled setting.

A complementary ConvNeXt-B experiment applies supervised-contrastive regularization to the representation. Macro F1-score increases from **73.15% to 73.93%** under Softmax, while the unweighted frozen Linear SVM reaches **74.16%** on the regularized representation. This result indicates that explicit representation regularization and downstream decision modeling can provide complementary gains.

Representation geometry is examined directly using within-class cosine distance, between-class centroid distance, class-centroid margin, macro 5-nearest-neighbor purity, and a class-balanced Fisher discriminant ratio. Across matched EfficientNetV2-M runs, all five measures improve from Base to ECA to CBAM. Within-class cosine distance decreases from **0.314 to 0.297 to 0.271**, while between-class centroid distance increases from **0.388 to 0.401 to 0.421** and macro 5-nearest-neighbor purity rises from **71.84% to 74.26% to 77.31%**.

---

## Spatial and Long-Tail Diagnostics

Spatial behavior is evaluated on the 5,800 IP102 test images that intersect the official bounding-box annotations. Across matched EfficientNetV2-M runs, the Grad-CAM pointing-hit rate increases from **67.4% for Base** to **72.8% for ECA** and **80.6% for CBAM**, while box-overlap energy increases from **0.492 to 0.538 to 0.624**. Box-region perturbation experiments show the same ordering, with stronger retention when annotated insect regions are preserved and greater sensitivity when they are removed.

The long-tail analysis divides the 102 IP102 classes into equal Head, Middle, and Tail groups by training frequency. Base-to-CBAM macro F1-score gains are **+2.40**, **+4.16**, and **+5.26 points**, respectively. Across all classes, the improvement has a moderate negative association with log training frequency, with Spearman \(\rho=-0.34\) and \(p<0.001\).

On the annotated subset, smaller insects remain more difficult, while the Base-to-CBAM macro F1-score gain is largest for the Small group. A structured audit of 200 misclassified images further identifies fine-grained visual ambiguity and weakly resolved, small, occluded, or blurred insects as the most frequent dominant error categories.

Together, these analyses connect predictive performance with representation structure, spatial behavior, and residual difficulty within the evaluated benchmark conditions.

---

## Computational Profile

Batch-1 latency is measured on an NVIDIA RTX 4070 Ti Super 16 GB using 200 warm-up predictions followed by 1,000 synchronized timed predictions.

| ConvNeXt-B configuration | Latency (ms/image) |
|---|---:|
| Backbone-only | 4.72 ± 0.08 |
| AIDER-CBAM Softmax | 5.28 ± 0.09 |
| AIDER-CBAM frozen SVM | 5.39 ± 0.09 |

For ConvNeXt-B, CBAM adds 0.56 ms/image relative to the backbone-only model. The complete frozen-SVM route adds a further 0.11 ms/image relative to AIDER-CBAM Softmax. The frozen route additionally requires the stored training-derived scaler and external classifier.

---

## Reproducibility

The reference environment uses Ubuntu 22.04.5 LTS, Python 3.11.11, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, `timm` 1.0.14, scikit-learn 1.6.1, CatBoost 1.2.8, CUDA 12.4, and an NVIDIA RTX 4070 Ti Super 16 GB GPU.

Neural training uses AdamW with a base learning rate of \(5\times10^{-4}\), five-epoch linear warmup, cosine decay to \(10^{-6}\), FP16 automatic mixed precision, and 100 training epochs with early stopping disabled. The ten matched seeds are `202601` through `202610`.

Detailed augmentation settings, classifier grids, backbone-specific attention insertion points, split manifests, statistical analyses, dataset-integrity records, and diagnostic protocols are provided in the accompanying Supplementary Material.

---

## Scope and Data Availability

The main conclusions are based on controlled experiments within IP102. Xie24 and D0 provide additional within-dataset evidence. Evaluation under new cameras, locations, illumination conditions, background distributions, insect populations, and acquisition processes remains an important direction for external field validation and domain adaptation. Further work also includes improved recognition of rare classes and small objects, together with direct evaluation on mobile and edge platforms.

IP102, Xie24, and D0 were obtained from their original sources and are not redistributed with this repository. Access and reuse remain subject to the terms specified by their respective providers. Source code, experiment configurations, split metadata, and supporting materials associated with this study are maintained in the repository.

---

## Citation

Citation information will be updated when the associated article is published.

For now, please cite the manuscript as:

> Quoc-Anh Nguyen, Thanh-Nghi Doan, and Huu-Hoa Nguyen  
> **“Attention-Guided Deep Embeddings for Fine-Grained Insect Recognition.”**

