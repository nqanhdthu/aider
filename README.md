# AIDER: Attention-Guided Deep Embeddings for Fine-Grained Insect Recognition

AIDER (**Attention-guided Insect Deep Embedding Recognition**) is a research framework for fine-grained insect recognition in cluttered field imagery. It studies two complementary aspects of recognition under matched experimental conditions:

1. **Attention-guided representation refinement**, which aims to emphasize discriminative insect morphology while reducing the influence of distracting background regions.
2. **Downstream decision modeling**, which evaluates whether classifiers operating on frozen deep embeddings can improve prediction without modifying the learned encoder representation.

This repository accompanies the manuscript:

> **Attention-Guided Deep Embeddings for Fine-Grained Insect Recognition**  
> Quoc-Anh Nguyen, Thanh-Nghi Doan, and Huu-Hoa Nguyen

---

## Overview

Fine-grained insect recognition is challenging because visually related categories may differ only in small morphological cues such as wing patterns, antennae, thorax markings, body contours, or localized textures. In field images, these cues can be obscured by background clutter, pose variation, scale changes, occlusion, and illumination differences.

AIDER provides a controlled experimental framework for studying how attention-guided feature refinement and downstream decision rules affect recognition performance.

The framework supports three representation settings:

- **Base**: the native backbone without an added AIDER attention module
- **ECA**: channel-only refinement using Efficient Channel Attention
- **CBAM**: channel-spatial refinement using the Convolutional Block Attention Module

The resulting feature maps are globally pooled to obtain image-level deep embeddings.

AIDER then supports two prediction routes:

- **End-to-end Softmax**, where the encoder and Softmax classifier are trained jointly
- **Frozen-embedding classification**, where the validation-selected encoder is frozen, embeddings are standardized using training-set statistics, and independent classifiers are fitted to the fixed representation

The frozen-embedding route changes the downstream decision rule but does not modify the learned embedding geometry.

---

## Supported Backbones

The study evaluates AIDER across multiple CNN and Transformer backbone families:

- MobileNetV3-Large
- EfficientNet-B5
- EfficientNetV2-S
- EfficientNetV2-M
- ConvNeXt-S
- ConvNeXt-B
- Swin-S

Attention preserves the native output dimension of each backbone.

---

## Downstream Classifiers

The frozen-embedding evaluation includes:

- Logistic Regression
- Linear SVM
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- CatBoost

Weighted and unweighted variants are evaluated separately where applicable.

All non-Softmax hyperparameters are selected independently for each encoder run using stratified five-fold cross-validation on the training embeddings only. Standardization is fitted independently within each cross-validation training fold.

---

## Representation-Learning Control

Standard AIDER learns pooled encoder representations using cross-entropy.

To provide an explicit representation-learning reference, the study also evaluates supervised-contrastive regularization with AIDER-CBAM and ConvNeXt-B. The control uses two augmented views and a training-only projection head. The projection head is discarded before evaluation, and final Softmax and frozen-SVM predictions operate on the original pooled ConvNeXt-B embedding.

This experiment distinguishes:

- attention-guided representation refinement
- explicit representation regularization
- downstream decision modeling after representation learning

---

## Datasets

The experiments use three insect image benchmarks.

### IP102

IP102 is the primary benchmark and contains:

- **102 classes**
- **75,222 images**
- **45,095 training images**
- **7,508 validation images**
- **22,619 test images**

The official classification split is retained.

IP102 is strongly long-tailed and contains substantial background clutter, fine-grained inter-class similarity, and intra-class variation.

### Xie24

Xie24 contains:

- **24 classes**
- **1,600 original images**
- **1,120 training images**
- **160 validation images**
- **320 test images**

Partitioning is performed at the original-image level before augmentation.

### D0

D0 contains:

- **40 classes**
- **4,508 images**
- **3,156 training images**
- **451 validation images**
- **901 test images**

Xie24 and D0 are used as secondary within-dataset evaluations rather than as external-domain validation.

---

## Data Integrity

Dataset integrity is handled explicitly.

- Exact duplicates are screened using SHA-256 hashes.
- Near-duplicate candidates are generated using perceptual hashing and then visually verified.
- Confirmed duplicate relations are grouped before partitioning for Xie24 and D0.
- Training augmentation is applied only after partitioning.
- The official IP102 split is retained and audited post hoc.

No cross-partition exact or visually confirmed near-duplicate pairs were identified under the reported audit procedure.

The image datasets themselves are third-party research datasets and are **not redistributed in this repository**.

---

## Experimental Protocol

The main experiments use:

- ImageNet-pretrained `timm` backbones
- AdamW optimization
- 100 training epochs
- validation macro F1-score for checkpoint selection
- ten matched random seeds: `202601` through `202610`
- accuracy, macro precision, macro recall, and macro F1-score as evaluation metrics

The main analysis compares AIDER-CBAM Softmax with the matched backbone-only Softmax model under ordinary cross-entropy without class weighting.

Additional analyses separate:

- representation refinement
- downstream classifier choice
- class-imbalance handling
- explicit representation regularization

---

## Class-Imbalance Controls

The study evaluates several imbalance-handling strategies:

- ordinary cross-entropy
- inverse-frequency weighted cross-entropy
- Balanced Softmax
- focal loss
- weighted and unweighted Logistic Regression
- weighted and unweighted Linear SVM

Class-frequency quantities are computed from the training split only.

---

## Statistical Analysis

The primary IP102 comparisons use matched random seeds and paired statistical analysis.

Reported analyses include:

- paired mean differences
- 95% confidence intervals
- two-sided paired t-tests
- paired-sample effect sizes
- Holm correction for multiple comparisons
- paired class-stratified bootstrap analysis with 10,000 resamples

The bootstrap analysis complements run-level uncertainty by evaluating finite-test-sample variability while preserving class structure and prediction pairing.

---

## Representation Analysis

The study directly evaluates the geometry of the learned embeddings using:

- within-class cosine distance
- between-class centroid distance
- class-centroid margin
- macro 5-nearest-neighbor purity
- class-balanced Fisher discriminant ratio

These analyses compare Base, ECA, and CBAM representations under matched conditions.

---

## Spatial Diagnostics

Spatial behavior is examined using both qualitative and quantitative analyses.

The quantitative evaluation uses the subset of IP102 test images with official bounding-box annotations and reports:

- Grad-CAM pointing-hit rate
- box-overlap energy
- box-region-only prediction retention
- box-masked prediction retention

These diagnostics assess alignment with annotated insect regions within the evaluated benchmark. They are not interpreted as external-domain validation or as proof of pixel-level localization.

---

## Long-Tail and Error Analysis

The study further analyzes:

- Head, Middle, and Tail class groups
- class-frequency relationships
- object-size effects
- confusion patterns
- structured residual errors

These analyses characterize where recognition remains difficult, particularly for visually similar categories, low-frequency classes, and small-object cases.

---

## Key Results

Across the six primary matched backbone comparisons on IP102, AIDER-CBAM Softmax improves performance over the corresponding backbone-only Softmax models by:

- **1.87 to 3.92 percentage points in accuracy**
- **2.24 to 4.70 percentage points in macro F1-score**

For AIDER-CBAM with ConvNeXt-B:

| Decision setting | Accuracy (%) | Macro F1 (%) |
|---|---:|---:|
| Softmax, unweighted CE | 80.13 | 73.15 |
| Unweighted frozen Linear SVM | 80.78 | 73.59 |
| Inverse-frequency weighted frozen Linear SVM | 81.16 | 74.25 |

The controlled analyses indicate that attention-guided representation refinement is the dominant source of improvement, while the frozen-embedding route and class-imbalance handling provide smaller complementary gains.

---

## Computational Profile

Resource measurements are reported on an NVIDIA RTX 4070 Ti Super with batch size 1.

For ConvNeXt-B:

| Configuration | Latency (ms/image) |
|---|---:|
| Backbone-only | 4.72 ± 0.08 |
| AIDER-CBAM Softmax | 5.28 ± 0.09 |
| AIDER-CBAM frozen SVM | 5.39 ± 0.09 |

The frozen route additionally requires the stored training-derived scaler and downstream classifier.

---

## Reproducibility Environment

The reference experiments use:

```text
Ubuntu 22.04.5 LTS
Python 3.11.11
PyTorch 2.6.0+cu124
torchvision 0.21.0+cu124
timm 1.0.14
scikit-learn 1.6.1
CatBoost 1.2.8
Pillow 11.1.0
ImageHash 4.3.2
CUDA 12.4
```

Hardware:

```text
Intel Core i7-13700K
32 GB system RAM
NVIDIA RTX 4070 Ti Super, 16 GB
```

Reproducibility is defined with respect to the documented environment, data partitions, random seeds, model configurations, and controlled random-number initialization. Bitwise-identical execution across different software releases, CUDA versions, or hardware platforms is not assumed.

---

## Usage

Before running experiments, prepare the required third-party datasets according to their original distribution terms and reproduce the dataset splits using the provided partition metadata.

Experiment settings should follow the configuration associated with the selected backbone, attention variant, decision route, and class-imbalance condition.

For frozen-embedding experiments:

1. Train the neural encoder and select the validation-best checkpoint.
2. Freeze the selected encoder.
3. Extract training, validation, and test embeddings.
4. Fit feature standardization using training embeddings only.
5. Tune non-Softmax classifiers using stratified five-fold cross-validation on training embeddings only.
6. Refit the selected classifier variant on the complete training embeddings.
7. Use the validation set for post-tuning classifier and route selection.
8. Evaluate the fixed configuration on the test set.

---

## Scope and Limitations

AIDER is a controlled empirical framework built from established attention, backbone, and classifier components. It does not introduce a new attention operator, embedding loss, optimization algorithm, or classifier.

The conclusions are restricted to the evaluated benchmark conditions. The current experiments do not establish robustness to changes in:

- camera hardware
- geographic location
- illumination
- background distribution
- insect population
- acquisition process

External field validation, domain adaptation, improved rare-class recognition, and deployment on mobile or edge hardware remain important directions for future work.

---

## Citation

Citation information will be updated when the associated article is published.

For now, please cite the manuscript as:

> Quoc-Anh Nguyen, Thanh-Nghi Doan, and Huu-Hoa Nguyen,  
> **“Attention-Guided Deep Embeddings for Fine-Grained Insect Recognition.”**

---

