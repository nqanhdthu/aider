# AIDER: Attention-Guided Deep Embeddings for Fine-Grained Insect Recognition

**AIDER** (Attention-guided Insect Deep Embedding Recognition) is a controlled empirical framework for fine-grained insect recognition in cluttered field imagery. It studies attention-guided representation refinement together with downstream decision modeling on globally pooled deep visual embeddings.

This repository accompanies the manuscript **“Attention-Guided Deep Embeddings for Fine-Grained Insect Recognition”** and provides source code, experiment configurations, data-partition manifests, integrity-audit records, and numerical results supporting the reported analyses.

## Overview

Fine-grained insect recognition is challenging because discriminative morphological cues may be small, visually similar across classes, or obscured by background clutter, pose variation, and occlusion. AIDER evaluates whether channel and channel-spatial attention can refine learned visual representations under controlled backbone-matched conditions, while separately examining how downstream decision models and class-imbalance handling affect recognition performance.

The study uses **IP102** as the primary benchmark. **Xie24** and **D0** are included as secondary within-dataset evaluations rather than as evidence of external-domain generalization.

## Highlights

- **Controlled representation refinement.** Backbone-only, ECA, and CBAM representations are evaluated under matched experimental settings, with the primary analysis comparing AIDER-CBAM Softmax against the corresponding backbone-only Softmax model across six CNN and Transformer backbone families.
- **Separated representation and decision effects.** Factorial and classifier controls distinguish attention-guided representation refinement from downstream classification and class-imbalance handling.
- **Multi-level evaluation.** The study includes paired statistical inference, embedding-geometry analysis, supervised-contrastive representation-learning controls, spatial diagnostics, long-tail and object-size analyses, structured error analysis, dataset-integrity auditing, and computational profiling.

Across the six primary matched backbone comparisons on IP102, **AIDER-CBAM Softmax improves accuracy by 1.87–3.92 percentage points and macro F1-score by 2.24–4.70 points** over the corresponding backbone-only Softmax models. Both members of each primary comparison use ordinary cross-entropy without class weighting.

## Repository Structure

~~~text
.
├── code/
│   ├── configs/
│   │   ├── training.yaml
│   │   ├── models.yaml
│   │   ├── datasets.yaml
│   │   └── classifiers.yaml
│   └── *.py
├── data/
│   ├── ip102_train.txt
│   ├── ip102_val.txt
│   ├── ip102_test.txt
│   ├── xie24_split_manifest.csv
│   ├── d0_split_manifest.csv
│   ├── ip102_bbox_test_subset.csv
│   └── integrity-audit files
├── results/
│   └── experimental and diagnostic result files
└── README.md
~~~

The repository is organized around three complementary groups of research artifacts:

- `code/` contains core implementation and analysis utilities together with the experiment specifications used in the study.
- `data/` contains fixed partition manifests and metadata supporting dataset-integrity and spatial analyses.
- `results/` contains run-level and summary numerical artifacts associated with the experiments and analyses reported in the manuscript and Supplementary Material.

## Experiment Configuration

The principal experimental specifications are recorded in `code/configs/`:

- `training.yaml` documents the software environment, optimization settings, data augmentation, repeated-run seeds, and training controls.
- `models.yaml` records backbone identifiers, input resolutions, embedding dimensions, normalization settings, and attention configurations.
- `datasets.yaml` records dataset partition information and duplicate-audit settings.
- `classifiers.yaml` records the training-only cross-validation protocol and frozen-embedding classifier settings.

The main experiments use ten matched seeds to support paired comparisons across configurations. Non-Softmax classifier hyperparameters are selected using stratified five-fold cross-validation on training embeddings only. Standardization statistics are also estimated from training embeddings only and then applied unchanged to validation and test embeddings.

## Data and Integrity

The image datasets used in this study are third-party research datasets and are **not redistributed** in this repository. Access and reuse remain subject to the terms specified by their original providers.

The repository provides the split and audit information used for the reported experiments, including:

- the official IP102 classification partitions used in the primary evaluation
- fixed Xie24 and D0 split manifests
- duplicate-audit records based on exact and perceptual-image checks
- cross-partition integrity summaries
- metadata for the bounding-box-annotated IP102 test subset used in the spatial diagnostics

For Xie24 and D0, duplicate grouping is performed before partitioning. The official IP102 classification split is retained and audited post hoc. Within the image-level information available from the source datasets, no cross-partition byte-identical images or visually confirmed near-duplicate pairs were identified in the evaluated partitions.

## Results and Supporting Analyses

The `results/` directory provides numerical artifacts supporting the main and supplementary analyses. These include:

- primary matched backbone comparisons
- attention and classifier factorial controls
- weighted and unweighted classifier analyses
- supervised-contrastive controls
- paired run-level differences
- embedding-geometry diagnostics
- bounding-box localization and perturbation diagnostics
- class-frequency and object-size analyses
- structured error summaries
- computational and resource measurements

The primary scientific comparison is the matched **AIDER-CBAM Softmax versus backbone-only Softmax** evaluation on IP102. Frozen-embedding classifier results are complementary downstream-route analyses and are kept separate from the primary representation-refinement comparison.

## Reproducibility Scope

The repository is intended to provide the implementation records and supporting artifacts needed to reproduce and inspect the experimental protocol reported in the associated manuscript.

Exact experimental settings should be taken from the configuration files and the accompanying manuscript and Supplementary Material. The result files preserve the numerical evidence underlying the reported aggregate comparisons and diagnostic analyses.

The repository does not redistribute the original image datasets, and the reported conclusions are limited to the evaluated benchmark conditions. In particular, the Xie24 and D0 experiments are within-dataset evaluations and should not be interpreted as cross-domain or field-deployment validation.

## Associated Manuscript

**Quoc-Anh Nguyen, Thanh-Nghi Doan, and Huu-Hoa Nguyen**

**“Attention-Guided Deep Embeddings for Fine-Grained Insect Recognition”**

Manuscript citation information will be updated upon publication.