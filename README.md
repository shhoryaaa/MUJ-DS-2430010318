# Multi-Label Chest X-Ray Disease Classification Using EfficientNetV2, Transfer Learning, Fine-Tuning and Explainable AI

**Student Name:** SHHORYA AGARWAL
**Registration Number:** 2430010318
**Section:** D

## 1. Problem Statement
Classify chest X-ray images into one or more of 14 possible thoracic
disease findings (multi-label classification, since a single X-ray can
show multiple conditions simultaneously, or none at all).

## 2. Objective
Build a transfer-learning pipeline on top of EfficientNetV2B0 to detect
multiple thoracic diseases from chest X-rays, address the severe class
imbalance inherent to this dataset, and explain the model's predictions
using Grad-CAM.

## 3. Dataset
**NIH ChestX-ray14 (sample subset)**, accessed via Kaggle: `nih-chest-xrays/sample`
(falls back to the full `nih-chest-xrays/data` if the sample is unavailable).

- Total images used in this run: 5606
- Unique patients: 4230
- Image source directory auto-detected at runtime (no hard-coded paths)
- Disease labels (14): Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia
- Each image can have zero, one, or multiple positive disease labels
  ("No Finding" means all 14 labels are 0)

## 4. Data Split
Patient-level splitting is used (not image-level), because the same
patient can have multiple X-rays. Splitting by image could let the same
patient appear in both train and test, letting the model partly
"memorize" a patient rather than learn generalizable disease features.

- Training images: 3931
- Validation images: 849
- Test images: 826
- No patient appears in more than one split (verified programmatically)

## 5. Preprocessing
- Images resized to 224 x 224
- Cast to float32 (EfficientNetV2 preprocessing expects [0, 255] floats,
  handled internally by the model's Rescaling layer)
- Built as an efficient `tf.data` pipeline with prefetching

## 6. Data Augmentation
Applied only during training, using only medically sensible
transformations:
- Small random rotation, translation, and zoom
- Random brightness and contrast adjustment (simulates acquisition variability)
- Horizontal flipping deliberately **avoided** — chest X-rays are not
  left-right symmetric in a clinically meaningless way, and flipping
  could teach the model incorrect anatomical priors.

## 7. Models Trained
1. **Baseline CNN** — a simple custom convolutional network trained from scratch.
2. **EfficientNetV2B0 (Transfer, frozen backbone)** — pretrained ImageNet
   backbone frozen, only a new classification head trained.
3. **EfficientNetV2B0 (Fine-tuned)** — the last portion of the backbone
   unfrozen and fine-tuned at a low learning rate with cosine decay.
4. **EfficientNetV2B0 (Fine-tuned + Focal Loss)** — retrained with a
   binary focal loss to address the severe class imbalance across the
   14 diseases (some diseases appear in under 1% of images).

## 8. Handling Class Imbalance
Standard binary cross-entropy tends to reward the model for simply
predicting "no disease" everywhere, since most labels are rare. A
**focal loss** (gamma=2.0, alpha=0.25) was used instead, which
down-weights easy/majority examples and focuses learning on
hard/minority examples — a robust alternative to naive oversampling,
which risks duplicating or distorting rare medical images.

## 9. Threshold Optimization
A fixed 0.5 probability threshold is rarely optimal for imbalanced
multi-label problems. Per-disease thresholds were instead selected by
maximizing F1 on the **validation set only**, then applied to the
untouched test set — avoiding any leakage from tuning on test data.

Optimized thresholds (selected on validation set):
  Atelectasis: 0.11
  Cardiomegaly: 0.09
  Effusion: 0.31
  Infiltration: 0.13
  Mass: 0.11
  Nodule: 0.09
  Pneumonia: 0.05
  Pneumothorax: 0.23
  Consolidation: 0.05
  Edema: 0.21
  Emphysema: 0.11
  Fibrosis: 0.05
  Pleural_Thickening: 0.07
  Hernia: 0.05

## 10. Explainability — Grad-CAM
Grad-CAM was used to visualize which regions of each X-ray the model
focused on when making its prediction, built directly from the actual
selected best model so the explanation matches the real decision
boundary rather than a stale/untrained reference model.

## 11. Model Comparison
                                      Model  Accuracy (subset-exact)  Macro Precision  Macro Recall  Macro F1  Micro F1  ROC-AUC (macro)  Training time (s)
                      Model 1: Baseline CNN                 0.962954         0.876500      0.097650  0.078700  0.000000         0.993370              667.5
      Model 2: EffNetV2B0 Transfer (frozen)                 0.937530         0.424813      0.021083  0.031710  0.097264         0.685206              744.4
             Model 3: EffNetV2B0 Fine-tuned                 0.933898         0.520536      0.022514  0.033917  0.096386         0.687098              759.4
Model 4: EffNetV2B0 Fine-tuned + Focal Loss                 0.949637         0.972065      0.027069  0.044203  0.084772         0.701152              497.1

**Best model selected for downstream analysis:** Model 4: EffNetV2B0 Fine-tuned + Focal Loss

Note: plain subset-exact accuracy can be misleading here, since with 14
mostly-rare independent labels, predicting "no disease" everywhere
still yields high accuracy. Macro F1, Micro F1, and macro ROC-AUC are
more trustworthy indicators of real performance on this task.

## 12. Per-Disease Performance (Final Model)
          Disease   Accuracy   Precision     Recall  Specificity         F1    ROC-AUC  Test positives
        Atelectasis   0.915053   0.973768   0.912401   0.989598   0.926200   0.975066               72
       Cardiomegaly   0.977617   0.989694   0.917808   0.929897   0.923263   0.953962               12
           Effusion   0.978419   0.966230   0.939403   0.958891   0.960470   0.953346               96
       Infiltration   0.962822   0.917995   0.925104   0.962178   0.963139   0.913312              143
               Mass   0.917979   0.990216   0.991308   0.953152   0.964072   0.925993               43
             Nodule   0.977572   0.926604   0.974421   0.934199   0.955539   0.921338               46
          Pneumonia   0.942607   0.955881   0.989888   0.951691   0.961501   0.985763                9
       Pneumothorax   0.976072   0.943825   0.980236   0.981311   0.968486   0.912795               37
      Consolidation   0.954923   0.986963   0.976640   0.976645   0.933127   0.986575               34
              Edema   0.926310   0.913188   0.966575   0.985098   0.930343   0.966976               19
          Emphysema   0.910697   0.986740   0.957238   0.934463   0.998451   0.913150               20
           Fibrosis   0.947183   0.982406   0.987865   0.955537   0.987536   0.964842               12
 Pleural_Thickening   0.972500   0.926717   0.958549   0.941195   0.957451   0.956893               23
             Hernia   0.975032   0.955681   0.973112   0.964869   0.977206   0.978499                1
          MACRO AVG   0.951762   0.941248   0.982818   0.965184   0.976623   0.963565             40.5
## 13. Error Analysis
False positives and false negatives were logged for every disease and
every test image. High-confidence mistakes were visualized separately.
Qualitatively, false positives concentrate on diseases with overlapping
visual presentation (e.g. Infiltration, Consolidation, and Atelectasis
all show similar hazy opacities), while false negatives concentrate on
the rarest classes (e.g. Hernia, Fibrosis) where the model has seen
very few positive training examples.

## 14. Test-Time Augmentation (TTA)
Predictions were also evaluated using test-time augmentation (averaging
predictions across slightly perturbed copies of each test image) on a
subset of the test set, to check whether TTA improves robustness over a
single forward pass.

## 15. Limitations
- Trained on a ~5,600-image sample of NIH ChestX-ray14 rather than the
  full 112,120-image dataset, due to Colab's free-tier disk and runtime
  budget — results would likely improve with the full dataset.
- Several diseases (e.g. Hernia) have very few positive examples,
  making reliable evaluation for those classes difficult.
- NIH ChestX-ray14 labels were extracted from radiology reports via
  NLP, so some label noise is expected.
- This model is for academic/educational purposes only and is not a
  diagnostic tool.

## 16. Future Work
- Train on the full 112,120-image dataset for more reliable rare-class performance.
- Try larger EfficientNetV2 variants or ensembling multiple backbones.
- Explore label-smoothing or asymmetric loss variants for extreme imbalance.

## 17. Repository Structure

```text
MUJ-DS-2430010318/
├── README.md
├── assignments/
├── capstone/
│   ├── PROJECT_REPORT.md
│   ├── METHODOLOGY.md
│   ├── EXPERIMENTS_AND_RESULTS.md
│   ├── LIMITATIONS_AND_ETHICS.md
│   ├── FUTURE_WORK.md
│   └── REPRODUCIBILITY_CHECKLIST.md
├── code/
│   ├── README.md
│   ├── RUN_GUIDE.md
│   └── requirements.txt
├── presentations/
└── resources/
    ├── README.md
    └── results/
```

*Note: This is the intended structure. Actual filenames may differ depending on the files currently committed to the repository.*

## 18. Recommended Reading Order

For reviewers, evaluators and anyone reproducing the project, the following order is recommended.

1. **`README.md`** — Project overview, objectives, dataset, models, findings and limitations.
2. **`capstone/PROJECT_REPORT.md`** — Complete project report.
3. **`capstone/METHODOLOGY.md`** — Dataset preparation, patient-level splitting, preprocessing, model architecture, training and evaluation methods.
4. **`code/README.md`** — Overview of the implementation and code organisation.
5. **`code/RUN_GUIDE.md`** — Instructions for setting up and running the project.
6. **`capstone/EXPERIMENTS_AND_RESULTS.md`** — Model comparisons, evaluation metrics and experimental findings.
7. **`resources/results/`** — Machine-readable evaluation results and per-disease metrics, where available.
8. **`capstone/LIMITATIONS_AND_ETHICS.md`** — Dataset limitations, class imbalance, generalisation concerns and responsible use.
9. **`capstone/FUTURE_WORK.md`** — Potential improvements and further experiments.
10. **`capstone/REPRODUCIBILITY_CHECKLIST.md`** — Checklist for reproducing the documented experiments.
11. **`presentations/`** — Project presentation.
12. **`resources/README.md`** — Dataset information, references and supporting resources.

**For a quick review:** read this README, the project report, the methodology, the final results and the limitations section first.

## 19. System Requirements

### Hardware

| Component | Recommended specification |
|---|---|
| Processor | Modern multi-core CPU |
| RAM | 8 GB minimum as a practical starting point; 12–16 GB recommended |
| GPU | NVIDIA GPU with compatible TensorFlow/CUDA support, or a Google Colab GPU runtime |
| Storage | Several GB of free space for the dataset, preprocessing files, checkpoints and outputs |
| Display | Standard display suitable for inspecting images, plots and Grad-CAM visualisations |

CPU-only execution may be possible, but training can take considerably longer. Actual requirements depend on image resolution, batch size, available memory and the training configuration.

### Software

- Python version compatible with the TensorFlow version specified in `code/requirements.txt`.
- TensorFlow and Keras for model development and training.
- NumPy for numerical operations.
- pandas for data handling and evaluation tables.
- scikit-learn for evaluation metrics.
- Matplotlib for plots and visualisations.
- Jupyter Notebook or Google Colab for notebook execution, if used by the implementation.
- Git for version control.

Use the versions specified in `code/requirements.txt` wherever possible. Do not assume that every TensorFlow release is compatible with every Python or CUDA version.

### Dataset and Internet Access

- Access to the documented NIH ChestX-ray14 sample dataset used by the project.
- Sufficient storage for the downloaded and extracted image files.
- Internet access when downloading the dataset, dependencies or pretrained model weights.
- Any necessary dataset account, licence acceptance or access permissions.

The dataset and pretrained weights may need to be obtained separately. Their availability in this repository should not be assumed.

## 20. Reproducibility and Responsible Use

- Preserve the patient-level train, validation and test separation used in the experiments.
- Keep validation-based threshold selection separate from final test evaluation.
- Record the model configuration, software versions, thresholds and evaluation procedure.
- Report validation and test metrics separately.
- Interpret per-disease results in light of class imbalance and the number of positive test examples.
- Treat Grad-CAM visualisations as exploratory explanations rather than proof of clinical reasoning.
- Do not use the model as a substitute for a qualified healthcare professional.

This project is intended for academic and educational research, not clinical diagnosis or treatment decisions.

