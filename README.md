# Real vs AI-Generated Image Detection

A machine learning pipeline for classifying images as real (human-captured) or AI-generated. Built as a capstone project for ADS 504 (Machine Learning) at the University of San Diego.

## Repository Layout

```
ADS504-Final/
├── data/
│   ├── images/          # Raw images (train / test / validation splits)
│   ├── arrays/          # Preprocessed 224x224x3 numpy arrays
│   └── metadata/        # Parquet files with labels and engineered features
├── features/            # Extracted feature sets (color histogram, GLCM, HOG, EfficientNet)
├── models/              # Trained model artifacts
├── experiments/         # Keras Tuner hyperparameter search results
├── config/              # Importable path configuration (paths.py)
├── utils/               # Helper modules for image reading and feature extraction
├── resources/           # Reference papers
├── MSADS_504_Final_Project.ipynb   # Main project notebook
├── Paola_s_Final_Notebook.ipynb    # EDA, classical ML, and CNN evaluation
├── Paola_s_Notebook-2.ipynb        # Additional exploration and modeling
└── Taylors_notebook.ipynb          # Feature engineering and model tuning
```

## Summary of Findings

We compared classical machine learning models (Logistic Regression, SVM, Random Forest) against a fine-tuned EfficientNetB0 CNN on a multi-source dataset of ~15,000 images compiled from LAION, Open Images, and the AI vs Human study.

Key results:

- **Random Forest on engineered metadata features** achieved the best performance at **92% accuracy** (AUC 0.98), with balanced precision and recall across both classes.
- Logistic Regression on metadata reached 81% accuracy (AUC 0.87).
- The fine-tuned EfficientNetB0 CNN reached 77-80% accuracy, underperforming the classical models trained on hand-crafted features.
- Engineered image properties (color statistics, edge density, texture descriptors, brightness, contrast) proved more discriminative than raw CNN embeddings for this task.

## Contact

For questions or collaboration, reach out to any of the authors:

| Name | Email |
|------|-------|
| Taylor Kirk | tkirk@sandiego.edu |
| Tommy Baron | tbarron@sandiego.edu |
| Paola Rodriguez | prodriguez2@sandiego.edu |

## References

- K. Balakrishna Maruthiram, G. Venkataramireddy, M. Klick. "Real VS AI Generated Image Detection and Classification." *IJIRT*, Volume 11, Issue 2, July 2024. IJIRT 166462.
- LAION (Large-scale Artificial Intelligence Open Network) dataset.
- Open Images dataset.
- AI vs Human Images study dataset.
