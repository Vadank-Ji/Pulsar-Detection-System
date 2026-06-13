# Pulsar Detection System

## Overview

A machine learning classification system to detect pulsars from the Fermi 4FGL gamma-ray catalog. This project leverages data from NASA's Fermi Gamma-ray Space Telescope to identify pulsar sources among celestial objects using advanced supervised learning techniques.

**Objective**: Build robust classifiers to distinguish pulsars (PSR, MSP) from other gamma-ray sources with optimized F1 scores for imbalanced class detection.

---

## Dataset

### Source
- **Fermi 4FGL Catalog** (Version 3.5): Complete catalog of gamma-ray sources detected by the Fermi Large Area Telescope
- **Total Records**: 7,195 celestial sources
- **Class Distribution**: 
  - Non-Pulsar: 6,875 (95.6%)
  - Pulsar: 320 (4.4%) — *Imbalanced dataset*

### Key Features

The dataset includes ~80 features across several categories:

#### Position & Localization
- `GLON`, `GLAT`: Galactic longitude/latitude
- `RAJ2000`, `DEJ2000`: Right ascension/declination (J2000)
- `Conf_68_SemiMajor`, `Conf_68_SemiMinor`: 68% confidence position uncertainty

#### Spectral Properties
- `SpectrumType`: Spectral model (PowerLaw, LogParabola, PLEC)
- `PL_Index`: Power-law spectral index
- `LP_Index`, `LP_beta`: Log-parabola spectral parameters
- `PLEC_IndexS`, `PLEC_ExpfactorS`: Power-law exponential cutoff parameters

#### Flux & Energy
- `Flux1000`: Photon flux above 1 GeV
- `Energy_Flux100`: Energy flux above 100 MeV
- `Pivot_Energy`: Reference energy for spectrum
- `Flux_Band`: Multi-band flux in 8 energy ranges (7195, 8)

#### Variability & Detection
- `Variability_Index`: **Strong pulsar discriminator** — pulsars show distinct variability patterns
- `Frac_Variability`: Fractional variability
- `Signif_Avg`: Average detection significance
- `Signif_Peak`: Peak detection significance
- `Npred`: Predicted photon count

#### Time-Series History
- `Flux_History`: 14-month brightness history (7195, 14)
- `Sqrt_TS_History`: Time-series detection significance

### Feature Recommendations

**Top Features (ML-Relevance)**:
- `LP_SigCurv`, `PLEC_SigCurv`: Curvature significance
- `Variability_Index`, `Frac_Variability`: Temporal behavior
- `LP_beta`, `PLEC_ExpfactorS`: Spectral characteristics
- `Signif_Avg`, `Flux1000`: Detection quality and brightness

**Features Dropped**:
- Leakage columns: `ASSOC_*`, `CLASS2`, counterpart associations
- Categorical metadata: `Source_Name`, `DataRelease`, `ROI_num`
- Position angles: `Conf_*_PosAng`
- Time-peak features: `Peak_Interval`, `Time_Peak` (sparse/weak signal)

---

## Project Structure

```
PulsarDetectionSystem/
├── proto.ipynb                      # Main notebook with EDA and models
├── gll_psc_v35.fit                 # Fermi 4FGL catalog (FITS format)
└── README.md                       # This file
```

---

## Installation

### Requirements
- Python 3.9+
- Jupyter Notebook

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn astropy torch tensorflow
```

**Key Libraries**:
- `scikit-learn`: ML models, pipelines, hyperparameter tuning
- `imbalanced-learn`: SMOTE for handling class imbalance
- `astropy`: FITS file handling (Fermi catalog)
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization

---

## Methodology

### 1. Data Preprocessing

**Steps**:
1. **Load FITS file**: Extract single-dimensional columns from Fermi catalog
2. **Create binary target**: `Is_Pulsar_Class = 1` if CLASS1 ∈ {PSR, MSP, psr, msp}, else 0
3. **Drop leakage columns**: Remove association and counterpart data
4. **Handle missing values**: Fill with median (numeric features)
5. **Remove infinities**: Replace `±inf` with NaN, then remove rows
6. **One-hot encode**: Categorical `SpectrumType` → binary features
7. **Train-test split**: 80/20 stratified split (preserve class ratio)

### 2. Class Imbalance Handling

**SMOTE (Synthetic Minority Over-Sampling)**:
- Applied inside pipelines to avoid data leakage
- Oversamples minority class during training only
- Parameters: `random_state=42` for reproducibility

**Class Weighting**:
- `balanced_subsample`: Automatic adjustment for tree-based models

### 3. Models Implemented

#### Random Forest Classifier (SMOTE + Hyperparameter Search)
**Pipeline**: SMOTE → RandomForestClassifier

**Hyperparameter Search** (RandomizedSearchCV):
- `n_estimators`: [100, 200, 500]
- `max_depth`: [None, 5, 10, 20]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `class_weight`: [None, 'balanced', 'balanced_subsample']

**CV Strategy**: StratifiedKFold (5 splits)  
**Optimization Metric**: F1-score (favors minority class recall)

**Baseline Performance**:
- Class 0 (Non-Pulsar): Precision=0.98, Recall=0.99, F1=0.99
- Class 1 (Pulsar): Precision=0.80, Recall=0.64, **F1=0.71**
- Confusion Matrix: [[1375, 0], [8, 56]]

#### Logistic Regression (Baseline)
**Pipeline**: StandardScaler → LogisticRegression

**Hyperparameter Search** (GridSearchCV):
- `C`: [0.01, 0.1, 1, 10]
- `solver`: liblinear
- `class_weight`: balanced

**CV Strategy**: StratifiedKFold (5 splits)  
**Optimization Metric**: Balanced Accuracy

### 4. Evaluation Metrics

- **Balanced Accuracy**: Average recall across classes
- **Precision/Recall/F1**: Per-class performance
- **Confusion Matrix**: True/false positives and negatives
- **ROC-AUC**: Discrimination ability across thresholds
- **Precision-Recall Curve**: F1-score threshold optimization
- **Cross-Validation**: 5-fold stratified CV with F1 scoring
- **Feature Importances**: Top 20 discriminative features

### 5. Threshold Tuning

For imbalanced datasets, the default 0.5 classification threshold may be suboptimal:

**Approach**:
1. Compute precision-recall curve on test set
2. Calculate F1 for all thresholds: `F1 = 2 * (Precision × Recall) / (Precision + Recall)`
3. Select threshold maximizing F1
4. Report classification metrics at tuned threshold

---

## Results Summary

### Random Forest (Best Performer)

| Metric | Class 0 | Class 1 | Overall |
|--------|---------|---------|---------|
| Precision | 0.98 | 0.80 | - |
| Recall | 0.99 | 0.64 | - |
| **F1-Score** | **0.99** | **0.71** | - |
| Support | 1375 | 64 | 1439 |

**Top Features** (by importance):
1. LP_SigCurv (0.089)
2. PLEC_SigCurv (0.082)
3. LP_beta (0.077)
4. Unc_PLEC_EPeak (0.058)
5. PLEC_ExpfactorS (0.054)

### Logistic Regression (Baseline)

| Metric | Class 0 | Class 1 | Overall |
|--------|---------|---------|---------|
| Precision | 1.00 | 0.23 | - |
| Recall | 0.85 | 0.95 | - |
| F1-Score | 0.92 | 0.36 | - |

**Observation**: LR achieves high pulsar recall (0.95) but low precision (0.23) → many false positives.

---

## Usage

### Running the Notebook

```bash
jupyter notebook proto.ipynb
```

**Cell Execution Order**:
1. Import libraries
2. Load FITS catalog
3. Extract & explore data
4. Data preprocessing (drop columns, handle NaN, one-hot encode)
5. Feature engineering (remove infinities, final feature set)
6. Train-test split
7. Train RandomForest with SMOTE + hyperparameter search
8. Train Logistic Regression
9. Evaluate both models (metrics, confusion matrices, ROC curves)
10. Threshold optimization & cross-validation

### Key Variables
- `X`, `y`: Full dataset features and labels (7194 samples)
- `X_train`, `y_train`: Training set (80%)
- `X_test`, `y_test`: Test set (20%)
- `model`: Fitted RandomForest pipeline
- `pred`, `probs`: Test predictions and probabilities

---

## Future Improvements

### 1. Advanced Resampling
- **SMOTEENN**: Combines SMOTE with edited nearest neighbors
- **ADASYN**: Adaptive synthetic sampling
- **Stratified subsampling**: Balanced undersampling

### 2. Ensemble Methods
- **XGBoost / LightGBM**: Gradient boosting with class weight tuning
- **Stacking**: RF + XGBoost + LR meta-learner optimizing F1
- **Voting Classifier**: Hard/soft voting combination

### 3. Feature Engineering
- **Multi-band features**: Expand `Flux_Band` (8 energy ranges)
- **Time-series statistics**: Mean/std/min/max of `Flux_History`
- **Polynomial interactions**: Combine spectral and variability features
- **PCA**: Dimensionality reduction for high-correlation features

### 4. Hyperparameter Optimization
- **Bayesian optimization**: More efficient than random search
- **Neural networks**: Deep learning with imbalanced loss functions

### 5. Interpretability
- **SHAP values**: Feature contribution analysis
- **LIME**: Local model-agnostic explanations
- **Permutation importance**: Alternative to tree importances

### 6. Deployment
- **Model serialization**: Save trained models (pickle/joblib)
- **REST API**: FastAPI server for real-time predictions
- **Web dashboard**: Streamlit/Dash for interactive exploration

---

## References

- **Fermi LAT Collaboration**: [4FGL Catalog Documentation](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4FGL/)
- **Pulsars in Fermi Dataset**: Pulsar identification via gamma-ray variability
- **Imbalanced Learning**: Chawla et al., SMOTE (2002)
- **scikit-learn Documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)

---

## Contact & Support

For questions or issues, refer to:
- Comments and markdown documentation in `proto.ipynb`
- Feature documentation table in notebook (see Dataset section above)
- Cross-validation and baseline model definitions in code cells

---

**Last Updated**: 2026  
**Status**: Active Development  
**License**: [MIT]
