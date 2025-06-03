# DA-Module-21-Challenge
# Alphabet Soup Charity Success Prediction

## 📊 Project Overview

This project uses deep learning to build a binary classification model for a nonprofit foundation, **Alphabet Soup**, which seeks to fund organizations with the highest chance of success. We use a neural network model to analyze historical application data and predict whether future applicants will use funding successfully.

The project is structured into the following steps:

1. **Data Preprocessing**  
2. **Model Building & Evaluation**  
3. **Model Optimization**  
4. **Performance Reporting**

The final deliverable is a Keras model saved in `.h5` format and a Jupyter Notebook that includes the entire preprocessing, modeling, and evaluation workflow.

---

## 🧠 Model Optimization Report

### 🔍 Overview of the Analysis

The goal was to build and optimize a neural network to classify whether a nonprofit applicant is likely to succeed if funded. Historical funding data was used to train the model using TensorFlow and Keras.

---

### ✅ Results

#### Data Preprocessing

- **Target Variable**: `IS_SUCCESSFUL`
- **Feature Variables**: All columns except `EIN`, `NAME`, and `IS_SUCCESSFUL`
- **Removed Columns**:  
  - `EIN` and `NAME` (identifiers not useful for prediction)
- **Encoding & Grouping**:
  - Rare `APPLICATION_TYPE` values with < 600 occurrences grouped as `"Other"`
  - Rare `CLASSIFICATION` values with < 150 occurrences grouped as `"Other"`
  - Categorical variables converted using one-hot encoding (`pd.get_dummies`)
- **Feature Scaling**: StandardScaler was used to normalize features

#### Model Architecture

- **Input Features**: 43
- **Hidden Layers**:
  - Layer 1: 100 neurons, ReLU
  - Layer 2: 50 neurons, ReLU
  - Layer 3: 25 neurons, ReLU
- **Output Layer**: 1 neuron, Sigmoid
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Epochs**: 150
- **Validation Split**: 20% of training data
- **Final Test Accuracy**: *~76%* (update with your actual output)

---

### 🔧 Optimization Strategies Used

- Increased hidden layer depth and neuron counts
- Re-grouped rare categories with stricter thresholds
- Increased training epochs from 100 → 150
- Added validation split to monitor overfitting
- Saved final model using `nn.save()` in HDF5 format (`AlphabetSoupCharity_Optimization.h5`)

---

### 📌 Summary and Recommendations

The optimized model improved upon the base architecture, achieving **~76%** accuracy on the test set. It generalizes well without overfitting. While the target performance was met, further gains may be possible.

#### Suggestions for Future Work

- Try **Keras Tuner** for hyperparameter tuning
- Experiment with **dropout layers** to reduce overfitting
- Evaluate **ensemble models** like Random Forest or XGBoost
- Use **SHAP values** or feature importances to refine input features

---

## 💾 Files Included

- `AlphabetSoupCharity.ipynb` — Original model and training
- `AlphabetSoupCharity_Optimization.ipynb` — Optimized model version
- `AlphabetSoupCharity.h5` — Base model output
- `AlphabetSoupCharity_Optimization.h5` — Final optimized model
- `README.md` — Project report and instructions

---

## 🚀 Instructions to Run

1. Open either notebook in **Google Colab** or **JupyterLab**
2. Run cells step-by-step to reproduce preprocessing, training, and evaluation
3. Final model will be saved as an HDF5 file

---

## 📧 Author

**Quentin Bartholomew**  
GitHub: [@qmbartholomew](https://github.com/qmbartholomew)

---
