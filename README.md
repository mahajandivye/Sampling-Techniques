# Sampling Techniques on Imbalanced Credit Card Dataset

## Objective
The objective of this assignment is to understand the importance of sampling techniques in handling highly imbalanced datasets and to analyze how different sampling strategies affect the performance of various machine learning models.

Imbalanced datasets are common in real-world problems such as credit card fraud detection, where incorrect predictions can lead to serious consequences. This project evaluates multiple sampling techniques and machine learning models to identify optimal combinations.

---

## Dataset
- Dataset: Credit Card Fraud Dataset
- Source: https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv
- Target Column: Class  
  - 0 → Non-Fraud  
  - 1 → Fraud  

The dataset is highly imbalanced and requires balancing before model training.

---

## Dataset Balancing
The dataset was converted into a balanced dataset using SMOTE (Synthetic Minority Oversampling Technique). This ensures equal representation of both classes during training and improves model reliability.

---

## Sampling Techniques Used

| Sampling ID | Technique |
|------------|----------|
| Sampling1 | Random Under Sampling |
| Sampling2 | Random Over Sampling |
| Sampling3 | SMOTE |
| Sampling4 | NearMiss |
| Sampling5 | SMOTEENN |

---

## Machine Learning Models Used

| Model ID | Algorithm |
|--------|-----------|
| M1 | Logistic Regression |
| M2 | Decision Tree |
| M3 | Random Forest |
| M4 | K-Nearest Neighbors (KNN) |
| M5 | Support Vector Machine (SVM) |

---

## Accuracy Results

| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|------|-----------|-----------|-----------|-----------|-----------|
| M1 | 50.10 | 52.24 | 63.18 | 69.23 | 70.12 |
| M2 | 59.25 | 65.27 | 68.72 | 28.36 | 30.25 |
| M3 | 90.45 | 72.41 | 32.17 | 42.58 | 41.85 |
| M4 | 78.25 | 56.24 | 47.23 | 33.44 | 40.12 |
| M5 | 81.25 | 12.85 | 57.36 | 32.25 | 52.74 |

---

## Best Sampling Technique per Model

| Model | Best Sampling Technique | Accuracy |
|------|------------------------|----------|
| M1 | Sampling5 (SMOTEENN) | 70.12 |
| M2 | Sampling3 (SMOTE) | 68.72 |
| M3 | Sampling1 (Under Sampling) | 90.45 |
| M4 | Sampling1 (Under Sampling) | 78.25 |
| M5 | Sampling1 (Under Sampling) | 81.25 |

---

## Discussion
The results clearly demonstrate that sampling techniques significantly influence model performance on imbalanced datasets. Tree-based models such as Random Forest perform better with under-sampling techniques, while linear models benefit from hybrid sampling methods. There is no single best sampling method; the optimal approach depends on the machine learning model used.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

---

## Repository Structure

Sampling_Assignment/
- Creditcard_data.csv
- sampling_assignment.ipynb
- README.md

---

## Conclusion
This assignment highlights the importance of selecting appropriate sampling techniques when working with imbalanced datasets. Proper sampling improves model accuracy and ensures fair learning, making machine learning systems more robust and reliable in real-world applications.
