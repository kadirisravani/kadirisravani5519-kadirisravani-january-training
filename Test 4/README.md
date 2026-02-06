Heart Disease Prediction Using Machine Learning

 Project Overview

This project focuses on predicting the presence of heart disease using multiple machine learning algorithms. The dataset is preprocessed, cleaned, and evaluated using different classification models to identify the best-performing model based on standard evaluation metrics.

The goal is to compare models such as Logistic Regression, Decision Tree, Random Forest, KNN, and SVM, and select the most suitable one for heart disease prediction.

 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve treatment outcomes. Traditional diagnosis can be time-consuming and dependent on expert availability. This project aims to automate heart disease prediction using machine learning techniques for faster and more accurate decision-making.

---

## 🗂 Dataset Description

* **Dataset Name:** `heart_disease_cleaned.csv`
* **Target Column:** `num`

  * `0` → No heart disease
  * `1–4` → Presence of heart disease (converted to binary `1`)
* **ID Column:** `id` (removed during training)

Technologies Used

* **Programming Language:** Python
* **Libraries:**

  * pandas, numpy
  * scikit-learn

---

## 🧹 Data Preprocessing Steps

1. Converted target variable into binary classification
2. Handled missing values using mean imputation
3. Removed duplicate records
4. Outlier detection and removal using IQR method
5. Encoded categorical variables using Label Encoding
6. Feature scaling using StandardScaler
7. Train-test split (80% training, 20% testing)

 Machine Learning Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

---

 Model Evaluation Metrics

Each model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

---

 Results Summary

Based on the evaluation results, the **Random Forest Classifier** performed best among all models.

**Best Model Performance:**

* Accuracy: ~82%
* Precision: ~77%
* Recall: ~81%
* F1 Score: ~79%

---

## 🏆 Conclusion

The Random Forest model achieved the highest accuracy and balanced performance across all metrics, making it the most suitable model for this heart disease prediction task. Ensemble methods performed better than individual models due to their ability to reduce overfitting and capture complex patterns.

---

## 🚀 How to Run the Project

1. Clone the repository
2. Install required libraries:

   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Place the dataset inside the `data/` folder
4. Run the Python script:

   ```bash
   python ml_models.py
   ```

---

## 📌 Future Enhancements

* Hyperparameter tuning
* Cross-validation
* Feature selection
* Deployment using Flask or Streamlit
