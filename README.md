# 🎓 Student Performance Prediction — Machine Learning Project

This repository contains a complete machine-learning workflow for predicting **student exam performance** using socio-economic, academic, and personal factors.  
The project covers **EDA**, **data preprocessing**, **encoding**, **model training**, and **evaluation**.

---

## 📁 Project Structure
├── Untitled-1.ipynb # Main project notebook
├── StudentPerformanceFactors.csv # Dataset (must be added manually)
└── README.md # Documentation


---

## 🎯 Objective

Build a regression model to predict **Exam_Score** using:

- Academic factors  
- Family & socio-economic attributes  
- School environment  
- Personal motivation metrics  

The notebook performs:

- Exploratory Data Analysis  
- Missing-value imputation  
- Ordinal & One-Hot Encoding  
- Numeric preprocessing  
- Linear Regression training  
- Model evaluation (RMSE, R²)  
- 5-Fold Cross-validation  

---

## 🧰 Technologies Used

- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn  
  - SimpleImputer  
  - OrdinalEncoder  
  - OneHotEncoder  
  - ColumnTransformer  
  - LinearRegression  
  - train_test_split  
  - cross_val_score  

---

## 📊 Dataset Overview

The dataset includes:

- **Ordinal features:**  
  Teacher_Quality, Parental_Education_Level, Distance_from_Home, Motivation_Level, Parental_Involvement, Family_Income, Peer_Influence  

- **Nominal features:**  
  Gender, Internet_Access, School_Type, Access_to_Resources, Extracurricular_Activities, Learning_Disabilities  

- **Numeric features:**  
  (various numerical indicators based on dataset)  

- **Target:**  
  Exam_Score  

---

## 🔧 Data Preprocessing

### 1. Missing Value Handling
```python
SimpleImputer(strategy="constant", fill_value="Missing")
2. Ordinal Encoding
ordinal_cols = [
    'Teacher_Quality',
    'Parental_Education_Level',
    'Distance_from_Home',
    'Motivation_Level',
    'Parental_Involvement',
    'Family_Income',
    'Peer_Influence'
]

ordinal_categories = [
    ['Low', 'Medium', 'High', 'Missing'],
    ['High School', 'College', 'Postgraduate', 'Missing'],
    ['Near', 'Moderate', 'Far', 'Missing'],
    ['Low', 'Medium', 'High'],
    ['Low', 'Medium', 'High'],
    ['Low', 'Medium', 'High'],
    ['Negative', 'Neutral', 'Positive']
]
3. One-Hot Encoding
onehot_cols = [
    'Access_to_Resources',
    'Extracurricular_Activities',
    'Gender',
    'Internet_Access',
    'Learning_Disabilities',
    'School_Type'
]
4. Numeric Features

numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns
5. Combined Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_transformer, ordinal_cols),
        ('onehot', onehot_transformer, onehot_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)
🤖 Model Training
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_df_final, y_train)
y_pred = model.predict(X_test_df_final)

📈 Model Evaluation
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

Cross-Validation
cv_scores = cross_val_score(model, X_train_df_final, y_train, cv=5, scoring='r2')

Outputs include:

RMSE

R² Score

Each fold's CV R²

Mean & standard deviation

📘 Prediction Comparison

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

🚀 How to Run the Project
1. Install Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
2. Add the Dataset

Place StudentPerformanceFactors.csv in the project folder.

3. Run Notebook

