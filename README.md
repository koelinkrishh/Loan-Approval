# Loan Approval Prediction:

An End-to-End Machine learning project which predicts whether an individual is eligible to be approved for his loan based on various personal and financial factors like his marital status,profession, etc.

-> Complete Classification ML model using various algorithms like logistic regression, xgboost, K-nearest neighbour and Artificial Neural Network(ANN), etc.

### 🚀 Features

- Data Cleaning & Preprocessing: Handling missing values, encoding categorical variables, and feature scaling.
- Exploratory Data Analysis (EDA): Visualizing key insights about loan applicants.
- Feature Engineering: Transforming raw data into meaningful features.
- Model Selection & Training: Comparing multiple classification algorithms.
- Hyperparameter Tuning: Optimizing model performance.
- Model Evaluation: Assessing accuracy, precision, recall, and F1-score.
- Deployment: Deploying the model using Flask/Streamlit.

### 📂 Dataset

- Features: Applicant's Income, Age, Loan Amount, Credit Score, Employment History, Number of credit lines, Interest rate on Loan, Education, Loan Terms, Debt to income ratio, Employment type, Marital Status, have mortgage,dependent,Co-signer and purpose for Loan.

- Target Variable: Loan Status (Approved/Not Approved)
- Source: Kaggle Loan Prediction Dataset.

### 📌 How to Run Project:

1. Clone the repository:
```
git clone https://github.com/your-repo/loan-approval-prediction.git
```
2. Navigate to the project directory: 
```
cd loan-approval-prediction
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Run the project:
```
python app.py  # For Flask
streamlit run app.py  # For Streamlit
```

### 📈 Results

The final model achieves _% accuracy on the test dataset, with a ROC-AUC score of _%. 

Feature importance analysis reveals that Credit History and Applicant Income are the most influential factors.
