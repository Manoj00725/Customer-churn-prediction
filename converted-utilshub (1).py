# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")
print("Libraries imported.")

try:
    df = pd.read_csv('your_dataset.csv')
    print("Dataset loaded successfully!")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: Dataset file not found. Please ensure 'your_dataset.csv' is in the correct directory.")
    exit()

# Initial exploration
print("\nDataset information:")
df.info()
print("\nSummary statistics of numerical features:")
print(df.describe())
print("\nValue counts of the target variable 'Churn':")
print(df['Churn'].value_counts())
sns.countplot(x='Churn', data=df)
plt.title('Distribution of Churn')
plt.show()

# Identify missing values
print("\nMissing values:")
print(df.isnull().sum())
# Handle missing values (if any) - Example: Imputing or dropping
# For 'TotalCharges', if it has missing values and is numeric, we might impute with the median.
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print("\nMissing values after handling:")
print(df.isnull().sum())

# Identify categorical and numerical features
categorical_features = df.select_dtypes(include='object').columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Remove the customer ID as it's usually not informative for the model
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)
    if 'customerID' in categorical_features:
        categorical_features.remove('customerID')
    print("\n'customerID' column removed.")
else:
    print("\n'customerID' column not found.")
print("\nCategorical Features:", categorical_features)
print("Numerical Features:", numerical_features)

# Encode categorical features
label_encoder = LabelEncoder()
for col in ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'Churn', 'gender']: # Include 'gender' if present
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])
# One-hot encode remaining categorical features (if any after initial encoding)
df = pd.get_dummies(df, columns=[col for col in categorical_features if col not in ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'Churn', 'gender']], drop_first=True)
print("\nProcessed dataset information after encoding:")
df.info()

# Scale numerical features
numerical_features = [col for col in numerical_features if col != 'Churn']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print("\nProcessed dataset information after scaling:")
df.info()
print("\nFirst 5 rows of the scaled data:")
print(df.head())

# ## 4. Feature Engineering (Optional)
# Example 1: Tenure Grouping
bins = [0, 12, 24, 36, 48, 60, np.inf]
labels = ['0-12 Months', '13-24 Months', '25-36 Months', '37-48 Months', '49-60 Months', '60+ Months']
if 'tenure' in df.columns:
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
    df = pd.get_dummies(df, columns=['Tenure_Group'], drop_first=True)
    print("\n'Tenure_Group' feature created and one-hot encoded.")
else:
    print("\n'tenure' column not found, skipping 'Tenure_Group' feature engineering.")
print("\nDataset information after optional feature engineering:")
df.info()
print("\nFirst 5 rows after optional feature engineering:")
print(df.head())

# ## 5. Model Selection
# Split the data into training and testing sets
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("\nShape of training data:", X_train.shape)
print("Shape of testing data:", X_test.shape)
print("Shape of testing target:", y_test.shape)
print("Shape of training target:", y_train.shape)

# Initialize models
logistic_regression = LogisticRegression(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)
print("\nModels initialized: Logistic Regression, Random Forest, Gradient Boosting.")

# ## 6. Model Training
# Train Logistic Regression
logistic_regression.fit(X_train, y_train)
print("\nLogistic Regression trained.")

# Train Random Forest
random_forest.fit(X_train, y_train)
print("Random Forest trained.")

# Train Gradient Boosting
gradient_boosting.fit(X_train, y_train)
print("Gradient Boosting trained.")

# ## 7. Model Evaluation
# Make predictions on the test set
y_pred_lr = logistic_regression.predict(X_test)
# Evaluate Logistic Regression
print("\n--- Logistic Regression Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluate Random Forest
y_pred_rf = random_forest.predict(X_test)
print("\n--- Random Forest Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluate Gradient Boosting
y_pred_gb = gradient_boosting.predict(X_test)
print("\n--- Gradient Boosting Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gb))
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(conf_matrix_gb, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix - Gradient Boosting')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance (for tree-based models like Random Forest)
if hasattr(random_forest, 'feature_importances_'):
    feature_importances_rf = pd.Series(random_forest.feature_importances_, index=X_train.columns)
    feature_importances_rf_sorted = feature_importances_rf.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances_rf_sorted.head(10), y=feature_importances_rf_sorted.head(10).index)
    plt.title('Top 10 Feature Importances - Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()
else:
    print("\nRandom Forest model does not have 'feature_importances_' attribute.")

# Feature Importance (for tree-based models like Gradient Boosting)
if hasattr(gradient_boosting, 'feature_importances_'):
    feature_importances_gb = pd.Series(gradient_boosting.feature_importances_, index=X_train.columns)
    feature_importances_gb_sorted = feature_importances_gb.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances_gb_sorted.head(10), y=feature_importances_gb_sorted.head(10).index)
    plt.title('Top 10 Feature Importances - Gradient Boosting')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()
else:
    print("\nGradient Boosting model does not have 'feature_importances_' attribute.")

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()