import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Path to the Telco Customer Churn dataset
# Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
DATASET_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Features chosen considering domain knowledge and importance
SELECTED_FEATURES = [
    "tenure",           # time customer has been with the business in months
    "MonthlyCharges",   # Amount billed on a monthly basis
    "TotalCharges",     # lifetime charges total
    "Contract",         # type of contract: month-to-month, one-year, two-year
    "InternetService",  # type of internet access: DSL, fiber optic, none
    "PaymentMethod",    # bill payment method used by the customer
    "SeniorCitizen",    # 1 if the age of the customer >= 65
    "PaperlessBilling", # electronic billing: yes or no
    "TechSupport",      # tech support services purchased by the customer
    "OnlineSecurity",   # online security services purchased by the customer
    "Churn"             # target variable: 1 if the customer churned, 0 otherwise
]

# Columns requiring label encoding prior to training
CATEGORICAL_COLS = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport",
    "OnlineSecurity"
]


def load_and_prepare_data():
    """
    Loads the Telco Churn dataset, does preprocessing, 
    and returns train-test splits that can be used to train models.

    Steps:
        1. Loads raw data in CSV format and selects features
        2. Corrects TotalCharges data type problem and imputes missing data
        3. Encodes target column with binary labels Yes/No as 1/0
        4. Labels categorical columns
        5. Stratifies train-test split

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """

    print("-" * 50)
    print("Phase 1: Data Preparation")
    print("-" * 50)

    # Raise an explicit exception if the dataset file does not exist
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'\n"
            "Please download from:\n"
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "and place the CSV file in the data/ folder."
        )

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Retain only the necessary columns in the dataframe
    df = df[SELECTED_FEATURES].copy()

    # The TotalCharges is a string in the raw dataframe
    # Zero tenure entries have whitespace; use coerce to convert such entries to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Binary conversion of churn labels to integer is essential for future processing
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Missing entries in the TotalCharges field must be replaced by the median value
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        print(f"Imputed {missing_count} missing values in TotalCharges with median")

    print(f"Churn rate: {df['Churn'].mean() * 100:.1f}%")

    # Use label encoding for categorical fields to allow numerical processing
    encoder = LabelEncoder()
    for col in CATEGORICAL_COLS:
        df[col] = encoder.fit_transform(df[col])

    # Separation of input features from the output label
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_names = X.columns.tolist()

    # Stratified split ensures that churn ratios are retained in both datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training set : {X_train.shape[0]} samples")
    print(f"Test set     : {X_test.shape[0]} samples")
    print(f"Features     : {feature_names}\n")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    load_and_prepare_data()
