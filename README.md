# Customer Churn Prediction — MLOps with MLflow

**AIN-3009 | Bahçeşehir University | Term Project**
**Student:** Abdin Abuhmaid — 2267570

---

## Overview

In this project, a full-stack pipeline for predicting customer churns in telecoms will be developed. The entire process of creating a pipeline from obtaining raw data through deploying and monitoring deployed models will take place on MLflow. For this purpose, IBM Telco Customer Churn Dataset will be used. The dataset contains 7,043 samples.

---

## Pipeline Stages

| Stage | Description                                                       |
|---|-------------------------------------------------------------------|
| Data Preparation | Loading, cleaning, encoding, and also splitting the dataset       |
| Experiment Tracking | Training the three classifiers with full parameter and metric logging |
| Hyperparameter Tuning | 50-trial Bayesian search using Hyperopt                           |
| Model Registry | Versioned model promotion from Staging to Production              |
| Deployment | Real-time and batch inference using registered model              |
| Performance Monitoring | Six-month drift simulation with automated alerting                |

---

## Project Structure

```
churn-mlops/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── data_prep.py      — data loading and preprocessing
│   ├── train.py          — model training and experiment logging
│   ├── tune.py           — hyperparameter optimization
│   ├── register.py       — model registry and stage transitions
│   ├── serve.py          — deployment and inference
│   └── monitor.py        — post-deployment monitoring
├── main.py               — runs full pipeline
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/abdinabuhmaid/churn-mlops.git
cd churn-mlops

# Install the dependencies
pip3 install -r requirements.txt

# Download the following dataset from Kaggle and place it in data/
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Run pipeline
python3 main.py

# Launch MLflow dashboard
python3 -m mlflow ui
# Open: http://127.0.0.1:5000
```

---

## Running of the Project

```bash
python3 main.py
```

The pipeline executes six phases in sequence:

1. Data preparation: encodes and cleans raw dataset
2. Model training: builds Logistic Regression, Random Forest, and Gradient Boosting classifiers
3. Hyperparameter tuning: runs 50 Hyperopt iterations for the best classifier
4. Model registry: moves the fine-tuned classifier to the Production stage
5. Deployment: scores samples and batches of customers
6. Monitoring: conducts six months of performance monitoring

Total execution time is about 60-70 seconds.

---

## MLflow Dashboard

Once you have completed the pipeline, run the UI to analyze your logged experiments:

```bash
python3 -m mlflow ui
```

- **Churn_Prediction**: has all the training and tuning iterations
- **Churn_Model_Monitoring**: has all the monthly performance and drift charts
- **Model registry**: shows the complete version history of the ChurnPredictionModel

---

## Models

| Model | Parameters | Role |
|---|---|---|
| Logistic Regression | C=1.0, max_iter=1000 | Baseline |
| Random Forest | n_estimators=100, max_depth=10 | Ensemble comparison |
| Gradient Boosting | n_estimators=250, lr=0.03 | Best performer |

---

## Results

- Best model: Gradient Boosting
- Evaluation metric: AUC-ROC (chosen for class imbalance)
- Post-tuning AUC: ~0.848
- Production version: ChurnPredictionModel v6
- Drift detected from month 4 onward — retraining recommended

---

## Technologies

| Tool | Purpose |
|---|---|
| MLflow | Experiment tracking, registry, and serving |
| Scikit-learn | Model training and evaluation |
| Hyperopt | Bayesian hyperparameter optimization |
| Pandas / NumPy | Data processing |
| Matplotlib | Visualization and monitoring charts |

---

## Submission Format

```
PRJ-AbdinAbuhmaid-2267570.zip
├── churn-mlops/
└── report.pdf
```
