# E-commerce Purchase Prediction System

## Overview

This project builds an end-to-end machine learning system to **predict whether a user will make a purchase** based on their behavior during an e-commerce session.

The system processes raw event-level data, converts it into meaningful session-level features, trains a machine learning model, and exposes predictions through both:
- a REST API using FastAPI
- a simple interactive UI using Streamlit

---

## Problem Statement

Given user interaction data such as views, cart actions, browsing duration, and product exploration, predict:

> **Will the user complete a purchase in this session?**

This is a **binary classification problem**:
- `1` → Purchase
- `0` → No purchase

---

## Dataset

- Source: Kaggle eCommerce behavior data from a multi-category store
- Data type: Event-level logs
- Each row represents an action such as:
  - `view`
  - `cart`
  - `purchase`

Because the raw dataset is event-level, it cannot be used directly for modeling. It first needs to be transformed into session-level features.

---

## Approach

### Step 1: Data Transformation

Raw data contains multiple rows per user session.

This project converts raw logs into **session-level data**:
- One row = one user session

This makes the dataset suitable for machine learning.

---

### Step 2: Feature Engineering

Created the following features:

| Feature | Description |
|--------|-------------|
| `view` | Number of product views in a session |
| `cart` | Number of cart additions in a session |
| `session_duration_sec` | Time spent in the session |
| `unique_products` | Number of unique products interacted with |
| `avg_price` | Average product price in the session |

---

### Step 3: Target Definition

```text
target = 1 → session contains purchase
target = 0 → session does not contain purchase
```

---

## Problems Solved During the Project

This project was not just about training a model. It also involved solving important real-world ML problems.

### 1. Data Leakage

An early version of the model used the `purchase` feature directly.

That caused the model to achieve nearly perfect performance, because:
- the feature was directly revealing the answer
- the model was effectively "cheating"

This was fixed by removing leaky features during model training.

Key learning:
- **high accuracy is not always good**
- the feature set must only contain information that would be available at prediction time

---

### 2. Class Imbalance

The dataset was heavily imbalanced:
- most sessions ended with **no purchase**
- only a small fraction ended with **purchase**

Because of this, a model could get high accuracy by simply predicting `0` for almost every case.

This was addressed by using:

```python
class_weight="balanced"
```

This improved the model’s ability to detect purchasing sessions instead of ignoring the minority class.

Key learning:
- **accuracy alone can be misleading**
- recall and precision matter much more in imbalanced classification tasks

---

### 3. Deployment

Most ML projects stop at notebook experimentation.

This project goes further by deploying the trained model in two ways:

#### FastAPI
Used to expose the model as a REST API for real-time predictions.

#### Streamlit
Used to build a lightweight user interface for testing predictions interactively.

Key learning:
- a useful ML project is not just a notebook
- deployment makes the model usable by people and other systems

---

## Model

- Algorithm: Logistic Regression
- Configuration: balanced class weights
- Objective: detect sessions likely to result in purchase

---

## Results

Final experiment using behavioral features including `cart` produced approximately:

| Metric | Value |
|------|------|
| Accuracy | 0.91 |
| Precision (purchase class) | 0.37 |
| Recall (purchase class) | 0.84 |
| F1-score (purchase class) | 0.52 |

### Confusion Matrix

```text
[[2382143  218424]
 [  25130  129513]]
```

### Interpretation

- The model captures a large portion of real buyers
- It favors higher recall, which is useful when missing buyers is costly
- Some false positives remain, but the model is useful for high-intent session detection

---

## Project Structure

```text
ecommerce_purchase_prediction/
│
├── api/
│   └── app.py
├── data/
│   └── model_ready_day1.csv
├── models/
│   └── model.pkl
├── notebooks/
├── src/
│   ├── data_pipeline.py
│   └── train_model.py
├── ui/
│   └── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Create and activate virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run data pipeline

```bash
python src/data_pipeline.py
```

### 4. Train model

```bash
python src/train_model.py
```

### 5. Run FastAPI

```bash
uvicorn api.app:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```
<img width="1213" height="797" alt="image" src="https://github.com/user-attachments/assets/2925ac7e-dd40-4787-bd6d-8eb29cdcdb8a" />

### 6. Run Streamlit UI

```bash
streamlit run ui/app.py
```
<img width="997" height="1048" alt="image" src="https://github.com/user-attachments/assets/81ae4cf1-d511-45e6-941f-2c9329de4198" />

---

## API Usage

### Endpoint

```text
POST /predict
```

### Request Body

```json
{
  "cart": 1,
  "view": 5,
  "session_duration_sec": 120,
  "unique_products": 3,
  "avg_price": 500
}
```

### Response

```json
{
  "purchase_prediction": 1,
  "purchase_probability": 0.82
}
```

---

## What I Learned

Through this project, I learned how to:

- transform raw event logs into ML-ready session features
- identify and fix data leakage
- handle imbalanced datasets
- evaluate models beyond simple accuracy
- package training code into reusable Python scripts
- deploy a trained ML model through API and UI layers

---

## Future Improvements

Potential next steps for improving this system:

- use time-aware features based only on pre-purchase behavior
- try tree-based boosting models such as XGBoost
- add model threshold tuning for business-specific trade-offs
- deploy the app to the cloud
- connect Streamlit UI to FastAPI instead of loading the model directly

---

## Conclusion

This project demonstrates a complete ML workflow:

```text
Raw Data → Feature Engineering → Model Training → Evaluation → API/UI Deployment
```

It is not just a notebook experiment. It is a deployable end-to-end machine learning project built around a practical e-commerce prediction problem.
