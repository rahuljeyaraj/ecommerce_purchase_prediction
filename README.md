# 🛒 E-commerce Purchase Prediction System

## 📌 Overview

This project builds an end-to-end machine learning system to **predict whether a user will make a purchase** based on their behavior during an e-commerce session.

The system processes raw event-level data, converts it into meaningful session-level features, trains a machine learning model, and exposes predictions through a REST API.

---

## 🎯 Problem Statement

Given user interaction data (views, cart actions, browsing behavior), predict:

> **Will the user complete a purchase in this session?**

This is a **binary classification problem**:
- `1` → Purchase
- `0` → No purchase

---

## 📊 Dataset

- Source: Kaggle (E-commerce behavior dataset)
- Data type: Event-level logs
- Each row represents an action:
  - `view`
  - `cart`
  - `purchase`

---

## 🧠 Approach

### Step 1: Data Transformation

Raw data is event-level (multiple rows per session).

We convert it into **session-level data**:
- One row = one user session

---

### Step 2: Feature Engineering

Created the following features:

| Feature | Description |
|--------|-------------|
| `view` | Number of product views |
| `cart` | Number of cart additions |
| `session_duration_sec` | Time spent in session |
| `unique_products` | Number of unique products viewed |
| `avg_price` | Average price of viewed products |

---

### Step 3: Target Definition

```text
target = 1 → session contains purchase
target = 0 → no purchase