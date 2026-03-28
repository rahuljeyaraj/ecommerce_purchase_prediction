import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


FILE_PATH = "../data/model_ready.csv"
MODEL_OUTPUT_PATH = "../models/model.pkl"


def load_data(file_path: str) -> pd.DataFrame:
    print("Loading model-ready data...")
    df = pd.read_csv(file_path)
    print("Data shape:", df.shape)
    return df


def prepare_features(df: pd.DataFrame):
    print("Preparing features and target...")

    X = df.drop(columns=["target", "purchase", "cart"], errors="ignore")
    y = df["target"]

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
    print("Feature columns:", X.columns.tolist())

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    print("Splitting train and test data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    print("Training model...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    print("Training complete.")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    print("Evaluating model...")

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_model(model, output_path: str) -> None:
    print(f"Saving model to {output_path} ...")
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print("Model saved.")


def main():
    df = load_data(FILE_PATH)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()