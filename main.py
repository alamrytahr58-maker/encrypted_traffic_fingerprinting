import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================
# 1. Configuration
# =========================

DATA_PATH = "data/traffic_features.csv"
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "random_forest_model.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# 2. Load Dataset
# =========================

def load_dataset(path):
    """
    Load the network traffic dataset from CSV file.
    """
    df = pd.read_csv(path)
    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    return df


# =========================
# 3. Preprocessing
# =========================

def preprocess_data(df):
    """
    Clean the dataset and prepare features and labels.
    This function assumes that the target column is named 'Label'.
    """

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Remove non-feature columns if they exist
    columns_to_drop = [
        "Flow ID",
        "Src IP",
        "Dst IP",
        "Source IP",
        "Destination IP",
        "Src Port",
        "Dst Port",
        "Source Port",
        "Destination Port",
        "Timestamp"
    ]

    for col in columns_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Make sure Label column exists
    if "Label" not in df.columns:
        raise ValueError("The dataset must contain a 'Label' column.")

    # Separate features and target
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # Convert any non-numeric feature columns into numeric values
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Remove columns that became completely empty
    X.dropna(axis=1, how="all", inplace=True)

    # Fill remaining missing values with median
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Preprocessing completed.")
    print("Number of features:", X.shape[1])
    print("Classes:", list(label_encoder.classes_))

    return X, y_encoded, label_encoder


# =========================
# 4. Train Model
# =========================

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Random Forest model trained successfully.")
    return model


# =========================
# 5. Evaluation
# =========================

def evaluate_model(model, X_test, y_test, label_encoder, experiment_name="Full Model"):
    """
    Evaluate model using accuracy, precision, recall, macro F1-score,
    classification report, and confusion matrix.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n==============================")
    print(f"Evaluation Results: {experiment_name}")
    print("==============================")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {experiment_name}")

    output_path = os.path.join(
        RESULTS_DIR,
        f"confusion_matrix_{experiment_name.replace(' ', '_').lower()}.png"
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")

    return {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1
    }


# =========================
# 6. Ablation Study
# =========================

def remove_timing_features(X):
    """
    Remove timing-based features for ablation study.
    Features containing IAT, Duration, or Time are removed.
    """

    timing_keywords = [
        "IAT",
        "Duration",
        "Time",
        "timestamp",
        "Timestamp"
    ]

    columns_to_remove = []

    for col in X.columns:
        for keyword in timing_keywords:
            if keyword.lower() in col.lower():
                columns_to_remove.append(col)
                break

    X_ablation = X.drop(columns=columns_to_remove, errors="ignore")

    print("\nAblation Study:")
    print("Removed timing features:", len(columns_to_remove))
    print("Remaining features:", X_ablation.shape[1])

    return X_ablation


# =========================
# 7. Cross Validation
# =========================

def run_cross_validation(model, X, y):
    """
    Run 5-fold cross-validation and report mean ± standard deviation.
    """

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    print("\nCross Validation Macro F1:")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std:  {scores.std():.4f}")

    return scores.mean(), scores.std()


# =========================
# 8. Main Program
# =========================

def main():
    # Load dataset
    df = load_dataset(DATA_PATH)

    # Preprocess dataset
    X, y, label_encoder = preprocess_data(df)

    # Split dataset into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("\nData Split:")
    print("Training set:", X_train.shape)
    print("Validation set:", X_val.shape)
    print("Test set:", X_test.shape)

    # Train full model
    full_model = train_random_forest(X_train, y_train)

    # Evaluate on validation set
    evaluate_model(
        full_model,
        X_val,
        y_val,
        label_encoder,
        experiment_name="Validation Full Model"
    )

    # Final evaluation on test set
    full_results = evaluate_model(
        full_model,
        X_test,
        y_test,
        label_encoder,
        experiment_name="Test Full Model"
    )

    # Cross validation
    run_cross_validation(full_model, X_train, y_train)

    # Save trained model
    joblib.dump(full_model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # =========================
    # Ablation Experiment
    # =========================

    X_no_timing = remove_timing_features(X)

    X_train_ab, X_temp_ab, y_train_ab, y_temp_ab = train_test_split(
        X_no_timing,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val_ab, X_test_ab, y_val_ab, y_test_ab = train_test_split(
        X_temp_ab,
        y_temp_ab,
        test_size=0.50,
        random_state=42,
        stratify=y_temp_ab
    )

    ablation_model = train_random_forest(X_train_ab, y_train_ab)

    ablation_results = evaluate_model(
        ablation_model,
        X_test_ab,
        y_test_ab,
        label_encoder,
        experiment_name="Test Without Timing Features"
    )

    # Compare results
    print("\n==============================")
    print("Final Comparison")
    print("==============================")
    print("Full Model:")
    print(f"Accuracy: {full_results['accuracy']:.4f}")
    print(f"Macro F1: {full_results['macro_f1']:.4f}")

    print("\nWithout Timing Features:")
    print(f"Accuracy: {ablation_results['accuracy']:.4f}")
    print(f"Macro F1: {ablation_results['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
