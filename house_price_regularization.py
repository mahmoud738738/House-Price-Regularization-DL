"""
Reducing Overfitting in House Price Classification.

Compares an unregularized Original Model against a Modified Model using
Dropout (rate=0.3) after removing duplicate rows and scaling strictly on train data.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

SEED = 42
DROPOUT_RATE = 0.3
FIGURES_DIR = "figures"


def load_and_preprocess_data(csv_path: str = "housepricedata.csv"):
    """Load dataset, drop duplicate rows, split, and scale strictly on X_train."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)

    dataset = df.values
    X = dataset[:, 0:10]
    Y = dataset[:, 10]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=1
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def build_model(dropout_rate: float = 0.0, name: str = "Model") -> Sequential:
    """Build 4x1000 ReLU network with optional Dropout regularization."""
    layers = [Input(shape=(10,))]
    for _ in range(4):
        layers.append(Dense(1000, activation="relu"))
        if dropout_rate > 0.0:
            layers.append(Dropout(dropout_rate))
    layers.append(Dense(1, activation="sigmoid"))

    model = Sequential(layers, name=name)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(
    model: Sequential,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
):
    """Train with EarlyStopping and evaluate on train, val, and test splits."""
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=20, mode="min", restore_best_weights=True
    )
    history = model.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, Y_val),
        verbose=1,
        callbacks=[early_stopping],
    )

    train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

    probs = model.predict(X_test, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "epochs_run": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "end_loss_gap": history.history["val_loss"][-1] - history.history["loss"][-1],
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "acc_gap": train_acc - test_acc,
        "test_f1": f1_score(Y_test, preds),
        "test_auc": roc_auc_score(Y_test, probs),
        "preds": preds,
    }
    return history, metrics


def save_plots(hist_orig, orig_metrics, hist_mod, mod_metrics, Y_test):
    """Save loss, accuracy, and confusion matrix comparison plots."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_style("whitegrid")

    ep_orig = range(1, orig_metrics["epochs_run"] + 1)
    ep_mod = range(1, mod_metrics["epochs_run"] + 1)

    plt.figure(figsize=(8.5, 5))
    plt.plot(ep_mod, hist_mod.history["loss"], label="Training Loss", color="#1f77b4", linewidth=2.2)
    plt.plot(ep_mod, hist_mod.history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2.2)
    plt.axvline(
        mod_metrics["best_epoch"],
        color="#2ca02c",
        linestyle="--",
        linewidth=1.6,
        label=f"Best Epoch ({mod_metrics['best_epoch']})",
    )
    plt.title("Modified Model (Dropout 0.3): Training Loss vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_modified_model_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(ep_orig, hist_orig.history["loss"], label="Original - Train Loss", color="#1f77b4", linestyle="--", linewidth=1.8)
    axes[0].plot(ep_orig, hist_orig.history["val_loss"], label="Original - Val Loss", color="#d62728", linewidth=2.2)
    axes[0].plot(ep_mod, hist_mod.history["loss"], label="Modified - Train Loss", color="#17becf", linestyle="--", linewidth=1.8)
    axes[0].plot(ep_mod, hist_mod.history["val_loss"], label="Modified - Val Loss", color="#2ca02c", linewidth=2.2)
    axes[0].set_title("Loss Comparison: Original vs. Modified Model")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(ep_orig, hist_orig.history["accuracy"], label="Original - Train Acc", color="#1f77b4", linestyle="--", linewidth=1.8)
    axes[1].plot(ep_orig, hist_orig.history["val_accuracy"], label="Original - Val Acc", color="#d62728", linewidth=2.2)
    axes[1].plot(ep_mod, hist_mod.history["accuracy"], label="Modified - Train Acc", color="#17becf", linestyle="--", linewidth=1.8)
    axes[1].plot(ep_mod, hist_mod.history["val_accuracy"], label="Modified - Val Acc", color="#2ca02c", linewidth=2.2)
    axes[1].set_title("Accuracy Comparison: Original vs. Modified Model")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_loss_and_accuracy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, preds, title, cmap in [
        (axes[0], orig_metrics["preds"], f"Original Model (Test Acc: {orig_metrics['test_acc']:.4f})", "Reds"),
        (axes[1], mod_metrics["preds"], f"Modified Model + Dropout (Test Acc: {mod_metrics['test_acc']:.4f})", "Blues"),
    ]:
        cm = confusion_matrix(Y_test, preds)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            xticklabels=["Pred: 0", "Pred: 1"],
            yticklabels=["Actual: 0", "Actual: 1"],
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(title)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_and_preprocess_data()

    tf.keras.utils.set_random_seed(SEED)
    orig_model = build_model(dropout_rate=0.0, name="Original_Model")
    hist_orig, orig_metrics = train_and_evaluate(
        orig_model, X_train, Y_train, X_val, Y_val, X_test, Y_test
    )

    tf.keras.utils.set_random_seed(SEED)
    mod_model = build_model(dropout_rate=DROPOUT_RATE, name="Modified_Model_Dropout")
    hist_mod, mod_metrics = train_and_evaluate(
        mod_model, X_train, Y_train, X_val, Y_val, X_test, Y_test
    )

    save_plots(hist_orig, orig_metrics, hist_mod, mod_metrics, Y_test)

    summary = pd.DataFrame([
        {"Model": "Original Model", **{k: round(v, 4) if isinstance(v, float) else v for k, v in orig_metrics.items() if k != "preds"}},
        {"Model": "Modified Model", **{k: round(v, 4) if isinstance(v, float) else v for k, v in mod_metrics.items() if k != "preds"}},
    ])
    print("\n=== Original vs. Modified Model Comparison ===")
    print(summary.set_index("Model").T.to_string())


if __name__ == "__main__":
    main()
