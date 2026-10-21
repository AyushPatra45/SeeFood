"""Final held-out test metrics and reproducible plots; never used to tune the model."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from seefood.data import ImageManifest
from seefood.model import load_checkpoint
from seefood.train import choose_device, run_epoch


def metric_summary(targets, scores, threshold):
    predicted = np.asarray(scores) >= threshold
    return {"accuracy": float(accuracy_score(targets, predicted)),
            "precision": float(precision_score(targets, predicted, zero_division=0)),
            "recall": float(recall_score(targets, predicted, zero_division=0)),
            "f1": float(f1_score(targets, predicted, zero_division=0)),
            "confusion_matrix": confusion_matrix(targets, predicted, labels=[0, 1]).tolist()}


def evaluate(data="data/raw", reports="reports", model_path="artifacts/model.pt", device_name="auto"):
    torch.set_num_threads(4)
    device, reports = choose_device(device_name), Path(reports)
    model, checkpoint = load_checkpoint(model_path, device)
    dataset = ImageManifest(data, reports / "manifest.json", "test")
    stats, scores, targets = run_epoch(model, DataLoader(dataset, batch_size=32), device)
    threshold = checkpoint["threshold"]
    metrics = metric_summary(targets, scores, threshold)
    metrics.update({"test_size": len(dataset), "threshold": threshold, "positive_class": "hot_dog",
                    "confusion_matrix_order": ["not_hot_dog", "hot_dog"], "loss": stats["loss"],
                    "at_default_threshold": metric_summary(targets, scores, 0.5),
                    "checkpoint_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest()})
    # Wilson interval: a useful indication of test-set uncertainty, not a guarantee.
    n, p, z = len(targets), metrics["accuracy"], 1.96
    center = (p + z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    metrics["accuracy_95_percent_wilson_interval"] = [float(center-radius), float(center+radius)]
    predictions = [{"path": row["path"], "true_label": int(y), "predicted_label": int(p >= threshold),
                    "probability_hot_dog": float(p), "correct": bool((p >= threshold) == y)}
                   for row, y, p in zip(dataset.rows, targets, scores)]
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "test_predictions.json").write_text(json.dumps(predictions, indent=2))
    errors = [r for r in predictions if not r["correct"]]
    errors.sort(key=lambda r: abs(r["probability_hot_dog"] - threshold), reverse=True)
    (reports / "error_cases.json").write_text(json.dumps(errors, indent=2))
    plt.rcParams.update({"figure.facecolor": "#faf9f6", "axes.facecolor": "#faf9f6", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.array(metrics["confusion_matrix"]), display_labels=["Not hotdog", "Hotdog"]).plot(ax=ax, cmap="Oranges", colorbar=False)
    ax.set_title(f"Held-out test set · n={len(dataset)}", pad=16)
    fig.tight_layout()
    fig.savefig(reports / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    history = json.loads((reports / "history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history)+1)
    for ax, key, title in zip(axes, ["loss", "accuracy"], ["Binary cross-entropy", "Accuracy at threshold 0.5"]):
        ax.plot(epochs, [r[f"train_{key}"] for r in history], label="Training", color="#ea5b2a", linewidth=2)
        ax.plot(epochs, [r[f"val_{key}"] for r in history], label="Validation", color="#326b58", linewidth=2)
        boundary = next((i+0.5 for i, r in enumerate(history) if r["stage"] == "finetune"), None)
        if boundary:
            ax.axvline(boundary, color="#777", linestyle="--", alpha=0.6, label="Fine-tuning begins")
        ax.set(xlabel="Epoch", title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(reports / "training_curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--model", default="artifacts/model.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args.data, args.reports, args.model, args.device)
