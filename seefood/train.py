"""Two-stage transfer learning. Select checkpoints and threshold using validation only."""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from seefood.data import ImageManifest
from seefood.model import CLASS_TO_IDX, build_model, configure_stage


def choose_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    # Small batches must not overwrite pretrained running means/variances.
    if training:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, scores, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images).flatten()
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            scores.extend(logits.detach().sigmoid().cpu().tolist())
            targets.extend(labels.cpu().tolist())
    scores, targets = np.array(scores), np.array(targets)
    return {"loss": total_loss / len(targets), "accuracy": float(np.mean((scores >= 0.5) == targets))}, scores, targets


def select_threshold(targets, scores):
    candidates = np.linspace(0.1, 0.9, 81)
    # Resolve ties toward the conventional 0.5 boundary.
    return float(max(candidates, key=lambda t: (f1_score(targets, scores >= t, zero_division=0), -abs(t - 0.5))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--output", default="artifacts/model.pt")
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.head_epochs, args.finetune_epochs) < 1:
        parser.error("Both stages require at least one epoch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = choose_device(args.device)
    reports = Path(args.reports)
    manifest = reports / "manifest.json"
    train_data = ImageManifest(args.data, manifest, "train", training=True)
    val_data = ImageManifest(args.data, manifest, "val")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers)
    model = build_model(pretrained=True).to(device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started, best_loss, history = time.perf_counter(), float("inf"), []
    config = vars(args) | {"architecture": "mobilenet_v2", "pretrained_weights": "IMAGENET1K_V2", "device": str(device),
                          "head_lr": 1e-3, "finetune_lr": 1e-5, "weight_decay": 1e-4,
                          "image_size": 224, "torch_version": str(torch.__version__), "python": platform.python_version()}
    print(json.dumps(config), flush=True)
    for stage, epochs, lr in [("head", args.head_epochs, 1e-3), ("finetune", args.finetune_epochs, 1e-5)]:
        if stage == "finetune":
            checkpoint = torch.load(output, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
        configure_stage(model, stage)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        stale, stage_best = 0, float("inf")
        for epoch in range(1, epochs + 1):
            train_stats, _, _ = run_epoch(model, train_loader, device, optimizer)
            val_stats, _, _ = run_epoch(model, val_loader, device)
            row = {"stage": stage, "epoch": epoch, "train_loss": train_stats["loss"], "val_loss": val_stats["loss"],
                   "train_accuracy": train_stats["accuracy"], "val_accuracy": val_stats["accuracy"],
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "elapsed_seconds": round(time.perf_counter() - started, 2)}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val_stats["loss"] < best_loss:
                best_loss = val_stats["loss"]
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "class_to_idx": CLASS_TO_IDX, "threshold": 0.5, "config": config,
                            "best_stage": stage, "best_epoch": epoch, "val_loss": best_loss}, output)
            if val_stats["loss"] < stage_best - 1e-4:
                stage_best, stale = val_stats["loss"], 0
            else:
                stale += 1
            (reports / "history.json").write_text(json.dumps(history, indent=2))
            if stale >= 3:
                print(f"Early stopping {stage}", flush=True)
                break
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    _, scores, targets = run_epoch(model, val_loader, device)
    threshold = select_threshold(targets, scores)
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, output)
    metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    metadata["training_seconds"] = round(time.perf_counter() - started, 2)
    metadata["threshold_selection"] = {"split": "val", "criterion": "maximum F1, ties closest to 0.5", "f1_at_0_5": f1_score(targets, scores >= 0.5), "f1_selected": f1_score(targets, scores >= threshold)}
    (output.parent / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (reports / "validation_predictions.json").write_text(json.dumps([{"path": row["path"], "label": int(y), "probability_hot_dog": float(p)} for row, y, p in zip(val_data.rows, targets, scores)], indent=2))
    print(f"Saved best model: {output}; validation-selected threshold: {threshold:.2f}", flush=True)


if __name__ == "__main__":
    main()
