"""
Softmax Multiclass Logistic Regression — Wine Dataset
======================================================
Algorithm: Multinomial (Softmax) Logistic Regression via PyTorch CrossEntropyLoss
Dataset:   sklearn.datasets.load_wine (178 samples, 13 features, 3 classes)

Math
----
Softmax output for class k:
    P(y=k | x) = exp(x^T θ_k + b_k) / Σ_j exp(x^T θ_j + b_j)

Cross-Entropy Loss (negative log-likelihood):
    J(Θ) = -(1/N) Σ_i log P(y=y_i | x_i)
          = -(1/N) Σ_i [ (x_i^T θ_{y_i} + b_{y_i}) - log Σ_j exp(x_i^T θ_j + b_j) ]

vs One-vs-Rest (OvR from task4_logreg):
  • Softmax normalises across ALL classes jointly → calibrated probabilities
  • OvR trains K independent binary classifiers → scores not normalised
  • Softmax has (K−1)×D fewer effective parameters (sum-to-1 constraint)

Regularisation variants compared: None, L1, L2, ElasticNet (via weight_decay + manual L1)
"""

import sys
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix


def get_task_metadata() -> dict:
    return {
        "series": "Logistic Regression",
        "level": "new_7",
        "id": "logreg_new7_softmax_wine",
        "algorithm": "Softmax (Multinomial) Logistic Regression with regularisation ablation",
        "description": (
            "Joint K-way (softmax) logistic regression on the Wine dataset (3 classes). "
            "Ablates four regularisation strategies: none, L2 (weight_decay), L1 (manual), "
            "and ElasticNet (L1+L2). Also adds 5-fold stratified cross-validation to report "
            "mean±std accuracy — more robust than a single train/val split."
        ),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": "sklearn load_wine — 178 samples, 13 features, 3 classes; 80/20 split + 5-fold CV.",
            "implementation": "nn.Linear(13,3) + CrossEntropyLoss; four regularisation variants; StratifiedKFold.",
            "evaluation": "Accuracy, macro-F1, confusion matrix; 5-fold mean±std accuracy for best variant.",
            "validation": "Best variant accuracy > 0.94; macro-F1 > 0.94; CV mean accuracy > 0.92.",
        },
    }


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size: int = 32, seed: int = 42):
    data = load_wine()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_val  = scaler.transform(X_val)

    to_t_x = lambda a: torch.tensor(a, dtype=torch.float32)
    to_t_y = lambda a: torch.tensor(a, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(to_t_x(X_tr), to_t_y(y_tr)), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(to_t_x(X_val), to_t_y(y_val)), batch_size=len(X_val)
    )
    return (train_loader, val_loader,
            (to_t_x(X_val), to_t_y(y_val)),
            X, y, scaler)


def build_model(in_features: int, n_classes: int, device: torch.device) -> nn.Module:
    model = nn.Linear(in_features, n_classes)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model.to(device)


def compute_loss(logits, labels, model, reg_type, lambda1, lambda2):
    ce = nn.functional.cross_entropy(logits, labels)
    if reg_type == "none":
        return ce
    w = model.weight
    if reg_type == "l2":
        return ce + lambda2 * (w ** 2).sum()
    if reg_type == "l1":
        return ce + lambda1 * w.abs().sum()
    if reg_type == "elasticnet":
        return ce + lambda1 * w.abs().sum() + lambda2 * (w ** 2).sum()
    raise ValueError(f"Unknown reg_type: {reg_type}")


def train(model, train_loader, val_loader, cfg, device):
    opt     = optim.Adam(model.parameters(), lr=cfg["lr"])
    reg     = cfg.get("reg_type", "none")
    l1, l2  = cfg.get("lambda1", 0.0), cfg.get("lambda2", 0.0)
    epochs  = cfg["epochs"]
    train_hist, val_hist = [], []

    for _ in range(epochs):
        model.train()
        losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = compute_loss(model(Xb), yb, model, reg, l1, l2)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_hist.append(float(np.mean(losses)))

        model.eval()
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                vl = nn.functional.cross_entropy(model(Xv), yv).item()
        val_hist.append(vl)

    return {"train_loss_history": train_hist, "val_loss_history": val_hist}


def evaluate(model, Xv, yv, device):
    model.eval()
    Xv, yv = Xv.to(device), yv.to(device)
    with torch.no_grad():
        preds = model(Xv).argmax(dim=1).cpu().numpy()
    y_np = yv.cpu().numpy()
    acc  = float((preds == y_np).mean())
    f1   = float(f1_score(y_np, preds, average="macro", zero_division=0))
    cm   = confusion_matrix(y_np, preds).tolist()
    return {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm}


def cross_val_accuracy(X_raw, y_raw, cfg, device, n_splits=5, seed=42):
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs   = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_raw, y_raw)):
        set_seed(seed + fold)
        X_tr, X_val = X_raw[tr_idx], X_raw[val_idx]
        y_tr, y_val = y_raw[tr_idx], y_raw[val_idx]

        sc = StandardScaler()
        X_tr  = sc.fit_transform(X_tr).astype(np.float32)
        X_val = sc.transform(X_val).astype(np.float32)

        to_t_x = lambda a: torch.tensor(a, dtype=torch.float32)
        to_t_y = lambda a: torch.tensor(a, dtype=torch.long)

        tl = DataLoader(TensorDataset(to_t_x(X_tr), to_t_y(y_tr)),
                        batch_size=32, shuffle=True)
        vl = DataLoader(TensorDataset(to_t_x(X_val), to_t_y(y_val)),
                        batch_size=len(X_val))

        m = build_model(X_raw.shape[1], len(np.unique(y_raw)), device)
        train(m, tl, vl, cfg, device)
        met = evaluate(m, to_t_x(X_val), to_t_y(y_val), device)
        accs.append(met["accuracy"])
    return float(np.mean(accs)), float(np.std(accs))


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Task: {get_task_metadata()['id']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    base_cfg   = {"lr": 1e-2, "epochs": 500}
    reg_cfgs   = {
        "none":       {**base_cfg, "reg_type": "none"},
        "l2":         {**base_cfg, "reg_type": "l2",         "lambda2": 1e-3},
        "l1":         {**base_cfg, "reg_type": "l1",         "lambda1": 5e-4},
        "elasticnet": {**base_cfg, "reg_type": "elasticnet", "lambda1": 3e-4, "lambda2": 3e-4},
    }

    train_loader, val_loader, (Xv, yv), X_raw, y_raw, _ = make_dataloaders()
    in_features = Xv.shape[1]
    n_classes   = int(y_raw.max()) + 1

    all_metrics   = {}
    all_histories = {}
    best_acc, best_cfg_name = -1.0, ""

    for name, cfg in reg_cfgs.items():
        set_seed(42)
        model   = build_model(in_features, n_classes, device)
        history = train(model, train_loader, val_loader, cfg, device)
        metrics = evaluate(model, Xv, yv, device)
        all_metrics[name]   = metrics
        all_histories[name] = history

        if metrics["accuracy"] > best_acc:
            best_acc, best_cfg_name = metrics["accuracy"], name

        print(f"  [{name:>11}]  "
              f"Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}")

    print(f"\nRunning 5-fold CV on best variant ({best_cfg_name}) …")
    cv_mean, cv_std = cross_val_accuracy(
        X_raw, y_raw, reg_cfgs[best_cfg_name], device
    )
    print(f"  CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    all_metrics["cv_best"] = {"mean_accuracy": cv_mean, "std_accuracy": cv_std,
                               "variant": best_cfg_name}

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/logreg_new7_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n--- Assertions ---")

    assert best_acc > 0.94, f"FAIL: best accuracy = {best_acc:.4f} (expected > 0.94)"
    print(f"[PASS] Best variant accuracy > 0.94: {best_acc:.4f} ({best_cfg_name})")

    best_f1 = all_metrics[best_cfg_name]["macro_f1"]
    assert best_f1 > 0.94, f"FAIL: best macro-F1 = {best_f1:.4f} (expected > 0.94)"
    print(f"[PASS] Best variant macro-F1 > 0.94: {best_f1:.4f}")

    assert cv_mean > 0.92, f"FAIL: 5-fold CV mean accuracy = {cv_mean:.4f} (expected > 0.92)"
    print(f"[PASS] 5-fold CV mean accuracy > 0.92: {cv_mean:.4f} ± {cv_std:.4f}")

    for name, hist in all_histories.items():
        assert hist["val_loss_history"][-1] < hist["val_loss_history"][0], (
            f"FAIL: [{name}] val CE did not decrease"
        )
    print("[PASS] All regularisation variants: val CE decreased over training")

    print("\n[SUCCESS] All assertions passed. Exiting 0.")
    sys.exit(0)
