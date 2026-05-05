"""
Logistic Regression with Dropout Regularization — Breast Cancer Dataset
========================================================================
Algorithm: Binary Logistic Regression + Dropout (Monte Carlo Dropout at inference)
Dataset:   sklearn.datasets.load_breast_cancer (569 samples, 30 features)

Math
----
Model with Dropout (p_drop probability of zeroing a unit during training):
    h = Dropout(p)(x)
    ŷ = σ(Wh + b),   σ(z) = 1/(1+e^{-z})

During training:  units dropped at random → acts as an ensemble of sub-networks.
During inference (MC Dropout): keep Dropout ON for T forward passes, then average:
    p̄(y=1|x) ≈ (1/T) Σ_t σ(W·Dropout_t(x) + b)
    uncertainty ≈ std_t[σ(W·Dropout_t(x) + b)]

Loss:  J(θ) = BCE(ŷ, y)   (no explicit penalty; Dropout provides implicit regularisation)
Optimizer: Adam

Compared variants: no_dropout (p=0), dropout_0.2, dropout_0.5
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
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def get_task_metadata() -> dict:
    return {
        "series": "Logistic Regression",
        "level": "new_5",
        "id": "logreg_new5_dropout_cancer",
        "algorithm": "Logistic Regression with Dropout + Monte Carlo Uncertainty Estimation",
        "description": (
            "Binary classification of malignant vs benign tumours (breast cancer) "
            "using a logistic unit with Dropout regularization. Compares three dropout "
            "rates (0, 0.2, 0.5) and demonstrates MC Dropout for prediction-interval "
            "uncertainty estimation. New optimizer feature: gradient clipping."
        ),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": "sklearn load_breast_cancer — 569 samples, 30 features; 80/20 split.",
            "implementation": "nn.Dropout + nn.Linear; Adam + gradient clipping; three p_drop variants.",
            "evaluation": "Accuracy, F1, ROC-AUC, confusion matrix; MC Dropout uncertainty on val set.",
            "validation": "Best variant F1 > 0.93; ROC-AUC > 0.97; MC std mean < 0.15.",
        },
    }


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DropoutLogisticRegressor(nn.Module):
    """Single linear layer with configurable input dropout."""

    def __init__(self, in_features: int, p_drop: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)
        self.linear  = nn.Linear(in_features, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))


def make_dataloaders(batch_size: int = 32, seed: int = 42):
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32).reshape(-1, 1)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_val  = scaler.transform(X_val)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xt, yt = to_t(X_tr), to_t(y_tr)
    Xv, yv = to_t(X_val), to_t(y_val)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=len(Xv))
    return train_loader, val_loader, (Xv, yv)


def train(model, train_loader, val_loader, cfg, device):
    opt    = optim.Adam(model.parameters(), lr=cfg["lr"])
    epochs = cfg["epochs"]
    clip   = cfg.get("grad_clip", 5.0)
    train_hist, val_hist = [], []

    for _ in range(epochs):
        model.train()
        losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(loss.item())
        train_hist.append(float(np.mean(losses)))

        model.eval()
        with torch.no_grad():
            for Xv, yv in val_loader:
                vl = nn.functional.binary_cross_entropy_with_logits(
                    model(Xv.to(device)), yv.to(device)
                ).item()
        val_hist.append(vl)

    return {"train_loss_history": train_hist, "val_loss_history": val_hist}


def evaluate(model, Xv, yv, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(Xv.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy().ravel()
    preds = (probs >= threshold).astype(int)
    y_np  = yv.numpy().ravel().astype(int)
    acc   = float((preds == y_np).mean())
    f1    = float(f1_score(y_np, preds, zero_division=0))
    auc   = float(roc_auc_score(y_np, probs))
    cm    = confusion_matrix(y_np, preds).tolist()
    return {"accuracy": acc, "f1": f1, "roc_auc": auc, "confusion_matrix": cm}


def mc_dropout_uncertainty(model, Xv, device, T=50, threshold=0.5):
    """Run T stochastic forward passes with Dropout ON to estimate uncertainty."""
    model.train()
    preds_T = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(Xv.to(device))
            preds_T.append(torch.sigmoid(logits).cpu().numpy().ravel())
    model.eval()

    preds_T   = np.stack(preds_T, axis=0)
    mean_pred = preds_T.mean(axis=0)
    std_pred  = preds_T.std(axis=0)
    return float(std_pred.mean()), float(std_pred.max())


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Task: {get_task_metadata()['id']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    cfg = {"lr": 5e-3, "epochs": 400, "batch_size": 32, "grad_clip": 5.0}
    train_loader, val_loader, (Xv, yv) = make_dataloaders(batch_size=cfg["batch_size"])
    in_features = Xv.shape[1]

    all_metrics   = {}
    all_histories = {}
    best_f1, best_tag = -1.0, ""

    for p_drop in (0.0, 0.2, 0.5):
        set_seed(42)
        model   = DropoutLogisticRegressor(in_features=in_features, p_drop=p_drop).to(device)
        history = train(model, train_loader, val_loader, cfg, device)
        metrics = evaluate(model, Xv, yv, device)
        mc_mean_std, mc_max_std = mc_dropout_uncertainty(model, Xv, device)

        tag = f"dropout_{p_drop:.1f}"
        metrics["mc_mean_std"] = mc_mean_std
        metrics["mc_max_std"]  = mc_max_std
        all_metrics[tag]   = metrics
        all_histories[tag] = history

        if metrics["f1"] > best_f1:
            best_f1, best_tag = metrics["f1"], tag

        print(f"  p_drop={p_drop:.1f}  "
              f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
              f"AUC={metrics['roc_auc']:.4f}  MC_std={mc_mean_std:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/logreg_new5_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nBest variant: {best_tag}  (F1={best_f1:.4f})")

    print("\n--- Assertions ---")

    assert best_f1 > 0.93, f"FAIL: best F1 = {best_f1:.4f} (expected > 0.93)"
    print(f"[PASS] Best variant F1 > 0.93: {best_f1:.4f} ({best_tag})")

    best_auc = all_metrics[best_tag]["roc_auc"]
    assert best_auc > 0.97, f"FAIL: ROC-AUC = {best_auc:.4f} (expected > 0.97)"
    print(f"[PASS] ROC-AUC > 0.97: {best_auc:.4f}")

    mc_std = all_metrics[best_tag]["mc_mean_std"]
    assert mc_std < 0.15, f"FAIL: MC Dropout mean std = {mc_std:.4f} (expected < 0.15)"
    print(f"[PASS] MC Dropout mean uncertainty < 0.15: {mc_std:.4f}")

    for tag, hist in all_histories.items():
        assert hist["val_loss_history"][-1] < hist["val_loss_history"][0], (
            f"FAIL: [{tag}] val loss did not decrease"
        )
    print("[PASS] All variants: val BCE decreased over training")

    print("\n[SUCCESS] All assertions passed. Exiting 0.")
    sys.exit(0)
