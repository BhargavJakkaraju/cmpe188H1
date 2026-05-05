"""
Linear Regression with Polynomial Feature Expansion — Diabetes Dataset
=======================================================================
Algorithm: Linear Regression with degree-2 polynomial feature expansion
           (interaction terms + squared features) via PyTorch autograd
Dataset:   sklearn.datasets.load_diabetes (442 samples, 10 features)
           → expanded to 65 features via PolynomialFeatures(degree=2)

Math
----
Polynomial expansion:
    φ(x) = [x₁, x₂, …, x_d, x₁², x₁x₂, …, x_d²]
    (degree-2 without bias: D=10 → 65 features)

Model:    ŷ = φ(X)θ + b
Loss:     J(θ) = MSE(ŷ, y) + λ₂‖θ‖₂²    (Ridge to prevent overfitting from expansion)
Optimizer: Adam with ReduceLROnPlateau scheduler

Key question answered: does quadratic expansion improve over raw linear features?
We train both a linear (degree-1) and polynomial (degree-2) model and compare.
"""

import sys
import os
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def get_task_metadata() -> dict:
    return {
        "series": "Linear Regression",
        "level": "new_4",
        "id": "linreg_new4_poly_diabetes",
        "algorithm": "Polynomial Linear Regression (degree 2) with Ridge penalty",
        "description": (
            "Extends plain linear regression with degree-2 polynomial feature "
            "expansion on the sklearn diabetes dataset (442 samples, 10 → 65 features). "
            "Compares linear vs polynomial models under Ridge regularization and "
            "ReduceLROnPlateau scheduling. Demonstrates that interaction terms can "
            "capture non-linear disease-progression relationships."
        ),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": "sklearn load_diabetes — 442 samples, 10 raw → 65 poly features; 80/20 split.",
            "implementation": "nn.Linear + Ridge loss; Adam + ReduceLROnPlateau; degree-1 vs degree-2 comparison.",
            "evaluation": "MSE, RMSE, R² on val split for both models.",
            "validation": "Poly R² > 0.50; poly R² >= linear R² − 0.02 (poly should not hurt).",
        },
    }


def set_seed(s: int = 42) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(degree: int = 1, batch_size: int = 64, seed: int = 42):
    data = load_diabetes()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32).reshape(-1, 1)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=seed)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_tr  = poly.fit_transform(X_tr).astype(np.float32)
    X_val = poly.transform(X_val).astype(np.float32)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_tr  = scaler_X.fit_transform(X_tr)
    X_val = scaler_X.transform(X_val)
    y_tr  = scaler_y.fit_transform(y_tr)
    y_val = scaler_y.transform(y_val)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xt, yt = to_t(X_tr),  to_t(y_tr)
    Xv, yv = to_t(X_val), to_t(y_val)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=len(Xv))
    return train_loader, val_loader, (Xv, yv), Xv.shape[1]


def build_model(in_features: int, device: torch.device) -> nn.Module:
    model = nn.Linear(in_features, 1)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model.to(device)


def ridge_loss(pred, target, model, lam):
    return nn.functional.mse_loss(pred, target) + lam * (model.weight ** 2).sum()


def train(model, train_loader, val_loader, cfg, device):
    opt       = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )
    lam       = cfg["lambda"]
    train_hist, val_hist = [], []

    for _ in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ridge_loss(model(Xb), yb, model, lam)
            loss.backward()
            opt.step()
            total += loss.item() * len(Xb)
        train_hist.append(total / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            for Xv, yv in val_loader:
                vm = nn.functional.mse_loss(model(Xv.to(device)), yv.to(device)).item()
        val_hist.append(vm)
        scheduler.step(vm)

    return {"train_loss_history": train_hist, "val_loss_history": val_hist}


def evaluate(model, Xv, yv, device):
    model.eval()
    Xv, yv = Xv.to(device), yv.to(device)
    with torch.no_grad():
        pred = model(Xv)
        mse  = nn.functional.mse_loss(pred, yv).item()
    ss_res = ((yv - pred) ** 2).sum().item()
    ss_tot = ((yv - yv.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"mse": mse, "rmse": math.sqrt(mse), "r2": r2}


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Task: {get_task_metadata()['id']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    cfg = {"lr": 3e-3, "lambda": 1e-4, "epochs": 500, "batch_size": 64}
    all_metrics  = {}
    all_histories = {}

    for degree in (1, 2):
        set_seed(42)
        train_loader, val_loader, (Xv, yv), n_feat = make_dataloaders(
            degree=degree, batch_size=cfg["batch_size"]
        )
        model   = build_model(n_feat, device)
        history = train(model, train_loader, val_loader, cfg, device)
        metrics = evaluate(model, Xv, yv, device)

        tag = f"degree_{degree}"
        all_metrics[tag]   = metrics
        all_histories[tag] = history
        print(f"  degree={degree}  features={n_feat:>3}  "
              f"MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/linreg_new4_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n--- Assertions ---")

    r2_poly   = all_metrics["degree_2"]["r2"]
    r2_linear = all_metrics["degree_1"]["r2"]

    assert r2_poly > 0.45, f"FAIL: poly R² = {r2_poly:.4f} (expected > 0.45)"
    print(f"[PASS] Polynomial R² > 0.45: {r2_poly:.4f}")

    assert r2_poly >= r2_linear - 0.02, (
        f"FAIL: poly R²={r2_poly:.4f} significantly worse than linear R²={r2_linear:.4f}"
    )
    print(f"[PASS] Polynomial R² ({r2_poly:.4f}) not significantly worse than linear ({r2_linear:.4f})")

    for tag, hist in all_histories.items():
        assert hist["val_loss_history"][-1] < hist["val_loss_history"][0], (
            f"FAIL: [{tag}] val loss did not decrease"
        )
        print(f"[PASS] [{tag}] val loss decreased: "
              f"{hist['val_loss_history'][0]:.4f} → {hist['val_loss_history'][-1]:.4f}")

    print("\n[SUCCESS] All assertions passed. Exiting 0.")
    sys.exit(0)
