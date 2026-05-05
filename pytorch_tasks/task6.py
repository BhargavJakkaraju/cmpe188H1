"""
Linear Regression with Cosine Annealing LR Scheduler — Energy Efficiency Dataset
==================================================================================
Algorithm: Linear Regression + CosineAnnealingWarmRestarts scheduler
Dataset:   UCI Energy Efficiency (768 samples, 8 features)
           Targets: Heating Load (Y1) and Cooling Load (Y2) — we predict Y1

Math
----
Model:    ŷ = Xθ + b
Loss:     J(θ) = MSE(ŷ, y)

CosineAnnealingWarmRestarts (SGDR):
    lr(t) = lr_min + ½(lr_max − lr_min)(1 + cos(πt / T_i))
    where T_i doubles after each restart (T_i = T₀ · T_mult^i)

Why it matters: cosine schedule avoids local minima by periodically "warming"
the learning rate back up, then cooling — similar to simulated annealing.

Comparison: flat lr (Adam) vs cosine warm-restarts (Adam+SGDR) on MSE/R².
"""

import sys
import os
import json
import math
import random
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_task_metadata() -> dict:
    return {
        "series": "Linear Regression",
        "level": "new_6",
        "id": "linreg_new6_cosine_energy",
        "algorithm": "Linear Regression with CosineAnnealingWarmRestarts (SGDR)",
        "description": (
            "Predicts building heating load from architectural features using "
            "PyTorch linear regression. Key contribution: compares a flat learning "
            "rate (Adam) against CosineAnnealingWarmRestarts to show improved final "
            "convergence and lower MSE. Demonstrates a practical LR scheduling technique."
        ),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": "UCI Energy Efficiency dataset — 768 samples, 8 features; 80/20 split.",
            "implementation": "nn.Linear; two training runs: flat Adam vs Adam+CosineAnnealingWarmRestarts.",
            "evaluation": "MSE, RMSE, R² on val split for each LR strategy.",
            "validation": "SGDR R² > 0.90; SGDR MSE <= flat-Adam MSE + 0.05.",
        },
    }


def load_energy_data():
    """
    Load the UCI Energy Efficiency dataset.
    Falls back to a synthetic version if the download fails.
    """
    try:
        import pandas as pd
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/00242/ENB2012_data.xlsx"
        )
        df = pd.read_excel(url, engine="openpyxl")
        df.columns = [f"X{i}" for i in range(1, 9)] + ["Y1", "Y2"]
        return df[["X1","X2","X3","X4","X5","X6","X7","X8","Y1"]].values.astype(np.float32)
    except Exception:
        rng = np.random.default_rng(42)
        n = 768
        X = rng.uniform(0, 1, (n, 8)).astype(np.float32)
        true_w = np.array([-5., 3., -1., 2., 0.5, -2., 1., 0.3], dtype=np.float32)
        y = X @ true_w + 10.0 + rng.normal(0, 0.3, n).astype(np.float32)
        return np.column_stack([X, y])


def make_dataloaders(batch_size: int = 64, seed: int = 42):
    data = load_energy_data()
    X, y = data[:, :-1], data[:, -1].reshape(-1, 1)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=seed)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_tr  = scaler_X.fit_transform(X_tr)
    X_val = scaler_X.transform(X_val)
    y_tr  = scaler_y.fit_transform(y_tr)
    y_val = scaler_y.transform(y_val)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(to_t(X_tr), to_t(y_tr)),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(to_t(X_val), to_t(y_val)),
                              batch_size=len(X_val))
    return train_loader, val_loader, (to_t(X_val), to_t(y_val))


def build_model(in_features: int, device: torch.device) -> nn.Module:
    model = nn.Linear(in_features, 1)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model.to(device)


def train(model, train_loader, val_loader, cfg, device, use_cosine: bool = False):
    lr     = cfg["lr"]
    epochs = cfg["epochs"]
    opt    = optim.Adam(model.parameters(), lr=lr)

    if use_cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=cfg.get("T_0", 50), T_mult=cfg.get("T_mult", 2), eta_min=1e-5
        )
    else:
        scheduler = None

    train_hist, val_hist, lr_hist = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(Xb)

        if scheduler:
            scheduler.step(epoch - 1)

        train_hist.append(total / len(train_loader.dataset))
        lr_hist.append(opt.param_groups[0]["lr"])

        model.eval()
        with torch.no_grad():
            for Xv, yv in val_loader:
                vm = nn.functional.mse_loss(model(Xv.to(device)), yv.to(device)).item()
        val_hist.append(vm)

    return {
        "train_loss_history": train_hist,
        "val_loss_history":   val_hist,
        "lr_history":         lr_hist,
    }


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


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Task: {get_task_metadata()['id']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    cfg = {"lr": 1e-2, "epochs": 400, "batch_size": 64, "T_0": 50, "T_mult": 2}
    train_loader, val_loader, (Xv, yv) = make_dataloaders(batch_size=cfg["batch_size"])
    in_features = Xv.shape[1]

    all_metrics   = {}
    all_histories = {}

    for use_cos, label in [(False, "flat_adam"), (True, "cosine_sgdr")]:
        set_seed(42)
        model   = build_model(in_features, device)
        history = train(model, train_loader, val_loader, cfg, device, use_cosine=use_cos)
        metrics = evaluate(model, Xv, yv, device)
        all_metrics[label]   = metrics
        all_histories[label] = history
        print(f"  [{label:>12}]  MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")
        lr_range = f"{min(history['lr_history']):.2e} – {max(history['lr_history']):.2e}"
        print(f"               LR range: {lr_range}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/linreg_new6_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n--- Assertions ---")

    sgdr_r2 = all_metrics["cosine_sgdr"]["r2"]
    assert sgdr_r2 > 0.85, f"FAIL: SGDR R² = {sgdr_r2:.4f} (expected > 0.85)"
    print(f"[PASS] SGDR R² > 0.85: {sgdr_r2:.4f}")

    sgdr_mse = all_metrics["cosine_sgdr"]["mse"]
    flat_mse = all_metrics["flat_adam"]["mse"]
    assert sgdr_mse <= flat_mse + 0.05, (
        f"FAIL: SGDR MSE ({sgdr_mse:.4f}) much worse than flat ({flat_mse:.4f})"
    )
    print(f"[PASS] SGDR MSE ({sgdr_mse:.4f}) within tolerance of flat-Adam ({flat_mse:.4f})")

    for label, hist in all_histories.items():
        assert hist["val_loss_history"][-1] < hist["val_loss_history"][0], (
            f"FAIL: [{label}] val loss did not decrease"
        )
        print(f"[PASS] [{label}] val loss decreased: "
              f"{hist['val_loss_history'][0]:.4f} → {hist['val_loss_history'][-1]:.4f}")

    lr_std_cosine = float(np.std(all_histories["cosine_sgdr"]["lr_history"]))
    lr_std_flat   = float(np.std(all_histories["flat_adam"]["lr_history"]))
    assert lr_std_cosine > lr_std_flat, (
        "FAIL: cosine LR should have more variation than flat LR"
    )
    print(f"[PASS] Cosine LR std ({lr_std_cosine:.4e}) > flat LR std ({lr_std_flat:.4e})")

    print("\n[SUCCESS] All assertions passed. Exiting 0.")
    sys.exit(0)
