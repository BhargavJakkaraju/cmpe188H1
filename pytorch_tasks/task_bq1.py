"""
BigQuery Task 1 — Linear Regression: Birth Weight Prediction
=============================================================
Dataset:   bigquery-public-data.samples.natality
           (US birth records 1969–2008; we sample years 2000–2008)
Target:    weight_pounds  (continuous — birth weight in lbs)
Features:  mother_age, gestation_weeks, year, is_male,
           mother_married (5 numeric/boolean predictors)

New beyond the sample notebook
-------------------------------
The BigQuery-intro notebook only runs exploratory queries (counts, averages).
This task goes further:
  • Uses TABLESAMPLE SYSTEM for reproducible random sampling directly in SQL
  • Engineers features and cleans nulls entirely in BigQuery SQL
  • Trains a PyTorch ElasticNet linear regression on the streamed data
  • Reports MSE, R², and coefficient magnitudes on a held-out split

Algorithm
---------
Model:   ŷ = Xθ + b      (nn.Linear)
Loss:    J(θ) = MSE + λ1·‖θ‖₁ + λ2·‖θ‖₂²   (ElasticNet)
Optimizer: Adam

BigQuery feature demonstrated
------------------------------
  TABLESAMPLE SYSTEM(p PERCENT) — native random sampling pushed to BQ
  (avoids ORDER BY RAND() full-table scan, much cheaper on large tables)
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from google.cloud import bigquery
    PROJECT_ID = os.environ.get("GCP_PROJECT", "cmpelkk")
    BQ_CLIENT  = bigquery.Client(project=PROJECT_ID)
    BQ_AVAILABLE = True
except Exception as e:
    print(f"[WARN] BigQuery unavailable ({e}). Falling back to synthetic data.")
    BQ_AVAILABLE = False


def get_task_metadata() -> dict:
    return {
        "series": "Linear Regression",
        "level": "bq_1",
        "id": "linreg_bq1_natality_birthweight",
        "algorithm": "Linear Regression (ElasticNet) — BigQuery natality dataset",
        "description": (
            "Predict US infant birth weight (weight_pounds) from mother age, "
            "gestation weeks, year, sex, and marital status. Data loaded from "
            "bigquery-public-data.samples.natality via TABLESAMPLE SYSTEM SQL. "
            "Trains a PyTorch ElasticNet model; extends the notebook's EDA-only approach."
        ),
        "interface_protocol": "pytorch_task_v1",
        "bigquery_features": [
            "TABLESAMPLE SYSTEM(p PERCENT) — cheap random sampling at storage layer",
            "IS NOT NULL filters and CAST in SQL for server-side cleaning",
            "client.query().to_dataframe() stream to pandas",
        ],
        "requirements": {
            "data": "bigquery-public-data.samples.natality, years 2000-2008, sampled 1%.",
            "implementation": "nn.Linear + ElasticNet loss; Adam; device-agnostic.",
            "evaluation": "MSE, R² on 20% val split.",
            "validation": "R² > 0.20 (birth weight is inherently noisy); MSE < 1.5.",
        },
    }


BQ_QUERY = """
SELECT
    CAST(mother_age        AS FLOAT64) AS mother_age,
    CAST(gestation_weeks   AS FLOAT64) AS gestation_weeks,
    CAST(year              AS FLOAT64) AS year,
    CAST(is_male           AS FLOAT64) AS is_male,
    CAST(mother_married    AS FLOAT64) AS mother_married,
    weight_pounds
FROM
    `bigquery-public-data.samples.natality`
    TABLESAMPLE SYSTEM (1 PERCENT)
WHERE
    year BETWEEN 2000 AND 2008
    AND weight_pounds   IS NOT NULL
    AND weight_pounds   BETWEEN 1.0 AND 15.0
    AND gestation_weeks IS NOT NULL
    AND gestation_weeks BETWEEN 20 AND 45
    AND mother_age      IS NOT NULL
    AND mother_married  IS NOT NULL
    AND is_male         IS NOT NULL
LIMIT 20000
"""

FEATURE_COLS = ["mother_age", "gestation_weeks", "year", "is_male", "mother_married"]
TARGET_COL   = "weight_pounds"


def _synthetic_fallback(n: int = 5000, seed: int = 42) -> "pd.DataFrame":
    """Generate plausible synthetic birth-weight data when BQ is unavailable."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    gestation = rng.normal(39, 2, n).clip(28, 45)
    mother_age = rng.integers(15, 45, n).astype(float)
    year = rng.integers(2000, 2009, n).astype(float)
    is_male = rng.integers(0, 2, n).astype(float)
    married = rng.integers(0, 2, n).astype(float)
    weight = (
        0.35 * gestation
        + 0.01 * mother_age
        - 0.005 * (year - 2000)
        + 0.10 * is_male
        + 0.05 * married
        + rng.normal(0, 0.6, n)
    )
    df = pd.DataFrame({
        "mother_age": mother_age, "gestation_weeks": gestation,
        "year": year, "is_male": is_male, "mother_married": married,
        "weight_pounds": weight,
    })
    return df


def load_data():
    """Load from BigQuery; fall back to synthetic data if unavailable."""
    if BQ_AVAILABLE:
        print("  Loading data from BigQuery (TABLESAMPLE 1% of natality 2000–2008) …")
        df = BQ_CLIENT.query(BQ_QUERY).to_dataframe()
        print(f"  Rows returned: {len(df):,}")
    else:
        import pandas as pd
        print("  [Fallback] Generating synthetic birth-weight data …")
        df = _synthetic_fallback()
        print(f"  Rows: {len(df):,}")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


def make_dataloaders(df, batch_size: int = 128, seed: int = 42):
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train  = scaler_X.fit_transform(X_train)
    X_val    = scaler_X.transform(X_val)
    y_train  = scaler_y.fit_transform(y_train)
    y_val    = scaler_y.transform(y_val)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xt, yt = to_t(X_train), to_t(y_train)
    Xv, yv = to_t(X_val),   to_t(y_val)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=len(Xv))

    return train_loader, val_loader, (Xv, yv), scaler_y


def build_model(in_features: int, device: torch.device) -> nn.Module:
    model = nn.Linear(in_features, 1)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model.to(device)


def elasticnet_loss(pred, target, model, l1, l2):
    mse = nn.functional.mse_loss(pred, target)
    w   = model.weight
    return mse + l1 * w.abs().sum() + l2 * (w ** 2).sum()


def train(model, train_loader, val_loader, cfg, device):
    opt    = optim.Adam(model.parameters(), lr=cfg["lr"])
    l1, l2 = cfg["lambda1"], cfg["lambda2"]
    train_hist, val_hist = [], []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = elasticnet_loss(model(Xb), yb, model, l1, l2)
            loss.backward()
            opt.step()
            total += loss.item() * len(Xb)
        train_hist.append(total / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            for Xv, yv in val_loader:
                vm = nn.functional.mse_loss(model(Xv.to(device)), yv.to(device)).item()
        val_hist.append(vm)

        if epoch % 100 == 0:
            print(f"    Epoch {epoch:>4}/{cfg['epochs']}  train={train_hist[-1]:.4f}  val_mse={vm:.4f}")

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


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta   = get_task_metadata()

    print(f"\n{'='*60}")
    print(f"Task : {meta['id']}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    df = load_data()
    print(f"\n  Dataset shape : {df.shape}")
    print(f"  Target stats  : mean={df[TARGET_COL].mean():.2f}  "
          f"std={df[TARGET_COL].std():.2f}  "
          f"min={df[TARGET_COL].min():.2f}  max={df[TARGET_COL].max():.2f}")

    cfg = {"lr": 5e-3, "lambda1": 1e-4, "lambda2": 1e-4, "epochs": 400, "batch_size": 256}
    train_loader, val_loader, (Xv, yv), scaler_y = make_dataloaders(
        df, batch_size=cfg["batch_size"]
    )
    in_features = Xv.shape[1]
    print(f"\n  Train batches : {len(train_loader)}   Val samples: {len(Xv)}")

    model   = build_model(in_features, device)
    print(f"\nTraining ElasticNet Linear Regression …")
    history = train(model, train_loader, val_loader, cfg, device)

    metrics = evaluate(model, Xv, yv, device)
    weights = model.weight.detach().cpu().numpy().flatten()

    print(f"\n{'─'*60}")
    print(f"  Val MSE  : {metrics['mse']:.4f}")
    print(f"  Val RMSE : {metrics['rmse']:.4f}  (standardised units)")
    print(f"  Val R²   : {metrics['r2']:.4f}")
    print(f"  Weights  : {dict(zip(FEATURE_COLS, np.round(weights, 4)))}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/linreg_bq1_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "config": cfg, "feature_cols": FEATURE_COLS}, f, indent=2)

    print(f"\n--- Assertions ---")

    assert metrics["r2"] > 0.20, (
        f"FAIL: R² = {metrics['r2']:.4f} (expected > 0.20). "
        "Gestation weeks should explain >20% variance in birth weight."
    )
    print(f"[PASS] R² > 0.20: {metrics['r2']:.4f}")

    assert metrics["mse"] < 1.5, (
        f"FAIL: MSE = {metrics['mse']:.4f} (expected < 1.5 in standardised space)"
    )
    print(f"[PASS] MSE < 1.5: {metrics['mse']:.4f}")

    assert history["val_loss_history"][-1] < history["val_loss_history"][0], (
        "FAIL: val loss did not decrease over training"
    )
    print(
        f"[PASS] Val loss decreased: "
        f"{history['val_loss_history'][0]:.4f} → {history['val_loss_history'][-1]:.4f}"
    )

    gestation_idx = FEATURE_COLS.index("gestation_weeks")
    assert weights[gestation_idx] > 0, (
        "FAIL: gestation_weeks weight should be positive (longer gestation → heavier baby)"
    )
    print(f"[PASS] gestation_weeks weight is positive: {weights[gestation_idx]:.4f}")

    print("\n[SUCCESS] All assertions passed. Exiting 0.")
    sys.exit(0)
