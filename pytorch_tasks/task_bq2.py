"""
BigQuery Task 2 — Logistic Regression: Maternal Cigarette Use Prediction
=========================================================================
Dataset:   bigquery-public-data.samples.natality
Target:    cigarette_use  (binary: True/False)
Features:  mother_age, gestation_weeks, year, is_male, mother_married,
           weight_pounds  (6 predictors)

New beyond the sample notebook
-------------------------------
The BigQuery-intro notebook only displays cigarette_use value counts as a
bar chart. This task goes further:
  • Uses a BigQuery WITH clause (CTE) to compute a class-balanced sample
    in SQL (50/50 smoker vs non-smoker) before streaming data to Python
  • Trains a binary logistic regression in PyTorch
  • Benchmarks against sklearn LogisticRegression
  • Reports F1, precision, recall, AUC-PR, and confusion matrix

BigQuery features demonstrated
-------------------------------
  • WITH … AS (CTE) for two-step balanced sub-sampling
  • TABLESAMPLE SYSTEM per stratum (efficient, no full-scan ORDER BY RAND)
  • UNION ALL to combine the two strata

Algorithm
---------
σ(z)  = 1 / (1 + e^{−z})
J(θ) = BCE(σ(Xθ), y) + λ · ‖θ‖₂²   (Ridge / L2 logistic regression)
Optimizer: Adam
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score,
)
from sklearn.linear_model import LogisticRegression as SklearnLR

try:
    from google.cloud import bigquery

    PROJECT_ID   = os.environ.get("GCP_PROJECT", "cmpelkk")
    BQ_CLIENT    = bigquery.Client(project=PROJECT_ID)
    BQ_AVAILABLE = True
except Exception as e:
    print(f"[WARN] BigQuery unavailable ({e}). Falling back to synthetic data.")
    BQ_AVAILABLE = False


def get_task_metadata() -> dict:
    return {
        "series": "Logistic Regression",
        "level": "bq_2",
        "id": "logreg_bq2_natality_cigarette",
        "algorithm": "Logistic Regression (L2 / Ridge) — BigQuery natality dataset",
        "description": (
            "Binary classification: predict whether a mother smoked during pregnancy "
            "from birth record features. Data loaded from BigQuery with a CTE-based "
            "balanced sample (equal smoker/non-smoker rows via UNION ALL). "
            "Extends the notebook's EDA-only approach to a full ML pipeline."
        ),
        "interface_protocol": "pytorch_task_v1",
        "bigquery_features": [
            "WITH … AS (CTE) for readable multi-step SQL",
            "UNION ALL of two TABLESAMPLE strata for class-balanced sampling",
            "Server-side IS NOT NULL filtering; CAST to FLOAT64",
        ],
        "requirements": {
            "data": "bigquery-public-data.samples.natality, 2000-2008, balanced smoker/non-smoker sample.",
            "implementation": "nn.Linear(6,1) + BCE + L2 penalty; Adam; compare with sklearn LR.",
            "evaluation": "Accuracy, F1, precision, recall, average precision (AUC-PR), confusion matrix.",
            "validation": "F1 > 0.60; accuracy > 0.60; within 5 pp of sklearn baseline.",
        },
    }


BQ_QUERY = """
WITH smokers AS (
  SELECT
      CAST(mother_age      AS FLOAT64) AS mother_age,
      CAST(gestation_weeks AS FLOAT64) AS gestation_weeks,
      CAST(year            AS FLOAT64) AS year,
      CAST(is_male         AS FLOAT64) AS is_male,
      CAST(mother_married  AS FLOAT64) AS mother_married,
      CAST(weight_pounds   AS FLOAT64) AS weight_pounds,
      1.0                              AS cigarette_use
  FROM `bigquery-public-data.samples.natality` TABLESAMPLE SYSTEM (2 PERCENT)
  WHERE cigarette_use = TRUE
    AND year BETWEEN 2000 AND 2008
    AND weight_pounds   IS NOT NULL
    AND gestation_weeks IS NOT NULL
    AND mother_married  IS NOT NULL
    AND is_male         IS NOT NULL
  LIMIT 5000
),
non_smokers AS (
  SELECT
      CAST(mother_age      AS FLOAT64) AS mother_age,
      CAST(gestation_weeks AS FLOAT64) AS gestation_weeks,
      CAST(year            AS FLOAT64) AS year,
      CAST(is_male         AS FLOAT64) AS is_male,
      CAST(mother_married  AS FLOAT64) AS mother_married,
      CAST(weight_pounds   AS FLOAT64) AS weight_pounds,
      0.0                              AS cigarette_use
  FROM `bigquery-public-data.samples.natality` TABLESAMPLE SYSTEM (2 PERCENT)
  WHERE cigarette_use = FALSE
    AND year BETWEEN 2000 AND 2008
    AND weight_pounds   IS NOT NULL
    AND gestation_weeks IS NOT NULL
    AND mother_married  IS NOT NULL
    AND is_male         IS NOT NULL
  LIMIT 5000
)
SELECT * FROM smokers
UNION ALL
SELECT * FROM non_smokers
"""

FEATURE_COLS = [
    "mother_age", "gestation_weeks", "year",
    "is_male", "mother_married", "weight_pounds",
]
TARGET_COL = "cigarette_use"


def _synthetic_fallback(n_per_class: int = 2000, seed: int = 42):
    """Plausible synthetic data when BigQuery is unavailable."""
    import pandas as pd
    rng = np.random.default_rng(seed)

    rows = []
    for label in [0, 1]:
        age    = rng.normal(25 + 2 * (1 - label), 5, n_per_class).clip(15, 45)
        gest   = rng.normal(39 - 0.5 * label, 1.5, n_per_class).clip(28, 45)
        year   = rng.integers(2000, 2009, n_per_class).astype(float)
        male   = rng.integers(0, 2, n_per_class).astype(float)
        mar    = rng.binomial(1, 0.8 - 0.3 * label, n_per_class).astype(float)
        weight = rng.normal(7.2 - 0.3 * label, 1.0, n_per_class).clip(2, 12)
        df_part = pd.DataFrame({
            "mother_age": age, "gestation_weeks": gest, "year": year,
            "is_male": male, "mother_married": mar, "weight_pounds": weight,
            "cigarette_use": float(label),
        })
        rows.append(df_part)
    return pd.concat(rows, ignore_index=True)


def load_data():
    if BQ_AVAILABLE:
        print("  Loading balanced sample from BigQuery (CTE + TABLESAMPLE + UNION ALL) …")
        df = BQ_CLIENT.query(BQ_QUERY).to_dataframe()
        print(f"  Rows returned: {len(df):,}  |  class balance: {df[TARGET_COL].value_counts().to_dict()}")
    else:
        print("  [Fallback] Generating synthetic maternal smoking data …")
        df = _synthetic_fallback()
        print(f"  Rows: {len(df):,}")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


def make_dataloaders(df, batch_size: int = 64, seed: int = 42):
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_tr)
    X_val   = scaler.transform(X_val)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(to_t(X_tr), to_t(y_tr)), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(to_t(X_val), to_t(y_val)), batch_size=len(X_val)
    )
    return train_loader, val_loader, (to_t(X_val), to_t(y_val)), X_tr, y_tr.ravel(), X_val, y_val.ravel()


def build_model(in_features: int, device: torch.device) -> nn.Module:
    model = nn.Linear(in_features, 1)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model.to(device)


def ridge_bce_loss(logits, target, model, lam):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    l2  = (model.weight ** 2).sum()
    return bce + lam * l2


def train(model, train_loader, val_loader, cfg, device):
    opt   = optim.Adam(model.parameters(), lr=cfg["lr"])
    lam   = cfg["lambda"]
    train_hist, val_hist = [], []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ridge_bce_loss(model(Xb), yb, model, lam)
            loss.backward()
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

        if epoch % 100 == 0:
            print(f"    Epoch {epoch:>4}/{cfg['epochs']}  train_loss={train_hist[-1]:.4f}  val_bce={vl:.4f}")

    return {"train_loss_history": train_hist, "val_loss_history": val_hist}


def evaluate(model, Xv, yv, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        Xv_d  = Xv.to(device)
        logits = model(Xv_d)
        probs  = torch.sigmoid(logits).cpu().numpy().ravel()
        preds  = (probs >= threshold).astype(int)

    y_np = yv.numpy().ravel().astype(int)
    acc  = float((preds == y_np).mean())
    f1   = float(f1_score(y_np, preds, zero_division=0))
    pr   = float(precision_score(y_np, preds, zero_division=0))
    rc   = float(recall_score(y_np, preds, zero_division=0))
    ap   = float(average_precision_score(y_np, probs))
    cm   = confusion_matrix(y_np, preds).tolist()

    return {"accuracy": acc, "f1": f1, "precision": pr, "recall": rc,
            "avg_precision_auc": ap, "confusion_matrix": cm}


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
    print(f"\n  Dataset shape       : {df.shape}")
    pos = int(df[TARGET_COL].sum())
    print(f"  Class balance       : {pos} smokers / {len(df)-pos} non-smokers")

    cfg = {"lr": 1e-3, "lambda": 1e-3, "epochs": 400, "batch_size": 64}
    train_loader, val_loader, (Xv_t, yv_t), X_tr_np, y_tr_np, X_val_np, y_val_np = (
        make_dataloaders(df, batch_size=cfg["batch_size"])
    )
    in_features = Xv_t.shape[1]
    print(f"  Train batches : {len(train_loader)}   Val samples: {len(Xv_t)}")

    model   = build_model(in_features, device)
    print(f"\nTraining L2 Logistic Regression …")
    history = train(model, train_loader, val_loader, cfg, device)
    metrics_pt = evaluate(model, Xv_t, yv_t, device)

    sk_clf   = SklearnLR(penalty="l2", max_iter=1000, random_state=42)
    sk_clf.fit(X_tr_np, y_tr_np)
    sk_preds = sk_clf.predict(X_val_np)
    sk_probs = sk_clf.predict_proba(X_val_np)[:, 1]
    metrics_sk = {
        "accuracy" : float((sk_preds == y_val_np).mean()),
        "f1"       : float(f1_score(y_val_np, sk_preds, zero_division=0)),
        "avg_precision_auc": float(average_precision_score(y_val_np, sk_probs)),
    }

    print(f"\n{'─'*60}")
    print("PyTorch L2 Logistic Regression:")
    for k, v in metrics_pt.items():
        if k != "confusion_matrix":
            print(f"  {k:25s}: {v:.4f}")
    print(f"  confusion_matrix      : {metrics_pt['confusion_matrix']}")
    print(f"\nsklearn baseline:")
    for k, v in metrics_sk.items():
        print(f"  {k:25s}: {v:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/logreg_bq2_metrics.json", "w") as f:
        json.dump({"pytorch": metrics_pt, "sklearn": metrics_sk, "config": cfg}, f, indent=2)

    print(f"\n--- Assertions ---")

    acc_pt = metrics_pt["accuracy"]
    assert acc_pt > 0.60, f"FAIL: accuracy = {acc_pt:.4f} (expected > 0.60)"
    print(f"[PASS] Accuracy > 0.60: {acc_pt:.4f}")

    f1_pt = metrics_pt["f1"]
    assert f1_pt > 0.60, f"FAIL: F1 = {f1_pt:.4f} (expected > 0.60)"
    print(f"[PASS] F1 > 0.60: {f1_pt:.4f}")

    acc_gap = abs(acc_pt - metrics_sk["accuracy"])
    assert acc_gap <= 0.05, (
        f"FAIL: accuracy gap vs sklearn = {acc_gap:.4f} (expected ≤ 0.05)"
    )
    print(f"[PASS] Within 5 pp of sklearn: PyTorch={acc_pt:.4f}  sklearn={metrics_sk['accuracy']:.4f}")

    assert history["val_loss_history"][-1] < history["val_loss_history"][0], (
        "FAIL: val BCE did not decrease"
    )
    print(f"[PASS] Val BCE decreased: {history['val_loss_history'][0]:.4f} → {history['val_loss_history'][-1]:.4f}")

    print("\n[SUCCESS] All assertions passed. Exiting 0.")
    sys.exit(0)
