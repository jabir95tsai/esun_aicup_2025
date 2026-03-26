"""LightGBM training and inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def train_and_predict(
    df_final_pd: pd.DataFrame,
    df_predict_pd: pd.DataFrame,
    feature_cols: Iterable[str],
    output_path: str | Path,
    model_cfg: Dict,
) -> pd.DataFrame:
    """Train LightGBM with CV and write submission.csv."""
    feature_cols = list(feature_cols)
    df_final_pd = df_final_pd.sort_values(by="acct").reset_index(drop=True)
    df_predict_pd = df_predict_pd.sort_values(by="acct").reset_index(drop=True)

    x_train = df_final_pd[feature_cols].copy()
    y_train = df_final_pd["label"].astype(int).copy()
    x_test = df_predict_pd[feature_cols].copy()

    for data in (x_train, x_test):
        for col in data.columns:
            if data[col].dtype == object or str(data[col].dtype) == "category":
                data[col] = pd.to_numeric(data[col], errors="coerce")
        data.fillna(0, inplace=True)

    cv_cfg = model_cfg["cv"]
    skf = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"],
    )
    th_grid = np.linspace(cv_cfg["threshold_min"], cv_cfg["threshold_max"], cv_cfg["threshold_steps"])

    best_iters = []
    oof_pred = np.zeros(len(x_train), dtype=float)

    for tr_idx, val_idx in skf.split(x_train, y_train):
        x_tr, x_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        alpha = model_cfg["pu_alpha"]
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        scale_pos_weight = alpha * (neg / max(pos, 1))

        clf = LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            n_estimators=model_cfg["n_estimators"],
            learning_rate=model_cfg["learning_rate"],
            num_leaves=model_cfg["num_leaves"],
            subsample=model_cfg["subsample"],
            colsample_bytree=model_cfg["colsample_bytree"],
            reg_alpha=model_cfg["reg_alpha"],
            reg_lambda=model_cfg["reg_lambda"],
            random_state=model_cfg["random_state"],
            n_jobs=model_cfg["n_jobs"],
            scale_pos_weight=scale_pos_weight,
            verbose=-1,
        )
        clf.fit(
            x_tr,
            y_tr,
            eval_set=[(x_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(stopping_rounds=model_cfg["early_stopping_rounds"], verbose=False)],
        )

        best_iters.append(clf.best_iteration_)
        val_prob = clf.predict_proba(x_val, num_iteration=clf.best_iteration_)[:, 1]
        oof_pred[val_idx] = val_prob

    _ = [f1_score(y_train, (oof_pred > th).astype(int)) for th in th_grid]

    final_n_estimators = max(model_cfg["min_final_estimators"], int(np.mean(best_iters)))
    pos_all = (y_train == 1).sum()
    neg_all = (y_train == 0).sum()
    scale_pos_weight_all = model_cfg["pu_alpha"] * (neg_all / max(pos_all, 1))

    final_clf = LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=final_n_estimators,
        learning_rate=model_cfg["learning_rate"],
        num_leaves=model_cfg["num_leaves"],
        subsample=model_cfg["subsample"],
        colsample_bytree=model_cfg["colsample_bytree"],
        reg_alpha=model_cfg["reg_alpha"],
        reg_lambda=model_cfg["reg_lambda"],
        random_state=model_cfg["random_state"],
        n_jobs=model_cfg["n_jobs"],
        scale_pos_weight=scale_pos_weight_all,
        verbose=-1,
    )
    final_clf.fit(x_train, y_train)
    test_prob = final_clf.predict_proba(x_test)[:, 1]

    topk_pct = model_cfg["topk_pct"]
    k = max(1, int(len(test_prob) * topk_pct))
    topk_threshold = np.partition(test_prob, -k)[-k]
    final_pred = (test_prob >= topk_threshold).astype(int)

    df_submission = pd.DataFrame({"acct": df_predict_pd["acct"], "label": final_pred}).sort_values(by="acct")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_submission.to_csv(output_dir / "submission.csv", index=False)
    return df_submission
