# -*- coding: utf-8 -*-
"""Korak 2: treniranje svih modela sa vise varijanti klasifikatora.

Varijante:
  A - XGBoost BEZ kalibracije (sirove predict_proba vrednosti)
  B - XGBoost + Platt (sigmoid) kalibracija
  C - XGBoost + isotonic kalibracija (originalni pristup)
  D - Undersampling negativne klase + XGBoost bez kalibracije
  E - Isolation Forest (anomaly detection, treniran samo na negativnoj klasi)
"""

import os, sys, json
import numpy as np
import joblib
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBRegressor, XGBClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src.config import (RANDOM, FEATURES, GRID,
                         TOP_N, MIN_PRECISION, TH_LO, TH_HI, TH_STEP)

M = os.path.join(BASE, "models")


def _find_best_threshold(y_score, y_true, is_anomaly=False):
    ts = (np.linspace(np.min(y_score), np.max(y_score), 100) if is_anomaly
          else np.round(np.arange(TH_LO, TH_HI + 1e-9, TH_STEP), 4))
    rows = []
    for t in ts:
        pred = (y_score <= t).astype(int) if is_anomaly else (y_score >= t).astype(int)
        rows.append(dict(t=float(t),
                         p=float(precision_score(y_true, pred, zero_division=0)),
                         r=float(recall_score(y_true, pred, zero_division=0))))
    valid = [r for r in rows if r["p"] >= MIN_PRECISION]
    best = max(valid or rows, key=lambda r: (r["r"], r["p"]))
    return best["t"], rows, best


def _class_weight(y):
    n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
    return n_neg / max(n_pos, 1)


def _train_eval(model, X_tr, y_tr, X_val, y_val, is_anomaly=False):
    model.fit(X_tr, y_tr)
    scores = (model.decision_function(X_val) if is_anomaly
              else model.predict_proba(X_val)[:, 1])
    th, sweep, best = _find_best_threshold(scores, y_val, is_anomaly)
    print(f"  Prag={th:.4f}  Recall={best['r']:.3f}  Precision={best['p']:.3f}")
    return model, th, sweep, best


def _save_json(data, filename):
    with open(os.path.join(M, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    s = joblib.load(os.path.join(BASE, "data", "splits.joblib"))
    X, Xv = s["X_train"], s["X_val"]
    yr = s["yreg_train"]
    yc, ycv = s["yclf_train"], s["yclf_val"]

    print("=" * 60)
    print("TRENIRANJE - HE TREBINJE I (vise varijanti)")
    print("=" * 60)

    dummy_reg = DummyRegressor(strategy="mean").fit(X, yr)
    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM).fit(X, yc)

    gs_reg = GridSearchCV(XGBRegressor(random_state=RANDOM, n_jobs=-2), GRID, cv=3,
                           scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    gs_reg.fit(X, yr)
    xgb_reg = gs_reg.best_estimator_
    print(f"XGB Reg: {gs_reg.best_params_} MAE_cv={-gs_reg.best_score_:.4f}")

    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=4,
        random_state=RANDOM, n_jobs=-1).fit(X, yr)

    cw = _class_weight(yc)
    xgb_kw = dict(n_estimators=200, max_depth=8, learning_rate=0.1,
                  scale_pos_weight=cw, random_state=RANDOM, n_jobs=-2,
                  eval_metric="logloss")

    variants = {}

    print("\n--- Varijanta A: XGBoost BEZ kalibracije ---")
    xgbA, tA, swA, bestA = _train_eval(XGBClassifier(**xgb_kw), X, yc, Xv, ycv)
    variants["A_raw"] = dict(naziv="XGBoost BEZ kalibracije",
                              threshold=tA, best=bestA, sweep=swA)

    print("\n--- Varijanta B: XGBoost + Platt (sigmoid) kalibracija ---")
    xgbB, tB, swB, bestB = _train_eval(
        CalibratedClassifierCV(XGBClassifier(**xgb_kw), method="sigmoid", cv=3),
        X, yc, Xv, ycv)
    variants["B_platt"] = dict(naziv="XGBoost + Platt (sigmoid)",
                                threshold=tB, best=bestB, sweep=swB)

    print("\n--- Varijanta C: XGBoost + isotonic kalibracija (original) ---")
    xgbC, tC, swC, bestC = _train_eval(
        CalibratedClassifierCV(XGBClassifier(**xgb_kw), method="isotonic", cv=3),
        X, yc, Xv, ycv)
    variants["C_iso"] = dict(naziv="XGBoost + isotonic (original)",
                              threshold=tC, best=bestC, sweep=swC)

    print("\n--- Varijanta D: Undersampling + XGBoost bez kalibracije ---")
    X_under, yc_under = RandomUnderSampler(sampling_strategy=0.3, random_state=RANDOM).fit_resample(X, yc)  # type: ignore
    xgbD, tD, swD, bestD = _train_eval(XGBClassifier(**xgb_kw), X_under, yc_under, Xv, ycv)
    variants["D_undersample"] = dict(naziv="Undersampling + XGBoost",
                                      threshold=tD, best=bestD, sweep=swD)

    print("\n--- Varijanta E: Isolation Forest ---")
    ifo, tE, swE, bestE = _train_eval(
        IsolationForest(n_estimators=200, contamination=0.02,
                        random_state=RANDOM, n_jobs=-1),
        X, yc, Xv, ycv, is_anomaly=True)
    variants["E_iforest"] = dict(naziv="Isolation Forest",
                                  threshold=tE, best=bestE, sweep=swE)

    print("\n--- Reference: RandomForest + SMOTE + isotonic ---")
    rf, tRF, swRF, bestRF = _train_eval(
        CalibratedClassifierCV(ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM)),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=12,
                                          min_samples_leaf=4, class_weight="balanced",
                                          random_state=RANDOM, n_jobs=-1)),
        ]), method="isotonic", cv=3), X, yc, Xv, ycv)
    variants["RF_ref"] = dict(naziv="RandomForest + SMOTE (ref)",
                               threshold=tRF, best=bestRF, sweep=swRF)

    print("\n--- Reference: LogisticRegression + SMOTE + isotonic ---")
    lr, tLR, swLR, bestLR = _train_eval(
        CalibratedClassifierCV(ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM)),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=2000,
                                      random_state=RANDOM)),
        ]), method="isotonic", cv=3), X, yc, Xv, ycv)
    variants["LR_ref"] = dict(naziv="LogisticRegression + SMOTE (ref)",
                               threshold=tLR, best=bestLR, sweep=swLR)

    imp = (xgb_reg.feature_importances_ +
           np.mean([c.estimator.feature_importances_
                     for c in xgbC.calibrated_classifiers_], axis=0)) / 2
    ranked = sorted(zip(FEATURES, imp), key=lambda x: -float(x[1]))
    topf = [f for f, _ in ranked[:TOP_N]]
    remf = [f for f, _ in ranked[TOP_N:]]
    print(f"\nTop {len(topf)}: {topf}\nOdbaceno: {remf}")

    xgb_reg_top = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                random_state=RANDOM, n_jobs=-2).fit(X[topf], yr)

    print("\n--- Top5 klasifikator: XGBoost + isotonic (top 5 atributa) ---")
    xgb_clf_top5, t_top5, sw_top5, best_top5 = _train_eval(
        CalibratedClassifierCV(XGBClassifier(**xgb_kw), method="isotonic", cv=3),
        X[topf], yc, Xv[topf], ycv)

    for name, mod in [
        ("dummy_regressor", dummy_reg),
        ("dummy_classifier", dummy_clf),
        ("xgb_regressor", xgb_reg),
        ("random_forest_regressor", rf_reg),
        ("xgb_regressor_top5", xgb_reg_top),
        ("xgb_classifier_raw", xgbA),
        ("xgb_classifier_platt", xgbB),
        ("xgb_classifier_iso", xgbC),
        ("xgb_classifier_undersample", xgbD),
        ("isolation_forest", ifo),
        ("random_forest_classifier", rf),
        ("logistic_regression", lr),
        ("xgb_classifier_top5", xgb_clf_top5),
    ]:
        joblib.dump(mod, os.path.join(M, f"{name}.joblib"))

    print("\n=== REZULTATI (Validation) ===")
    print(f"{'Varijanta':<35} {'Prag':>8} {'Recall':>8} {'Prec':>8}")
    print("-" * 60)
    for vd in variants.values():
        b = vd["best"]
        print(f"{vd['naziv']:<35} {vd['threshold']:>8.4f} {b['r']:>8.3f} {b['p']:>8.3f}")

    _save_json(variants, "variants.json")
    _save_json(dict(
        threshold=tC,
        thresholds_all_models=dict(raw=tA, platt=tB, iso=tC,
                                    undersample=tD, iforest=tE, rf=tRF, logreg=tLR),
        criterion=f"max recall; precision>={MIN_PRECISION*100:.0f}%",
    ), "threshold.json")
    _save_json(dict(
        svi_atributi=FEATURES,
        zadrzani_atributi=topf,
        odbaceni_atributi=remf,
        vaznost_atributa={f: float(v) for f, v in ranked},
        top5_clf_threshold=t_top5,
    ), "feature_selection.json")
    _save_json(dict(
        xgb_regressor=dict(best_params=gs_reg.best_params_,
                            best_mae_cv=float(gs_reg.best_score_)),
    ), "grid_search.json")

    print("\nSacuvani svi modeli u models/")


if __name__ == "__main__":
    main()
