# -*- coding: utf-8 -*-
"""Korak 3: Evaluacija svih varijanti klasifikatora."""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score,
    average_precision_score, confusion_matrix,
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


def reg_metrike(y_true, y_pred):
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def clf_metrike(y_true, y_pred, proba=None):
    m = {
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "cm":        confusion_matrix(y_true, y_pred).tolist(),
    }
    if proba is not None:
        m["PR_AUC"] = float(average_precision_score(y_true, proba))
    return m


def _iforest_pred(model: IsolationForest, X, threshold):
    scores = model.decision_function(X)
    pred = (scores <= threshold).astype(int)
    return pred, scores


def main():
    print("=== Korak 3: Evaluacija svih varijanti ===\n")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    splits = joblib.load(os.path.join(DATA_DIR, "splits.joblib"))
    feature_cols = splits["feature_cols"]
    Xv, yrv, ycv = splits["X_val"], splits["yreg_val"], splits["yclf_val"]
    Xt, yrt, yct = splits["X_test"], splits["yreg_test"], splits["yclf_test"]

    with open(os.path.join(MODELS_DIR, "variants.json"), encoding="utf-8") as f:
        variants_data = json.load(f)

    print(f"Ucitano {len(variants_data)} varijanti klasifikatora")

    reg_model = joblib.load(os.path.join(MODELS_DIR, "xgb_regressor.joblib"))
    reg_top5_model = joblib.load(os.path.join(MODELS_DIR, "xgb_regressor_top5.joblib"))
    rf_reg_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_regressor.joblib"))
    dummy_reg = joblib.load(os.path.join(MODELS_DIR, "dummy_regressor.joblib"))
    dummy_clf = joblib.load(os.path.join(MODELS_DIR, "dummy_classifier.joblib"))

    with open(os.path.join(MODELS_DIR, "feature_selection.json"), encoding="utf-8") as f:
        fs_json = json.load(f)
    top5_features = fs_json["zadrzani_atributi"]

    variant_models = {
        "A_raw":          joblib.load(os.path.join(MODELS_DIR, "xgb_classifier_raw.joblib")),
        "B_platt":        joblib.load(os.path.join(MODELS_DIR, "xgb_classifier_platt.joblib")),
        "C_iso":          joblib.load(os.path.join(MODELS_DIR, "xgb_classifier_iso.joblib")),
        "D_undersample":  joblib.load(os.path.join(MODELS_DIR, "xgb_classifier_undersample.joblib")),
        "E_iforest":      joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib")),
        "RF_ref":         joblib.load(os.path.join(MODELS_DIR, "random_forest_classifier.joblib")),
        "LR_ref":         joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib")),
    }

    clf_top5_model = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier_top5.joblib"))
    clf_top5_threshold = fs_json.get("top5_clf_threshold", 0.02)

    rezultati = {"validation": {}, "test": {}, "variants": {}}

    # regresija
    rezultati["validation"]["Baseline_regresor"] = reg_metrike(yrv, dummy_reg.predict(Xv))
    rezultati["validation"]["XGBoost_regresor"]  = reg_metrike(yrv, reg_model.predict(Xv))
    rezultati["validation"]["RandomForest_regresor"] = reg_metrike(yrv, rf_reg_model.predict(Xv))
    rezultati["test"]["Baseline_regresor"] = reg_metrike(yrt, dummy_reg.predict(Xt))
    rezultati["test"]["XGBoost_regresor"]  = reg_metrike(yrt, reg_model.predict(Xt))
    rezultati["test"]["RandomForest_regresor"] = reg_metrike(yrt, rf_reg_model.predict(Xt))

    rezultati["validation"]["XGBoost_regresor_top5"] = reg_metrike(yrv, reg_top5_model.predict(Xv[top5_features]))
    rezultati["test"]["XGBoost_regresor_top5"] = reg_metrike(yrt, reg_top5_model.predict(Xt[top5_features]))

    # klasifikator top5 - evaluacija za selekciju atributa
    proba_clf_top5_val  = clf_top5_model.predict_proba(Xv[top5_features])[:, 1]
    pred_clf_top5_val   = (proba_clf_top5_val >= clf_top5_threshold).astype(int)
    proba_clf_top5_test = clf_top5_model.predict_proba(Xt[top5_features])[:, 1]
    pred_clf_top5_test  = (proba_clf_top5_test >= clf_top5_threshold).astype(int)
    m_top5_val  = clf_metrike(ycv, pred_clf_top5_val, proba_clf_top5_val)
    m_top5_val["threshold"] = float(clf_top5_threshold)
    m_top5_test = clf_metrike(yct, pred_clf_top5_test, proba_clf_top5_test)
    m_top5_test["threshold"] = float(clf_top5_threshold)
    rezultati["validation"]["XGBoost_klasifikator_top5"] = m_top5_val
    rezultati["test"]["XGBoost_klasifikator_top5"] = m_top5_test

    # referentni klasifikator (svi atributi, C_iso) za poređenje
    proba_clf_svi_val  = variant_models["C_iso"].predict_proba(Xv)[:, 1]
    pred_clf_svi_val   = (proba_clf_svi_val >= variants_data["C_iso"]["threshold"]).astype(int)
    proba_clf_svi_test = variant_models["C_iso"].predict_proba(Xt)[:, 1]
    pred_clf_svi_test  = (proba_clf_svi_test >= variants_data["C_iso"]["threshold"]).astype(int)
    m_clf_svi_val  = clf_metrike(ycv, pred_clf_svi_val, proba_clf_svi_val)
    m_clf_svi_test = clf_metrike(yct, pred_clf_svi_test, proba_clf_svi_test)
    rezultati["validation"]["XGBoost_klasifikator_svi"] = m_clf_svi_val
    rezultati["test"]["XGBoost_klasifikator_svi"] = m_clf_svi_test

    # baseline classification
    bl_proba_val = dummy_clf.predict_proba(Xv)[:, 1]
    bl_pred_val  = (bl_proba_val >= 0.5).astype(int)
    bl_proba_test = dummy_clf.predict_proba(Xt)[:, 1]
    bl_pred_test  = (bl_proba_test >= 0.5).astype(int)
    rezultati["validation"]["Baseline_klasifikator"] = clf_metrike(ycv, bl_pred_val, bl_proba_val)
    rezultati["validation"]["Baseline_klasifikator"]["threshold"] = 0.5
    rezultati["test"]["Baseline_klasifikator"] = clf_metrike(yct, bl_pred_test, bl_proba_test)
    rezultati["test"]["Baseline_klasifikator"]["threshold"] = 0.5

    # variantni klasifikatori
    print(f"\n{'Varijanta':<35} {'Set':<6} {'Prag':>8} {'Recall':>8} {'Prec':>8} {'F1':>8} {'PR-AUC':>8}")
    print("-" * 90)

    for code, vd in variants_data.items():
        model = variant_models[code]
        th = vd["threshold"]

        if code == "E_iforest":
            pred_val, scores_val = _iforest_pred(model, Xv, th)
            pred_test, scores_test = _iforest_pred(model, Xt, th)
            proba_val = -scores_val
            proba_test = -scores_test
        else:
            proba_val = model.predict_proba(Xv)[:, 1]
            pred_val  = (proba_val >= th).astype(int)
            proba_test = model.predict_proba(Xt)[:, 1]
            pred_test  = (proba_test >= th).astype(int)

        m_val = clf_metrike(ycv, pred_val, proba_val)
        m_val["threshold"] = float(th)
        m_test = clf_metrike(yct, pred_test, proba_test)
        m_test["threshold"] = float(th)

        rezultati["variants"][code] = {
            "naziv": vd["naziv"],
            "threshold": float(th),
            "validation": m_val,
            "test": m_test,
        }

        print(f"{vd['naziv']:<35} val   {th:>8.4f} {m_val['recall']:>8.3f} {m_val['precision']:>8.3f} {m_val['f1']:>8.3f} {m_val.get('PR_AUC',0):>8.3f}")
        print(f"{'':<35} test  {'':>8} {m_test['recall']:>8.3f} {m_test['precision']:>8.3f} {m_test['f1']:>8.3f} {m_test.get('PR_AUC',0):>8.3f}")

    # poredenje varijanti - grafik
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    nazivi = [vd["naziv"] for vd in variants_data.values()]
    recalls_val = [variants_data[c]["best"]["r"] for c in variants_data]
    precs_val   = [variants_data[c]["best"]["p"] for c in variants_data]
    pragovi      = [variants_data[c]["threshold"] for c in variants_data]

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(nazivi)))
    x = np.arange(len(nazivi))
    w = 0.35

    axes[0].bar(x, recalls_val, w, color="#3b82c4")
    axes[0].set_xticks(x); axes[0].set_xticklabels([n[:12] for n in nazivi], rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Recall (validation, vise = bolje)")
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(1.0, ls="--", color="green", alpha=0.3)

    axes[1].bar(x, precs_val, w, color="#d9534f")
    axes[1].set_xticks(x); axes[1].set_xticklabels([n[:12] for n in nazivi], rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Precision (validation, vise = bolje)")
    axes[1].set_ylim(0, max(max(precs_val) * 1.2, 0.3))

    axes[2].bar(x, pragovi, w, color="#5cb85c")
    axes[2].set_xticks(x); axes[2].set_xticklabels([n[:12] for n in nazivi], rotation=45, ha="right", fontsize=8)
    axes[2].set_title("Prag odluke (validation)")
    axes[2].axhline(0.5, ls="--", color="red", alpha=0.5)
    axes[2].set_ylim(0, max(max(pragovi) * 1.2, 0.6))

    fig.suptitle("Poredjenje varijanti klasifikatora na validation skupu", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "variants_comparison.png"))
    plt.close(fig)
    print("\nSacuvan grafik: variants_comparison.png")

    # confusion matrix za svaku varijantu
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, (code, vd) in enumerate(variants_data.items()):
        if i >= 8: break
        cm_data = rezultati["variants"][code]["test"]["cm"]
        sns.heatmap(np.array(cm_data), annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"], ax=axes[i])
        axes[i].set_title(f"{vd['naziv']}\nPrag={vd['threshold']:.3f}", fontsize=9)
        axes[i].set_xlabel("Pred"); axes[i].set_ylabel("Stvarno")
    for j in range(len(variants_data), 8):
        axes[j].set_visible(False)
    fig.suptitle("Confusion matrice - test skup", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "all_confusion_matrices.png"))
    plt.close(fig)
    print("Sacuvan grafik: all_confusion_matrices.png")

    # distribution of probabilities for each variant
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, (code, vd) in enumerate(variants_data.items()):
        if i >= 8: break
        model = variant_models[code]
        th = vd["threshold"]
        if code == "E_iforest":
            _, scores = _iforest_pred(model, Xt, th)
            prob = -scores
        else:
            prob = model.predict_proba(Xt)[:, 1]

        pos_prob = prob[yct.values == 1]
        neg_prob = prob[yct.values == 0]
        axes[i].hist(neg_prob, bins=50, alpha=0.7, label=f"Neg (n={len(neg_prob)})", color="#3b82c4")
        axes[i].hist(pos_prob, bins=30, alpha=0.9, label=f"Pos (n={len(pos_prob)})", color="#d9534f")
        axes[i].axvline(th, ls="--", color="black", alpha=0.7, label=f"Prag={th:.3f}")
        axes[i].set_title(vd["naziv"], fontsize=9)
        axes[i].legend(fontsize=7)
        axes[i].set_yscale("log")
    for j in range(len(variants_data), 8):
        axes[j].set_visible(False)
    fig.suptitle("Distribucija predikcija po varijantama (test skup, log skala)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "prob_distributions.png"))
    plt.close(fig)
    print("Sacuvan grafik: prob_distributions.png")

    # threshold sweep grafik
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, (code, vd) in enumerate(variants_data.items()):
        if i >= 8: break
        sweep = vd.get("sweep", [])
        if sweep:
            ts = [r["t"] for r in sweep]
            rs = [r["r"] for r in sweep]
            ps = [r["p"] for r in sweep]
            axes[i].plot(ts, rs, "b-", label="Recall", linewidth=2)
            axes[i].plot(ts, ps, "r-", label="Precision", linewidth=2)
            axes[i].axvline(vd["threshold"], ls="--", color="black", alpha=0.5)
            axes[i].axhline(0.05, ls=":", color="gray", alpha=0.3)
            axes[i].set_title(vd["naziv"], fontsize=9)
            axes[i].set_xlabel("Prag"); axes[i].set_ylabel("Vrednost")
            axes[i].legend(fontsize=7)
            axes[i].set_ylim(-0.05, 1.1)
    for j in range(len(variants_data), 8):
        axes[j].set_visible(False)
    fig.suptitle("Sweep pragova po varijantama (validation)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "threshold_sweeps.png"))
    plt.close(fig)
    print("Sacuvan grafik: threshold_sweeps.png")

    # feature importance grafikoni
    importance_data = fs_json["vaznost_atributa"]
    sorted_imp = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
    features_sorted = [f for f, _ in sorted_imp]
    scores_sorted = [s for _, s in sorted_imp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors_bar = ["#3b82c4" if f in top5_features else "#d1d5db" for f in features_sorted]
    ax1.barh(range(len(features_sorted)), scores_sorted, color=colors_bar)
    ax1.set_yticks(range(len(features_sorted)))
    ax1.set_yticklabels(features_sorted)
    ax1.set_xlabel("Vaznost (prosek reg + clf)")
    ax1.set_title("Feature Importance - zajednicki rang")
    ax1.invert_yaxis()
    for i, (f, s) in enumerate(sorted_imp):
        ax1.text(s + 0.005, i, f"{s:.3f}", va="center", fontsize=7)

    reg_imp = dict(zip(feature_cols, reg_model.feature_importances_))
    colors_reg = ["#3b82c4" if f in top5_features else "#d1d5db" for f in features_sorted]
    ax2.barh(range(len(features_sorted)), [reg_imp[f] for f in features_sorted], color=colors_reg)
    ax2.set_yticks(range(len(features_sorted)))
    ax2.set_yticklabels(features_sorted)
    ax2.set_xlabel("Vaznost")
    ax2.set_title("Feature Importance - Regresija (XGBoost)")
    ax2.invert_yaxis()

    fig.suptitle("Vizualna analiza najuticajnijih atributa", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_selection.png"))
    plt.close(fig)
    print("Sacuvan grafik: feature_selection.png")

    isoct = variant_models["C_iso"]
    clf_imp = np.mean([c.estimator.feature_importances_
                       for c in isoct.calibrated_classifiers_], axis=0)
    clf_imp_dict = dict(zip(feature_cols, clf_imp))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors_clf = ["#d9534f" if f in top5_features else "#d1d5db" for f in features_sorted]
    ax.barh(range(len(features_sorted)), [clf_imp_dict[f] for f in features_sorted], color=colors_clf)
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted)
    ax.set_xlabel("Vaznost")
    ax.set_title("Feature Importance - Klasifikacija (XGBoost + isotonic)")
    ax.invert_yaxis()
    for i, f in enumerate(features_sorted):
        ax.text(clf_imp_dict[f] + 0.003, i, f"{clf_imp_dict[f]:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_selection_clf.png"))
    plt.close(fig)
    print("Sacuvan grafik: feature_selection_clf.png")

    # saving results
    rezultati["feature_selection"] = fs_json
    with open(os.path.join(BASE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rezultati, f, indent=2, ensure_ascii=False)

    najbolji = sorted(rezultati["variants"].items(),
                       key=lambda kv: (kv[1]["test"]["recall"],
                                       -kv[1]["test"]["cm"][0][1],
                                       kv[1]["test"]["f1"]),
                       reverse=True)

    tumacenje = [
        "TUMACENJE REZULTATA - HE TREBINJE I (vise varijanti)",
        "=" * 55,
        "",
        "1. REGRESIJA (broj agregata 0-3)",
        f"   XGBoost (svi atributi)    MAE={rezultati['test']['XGBoost_regresor']['MAE']:.3f}  "
        f"R2={rezultati['test']['XGBoost_regresor']['R2']:.3f}",
        f"   XGBoost (top 5 atributa)  MAE={rezultati['test']['XGBoost_regresor_top5']['MAE']:.3f}  "
        f"R2={rezultati['test']['XGBoost_regresor_top5']['R2']:.3f}",
        f"   RF (svi atributi)         MAE={rezultati['test']['RandomForest_regresor']['MAE']:.3f}  "
        f"R2={rezultati['test']['RandomForest_regresor']['R2']:.3f}",
        f"   Baseline                  MAE={rezultati['test']['Baseline_regresor']['MAE']:.3f}",
        "",
        "2. SELEKCIJA ATRIBUTA",
        f"   Zadrzani (top {len(top5_features)}):    {', '.join(top5_features)}",
        f"   Odbaceni ({len(fs_json['odbaceni_atributi'])}): {', '.join(fs_json['odbaceni_atributi'])}",
        "",
        "   Poredjenje modela sa svim vs top atributima:",
        f"   Svi atributi: MAE={rezultati['test']['XGBoost_regresor']['MAE']:.3f}, R2={rezultati['test']['XGBoost_regresor']['R2']:.3f}",
        f"   Top atributi: MAE={rezultati['test']['XGBoost_regresor_top5']['MAE']:.3f}, R2={rezultati['test']['XGBoost_regresor_top5']['R2']:.3f}",
        f"   Razlika MAE: {(rezultati['test']['XGBoost_regresor_top5']['MAE'] - rezultati['test']['XGBoost_regresor']['MAE']):.4f}",
        f"   Razlika R2:  {(rezultati['test']['XGBoost_regresor_top5']['R2'] - rezultati['test']['XGBoost_regresor']['R2']):.4f}",
        "",
        "   Poredjenje KLASIFIKATORA sa svim vs top atributima (XGBoost + isotonic):",
        f"   Svi atributi: Recall={rezultati['test']['XGBoost_klasifikator_svi']['recall']:.3f}, Precision={rezultati['test']['XGBoost_klasifikator_svi']['precision']:.3f}, PR-AUC={rezultati['test']['XGBoost_klasifikator_svi'].get('PR_AUC',0):.3f}",
        f"   Top atributi: Recall={rezultati['test']['XGBoost_klasifikator_top5']['recall']:.3f}, Precision={rezultati['test']['XGBoost_klasifikator_top5']['precision']:.3f}, PR-AUC={rezultati['test']['XGBoost_klasifikator_top5'].get('PR_AUC',0):.3f}",
        f"   Razlika Recall:   {(rezultati['test']['XGBoost_klasifikator_top5']['recall'] - rezultati['test']['XGBoost_klasifikator_svi']['recall']):.3f}",
        f"   Razlika PR-AUC:   {(rezultati['test']['XGBoost_klasifikator_top5'].get('PR_AUC',0) - rezultati['test']['XGBoost_klasifikator_svi'].get('PR_AUC',0)):.3f}",
        "",
        "3. KLASIFIKACIJA - POREDJENJE VARIJANTI (test skup)",
        "",
    ]

    for rank, (code, data) in enumerate(najbolji, 1):
        t = data["test"]
        cm = np.array(t["cm"])
        tn, fp, fn, tp = cm.ravel()
        tumacenje.append(
            f"   #{rank} {data['naziv']} (prag={t['threshold']:.3f})"
        )
        tumacenje.append(
            f"      Recall={t['recall']:.3f}  Precision={t['precision']:.3f}  "
            f"F1={t['f1']:.3f}  PR-AUC={t.get('PR_AUC',0):.3f}"
        )
        tumacenje.append(
            f"      TN={tn}  FP={fp}  FN={fn}  TP={tp}  "
            f"({tp}/{tp+fn} preliva detektovano)"
        )

    tumacenje += [
        "",
        "4. ZAKLJUCAK",
        "   Cilj je bio istraziti da li se prag odluke moze pribliziti vrednosti 0.5",
        "   primenom razlicitih tehnika kalibracije i balansiranja podataka.",
        "",
        "   Varijante koje daju najvise pragove (blize 0.5) uz ocuvanje visokog",
        "   recall-a su najbolji kandidati za zamenu originalnog pristupa.",
        "",
        "   Najuticajniji atributi (feature importance):",
    ]
    for rank, (f, s) in enumerate(sorted_imp, 1):
        tumacenje.append(f"      {rank}. {f} = {s:.4f}")
    tumacenje += [
        "",
        "5. METODOLOGIJA",
        "   - Split 70/10/20, stratifikovan po Preliv_Status",
        "   - Varijanta A: XGBoost BEZ kalibracije (sirovi predict_proba)",
        "   - Varijanta B: XGBoost + Platt (sigmoid) kalibracija",
        "   - Varijanta C: XGBoost + isotonic kalibracija (originalni pristup)",
        "   - Varijanta D: RandomUnderSampler(0.3) + XGBoost bez kalibracije",
        "   - Varijanta E: Isolation Forest (contamination=0.02)",
        "   - RF_ref: RandomForest + SMOTE + isotonic",
        "   - LR_ref: LogisticRegression + SMOTE + isotonic",
        "",
        "   random_state=42 svuda za reproducibilnost.",
        "   Validation skup za sve odluke; test skup evaluiran samo jednom.",
    ]

    with open(os.path.join(BASE_DIR, "tumacenje.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(tumacenje))

    print("\n" + "=" * 55)
    print("RANG LISTA VARIJANTI (test skup):")
    for rank, (code, data) in enumerate(najbolji, 1):
        t = data["test"]
        cm = np.array(t["cm"])
        tn, fp, fn, tp = cm.ravel()
        print(f"  #{rank} {data['naziv']:<30} "
              f"Prag={t['threshold']:.3f}  "
              f"Recall={t['recall']:.3f}  "
              f"Prec={t['precision']:.3f}  "
              f"F1={t['f1']:.3f}  "
              f"TP={tp}/{tp+fn}")

    print("\nSacuvano: metrics.json, tumacenje.txt, plots/*.png")


if __name__ == "__main__":
    main()
