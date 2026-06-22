# -*- coding: utf-8 -*-
"""Korak 1: priprema podataka - EDA, ciscenje, enkodiranje, split 70/10/20."""

import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src.config import (RANDOM, FEATURES, TARGET_REG, TARGET_CLF,
                         VALID_DOBA, VALID_PRITISAK, PHYS_RANGES)

D = os.path.join(BASE, "data")
M = os.path.join(BASE, "models")
P = os.path.join(BASE, "plots")
os.makedirs(D, exist_ok=True)
os.makedirs(M, exist_ok=True)
os.makedirs(P, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


def _eda(df):
    print("\n" + "=" * 60)
    print("EDA — Eksplorativna analiza podataka")
    print("=" * 60)

    print(f"\nOblik dataset-a: {df.shape}")
    print(f"\nTipovi kolona:\n{df.dtypes}")
    print(f"\nNedostajuce vrijednosti:\n{df.isnull().sum()[df.isnull().sum()>0]}")
    print(f"\nDuplikati: {df.duplicated().sum()}")
    print(f"\nPeriod: {df['Datum'].min().date()} — {df['Datum'].max().date()}")

    print(f"\nDeskriptivna statistika (numericke):\n{df.describe().round(2)}")

    print(f"\nPreliv_Status distribucija:")
    print(df["Preliv_Status"].value_counts())
    print(f"  Udio pozitivne klase: {df['Preliv_Status'].mean()*100:.2f}%")

    print(f"\nAngazovani_Agregati distribucija:")
    print(df["Angazovani_Agregati"].value_counts().sort_index())

    print(f"\nDoba_Godine distribucija:")
    print(df["Doba_Godine"].value_counts())

    print(f"\nPritisak_Mreze distribucija:")
    print(df["Pritisak_Mreze"].value_counts())

    # 1. Distribucija target varijabli
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    agg_counts = df["Angazovani_Agregati"].value_counts().sort_index()
    colors_agg = ["#3b82c4"] * len(agg_counts)
    ax1.bar(agg_counts.index.astype(str), agg_counts.values, color=colors_agg, edgecolor="white")
    for i, v in enumerate(agg_counts.values):
        ax1.text(i, v + max(agg_counts.values)*0.02, str(v), ha="center", fontsize=10, fontweight="bold")
    ax1.set_title("Distribucija — Angazovani Agregati", fontweight="bold")
    ax1.set_xlabel("Broj agregata"); ax1.set_ylabel("Broj dana")

    preliv_counts = df["Preliv_Status"].value_counts().sort_index()
    colors_preliv = ["#5cb85c", "#d9534f"]
    bars = ax2.bar(["Bez preliva (0)", "Preliv (1)"], preliv_counts.values, color=colors_preliv, edgecolor="white")
    for b, v in zip(bars, preliv_counts.values):
        ax2.text(b.get_x()+b.get_width()/2, v+max(preliv_counts.values)*0.02, str(v), ha="center", fontsize=10, fontweight="bold")
    ax2.set_title("Distribucija — Preliv Status", fontweight="bold")
    ax2.set_ylabel("Broj dana")

    fig.suptitle("Distribucija target varijabli", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(P, "eda_target_distribution.png"))
    plt.close(fig)
    print("Sacuvano: eda_target_distribution.png")

    # 2. Histogrami numerickih atributa
    num_cols = ["Vodostaj_Bileca","Dotok_Prethodni_Dan","Padavine_Trebinje",
                "Temperatura_Vazduha","Padavine_Sutra","Padavine_Za2Dana",
                "Promjena_Vodostaja"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=60, color="#3b82c4", alpha=0.85, edgecolor="white")
        axes[i].axvline(df[col].mean(), ls="--", color="red", alpha=0.7, label=f"Mean={df[col].mean():.1f}")
        axes[i].axvline(df[col].median(), ls=":", color="black", alpha=0.7, label=f"Med={df[col].median():.1f}")
        axes[i].set_title(col.replace("_"," "), fontsize=9)
        axes[i].legend(fontsize=7)
    axes[7].set_visible(False)
    axes[8].set_visible(False)
    fig.suptitle("Distribucija numerickih atributa", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(P, "eda_distributions.png"))
    plt.close(fig)
    print("Sacuvano: eda_distributions.png")

    # 3. Korelaciona matrica
    corr_df = df[num_cols+["Angazovani_Agregati","Preliv_Status"]]
    corr = corr_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                mask=mask, square=True, linewidths=0.5,
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Korelaciona matrica — svi numericki atributi i targeti", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(P, "eda_correlation.png"))
    plt.close(fig)
    print("Sacuvano: eda_correlation.png")

    # 4. Vremenske serije kljucnih atributa (poslednje 3 godine)
    recent = df[df["Datum"] >= df["Datum"].max()-pd.DateOffset(years=3)]
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    axes[0].plot(recent["Datum"], recent["Vodostaj_Bileca"], color="#3b82c4", linewidth=0.7)
    axes[0].set_ylabel("Vodostaj (m)")
    axes[0].set_title("Vodostaj Bileca — poslednje 3 godine", fontweight="bold")

    axes[1].plot(recent["Datum"], recent["Dotok_Prethodni_Dan"], color="#f0ad4e", linewidth=0.7)
    axes[1].set_ylabel("Dotok (m3/s)")
    axes[1].set_title("Dotok — poslednje 3 godine", fontweight="bold")

    axes[2].bar(recent["Datum"], recent["Padavine_Trebinje"], color="#5bc0de", width=1)
    axes[2].set_ylabel("Padavine (mm)")
    axes[2].set_title("Padavine Trebinje — poslednje 3 godine", fontweight="bold")

    preliv_days = recent[recent["Preliv_Status"]==1]
    axes[3].scatter(preliv_days["Datum"], [1]*len(preliv_days),
                    color="#d9534f", s=20, alpha=0.8, marker="|")
    axes[3].set_ylabel("Preliv")
    axes[3].set_title("Dani sa prelivom — poslednje 3 godine", fontweight="bold")
    axes[3].set_ylim(-0.1, 1.5)

    fig.tight_layout()
    fig.savefig(os.path.join(P, "eda_timeseries.png"))
    plt.close(fig)
    print("Sacuvano: eda_timeseries.png")

    # 5. Box plotovi po sezoni
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    sezone_order = ["Proljece","Ljeto","Jesen","Zima"]
    palette = {"Proljece":"#5cb85c","Ljeto":"#d9534f","Jesen":"#f0ad4e","Zima":"#3b82c4"}
    cols_box = ["Vodostaj_Bileca","Dotok_Prethodni_Dan","Padavine_Trebinje",
                "Temperatura_Vazduha","Padavine_Sutra","Padavine_Za2Dana","Promjena_Vodostaja"]
    for i, col in enumerate(cols_box):
        bp = df.boxplot(column=col, by="Doba_Godine", ax=axes[i], patch_artist=True,
                        grid=False, return_type="dict")
        for patch, sez in zip(bp[col]["boxes"], sezone_order):
            patch.set_facecolor(palette[sez])
        axes[i].set_title(col.replace("_"," "), fontsize=9)
        axes[i].set_xlabel("")
    axes[7].set_visible(False)
    fig.suptitle("Box plotovi numerickih atributa po sezoni", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(P, "eda_boxplot_season.png"))
    plt.close(fig)
    print("Sacuvano: eda_boxplot_season.png")

    # 6. Atributi razlozeni po Preliv_Status
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, col in enumerate(cols_box):
        for status, color, label in [(0,"#5cb85c","Bez preliva"),(1,"#d9534f","Preliv")]:
            data = df[df["Preliv_Status"]==status][col]
            axes[i].hist(data, bins=50, alpha=0.6, color=color, label=label, edgecolor="white")
        axes[i].set_title(col.replace("_"," "), fontsize=9)
        axes[i].legend(fontsize=7)
    axes[7].set_visible(False)
    fig.suptitle("Distribucija atributa razlozeno po Preliv_Status", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(P, "eda_by_preliv.png"))
    plt.close(fig)
    print("Sacuvano: eda_by_preliv.png")

    # 7. Agregati po sezoni i pritisku mreze
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    agg_season = df.groupby("Doba_Godine")["Angazovani_Agregati"].mean()
    agg_season = agg_season.reindex(sezone_order)
    ax1.bar(agg_season.index, agg_season.values, color=[palette[s] for s in agg_season.index], edgecolor="white")
    ax1.set_title("Prosjecan broj agregata po sezoni", fontweight="bold")
    ax1.set_ylabel("Prosjecan broj agregata")

    agg_prit = df.groupby("Pritisak_Mreze")["Angazovani_Agregati"].mean()
    prit_order = ["Nisko","Srednje","Visoko"]
    agg_prit = agg_prit.reindex(prit_order)
    ax2.bar(agg_prit.index, agg_prit.values, color=["#5cb85c","#f0ad4e","#d9534f"], edgecolor="white")
    ax2.set_title("Prosjecan broj agregata po pritisku mreze", fontweight="bold")
    ax2.set_ylabel("Prosjecan broj agregata")

    fig.suptitle("Agregati — sezonski i operativni obrasci", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(P, "eda_agregati_patterns.png"))
    plt.close(fig)
    print("Sacuvano: eda_agregati_patterns.png")

    print("\n" + "=" * 60)
    print("EDA zavrsena. Svi grafici u plots/")
    print("=" * 60 + "\n")


def main():
    df = pd.read_csv(os.path.join(D, "hetI.csv"))
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y.", errors="coerce")

    df = df.dropna().drop_duplicates().copy()
    df = df.dropna(subset=["Datum"]).sort_values("Datum").reset_index(drop=True)

    bad = set(df["Doba_Godine"].unique()) - VALID_DOBA
    if bad: df = df[df["Doba_Godine"].isin(VALID_DOBA)]
    bad = set(df["Pritisak_Mreze"].unique()) - VALID_PRITISAK
    if bad: df = df[df["Pritisak_Mreze"].isin(VALID_PRITISAK)]

    for col, (lo, hi) in PHYS_RANGES.items():
        mask = (df[col] < lo) | (df[col] > hi)
        if mask.sum(): df = df[~mask]

    df = df.reset_index(drop=True)
    print(f"Ocisceno redova: {len(df)}")

    _eda(df)

    le_doba = LabelEncoder().fit(sorted(VALID_DOBA))
    le_prit = LabelEncoder().fit(sorted(VALID_PRITISAK))
    df["Doba_Godine_enc"] = le_doba.transform(df["Doba_Godine"])  # type: ignore
    df["Pritisak_Mreze_enc"] = le_prit.transform(df["Pritisak_Mreze"])  # type: ignore

    X = df[FEATURES]
    yreg = df[TARGET_REG]
    yclf = df[TARGET_CLF]

    X_tmp, X_test, yr_tmp, yr_test, yc_tmp, yc_test = train_test_split(
        X, yreg, yclf, test_size=0.2, stratify=yclf, random_state=RANDOM)
    X_train, X_val, yr_train, yr_val, yc_train, yc_val = train_test_split(
        X_tmp, yr_tmp, yc_tmp, test_size=0.125, stratify=yc_tmp, random_state=RANDOM)

    joblib.dump(le_doba, os.path.join(M, "le_doba_godine.joblib"))
    joblib.dump(le_prit, os.path.join(M, "le_pritisak_mreze.joblib"))
    joblib.dump({
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "yreg_train": yr_train, "yreg_val": yr_val, "yreg_test": yr_test,
        "yclf_train": yc_train, "yclf_val": yc_val, "yclf_test": yc_test,
        "feature_cols": FEATURES,
    }, os.path.join(D, "splits.joblib"))
    df.to_csv(os.path.join(D, "hetI_clean.csv"), index=False)

    print(f"train:{len(X_train)} val:{len(X_val)} test:{len(X_test)}")
    for name, y in [("train", yc_train), ("val", yc_val), ("test", yc_test)]:
        print(f"  preliv=1 u {name}: {y.sum()} ({y.mean()*100:.2f}%)")
    print("Sacuvano: splits.joblib, hetI_clean.csv, le_*.joblib")


if __name__ == "__main__":
    main()
