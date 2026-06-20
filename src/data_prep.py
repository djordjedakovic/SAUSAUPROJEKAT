# -*- coding: utf-8 -*-
"""Korak 1: priprema podataka - ciscenje, enkodiranje, split 70/10/20."""

import os, sys
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src.config import (RANDOM, FEATURES, TARGET_REG, TARGET_CLF,
                         VALID_DOBA, VALID_PRITISAK, PHYS_RANGES)

D = os.path.join(BASE, "data")
M = os.path.join(BASE, "models")
os.makedirs(D, exist_ok=True)
os.makedirs(M, exist_ok=True)


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
