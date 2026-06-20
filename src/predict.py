# -*- coding: utf-8 -*-
"""Prognoza za 7 dana - HE Trebinje I."""

import os, sys, json
import numpy as np
import pandas as pd
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src.config import FEATURES, RAW_KEYS, PHYS_RANGES, VALID_DOBA, VALID_PRITISAK

MODELS = os.path.join(BASE, "models")
_cache = {}


def _load():
    if _cache: return _cache
    _cache["reg"] = joblib.load(os.path.join(MODELS, "xgb_regressor.joblib"))
    _cache["clf"] = joblib.load(os.path.join(MODELS, "xgb_classifier_iso.joblib"))
    _cache["le_doba"] = joblib.load(os.path.join(MODELS, "le_doba_godine.joblib"))
    _cache["le_prit"] = joblib.load(os.path.join(MODELS, "le_pritisak_mreze.joblib"))
    with open(os.path.join(MODELS, "threshold.json"), encoding="utf-8") as f:
        _cache["threshold"] = json.load(f)["threshold"]
    return _cache


def forecast_7days(days):
    if len(days) != 7:
        raise ValueError(f"Treba 7 dana, dobijeno {len(days)}")

    for i, day in enumerate(days):
        missing = [k for k in RAW_KEYS if k not in day]
        if missing:
            raise ValueError(f"Dan {i+1}: fale kljucevi {missing}")

        num_cols = ["Vodostaj_Bileca","Dotok_Prethodni_Dan","Padavine_Trebinje",
                     "Temperatura_Vazduha","Padavine_Sutra","Padavine_Za2Dana",
                     "Promjena_Vodostaja"]
        for k in num_cols:
            lo, hi = PHYS_RANGES[k]
            if not lo <= day[k] <= hi:
                raise ValueError(f"Dan {i+1}: {k}={day[k]} van [{lo},{hi}]")

        if day["Doba_Godine"] not in VALID_DOBA:
            raise ValueError(f"Dan {i+1}: Doba_Godine='{day['Doba_Godine']}' nije validna")
        if day["Pritisak_Mreze"] not in VALID_PRITISAK:
            raise ValueError(f"Dan {i+1}: Pritisak_Mreze='{day['Pritisak_Mreze']}' nije validan")

    c = _load()
    df = pd.DataFrame(days)
    df["Doba_Godine_enc"] = c["le_doba"].transform(df["Doba_Godine"])
    df["Pritisak_Mreze_enc"] = c["le_prit"].transform(df["Pritisak_Mreze"])
    X = df[FEATURES]

    kontinualno = c["reg"].predict(X)
    celobrojno = np.clip(np.rint(kontinualno), 0, 3).astype(int)
    proba = c["clf"].predict_proba(X)[:, 1]
    rizik = (proba >= c["threshold"]).astype(int)

    return [
        dict(
            dan=i + 1,
            agregati_kontinualno=float(kontinualno[i]),
            agregati=int(celobrojno[i]),
            preliv_vjerovatnoca=float(proba[i]),
            preliv_rizik=int(rizik[i]),
            prag=float(c["threshold"]),
        )
        for i in range(7)
    ]


def demo_days():
    base = {"Temperatura_Vazduha": 12, "Doba_Godine": "Proljece", "Pritisak_Mreze": "Srednje"}
    plan = [
        (388.5, 180, 12, 35, 60, 0.05),
        (388.9, 220, 35, 60, 80, 0.09),
        (389.6, 300, 60, 80, 90, 0.15),
        (390.4, 380, 80, 90, 40, 0.20),
        (391.1, 420, 90, 40, 10, 0.22),
        (391.6, 400, 40, 10,  0, 0.12),
        (391.8, 320, 10,  0,  0, 0.05),
    ]
    return [
        {**base, "Vodostaj_Bileca": v, "Dotok_Prethodni_Dan": d,
         "Padavine_Trebinje": p, "Padavine_Sutra": s,
         "Padavine_Za2Dana": z, "Promjena_Vodostaja": pr}
        for v, d, p, s, z, pr in plan
    ]


def main():
    print("Demo prognoza (7 dana):")
    for r in forecast_7days(demo_days()):
        alarm = "!!! RIZIK" if r["preliv_rizik"] else "ok"
        print(f"  Dan {r['dan']}: agregati={r['agregati']} "
              f"P(preliv)={r['preliv_vjerovatnoca']:.3f} "
              f"prag={r['prag']:.3f} -> {alarm}")


if __name__ == "__main__":
    main()
