# HE Trebinje I — Predikcija vremenskih serija

Projekat iz predmeta **SAUSAU** (Nivo 3 — sopstveni dataset i trening modela).

Sistem masinskog ucenja za upravljanje hidroelektranom Trebinje I:

1. **Regresija** — predikcija broja angazovanih turbinskih agregata (0-3)
2. **Klasifikacija** — predikcija rizika od prelivanja Bileckog jezera (0/1)

## Dataset

Sopstveno prikupljeni i ocisceni podaci o Bileckom jezeru (HE Trebinje I):
- **Period**: 2000–2027 (~10,000 dana)
- **Atributi**: Vodostaj, dotok, padavine, temperatura, doba godine, pritisak mreze, promjena vodostaja
- **Target 1** (regresija): Angazovani_Agregati (0-3)
- **Target 2** (klasifikacija): Preliv_Status (0/1, svega 1.28% pozitivnih)

## Struktura projekta

```
PROJEKTOVANJE/
├── pyproject.toml
├── README.md
├── .gitignore
├── data/
│   ├── hetI.csv              # originalni dataset
│   ├── hetI_clean.csv        # ocisceni dataset
│   └── splits.joblib         # train/val/test splitovi
├── models/                   # sacuvani modeli, enkoderi, pragovi
├── plots/                    # generisani grafici
├── app/
│   └── app.py                # Streamlit interaktivna aplikacija
├── src/
│   ├── config.py             # konstante
│   ├── data_prep.py          # KORAK 1: priprema podataka
│   ├── train.py              # KORAK 2: treniranje vise varijanti
│   ├── evaluate.py           # KORAK 3: evaluacija i poredjenje
│   └── predict.py            # KORAK 4: 7-dnevna prognoza
├── metrics.json
└── tumacenje.txt
```

## Pokretanje (redom)

```bash
# Instalacija zavisnosti
pip install -r requirements.txt

# Korak 1: Priprema podataka
python src/data_prep.py

# Korak 2: Treniranje svih varijanti
python src/train.py

# Korak 3: Evaluacija i poredjenje varijanti
python src/evaluate.py

# Korak 4: Demo prognoza
python src/predict.py

# Streamlit aplikacija
streamlit run app/app.py
```

## Istrazena arhitektura i varijante

Za **regresiju** koristi se XGBoost i Random Forest regresor.

Za **klasifikaciju** je istrazeno **7 varijanti** kako bi se istrazila mogucnost priblizavanja praga odluke vrednosti 0.5:

| Varijanta | Opis |
|-----------|------|
| **A: Raw** | XGBoost BEZ kalibracije (sirove `predict_proba` vrednosti) |
| **B: Platt** | XGBoost + Platt (sigmoid) kalibracija |
| **C: Isotonic** | XGBoost + isotonic kalibracija (originalni pristup) |
| **D: Undersample** | `RandomUnderSampler(0.3)` + XGBoost bez kalibracije |
| **E: Isolation Forest** | Anomaly detection — treniran samo na danima bez preliva |
| **RF ref** | Random Forest + SMOTE + isotonic (referentni model) |
| **LR ref** | Logistic Regression + SMOTE + isotonic (referentni model) |

## Metodologija

- **Split**: 70/10/20 (train/validation/test), stratifikovan po `Preliv_Status`
- **`random_state=42`** svuda za reproducibilnost
- **Prag odluke** optimizovan na validation skupu za svaku varijantu posebno:
  - Sweep od 0.01 do 0.99 (korak 0.01)
  - Kriterijum: **max recall** uz precision >= 5%
- **Validation** za sve odluke, **test** evaluiran samo jednom
- **Recall iznad svega**: propusten preliv moze ostetiti branu; lazan alarm samo trosi paznju dispecera

## Metrike

- **Regresija**: MAE, RMSE, R²
- **Klasifikacija**: Recall, Precision, F1, PR-AUC, Confusion Matrix
- Accuracy se ne koristi — baseline daje 98.7% accuracy ali recall=0

## Streamlit aplikacija

Interaktivna aplikacija sa 3 taba:
1. **Podaci** — EDA, distribucije, korelacije, sezonski pregled
2. **Modeli** — poredjenje svih varijanti, metrike, grafici
3. **Prognoza** — 7-dnevna prognoza agregata i rizika preliva

## Tehnologije

- Python 3.10+, uv za upravljanje projektom
- XGBoost, scikit-learn, imbalanced-learn
- pandas, numpy, matplotlib, seaborn
- Streamlit za interaktivnu vizuelizaciju
