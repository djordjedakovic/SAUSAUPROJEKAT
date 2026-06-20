# Upravljanje hidroelektranom Trebinje I — ML sistem

Sistem masinskog ucenja koji simulira rad dispecera u HE Trebinje I na
osnovu podataka o Bileckom jezeru:

1. **Regresija** — koliko turbinskih agregata (0-3) treba ukljuciti
2. **Klasifikacija** — da li postoji rizik od prelivanja jezera (0/1)

## Struktura projekta

```
PROJEKTOVANJE/
├── data/
│   ├── hetI.csv             # originalni dataset
│   ├── hetI_clean.csv       # ocisceni dataset (poslije validacije)
│   └── splits.joblib        # train/val/test splitovi
├── models/                  # sacuvani modeli (vise varijanti), enkoderi i prag
├── plots/                   # generisani grafici
├── app/app.py               # Streamlit aplikacija (3 taba)
├── src/
│   ├── data_prep.py         # KORAK 1
│   ├── train.py             # KORAK 2
│   ├── evaluate.py          # KORAK 3
│   └── predict.py           # KORAK 4 (forecast_7days)
├── metrics.json             # sve metrike (validation + test + varijante)
└── tumacenje.txt            # tumacenje rezultata
```

## Pokretanje (redom)

```bash
pip install -r requirements.txt

python src/data_prep.py   # priprema, validacija, split, enkoderi
python src/train.py       # treniranje 7 varijanti + optimizacija praga
python src/evaluate.py    # metrike, grafici, poredjenje, metrics.json
python src/predict.py     # demo 7-dnevne prognoze u konzoli

streamlit run app/app.py  # interaktivna aplikacija
```

## Varijante klasifikatora

Projekat istrazuje 7 razlicitih pristupa klasifikaciji rizika preliva:

| Kod | Naziv | Opis |
|-----|-------|------|
| A_raw | XGBoost BEZ kalibracije | Sirovi predict_proba score-ovi |
| B_platt | XGBoost + Platt | sigmoid kalibracija |
| C_iso | XGBoost + isotonic | isotonic kalibracija (original) |
| D_undersample | Undersampling + XGBoost | RandomUnderSampler(0.3) |
| E_iforest | Isolation Forest | Anomaly detection |
| RF_ref | RandomForest + SMOTE | Referentni model |
| LR_ref | LogisticRegression + SMOTE | Referentni model |

## Metodologija (kljucna pravila)

- **Split 70/10/20** (train/validation/test), stratifikovan po `Preliv_Status`
- **SMOTE samo na train skupu** — unutar `ImbPipeline`, nikad na val/test
- **Prag odluke** optimizovan na validation skupu za svaku varijantu posebno
- **`random_state=42` svuda** — reproducibilnost
- **Recall iznad svega**: propusten preliv moze trajno ostetiti branu
- **PR-AUC** za prag-nezavisno poredjenje varijanti

## Rezultati

Pogledati `metrics.json` i `tumacenje.txt` nakon pokretanja evaluate.py.
Grafike u `plots/` folderu prikazuju poredjenje svih varijanti.
