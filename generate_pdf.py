# -*- coding: utf-8 -*-
"""Generisi dokumentacija.pdf za HE Trebinje I projekat."""
import os, json
from fpdf import FPDF

BASE = os.path.dirname(os.path.abspath(__file__))

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_font("DejaVu", "", r"C:\Windows\Fonts\arial.ttf")
pdf.add_font("DejaVu", "B", r"C:\Windows\Fonts\arialbd.ttf")
pdf.add_font("DejaVu", "I", r"C:\Windows\Fonts\ariali.ttf")

def title_page():
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("DejaVu", "B", 28)
    pdf.cell(0, 14, "HE Trebinje I", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("DejaVu", "", 16)
    pdf.cell(0, 10, "Predikcija vremenskih serija", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.cell(0, 10, "Regresija agregata i klasifikacija rizika preliva", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("DejaVu", "I", 13)
    pdf.cell(0, 8, "Projekat iz predmeta SAUSAU", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "RA156-2023", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("DejaVu", "", 11)
    pdf.cell(0, 7, "Autor: Đorđe Đaković", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "2026", align="C", new_x="LMARGIN", new_y="NEXT")

def heading(text, level=1):
    sizes = {1: 16, 2: 13, 3: 11}
    pdf.set_font("DejaVu", "B", sizes.get(level, 11))
    pdf.ln(4)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

def body(text):
    pdf.set_font("DejaVu", "", 10)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(1)

def code_block(text):
    pdf.set_font("DejaVu", "", 8)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_font("DejaVu", "", 8)
    lines = text.strip().split("\n")
    for line in lines:
        pdf.set_x(15)
        pdf.cell(0, 4.5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

def bullet(text):
    pdf.set_font("DejaVu", "", 10)
    pdf.cell(8, 5.5, "•", new_x="RIGHT", new_y="LAST")
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(0.5)

def table(headers, rows):
    pdf.set_font("DejaVu", "B", 8.5)
    col_w = pdf.epw / len(headers)
    for h in headers:
        pdf.cell(col_w, 6.5, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("DejaVu", "", 8.5)
    for row in rows:
        for cell in row:
            pdf.cell(col_w, 6, str(cell), border=1, align="C")
        pdf.ln()
    pdf.ln(3)

# ── Cover ──
title_page()

# ── 1. Uvod ──
pdf.add_page()
heading("1. Uvod")
body(
    "Sistem masinskog ucenja za upravljanje hidroelektranom Trebinje I. "
    "Projekat simulira rad dispecera koji na osnovu istorijskih i trenutnih podataka "
    "donosi dve kljucne odluke:"
)
bullet("Regresija — koliko od ukupno 3 turbinska agregata treba aktivirati")
bullet("Klasifikacija — da li postoji rizik od prelivanja Bileckog jezera (0/1)")

heading("1.1 Opis problema", 2)
body(
    "Planiranje angazmana agregata — dispecer svaki dan mora odluciti koliko od ukupno 3 "
    "turbinska agregata treba aktivirati, uzimajuci u obzir trenutni vodostaj jezera, "
    "dotok vode, pritisak elektricne mreze i vremensku prognozu. Prevelik broj aktivnih "
    "agregata nepotrebno prazni rezervoar, a premali ne iskoristava dostupnu energiju."
)
body(
    "Detekcija rizika od preliva — kada nivo jezera poraste iznad maksimalno dozvoljene kote, "
    "dolazi do nekontrolisanog prelivanja vode preko brane. Preliv je izuzetno opasan dogadjaj "
    "koji moze uzrokovati trajno ostecenje konstrukcije brane, poplavu nizvodnih naselja i "
    "gubitak ljudskih zivota. Dispecer mora blagovremeno prepoznati rizik i pravovremeno "
    "reagovati."
)

# ── 2. Dataset ──
heading("2. Dataset")
body(
    "Sopstveno prikupljeni i ocisceni podaci o Bileckom jezeru (HE Trebinje I):"
)
bullet("Period: 2000–2027 (~10,000 dana)")
bullet("Atributi: Vodostaj, dotok, padavine, temperatura, doba godine, pritisak mreze, promjena vodostaja")
bullet("Target 1 (regresija): Angazovani_Agregati (0–3)")
bullet("Target 2 (klasifikacija): Preliv_Status (0/1, svega 1.28% pozitivnih — ekstreman imbalance)")

heading("2.1 Atributi", 2)
table(
    ["Atribut", "Opis", "Opseg"],
    [
        ["Vodostaj_Bileca", "Nivo vode u jezeru (m)", "300–420"],
        ["Dotok_Prethodni_Dan", "Dotok vode prethodnog dana (m³/s)", "0–2000"],
        ["Padavine_Trebinje", "Padavine u Trebinju (mm)", "0–500"],
        ["Temperatura_Vazduha", "Temperatura vazduha (°C)", "-30–50"],
        ["Doba_Godine", "Sezona (Zima/Proljece/Ljeto/Jesen)", "4 kategorije"],
        ["Pritisak_Mreze", "Pritisak elektricne mreze", "3 kategorije"],
        ["Padavine_Sutra", "Prognoza padavina za sutra (mm)", "0–500"],
        ["Padavine_Za2Dana", "Prognoza padavina za 2 dana (mm)", "0–500"],
        ["Promjena_Vodostaja", "Dnevna promjena vodostaja (m)", "-5–5"],
    ]
)

# ── 3. Struktura ──
heading("3. Struktura projekta")
code_block("""PROJEKTOVANJE/
  data/
    hetI.csv              # originalni dataset
    hetI_clean.csv        # ocisceni dataset
    splits.joblib         # train/val/test splitovi
  models/                 # sacuvani modeli i enkoderi
  plots/                  # generisani grafici
  app/
    app.py                # Streamlit aplikacija
  src/
    config.py             # konstante
    data_prep.py          # KORAK 1: priprema podataka
    train.py              # KORAK 2: treniranje
    evaluate.py           # KORAK 3: evaluacija
    predict.py            # KORAK 4: 7-dnevna prognoza
  metrics.json            # sve metrike
  tumacenje.txt           # tumacenje rezultata
  dokumentacija.pdf       # ova dokumentacija""")

# ── 4. Metodologija ──
heading("4. Metodologija")
bullet("Split: 70/10/20 (train/validation/test), stratifikovan po Preliv_Status")
bullet("random_state=42 svuda za reproducibilnost")
bullet("SMOTE samo na train skupu — unutar ImbPipeline, nikad na val/test")
bullet("Prag odluke optimizovan na validation skupu za svaku varijantu posebno: sweep 0.01–0.99")
bullet("Kriterijum: max recall uz precision >= 5%")
bullet("Validation za sve odluke; test skup evaluiran samo jednom")
bullet("Recall iznad svega — propusten preliv moze ostetiti branu")

# ── 5. Varijante ──
heading("5. Varijante klasifikatora")
body("Istrazivano je 7 varijanti klasifikacije kako bi se prag odluke priblizio 0.5:")
table(
    ["Kod", "Naziv", "Opis"],
    [
        ["A_raw", "XGBoost BEZ kalibracije", "Sirovi predict_proba score-ovi"],
        ["B_platt", "XGBoost + Platt", "Sigmoid kalibracija"],
        ["C_iso", "XGBoost + isotonic", "Isotonic kalibracija (original)"],
        ["D_undersample", "Undersampling + XGBoost", "RandomUnderSampler(0.3)"],
        ["E_iforest", "Isolation Forest", "Anomaly detection"],
        ["RF_ref", "RandomForest + SMOTE", "Referentni model"],
        ["LR_ref", "LogisticRegression + SMOTE", "Referentni model"],
    ]
)

# ── 6. Rezultati ──
heading("6. Rezultati")

heading("6.1 Regresija (test skup)", 2)
metrics_path = os.path.join(BASE, "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, encoding="utf-8") as f:
        m_data = json.load(f)
    test = m_data.get("test", {})
    reg_xgb = test.get("XGBoost_regresor", {})
    reg_rf = test.get("RandomForest_regresor", {})
    reg_bl = test.get("Baseline_regresor", {})
    reg_top = test.get("XGBoost_regresor_top5", {})
    table(
        ["Model", "MAE", "RMSE", "R²"],
        [
            ["XGBoost (svi)", f"{reg_xgb.get('MAE',0):.3f}", f"{reg_xgb.get('RMSE',0):.3f}", f"{reg_xgb.get('R2',0):.3f}"],
            ["XGBoost (top5)", f"{reg_top.get('MAE',0):.3f}", f"{reg_top.get('RMSE',0):.3f}", f"{reg_top.get('R2',0):.3f}"],
            ["Random Forest", f"{reg_rf.get('MAE',0):.3f}", f"{reg_rf.get('RMSE',0):.3f}", f"{reg_rf.get('R2',0):.3f}"],
            ["Baseline", f"{reg_bl.get('MAE',0):.3f}", "-", "-"],
        ]
    )

heading("6.2 Klasifikacija — poređenje varijanti (test skup)", 2)
variants = m_data.get("variants", {})
if variants:
    rows = []
    for code, data in sorted(variants.items(),
                              key=lambda kv: (kv[1]["test"]["recall"],
                                              kv[1]["test"]["f1"]),
                              reverse=True):
        t = data["test"]
        cm_list = t.get("cm", [[0,0],[0,0]])
        tn, fp = cm_list[0][0], cm_list[0][1]
        fn, tp = cm_list[1][0], cm_list[1][1]
        rows.append([
            data["naziv"][:22],
            f"{t['threshold']:.3f}",
            f"{t['recall']:.3f}",
            f"{t['precision']:.3f}",
            f"{t['f1']:.3f}",
            f"{t.get('PR_AUC',0):.3f}",
            f"{tp}/{tp+fn}",
            str(fp),
            str(tn),
        ])
    table(
        ["Varijanta", "Prag", "Recall", "Prec", "F1", "PR-AUC", "TP", "FP", "TN"],
        rows,
    )

heading("6.3 Selekcija atributa", 2)
body("Najuticajniji atributi (prosek feature importance regresije i klasifikacije):")
fs_path = os.path.join(BASE, "models", "feature_selection.json")
if os.path.exists(fs_path):
    with open(fs_path, encoding="utf-8") as f:
        fs = json.load(f)
    imp = sorted(fs.get("vaznost_atributa", {}).items(), key=lambda x: -x[1])
    for i, (feat, val) in enumerate(imp, 1):
        marker = " +" if feat in fs.get("zadrzani_atributi", []) else " -"
        body(f"  {i}. {feat} = {val:.4f}{marker}")
    body(f"\nZadrzano (top 5): {', '.join(fs.get('zadrzani_atributi', []))}")
    body(f"Odbaceno (4): {', '.join(fs.get('odbaceni_atributi', []))}")

# ── 7. Pokretanje ──
heading("7. Pokretanje")
body("Instalacija zavisnosti:")
code_block("pip install -r requirements.txt")
body("Pipeline (redom):")
code_block("""python src/data_prep.py    # KORAK 1: priprema, validacija, split
python src/train.py        # KORAK 2: treniranje 7 varijanti
python src/evaluate.py     # KORAK 3: evaluacija, metrike, grafici
python src/predict.py      # KORAK 4: demo 7-dnevne prognoze""")
body("Streamlit aplikacija:")
code_block("streamlit run app/app.py")
# Alternative:
code_block("python -m streamlit run app/app.py")

# ── 8. Tehnologije ──
heading("8. Tehnologije")
bullet("Python 3.10+")
bullet("scikit-learn 1.3+ — Random Forest, Logistic Regression, LabelEncoder")
bullet("imbalanced-learn — SMOTE, RandomUnderSampler")
bullet("XGBoost 2.0+ — XGBRegressor, XGBClassifier")
bullet("pandas, numpy, matplotlib, seaborn")
bullet("Streamlit — interaktivna vizuelizacija")
bullet("joblib — serijalizacija modela")

# ── 9. Metrike ──
heading("9. Metrike")
bullet("Regresija: MAE, RMSE, R²")
bullet("Klasifikacija: Recall, Precision, F1, PR-AUC, Confusion Matrix")
bullet("Accuracy se ne koristi — baseline daje 98.7% accuracy ali recall=0")

# ── 10. Zakljucak ──
heading("10. Zakljucak")
body(
    "Cilj projekta je bio istraziti da li se prag odluke klasifikatora moze "
    "pribliziti vrednosti 0.5 primenom razlicitih tehnika kalibracije i "
    "balansiranja podataka, uz ocuvanje visokog recall-a (detekcije rizika "
    "od preliva). Najuspesnije varijante su Undersampling + XGBoost (prag=0.100, "
    "recall=0.962) i XGBoost + isotonic (prag=0.030, recall=0.885). "
    "Za regresiju, XGBoost postize MAE=0.126 i R²=0.884, sto znacajno "
    "nadmasuje baseline (MAE=0.629)."
)

# ── Save ──
output = os.path.join(BASE, "dokumentacija.pdf")
pdf.output(output)
print(f"PDF generisan: {output}")
