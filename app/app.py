# -*- coding: utf-8 -*-
"""HE Trebinje I - Streamlit app. Pokretanje: streamlit run app/app.py"""

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src.predict import forecast_7days, demo_days

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="HE Trebinje I", layout="wide")

PLOTS = os.path.join(BASE, "plots")
DATA = os.path.join(BASE, "data")


@st.cache_data
def _df():
    df = pd.read_csv(os.path.join(DATA, "hetI_clean.csv"))
    df["Datum"] = pd.to_datetime(df["Datum"])
    return df


@st.cache_data
def _metrics():
    p = os.path.join(BASE, "metrics.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _fs():
    p = os.path.join(BASE, "models", "feature_selection.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    pass


df = _df()
m = _metrics()
fs_data = _fs()

st.title("HE Trebinje I")
st.caption("Planiranje agregata (regresija) + rizik preliva (klasifikacija).")

t1, t2, t3 = st.tabs(["Podaci", "Modeli", "Prognoza"])

# ============================================================ TAB 1
with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dana", f"{len(df):,}")
    c2.metric("Period", f"{df['Datum'].dt.year.min()}-{df['Datum'].dt.year.max()}")
    c3.metric("Preliva", int(df["Preliv_Status"].sum()))
    c4.metric("Udio", f"{df['Preliv_Status'].mean()*100:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df["Angazovani_Agregati"].value_counts().sort_index().plot.bar(ax=ax, color="#3b82c4")
        ax.set_title("Agregati 0-3"); st.pyplot(fig); plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df["Preliv_Status"].value_counts().sort_index().plot.bar(
            ax=ax, color=["#3b82c4", "#d9534f"])
        ax.set_yscale("log"); ax.set_title("Preliv"); st.pyplot(fig); plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.boxplot(data=df, x="Angazovani_Agregati", y="Vodostaj_Bileca",
                    hue="Angazovani_Agregati", palette="Blues", legend=False, ax=ax)
        ax.set_title("Vodostaj vs agregati"); st.pyplot(fig); plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.boxplot(data=df, x="Doba_Godine", y="Padavine_Trebinje",
                    order=["Zima","Proljece","Ljeto","Jesen"],
                    hue="Doba_Godine", palette="viridis", legend=False, ax=ax)
        ax.set_title("Padavine po dobima"); st.pyplot(fig); plt.close(fig)

    sezona = df.groupby("Doba_Godine").agg(
        dotok=("Dotok_Prethodni_Dan","mean"),
        padavine=("Padavine_Trebinje","mean"),
        agregati=("Angazovani_Agregati","mean"),
        preliv=("Preliv_Status","mean"),
    ).reindex(["Zima","Proljece","Ljeto","Jesen"]).round(3)
    st.subheader("Sezonski pregled")
    st.dataframe(sezona, width="stretch")

    nc = ["Vodostaj_Bileca","Dotok_Prethodni_Dan","Padavine_Trebinje","Temperatura_Vazduha",
          "Padavine_Sutra","Padavine_Za2Dana","Promjena_Vodostaja",
          "Angazovani_Agregati","Preliv_Status"]
    corr_matrix = pd.DataFrame(df[nc]).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, annot_kws={"size": 7})
    st.subheader("Korelacije"); st.pyplot(fig); plt.close(fig)

    st.subheader("Sezonski trend")
    df_plot = df.assign(Mjesec=df["Datum"].dt.month)
    mj = df_plot.groupby("Mjesec")[["Vodostaj_Bileca","Dotok_Prethodni_Dan"]].mean()
    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    ax2 = ax1.twinx()
    ax1.plot(mj.index, mj["Vodostaj_Bileca"], "b-o", label="Vodostaj")
    ax2.bar(mj.index, mj["Dotok_Prethodni_Dan"], alpha=0.3, color="teal", label="Dotok")
    ax1.set_xticks(range(1,13))
    ax1.set_xticklabels(["Jan","Feb","Mar","Apr","Maj","Jun","Jul","Avg","Sep","Okt","Nov","Dec"])
    fig.legend(loc="upper right"); st.pyplot(fig); plt.close(fig)

# ============================================================ TAB 2
with t2:
    if not m:
        st.warning("Pokreni evaluate.py prvo.")
        st.stop()

    test = m["test"]
    th_path = os.path.join(BASE, "models", "threshold.json")
    if os.path.exists(th_path):
        with open(th_path, encoding="utf-8") as f:
            threshold = json.load(f)["threshold"]
    else:
        threshold = 0.02

    st.subheader("Regresija (test skup)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (svi attr.)", f"{test['XGBoost_regresor']['MAE']:.3f}",
              delta=f"top5: {test.get('XGBoost_regresor_top5',{}).get('MAE',0):.3f}", delta_color="off")
    c2.metric("RMSE (svi attr.)", f"{test['XGBoost_regresor']['RMSE']:.3f}",
              delta=f"top5: {test.get('XGBoost_regresor_top5',{}).get('RMSE',0):.3f}", delta_color="off")
    c3.metric("R2 (svi attr.)", f"{test['XGBoost_regresor']['R2']:.3f}",
              delta=f"top5: {test.get('XGBoost_regresor_top5',{}).get('R2',0):.3f}", delta_color="off")
    c4.metric("Baseline MAE", f"{test['Baseline_regresor']['MAE']:.3f}")

    st.info("Accuracy se ne koristi: baseline ima ~98.7% accuracy ali recall=0.")

    if m.get("variants"):
        st.subheader("Poredjenje varijanti klasifikatora (test skup)")
        rows = []
        for code, data in m["variants"].items():
            tdata = data["test"]
            cm_data = np.array(tdata["cm"])
            tn, fp, fn, tp = cm_data.ravel()
            rows.append({
                "Varijanta": data["naziv"],
                "Prag": f"{tdata['threshold']:.3f}",
                "Recall": f"{tdata['recall']:.3f}",
                "Precision": f"{tdata['precision']:.3f}",
                "F1": f"{tdata['f1']:.3f}",
                "PR-AUC": f"{tdata.get('PR_AUC',0):.3f}",
                "TP": f"{tp}/{tp+fn}",
                "FP": str(fp),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Varijanta"), width="stretch")

    for fname, label in [
        ("variants_comparison.png", "Poredjenje varijanti"),
        ("threshold_sweeps.png", "Sweep pragova"),
        ("all_confusion_matrices.png", "Sve confusion matrice"),
        ("prob_distributions.png", "Distribucije verovatnoca"),
    ]:
        pth = os.path.join(PLOTS, fname)
        if os.path.exists(pth):
            st.subheader(label); st.image(pth)

    if fs_data:
        st.subheader("Selekcija atributa")
        c1, c2, c3 = st.columns(3)
        c1.metric("Zadrzani (top 5)", ", ".join(fs_data.get("zadrzani_atributi", [])))
        c2.metric("Odbaceni (4)", ", ".join(fs_data.get("odbaceni_atributi", [])))
        with c3:
            st.write("**Vaznost atributa:**")
            for f, v in sorted(fs_data.get("vaznost_atributa", {}).items(), key=lambda x: -x[1]):
                st.caption(f"{f}: {v:.4f}")

        # Poređenje svi vs top5 atributa — regresija i klasifikacija
        st.subheader("Svi vs Top 5 atributa — poređenje")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Regresija (XGBoost)**")
            reg_svi = test.get("XGBoost_regresor", {})
            reg_top = test.get("XGBoost_regresor_top5", {})
            st.dataframe(pd.DataFrame({
                "": ["Svi atributi (9)", "Top 5 atributa"],
                "MAE": [f"{reg_svi.get('MAE',0):.3f}", f"{reg_top.get('MAE',0):.3f}"],
                "RMSE": [f"{reg_svi.get('RMSE',0):.3f}", f"{reg_top.get('RMSE',0):.3f}"],
                "R2": [f"{reg_svi.get('R2',0):.3f}", f"{reg_top.get('R2',0):.3f}"],
            }).set_index(""))
        with col2:
            st.markdown("**Klasifikacija (XGBoost + isotonic)**")
            clf_svi = test.get("XGBoost_klasifikator_svi", {})
            clf_top = test.get("XGBoost_klasifikator_top5", {})
            st.dataframe(pd.DataFrame({
                "": ["Svi atributi (9)", "Top 5 atributa"],
                "Recall": [f"{clf_svi.get('recall',0):.3f}", f"{clf_top.get('recall',0):.3f}"],
                "Precision": [f"{clf_svi.get('precision',0):.3f}", f"{clf_top.get('precision',0):.3f}"],
                "PR-AUC": [f"{clf_svi.get('PR_AUC',0):.3f}", f"{clf_top.get('PR_AUC',0):.3f}"],
            }).set_index(""))

        for fn in ["feature_selection.png", "feature_selection_clf.png"]:
            pth = os.path.join(PLOTS, fn)
            if os.path.exists(pth):
                st.image(pth)

    tp = os.path.join(BASE, "tumacenje.txt")
    if os.path.exists(tp):
        with st.expander("Tumacenje"):
            with open(tp, encoding="utf-8") as f:
                st.text(f.read())

# ============================================================ TAB 3
with t3:
    st.subheader("Prognoza za 7 dana")
    if st.button("Ucitaj demo"):
        st.session_state["demo"] = demo_days()

    demo = st.session_state.get("demo")
    days = []
    for i in range(7):
        defaults = demo[i] if demo else {}
        with st.expander(f"Dan {i+1}", expanded=i == 0):
            c1, c2, c3 = st.columns(3)
            vod = c1.number_input("Vodostaj (m)", 360., 400.,
                                  float(defaults.get("Vodostaj_Bileca", 383.)),
                                  0.1, key=f"v{i}")
            dot = c1.number_input("Dotok (m3/s)", 0., 800.,
                                  float(defaults.get("Dotok_Prethodni_Dan", 100.)),
                                  1., key=f"d{i}")
            pad = c1.number_input("Padavine (mm)", 0., 300.,
                                  float(defaults.get("Padavine_Trebinje", 0.)),
                                  0.5, key=f"p{i}")
            tmp = c2.number_input("Temp (C)", -20., 45.,
                                  float(defaults.get("Temperatura_Vazduha", 15.)),
                                  0.5, key=f"t{i}")
            ps  = c2.number_input("Padavine sutra", 0., 300.,
                                  float(defaults.get("Padavine_Sutra", 0.)),
                                  0.5, key=f"s{i}")
            p2  = c2.number_input("Padavine +2d", 0., 300.,
                                  float(defaults.get("Padavine_Za2Dana", 0.)),
                                  0.5, key=f"z{i}")
            doba_opts = ["Zima","Proljece","Ljeto","Jesen"]
            do  = c3.selectbox("Doba", doba_opts,
                               doba_opts.index(defaults.get("Doba_Godine","Proljece")),
                               key=f"do{i}")
            prit_opts = ["Nisko","Srednje","Visoko"]
            pr  = c3.selectbox("Pritisak", prit_opts,
                               prit_opts.index(defaults.get("Pritisak_Mreze","Srednje")),
                               key=f"pr{i}")
            pv  = c3.number_input("Promjena vodostaja", -2., 2.,
                                  float(defaults.get("Promjena_Vodostaja", 0.)),
                                  0.01, key=f"pv{i}")
        days.append(dict(
            Vodostaj_Bileca=vod, Dotok_Prethodni_Dan=dot, Padavine_Trebinje=pad,
            Temperatura_Vazduha=tmp, Doba_Godine=do, Pritisak_Mreze=pr,
            Padavine_Sutra=ps, Padavine_Za2Dana=p2, Promjena_Vodostaja=pv,
        ))

    if st.button("Prognoza", type="primary"):
        try:
            out = pd.DataFrame(forecast_7days(days))
        except ValueError as e:
            st.error(str(e))
            st.stop()
        else:
            out["status"] = np.where(out["preliv_rizik"], "RIZIK", "OK")

            st.subheader("Rezultati")
            st.dataframe(
                out[["dan","agregati","agregati_kontinualno","preliv_vjerovatnoca","status"]]
                .set_axis(["Dan", "Agregati", "Kont.", "P(preliv)", "Status"], axis=1)
                .style.format({"Kont.":"{:.2f}","P(preliv)":"{:.3f}"}),
                width="stretch",
            )

            fig, ax1 = plt.subplots(figsize=(9, 4))
            ax1.bar(out["dan"], out["agregati"], color="#3b82c4", label="Agregati")
            ax1.set_xlabel("Dan"); ax1.set_ylabel("Agregata"); ax1.set_ylim(0, 3.4)
            ax2 = ax1.twinx()
            colors = ["#d9534f" if r else "#5cb85c" for r in out["preliv_rizik"]]
            ax2.bar(out["dan"], out["preliv_vjerovatnoca"], color=colors, alpha=0.7)
            ax2.axhline(threshold if m else 0.5, ls="--", color="black", alpha=0.5)
            ax2.set_ylabel("P(preliv)")
            ax2.set_ylim(0, 1.05)
            ax1.set_title("Plan agregata i rizik preliva")
            st.pyplot(fig); plt.close(fig)

            if out["preliv_rizik"].any():
                risk_days = out[out["preliv_rizik"] == 1]["dan"].to_list()
                st.error(f"Rizik za dane: {risk_days}! Angazuj sve agregate.")
            else:
                st.success("Nema rizika u narednih 7 dana.")
