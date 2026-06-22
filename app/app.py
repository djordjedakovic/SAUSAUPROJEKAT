# -*- coding: utf-8 -*-
"""HE Trebinje I - Streamlit app. Pokretanje: streamlit run app/app.py"""

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src.predict import forecast_7days, demo_days

st.set_page_config(page_title="HE Trebinje I", layout="wide")

MODELS = os.path.join(BASE, "models")

th_path = os.path.join(MODELS, "threshold.json")
if os.path.exists(th_path):
    with open(th_path, encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]
else:
    threshold = 0.02


def main():
    pass


st.title("HE Trebinje I")
st.caption("Planiranje agregata i rizik preliva — prognoza za 7 dana.")

st.subheader("Unos podataka za 7 dana")

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
        ax2.axhline(threshold, ls="--", color="black", alpha=0.5)
        ax2.set_ylabel("P(preliv)")
        ax2.set_ylim(0, 1.05)
        ax1.set_title("Plan agregata i rizik preliva")
        st.pyplot(fig); plt.close(fig)

        if out["preliv_rizik"].any():
            risk_days = out[out["preliv_rizik"] == 1]["dan"].to_list()
            st.error(f"Rizik za dane: {risk_days}! Angazuj sve agregate.")
        else:
            st.success("Nema rizika u narednih 7 dana.")
