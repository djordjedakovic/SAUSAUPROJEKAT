# -*- coding: utf-8 -*-
"""Zajednicke konstante za HE Trebinje I."""

RANDOM = 42

FEATURES = [
    "Vodostaj_Bileca","Dotok_Prethodni_Dan","Padavine_Trebinje","Temperatura_Vazduha",
    "Doba_Godine_enc","Pritisak_Mreze_enc","Padavine_Sutra","Padavine_Za2Dana","Promjena_Vodostaja",
]

RAW_KEYS = [
    "Vodostaj_Bileca","Dotok_Prethodni_Dan","Padavine_Trebinje","Temperatura_Vazduha",
    "Doba_Godine","Pritisak_Mreze","Padavine_Sutra","Padavine_Za2Dana","Promjena_Vodostaja",
]

TARGET_REG = "Angazovani_Agregati"
TARGET_CLF = "Preliv_Status"

VALID_DOBA = {"Zima","Proljece","Ljeto","Jesen"}
VALID_PRITISAK = {"Nisko","Srednje","Visoko"}

PHYS_RANGES = {
    "Vodostaj_Bileca":(300,420),   "Dotok_Prethodni_Dan":(0,2000),
    "Padavine_Trebinje":(0,500),    "Temperatura_Vazduha":(-30,50),
    "Padavine_Sutra":(0,500),       "Padavine_Za2Dana":(0,500),
    "Promjena_Vodostaja":(-5,5),    "Angazovani_Agregati":(0,3),
    "Preliv_Status":(0,1),
}

GRID = {"n_estimators":[100,200,300],"max_depth":[4,6,8],"learning_rate":[0.05,0.1]}

TOP_N = 5
MIN_PRECISION = 0.05
TH_LO = 0.01
TH_HI = 0.99
TH_STEP = 0.01
