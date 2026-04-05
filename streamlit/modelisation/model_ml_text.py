# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 09:54:44 2025

@author: Duong
"""

import os
import joblib
from functools import lru_cache

# ============================================================
# CHEMINS DES FICHIERS DU MODÈLE
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data_streamlit")

VECTORIZER_PATH = os.path.join(DATA_DIR, "ML_text4_tfidf_3.joblib")
MODEL_PATH = os.path.join(DATA_DIR, "ML_text4_svc_3.joblib")

PREPROCESS_IMG = os.path.join(DATA_DIR, "ML_text_clean.jpg")
CONFUSION_MATRIX_IMG = os.path.join(DATA_DIR, "matrice_confusion_svc_3.png")

# ➕ NEW : chemin de l’image des résultats
RESULTS_IMG = os.path.join(DATA_DIR, "ML_Text_Tableau_Model_F1-Score.jpg")

# ============================================================
# DICTIONNAIRE DES CODES
# ============================================================
CODE_LABELS = {
    10: "livres",
    40: "import_jeux",
    50: "console_jeux",
    60: "console_portable",
    1140: "figurines",
    1160: "cartes_collection",
    1180: "jeux_plateau",
    1280: "jouets",
    1281: "jeux_enfants",
    1300: "modelisme",
    1301: "vetements_enfants",
    1302: "gadgets",
    1320: "puericulture",
    1560: "mobilier",
    1920: "literie",
    1940: "alimentation",
    2060: "deco_lumiere",
    2220: "animaux",
    2280: "presse",
    2403: "livres",
    2462: "gaming",
    2522: "papeterie",
    2582: "jardin",
    2583: "piscine",
    2585: "outils_jardin",
    2705: "roman",
    2905: "jeux_telechargement",
}

def get_label_from_code(code: int) -> str:
    return CODE_LABELS.get(code, "Signification inconnue")


# ============================================================
# CHARGEMENT AVEC CACHE
# ============================================================
@lru_cache(maxsize=1)
def load_vectorizer():
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer introuvable : {VECTORIZER_PATH}")
    return joblib.load(VECTORIZER_PATH)


@lru_cache(maxsize=1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# ============================================================
# PRÉDICTION
# ============================================================
def predict_text(model, vectorizer, text):
    X = vectorizer.transform([text])
    pred = model.predict(X)
    return int(pred[0])


# ============================================================
# FONCTIONS POUR LES IMAGES DE DOCUMENTATION
# ============================================================
def get_preprocessing_image():
    return PREPROCESS_IMG if os.path.exists(PREPROCESS_IMG) else None

def get_confusion_matrix_image():
    return CONFUSION_MATRIX_IMG if os.path.exists(CONFUSION_MATRIX_IMG) else None

# ➕ NEW : Image des résultats (tableau F1-Score)
def get_results_image():
    return RESULTS_IMG if os.path.exists(RESULTS_IMG) else None





