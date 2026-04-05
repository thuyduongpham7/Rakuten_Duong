# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 09:53:32 2025

@author: Duong
"""

import streamlit as st
import sys
import os

# ============================================================
# AJOUT DU DOSSIER "modelisation" AU PYTHONPATH
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # STREAMLIT/pages
BASE_DIR = os.path.dirname(CURRENT_DIR)                       # STREAMLIT/
MODELISATION_DIR = os.path.join(BASE_DIR, "modelisation")

if MODELISATION_DIR not in sys.path:
    sys.path.append(MODELISATION_DIR)

# Importation des fonctions backend
from model_ml_text import (
    load_model,
    load_vectorizer,
    predict_text,
    get_label_from_code,
    get_preprocessing_image,
    get_confusion_matrix_image,
    get_results_image    # ➕ NEW
)

# ============================================================
# SESSION STATE
# ============================================================
if "ml_model" not in st.session_state:
    st.session_state.ml_model = None

if "ml_vectorizer" not in st.session_state:
    st.session_state.ml_vectorizer = None

# ============================================================
# TITRE DE LA PAGE
# ============================================================
st.title("🔍 Machine Learning — Classification de texte")
st.write("Page de prédiction de texte via un modèle SVC + TF-IDF.")
st.markdown("---")

# ============================================================
# SECTION 1 — CHARGEMENT + PRÉDICTION
# ============================================================
st.header("📌 Chargement du modèle & prédiction")

# Bouton de chargement
load_btn = st.button("Charger le modèle")

if load_btn:
    try:
        st.session_state.ml_model = load_model()
        st.session_state.ml_vectorizer = load_vectorizer()
        st.success("Modèle et vectorizer chargés avec succès ✔")
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")

if st.session_state.ml_model and st.session_state.ml_vectorizer:
    st.info("Modèle et vectorizer déjà chargés en mémoire.")

# Zone de prédiction
st.subheader("🔠 Prédiction de texte")

if st.session_state.ml_model is None or st.session_state.ml_vectorizer is None:
    st.warning("Chargez le modèle avant de faire une prédiction.")
else:
    input_text = st.text_area(
        "Texte à analyser",
        placeholder="Exemple : piscine en bois"
    )

    if st.button("🔮 Lancer la prédiction"):
        if not input_text.strip():
            st.warning("Veuillez saisir un texte.")
        else:
            try:
                pred_code = predict_text(
                    st.session_state.ml_model,
                    st.session_state.ml_vectorizer,
                    input_text
                )
                label = get_label_from_code(pred_code)

                st.write("### 🎯 Classe prédite :")
                st.success(f"**{pred_code}**")

                st.write("### 📘 Catégorie :")
                st.info(f"**{label}**")

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

st.markdown("---")

# ============================================================
# SECTION 2 — CRÉATION DES MODÈLES
# ============================================================
st.header("📘 Création des modèles")

# Étapes
st.subheader("🛠️ Les étapes")
st.markdown("""
1. Chargement et préparation du dataset  
2. Nettoyage et prétraitements du texte  
3. Vectorisation TF-IDF  
4. Tests de plusieurs modèles (LinearSVC, Logistic Regression…)  
5. Sélection du meilleur modèle  
6. Évaluation finale  
""")

# Prétraitements
st.subheader("🧹 Prétraitements du texte")

preprocess_img = get_preprocessing_image()
if preprocess_img:
    st.image(preprocess_img, caption="Étapes de nettoyage du texte", use_container_width=True)
else:
    st.error("Image des prétraitements introuvable.")

# Modèles testés
st.subheader("🤖 Les modèles testés")
st.markdown("""
- Logistic Regression 
- Linear SVC (SVM)
- Naive Bayes 
- Random Forest   
- Gradient Boosting
""")

# ============================================================
# SECTION — RESULTATS (NOUVELLE SECTION)
# ============================================================
st.subheader("📈 Résultats")

results_img = get_results_image()
if results_img:
    st.image(results_img, caption="F1-Score des modèles testés", use_container_width=True)
else:
    st.error("Image des résultats introuvable.")

# ============================================================
# MATRICE DE CONFUSION
# ============================================================
st.subheader("📊 Matrice de confusion")

cm_img_path = os.path.join(BASE_DIR, "data_streamlit", "MLsimple_Text8_MatriceConfusion_svc3_001.jpg")

if os.path.exists(cm_img_path):
    st.image(cm_img_path, caption="Matrice de confusion du modèle choisi (Linear SVC - essai 3)", use_container_width=True)
else:
    st.error("Image de la matrice de confusion introuvable.")


# ============================================================
# SECTION — IMPORTANCE DES MOTS POUR UNE CLASSE
# ============================================================
st.subheader("🧩 Importance des mots pour une classe")

importance_img_path = os.path.join(BASE_DIR, "data_streamlit", "MLsimple_Text8_graphMot_003_piscine.jpg")

if os.path.exists(importance_img_path):
    st.image(importance_img_path, caption="Modèle choisi (Linear SVC - essai 3) - Importance des mots pour une classe", use_container_width=True)
else:
    st.error("Image 'MLsimple_Text8_graphMot_003_piscine.jpg' introuvable dans data_streamlit.")








