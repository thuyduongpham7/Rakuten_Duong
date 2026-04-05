# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:08:34 2025

@author: Duong
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

# --- Correction des chemins ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # STREAMLIT/pages
STREAMLIT_DIR = os.path.dirname(CURRENT_DIR)                      # STREAMLIT/
PROJECT_DIR = STREAMLIT_DIR

if STREAMLIT_DIR not in sys.path:
    sys.path.append(STREAMLIT_DIR)

from modelisation.model_dl_image import (
    load_history, load_model, predict_image, generate_gradcams_for_last_n_conv_layers,
    decode_class
)

# =======================================
#   INITIALISATION SESSION STATE
# =======================================
if "preds" not in st.session_state:
    st.session_state.preds = None

if "img_path" not in st.session_state:
    st.session_state.img_path = None


# =======================================================
# PAGE STREAMLIT — DL IMAGE
# =======================================================
st.title("🧠 Deep Learning — Classification d’Images")
st.write("Page de classification + Grad-CAM.")

# =======================================================
#  🔥 PRÉDICTION EN PREMIER
# =======================================================

# -------------------------------------------------------
# SECTION 1 : Charger le modèle
# -------------------------------------------------------
st.markdown("---")
st.subheader("📌 Charger le modèle")

load_model_btn = st.checkbox("Charger le modèle maintenant")

model = None
if load_model_btn:
    try:
        model = load_model()
        st.success("Modèle chargé !")
    except Exception as e:
        st.error(f"Erreur : {e}")

# -------------------------------------------------------
# SECTION 2 : Prédiction
# -------------------------------------------------------
st.markdown("---")
st.subheader("🔮 Prédiction sur une image")

uploaded = st.file_uploader(
    "Uploader une image",
    type=["jpg", "png", "jpeg"]
)

img = None

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    temp_path = "temp_uploaded_image.png"
    img.save(temp_path)

    st.session_state.img_path = temp_path
    st.image(img, caption="Image sélectionnée", width=200)

    if st.button("🔍 Lancer la prédiction"):
        if not load_model_btn:
            st.warning("Chargez d’abord le modèle.")
        else:
            try:
                preds = predict_image(model, st.session_state.img_path).flatten()
                st.session_state.preds = preds
                st.success("Prédiction effectuée !")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

# -------------------------------------------------------
# AFFICHAGE DES PRÉDICTIONS
# -------------------------------------------------------
if st.session_state.preds is not None:
    preds = st.session_state.preds

    st.write("### 🔢 Probabilités (top 5)")
    top5_idx = preds.argsort()[-5:][::-1]

    for i in top5_idx:
        code, cat = decode_class(i)
        st.write(f"• Code **{code}** — *{cat}* : {preds[i]:.4f}")

    # ------------------------------------------
    # SECTION Grad-CAM
    # ------------------------------------------
    st.write("### 🔥 Heatmaps Grad-CAM")

    show_gradcam = st.checkbox("Afficher les 3 heatmaps Grad-CAM (top1)")

    if show_gradcam:
        if model is None:
            st.warning("Veuillez d'abord charger le modèle.")
        else:
            try:
                top1 = int(np.argmax(preds))
                code_top1, cat_top1 = decode_class(top1)
                st.write(f"Classe prédite : Code **{code_top1}** — *{cat_top1}*")

                heatmaps, layer_names = generate_gradcams_for_last_n_conv_layers(
                    model=model,
                    img_path=st.session_state.img_path,
                    class_index=top1,
                    target_size=(128, 128),
                    n_layers=3
                )

                n = len(heatmaps)
                fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

                if n == 1:
                    axes = [axes]

                for ax, hm, name in zip(axes, heatmaps, layer_names):
                    ax.imshow(hm, cmap="jet")
                    ax.set_title(name)
                    ax.axis("off")

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Erreur lors de la génération des heatmaps : {e}")


# =======================================================
# SECTION : CRÉATION DES MODÈLES
# =======================================================
st.markdown("---")
st.header("🏗️ Création des modèles")

# --- 1. LES ÉTAPES ---
st.subheader("🔹 Les étapes")
st.write("""
1. Chargement et préparation des données  
2. Prétraitement des images  
3. Tests de modèles CNN personnalisés  
4. Tests de modèles CNN avec Transfer Learning  
5. Analyse des performances  
""")

# --- 2. PRÉTRAITEMENT ---
st.subheader("🖼️ Prétraitement des images")

before_path = os.path.join(PROJECT_DIR, "data_streamlit", "ML_img5_002_AvantPretraitement.jpg")
after_path = os.path.join(PROJECT_DIR, "data_streamlit", "ML_img5_003_ApresPretraitement.jpg")

col1, col2 = st.columns(2)
with col1:
    st.image(before_path, caption="Avant le prétraitement", use_container_width=True)
with col2:
    st.image(after_path, caption="Après le prétraitement", use_container_width=True)

# --- 3. MODÈLES TESTÉS ---
st.subheader("🧪 Les modèles testés")
st.write("""
• **≈40 essais** sur **≈12 jours**  
• **Modèles CNN personnalisés**  
• **Transfer Learning** : EfficientNetB0 , MobileNetV2 , ResNet50 , 
Vision Transformers , ConvNeXt , etc.
""")

# --- 4. RÉSULTATS ---
st.subheader("📊 Résultats")
st.write("""
- **CNN personnalisé** → *F1-score weighted* = **0,4427**  
- **MobileNetV2 (Transfer Learning)** → *F1-score weighted* = **0,5166**  
""")

# --- 5. COURBES LOSS + ACCURACY ---
st.subheader("📈 Courbes Loss et Accuracy")

cnn_loss_acc = os.path.join(PROJECT_DIR, "data_streamlit", "DL_CNN_img_53_002.jpg")
mobilenet_loss_acc = os.path.join(PROJECT_DIR, "data_streamlit", "Laurent_DL_CNN_MobileNetV2_004.png")

col3, col4 = st.columns(2)
with col3:
    st.image(cnn_loss_acc, caption="CNN personnalisé — Loss & Accuracy", use_container_width=True)
with col4:
    st.image(mobilenet_loss_acc, caption="MobileNetV2 — Loss & Accuracy", use_container_width=True)

# --- 6. MATRICE DE CONFUSION ---
st.subheader("🧩 Matrice de confusion")

matrix_path = os.path.join(PROJECT_DIR, "data_streamlit", "DL_CNN_img_53_006_MatriceConfusion_pourcent.jpg")
st.image(matrix_path, caption="Matrice de confusion (en proportion)", use_container_width=True)
