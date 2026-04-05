# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import streamlit as st
import os
import sys

# ===========================================================
# AJOUT DU PATH GLOBAL POUR PERMETTRE LES IMPORTS INTERNES
# ===========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

# ===========================================================
# CONFIGURATION DE LA PAGE PRINCIPALE
# ===========================================================
st.set_page_config(
    page_title="Rakuten Streamlit App",
    page_icon="📦",
    layout="wide"
)

st.title(" Projet Rakuten - Datascientest")
st.write("Bienvenue dans l'application Streamlit dédiée au challenge Rakuten.")

st.write("par Thuy Duong PHAM - Laurent FAN - Christel BON ")

st.write(" décembre 2025 ") 

st.markdown("---")

st.info("Sélectionne une page dans le menu latéral pour commencer.")

 

