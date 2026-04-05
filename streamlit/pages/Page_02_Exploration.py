# -*- coding: utf-8 -*-
"""
Created on 

@author: Christel 
"""

import streamlit as st

st.title( "Exploration des données" )
 

col1, col2, col3 = st.columns(3)
col1.metric("Produits", "85K")
col2.metric("Catégories", "27")
col3.metric("Images", "1 / produit")

col4, col5, col6 = st.columns(3)
col4.metric("Doublons titre", "3 %")
col5.metric("Doublons description", "14 %")
col6.metric("Descriptions manquantes", "35 %")



st.markdown("---")

st.subheader( "Descriptions manquantes" )

image = './images/Explo_Nan.png'
st.image( image, caption = " " )



st.markdown("---")

st.subheader( "Exemples" )

col_left, col_right = st.columns([1, 1])

with col_left:
    st.image('./images/Explo_nuage_mots_piscine.png', 
             caption="Classe la plus fréquente")
    st.image('./images/Explo_nuage_mots_jeu_figurine.png', 
             caption="Classe la plus rare")

with col_right:
    st.image('./images/Explo_images.png', 
             caption="Images produit")
    



st.markdown("---")

st.subheader( "Déséquilibre des classes" )

image = './images/Explo_desequilibre.png'
st.image( image, caption = " Distribution de la variable cible " )



