# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import streamlit as st

st.title("Challenge Rakuten")

st.subheader(" Classification de produits e-commerce")

st.write("Ce projet explore les approches Machine Learning et Deep Learning appliquées à la classification des produits d'une marketplace. ")

st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("Benchmark perf (F1 score)", "")
col2.metric("Text", "0,81")
col3.metric("Images", "0,55")

st.markdown("---")

image = './images/Intro_inventaire.png'
st.image( image, caption = "La classification, ou l'art de structurer le chaos pour éclairer les décisions" )
         

