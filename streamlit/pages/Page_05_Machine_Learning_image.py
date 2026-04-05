# -*- coding: utf-8 -*-
"""
Created on 

@author: Christel
"""

import streamlit as st
import pandas as pd
 
 

# TITRE DE LA PAGE
st.title("Machine Learning — Classification des images")


results = {
    "Logistic Regression": ["0.24"],
    "K Nearest Neighbors": ["0.33"],
    "Random Forest": ["0.40"],
    "Decision Tree": ["0.33"]
}
df_results = pd.DataFrame(results, index=["F1-score"])

st.subheader("F1-score")
st.dataframe(df_results, use_container_width=True)



st.markdown("---")

st.subheader( "Matrice de confusion (Random Forest - images)" )

image = './images/ML_img-matrice_confusion_RF_images.png'
st.image( image )




st.markdown("---")

st.subheader( "Exemples (Random Forest - images)" )

image = './images/ML_img-Success.png'
st.image( image )

image = './images/ML_img-Erreurs.png'
st.image( image )



st.markdown("---")

st.subheader( "Taux d'erreur par classe (Random Forest - images)" )

image = './images/ML_img-Barplot_Erreur_par_classe.png'
st.image( image, caption = "" )


 
