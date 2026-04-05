# Rakuten_Duong
👤 Auteur
Projet réalisé par Thuy Duong PHAM, en collaboration avec Christel BON et Laurent FAN

📁 Structure du projet
├── models/            # Modèles entraînés
├── notebooks/         # Analyses exploratoires et expérimentations
├── reports/           # Rapport du projet
├── streamlit          # App streamlit pour visualiser/expliquer le projet
└── README.md

🎯 Objectifs
Prédire la catégorie (code produit) d’un produit à partir des textes et images

Comparer différentes approches :
  Machine Learning classique
  Deep Learning
  
Optimiser les performances du modèle

📊 Données

Dataset : Rakuten 
Type de données :
  Texte (titres, descriptions)
  Images 
  
Nombre d’observations : 85K
Nombre de classes : 27

⚙️ Méthodologie

🔹 Préprocessing

Nettoyage du texte (suppression des stopwords, caractères spéciaux…),
Vectorisation (TF-IDF),
Traitement des images (redimensionnement, normalisation)

🔹 Modèles utilisés

📌 Machine Learning pour textes:
Logistic Regression, Linear SVC, Naive Bayes, Random Forest, Gradient Boosting

📌 Machine Learning pour images:
Logistic Regression, Random Forest, KNN

📌 Deep Learning pour images:
CNN personnalisés

📈 Résultats
Modèle	                        F1-score
Linear SVC (pour textes)	      81%
CNN personnalisé (pour images)	44%

👉 Meilleur modèle : Linear SVC (pour textes)
👉 Score final : 81%

