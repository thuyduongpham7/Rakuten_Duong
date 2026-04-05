import streamlit as st



st.header( " Synthèse des résultats (F1 score)" )

col1, col2 = st.columns(2)
col1.metric("ML text", "0,81")
col2.metric("ML images", "0,40")

col3, col4 = st.columns(2)
col3.metric("DL text", "0,72")
col4.metric("DL images", "0,52")

 

# Définition des slides
slides = [
    {
        "kind": "text",
        "title": "Conclusions et retours d'expériences",
        "content": [
            "• Projet d'apparence simple mais en réalité complexe (déséquilibre, multilingues, catégories similaires)",
            "• Texte : Deep learning pas forcément nécessaire",
            "• Organisation du projet : parallélisation des tâches",
            "• Respecter les délais : limiter les tests, temps/ressources (CPU/GPU)",
            "• Approfondissement des sujets acquis pendant la formation"
        ]
    },
    {
        "kind": "text",
        "title": "Points d'attention",
        "content": [
            "• Appronfondir le preprocessing (nettoyage doublons/NA, textes multilingues, similitude des classes)",
            "• Valider la pipeline de base (commencer par un jeu de données réduit)",
            "• Bien étudier les paramètres et les hyper-paramètres",
            "• Utilisation de GPU pour le deep learning, attention à la mémoire",
            "• Sauvegarder souvent et pendant l'apprentissage en DL",
        ]
    },
    {
        "kind": "text",
        "title": "Pistes d'améliorations",
        "content": [
            "• Preprocessing : nettoyage, traiter les symptômes de déséquilibre",
            "• Texte : explorer les descriptions, fine-tuning, analyser les cas problématiques",
            "• Images : ConvNeXt, Vision Transformers, CLIP, LoRA, Adapters",
            "• Stratégie : multi-modal pipeline, stacking, voting)",
            "• LLM multi-modaux pour les cas difficiles"
        ]
    },
    {
        "kind": "text",
        "title": "What's next?",
        "content": [
            "• Affiner le besoin du client, définir les critères",
            "• Évolution du modèle",
            "• Industrialisation"
        ]
    },
    {
        "kind": "text",
        "title": "Remerciements",
        "content": [
            "• Professeurs pour les Master Classes",
            "• Kylian pour sa patience, ses conseils et son expertise",
            "• Public pour l'écoute",
        ]
    }
]

# Affichage de tous les slides sur une seule page
for slide in slides:
    st.divider()
    st.header(slide["title"])
    if slide["kind"] == "text":
        for line in slide["content"]:
            st.write(line)
    else:
        for img in slide["images"]:
            st.image(img, width="stretch")
        for line in slide["content"]:
            st.write(line)
