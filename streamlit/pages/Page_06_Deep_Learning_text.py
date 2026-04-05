import streamlit as st

# Définition des slides
slides = [
    {
        "kind": "text",
        "title": "Classification en Deep Learning",
        "content": [
            "• Bons résultats avec Tf-Idf et ML",
            "• Peut-on réutiliser Tf-Idf directement?",
            "• Quel modèle choisir ?",
        ]
    },
    {
        "kind": "text",
        "title": "Tf-Idf : Dimensions",
        "content": [
            "• 200 000 word tokens pour 70K lignes",
            "   ⇒ 200 000 dimensions ?",
            "• 700 000 en n-gram {2,4}",
            "• 600 000 en n-gram char {3,6}",
        ]
    },
    {
        "kind": "image",
        "title": "Tf-Idf : Matrice creuse",
        "images": ["images/tfidf_terms_histo.png"],
        "content": [
            "• 99% ont moins de 200 tokens",
            "• Nombre moyen de token par produit: 39",
            "• Bien gérée par sklearn et modèles ML (PCA intégré)",
            "• Non adapté aux réseaux de neurones : calcul lourd",
            "• Solutions:",
            "   - PCA sur Tf-Idf : sklearn TruncatedSVD",
            "   - Vectorisation : extraire le sens du texte (Word2Vec, fastText)",
            "   - Entrainement par batch : adapter le vocabulaire (BERT, RoBERTa)"
        ]
    },
    {
        "kind": "image",
        "title": "Tf-Idf n-gram mots : Macro F1 0.56",
        "images": ["images/tfidf_word_train_history.png", "images/tfidf_word_confusion.png"],
        "content": [
            " - Modèle utilisé : 2 couches Dense (256 et 128 neurones) + activation softmax"
        ]
    },
    {
        "kind": "image",
        "title": "Tf-Idf n-gram caractères Macro F1: 0.73",
        "images": ["images/tfidf_char_train_history.png", "images/tfidf_char_confusion.png"],
        "content": [
            "- Modèle utilisé : 2 couches Dense (256 et 128 neurones) + activation softmax"
        ]
    },
    {
        "kind": "image",
        "title": "Vectorisation sémantique Macro F1: 0.72",
        "images": ["images/sentense_vec_train_history.png","images/sentense_vec_confusion.png"],
        "content": [
            "- Modèle : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "- Similarité (sklearn cosine) : classification par centroïdes, F1 score ≈ 0.45"
        ]
    },
    {
        "kind": "text",
        "title": "Sentence vectorizer comparé au ML",
        "content": [
            "Sur 27 classes, le modèle vectorisé dépasse le SVC en F1-score seulement sur 4 classes",
            "La seule amélioration substantielle concerne la classe 2705 (+19 points F1)",
            "Avec plus de temps il faudrait tester d'autres modèles"
        ]
    },
    {
        "kind": "text",
        "title": "Exemples de cas difficiles",
        "content": [
            "==========10 : Livres",
            "Seigneurs Rebelles Morvan",
            "Sculpting Hillsides Decorative Concrete",
            "Abnormal Psychology Study Guide",
            "Wer Ist Alexander Grothendieck? Anarchie Mathematik Spiritualität Einsamkeit - Eine Biographie - Teil 3: Spiritualität",
            "Venu J'ai Vu N'y Crois Plus / Ba Omar / Réf54712",
            "==========2705 : Romans historiques et littérature",
            "Roman D'un Fripon",
            "Concours Cadre Santé 2017-2018",
            "P'tits Toquades - Wok",
            "Rock & Sex (Vol. 1)",
            "L'anxiété Comment S'en Sortir : Pistes Réflexions ..."
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
