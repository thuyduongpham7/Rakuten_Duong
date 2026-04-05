# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:06:35 2025

@author: Duong
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

# =======================================================
# RÉPERTOIRE DU PROJET
# =======================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data_streamlit")

CSV_PATH = os.path.join(DATA_DIR, "Rakuten_imagepath.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model_59.keras")
HISTORY_PATH = os.path.join(DATA_DIR, "history_59.json")
IMAGE_FOLDER = os.path.join(DATA_DIR, "processed_128")

SEED = 42

# =======================================================
#  MAPPING CLASSES → CODE PRODUIT + CATÉGORIE
# =======================================================
CLASS_MAPPING = {
    0: (10, "livres"),
    1: (40, "import_jeux"),
    2: (50, "console_jeux"),
    3: (60, "console_portable"),
    4: (1140, "figurines"),
    5: (1160, "cartes_collection"),
    6: (1180, "jeux_plateau"),
    7: (1280, "jouets"),
    8: (1281, "jeux_enfants"),
    9: (1300, "modelisme"),
    10: (1301, "vetements_enfants"),
    11: (1302, "gadgets"),
    12: (1320, "puericulture"),
    13: (1560, "mobilier"),
    14: (1920, "literie"),
    15: (1940, "alimentation"),
    16: (2060, "deco_lumiere"),
    17: (2220, "animaux"),
    18: (2280, "presse"),
    19: (2403, "livres"),
    20: (2462, "gaming"),
    21: (2522, "papeterie"),
    22: (2582, "jardin"),
    23: (2583, "piscine"),
    24: (2585, "outils_jardin"),
    25: (2705, "roman"),
    26: (2905, "jeux_telechargement"),
}

def decode_class(index: int):
    """Retourne (code_produit, categorie) depuis un index prédictif."""
    return CLASS_MAPPING.get(index, ("?", "inconnu"))

# =======================================================
# CHARGER L’HISTORIQUE
# =======================================================
def load_history():
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"Fichier introuvable : {HISTORY_PATH}")

    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(len(acc))

    return acc, val_acc, loss, val_loss, epochs

# =======================================================
# CHARGER LE MODÈLE
# =======================================================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

# =======================================================
# PRÉDICTION
# =======================================================
def load_and_preprocess_image(img_path, target_size=(128, 128)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(model, img_path):
    img = load_and_preprocess_image(img_path)
    return model.predict(img)

# =======================================================
# GRAD-CAM (pour plusieurs couches)
# =======================================================
def _find_last_n_conv_layer_names(model, n=3):
    conv_names = []
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_names.append(layer.name)
            if len(conv_names) >= n:
                break
    return conv_names

def make_gradcam_heatmap_for_layer(model, img_array, layer_name, class_index=None):
    last_conv = model.get_layer(layer_name)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out_np = conv_out[0].numpy()
    pooled_np = pooled_grads.numpy()

    for i in range(pooled_np.shape[-1]):
        conv_out_np[:, :, i] *= pooled_np[i]

    heatmap = np.sum(conv_out_np, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap

def generate_gradcams_for_last_n_conv_layers(model, img_path, class_index=None, target_size=(128, 128), n_layers=3):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_tensor)
    if class_index is None:
        class_index = int(np.argmax(preds[0]))

    conv_names = _find_last_n_conv_layer_names(model, n=n_layers)

    heatmaps_resized = []
    layer_names = []

    for name in conv_names:
        try:
            heatmap = make_gradcam_heatmap_for_layer(
                model, img_tensor, layer_name=name, class_index=class_index
            )
            heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], target_size).numpy()
            heatmap_resized = np.squeeze(heatmap_resized)
            if np.max(heatmap_resized) != 0:
                heatmap_resized /= np.max(heatmap_resized)

            heatmaps_resized.append(heatmap_resized)
            layer_names.append(name)

        except Exception:
            continue

    return heatmaps_resized, layer_names


