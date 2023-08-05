import numpy as np
import tensorflow as tf
import lrp_toolbox as LRP
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Funktion zum Laden und Vorbereiten des Bildes
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)
    return processed_img, img_array

# Lade das VGG19-Modell (mit der vollständig verbundenen Schicht)
model = VGG19(weights='imagenet')

# Pfade zu den Bilddateien
image_path_1 = 'pfad/zum/bild1.jpg'
image_path_2 = 'pfad/zum/bild2.jpg'

# Lade und verarbeite die Bilder
processed_image_1, original_image_1 = load_and_preprocess_image(image_path_1)
processed_image_2, original_image_2 = load_and_preprocess_image(image_path_2)

# Klassifiziere das erste Bild
predictions_1 = model.predict(processed_image_1)
decoded_predictions_1 = decode_predictions(predictions_1, top=1)[0]

# Klassifiziere das zweite Bild
predictions_2 = model.predict(processed_image_2)
decoded_predictions_2 = decode_predictions(predictions_2, top=1)[0]

# LRP auf das erste Bild anwenden
lrp_1 = LRP(model)
relevance_1 = lrp_1.lrp(original_image_1, decoded_predictions_1[0][1])

# LRP auf das zweite Bild anwenden
lrp_2 = LRP(model)
relevance_2 = lrp_2.lrp(original_image_2, decoded_predictions_2[0][1])

# Funktion zur Normalisierung der Relevanzen für die Heatmap-Darstellung
def normalize_relevance(relevance):
    min_val = np.min(relevance)
    max_val = np.max(relevance)
    normalized_relevance = (relevance - min_val) / (max_val - min_val)
    return normalized_relevance

# Funktion zur Darstellung der Heatmap
def plot_heatmap(image, relevance, prediction, title):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image[0].astype(np.uint8))
    plt.axis('off')
    plt.title(f"Prediction: {prediction[1]} ({prediction[2]:.4f})")

    plt.subplot(1, 2, 2)
    normalized_relevance = normalize_relevance(relevance[0])
    plt.imshow(normalized_relevance, cmap='hot', alpha=0.8)
    plt.imshow(image[0].astype(np.uint8), cmap='gray', alpha=0.5)
    plt.axis('off')
    plt.title(title)
    plt.show()

# Visualisiere die Heatmaps
plot_heatmap(original_image_1, relevance_1, decoded_predictions_1[0], "LRP Heatmap - Bild 1")
plot_heatmap(original_image_2, relevance_2, decoded_predictions_2[0], "LRP Heatmap - Bild 2")
