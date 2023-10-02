import sys

from matplotlib import pyplot as plt
from explanation_handler import ExplanationHandler
from image_handler import ImageHandler
from model_handler import ModelHandler
import platform

def check_os():
    system = platform.system()
    if system == 'Windows':
        return 'C:\\Users\\icke\\Desktop\\img\\448\\b8286429e9245e0dfe7b9bcaa5200df2de0499de.jpeg'
    elif system == 'Darwin':
        return '/Users/valerijamadzoska/Desktop/bilderBA/rooster-1867562_640.jpg'
    else:
        return 'Unknown'

if __name__ == "__main__":
    model_handler = ModelHandler()
    image_handler = ImageHandler()
    explanation_handler = ExplanationHandler(model_handler.model)
    model = model_handler.load_pretrained_model()
    image_path = check_os()
    input_tensor = image_handler.preprocess_image(image_path)
    predicted_label = model_handler.predict_label(model, input_tensor)

    print('Predicted label:', predicted_label)

    user_input = input('Prediction true or false?: ')
    print('You entered:', user_input)

    explanation_handler.explain(model, input_tensor, image_path, "vgg", predicted_label, user_input)
