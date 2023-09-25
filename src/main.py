import sys

from matplotlib import pyplot as plt
from explanation_handler import ExplanationHandler
from image_handler import ImageHandler
from model_handler import ModelHandler
import platform

def check_os():
    system = platform.system()
    if system == 'Windows':
        #C:\\Users\\icke\\Downloads\\ILSVRC2012_test_00000090.jpg
        return 'C:\\Users\\icke\\Desktop\\InputImg\\ILSVRC2012_val_00002266.jpg'
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
    explanation_handler.explain(model, input_tensor, image_path, "vgg", predicted_label)