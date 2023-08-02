import sys

#/Users/valerijamadzoska/Desktop/human-centeredXAI/src
#C:/Users/icke/Desktop/human-centeredXAI-1/src
sys.path.insert(0, '/Users/valerijamadzoska/Desktop/human-centeredXAI/src')

from matplotlib import pyplot as plt
from explanation_handler import ExplanationHandler
from image_handler import ImageHandler
from model_handler import ModelHandler


if __name__ == "__main__":
    model_handler = ModelHandler()
    image_handler = ImageHandler()
    explanation_handler = ExplanationHandler(model_handler.model)
    model = model_handler.load_pretrained_model()
    #/Users/valerijamadzoska/Desktop/bilderBA/rooster-1867562_640.jpg
    #C:\\Users\\icke\\Downloads\\rooster-1867562_1280.jpg
    image_path = "/Users/valerijamadzoska/Desktop/bilderBA/rooster-1867562_640.jpg"
    input_tensor = image_handler.preprocess_image(image_path)
    predicted_label = model_handler.predict_label(model, input_tensor)

    print('Predicted label:', predicted_label)
    explanation_handler.explain(model, input_tensor, image_path, "vgg", segmentation_threshold=0.2)

    plt.show()