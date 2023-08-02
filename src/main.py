import sys

sys.path.insert(0, 'C:/Users/icke/Desktop/human-centeredXAI-1/src')

from matplotlib import pyplot as plt

from explanation_handler import ExplanationHandler
from image_handler import ImageHandler
from model_handler import ModelHandler


#def main():
    ##HIER IST WIEDER KI JIZZ DER SPÄTER ANGEPASST WERDEN MUSS
    # # Initialize handlers
    # model_handler = ModelHandler()
    # image_handler = ImageHandler()
    # explanation_handler = ExplanationHandler(model_handler.model)

    # # Path to the input image
    # image_path = "data/images/your_image.jpg"

    # # Preprocess the image
    # input_tensor = image_handler.preprocess_image(image_path)

    # # Predict the label
    # predicted_label = model_handler.predict_label(input_tensor)
    # print('Predicted label:', predicted_label)

    # # Generate and visualize the explanation
    # explanation_handler.explain(input_tensor, image_path, "vgg", save=False, segmentation_threshold=0.2)


    

if __name__ == "__main__":
    model_handler = ModelHandler()
    image_handler = ImageHandler()
    explanation_handler = ExplanationHandler(model_handler.model)
    model = model_handler.load_pretrained_model()
    image_path = "C:\\Users\\icke\\Downloads\\rooster-1867562_1280.jpg"
    input_tensor = image_handler.preprocess_image(image_path)
    predicted_label = model_handler.predict_label(model, input_tensor)

    print('Predicted label:', predicted_label)
    explanation_handler.explain(model, input_tensor, image_path, "vgg", save=False, segmentation_threshold=0.2)
    #heatmap_figure = explanation_handler.explain(model, input_tensor, image_path, "vgg", save=False, segmentation_threshold=0.2)

    plt.show()
