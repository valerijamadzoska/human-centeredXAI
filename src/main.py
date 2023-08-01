from src.model_handler import ModelHandler
from src.image_handler import ImageHandler
from src.explanation_handler import ExplanationHandler

def main():
    # Initialize handlers
    model_handler = ModelHandler()
    image_handler = ImageHandler()
    explanation_handler = ExplanationHandler(model_handler.model)

    # Path to the input image
    image_path = "data/images/your_image.jpg"

    # Preprocess the image
    input_tensor = image_handler.preprocess_image(image_path)

    # Predict the label
    predicted_label = model_handler.predict_label(input_tensor)
    print('Predicted label:', predicted_label)

    # Generate and visualize the explanation
    explanation_handler.explain(input_tensor, image_path, "vgg", save=False, segmentation_threshold=0.2)

if __name__ == "__main__":
    main()
