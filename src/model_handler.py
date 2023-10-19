import torchvision.models as models
import torch

class ModelHandler:
    """
    A handler for loading a pretrained model and making predictions.
    """
    def __init__(self):
        #Initializes the ModelHandler by loading a pretrained model and setting it to evaluation mode.
        self.model = self.load_pretrained_model()
        self.model.eval()

    #Loads a pretrained VGG19 model.
    def load_pretrained_model(self):
        return models.vgg19(pretrained=True).eval()

    def predict_label(self,model,input_tensor):
        """
        Predicts the label of an input tensor using a pretrained model.

        Args:
            model (torch.nn.Module): The pretrained model used for prediction.
            input_tensor (torch.Tensor): The input tensor to predict the label for.

        Returns:
            str: The label of the highest prediction.
        """
        with torch.no_grad():
            output = model(input_tensor)
        with open('data/imagenet_labels.txt') as f:
            labels = [line.strip() for line in f.readlines()]
        _, predicted_idx = torch.max(output, 1)
        return labels[predicted_idx.item()]
