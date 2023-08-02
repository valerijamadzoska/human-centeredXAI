import torchvision.models as models
import torch

class ModelHandler:
    def __init__(self):
        self.model = self.load_pretrained_model()
        self.model.eval()

    def load_pretrained_model(self):
        return models.vgg16(pretrained=True).eval()

    def predict_label(self,model,input_tensor):
        with torch.no_grad():
            output = model(input_tensor)
        with open('data/imagenet_labels.txt') as f:
            labels = [line.strip() for line in f.readlines()]
        _, predicted_idx = torch.max(output, 1)
        return labels[predicted_idx.item()]
