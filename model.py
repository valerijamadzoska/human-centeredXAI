import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from LRP_heatmap_segmentation import explain
import matplotlib.pyplot as plt

# Load the pre-trained VGG16 model
#https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
import LRP_heatmap
import LRP_heatmap_segmentation

model = models.vgg16(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Load and preprocess the image
image_path = '//Users/valerijamadzoska/Desktop/bilderBA/rooster-1867562_640.jpg'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)

# Perform the forward pass
with torch.no_grad():
    output = model(input_tensor)

# Load the labels for ImageNet classes
with open('imagenet_labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]
#print('Number of labels:', len(labels))
#print('Labels:', labels)


# Get the predicted class
_, predicted_idx = torch.max(output, 1)
predicted_label = labels[predicted_idx.item()]

# Print the predicted class label
print('Predicted label:', predicted_label)

heatmap_figure = explain(model, input_tensor, image_path, "vgg", save=True, segmentation_threshold=0.2)
print('Ende?')
# If you want to display the heatmap with segmentation overlay
plt.show()

# If you want to save the figure, you can use:
# heatmap_figure.savefig("path/to/save/heatmap_segmented.jpg")
#LRP_heatmap.explain(model=model, img=input_tensor, file=image_path, model_str="vgg", save=True)
#LRP_heatmap_segmentation.explain(model, input_tensor, image_path, "vgg", save=True, segmentation_threshold=0.2)