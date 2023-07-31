import os
import copy
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import importlib.util
from matplotlib.colors import ListedColormap

# Loading and Preprocessing
def load_pretrained_model():
    model = models.vgg16(pretrained=True)
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def predict_label(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    with open('imagenet_labels.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    _, predicted_idx = torch.max(output, 1)
    return labels[predicted_idx.item()]

# Explanation and Visualization
def create_cmap():
    my_cmap = plt.cm.bwr(np.arange(plt.cm.bwr.N))
    my_cmap[:, 0:3] *= 0.85
    return ListedColormap(my_cmap)

def normalize_img(img, b):
    return np.clip((img + b) / (2 * b), 0, 1)

def heatmap(R, sx, sy, img, name, save=True):
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))

    my_cmap = create_cmap()

    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')

    if R.shape[0] == 1:
        R = np.transpose(R, (1, 2, 0))

    heatmap_img_resized = cv2.resize(R, (img.shape[1], img.shape[0]))
    heatmap_img_resized_normalized = normalize_img(heatmap_img_resized, b)

    plt.imshow(heatmap_img_resized_normalized, cmap=my_cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(orientation="horizontal", shrink=0.75, ticks=[-1, 0, 1])
    plt.imshow(img, cmap='gray', interpolation='None', alpha=0.15)
    cbar.ax.set_xticklabels(["least relevant", "", "most relevant"])

    if save:
        os.makedirs('results', exist_ok=True)
        plt.imsave(os.path.join('results', name + "_heatmap.jpg"), heatmap_img_resized, cmap=my_cmap, vmin=-b, vmax=b)

    #plt.show()

    return

def heatmap_with_segmentation(heatmap_img, binary_mask, sx, sy, img, name, threshold=0.5, save=True):
    b = 10 * ((np.abs(heatmap_img) ** 3.0).mean() ** (1.0 / 3))
    black_pixel = (0, 0, 0)  # Represents the RGB values for black (0, 0, 0)

    my_cmap = create_cmap()

    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')

    heatmap_img_resized = cv2.resize(heatmap_img, (img.shape[2], img.shape[3]))

    if heatmap_img_resized.ndim == 3:
        heatmap_img_resized = np.transpose(heatmap_img_resized, (1, 2, 0))
    elif heatmap_img_resized.ndim == 2:
        heatmap_img_resized = np.expand_dims(heatmap_img_resized, axis=2)
        heatmap_img_resized = np.repeat(heatmap_img_resized, 3, axis=2)
    else:
        pass

    heatmap_img_resized_normalized = normalize_img(heatmap_img_resized, b)

    plt.imshow(heatmap_img_resized, cmap=my_cmap, vmin=-b, vmax=b)
    cbar = plt.colorbar(orientation="horizontal", shrink=0.75, ticks=[-1, 0, 1])

    img_np = img.squeeze().cpu().numpy()
    img_for_overlay = np.transpose(img_np, (1, 2, 0))

    plt.imshow(heatmap_img_resized_normalized, cmap=my_cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(orientation="horizontal", shrink=0.75, ticks=[-1, 0, 1])

    binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), (img.shape[1], img.shape[0]))
    binary_mask_expanded = np.expand_dims(binary_mask_resized, axis=2)
    binary_mask_expanded = np.repeat(binary_mask_expanded, 3, axis=2)

    segmented_img = copy.deepcopy(img_for_overlay)

    binary_mask_expanded = np.tile(binary_mask[:, :, np.newaxis], (1, 1, segmented_img.shape[2]))
    segmented_img[binary_mask_expanded[:, :, 0] == 0] = black_pixel

    segmented_img_normalized = np.clip(segmented_img / 255.0, 0, 1)

    plt.imshow(heatmap_img_resized_normalized, cmap='coolwarm', vmin=0, vmax=1)
    plt.imshow(segmented_img_normalized * 255, alpha=0.5, vmin=0, vmax=255)
    plt.axis('off')

    if save:
        os.makedirs('results', exist_ok=True)
        img_min = segmented_img.min()
        img_max = segmented_img.max()
        if img_min != img_max:
            segmented_img_normalized = (segmented_img - img_min) / (img_max - img_min)
        else:
            segmented_img_normalized = np.zeros_like(segmented_img)

        segmented_img_normalized = (segmented_img_normalized * 255).astype(np.uint8)
        plt.imsave(os.path.join('results', name + "_segmented.jpg"), segmented_img_normalized)

    plt.show()
    plt.close()

    return

# Converting Layers and Explanation
def newlayer(layer, g):
    layer = copy.deepcopy(layer)

    try:
        layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError:
        pass

    try:
        layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError:
        pass

    return layer

def toconv(layers, model):
    newlayers = []
    for i, layer in enumerate(layers):

        if isinstance(layer, torch.nn.Linear):
            newlayer = None
            if model == "alexnet":
                if i == 1:
                    m, n = 256, layer.weight.shape[0]
                    newlayer = torch.nn.Conv2d(m, n, 6)
                    newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n, m, 6, 6))
                else:
                    m, n = layer.weight.shape[1], layer.weight.shape[0]
                    newlayer = torch.nn.Conv2d(m, n, 1)
                    newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n, m, 1, 1))
            else:
                if i == 0:
                    m, n = 512, layer.weight.shape[0]
                    newlayer = torch.nn.Conv2d(m, n, 7)
                    newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n, m, 7, 7))
                else:
                    m, n = layer.weight.shape[1], layer.weight.shape[0]
                    newlayer = torch.nn.Conv2d(m, n, 1)
                    newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n, m, 1, 1))

            newlayer.bias = torch.nn.Parameter(layer.bias)
            newlayers += [newlayer]

        else:
            newlayers += [layer]

    return newlayers

def explain(model, img, file, model_str, save=True, segmentation_threshold=0.5):
    results_path = 'results/LRP/'
    os.makedirs(results_path, exist_ok=True)
    name = os.path.splitext(os.path.basename(file))[0] + "_" + model_str + '.jpg'
    full_path = os.path.join(results_path, name)
    X = copy.deepcopy(img)

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    layers = list(model._modules['features']) + toconv(list(model._modules['classifier']), model_str)
    L = len(layers)

    A = [X] + [None] * L
    for l in range(L):
        A[l + 1] = layers[l].forward(A[l])

    scores = np.array(A[-1].data.view(-1))
    ind = np.argsort(-scores)
    topClass = ind[0]
    T = torch.FloatTensor((1.0 * (np.arange(1000) == topClass).reshape([1, 1000, 1, 1])))

    R = [None] * L + [(A[-1] * T).data]
    for l in range(1, L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l], torch.nn.MaxPool2d):
            layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):
            if l <= 16:
                rho = lambda p: p + 0.25 * p.clamp(min=0)
                incr = lambda z: z + 1e-9
            elif 17 <= l <= 30:
                rho = lambda p: p
                incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
            elif l >= 31:
                rho = lambda p: p
                incr = lambda z: z + 1e-9

            z = incr(newlayer(layers[l], rho).forward(A[l]))
            s = (R[l + 1] / z).data
            (z * s).sum().backward()
            c = A[l].grad
            R[l] = (A[l] * c).data

        else:
            R[l] = R[l + 1]

    if model_str == "alexnet":
        layers_map = [15, 10, 7, 1]
    else:
        layers_map = [31, 21, 11, 1]

    name = os.path.splitext(file)[0]
    name = name + "_" + model_str
    for i, l in enumerate(layers_map):
        if l == layers_map[-1] and model_str == "vgg" and False:
            print('blaaaaaaaaaaaaaaa')
        elif False:
            heatmap_(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)

    A[0] = A[0].data.requires_grad_(True)

    lb = (A[0].data * 0 + (0 - mean) / std).requires_grad_(True)
    hb = (A[0].data * 0 + (1 - mean) / std).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9
    z -= newlayer(layers[0], lambda p: p.clamp(min=0)).forward(lb)
    z -= newlayer(layers[0], lambda p: p.clamp(max=0)).forward(hb)
    s = (R[1] / z).data
    (z * s).sum().backward()
    c, cp, cm = A[0].grad, lb.grad, hb.grad
    R[0] = (A[0] * c + lb * cp + hb * cm).data

    input_np = input_tensor.squeeze().cpu().numpy()
    img_for_overlay = np.transpose(input_np, (1, 2, 0))

    relevance_map = np.array(R[0][0]).sum(axis=0)
    binary_mask = relevance_map >= segmentation_threshold

    if model_str == "alexnet":
        heatmap_with_segmentation(relevance_map, binary_mask, 10, 10, img_for_overlay, name, save=save)
    else:
        heatmap(relevance_map, 10, 10, img_for_overlay, name, save=save)


if __name__ == "__main__":
    model = load_pretrained_model()
    image_path = "C:\\Users\\icke\\Downloads\\rooster-1867562_1280.jpg"
    input_tensor = preprocess_image(image_path)
    predicted_label = predict_label(model, input_tensor)

    print('Predicted label:', predicted_label)

    heatmap_figure = explain(model, input_tensor, image_path, "vgg", save=False, segmentation_threshold=0.2)

    plt.show()
