import copy
#import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import torch

import sys
sys.path.append('/Users/valerijamadzoska/Library/Python/3.8/lib/python/site-packages')

import cv2

"""
Model code and utility functions downloaded from   :
    https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/tree/main
Adjustments to the code are marked
"""
# --------------------------------------
# Visualizing data
# --------------------------------------
# modified for easier saving
def heatmap(R, sx, sy, img, name, save=True):
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.bwr(np.arange(plt.cm.bwr.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')

    # modified
    if save:
        plt.imsave(name + ".jpg", R, cmap=my_cmap, vmin=-b, vmax=b)

    img = img.squeeze()
    img = torch.permute(img, [1, 2, 0])
    img = np.matmul(img[..., :3], [0.299, 0.587, 0.114])
    img = img[:, :, np.newaxis]

    # Transpose heatmap_img if necessary
    if R.shape[0] == 1:
        R = np.transpose(R, (1, 2, 0))

    plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b)
    cbar = plt.colorbar(orientation="horizontal", shrink=0.75, ticks=[-1, 0, 1])
    plt.imshow(img, cmap='gray', interpolation='None', alpha=0.15)
    cbar.ax.set_xticklabels(["least relevant", "", "most relevant"])
    plt.show()

    fig = plt.gcf()
    plt.close()

    return fig


# --------------------------------------
# Visualizing data
# --------------------------------------
# modified
def heatmap_with_segmentation(heatmap_img, binary_mask, sx, sy, img, name, threshold=0.5, save=True):
    b = 10 * ((np.abs(heatmap_img) ** 3.0).mean() ** (1.0 / 3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.bwr(np.arange(plt.cm.bwr.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')

    # Create a binary mask based on the threshold
    binary_mask = np.abs(heatmap_img) >= threshold
    binary_mask = binary_mask.squeeze()
    binary_mask = np.stack((binary_mask, binary_mask, binary_mask), axis=-1)
   

    # Convert binary_mask to a NumPy array
    binary_mask_np = binary_mask.cpu().numpy()


    # Overlay the segmentation mask on the original image
    segmented_img = copy.deepcopy(img)
    binary_mask = torch.Tensor(binary_mask).type_as(segmented_img)

    # Convert the list [0, 0, 0] to a NumPy array
    black_pixel = np.array([0, 0, 0], dtype=np.uint8)

    # Convert segmented_img to a NumPy array
   # segmented_img = segmented_img.numpy()
    # Convert segmented_img back to a PyTorch tensor
   # segmented_img = torch.from_numpy(segmented_img).type_as(segmented_img)


       # Expand binary_mask to match the shape of segmented_img
    #binary_mask_expanded = np.expand_dims(binary_mask, axis=0)
    #binary_mask_expanded = np.repeat(binary_mask_expanded, segmented_img.shape[0], axis=0)
   # binary_mask_expanded = np.repeat(binary_mask_expanded, segmented_img.shape[1], axis=1)
   # binary_mask_expanded = np.repeat(binary_mask_expanded, segmented_img.shape[2], axis=2)

     # Resize binary mask to match the shape of segmented_img
    binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), (segmented_img.shape[2], segmented_img.shape[1]))

    # Convert binary_mask_resized to a PyTorch tensor
    binary_mask_resized = torch.Tensor(binary_mask_resized).type_as(segmented_img)

    # Convert binary_mask to a PyTorch tensor and expand dimensions to match segmented_img
    binary_mask_expanded = torch.Tensor(binary_mask).type_as(segmented_img).unsqueeze(0).unsqueeze(0)

    # Convert non_segmented_mask to a PyTorch tensor and expand dimensions to match segmented_img
    non_segmented_mask = ~binary_mask_expanded.byte()  # Convert to a boolean mask
    non_segmented_mask = non_segmented_mask.type_as(segmented_img).expand_as(segmented_img)

    # Set non-segmented regions to black
    segmented_img[~binary_mask_np] = black_pixel

   

    if save:
        # Save the segmented image
        plt.imsave(name, np.transpose(segmented_img, (1, 2, 0)))  # Transpose for correct image format
    else:
        # Display the plot with the segmented image overlay
        plt.imshow(np.transpose(heatmap_img, (1, 2, 0)), cmap='coolwarm', vmin=-b, vmax=b)
        plt.imshow(np.transpose(segmented_img, (1, 2, 0)), alpha=0.5)  # Reshape to (224, 224, 3) for display
        plt.axis('off')
        plt.show()  # Corrected to plt.show()

    # plt.close()
    fig = plt.gcf()
    plt.close()

    return fig



# --------------------------------------------------------------
# Clone a layer and pass its parameters through the function g
# --------------------------------------------------------------

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


# --------------------------------------------------------------
# convert VGG classifier's dense layers to convolutional layers
# --------------------------------------------------------------

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
                if i == 0:  # 0 for vgg and 1 for alex
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
    """
    :param picture: at the moment string to picture location, can be changed to the picture itself
    :param model: the model to use, not the name the whole model itself
    :param model_str: name of the model we use
    :param save: if we want to save the results or not
    :return: None
    """
    # modified
    results_path = 'results/LRP/'
    os.makedirs(results_path, exist_ok=True)  # Erstellt den Ordner, wenn er nicht existiert
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

    # for i in ind[:10]:
    #    print('%20s (%3d): %6.3f' % (model.labels[i], i, scores[i]))

    topClass = ind[0]
    T = torch.FloatTensor((1.0 * (np.arange(1000) == topClass).reshape([1, 1000, 1, 1])))  # mask of output class
    R = [None] * L + [(A[-1] * T).data]  # Relevance list, with las being T
    for l in range(1, L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l], torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):
            # roh rules
            if l <= 16:       rho = lambda p: p + 0.25 * p.clamp(min=0); incr = lambda z: z + 1e-9
            if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z + 1e-9 + 0.25 * (
                    (z ** 2).mean() ** .5).data
            if l >= 31:       rho = lambda p: p;                       incr = lambda z: z + 1e-9

            # lRP math
            z = incr(newlayer(layers[l], rho).forward(A[l]))  # step 1

            s = (R[l + 1] / z).data  # step 2
            (z * s).sum().backward();
            c = A[l].grad  # step 3
            R[l] = (A[l] * c).data  # step 4

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
            heatmap(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5, name, save)

        elif False : # we dont need to see each layer
            heatmap_(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)


    A[0] = A[0].data.requires_grad_(True)

    lb = (A[0].data * 0 + (0 - mean) / std).requires_grad_(True)
    hb = (A[0].data * 0 + (1 - mean) / std).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9  # step 1 (a)
    z -= newlayer(layers[0], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
    z -= newlayer(layers[0], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
    s = (R[1] / z).data  # step 2
    (z * s).sum().backward()
    c, cp, cm = A[0].grad, lb.grad, hb.grad  # step 3
    R[0] = (A[0] * c + lb * cp + hb * cm).data

    # Compute the binary mask based on the threshold
    relevance_map = np.array(R[0][0]).sum(axis=0)
    binary_mask = relevance_map >= segmentation_threshold
    # modified
    if save:
        if model_str == "alexnet":

            return heatmap_with_segmentation(np.array(R[0][0]).sum(axis=0), binary_mask, 3.5, 3.5, img, name, threshold=segmentation_threshold, save=True)
        else:

            return heatmap_with_segmentation(np.array(R[0][0]).sum(axis=0), binary_mask, 3.5, 3.5, img, name, threshold=segmentation_threshold, save=False)


