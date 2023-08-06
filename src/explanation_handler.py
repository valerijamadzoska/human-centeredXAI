import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
import copy
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist
import networkx as nx
import numpy.ma as ma


class ExplanationHandler:
    def __init__(self, model):
        self.model = model

    def create_cmap(self):
        my_cmap = plt.cm.bwr(np.arange(plt.cm.bwr.N))
        my_cmap[:, 0:3] *= 0.85
        return ListedColormap(my_cmap)

    def heatmap(self, R, sx, sy, img, name=None, save=False):
        #zunächst Originalbild kopieren und für imshow vorbereiten
        image_copy = np.copy(img)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])  
        image_copy = img.squeeze().numpy().transpose(1,2,0) * std + mean 
        image_copy = (image_copy * 255).astype(np.uint8) # Convert to 8-bit values

        my_cmap = self.create_cmap()
        b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))

        # R<0 soll nicht angezeigt werden (die blauen Pixel)
        R[R < 0] = 0

        plt.figure(figsize=(sx, sy))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')

        plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b) 
        plt.imshow(image_copy, alpha=0.15) #Originalbild mit Transparenz

        plt.show()
    
    def segmentation(self, heatmap_data, original_img):
        """
        Visualizes the relevance map of LRP as segments on top of the original image.
        :param heatmap_data: Heatmap data (relevance map) from the LRP.
        :param original_img: Original image on which the relevance map is overlayed.
        :return: Image with overlayed relevance segments.
        """

        # Convert heatmap_data to 8-bit grayscale
        heatmap_8bit = (heatmap_data * 255).astype(np.uint8)

        # Apply a simple binarization to segment the relevance 
        _, binary_map = cv2.threshold(heatmap_8bit, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the binary map
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Overlay the contours on the original image
        img_with_contours = original_img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 1)  # Draw contours with green lines

        return img_with_contours

    def graph(self, relevance_map, img, threshold=0.0007):
        # mean and standard deviation values used for image preprocessing
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225])  

        G = nx.Graph()
        
        height, width = relevance_map.shape
        for y in range(height):
            for x in range(width):
                for dy, dx in [(0, 1), (1, 0)]:
                    if 0 <= y + dy < height and 0 <= x + dx < width:
                        node1 = y * width + x
                        node2 = (y + dy) * width + (x + dx)
                        weight = abs(relevance_map[y, x] - relevance_map[y + dy, x + dx])
                        if weight > threshold:
                            G.add_edge(node1, node2, weight=weight)

        pos = {y * width + x: (x, -y) for y in range(height) for x in range(width)}
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

        plt.figure(figsize=(5, 5)) # figure size

        # Convert the tensor to a numpy array and de-normalize
        img = img.squeeze().numpy().transpose(1,2,0) * std + mean 
        img = (img * 255).astype(np.uint8) # Convert to 8-bit values

        plt.imshow(img, extent=[0, width, -height, 0], alpha=0.3) # alpha=transparency

        nx.draw(G, pos, with_labels=False, node_size=10, edge_color=edge_weights, edge_cmap=plt.cm.Blues)
        
        plt.show()

    def newlayer(self, layer, g):
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

    def toconv(self, layers, model):
        newlayers = []
        for i, layer in enumerate(layers):
            if isinstance(layer, torch.nn.Linear):
                if model == "vgg" and i == 0:
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
    
    def explain(self, model, img, file, model_str, save=True):
        """
        :param picture: at the moment string to picture location, can be changed to the picture itself
        :param model: the model to use, not the name the whole model itself
        :param model_str: name of the model we use
        :param save: if we want to save the results or not
        :return: None
        """
        X = copy.deepcopy(img)

        mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        layers = list(model._modules['features']) + self.toconv(list(model._modules['classifier']), model_str)
        L = len(layers)

        A = [X] + [None] * L
        for l in range(L):
            A[l + 1] = layers[l].forward(A[l])

        scores = np.array(A[-1].data.view(-1))
        ind = np.argsort(-scores)

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
                z = incr(self.newlayer(layers[l], rho).forward(A[l]))  # step 1

                s = (R[l + 1] / z).data  # step 2
                (z * s).sum().backward();
                c = A[l].grad  # step 3
                R[l] = (A[l] * c).data  # step 4

            else:

                R[l] = R[l + 1]
        
        A[0] = A[0].data.requires_grad_(True)

        lb = (A[0].data * 0 + (0 - mean) / std).requires_grad_(True)
        hb = (A[0].data * 0 + (1 - mean) / std).requires_grad_(True)

        z = layers[0].forward(A[0]) + 1e-9  # step 1 (a)
        z -= self.newlayer(layers[0], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
        z -= self.newlayer(layers[0], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
        s = (R[1] / z).data  # step 2
        (z * s).sum().backward()
        c, cp, cm = A[0].grad, lb.grad, hb.grad  # step 3
        R[0] = (A[0] * c + lb * cp + hb * cm).data

        #HEATMAP
        heatmap_fig = self.heatmap(np.array(R[l][0]).sum(axis=0), 3.5, 3.5, img)
        print(heatmap_fig)
        #SEGMENTATION
        original_img_np = img.squeeze().numpy().transpose(1,2,0)
        original_img_np = (original_img_np * 255).astype(np.uint8)
        segmented_img = self.segmentation(np.array(R[l][0]).sum(axis=0), original_img_np)
        plt.imshow(segmented_img)
        plt.axis('off')
        plt.show()
        #GRAPH
        relevance_map = np.array(R[l][0]).sum(axis=0)
        #print("Min value:", relevance_map.min())
        #print("Max value:", relevance_map.max())
        graph_img = self.graph(relevance_map, img)

