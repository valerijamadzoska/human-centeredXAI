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
from skimage.transform import resize
from sklearn.cluster import KMeans



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
        return R
    
    def segmentation(self, heatmap, n_clusters=2):
        # Form des Heatmaps anpassen, um es als Eingabe für k-Means verwenden zu können
        x, y = heatmap.shape
        heatmap_reshaped = heatmap.reshape((-1, 1))
        
        # k-Means-Algorithmus anwenden
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(heatmap_reshaped)
        
        # Die resultierenden Cluster-Labels in die ursprüngliche Form zurückkonvertieren
        segmented_heatmap = kmeans.labels_.reshape((x, y))
        
        return segmented_heatmap

    # threshold=0.0007, block_size=2
    def graph(self, relevance_map, img, threshold=0.0007, block_size=2):
        #block size=Größe der Blöcke in denen die Pixel zusammengefasst sind
        # mean and standard deviation values used for image preprocessing
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225])  

        G = nx.Graph()

        height, width = relevance_map.shape

        # Schleife über das Relevanz-Map in Blöcken der Größe block_size
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Überprüfung der benachbarten Blöcke
                for dy, dx in [(0, block_size), (block_size, 0)]:
                    if 0 <= y + dy < height and 0 <= x + dx < width:
                        # Ermittlung der Knoten-ID basierend auf der aktuellen Blockposition
                        node1 = (y // block_size) * (width // block_size) + (x // block_size)
                        node2 = ((y + dy) // block_size) * (width // block_size) + ((x + dx) // block_size)
                        
                        # Gewicht des Knotens basierend auf dem mittleren Relevanzwert des Blocks berechnen
                        weight = abs(np.mean(relevance_map[y:y+block_size, x:x+block_size]) - 
                                    np.mean(relevance_map[y+dy:y+dy+block_size, x+dx:x+dx+block_size]))
                        if weight > threshold:
                            G.add_edge(node1, node2, weight=weight)

        # Positionen für jeden Knoten (Zentrum des Blocks) berechnen
        pos = {(y // block_size) * (width // block_size) + (x // block_size): ((x + block_size // 2), -(y + block_size // 2)) 
            for y in range(0, height, block_size) 
            for x in range(0, width, block_size)}

        # Liste der Gewichtungen für jede Kante erstellen
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

        plt.figure(figsize=(5, 5))  # figure size 

        # Bild von Tensor in Numpy Array umwandeln und de-normalisieren
        img = img.squeeze().numpy().transpose(1,2,0) * std + mean 
        img = (img * 255).astype(np.uint8)  # Umwandlung in 8-Bit-Werte

        plt.imshow(img, extent=[0, width, -height, 0], alpha=0.3)
        nx.draw(G, pos, with_labels=False, node_size=10, edge_color=edge_weights, edge_cmap=plt.cm.Blues)
        plt.show()
        return G, pos, width, height, img


    def fully_connect_and_plot(self, G, img, pos, width, height):
        fully_connected_G = nx.Graph()
        nodes = list(G.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Gewicht zw. Knoten 
                weight = 1.0
                fully_connected_G.add_edge(node1, node2, weight=weight)

        plt.figure(figsize=(5, 5))  # Figure size
        plt.imshow(img, extent=[0, width, -height, 0], alpha=0.3)

        # Gewichtungen für die Kanten 
        edge_weights = [fully_connected_G[u][v]['weight'] for u, v in fully_connected_G.edges()]

        nx.draw(fully_connected_G, pos, with_labels=False, node_size=10, edge_color=edge_weights, edge_cmap=plt.cm.Blues)
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

        #SEGMENTATION
        segmented_heatmap = self.segmentation(heatmap_fig)
        plt.imshow(segmented_heatmap, cmap='tab20b')
        plt.show()
        
        #GRAPH
        relevance_map = np.array(R[l][0]).sum(axis=0)
        G, pos, width, height, img = self.graph(relevance_map, img) 
        self.fully_connect_and_plot(G, img, pos, width, height)
