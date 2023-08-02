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

class ExplanationHandler:
    def __init__(self, model):
        self.model = model

    def normalize_img(self, img, b):
        return np.clip((img + b) / (2 * b), 0, 1)

    #VORSCHLAG VON DER KI, AM BESTEN IN SEPERATEM SCHRITT MIT DER URSPRUNGS EXPLAIN METHODE ERSETZEN
    # def explain(self, img, file, model_str, technique="heatmap", addon="default", save=True, segmentation_threshold=0.5):
    #     explanation_methods = {
    #         "heatmap": self.heatmap,
    #         "heatmap_with_segmentation": self.heatmap_with_segmentation,
    #         "not_yet_known": self.not_yet_known
    #     }
        
    #     addon_methods = {
    #         "default": self.addon_default,
    #         "a": self.addon_a,
    #         "b": self.addon_b,
    #         "c": self.addon_c
    #     }

    #     explanation_method = explanation_methods.get(technique, self.heatmap)
    #     addon_method = addon_methods.get(addon, self.addon_default)
        
    #     explanation_result = explanation_method(img, file, model_str, save, segmentation_threshold)
    #     addon_result = addon_method(explanation_result)

    #     return addon_result

    def heatmap(self, R, sx, sy, img, name, save=True):
        b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))

        my_cmap = create_cmap()

        plt.figure(figsize=(sx, sy))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')

        if R.shape[0] == 1:
            R = np.transpose(R, (1, 2, 0))

        heatmap_img_resized = cv2.resize(R, (img.shape[1], img.shape[0]))
        heatmap_img_resized_normalized = self.normalize_img(heatmap_img_resized, b)

        plt.imshow(heatmap_img_resized_normalized, cmap=my_cmap, vmin=0, vmax=1)
        cbar = plt.colorbar(orientation="horizontal", shrink=0.75, ticks=[-1, 0, 1])
        plt.imshow(img, cmap='gray', interpolation='None', alpha=0.15)
        cbar.ax.set_xticklabels(["least relevant", "", "most relevant"])

        if save:
            os.makedirs('results', exist_ok=True)
            plt.imsave(os.path.join('results', name + "_heatmap.jpg"), heatmap_img_resized, cmap=my_cmap, vmin=-b, vmax=b)

    #plt.show()
        return

    def heatmap_with_segmentation(self, heatmap_img, binary_mask, sx, sy, img, name, threshold=0.5, save=True):
        b = 10 * ((np.abs(heatmap_img) ** 3.0).mean() ** (1.0 / 3))
        black_pixel = (0, 0, 0)  # Represents the RGB values for black (0, 0, 0)

        my_cmap = self.create_cmap()

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

        heatmap_img_resized_normalized = self.normalize_img(heatmap_img_resized, b)

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

    def not_yet_known(self, img, file, model_str, save, segmentation_threshold):
        # Code for the unknown technique
        pass

    def addon_default(self, explanation_result):
        # Code for default addon
        return explanation_result

    def addon_a(self, explanation_result):
        # Code for addon a
        pass

    def addon_b(self, explanation_result):
        # Code for addon b
        pass

    def addon_c(self, explanation_result):
        # Code for addon c
        pass


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

    



    def explain(self, model, img, file, model_str, save=True, segmentation_threshold=0.5):
        results_path = 'results/LRP/'
        os.makedirs(results_path, exist_ok=True)
        name = os.path.splitext(os.path.basename(file))[0] + "_" + model_str + '.jpg'
        full_path = os.path.join(results_path, name)
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

                z = incr(self.newlayer(layers[l], rho).forward(A[l]))
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
        z -= self.newlayer(layers[0], lambda p: p.clamp(min=0)).forward(lb)
        z -= self.newlayer(layers[0], lambda p: p.clamp(max=0)).forward(hb)
        s = (R[1] / z).data
        (z * s).sum().backward()
        c, cp, cm = A[0].grad, lb.grad, hb.grad
        R[0] = (A[0] * c + lb * cp + hb * cm).data

        input_np = img.squeeze().cpu().numpy()
        img_for_overlay = np.transpose(input_np, (1, 2, 0))

        relevance_map = np.array(R[0][0]).sum(axis=0)
        binary_mask = relevance_map >= segmentation_threshold

        if model_str == "alexnet":
            self.heatmap_with_segmentation(relevance_map, binary_mask, 10, 10, img_for_overlay, name, save=save)
        else:
            self.heatmap(relevance_map, 10, 10, img_for_overlay, name, save=save)


def create_cmap():
        my_cmap = plt.cm.bwr(np.arange(plt.cm.bwr.N))
        my_cmap[:, 0:3] *= 0.85
        return ListedColormap(my_cmap)
