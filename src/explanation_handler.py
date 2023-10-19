import torch
import copy
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors



class ExplanationHandler:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def prepare_image_for_display(img):
        image_copy = np.copy(img)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_copy = img.squeeze().numpy().transpose(1, 2, 0) * std + mean
        return (image_copy * 255).astype(np.uint8)  # Convert to 8-bit values
    
    @staticmethod
    def create_output_dir():
        output_dir = "Output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    @staticmethod
    def format_label(predicted_label):
        return predicted_label.replace(':', '_').replace(' ', '_').replace(',', '')
# --------------------------------------
# Visualizing data
# --------------------------------------
    def heatmap(self, R, sx, sy, img, predicted_label, user_input, save=True):
        image_copy = self.prepare_image_for_display(img)

        # Create a figure and a set of subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(sx, sy))

        # Original Image
        axes[0].imshow(image_copy)
        axes[0].set_title("Originalbild")
        axes[0].axis('off')

        # Heatmap
        im = axes[1].imshow(R, cmap='cividis')
        axes[1].set_title("Ergebnis")
        axes[1].axis('off')

        # Add colorbar with only min, 0 and max value
        cbar_min = np.nanmin(R)
        cbar_max = np.nanmax(R)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                            ticks=[cbar_min, 0, cbar_max])
        cbar.ax.set_xticklabels(['spricht gegen die Entscheidung', 'nicht relevant für die Entscheidung',
                                 'spricht für die Entscheidung'])

        if save:
            output_dir = self.create_output_dir()
            label_str = self.format_label(predicted_label)
            output_path = os.path.join(output_dir, f"heatmap{label_str + user_input}.png")
            try:
                plt.savefig(output_path, bbox_inches='tight', format='png')
            except Exception as e:
                print(f"Fehler beim Speichern des Bildes: {e}")

        plt.show()
        return R
    
    def save_relevance_map_as_csv(self, R, predicted_label):
        output_dir = self.create_output_dir()
        label_str = self.format_label(predicted_label)

        csv_path = os.path.join(output_dir, f"heatmap{label_str}.csv")

        try:
            np.savetxt(csv_path, R, delimiter=",")
            print(f"Relevanzkarte erfolgreich gespeichert unter: {csv_path}")
        except Exception as e:
            print(f"Fehler beim Speichern der Relevanzkarte: {e}")

    def cluster_relevance_map(self, R, img, predicted_label, user_input, save=True, colors=None):
        # Default colors
        colors = colors or ['indigo', 'darkcyan', 'yellow']
        cmap = mcolors.ListedColormap(colors)

        # BoundaryNorm
        norm = mcolors.BoundaryNorm(boundaries=[0,1,2,3], ncolors=cmap.N)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=3).fit(R.reshape(-1, 1))
        labels = kmeans.predict(R.reshape(-1, 1))

         # sorting of Cluster-Centroids and Cluster-IDs
        sorted_centroids_idx = np.argsort(kmeans.cluster_centers_.squeeze())
        
        # Setzen der Cluster-Labels basierend auf den sortierten Cluster-Centroids
        for i, cluster_id in enumerate(sorted_centroids_idx):
            labels[labels == cluster_id] = i + 10  # Temporary IDs
        labels -= 10  # back to 0, 1, 2

        # Prepare the clustered image
        cluster_map = labels.reshape(R.shape)

        # Create a figure and a set of subplots
        image_copy = self.prepare_image_for_display(img)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axes[0].imshow(image_copy)
        axes[0].set_title("Originalbild")
        axes[0].axis('off')

        # Display the clustered heatmap
        img = axes[1].imshow(cluster_map, cmap=cmap, norm=norm)
        cbar = fig.colorbar(img, ax=axes.ravel().tolist(), orientation='horizontal', boundaries=[0,1,2,3])
        cbar.ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
        cbar.ax.set_xticklabels(['', 'Nicht Relevant', '', 'Wichtiger Bereich', '', 'Wichtigster Bereich'])
        # Adjust the tick label position to be left of the color blocks
        for label in cbar.ax.xaxis.get_ticklabels():
            label.set_horizontalalignment('center')
        axes[1].set_title("Ergebnis")
        axes[1].axis('off')

        if save:
            self.save_clustered_image(fig, predicted_label, user_input)

        plt.show()
        return cluster_map

    def contour_relevance_map(self, R, img, predicted_label, user_input, save=True):
        image_copy = self.prepare_image_for_display(img)

        # Create a figure and a set of subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 13))
        axes[0].imshow(image_copy)
        axes[0].set_title("Originalbild")
        axes[0].axis('off')

        # Display the heatmap with contours
        im = axes[1].imshow(R, cmap='cividis')
        #levels: kalkuliert RelevanceMap Data
        axes[1].contour(R, levels=[R.mean() + R.std()], colors='black')
        axes[1].set_title("Ergebnis")
        axes[1].axis('off')

        # Display cbar
        cbar_min = np.nanmin(R)
        cbar_max = np.nanmax(R)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', ticks=[cbar_min, 0, cbar_max])
        cbar.ax.set_xticklabels(['spricht gegen die Entscheidung','nicht relevant für die Entscheidung',  'spricht für die Entscheidung'])
        

        if save:
            self.save_contour_image(fig, predicted_label, user_input)

        plt.show()

    def overlay_clustered_on_grayscale(self, img, clustered_image, predicted_label, user_input, alpha=0.5, save=True):
        image_copy = self.prepare_image_for_display(img)

        # Convert original image to grayscale
        grayscale_image = np.dot(image_copy[..., :3], [0.2989, 0.5870, 0.1140])
        grayscale_image_rgb = np.stack([grayscale_image] * 3, axis=-1).astype(np.uint8)

        # Create a figure and a set of subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axes[0].imshow(image_copy)
        axes[0].set_title("Originalbild")
        axes[0].axis('off')

        # Display the overlay of clustered map on grayscale image
        axes[1].imshow(grayscale_image_rgb, alpha=alpha - 0.1)
        im = axes[1].imshow(clustered_image, alpha=alpha)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', ticks=[0, 1, 2])
        cbar.ax.set_xticklabels(['Nicht Relevant', 'Wichtiger Bereich', 'Wichtigster Bereich'])
        axes[1].set_title("Ergebnis")
        axes[1].axis('off')

        if save:
            self.save_overlay_image(fig, predicted_label, user_input)

        plt.show()

    def save_clustered_image(self, fig, predicted_label, user_input):
        output_dir = self.create_output_dir()
        label_str = self.format_label(predicted_label)
        output_path = os.path.join(output_dir, f"clustered_heatmap_{label_str + user_input}.png")
        self.save_image(fig, output_path)

    def save_contour_image(self, fig, predicted_label, user_input):
        output_dir = self.create_output_dir()
        label_str = self.format_label(predicted_label)
        output_path = os.path.join(output_dir, f"contour_map{label_str + user_input}.png")
        self.save_image(fig, output_path)

    def save_overlay_image(self, fig, predicted_label, user_input):
        output_dir = self.create_output_dir()
        label_str = self.format_label(predicted_label)
        output_path = os.path.join(output_dir, f"clusteredOverlay{label_str + user_input}.png")
        self.save_image(fig, output_path)

    @staticmethod
    def save_image(fig, output_path):
        try:
            fig.savefig(output_path, bbox_inches='tight', format='png')
        except Exception as e:
            print(f"Fehler beim Speichern des Bildes: {e}")


#LRP Model code (explain()) and utility functions (toncov(), newlayer()) downloaded from:
#    https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/tree/main

# --------------------------------------------------------------
# Clone a layer and pass its parameters through the function g
# --------------------------------------------------------------
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
# --------------------------------------------------------------
# convert VGG classifier's dense layers to convolutional layers
# --------------------------------------------------------------
    def toconv(self, layers):
        newlayers = []
        for i, layer in enumerate(layers):
            if isinstance(layer, torch.nn.Linear):
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

    def explain(self, model, img, predicted_label, user_input, save=True):
        """
    Explains the prediction of a model on an image by visualizing the relevance of each part of the image 
    to the model's decision. 
    Parameters:
    - model (torch.nn.Module): used pretrained model
    - img (torch.Tensor): input image tensor 
    - predicted_label (str): predicted label
    - user_input (str): user input for saving the img
    - save (bool, optional): save the img?
    """
        X = copy.deepcopy(img)

        mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        layers = list(model._modules['features']) + self.toconv(list(model._modules['classifier']))
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

# --------------------------------------------------------------
# Method calls
# --------------------------------------------------------------
        heatmap_fig = self.heatmap(np.array(R[l][0]).sum(axis=0), 18, 13, img, predicted_label, user_input)
        clustered_image = self.cluster_relevance_map(np.array(R[l][0]).sum(axis=0), img, predicted_label, user_input)
        self.overlay_clustered_on_grayscale(img, clustered_image, predicted_label, user_input, alpha=0.5) 
        self.contour_relevance_map(np.array(R[l][0]).sum(axis=0), img, predicted_label, user_input) 
        #self.save_relevance_map_as_csv(np.array(R[l][0]).sum(axis=0), predicted_label)

    




        
