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


class ExplanationHandler:
    def __init__(self, model):
        self.model = model

    def normalize_img(self, img, b):
        return np.clip((img + b) / (2 * b), 0, 1)

    def create_cmap(self):
        my_cmap = plt.cm.bwr(np.arange(plt.cm.bwr.N))
        my_cmap[:, 0:3] *= 0.85
        return ListedColormap(my_cmap)

    def heatmap(self, R, sx, sy, img, name, save=True):
        b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))
        my_cmap = self.create_cmap()

        plt.figure(figsize=(sx, sy))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')

        if R.shape[0] == 1:
            R = np.transpose(R, (1, 2, 0))

        heatmap_img_resized = cv2.resize(R, (img.shape[1], img.shape[0]))
        heatmap_img_resized_normalized = self.normalize_img(heatmap_img_resized, b)

        plt.imshow(heatmap_img_resized_normalized, cmap=my_cmap, vmin=0, vmax=1)
        plt.imshow(img, cmap='gray', interpolation='None', alpha=0.15)
        plt.colorbar(orientation="horizontal", shrink=0.75, ticks=[-1, 0, 1])

        if save:
            os.makedirs('results', exist_ok=True)
            plt.imsave(os.path.join('results', name + "_heatmap.jpg"), heatmap_img_resized, cmap=my_cmap, vmin=-b, vmax=b)

        plt.show()
        plt.close()


    def heatmap_with_segmentation(self, heatmap_img, binary_mask, sx, sy, img, name, threshold=0.5, save=True):
        b = 10 * ((np.abs(heatmap_img) ** 3.0).mean() ** (1.0 / 3))
        black_pixel = (0, 0, 0)  # Represents the RGB values for black (0, 0, 0)
        my_cmap = self.create_cmap()

        plt.figure(figsize=(sx, sy))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')

    # Ensure heatmap_img is 2D and has the same shape as the input image
        heatmap_img = heatmap_img.squeeze()
        heatmap_img_resized = cv2.resize(heatmap_img, (img.shape[1], img.shape[0]))

        if heatmap_img_resized.ndim == 2:
            heatmap_img_resized = np.expand_dims(heatmap_img_resized, axis=2)
            heatmap_img_resized = np.repeat(heatmap_img_resized, 3, axis=2)

        heatmap_img_resized_normalized = self.normalize_img(heatmap_img_resized, b)

        plt.imshow(heatmap_img_resized_normalized, cmap='coolwarm', vmin=0, vmax=1)
        plt.colorbar(orientation="horizontal", shrink=0.75, ticks=[-1, 0, 1])

    # Resize binary_mask to the same shape as the input image
        binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), (img.shape[1], img.shape[0]))
        binary_mask_expanded = np.expand_dims(binary_mask_resized, axis=2)
        binary_mask_expanded = np.repeat(binary_mask_expanded, 3, axis=2)

        segmented_img = copy.deepcopy(img)
        binary_mask_expanded = np.tile(binary_mask[:, :, np.newaxis], (1, 1, segmented_img.shape[2]))
        segmented_img[binary_mask_expanded[:, :, 0] == 0] = black_pixel

        segmented_img_normalized = np.clip(segmented_img / 255.0, 0, 1)

        plt.imshow(segmented_img_normalized * 255, cmap='coolwarm', alpha=0.5, vmin=0, vmax=255)  # Overlay segmented regions

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

    
    def heatmap_with_graph(self, R, sx, sy, img, name, num_nodes, save=True):
        b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))
        my_cmap = self.create_cmap()

        plt.figure(figsize=(sx, sy))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')

        if R.shape[0] == 1:
            R = np.transpose(R, (1, 2, 0))

        heatmap_img_resized = cv2.resize(R, (img.shape[1], img.shape[0]))
        heatmap_img_resized_normalized = self.normalize_img(heatmap_img_resized, b)

        # Find the coordinates of the top N most relevant pixels
        nodes = np.unravel_index(np.argsort(heatmap_img_resized.ravel())[-num_nodes:], heatmap_img_resized.shape)
        edges = connect_neighboring_nodes(nodes, threshold=10)
        # Remove the batch dimension
        img = img[0]

        # Transpose the dimensions to (height, width, channels)
        #img = np.transpose(img, (1, 2, 0))

        # If the image data is in the range [0, 255], you can normalize to [0, 1]
        img = img / 255.0

        # Display the image
        plt.imshow(img, interpolation='None', alpha=0.15)
        plt.imshow(heatmap_img_resized_normalized, cmap=my_cmap, vmin=0, vmax=1)

        # Plotting nodes
        for node in zip(*nodes):
            plt.scatter(*node, c='red')

        # Plotting edges
        for edge in edges:
            node_start = nodes[0][edge[0]], nodes[1][edge[0]]
            node_end = nodes[0][edge[1]], nodes[1][edge[1]]
            plt.plot(*zip(node_start, node_end), c='blue')

        plt.colorbar(orientation="horizontal", shrink=0.75, ticks=[-1, 0, 1])

        if save:
            os.makedirs('results', exist_ok=True)
            plt.imsave(os.path.join('results', name + "_heatmap_with_nodes.jpg"), heatmap_img_resized, vmin=-b, vmax=b)

        plt.show()
        plt.close()


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

    
    def explain(self, model, input_image, input_file_path, model_name, save=True, segmentation_threshold=0.5):
        # Define the path to save the results
        results_path = 'results/LRP/'
        # Create the directory if it doesn't exist
        os.makedirs(results_path, exist_ok=True)
        # Generate a filename based on the input file's name and the model name
        name = os.path.splitext(os.path.basename(input_file_path))[0] + "_" + model_name + '.jpg'
        # Deep copy the image to avoid changes to the original
        copied_image = copy.deepcopy(input_image)


        # Define the mean and std used for normalization during pre-processing of the image
        normalization_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        normalization_std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)


        # Obtain the layers from the model
        model_layers = list(self.model._modules['features']) + self.toconv(list(self.model._modules['classifier']), model_name)
        # Get the number of layers
        num_layers = len(model_layers)
        # Initialize the activation list
        activations = [copied_image] + [None] * num_layers


        # Forward pass through the network and store the activations for each layer
        for layer_index in range(num_layers):
            activations[layer_index + 1] = model_layers[layer_index].forward(activations[layer_index])

        # Calculate the scores for the forward pass
        class_scores = np.array(activations[-1].data.view(-1))
        # Get the class with the highest score
        top_class_index = np.argsort(-class_scores)[0]
        #Encode the target class in a tensor
        target_class_tensor = torch.FloatTensor((1.0 * (np.arange(1000) == top_class_index).reshape([1, 1000, 1, 1])))
        
        # Initialize the relevance list
        relevances = [None] * num_layers + [(activations[-1] * target_class_tensor).data]

        # Backward pass through the network to calculate the relevance for each layer
        for layer_index in range(1, num_layers)[::-1]:
            activations[layer_index] = (activations[layer_index].data).requires_grad_(True)
            # Adjust the layer if it's MaxPool2d to AvgPool2d for the backpropagation of relevance
            if isinstance(model_layers[layer_index], torch.nn.MaxPool2d):
                model_layers[layer_index] = torch.nn.AvgPool2d(2)
            # If the layer is either Conv2d or AvgPool2d, perform LRP using the rules defined in the code
            if isinstance(model_layers[layer_index], torch.nn.Conv2d) or isinstance(model_layers[layer_index], torch.nn.AvgPool2d):
                # Choose different rules based on the layer's index
                if layer_index <= 16:
                    positive_part_function = lambda p: p + 0.25 * p.clamp(min=0)
                    increment_function = lambda z: z + 1e-9
                elif 17 <= layer_index <= 30:
                    positive_part_function = lambda p: p
                    increment_function = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
                elif layer_index >= 31:
                    positive_part_function = lambda p: p
                    increment_function = lambda z: z + 1e-9

                # Calculate the neuron activation
                neuron_activation = increment_function(self.newlayer(model_layers[layer_index], positive_part_function).forward(activations[layer_index]))
                # Calculate the propagated relevance
                propagated_relevance = (relevances[layer_index + 1] / neuron_activation).data
                # Backward the result
                (neuron_activation * propagated_relevance).sum().backward()
                # Get the gradient
                gradient = activations[layer_index].grad
                # Calculate the relevance at this layer
                relevances[layer_index] = (activations[layer_index] * gradient).data

            else:
                relevances[layer_index] = relevances[layer_index + 1]

        # Prepare a list of layers to generate the relevance map
        layers_map = [31, 21, 11, 1]

        # Generate a name for the file based on the input file and model
        name = os.path.splitext(input_file_path)[0]
        name = name + "_" + model_name

        # At each layer in the layers_map, generate a relevance map (omitted in the provided code)
        for i, layer_index in enumerate(layers_map):
            if layer_index == layers_map[-1] and model_name == "vgg" and False:
                pass
            elif False:
                self.heatmap_(np.array(relevances[layer_index][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)
        
        
        # Backpropagate the relevance all the way to the input layer
        activations[0] = activations[0].data.requires_grad_(True)

        # Calculate the lower and upper relevance boundaries for the input layer
        lower_relevance_boundary = (activations[0].data * 0 + (0 - normalization_mean) / normalization_std).requires_grad_(True)
        upper_relevance_boundary = (activations[0].data * 0 + (1 - normalization_mean) / normalization_std).requires_grad_(True)

        # Calculate the neuron activation and the propagated relevance for the input layer
        neuron_activation = model_layers[0].forward(activations[0]) + 1e-9
        neuron_activation -= self.newlayer(model_layers[0], lambda p: p.clamp(min=0)).forward(lower_relevance_boundary)
        neuron_activation -= self.newlayer(model_layers[0], lambda p: p.clamp(max=0)).forward(upper_relevance_boundary)
        propagated_relevance = (relevances[1] / neuron_activation).data
        (neuron_activation * propagated_relevance).sum().backward()
        gradient, lower_relevance_gradient, upper_relevance_gradient = activations[0].grad, lower_relevance_boundary.grad, upper_relevance_boundary.grad
        relevances[0] = (activations[0] * gradient + lower_relevance_boundary * lower_relevance_gradient + upper_relevance_boundary * upper_relevance_gradient).data

        # Prepare the image for overlay
        image_np_array = input_image.squeeze().cpu().numpy()
        image_for_overlay = np.transpose(image_np_array, (1, 2, 0))

        # Generate the relevance map for the input layer
        input_relevance_map = np.array(relevances[0][0]).sum(axis=0)
        # Create a binary mask from the relevance map
        relevance_binary_mask = input_relevance_map >= segmentation_threshold
        
        #self.heatmap_with_segmentation(input_relevance_map, relevance_binary_mask, 10, 10, image_for_overlay, name, save=save)
        #self.heatmap(input_relevance_map, 10, 10, image_for_overlay, name, save=save)
        self.heatmap_with_graph(input_relevance_map, 10, 10, image_for_overlay, name, 100, save=True)


def connect_neighboring_nodes(nodes, threshold):
    edges = []
    nodes_array = np.column_stack(nodes)  # Convert the tuple of arrays into a 2D array
    distances = cdist(nodes_array, nodes_array, 'euclidean')
    for i in range(len(nodes_array)):
        for j in range(i + 1, len(nodes_array)):
            if distances[i, j] < threshold:
                edges.append((i, j))
    return edges
