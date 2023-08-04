import torch
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.io import read_image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from torchvision.utils import draw_keypoints
# Define transformations
def transform(img):
    img = F.to_tensor(img).unsqueeze(0)
    return img

# Segmentation Model
model = fcn_resnet50(pretrained=True, progress=False)
model = model.eval()

dog_list = [...]  # Replace this with a list of dog images
batch = torch.stack([transform(d) for d in dog_list])
output = model(batch)['out']
print(output.shape, output.min().item(), output.max().item())

# Keypoint Detection Model
person_int = read_image("/Users/valerijamadzoska/Desktop/bilderBA/dog-4988985_640.jpg")
person_float = transform(person_int)

model = keypointrcnn_resnet50_fpn(pretrained=True, progress=False)
model = model.eval()

outputs = model([person_float])
kpts = outputs[0]['keypoints']
scores = outputs[0]['scores']

detect_threshold = 0.75
idx = torch.where(scores > detect_threshold)
keypoints = kpts[idx]

# Visualization
def show(img):
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

res = draw_keypoints(person_int, keypoints, colors="blue", radius=3)
show(res)

# Define the function to draw keypoints along with connections
def draw_keypoints_and_skeleton(image, keypoints, connections, color="blue", radius=4, width=3):
    image_with_keypoints = draw_keypoints(image, keypoints, colors=color, radius=radius)

    # Convert to numpy for Matplotlib plotting
    if torch.is_tensor(image_with_keypoints):
        image_with_keypoints = image_with_keypoints.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots()
    ax.imshow(image_with_keypoints)

    # Draw connections
    keypoints_np = keypoints.numpy()
    for start, end in connections:
        if start < len(keypoints_np) and end < len(keypoints_np):
            x1, y1, _ = keypoints_np[start]
            x2, y2, _ = keypoints_np[end]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width)

    plt.show()

# Call the function with the keypoints, image, and connections
draw_keypoints_and_skeleton(person_int, keypoints[0], connect_skeleton)
