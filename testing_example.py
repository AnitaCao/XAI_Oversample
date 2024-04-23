import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)
model.eval()

# Define the transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
img = Image.open('data_examples/horse.jpg')
tensor = preprocess(img).unsqueeze(0)

# Register hooks
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# Register the hooks on the last convolutional layer
handle_forward = model.layer4[2].conv3.register_forward_hook(forward_hook)
handle_backward = model.layer4[2].conv3.register_backward_hook(backward_hook)


output = model(tensor)

# Get the score for the target class
_, predicted = torch.max(output, 1)
target_class = predicted.item()

# Zero out all other scores apart from the target class
one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
one_hot_output[0][target_class] = 1

# Backward pass
model.zero_grad()
output.backward(gradient=one_hot_output.to(output.device), retain_graph=True)

grads = gradients[0].cpu().data.numpy()
# Pool the gradients across the channels
pooled_grads = np.mean(grads, axis=(2, 3))

# Get the feature maps
feature_map = feature_maps[0].cpu().data.numpy()

# Weight the feature maps by the pooled gradients
for i in range(pooled_grads.shape[1]):
    feature_map[:, i, :, :] *= pooled_grads[i]

# Average the weighted feature maps across the channels
cam = np.mean(feature_map, axis=1)

# Apply ReLU to the class activation map
cam = np.maximum(cam, 0)

# Resize the class activation map to match the original image size
cam = cv2.resize(cam[0], (img.width, img.height))

# Normalize the class activation map
cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

# Display the class activation map
plt.imshow(cam, cmap='hot')
plt.axis('off')
plt.show()

# Remove the hooks
handle_forward.remove()
handle_backward.remove()