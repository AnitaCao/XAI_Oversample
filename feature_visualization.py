import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from CMO.imbalance_data.imbalance_iNaturalist import load_imb_inaturalist, Imb_iNat_Dataset

# Load the pretrained ResNet-50 model
model = models.resnet50(pretrained=True)
model.eval()

def get_features(module, input, output):
    features.append(output.view(output.size(0), -1).detach().cpu().numpy())

features = []
handle = model.avgpool.register_forward_hook(get_features)

# Define your data transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_path = "D:/anita/Research/iNaturalist/train/"
train_images_list, train_labels_list, _, _ = load_imb_inaturalist(data_path, transform, transform)
train_dataset = Imb_iNat_Dataset(data_path,train_images_list, train_labels_list,transform)

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# Pass the dataset through the model to get features
labels = []
with torch.no_grad():
    for inputs, targets in dataloader:
        _ = model(inputs)
        labels.extend(targets.numpy())

handle.remove()
features = np.concatenate(features, axis=0)
labels = np.array(labels)

'''
# Compute pairwise distances
distances = pdist(features, metric='euclidean')
distance_matrix = squareform(distances)

# Visualize pairwise distances using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(distance_matrix, cmap='viridis')
plt.title('Pairwise Distance Matrix of Features')
plt.show()
'''

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(features)

class_labels = np.unique(labels)

# Plot the reduced features
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.legend(handles=scatter.legend_elements(), labels=class_labels)
plt.colorbar(scatter)
plt.title('t-SNE of ResNet-50 Features')
plt.show()


# Plot the reduced features
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title('t-SNE of ResNet-50 Features')

# Annotate each class
class_centroids = []

for label in class_labels:
    class_points = reduced_features[labels == label]
    centroid = class_points.mean(axis=0)
    class_centroids.append(centroid)
    plt.annotate(train_dataset.classes[label], (centroid[0], centroid[1]), fontsize=12, ha='center', backgroundcolor='white')

plt.show()