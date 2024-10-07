import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from CMO.imbalance_data.imbalance_iNaturalist import load_imb_inaturalist, Imb_iNat_Dataset
from torchvision import models, transforms
import torch
from torch.utils.data import DataLoader

# Load the pretrained ResNet-50 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)
model.eval()
model.to(device)

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
long_tail = False
random_select = False
train_images_list, train_labels_list, _, _ = load_imb_inaturalist(data_path, transform, transform, long_tail, random_select)
train_dataset = Imb_iNat_Dataset(data_path,train_images_list, train_labels_list,transform)

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# Pass the dataset through the model to get features
labels = []
with torch.no_grad():
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        _ = model(inputs)
        labels.extend(targets.numpy())

handle.remove()
features = np.concatenate(features, axis=0)
labels = np.array(labels)

reducer = umap.UMAP(n_components=2, random_state=42)
reduced_features = reducer.fit_transform(features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title('UMAP of ResNet-50 Features')
plt.show()

class_labels = np.unique(labels)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title('UMAP of ResNet-50 Features')

class_centroids = []

for label in class_labels:
    class_points = reduced_features[labels == label]
    centroid = class_points.mean(axis=0)
    class_centroids.append(centroid)
    plt.annotate(label, (centroid[0], centroid[1]), fontsize=8, ha='center')

plt.show()