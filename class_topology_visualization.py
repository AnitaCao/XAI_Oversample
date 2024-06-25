import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
import seaborn as sns

def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def get_features_hook(module, input, output, features):
    features.append(output.view(output.size(0), -1).detach().cpu().numpy())

def load_data(data_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, dataloader

def extract_features(model, dataloader):
    features = []
    labels = []
    handle = model.avgpool.register_forward_hook(lambda m, i, o: get_features_hook(m, i, o, features))
    with torch.no_grad():
        for inputs, targets in dataloader:
            _ = model(inputs)
            labels.extend(targets.numpy())
    handle.remove()
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels

def compute_pairwise_distances(features):
    distances = pdist(features, metric='euclidean')
    distance_matrix = squareform(distances)
    return distance_matrix

def plot_distance_matrix(distance_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(distance_matrix, cmap='viridis')
    plt.title('Pairwise Distance Matrix of Features')
    plt.show()

def reduce_dimensionality(features):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)
    return reduced_features

def plot_tsne(reduced_features, labels, class_names):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
    plt.colorbar(scatter)
    plt.title('t-SNE of ResNet-50 Features')
    plt.show()

def compute_class_centroids(features, labels):
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        centroids.append(features[labels == label].mean(axis=0))
    centroids = np.array(centroids)
    return centroids, unique_labels

def hierarchical_clustering(centroids, class_names):
    dist_matrix = cdist(centroids, centroids, metric='euclidean')
    linkage_matrix = linkage(dist_matrix, method='average')
    dendro = dendrogram(linkage_matrix, labels=class_names, leaf_rotation=90)
    topological_order = leaves_list(linkage_matrix)
    plt.figure(figsize=(10, 8))
    dendrogram(linkage_matrix, labels=class_names, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Class')
    plt.ylabel('Distance')
    plt.show()
    return topological_order

def main():
    data_path = 'path/to/your/training_dataset'
    model = load_model()
    dataset, dataloader = load_data(data_path)
    features, labels = extract_features(model, dataloader)
    
    distance_matrix = compute_pairwise_distances(features)
    plot_distance_matrix(distance_matrix)
    
    reduced_features = reduce_dimensionality(features)
    plot_tsne(reduced_features, labels, dataset.classes)
    
    centroids, unique_labels = compute_class_centroids(features, labels)
    topological_order = hierarchical_clustering(centroids, dataset.classes)
    print("Topological order of classes:", unique_labels[topological_order])

if __name__ == "__main__":
    main()