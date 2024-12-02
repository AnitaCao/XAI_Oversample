import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import numpy as np
from scipy.spatial.distance import cdist
from CMO.imbalance_data.imbalance_iNaturalist import load_imb_inaturalist, Imb_iNat_Dataset

#get the centroid to centroid distance matrix of each classes.
def get_distance_matrix(dataloader):
    # Step 1: Load the pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # Step 2: Modify the model to output features before the classification layer
    # Remove the final fully connected layer (which is for classification)
    model = nn.Sequential(*list(model.children())[:-1])

    # Set the model to evaluation mode,move the model to GPU if available
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Step 4: Extract features and store them in a matrix
    features = []
    labels_list = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten the output
            features.append(outputs.cpu().numpy())
            labels_list.append(targets.cpu().numpy())

    # Convert the list of features and labels to numpy arrays
    features = np.concatenate(features, axis=0) #shape (n_samples, n_features)
    labels = np.concatenate(labels_list, axis=0) #shape (n_samples,)
    
    # Step 5: Computer centroids for each class
    class_labels = np.unique(labels)
    class_centroids = []
    for label in class_labels:
        class_features = features[labels == label]
        class_centroid = np.mean(class_features, axis=0)
        class_centroids.append(class_centroid)

    class_centroids = np.array(class_centroids) #shape (n_classes, n_features)

    # Step 6: Compute pairwise distances between class centroids
    distance_matrix = cdist(class_centroids, class_centroids, metric='euclidean')
    
    return distance_matrix, class_labels


# Convert the centroid distance matrix to proabbilites. 
# For each class, assign probabilities based on the inverse distance
# The closer the class centroids are, the higher the probability
def get_probability_matrix(distance_matrix):
    num_classes = distance_matrix.shape[0]
    probability_matrix = np.zeros_like(distance_matrix)

    # set the diagonal to infinity to avoid dominating the normalization
    np.fill_diagonal(distance_matrix, np.inf)

    for i in range(num_classes):
        distances = distance_matrix[i]

        # Inverse distances to give higher weight to further classes
        inverse_distances = 1 / (distances + 1e-10)  # Adding a small value to avoid division by zero

        # Normalize inverse distances to sum to 1
        probabilities = inverse_distances / np.sum(inverse_distances)
        
        probability_matrix[i] = probabilities
    return probability_matrix

def sample_next_class(probability_matrix, class_labels, sampled_classes):
    # Get the probability of currently sampled classes
    class_probabilities = probability_matrix[sampled_classes]
    
    #randomly sample the next class based on the probability
    next_class = np.random.choice(class_labels, p=class_probabilities)
    
    return next_class  


# Visualize the probability matrix
def visualize_probability_matrix(probability_matrix):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(probability_matrix, annot=True, fmt=".3f", cmap='Blues')
    plt.title('Class Probability Matrix Based on Centroid Distances')
    plt.xlabel('Classes')
    plt.ylabel('Classes')
    plt.show()
    
def visualize_distance_matrix(distance_matrix, class_labels):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 10))
    sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap='viridis', xticklabels=class_labels, yticklabels=class_labels) 
    plt.title('Pairwise Distance Matrix Between Class Centroids')
    plt.show()     



def get_next_batch(images1, labels1, data_loader, probability_matrix):
    batch_size = labels1.size(0)
    next_images = []
    next_labels = []
    train_dataset = data_loader.dataset
    label_list = data_loader.dataset.labels
    labels_array = np.array(label_list)
    
    for i in range(batch_size):
        current_label = labels1[i].item()
        
        # Get the probability distribution for the current class
        probabilities = probability_matrix[int(current_label)]
        
        # Sample the next class label based on the probability distribution
        next_class_label = np.random.choice(len(probabilities), p=probabilities)
        
        # randomly sample an image which belongs to the next class based on the next_class_label
        next_class_indices = np.where(labels_array == next_class_label)[0]

        next_idx = np.random.choice(next_class_indices)
        
        
        # Retrieve the next sample from the dataset
        next_image, next_label = train_dataset[next_idx]
        
        
        next_images.append(next_image)
        next_labels.append(next_label)
        
    #stack the images and labels into a batch tensor
    next_images = torch.stack(next_images)
    next_labels = torch.tensor(next_labels)
    
    return next_images, next_labels
    


#Testing

'''
# Step 3: Define data transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_path = "D:/anita/Research/iNaturalist/train/"
long_tail = False
random_select = False
train_images_list, train_labels_list, _, _ = load_imb_inaturalist(data_path, transform, transform, long_tail, random_select)
train_dataset = Imb_iNat_Dataset(data_path,train_images_list, train_labels_list,transform)

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

distance_matrix, class_labels = get_distance_matrix(dataloader)
visualize_distance_matrix(distance_matrix, class_labels)

probability_matrix = get_probability_matrix(distance_matrix)
visualize_probability_matrix(probability_matrix)

'''
