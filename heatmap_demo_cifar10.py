import torch
import torchvision
from torchvision import models, transforms
from gradcam import GradCAM
import utils as util
#from utils import load_image, save_img_with_heatmap, check_path_exist, apply_transforms, save_heatmap
import time
import json
#from utils import get_transform
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as transforms

def denormalize(image_batch, means=[0.5, 0.5, 0.5], stds=[0.5, 0.5, 0.5]):
    means = torch.tensor(means).view(1, 3, 1, 1)
    stds = torch.tensor(stds).view(1, 3, 1, 1)
    return image_batch * stds + means

def saliency_visualisation(img, saliency):
    fig, ax = plt.subplots(1,3)
    img = img*255
    img_heatmap = util.save_img_with_heatmap(img, saliency, None, style='zhou', normalise=True)
    # plt.imshow((img_heatmap[:, :, ::-1]).astype(np.uint8))
    ax[0].imshow((img[:, :, ::-1]).astype(np.uint8))
    ax[1].imshow((img_heatmap[:, :, ::-1]).astype(np.uint8))
    heatmap = util.save_heatmap(saliency, None, normalise=True)
    ax[2].imshow((heatmap[:, :, ::-1]).astype(np.uint8))
    plt.axis('off')

def main():

    transform_cifar10 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_imageNet = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor()           # Convert image to tensor
    ])

    model = models.resnet50(pretrained=True)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    model = model.cuda()
    model.eval()
    gc = GradCAM(model, target_layer='layer4')


    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    #get images from the testloader


    for i, data in enumerate(testloader):
        images, labels = data #images has shape [4, 3, 32, 32] 
        image_denormalized = denormalize(images)  
        images = images.cuda()
        start = time.time()
        saliencys, _ = gc(images, None)
        print('Total time: {:.2f} second'.format((time.time()-start)))
        threshold = 0.5  # adjust this value to suit your case

        for j, saliency in enumerate(saliencys):
            image = image_denormalized[j].squeeze(0).cpu().numpy() 
            image = np.transpose(image, (1, 2, 0))
            saliency = cv2.resize(np.squeeze(saliency), (32,32))
            image = cv2.resize(image, (32,32))
            print('Saliency generated by Grad-CAM')
            saliency_visualisation(image, saliency)

            #add bouding box around ROI
            # Convert the saliency map to binary
            _, binary_saliency = cv2.threshold(saliency, threshold, 255, cv2.THRESH_BINARY)

            # Convert binary_saliency to 8-bit image
            binary_saliency = np.uint8(binary_saliency)
            img_np = np.array(image)
             # Find contours in the binary image
            contours, _ = cv2.findContours(binary_saliency, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            '''
            # For each contour, find the bounding rectangle and draw it on the original image
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 1)
            '''

            #find the bounding box of the largest contour
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 1)

            # Display the image with bounding boxes
            plt.imshow(img_np)
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    main()


