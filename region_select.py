import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from datasets import load_dataset

class GradCAM:
    def __init__(self, model, target_layer='layer4'):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        self.handles = []
        
        # Register hooks
        for name, module in self.model.named_modules():
            if name == target_layer:
                self.handles.append(module.register_forward_hook(self._forward_hook))
                self.handles.append(module.register_backward_hook(self._backward_hook))
                break
    
    def _forward_hook(self, module, input, output):
        self.activations = output
        
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def get_cam(self, x, class_idx=None):
        """
        Generate CAM for input images (works for both ImageNet and CIFAR)
        Args:
            x: input tensor (B, C, H, W)
            class_idx: target class indices (optional)
        Returns:
            cam_maps: normalized activation maps
            probs: softmax probabilities
        """
        # Resize small images (e.g., CIFAR) to ImageNet size
        if x.shape[-1] < 224:
            x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
            
        batch_size = x.size(0)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        
        if class_idx is None:
            class_idx = logits.max(1)[-1]
            
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[torch.arange(batch_size), class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        cam = torch.mul(self.activations, weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize CAM to original input size
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # Normalize each CAM independently
        cam = cam.view(batch_size, -1)
        cam_min = cam.min(1, keepdim=True)[0]
        cam_max = cam.max(1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)
        cam = cam.view(batch_size, 1, x.shape[2], x.shape[3])
        
        return cam.detach(), probs.detach()
    
    def __call__(self, x, class_idx=None):
        return self.get_cam(x, class_idx)

def generate_mixing_masks(cam_maps, lam):
    """Generate binary masks from CAM maps"""
    cam_maps = cam_maps.cpu().numpy()
    batch_size = cam_maps.shape[0]
    masks = []
    actual_lams = []
    
    threshold = 1 - lam
    
    for i in range(batch_size):
        cam = cam_maps[i, 0]
        cam = cv2.GaussianBlur(cam, (5, 5), 0)
        mask = (cam > threshold).astype(np.float32)
        
        # Refinement
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        actual_lam = 1 - mask.mean()
        masks.append(mask)
        actual_lams.append(actual_lam)
        
    masks = np.array(masks)  # Combine the list of numpy arrays into a single numpy ndarray
    masks = torch.from_numpy(masks).float().unsqueeze(1).cuda()  # Convert to PyTorch tensor

    
    #masks = torch.FloatTensor(masks).unsqueeze(1).cuda()
    actual_lams = torch.FloatTensor(actual_lams).cuda()
    
    return masks, actual_lams

def visualize_cam(images, cam_maps, masks=None):
    """Visualize original images, CAM maps, and masks"""
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    images_denorm = images.detach() * std + mean
    
    num_images = images.size(0)
    num_cols = 3 if masks is None else 4
    plt.figure(figsize=(4*num_cols, 4*num_images))
    
    for idx in range(num_images):
        # Original image
        plt.subplot(num_images, num_cols, idx*num_cols + 1)
        img = images_denorm[idx].cpu().numpy().transpose(1, 2, 0)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f'Original {idx+1}')
        plt.axis('off')
        
        # CAM heatmap
        plt.subplot(num_images, num_cols, idx*num_cols + 2)
        cam = cam_maps[idx, 0].cpu().numpy()
        plt.imshow(cam, cmap='jet')
        plt.title(f'CAM Map {idx+1}')
        plt.axis('off')
        
        # Overlay
        plt.subplot(num_images, num_cols, idx*num_cols + 3)
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.7 * img + 0.3 * heatmap/255)
        overlay = np.clip(overlay / overlay.max(), 0, 1)
        plt.imshow(overlay)
        plt.title(f'Overlay {idx+1}')
        plt.axis('off')
        
        if masks is not None:
            plt.subplot(num_images, num_cols, idx*num_cols + 4)
            mask = masks[idx, 0].cpu().numpy()
            plt.imshow(mask, cmap='gray')
            plt.title(f'Mask {idx+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def get_bounding_box_from_mask(mask):
    """Convert mask to bounding box coordinates"""
    # mask: (H, W) binary mask
    y_indices, x_indices = np.where(mask > 0.5)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    
    x1, x2 = x_indices.min(), x_indices.max()
    y1, y2 = y_indices.min(), y_indices.max()
    
    # Add small padding
    pad = 5
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(mask.shape[1], x2 + pad)
    y2 = min(mask.shape[0], y2 + pad)
    
    return (x1, y1, x2, y2)
    
def visualize_cam_with_bbox(images, cam_maps, masks=None):
    """
    Visualize original images, CAM maps, masks, and bounding boxes
    Args:
        images: (B, C, H, W) tensor of images
        cam_maps: (B, 1, H, W) tensor of CAM maps
        masks: (B, 1, H, W) tensor of binary masks
    """
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    images_denorm = images.detach() * std + mean
    
    num_images = images.size(0)
    num_cols = 4  # Original, CAM, Mask+Box, Overlay+Box
    plt.figure(figsize=(4*num_cols, 4*num_images))
    
    for idx in range(num_images):
        # Original image
        plt.subplot(num_images, num_cols, idx*num_cols + 1)
        img = images_denorm[idx].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f'Original {idx+1}')
        plt.axis('off')
        
        # CAM heatmap
        plt.subplot(num_images, num_cols, idx*num_cols + 2)
        cam = cam_maps[idx, 0].cpu().numpy()
        plt.imshow(cam, cmap='jet')
        plt.title(f'CAM Map {idx+1}')
        plt.axis('off')
        
        # Mask with bounding box
        plt.subplot(num_images, num_cols, idx*num_cols + 3)
        if masks is not None:
            mask = masks[idx, 0].cpu().numpy()
            plt.imshow(mask, cmap='gray')
            # Draw bounding box
            bbox = get_bounding_box_from_mask(mask)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
        plt.title(f'Mask+Box {idx+1}')
        plt.axis('off')
        
        # Overlay with bounding box
        plt.subplot(num_images, num_cols, idx*num_cols + 4)
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.7 * img + 0.3 * heatmap/255)
        overlay = np.clip(overlay / overlay.max(), 0, 1)
        plt.imshow(overlay)
        # Draw bounding box on overlay
        if masks is not None:
            bbox = get_bounding_box_from_mask(mask)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
        plt.title(f'Overlay+Box {idx+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_test_batch(image_paths, is_cifar=False):
    """
    Create a batch of images from local paths
    Args:
        image_paths: list of image file paths
        is_cifar: whether to use CIFAR size and normalization
    Returns:
        batch_tensor: (N, C, H, W) tensor normalized and ready for model
    """
    if is_cifar:
        # CIFAR normalization and size
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    else:
        # ImageNet normalization and size
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # Load and transform images
    batch = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            batch.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    
    # Stack into batch
    batch_tensor = torch.stack(batch).cuda()
    return batch_tensor


def augment_roi(roi, augmentation_types=None):

    if augmentation_types is None:
        augmentation_types = ['all']
        
    augmented_rois = []
    roi_np = roi.permute(1, 2, 0).cpu().numpy()
    h, w, _ = roi_np.shape

    # Original ROI
    augmented_rois.append(roi)
    
    if 'scale' in augmentation_types or 'all' in augmentation_types:
        for scale in [0.8, 1.2]:
            scaled_size = (int(w * scale), int(h * scale))
            scaled_roi = cv2.resize(roi_np, scaled_size, interpolation=cv2.INTER_LINEAR)
            scaled_roi = torch.from_numpy(scaled_roi).permute(2, 0, 1).float().cuda()
            augmented_rois.append(scaled_roi)
    
    if 'rotate' in augmentation_types or 'all' in augmentation_types:
        # Multiple rotation angles
        center = (w // 2, h // 2)
        for angle in [-30, 30]:  # Less extreme angles
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_roi = cv2.warpAffine(roi_np, rotation_matrix, (w, h), 
                                       borderMode=cv2.BORDER_REFLECT)
            rotated_roi = torch.from_numpy(rotated_roi).permute(2, 0, 1).float().cuda()
            augmented_rois.append(rotated_roi)
    
    if 'flip' in augmentation_types or 'all' in augmentation_types:
        flipped_roi = cv2.flip(roi_np, 1)
        flipped_roi = torch.from_numpy(flipped_roi).permute(2, 0, 1).float().cuda()
        augmented_rois.append(flipped_roi)
    
    return augmented_rois

# Mix augmented ROIs with different backgrounds

def mix_with_augmented_roi(backgrounds, bbox, augmented_rois):

    augmented_images = []
    lam_list = []
    x1, y1, x2, y2 = bbox
    original_w = x2 - x1
    original_h = y2 - y1
    
    for idx, aug_roi in enumerate(augmented_rois):
        new_image = backgrounds[idx].clone()
        
        # Get the size of the augmented ROI
        roi_h, roi_w = aug_roi.shape[1], aug_roi.shape[2]
        
        if roi_h != original_h or roi_w != original_w:
            # Calculate the maximum space we can use within the original bbox
            max_w = min(roi_w, original_w)
            max_h = min(roi_h, original_h)
            
            # For larger ROIs (rotated or scaled up), take the center portion
            if roi_w > original_w or roi_h > original_h:
                start_roi_y = (roi_h - max_h) // 2
                start_roi_x = (roi_w - max_w) // 2
                roi_portion = aug_roi[:, start_roi_y:start_roi_y+max_h, 
                                       start_roi_x:start_roi_x+max_w]
                new_image[:, y1:y1+max_h, x1:x1+max_w] = roi_portion
            else:
                # For smaller ROIs (scaled down), center them in the bbox
                start_y = y1 + (original_h - roi_h) // 2
                start_x = x1 + (original_w - roi_w) // 2
                new_image[:, start_y:start_y+roi_h, start_x:start_x+roi_w] = aug_roi
        else:
            new_image[:, y1:y2, x1:x2] = aug_roi
            
        augmented_images.append(new_image)
        
        # Calculate lambda: scaling factor affects the ROI area
        scaling_factor = (roi_h * roi_w) / (original_h * original_w)
        original_foreground_area = original_h * original_w
        scaled_foreground_area = scaling_factor * original_foreground_area
        
        total_area = new_image.shape[1] * new_image.shape[2]  # H * W of the background
        lam = scaled_foreground_area / total_area
        lam_list.append(lam)
    
    return augmented_images, lam_list


def visualize_augmented_results(original_image, augmented_images, bbox):
    """
    Visualize original image and all augmented versions.
    
    Args:
        original_image (torch.Tensor): Original image (C, H, W)
        augmented_images (list): List of augmented images
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
    """
    num_images = len(augmented_images) + 1
    fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 4))
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    
    # Show original image
    img = original_image * std + mean
    img = img.permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(np.clip(img, 0, 1))
    axes[0].add_patch(plt.Rectangle((bbox[0], bbox[1]), 
                                  bbox[2]-bbox[0], bbox[3]-bbox[1],
                                  fill=False, color='red'))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show augmented images
    for i, aug_img in enumerate(augmented_images):
        img = aug_img* std + mean
        img = img.permute(1, 2, 0).cpu().numpy()
        axes[i+1].imshow(np.clip(img, 0, 1))
        axes[i+1].add_patch(plt.Rectangle((bbox[0], bbox[1]), 
                                        bbox[2]-bbox[0], bbox[3]-bbox[1],
                                        fill=False, color='red'))
        axes[i+1].set_title(f'Augmented {i+1}')
        axes[i+1].axis('off')
    
    plt.tight_layout()    
    save_path = "plots_from_xai_cmo_augmentation.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {save_path}")
    
    plt.show()

def generate_mixed_images_with_augmentation(images, backgrounds, masks, types=['scale', 'rotate', 'flip']):
    """
    Process images with augmentation and mixing.
    
    Args:
        images (torch.Tensor): Input images (N, C, H, W)
        backgrounds (torch.Tensor): Background images (N * num_augs, C, H, W)
        masks (torch.Tensor): Mixing masks (N, 1, H, W)
        types (list): List of augmentation types to apply
    """
    augmented_images = []
    lam_list = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #check if the number of backgrounds is 6 times the number of images. if not, randomly select backgorunds for the remaining images.
    if len(backgrounds) != 6*len(images):
        # Randomly select backgrounds for the remaining images
        num_remaining = len(images) * 6 - len(backgrounds)
        indices = torch.randint(0, len(backgrounds), (num_remaining,))
        remaining_backgrounds = backgrounds[indices]

        # Concatenate the sampled backgrounds to the existing ones
        backgrounds = torch.cat([backgrounds, remaining_backgrounds], dim=0)
    
    # Calculate number of augmentations per image
    num_augs_per_image = 6  # 1 original + 2 scales + 2 rotations + 1 flip
    
    # Process each image in the batch
    for idx in range(len(images)):
        # Get mask for current image
        mask = masks[idx, 0].cpu().numpy()
        
        # Calculate background slice indices for this image
        bg_start = idx * num_augs_per_image
        bg_end = (idx + 1) * num_augs_per_image
        
        # Get corresponding backgrounds for this image
        image_backgrounds = backgrounds[bg_start:bg_end]
    
        # Get bounding box
        bbox = get_bounding_box_from_mask(mask)
        if bbox is not None:
            # Get ROI
            x1, y1, x2, y2 = bbox
            roi = images[idx, :, y1:y2, x1:x2]
            
            # Generate augmented ROIs
            augmented_rois = augment_roi(roi, types)
            
            # Apply augmented ROIs to different background images
            mixed_images, lam_list = mix_with_augmented_roi(
                #images[idx], 
                image_backgrounds, 
                bbox, 
                augmented_rois
            )
            
            # Move augmented images to cuda
            mixed_images = [img.to(device) for img in mixed_images]
            
            augmented_images.extend(mixed_images)
        
            # Visualize results
            # visualize_augmented_results(images[idx].to(device), mixed_images, bbox)
            #rint(f"Processed image {idx+1}: ROI shape {roi.shape}, Generated {len(augmented_rois)} augmentations")
        else:
            print(f"No valid bounding box found for image {idx+1}")
            
    return augmented_images, lam_list


def generate_mixed_images_without_augmentation(images, backgrounds, masks):
    """
    Generate mixed images without applying augmentations in a 1-to-1 manner.
    
    Args:
        images (torch.Tensor): Input images (N, C, H, W).
        backgrounds (torch.Tensor): Background images (N, C, H, W).
        masks (torch.Tensor): Binary masks (N, 1, H, W).
    
    Returns:
        mixed_images: List of mixed images (N).
        lam_list: List of lambda values representing the ratio of the foreground region to the whole background image.
    """
    mixed_images = []
    lam_list = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Iterate over each image, background, and mask
    for idx in range(len(images)):
        # Get the mask for the current image
        mask = masks[idx, 0].cpu().numpy()
        
        # Extract the bounding box from the mask
        bbox = get_bounding_box_from_mask(mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            
            # Extract ROI from the original image
            roi = images[idx, :, y1:y2, x1:x2]
            
            # Clone the corresponding background
            new_image = backgrounds[idx].clone()
            
            # Insert the ROI into the background
            new_image[:, y1:y2, x1:x2] = roi
            
            # Compute the foreground region area
            foreground_area = (y2 - y1) * (x2 - x1)
            
            # Compute the total background area
            total_area = new_image.size(1) * new_image.size(2)  # H * W
            
            # Compute lambda: foreground area / total area
            lam = foreground_area / total_area
            lam_list.append(lam)
            
            # Add the mixed image to the list
            mixed_images.append(new_image.to(device))
        else:
            print(f"No valid bounding box found for image {idx+1}")
    
    return mixed_images, lam_list

def get_backgrounds(loader, num_batches=6):
    """
    Fetches `num_batches` of images from the DataLoader to create a diverse backgrounds batch.
    Args:
        loader (DataLoader): PyTorch DataLoader object.
        num_batches (int): Number of batches to fetch.
    Returns:
        torch.Tensor: Combined tensor of images from all fetched batches.
    """
    backgrounds = []
    batch_count = 0
    for batch in loader:
        if batch_count == num_batches:
            break
        images = batch['img']
        backgrounds.append(images)
        batch_count += 1

    # Concatenate all batches along the batch dimension
    #backgrounds = [torch.tensor(bg).cuda() if not isinstance(bg, torch.Tensor) else bg for bg in backgrounds]
    return torch.cat(backgrounds, dim=0).cuda()

# Example usage:
def main():
    
    # Setup models
    model = models.resnet50(pretrained=True).cuda().eval()
    grad_cam = GradCAM(model, 'layer4')
    
    #test for cifar data.
    train_dataset = load_dataset("tomas-gajarsky/cifar100-lt", 'r-10', split="train")
    
        # Convert the dataset to include tensors
    def transform_batch(examples):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        images = []
        for img in examples['img']:
            img = Image.fromarray(np.array(img)).resize((224, 224))
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            img = (img - mean[:, None, None]) / std[:, None, None]
            images.append(img)
        return {'img': torch.stack(images), 'label': examples['fine_label']}

    train_dataset.set_transform(transform_batch)
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=5, shuffle=True,
            num_workers=0, pin_memory=True)
    
    print(f"Number of batches: {len(train_loader)}")

    
    #get two batches of images with batch size 5 for testing
    for i, batch in enumerate(train_loader):
        if i == 2:
            break
        images = batch['img'].cuda()
   
        print(f"Batch {i+1}: {images.size()} images")
        
        backgrounds = get_backgrounds(train_loader, num_batches=6)
        print(f"Backgrounds shape: {backgrounds.shape}")
        
        print("Backgrounds shape:", backgrounds.shape)
        
        # Process images
        cam_maps, probs = grad_cam(images)
        print("CAM maps shape:", cam_maps.shape)
        
        masks, actual_lams = generate_mixing_masks(cam_maps, lam=0.7)
        
        visualize_cam_with_bbox(images, cam_maps, masks)
        
        
        # Generate mixed images
        generate_mixed_images_with_augmentation(images, backgrounds, masks, types=['scale', 'rotate', 'flip'])
    
   
if __name__ == "__main__":
    main()