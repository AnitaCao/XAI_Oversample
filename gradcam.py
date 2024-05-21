import torch
import torch.nn.functional as F

class BaseCAM(object):
    def __init__(self, model, target_layer="module.layer4.2"):
        super(BaseCAM, self).__init__()
        self.model = model.eval()
        self.gradients = dict()
        self.activations = dict()

        for module in self.model.named_modules():
            if module[0] == target_layer:
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    def forward(self, x, class_idx=None, retain_graph=False):
        raise NotImplementedError

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)
    

class GradCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2"):
        super().__init__(model, target_layer)

    def forward(self, x, class_idx=None, retain_graph=False):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)

        x = x.to(next(self.model.parameters()).device)
        b, c, h, w = x.size()

        # predication on raw x
        #print("checking input x values to see if there is any non or inf values:")
        #print(torch.isnan(x).any(), torch.isinf(x).any())

        logit = self.model(x) #logit has shape [b, num_classes]
        softmax = F.softmax(logit, dim=1)

        if class_idx is None:
            indices = logit.max(1)[-1]
            score = logit[torch.arange(logit.shape[0]), indices]#.squeeze()    
        else:
            score = logit[:, class_idx]#.squeeze()
            # score = logit[:, class_idx]

        if b > 1:
            retain_graph = True

        self.model.zero_grad()
        gradients_list = []

        
        for i, item in enumerate(score):  
            #print(item)
            if item.numel() == 1:  # Check if item is a scalar
                item.backward(retain_graph=retain_graph)  
                gradients = self.gradients['value'].data[i]
                gradients_list.append(gradients)
            else:
                print(f"Element {i} of 'score' is not a scalar, it has {item.numel()} elements")

        gradients = torch.stack(gradients_list, dim=0)
        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)

        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        # saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        # saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

        saliency_map_shape = saliency_map.shape
        saliency_map = saliency_map.view(saliency_map.shape[0], -1)
        saliency_map_min, saliency_map_max = saliency_map.min(1, keepdim=True)[0], saliency_map.max(1, keepdim=True)[0]
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
        saliency_map = saliency_map.view(saliency_map_shape)

        # import cv2
        # import numpy as np
        # map = saliency_map.cpu().data
        # map = cv2.applyColorMap(np.uint8(255 * map.squeeze()), cv2.COLORMAP_JET)
        # cv2.imwrite('test.jpg', map)

        return saliency_map.detach().cpu().numpy(), softmax.detach()

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)