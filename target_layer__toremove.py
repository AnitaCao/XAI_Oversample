#!/usr/bin/env python3

class TargetLayer(object):
    def __init__(self, arch, target_layer_name):
        super(TargetLayer, self).__init__()
        self.arch = arch
        self.target_layer_name = target_layer_name

    def forward(self, model_type):
        if 'resnet' in model_type.lower():
            target_layer = self.find_resnet_layer(self.arch, self.target_layer_name)
        else:
            target_layer = self.find_layer()
        return target_layer

    def __call__(self, model_type):
        return self.forward(model_type)

    @staticmethod
    def find_resnet_layer(arch, target_layer_name):
        """Find resnet layer to calculate GradCAM and GradCAM++
        Args:
            arch: default torchvision densenet models
            target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
                target_layer_name = 'conv1'
                target_layer_name = 'layer1'
                target_layer_name = 'layer1_basicblock0'
                target_layer_name = 'layer1_basicblock0_relu'
                target_layer_name = 'layer1_bottleneck0'
                target_layer_name = 'layer1_bottleneck0_conv1'
                target_layer_name = 'layer1_bottleneck0_downsample'
                target_layer_name = 'layer1_bottleneck0_downsample_0'
                target_layer_name = 'avgpool'
                target_layer_name = 'fc'
        Return:
            target_layer: found layer. this layer will be hooked to get forward/backward pass information.
        """
        if target_layer_name is None:
            target_layer_name = 'layer4'

        if 'layer' in target_layer_name:
            hierarchy = target_layer_name.split('_')
            layer_num = int(hierarchy[0].lstrip('layer'))
            if layer_num == 1:
                target_layer = arch.layer1
            elif layer_num == 2:
                target_layer = arch.layer2
            elif layer_num == 3:
                target_layer = arch.layer3
            elif layer_num == 4:
                target_layer = arch.layer4
            else:
                raise ValueError('unknown layer : {}'.format(target_layer_name))

            if len(hierarchy) >= 2:
                bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
                target_layer = target_layer[bottleneck_num]

            if len(hierarchy) >= 3:
                target_layer = target_layer._modules[hierarchy[2]]

            if len(hierarchy) == 4:
                target_layer = target_layer._modules[hierarchy[3]]

        else:
            target_layer = arch._modules[target_layer_name]

        return target_layer


    @staticmethod
    def find_layer(arch, target_layer_name):
        """Find target layer to calculate CAM.
            : Args:
                - **arch - **: Self-defined architecture.
                - **target_layer_name - ** (str): Name of target class.
            : Return:
                - **target_layer - **: Found layer. This layer will be hooked to get forward/backward pass information.
        """

        if target_layer_name.split('_') not in arch._modules.keys():
            raise Exception("Invalid target layer name.")
        target_layer = arch._modules[target_layer_name]
        return target_layer