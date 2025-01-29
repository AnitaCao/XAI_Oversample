# original code: https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
import math
import os
import sys
import random
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
#from tensorboardX import SummaryWriter
from imbalance_data.lt_data import LT_Dataset, Imb_Dataset, LT_Dataset_ROI
from losses import LDAMLoss, BalancedSoftmaxLoss
import warnings
from torch.nn import Parameter
import torch.nn.functional as F
from util.util import *
from util.randaugment import rand_augment_transform
import util.moco_loader as moco_loader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_utils
import region_select_copy as region_select
from opts import parser
from PIL import Image

import torchvision.transforms.functional as TF


best_acc1 = 0

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, args.loss_type, args.train_rule, args.cut_mix, str(args.imb_factor),
        str(args.rand_number),
        str(args.mixup_prob), args.exp_str])
    prepare_folders(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda
    args.gpu = gpu
   
    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = args.num_classes
    model = getattr(models, args.arch)(pretrained=False)
    if args.num_classes == 10:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    print(model)

 
    XAI_model = models.resnet50(pretrained=True) # load a pretrained model for XAI heatmap generation
    XAI_model = XAI_model.cuda(args.gpu)
    XAI_model.eval()
    
    gc = region_select.GradCAM(XAI_model, target_layer='layer4')
    
    if args.loss_type == 'LDAM':
        num_ftrs = model.fc.in_features
        model.fc = NormedLinear(num_ftrs, num_classes)
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # optionally resume from a checkpoint

    cudnn.benchmark = True

    if args.use_randaug:
        print("use randaug!!")
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45),
                         img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            normalize,
        ]
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    else:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_mask = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor(),  # Convert mask to tensor (values remain as integers)
        ])

    print(args)
    '''
    Current dataset is a long tailed dataset with 1000 classes (from CMO paper).
    TODO: Create different Datases with different imbalance ratio. 
    '''
    
    if args.num_classes == 1000:
        print("Using ImageNet_LT dataset")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(script_dir, 'imbalance_data', 'ImageNet_LT_train.txt')
        val_path = os.path.join(script_dir, 'imbalance_data', 'ImageNet_LT_val.txt')
        roi_path = os.path.join(script_dir, 'imbalance_data', 'dino_sam_region_images')
        
        train_dataset = LT_Dataset(args.root, train_path, transform_train,
                               use_randaug=args.use_randaug)
        val_dataset = LT_Dataset(args.root, val_path, transform_val)
        
        roi_dataset = LT_Dataset_ROI(roi_path, transform_train, transform_mask)
        print("Number of classes in roi_dataset: ", len(np.unique(roi_dataset.targets)))
        
        num_classes = len(np.unique(train_dataset.targets))
        
        assert num_classes == 1000
    else:
        return
        #data_path = os.path.join(args.root, 'train')
        #train_dataset, val_dataset = load_imb_imagenet(data_path, transform_train, transform_val)
        #num_classes = len(np.unique(train_dataset.targets))

    print("Number of classes: ", num_classes)
    print("Number of training samples: ", len(np.unique(val_dataset.targets)))
    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    print('cls num list:')
    print(cls_num_list)
    
    cls_num_list_in_roi = [0] * num_classes
    for label_ in roi_dataset.targets:
        cls_num_list_in_roi[label_] += 1
    print('cls_num_list_in_roi:')
    print(cls_num_list_in_roi)

    #args.cls_num_list = cls_num_list
    args.cls_num_list = cls_num_list
    train_cls_num_list = np.array(cls_num_list)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    weighted_train_loader = None
    weighted_cls_num_list = [0] * num_classes

    if args.cut_mix.startswith('CMO'):
        
        cls_weight = 1.0 / (np.array(cls_num_list) ** args.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        
        samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        
        print(samples_weight)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                  replacement=True)
        weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            num_workers=args.workers, pin_memory=True,
                                                            sampler=weighted_sampler, drop_last=True)
        
    if args.cut_mix == "CMO_OBJ":
        cls_weight_roi = 1.0 / (np.array(cls_num_list_in_roi) ** args.weighted_alpha)
        cls_weight_roi = cls_weight_roi / np.sum(cls_weight_roi) * len(cls_num_list_in_roi)
        
        samples_weight_roi = np.array([cls_weight_roi[t] for t in roi_dataset.targets])
        samples_weight_roi = torch.from_numpy(samples_weight_roi)
        samples_weight_roi = samples_weight_roi.double()
        #print(samples_weight_roi)
        weighted_sampler_roi = torch.utils.data.WeightedRandomSampler(samples_weight_roi, len(samples_weight_roi),
                                                                  replacement=True)
        roi_loader = torch.utils.data.DataLoader(roi_dataset, batch_size=args.batch_size, num_workers=args.workers, sampler=weighted_sampler_roi, drop_last=True)
        
        '''
        #testing the weighted sampler
        for i, batch in enumerate(weighted_train_loader):
            input, target = batch['img'], batch['label']
            #print("target: ", target)
            for t in target:
                weighted_cls_num_list[t] += 1
        print("weighted cls num list:")
        print(weighted_cls_num_list)
        print("total number of samples: ", sum(weighted_cls_num_list))
        print("weighted sampler testing finished")
        '''    

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda(args.gpu)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    #tf_writer = SummaryWriter(log_dir=os.path.join(args["root_log"], args["store_name"]))

    start_time = time.time()
    print("Training started!")

    for epoch in range(args.start_epoch, args.epochs):

        if args.use_randaug:
            paco_adjust_learning_rate(optimizer, epoch, args)
        else:
            adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'CBReweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')


        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'BS':
            criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list_cuda).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, model, gc, criterion, optimizer, epoch, args, log_training, weighted_train_loader, roi_loader)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        #tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
        }, is_best, epoch + 1)

    end_time = time.time()

    print("It took {} to execute the program".format(hms_string(end_time - start_time)))
    log_testing.write("It took {} to execute the program".format(hms_string(end_time - start_time)) + '\n')
    log_testing.flush()


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def train(train_loader, model, gradcam, criterion, optimizer, epoch, args, log, weighted_train_loader=None, roi_loader=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    end = time.time()
    if args.cut_mix.startswith('CMO') and args.start_cut_mix < epoch < (args.epochs - args.end_cut_mix):
        inverse_iter = iter(weighted_train_loader)

    for i, (input, target, path) in enumerate(train_loader):
        
        if args.cut_mix.startswith('CMO') and args.start_cut_mix < epoch < (
                args.epochs - args.end_cut_mix):
            try:
                input2, target2, path2 = next(inverse_iter) 
            except:
                inverse_iter = iter(weighted_train_loader)
                input2, target2, path2 = next(inverse_iter)
                
            input2 = input2[:input.size()[0]] # make sure the batch size is the same
            target2 = target2[:target.size()[0]] # make sure the batch size is the same
            input2 = input2.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        #input = input.cuda(args.gpu, non_blocking=True)
        #target = target.cuda(args.gpu, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True) 
       
        r = np.random.rand(1)
        #r = 0.01
        if args.cut_mix == 'CMO' and args.start_cut_mix < epoch < (
                args.epochs - args.end_cut_mix) and r < args.mixup_prob:
            
            lam = np.random.beta(args.beta, args.beta)
            
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2] # generate mixed sample
            
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))  # adjust lambda to exactly match pixel ratio
            
            output = model(input)
            loss = criterion(output, target) * lam + criterion(output, target2) * (1. - lam)

        elif args.cut_mix == 'CMO_XAI' and args.start_cut_mix < epoch < (
                args.epochs - args.end_cut_mix) and r < args.mixup_prob:
          
            lam = np.random.beta(args.beta, args.beta)
            start = time.time()
            
            cam_maps, probs = gradcam(input2)
            masks, lams_ori = region_select.generate_mixing_masks(cam_maps, lam=0.7)

            time1 = time.time()
            lam_list = []
            
            if args.data_aug:
                # get 6 batchs of images from trainloader to use as backgrounds
                backgrounds = []
                backgrounds_labels = []
                for j in range(6):
                    batch = next(iter(train_loader))
                    b_imgs = batch[0][:input.size()[0]]
                    b_labels = batch[1][:input.size()[0]]
                    if isinstance(b_imgs, list):
                        # Convert nested list to tensor
                        b_imgs = torch.stack([torch.stack([torch.stack(channel) for channel in img], dim=0) for img in b_imgs], dim=0).permute(3,0,1,2)
                        b_imgs = b_imgs.float()
                    backgrounds.append(b_imgs)
                    backgrounds_labels.append(b_labels)
                    
                backgrounds = torch.cat(backgrounds, dim=0)
                backgrounds_labels = torch.cat(backgrounds_labels,dim=0)
                
                mixed_imgs_list, lam_list = region_select.generate_mixed_images_with_augmentation(input2, backgrounds, masks, types=['scale', 'rotate', 'flip'])  
                
                input = torch.stack(mixed_imgs_list, dim=0).cuda(args.gpu, non_blocking=True)
                
                target = backgrounds_labels.cuda(args.gpu, non_blocking=True)  #background labels
                target2 = target2.repeat_interleave(6, dim=0).cuda(args.gpu, non_blocking=True)  #foreground labels
                
            else:
                backgrounds = input
                input, lam_list = region_select.generate_mixed_images_without_augmentation(input2, backgrounds, masks)


            input = input.cuda(args.gpu, non_blocking=True)
        
            output = model(input)
            loss = criterion(output, target) * (1. - torch.tensor(lam_list).cuda(args.gpu)) + criterion(output, target2) *  torch.tensor(lam_list).cuda(args.gpu)
            loss = loss.mean()        
        elif args.cut_mix == 'CMO_OBJ' and args.start_cut_mix< epoch < (
                args.epochs - args.end_cut_mix) and r < args.mixup_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            print("---Calling Object Detection. Batch number: ", i)
            start = time.time()
            
            input2= next(iter(roi_loader))  #get a batch from roi_loader as foreground  
            foreground = input2[:input.size()[0]]
            
            input2_imgs = foreground[0].cuda(args.gpu, non_blocking=True)
            input2_labels = foreground[1].cuda(args.gpu, non_blocking=True)
            input2_masks = foreground[2].cuda(args.gpu, non_blocking=True)
            
            if args.data_aug:
                print("Data Augmentation")  
                
                augmented_rois, augmented_masks = augment_rois_and_masks(
                input2_imgs, input2_masks, augmentation_types=['scale', 'rotate', 'flip'])
                
                lams = 1 - augmented_masks.sum(dim=(1,2,3)) / (224*224)
                # get 6 batchs of images from trainloader to use as backgrounds
                backgrounds = []
                backgrounds_labels = []
                for j in range(6):
                    batch = next(iter(train_loader))
                    b_imgs = batch[0][:input.size()[0]]
                    b_labels = batch[1][:input.size()[0]]
                    if isinstance(b_imgs, list):
                        # Convert nested list to tensor
                        b_imgs = torch.stack([torch.stack([torch.stack(channel) for channel in img], dim=0) for img in b_imgs], dim=0).permute(3,0,1,2)
                        b_imgs = b_imgs.float()
                    backgrounds.append(b_imgs)
                    backgrounds_labels.append(b_labels)
                    
                backgrounds = torch.cat(backgrounds, dim=0)
                
                mixed_input = backgrounds.cuda(args.gpu, non_blocking=True) * (1. - augmented_masks.cuda(args.gpu, non_blocking=True)) + augmented_rois.cuda(args.gpu, non_blocking=True) * augmented_masks.cuda(args.gpu, non_blocking=True)
                
                #get foreground labels
                target = torch.cat(backgrounds_labels, dim=0).cuda(args.gpu, non_blocking=True)
                target2 = input2_labels.repeat_interleave(6, dim=0).cuda(args.gpu, non_blocking=True)
                 
                              
            else:
                #get the mask roi ratio, lam should a list of lam values for each mask in the batch
                lams = 1 - input2_masks.sum(dim=(1,2,3)) / (224*224)

                # Patch ROIs into the background batch
                mixed_input = input * (1 - input2_masks) + input2_imgs.cuda(args.gpu, non_blocking=True) * input2_masks
                target2 = input2_labels.cuda(args.gpu, non_blocking=True)
            
            output = model(mixed_input)
            loss = criterion(output, target) * lams.cuda(args.gpu) + criterion(output, target2) * (1. - lams.cuda(args.gpu))
            loss = loss.mean()

            '''
            # Unnormalize input2_imgs for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(args.gpu)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda(args.gpu)
            # visualize one input and one mask and the mixed_input
            input_img = input[0].cpu()
            input_img = input_img * std[0].cpu() + mean[0].cpu()  # Reverse normalization
            input_img_np = input_img.permute(1, 2, 0).numpy()
            input_img_np = np.clip(input_img_np, 0, 1)
            plt.imshow(input_img_np)
            plt.axis('off')
            plt.savefig('input_image__.png')
            
            input2_img = input2_imgs[0].cpu()
            input2_img = input2_img * std[0].cpu() + mean[0].cpu()  # Reverse normalization
            input2_img_np = input2_img.permute(1, 2, 0).numpy()
            input2_img_np = np.clip(input2_img_np, 0, 1)
            plt.imshow(input2_img_np)
            plt.axis('off')
            plt.savefig('input2_image__.png')
            
            mixed_image = mixed_input[0].cpu()
            mixed_image = mixed_image * std[0].cpu() + mean[0].cpu()  # Reverse normalization
            mixed_image_np = mixed_image.permute(1, 2, 0).numpy()
            mixed_image_np = np.clip(mixed_image_np, 0, 1)
            plt.imshow(mixed_image_np)
            plt.axis('off')
            plt.savefig('mixed_input_image__.png')
            '''           
                     
        else: 
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            output = model(input) #output.size() = [128, 10], 128 is batch size, 10 is number of classes
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    #tf_writer.add_scalar('loss/train', losses.avg, epoch)
    #tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    #tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    #tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_bbox_withcenter(size, lam, cx, cy):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch, args, log=None, flag='val'):
    def compute_classwise_accuracy(conf_matrix):
        cls_cnt = conf_matrix.sum(axis=1)
        cls_hit = np.diag(conf_matrix)
        cls_acc = np.where(cls_cnt > 0, cls_hit / cls_cnt, 0)  # Avoid division by zero
        return cls_acc, cls_hit, cls_cnt
    
    def compute_imbalance_metrics(cls_acc):
        """Compute Gmean, ACSA, and overall accuracy."""
        gmean = np.prod(cls_acc) ** (1.0 / len(cls_acc))  # Geometric mean accuracy
        acsa = np.mean(cls_acc)  # Average class-specific accuracy
        acc = np.sum(cls_hit) / np.sum(cls_cnt)  # Overall accuracy
        return gmean, acsa, acc    
    
    batch_time, losses, top1, top5 = [AverageMeter(name, fmt) for name, fmt in 
                                      [('Time', ':6.3f'), ('Loss', ':.4e'), ('Acc@1', ':6.2f'), ('Acc@5', ':6.2f')]]

    # switch to evaluate mode
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            input, target = batch[0], batch[1]
        
            if isinstance(input, list):
                # Convert nested list to tensor
                input = torch.stack([torch.stack([torch.stack(channel) for channel in img], dim=0) for img in input], dim=0).permute(3,0,1,2)
                input = input.float()
            
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_acc, cls_hit, cls_cnt = compute_classwise_accuracy(cf)
        gmean, acsa, acc = compute_imbalance_metrics(cls_acc)
        
        out_gm_acsa_acc = f'{flag} Gmean: {gmean:.3f}, ACSA: {acsa:.3f}, ACC: {acc:.3f}'
        out_cls_acc = f'{flag} Class Accuracy: {np.array2string(cls_acc, separator=",", formatter={"float_kind": lambda x: "%.3f" % x})}'
        
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        
        output = (f'{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.5f}')
        
        print(output)
        if log:
            log.write(output + '\n')
            log.write(out_gm_acsa_acc + '\n')
            log.flush()
        #print(out_cls_acc)

        if args.imb_factor == 0.01:
            many_shot = train_cls_num_list > 100
            medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
            few_shot = train_cls_num_list < 20
            print("many avg, med avg, few avg", float(sum(cls_acc[many_shot]) * 100 / sum(many_shot)),
                  float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot)),
                  float(sum(cls_acc[few_shot]) * 100 / sum(few_shot)))

        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.write(out_gm_acsa_acc + '\n')
            log.flush()

        #tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
        #tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        #tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        #tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def paco_adjust_learning_rate(optimizer, epoch, args):
    # experiments as PaCo (ICCV'21) setting.
    warmup_epochs = 10
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= warmup_epochs:
        lr = args.lr / warmup_epochs * (epoch + 1)
    elif epoch > 360:
        lr = args.lr * 0.01
    elif epoch > 320:
        lr = args.lr * 0.1
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def augment_rois_and_masks(rois, masks, augmentation_types=None):
    """
    Augment a batch of foreground ROIs and their corresponding masks with consistent transformations.

    Parameters:
        rois: Tensor [B, C, H, W] - Batch of foreground images (ROIs).
        masks: Tensor [B, 1, H, W] - Batch of corresponding masks.
        augmentation_types: list (Types of augmentations to apply, e.g., ['scale', 'rotate', 'flip']).

    Returns:
        Tuple of Tensors: (augmented_rois, augmented_masks)
    """
    augmented_rois, augmented_masks = [], []
    batch_size = rois.shape[0]
    
    # Process each ROI and mask in the batch
    for i in range(batch_size):
        roi = TF.to_pil_image(rois[i])  # Convert ROI tensor to PIL
        mask = TF.to_pil_image(masks[i][0])  # Convert mask tensor to PIL (single channel)
        
        # Store original ROI and mask
        augmented_rois.append(TF.to_tensor(roi))
        augmented_masks.append(TF.to_tensor(mask))  # Add channel dimension back
        
        # Apply scaling
        if augmentation_types is None or 'scale' in augmentation_types:
            for scale in [0.8, 1.2]:
                new_size = (int(roi.size[1] * scale), int(roi.size[0] * scale))  # (H, W)
                roi_rescaled = TF.resize(roi, size=new_size)
                mask_rescaled = TF.resize(mask, size=new_size, interpolation=TF.InterpolationMode.NEAREST)
                # if the scaled size is smaller than 224, pad the image
                if roi_rescaled.size[0] < 224 or roi_rescaled.size[1] < 224:
                    roi_rescaled = TF.pad(roi_rescaled, (0, 0, 224 - roi_rescaled.size[1], 224 - roi_rescaled.size[0]))
                    mask_rescaled = TF.pad(mask_rescaled, (0, 0, 224 - mask_rescaled.size[1], 224 - mask_rescaled.size[0]))
                    
                # if the scaled size is larger than 224, center crop the image
                if roi_rescaled.size[0] > 224 or roi_rescaled.size[1] > 224:
                    roi_rescaled = TF.center_crop(roi_rescaled, (224, 224))
                    mask_rescaled = TF.center_crop(mask_rescaled, (224, 224))
                    
                augmented_rois.append(TF.to_tensor(roi_rescaled))
                augmented_masks.append(TF.to_tensor(mask_rescaled))
        
        # Apply rotation
        if augmentation_types is None or 'rotate' in augmentation_types:
            for angle in [-30, 30]:
                roi_rotated = TF.rotate(roi, angle)
                mask_rotated = TF.rotate(mask, angle)
                augmented_rois.append(TF.to_tensor(roi_rotated))
                augmented_masks.append(TF.to_tensor(mask_rotated))
        
        # Apply flipping
        if augmentation_types is None or 'flip' in augmentation_types:
            roi_flipped = TF.hflip(roi)
            mask_flipped = TF.hflip(mask)
            augmented_rois.append(TF.to_tensor(roi_flipped))
            augmented_masks.append(TF.to_tensor(mask_flipped))
            
    
    # Combine augmented ROIs and masks into batches
    augmented_rois = torch.stack(augmented_rois)
    augmented_masks = torch.stack(augmented_masks)
    
    return augmented_rois, augmented_masks  
   
    

if __name__ == '__main__':
    main()