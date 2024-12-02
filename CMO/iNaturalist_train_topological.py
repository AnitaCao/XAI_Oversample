# reference code: https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
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
from tensorboardX import SummaryWriter
from imbalance_data.imbalance_iNaturalist import load_imb_inaturalist, Imb_iNat_Dataset
from losses import LDAMLoss, BalancedSoftmaxLoss
import warnings
from torch.nn import Parameter
import torch.nn.functional as F
from util.util import *
from util.randaugment import rand_augment_transform
import util.moco_loader as moco_loader
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import gradcam
import feature_visualization_and_distance as fvd

#from utils import load_image, save_img_with_heatmap, check_path_exist, apply_transforms, save_heatmap


args = {}
args['dataset'] = 'iNaturalist_lt' #iNaturalist, iNaturalist_lt, iNaturalist_lt_Radom
#args['root'] = 'D:/anita/Research/iNaturalist/' #my local machine 
args['root'] = '/home/tcvcs/image_datasets/iNaturalist/' #server
args['arch'] = 'resnet50'
args['loss_type'] = 'BS'
args['train_rule'] = 'DRW'
args['imb_factor'] = 0.01  # this is not used in the LT_Dataset class. TODO: Implement a new dataset based on imb_factor
args['rand_number'] = 0
args['mixup_prob'] = 0.5
args['exp_str'] = 'exp'
args['seed'] = None
args['gpu'] = 0
args['resume'] = False
args['start_epoch'] = 0
args['batch_size'] = 16 #128
args['workers'] = 4
args['print_freq'] = 100
args['lr'] = 0.1
args['momentum'] = 0.9
args['weight_decay'] = 5e-4
args['epochs'] = 200
args['start_data_aug'] = 25 # start data augmentation after 25th epoch
args['end_data_aug'] = 25 # do not use data augmentation for the last 25 epoches
args['use_randaug'] = False
args['beta'] = 1
args['data_aug'] = 'CMO_XAI_MASK' # 'CMO' or 'CMO_XAI' or 'CMO_XAI_MASK'
args['sample_method'] = 'topological' #'topological' or 'freqeuncy' #by defaut, it is frequency
args['weighted_alpha'] = 0.5
args['num_classes'] = 10 #93
args["long_tail"] = False #10 classese for imb or 93 classes for long tail
args["random_select"] = False
args['root_log'] = './logs'
args['root_model'] = './checkpoint'
args['store_name'] = '_'.join([args['dataset'], args['arch'], args['loss_type'], args['train_rule'], args['data_aug'], str(args['imb_factor']), str(args['rand_number']), str(args['mixup_prob']), args['exp_str']])
args["cls_num_list"] = []


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
    prepare_folders(args)
    #if args['cos']:
    #    print("use cosine LR")

    if args['seed'] is not None:
        torch.manual_seed(args['seed'])
        cudnn.deterministic = True
        np.random.seed(args['seed'])
        random.seed(args['seed'])
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args['gpu'] is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args['gpu'], ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda

    args['gpu'] = gpu
    if args['gpu'] is not None:
        print("Use GPU: {} for training".format(args['gpu']))

    # create model
    print("=> creating model '{}'".format(args["arch"]))
    num_classes = args['num_classes']
    model = getattr(models, args["arch"])(pretrained=True) #use pretrained model for smaller dataset.
    #if args['num_classes'] == 10:
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    XAI_model = models.resnet50(pretrained=True) # load a pretrained model for XAI heatmap generation
    XAI_model = XAI_model.eval()
    XAI_model = XAI_model.cuda()
    target_layer = 'layer4'
    gc = gradcam.GradCAM(XAI_model, target_layer)


    if args['loss_type'] == 'LDAM':
        num_ftrs = model.fc.in_features
        model.fc = NormedLinear(num_ftrs, num_classes)

    if args['gpu'] is not None:
        torch.cuda.set_device(args['gpu'])
        model = model.cuda(args['gpu'])
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args['lr'],
                                momentum=args['momentum'],
                                weight_decay=args['weight_decay'])

    # optionally resume from a checkpoint
    if args['resume']:
        if os.path.isfile(args['resume']):
            print("=> loading checkpoint '{}'".format(args['resume']))
            checkpoint = torch.load(args['resume'], map_location='cuda:0')
            args['start_epoch'] = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args['gpu'] is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args['gpu'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))

    cudnn.benchmark = True

    # Data loading code
    if args['use_randaug']:
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
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
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

    data_path = os.path.join(args['root'], 'train')

    train_images_list, train_labels_list, val_images_list, val_labels_list = load_imb_inaturalist(data_path, transform_train, transform_val, long_tail=args["long_tail"], random_select=args["random_select"])
    
    train_dataset = Imb_iNat_Dataset(data_path,train_images_list, train_labels_list,transform_train)
    val_dataset = Imb_iNat_Dataset(data_path,val_images_list, val_labels_list,transform_val)
    
    
    num_classes = len(np.unique(train_dataset.targets))

    print("Number of classes: ", num_classes)
    print("Number of training samples: ", len(train_dataset))
    #print("Number of training samples: ", len(np.unique(val_dataset.targets)))
    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    print('cls num list:')
    print(cls_num_list)

    args['cls_num_list'] = cls_num_list
    train_cls_num_list = np.array(cls_num_list)


    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=(train_sampler is None),
        num_workers=args['workers'], pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args['batch_size'], shuffle=False,
        num_workers=args['workers'], pin_memory=True)

    weighted_train_loader = None

    if args['data_aug'].startswith('CMO'):
        cls_weight = 1.0 / (np.array(cls_num_list) ** args['weighted_alpha'])
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print(samples_weight)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                  replacement=True)
        weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'],
                                                            num_workers=args['workers'], pin_memory=True,
                                                            sampler=weighted_sampler)

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()

    # init log for training
    log_training = open(os.path.join(args["root_log"], args["store_name"], 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args["root_log"], args["store_name"], 'log_test.csv'), 'w')
    with open(os.path.join(args["root_log"], args["store_name"], 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args["root_log"], args["store_name"]))
    
    start_time = time.time()
    print("Training started!")

    for epoch in range(args['start_epoch'], args['epochs']):
        if args['use_randaug']:
            paco_adjust_learning_rate(optimizer, epoch, args)
        else:
            adjust_learning_rate(optimizer, epoch, args)

        if args['train_rule'] == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args['train_rule'] == 'CBReweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args['gpu'])
        elif args['train_rule'] == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args['gpu'])
        else:
            warnings.warn('Sample rule is not listed')

        if args['loss_type'] == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args['gpu'])
        elif args['loss_type'] == 'BS':
            criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list_cuda).cuda(args['gpu'])
        elif args['loss_type'] == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args['gpu'])
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, model, gc, criterion, optimizer, epoch, args, log_training, tf_writer,weighted_train_loader=weighted_train_loader)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)

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



def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def train(train_loader, model, gc, criterion, optimizer, epoch, args, log,
              tf_writer, weighted_train_loader=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    end = time.time()
    if args['data_aug'].startswith('CMO'):
        weighted_train_loader = iter(weighted_train_loader)

    for i, (input, target) in enumerate(train_loader):
        if args['data_aug'].startswith('CMO') and args['start_data_aug'] < epoch < (
                args['epochs'] - args['end_data_aug']):
            print("Using topological data selection")
            time1 = time.time()
        
            if args['sample_method'] == 'topological':
                
                distance_matrix, class_labels = fvd.get_distance_matrix(train_loader)
                
                #fvd.visualize_distance_matrix(distance_matrix, class_labels)

                probability_matrix = fvd.get_probability_matrix(distance_matrix)
                
                #fvd.visualize_probability_matrix(probability_matrix)
                
                input2, target2 = fvd.get_next_batch(input, target, train_loader, probability_matrix)                  
            else: 
                try:
                    input2, target2 = next(weighted_train_loader)
                except:
                    weighted_train_loader = iter(weighted_train_loader)
                    input2, target2 = next(weighted_train_loader)
                            
            time2 = time.time()
            print('Total time to get next batch with topological selection is: {:.2f} second'.format((time2-time1)))
                
            input2 = input2[:input.size()[0]] # make sure the batch size is the same
            target2 = target2[:target.size()[0]] # make sure the batch size is the same
            
            input2 = input2.cuda(args['gpu'], non_blocking=True)
            target2 = target2.cuda(args['gpu'], non_blocking=True)
            
            

        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda(args['gpu'], non_blocking=True) #16, 3, 224, 224
        target = target.cuda(args['gpu'], non_blocking=True) #16
        # Data augmentation
        r = np.random.rand(1)

        if args['data_aug'] == 'CMO' and args['start_data_aug'] < epoch < (
                args['epochs'] - args['end_data_aug']) and r < args["mixup_prob"]:
            # generate mixed sample
            lam = np.random.beta(args["beta"], args["beta"])

            input_ori = input.clone() # save the original input for display purposes

            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam) 
            input[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            output = model(input)
            loss = criterion(output, target) * lam + criterion(output, target2) * (1. - lam)
   
            #print("just for testing purposes: CMO random mixup images: ")
            #testing_plot(input_ori, input2, input)

        elif args['data_aug'] == 'CMO_XAI' and args['start_data_aug'] < epoch < (
                args['epochs'] - args['end_data_aug']) and r < args["mixup_prob"]:
            # generate mixed sample
            lam = np.random.beta(args["beta"], args["beta"])
            print("---Calling XAI_Box. Batch number: ", i)
            start = time.time()
            saliencys, _ = gc(input2, None) 
            
            #saliency_visualisation(input2, saliencys)
            
            time1 = time.time()
            print('Total time to generate saliencys is: {:.2f} second'.format((time1-start))) 

            bounding_list = XAI_box(saliencys, lam) # TODO: lam should be used as the threshold for the ROI, current threshold is 0.5
            #print("Bounding box list: ", bounding_list)
            

            #repace input's area with input2's area based on bounding box
            input_ori = input.clone() # save the original input for display purposes
            masks = torch.zeros_like(input)  # Create a zero mask for all images
            lam_list = []
            for j in range(len(bounding_list)):
                bbx1, bby1, bbx2, bby2 = bounding_list[j]
                masks[j, :, bby1:bby2, bbx1:bbx2] = 1  # Set mask to 1 within the bounding box
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                lam_list.append(lam)

            # Use the mask to blend input1 and input2
            input = masks * input2 + (1 - masks) * input
            output = model(input)
            loss = criterion(output, target) * torch.tensor(lam_list).cuda(args['gpu']) + criterion(output, target2) * (1. - torch.tensor(lam_list).cuda(args['gpu']))
            loss = loss.mean()
            #print("Just for testing purposes: CMO+XAI mixup images: ")
            #testing_plot(input_ori, input2, input)
        
        #Use mask to blend input1 and input2 instead of using bouding box.    
        elif args['data_aug'] == 'CMO_XAI_MASK' and args['start_data_aug'] < epoch < (
            args['epochs'] - args['end_data_aug']) and r < args["mixup_prob"]:
                        # generate mixed sample
            lam = np.random.beta(args["beta"], args["beta"])
            print("---Calling XAI_MASK. Batch number: ", i)
            start = time.time()
            saliencys, _ = gc(input2, None) 
            
            #saliency_visualisation(input2, saliencys)
            
            time1 = time.time()
            print('Total time to generate saliencys is: {:.2f} second'.format((time1-start))) 

            MASK_list, lam_list = XAI_MASK(saliencys, lam) # TODO: lam should be used as the threshold for the ROI, current threshold is 0.5
            
            # Convert refined_binary_masks to a torch tensor if it isn't already
            MASK_list = torch.tensor(MASK_list, dtype=torch.float32).cuda(args['gpu'])

            #repace input's area with input2's area based on bounding box
            input_ori = input.clone() # save the original input for display purposes

            # Use the mask to blend input1 and input2
            input = MASK_list * input2 + (1 - MASK_list) * input
            output = model(input)
            loss = criterion(output, target) * torch.tensor(lam_list).cuda(args['gpu']) + criterion(output, target2) * (1. - torch.tensor(lam_list).cuda(args['gpu']))
            loss = loss.mean()
            #print("Just for testing purposes: CMO+XAI mixup images: ")
            #testing_plot(input_ori, input2, input)              
        else:
            output = model(input)
            loss = criterion(output, target).mean()

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

        if i % args['print_freq'] == 0:
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

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def XAI_MASK(saliencys, lam):
    # saliency is a batch of saliency maps. For example: 32 by 1 by 224 by 224   
    #TODO: threshold should be related to lam.
    threshold = 0.5
    binary_masks = (saliencys > threshold).astype(np.uint8)
    #binary_masks = np.squeeze(binary_masks, axis=1)
    
    refined_binary_masks = np.array([refine_mask(mask[0]) for mask in binary_masks])   
    # Calculate the share of the ROI for each mask in the batch
    roi_shares = np.mean(refined_binary_masks, axis=(1, 2))

    return binary_masks,roi_shares

def refine_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    return refined_mask

def XAI_box(saliencys, lam):

    # saliency is a batch of saliency maps. For example: 32 by 1 by 224 by 224   
    #TODO: threshold should be related to lam.
    threshold = 0.5 

    '''
    #for each saliency map in the batch, print out the min and max values
    for k in range(len(saliencys)):
        saliency = saliencys[k].squeeze(0)
        print("for image ", k, "min: ", np.min(saliency), "max: ", np.max(saliency))
    '''
    bboxes = getCountourList(threshold, saliencys)
    return bboxes

def getCountour(threshold, saliency):
     #add bouding box around ROI 
    saliency = saliency.squeeze(0)

    _, binary_saliency = cv2.threshold(saliency, threshold, 255, cv2.THRESH_BINARY) # Convert the saliency map to binary
    # Convert binary_saliency to 8-bit image
    binary_saliency = np.uint8(binary_saliency)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_saliency, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("!!!No contour found!!!")
        print(saliency)
        print(binary_saliency)
        return 0, 0, 0, 0
    
    largest_contour = max(contours, key=cv2.contourArea)
    # Compute the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    bbx1=x 
    bby1=y
    bbx2=x+w
    bby2=y+h 

    return bbx1, bby1, bbx2, bby2

def getCountourList(threshold, saliencys):
    bbx1_list = []
    bby1_list = []
    bbx2_list = []
    bby2_list = []
    for saliency in saliencys:
        bbx1, bby1, bbx2, bby2 = getCountour(threshold, saliency)
        bbx1_list.append(bbx1)
        bby1_list.append(bby1)
        bbx2_list.append(bbx2)
        bby2_list.append(bby2)
    #create a list of bounding box coordinates, for each image in the batch, shape 32 by 4
    bboxes = np.array([bbx1_list, bby1_list, bbx2_list, bby2_list]).T
    return bboxes


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


def validate(val_loader, model, criterion, epoch,args, log, tf_writer, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args['gpu'], non_blocking=True)
            target = target.cuda(args['gpu'], non_blocking=True)

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

            if i % args['print_freq'] == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1) # the number of ground truth class
        cls_hit = np.diag(cf) # the number of correctly predicted class
        cls_acc = cls_hit / cls_cnt # the accuracy of each class, namely, tpr = true positive / total positive
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        print(output)
        
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(out_cls_acc)
        
        #calcuate Gmean, ACSA, ACC
        gmean = np.prod(cls_acc) ** (1.0 / len(cls_acc))
        acsa = np.mean(cls_acc)
        acc = np.sum(cls_hit) / np.sum(cls_cnt)
        out_gm_acsa_acc = '%s Gmean: %.3f, ACSA: %.3f, ACC: %.3f' % (flag, gmean, acsa, acc)
        #print('Gmean: %.3f, ACSA: %.3f, ACC: %.3f' % (gmean, acsa, acc))
        print(out_gm_acsa_acc)
        
        
        
        #[2000, 1500, 1000, 800, 650, 500, 350, 200, 120, 60]
        many = train_cls_num_list > 1000
        medium = (train_cls_num_list <= 1000) & (train_cls_num_list > 200)
        few = train_cls_num_list <= 200
        print("many avg(>1000), med avg(200-1000), few avg(<200)", float(sum(cls_acc[many]) * 100 / sum(many)),
              float(sum(cls_acc[medium]) * 100 / sum(medium)),
              float(sum(cls_acc[few]) * 100 / sum(few)))
        
        '''
        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list > 20)
        few_shot = train_cls_num_list <= 20
        print("many avg, med avg, few avg", float(sum(cls_acc[many_shot]) * 100 / sum(many_shot)),
              float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot)),
              float(sum(cls_acc[few_shot]) * 100 / sum(few_shot)))
        '''

        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.write(out_gm_acsa_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args['lr'] * epoch / 5
    elif epoch > 80:
        lr = args['lr'] * 0.01
    elif epoch > 60:
        lr = args['lr'] * 0.1
    else:
        lr = args['lr']

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def paco_adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    warmup_epochs = 10

    lr = args['lr']
    if epoch < warmup_epochs:
        lr = lr / warmup_epochs * (epoch + 1)
    elif args['cos']:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (args['epochs'] - warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args['lr_steps']:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def testing_plot(input_ori, input2, mixed_input):
    # Denormalize the images
    img1 = denormalize(input_ori.cpu())
    img2 = denormalize(input2.cpu())
    img3 = denormalize(mixed_input.cpu())

    img_index = 5
    img1 = img1[img_index].numpy()
    img2 = img2[img_index].numpy()
    img3 = img3[img_index].numpy()

    # Move color channel to last dimension
    img1 = np.transpose(img1, (1, 2, 0)) # (3, 224, 224) -> (224, 224, 3)
    img2 = np.transpose(img2, (1, 2, 0))
    img3 = np.transpose(img3, (1, 2, 0))

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Show images
    axs[0].imshow(img1)
    axs[0].set_title('Original Image')

    axs[1].imshow(img2)
    axs[1].set_title('Input2 Image')

    axs[2].imshow(img3)
    axs[2].set_title('Mixed Input Image')

    # Remove axis
    for ax in axs:
        ax.axis('off')

    plt.show()


def denormalize(image_batch, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    means = torch.tensor(means).view(1, 3, 1, 1)
    stds = torch.tensor(stds).view(1, 3, 1, 1)
    return image_batch * stds + means

#for ploting purposes, only take one image in the batch
def saliency_visualisation(batch, saliencys):
    imgs = denormalize(batch.cpu())
    img = imgs[15].numpy()
    img = np.transpose(img, (1, 2, 0)) # move color channel to last dimension
    saliency = saliencys[15].squeeze(0)
    #saliency = cv2.resize(saliency, (224, 224))
    #print(np.min(saliency))
    #print(np.max(saliency))

    fig, ax = plt.subplots(1,3)
    
    img = img * 255
    img_heatmap = utils.save_img_with_heatmap(img, saliency, None, style='zhou', normalise=True)
    heatmap = utils.save_heatmap(saliency, None, normalise=True)

    threshold = 0.5  
    _, binary_saliency = cv2.threshold(saliency, threshold, 255, cv2.THRESH_BINARY) # Convert the saliency map to binary

    # Convert binary_saliency to 8-bit image
    binary_saliency = np.uint8(binary_saliency)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_saliency, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_np = np.array(img).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np) # Make sure the array is contiguous for cv2.Rectangle
    #print("Data type:", img_np.dtype)
    #print("Shape:", img_np.shape)
    #print("Is contiguous:", img_np.flags['C_CONTIGUOUS'])

    #img_np_blank = np.zeros((224, 224, 3), dtype=np.uint8)
    #print("Data type:", img_np_blank.dtype)
    #print("Shape:", img_np_blank.shape)
    #print("Is contiguous:", img_np_blank.flags['C_CONTIGUOUS'])

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

    ax[0].imshow((img_heatmap).astype(np.uint8))
    ax[1].imshow((heatmap[:, :, ::-1]).astype(np.uint8))
    ax[2].imshow((img_np).astype(np.uint8))
    plt.axis('off')


if __name__ == '__main__':
    main()