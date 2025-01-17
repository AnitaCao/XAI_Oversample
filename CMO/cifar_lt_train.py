# original code: https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
import random
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import load_dataset
import models
import torchvision.models as torch_models
from PIL import Image

#from losses import LDAMLoss, BalancedSoftmaxLoss
from losses import *
from opts import parser
import warnings
from util.util import *
from util.autoaug import CIFAR10Policy, Cutout
import util.moco_loader as moco_loader
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_utils
import region_select_copy as region_select



''' args['cut_mix']: CMO, CMO_XAI, CMO_XAI_MASK, TODO: CMO_OBJ_DET
Data augmentation type is for the mixup. CMO is the original random cut, 
CMO_XAI is gradcam based bounding box cut, CMO_XAI_MASK is gradcam based mask cut.
TODO: Implement a new data augmentation method based on the bounding box from object detection
instead of the bounding box from the saliency map.

args['sample_method']: topological or frequency. 
topological is the new data selection method based on the distance of the class centroids of the training samples.
frequency is the original data selection method based on the frequency of the classes in the training samples.

'''

best_acc1 = 0

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
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = args.num_classes
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)
    print(model)

    XAI_model = torch_models.resnet50(pretrained=True) # load a pretrained model for XAI heatmap generation
    XAI_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    num_ftrs = XAI_model.fc.in_features
    XAI_model.fc = nn.Linear(num_ftrs, num_classes) 
    print("XAI model: ", XAI_model) 

    XAI_model = XAI_model.cuda(args.gpu)
    XAI_model.eval()
    
    gc = region_select.GradCAM(XAI_model, 'layer4')

    if args.gpu is not None:
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


    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    if args.use_randaug:
        """
        if use_randaug == True, we follow randaug following PaCo's setting (ICCV'2021),
        400 epoch & Randaug 
        https://github.com/dvlab-research/Parametric-Contrastive-Learning/blob/main/LT/paco_cifar.py
        """
        print("use randaug!!")
        augmentation_regular = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),  # add AutoAug
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]

        augmentation_sim_cifar = [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar)]

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    print(args)

    
    if num_classes == 10:
        train_dataset = load_dataset("tomas-gajarsky/cifar10-lt", 'r-10', split="train")
        val_dataset = load_dataset("tomas-gajarsky/cifar10-lt", 'r-10', split="test")  
    elif num_classes == 100:
        train_dataset = load_dataset("tomas-gajarsky/cifar100-lt", 'r-10', split="train")
        val_dataset = load_dataset("tomas-gajarsky/cifar100-lt", 'r-10', split="test")
    else:
        warnings.warn('Dataset is not listed')
        return
    #calculate the number of samples in each class from the training dataset
    if num_classes == 100:
        label_name = 'fine_label' 
    else:
        label_name = 'label'
        
    cls_num_list = [0] * num_classes
    for example in train_dataset:
        label = example[label_name]
        cls_num_list[label] += 1
    
    print('cls num list:') 
    print(cls_num_list) 
    print("total number of samples: ", sum(cls_num_list)) #total number of samples:  12006 for cifar10_lt or 19573 for cifar100_lt
    args.cls_num_list = cls_num_list
    train_cls_num_list = np.array(cls_num_list) 
    
    toy_dataset = train_dataset.select(range(100))  # Select the first 100 samples

    train_dataset = train_dataset.map(lambda x: {'img': transform_train(Image.fromarray(np.array(x['img']))), 'label': x[label_name]}, batched=False)
    val_dataset = val_dataset.map(lambda x: {'img': transform_val(Image.fromarray(np.array(x['img']))), 'label': x[label_name]}, batched=False)
    
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    weighted_train_loader = None
    weighted_cls_num_list = [0] * num_classes

    if args.cut_mix.startswith('CMO'):
        targets = train_dataset['label']
        cls_weight = 1.0 / (np.array(cls_num_list) ** args.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        
        samples_weight = np.array([cls_weight[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        
        print(samples_weight)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                  replacement=True)
        weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            num_workers=args.workers, pin_memory=True,
                                                            sampler=weighted_sampler)
        
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
        train(train_loader, model, gc, criterion, optimizer, epoch, args, log_training, weighted_train_loader)

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


def train(train_loader, model, gradcam, criterion, optimizer, epoch, args, log, weighted_train_loader=None):
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

    for i, batch in enumerate(train_loader):
        input, target = batch['img'], batch['label']
        
        if isinstance(input, list):
            # Convert nested list to tensor
            input = torch.stack([torch.stack([torch.stack(channel) for channel in img], dim=0) for img in input], dim=0).permute(3,0,1,2)
            input = input.float()
        
        if args.cut_mix.startswith('CMO') and args.start_cut_mix < epoch < (args.epochs - args.end_cut_mix):
            try:
                batch2 = next(inverse_iter)  #for forground
                input2, target2 = batch2['img'], batch2['label']
            except:
                inverse_iter = iter(weighted_train_loader)
                batch2 = next(inverse_iter)
                input2, target2 = batch2['img'], batch2['label']
                
            if isinstance(input2, list):
            # Convert nested list to tensor
                input2 = torch.stack([torch.stack([torch.stack(channel) for channel in img], dim=0) for img in input2], dim=0).permute(3,0,1,2)
                input2 = input2.float()
            
            input2 = input2[:input.size()[0]]
            target2 = target2[:target.size()[0]]
            input2 = input2.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
       
        r = np.random.rand(1)
        #r = 0.01
        if args.cut_mix == 'CMO' and args.start_cut_mix < epoch < (args.epochs - args.end_cut_mix) and r < args.mixup_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)

            input_ori = input.clone() # save the original input for display purposes

            input[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target) * lam + criterion(output, target2) * (1. - lam)
            #testing_plot(input, input2)
            #print("just for testing purposes: CMO random mixup images: ")
            #testing_plot(input_ori, input2, input)

        elif args.cut_mix == 'CMO_XAI' and args.start_cut_mix < epoch < (
                args.epochs - args.end_cut_mix) and r < args.mixup_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            #print("---Calling XAI_Box. Batch number: ", i)
            start = time.time() 
            # Process images
            cam_maps, probs = gradcam(input2)
            #print("CAM maps shape:", cam_maps.shape)
        
            masks, actual_lams = region_select.generate_mixing_masks(cam_maps, lam=0.7)
            #region_select.visualize_cam_with_bbox(input2, cam_maps, masks)
            time1 = time.time()
            print('Total time to generate saliencys is: {:.2f} second'.format((time1-start)))  
  
            input_ori = input.clone() # save the original input for display purposes
            lam_list = []
            #print("original input type: ", type(input))
            #print("original tensor shape: ", input.shape) # torch.Size([16, 3, 224, 224])
            
            if args.data_aug:
                # get 6 batchs of images from trainloader to use as backgrounds
                backgrounds = []
                backgrouds_labels = []
                for j in range(6):
                    batch = next(iter(train_loader))
                    #make sure batch size is the same as the input size
                    background = batch['img'][:input.size()[0]]
                    b_labels = batch['label'][:input.size()[0]]
                    if isinstance(background, list):
                        # Convert nested list to tensor
                        background = torch.stack([torch.stack([torch.stack(channel) for channel in img], dim=0) for img in background], dim=0).permute(3,0,1,2)
                        background = background.float()
                    backgrounds.append(background)
                    backgrouds_labels.append(b_labels)
                backgrounds = torch.cat(backgrounds, dim=0)
                backgrouds_labels = torch.cat(backgrouds_labels, dim=0)
                
                mixed_imgs_list, lam_list = region_select.generate_mixed_images_with_augmentation(input2, backgrounds, masks, types=['scale', 'rotate', 'flip'])  
                    
                #print(f"Mixed images list length: {len(mixed_imgs_list)}")
                #print(f"First mixed image shape: {mixed_imgs_list[0].shape}")
                
                input = torch.stack(mixed_imgs_list, dim=0).cuda(args.gpu, non_blocking=True)
        
                #target = target.repeat_interleave(6, dim=0)
                target = backgrouds_labels.cuda(args.gpu, non_blocking=True)
                target2 = target2.repeat_interleave(6, dim=0).cuda(args.gpu, non_blocking=True)
                #print("Generated input type: ", type(input))
                #print("Generated tensor shape: ", input.shape) #torch.Size([288, 224, 224])
                

            else:
                backgrounds = input
                input, lam_list = region_select.generate_mixed_images_without_augmentation(input2, backgrounds, masks)

            output = model(input.cuda(args.gpu, non_blocking=True))
            if output.size(0) != target.size(0):
                print("Output size: ", output.size())
                print("Target size: ", target.size())
            if output.size(0) != target2.size(0):
                print("Output size: ", output.size())
                print("Target2 size: ", target2.size())
            loss = criterion(output, target) * torch.tensor(lam_list).cuda(args.gpu) + criterion(output, target2) * (1. - torch.tensor(lam_list).cuda(args.gpu))
            loss = loss.mean()
            #print("just for testing purposes: CMO+XAI mixup images: ")
            #testing_plot(input_ori, input2, input)
                #Use mask to blend input1 and input2 instead of using bouding box.    
        #elif args.cut_mix == 'CMO_XAI_MASK' and args.start_cut_mix < epoch < (
            #args.epochs - args.end_cut_mix) and r < args.mixup_prob:
                        # generate mixed sample
            #TODO: use mask instead of bounding box to blend input1 and input2

        elif args.cut_mix == 'CMO_OBJ_DET' and args.start_cut_mix< epoch < (
                args.epochs - args.end_cut_mix) and r < args.mixup_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            print("---Calling Object Detection. Batch number: ", i)
            start = time.time()
            #TODO: Implement object detection based bounding box cut
            if args.data_aug:
                #TODO: Implement data augmentation for the selected region of the foreground 
                #before mixing with the background
                print("Data augmentation for the selected region of the foreground")
        else:
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
        for i, batch in enumerate(val_loader):
            input, target = batch['img'], batch['label']
        
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
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        # print(out_cls_acc)

        #calcuate Gmean, ACSA, ACC
        gmean = np.prod(cls_acc) ** (1.0 / len(cls_acc))
        acsa = np.mean(cls_acc)
        acc = np.sum(cls_hit) / np.sum(cls_cnt)
        out_gm_acsa_acc = '%s Gmean: %.3f, ACSA: %.3f, ACC: %.3f' % (flag, gmean, acsa, acc)
        #print('Gmean: %.3f, ACSA: %.3f, ACC: %.3f' % (gmean, acsa, acc))
        print(out_gm_acsa_acc)

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

def testing_plot(input, input2):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    grid_img = make_grid(input[:20], nrow=5)
    grid_img = grid_img.permute(1, 2, 0)
    grid_img = grid_img.cpu().numpy()
    grid_img2 = make_grid(input2[:20], nrow=5)
    grid_img2 = grid_img2.permute(1, 2, 0)
    grid_img2 = grid_img2.cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img)
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img2)
    plt.axis('off')
    plt.show()

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
    img1 = np.transpose(img1, (1, 2, 0)) # (3, 32, 32) -> (32, 32, 3)
    img2 = np.transpose(img2, (1, 2, 0))
    img3 = np.transpose(img3, (1, 2, 0))

    # Create subplots
    fig, axs = plt.subplots(1, 3)

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


def denormalize(image_batch,  means = [0.4914, 0.4822, 0.4465], stds = [0.2023, 0.1994, 0.2010]):
    means = torch.tensor(means).view(1, 3, 1, 1)
    stds = torch.tensor(stds).view(1, 3, 1, 1)
    return image_batch * stds + means


if __name__ == '__main__':
    main()