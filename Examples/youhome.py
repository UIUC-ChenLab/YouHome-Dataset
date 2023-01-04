from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data.sampler import *
# import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import utils.utils as util
import utils.quantization as q
from model.vanilla.youhome_resnet import resnet

import numpy as np
import os, time, sys
import copy

import argparse
import random

#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--data_dir', '-d', type=str, default='/tmp/cifar10_data',
                    help='path to the dataset directory')
parser.add_argument('--arch', metavar='ARCH', default='resnet', help='Choose a model')
parser.add_argument('--save', '-s', action='store_true', help='save the model')
parser.add_argument('--save_folder', type=str, default='./save/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes')
parser.add_argument('--gtarget', '-g', type=float, default=0.0)
parser.add_argument('--finetune', '-f', action='store_true', help='finetune the model')
parser.add_argument('--test', '-t', action='store_true', help='test only')
parser.add_argument('--resume', '-r', type=str, default=None,
                    help='path of the model checkpoint for resuming training')

parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')
args = parser.parse_args()

#########################
# parameters 

batch_size = args.batch_size
num_epoch = args.epochs
# batch_size = 128
# num_epoch = 250
_LAST_EPOCH = -1 #last_epoch arg is useful for restart
# _WEIGHT_DECAY = 1e-5 # 0.
# candidates = ['binput-prerprelu-pg', 'binput-prerprelu-1bit', 'binput-prerprelu-2bit', 'prerprelu-resnet20']
_ARCH = args.arch
# this_file_path = os.path.dirname(os.path.abspath(__file__))
save_folder = args.save_folder
drop_last = True if 'binput' in _ARCH else False
#########################

#----------------------------
# Load the YouHome dataset.
#----------------------------

def load_dataset():
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),

        # util.Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        normalize
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transform=train_transforms
    )
    train_dataset_small = torch.utils.data.Subset(
        train_dataset, 
        random.sample(range(len(train_dataset)), k=int(len(train_dataset)/20)))

    trainloader = torch.utils.data.DataLoader(
        train_dataset_small, batch_size=batch_size, shuffle=True,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=drop_last
    )

    val_dataset = torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            # normalize
        ]))
    val_dataset_small = torch.utils.data.Subset(
        val_dataset,
        random.sample(range(len(val_dataset)), k=int(len(val_dataset)/10)))

    valloader = torch.utils.data.DataLoader(
        val_dataset_small,
        batch_size=batch_size, shuffle=False,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=drop_last
    )

    return trainloader, valloader


# #----------------------------
# # Define the model.
# #----------------------------

# def generate_model(model_arch):
#     if 'prerprelu-resnet20' in model_arch:
#         import model.quantization.prerprelu_resnet20 as m
#         return m.resnet20()
#     elif 'binput-prerprelu-pg' in model_arch:
#         import model.quantization.binput_prerprelu_resnet20_pg as m
#         return m.resnet20(batch_size=batch_size, num_gpus=torch.cuda.device_count())
#     elif 'binput-prerprelu-1bit' in model_arch:
#         import model.quantization.binput_prerprelu_resnet20_1bit as m
#         return m.resnet20(batch_size=batch_size, num_gpus=torch.cuda.device_count())
#     elif 'binput-prerprelu-2bit' in model_arch:
#         import model.quantization.binput_prerprelu_resnet20_2bit as m
#         return m.resnet20(batch_size=batch_size, num_gpus=torch.cuda.device_count())
#     else:
#         raise NotImplementedError("Model architecture is not supported.")


#----------------------------
# Train the network.
#----------------------------

def train_model(trainloader, testloader, net, 
                optimizer, scheduler, start_epoch, device, log):
    # define the loss function
    criterion = (nn.CrossEntropyLoss().cuda() 
                if torch.cuda.is_available() else nn.CrossEntropyLoss())

    best_acc = 0.0
    best_model = copy.deepcopy(net.state_dict())

    start_time = time.time()
    epoch_time = util.AverageMeter('Time/Epoch', ':.2f')

    recorder = util.RecorderMeter(num_epoch)
    
    train_los = 0
    val_los = 0
    train_acc = 0
    val_acc = 0

    for epoch in range(start_epoch, num_epoch): # loop over the dataset multiple times

        # set printing functions
        # batch_time = util.AverageMeter('Time/batch', ':.2f')
        # losses = util.AverageMeter('Loss', ':6.2f')
        # top1 = util.AverageMeter('Acc', ':6.2f')
        # progress = util.ProgressMeter(
        #                 len(trainloader),
        #                 [losses, top1, batch_time],
        #                 prefix="Epoch: [{}]".format(epoch+1)
        #                 )

        # switch the model to the training mode
        net.train()

        current_learning_rate = optimizer.param_groups[0]['lr']
        # print_log('current learning rate = {}'.format(current_learning_rate), log)
        

        need_hour, need_mins, need_secs = util.convert_secs2time(
            epoch_time.avg * (num_epoch - epoch))
        avg_hour, avg_mins, avg_secs = util.convert_secs2time(
            epoch_time.avg)
        need_time = '[Need: {:02d}h{:02d}m]'.format(need_hour, need_mins)
        avg_time = '[Avg: {:02d}m{:02d}s]'.format(avg_mins, avg_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}{:s} [LR={:6.5f}]'.format(util.time_string(), epoch, num_epoch,
                                                                                   need_time, avg_time, current_learning_rate,)
            + ' [Best Acc={:.2f}]'.format(best_acc), log)

        # each epoch
        batch_time = util.AverageMeter('Time/batch', ':.2f')
        losses = util.AverageMeter('Loss', ':6.2f')
        top1 = util.AverageMeter('Acc', ':6.2f')
        top5 = util.AverageMeter('Acc', ':6.2f')
        
        start = time.time()
        end = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if 'pg' in _ARCH:
                for name, param in net.named_parameters():
                    if 'threshold' in name:
                        loss += 0.00001 * 0.5 * torch.norm(param-args.gtarget) * torch.norm(param-args.gtarget)
            loss.backward()
            optimizer.step()

            # # measure accuracy and record loss
            # _, batch_predicted = torch.max(outputs.data, 1)
            # batch_accu = 100.0 * (batch_predicted == labels).sum().item() / labels.size(0)
            # losses.update(loss.item(), labels.size(0))
            # top1.update(batch_accu, labels.size(0))

            # measure accuracy and record loss
            def accuracy(outputs, labels, topk=(1,)):
                """Computes the precision@k for the specified values of k"""
                with torch.no_grad():
                    maxk = max(topk)
                    batch_size = labels.size(0)

                    _, pred = outputs.topk(maxk, 1, True, True)
                    pred = pred.t()
                    correct = pred.eq(labels.view(1, -1).expand_as(pred))

                    res = []
                    for k in topk:
                        correct_k = correct[:k].reshape(-1).float().sum(0)
                        res.append(correct_k.mul_(100.0 / batch_size))
                    return res

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            elapsed = end - start

            prt_str = ('\r    Epoch: [%d][%d/%d]  Time %.3f  Loss %.4f  Prec@1 %.4f  Prec@5 %.4f'
                   %(epoch + 1, i + 1, len(trainloader), elapsed,
                    losses.val, top1.val, top5.val))
            sys.stdout.write("\b" * (len(prt_str))) 
            sys.stdout.write(prt_str)
            sys.stdout.flush()

        print_log(
            '\n    **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                  error1=100 - top1.avg), log)
            # if i % 100 == 99:    
            #     # print statistics every 100 mini-batches each epoch
            #     progress.display(i) # i = batch id in the epoch

        # update the learning rate
        scheduler.step()

        # print test accuracy every few epochs
        if epoch % 1 == 0:
            # print_log('epoch {}'.format(epoch+1), log)
            epoch_acc, epoch_los = test_accu(testloader, net, device, log)
            if 'pg' in _ARCH:
                sparsity(testloader, net, device)
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(net.state_dict())
            # print_log("The best test accuracy so far: {:.1f}".format(best_acc), log)

            # save the model if required
            if args.save:
                print_log("\r    Saving the trained model and states.", log)
                # this_file_path = os.path.dirname(os.path.abspath(__file__))
                # save_folder = os.path.join(this_file_path, 'save_CIFAR10_model')
                util.save_models(best_model, save_folder,
                        suffix=_ARCH+'-finetune' if args.finetune else _ARCH)
                """
                states = {'epoch':epoch+1, 
                          'optimizer':optimizer.state_dict(), 
                          'scheduler':scheduler.state_dict()}
                util.save_states(states, save_folder, suffix=_ARCH)
                """
        train_los = losses.avg
        train_acc = top1.avg
        val_los = epoch_los
        val_acc = epoch_acc
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        recorder.plot_curve(os.path.join(save_folder, 'curve.png'))

    print_log('Finished Training', log)


#----------------------------
# Test accuracy.
#----------------------------

def test_accu(testloader, net, device, log, num_iter = 1):
    net.to(device)
    correct = 0
    total = 0
    running_loss = 0.0
    # switch the model to the evaluation mode
    net.eval()
    with torch.no_grad():
        iter_count = 0
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            criterion = (nn.CrossEntropyLoss().cuda() 
                if torch.cuda.is_available() else nn.CrossEntropyLoss())
            loss = criterion(outputs, labels)
            if 'pg' in _ARCH:
                for name, param in net.named_parameters():
                    if 'threshold' in name:
                        loss += 0.00001 * 0.5 * torch.norm(param-args.gtarget) * torch.norm(param-args.gtarget)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iter_count += 1
            # if iter_count == num_iter:
            #     break

    accuracy = 100.0 * correct / total
    print_log('\r    Accuracy of the network on the test images: %.1f %%' % (accuracy), log)
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_acc, val_loss


#----------------------------
# Report sparsity in PG
#----------------------------

def sparsity(testloader, net, device):
    cnt_out = np.zeros(18) # this 9 is hardcoded for ResNet-20
    cnt_high = np.zeros(18) # this 9 is hardcoded for ResNet-20
    num_out = []
    num_high = []

    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, q.PGBinaryConv2d):
            num_out.append(m.num_out)
            num_high.append(m.num_high)

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            """ calculate statistics per PG layer """
            net.apply(_report_sparsity)
            cnt_out += np.array(num_out)
            cnt_high += np.array(num_high)
            num_out = []
            num_high = []
    # print('Sparsity of the update phase: %.1f %%' % (100.0-np.sum(cnt_high)*1.0/np.sum(cnt_out)*100.0))

#----------------------------
# Log
#----------------------------

def print_log(print_string, log):
    text = '{}'.format(print_string)
    print(text)
    log.write('{}\n'.format(print_string))
    log.flush()


#----------------------------
# Main function.
#----------------------------

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    log = open(os.path.join(args.save_folder, 'log.txt'), 'w')
    print_log('save path : {}'.format(args.save_folder), log)
    # state = {k: v for k, v in args}
    # print_log(state, log)
    print_log("python version : {}".format(
        sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(
        torch.backends.cudnn.version()), log)

    print_log("Available GPUs: {}".format(torch.cuda.device_count()), log)

    print_log("Create resnet18 model.", log)

    # net = models.resnet18(pretrained=True)
    # net.fc = nn.Linear(512, args.num_classes)
    net = resnet(num_classes=args.num_classes)
    # net = generate_model(_ARCH)
    print_log("{} \n".format(net), log)
    n_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print_log("Number of parameters: {}\n".format(n_param), log)


    if torch.cuda.device_count() > 1:
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        print_log("Activate multi GPU support.", log)
        net = nn.DataParallel(net)
    net.to(device)

    #------------------
    # Load model params
    #------------------
    if args.resume is not None or args.test:
        model_path = args.resume
        if os.path.exists(model_path):
            print_log("Loading trained model from {}.".format(model_path), log)
            net.load_state_dict(torch.load(model_path), strict=True)
        else:
            raise ValueError("Model not found.")

    #-----------------
    # Prepare Data
    #-----------------
    print_log("Loading data.", log)
    trainloader, testloader = load_dataset()

    #-----------------
    # Test
    #-----------------
    if args.test:
        print_log("Mode: Test only.", log)
        test_accu(testloader, net, device, log)
        if 'pg' in _ARCH:
            sparsity(testloader, net, device)

    #-----------------
    # Finetune
    #-----------------
    elif args.finetune:
        print_log("num epochs = {}".format(num_epoch), log)
        # initial_lr = 1e-5
        print_log("init lr = {}".format(args.learning_rate), log)
        optimizer = optim.Adam(net.parameters(),
                          lr = args.learning_rate,
                          weight_decay=0.)
        lr_decay_milestones = [100, 150, 200] #[150, 250, 300]#[200, 250, 300]#[50, 100] 
        print_log("milestones = {}".format(lr_decay_milestones), log)
        scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=lr_decay_milestones,
                            gamma=0.1,
                            last_epoch=_LAST_EPOCH)
        start_epoch=0
        print_log("Start finetuning.", log)
        train_model(trainloader, testloader, net,
                    optimizer, scheduler, start_epoch, device, log)
        test_accu(testloader, net, device, log, num_iter = 1)

    #-----------------
    # Train
    #-----------------
    else:
        #-----------
        # Optimizer
        #-----------
        # initial_lr = 1e-3
        optimizer = optim.Adam(net.parameters(),
                          lr = args.learning_rate,
                          weight_decay=args.decay)

        #-----------
        # Scheduler
        #-----------
        print_log("Use linear learning rate decay.", log)
        lambda1 = lambda epoch : (1.0-epoch/num_epoch) # linear decay
        #lambda1 = lambda epoch : (0.7**epoch) # exponential decay
        scheduler = optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lr_lambda=lambda1,
                            last_epoch=_LAST_EPOCH)

        start_epoch = 0
        print_log("Start training.", log)
        train_model(trainloader, testloader, net, 
                    optimizer, scheduler, start_epoch, device, log)
        test_accu(testloader, net, device, log)

if __name__ == "__main__":
    main()

