#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os, sys, pathlib, random, time, pickle, copy
from tqdm import tqdm
import json

import nflib
from nflib.flows import SequentialFlow, ActNorm, ActNorm2D, BatchNorm1DFlow, BatchNorm2DFlow
import nflib.res_flow as irf

import torch.optim as optim
from torch.utils import data

# device = torch.device("cpu")


# In[24]:


#### ["mnist", "fmnist", "cifar10", cifar100]
# experiment = "fmnist"
# invertible_backbone = True
# connected_classifier = True
# experiment_index = 0

def benchmark(experiment, invertible_backbone, connected_classifier, experiment_index, cuda=0, linear_clf=False):
    seed = [123, 456, 789][experiment_index]

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
        
    EPOCHS = None
    bs = None
    lr = 3e-4
    if experiment=="mnist" or experiment=="fmnist":
        EPOCHS = 100
        bs = 50
    elif experiment == "cifar10":
        EPOCHS = 400
        bs = 64
    elif experiment == "cifar100":
        EPOCHS = 600
        bs = 128

    
    # EPOCHS = 10
    
    device = torch.device(f"cuda:{cuda}")

    # ### Datasets
    train_dataset = None
    test_dataset = None
    if experiment == "mnist" or experiment == "fmnist":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if experiment == "mnist":
            train_dataset = datasets.MNIST(root="./data/", train=True, download=True, transform=train_transform)
            test_dataset = datasets.MNIST(root="./data/", train=False, download=True, transform=test_transform)
        elif experiment == "fmnist":
            train_dataset = datasets.FashionMNIST(root="./data/", train=True, download=True, transform=train_transform)
            test_dataset = datasets.FashionMNIST(root="./data/", train=False, download=True, transform=test_transform)

    elif experiment == "cifar10":
        cifar_train = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], # mean=[0.5071, 0.4865, 0.4409] for cifar100
                std=[0.2023, 0.1994, 0.2010], # std=[0.2009, 0.1984, 0.2023] for cifar100
            ),
        ])

        cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], # mean=[0.5071, 0.4865, 0.4409] for cifar100
                std=[0.2023, 0.1994, 0.2010], # std=[0.2009, 0.1984, 0.2023] for cifar100
            ),
        ])

        train_dataset = datasets.CIFAR10(root="./data/", train=True, download=True, transform=cifar_train)
        test_dataset = datasets.CIFAR10(root="./data/", train=False, download=True, transform=cifar_test)

    elif experiment == "cifar100":
        cifar_train = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4865, 0.4409],
                std=[0.2009, 0.1984, 0.2023],
            ),
        ])

        cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4865, 0.4409],
                std=[0.2009, 0.1984, 0.2023],
            ),
        ])

        train_dataset = datasets.CIFAR100(root="./data/", train=True, download=True, transform=cifar_train)
        test_dataset = datasets.CIFAR100(root="./data/", train=False, download=True, transform=cifar_test)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, num_workers=8)


    # ### Model

    actf = irf.Swish
    # actf = irf.LeakyReLU
    Norm1D = ActNorm
    Norm2D = ActNorm2D
    # if experiment == 'cifar100':
        # Norm1D = BatchNorm1DFlow
        # Norm2D = BatchNorm2DFlow
        # Norm1D = nn.BatchNorm2d
        # Norm2D = nn.BatchNorm1d

    if experiment.startswith("cifar"):
        Norm1D = BatchNorm1DFlow
        Norm2D = BatchNorm2DFlow
        # Norm1D = ActNorm
        # Norm2D = ActNorm2D
        flows = [
            Norm2D(3),
            irf.ConvResidualFlow([3, 32, 32], [32, 32], kernels=5, activation=actf),
            irf.InvertiblePooling(2),
            Norm2D(12),
            irf.ConvResidualFlow([12, 16, 16], [64, 64], kernels=5, activation=actf),
            Norm2D(12),
            irf.ConvResidualFlow([12, 16, 16], [64, 64], kernels=5, activation=actf),
            irf.InvertiblePooling(2),
            Norm2D(48),
            irf.ConvResidualFlow([48, 8, 8], [128, 128], kernels=5, activation=actf),
            Norm2D(48),
            irf.ConvResidualFlow([48, 8, 8], [128, 128], kernels=5, activation=actf),
            irf.InvertiblePooling(2),
            Norm2D(192),
            irf.ConvResidualFlow([192, 4, 4], [256, 256], kernels=5, activation=actf),
            Norm2D(192),
            irf.ConvResidualFlow([192, 4, 4], [256, 256], kernels=5, activation=actf),
            Norm2D(192),
            irf.Flatten(img_size=[192, 4, 4]),
        #     nn.Linear(3072, 3072, bias=False),
            Norm1D(3072),
                ]

    elif experiment.endswith("mnist"):
        ### FOR CNN use this
#         flows = [
#             Norm2D(1),
#             irf.ConvResidualFlow([1, 28, 28], [16], kernels=3, activation=actf),
#             irf.InvertiblePooling(2),
#             Norm2D(4),
#             irf.ConvResidualFlow([4, 14, 14], [64], kernels=3, activation=actf),
#             irf.InvertiblePooling(2),
#             Norm2D(16),
#             irf.ConvResidualFlow([16, 7, 7], [64, 64], kernels=3, activation=actf),
#             Norm2D(16),
#             irf.Flatten(img_size=[16, 7, 7]),
#             Norm1D(16*7*7),
#                 ]
        ### for MLP use this
        flows = [
            irf.Flatten(img_size=(1, 28, 28)),
            Norm1D(784),
            irf.ResidualFlow(784, [784], activation=actf),
            Norm1D(784),
            irf.ResidualFlow(784, [784], activation=actf),
            Norm1D(784),
                ]
    backbone = nn.Sequential(*flows).to(device)


    # In[23]:
    def get_children(module):
        child = list(module.children())
        if len(child) == 0:
            return [module]
        children = []
        for ch in child:
            grand_ch = get_children(ch)
            children+=grand_ch
        return children

    def remove_spectral_norm(model):
        for child in get_children(model):
            if hasattr(child, 'weight'):
                print("Yes", child)
                try:
                    irf.remove_spectral_norm_conv(child)
                    print("Success : irf conv")
                except Exception as e:
#                     print(e)
                    print("Failed : irf conv")
                
                try:
                    irf.remove_spectral_norm(child)
                    print("Success : irf lin")
                except Exception as e:
#                     print(e)
                    print("Failed : irf lin")
                    
                try:
                    nn.utils.remove_spectral_norm(child)
                    print("Success : nn")
                except Exception as e:
#                     print(e)
                    print("Failed : nn")
        return


    # In[25]:
    if not invertible_backbone:
        for xx, _ in train_loader:
#         xx, _ = iter(train_loader).next()
            _ = backbone(xx.to(device))
            remove_spectral_norm(backbone)
            break


    # In[137]:
    class ConnectedClassifier_Linear(nn.Module):

        def __init__(self,input_dim, num_sets, output_dim, inv_temp=1):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_sets = num_sets
            self.inv_temp = nn.Parameter(torch.ones(1)*inv_temp)

            self.linear = nn.Linear(input_dim, num_sets)

            init_val = torch.randn(num_sets, output_dim)
            for ns in range(num_sets):
                init_val[ns, ns%output_dim] = 5
            self.cls_weight = nn.Parameter(init_val)

            self.cls_confidence = None


        def forward(self, x, hard=False):
            x = self.linear(x)*torch.exp(self.inv_temp)
            if hard:
                x = torch.softmax(x*1e5, dim=1)
            else:
                x = torch.softmax(x, dim=1)
    #             x = torch.softmax(x*self.inv_temp, dim=1)
            self.cls_confidence = x
    #         c = torch.softmax(self.cls_weight, dim=1)
            c = self.cls_weight
            return x@c ## since both are normalized, it is also normalized


    # In[31]:
    class ConnectedClassifier_Distance(nn.Module):

        def __init__(self,input_dim, num_sets, output_dim, inv_temp=1):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_sets = num_sets
            self.inv_temp = nn.Parameter(torch.ones(1)*inv_temp)

            self.centers = nn.Parameter(torch.rand(num_sets, input_dim)*2-1)
            self.bias = nn.Parameter(torch.zeros(1, num_sets))
    #         self.cls_weight = nn.Parameter(torch.ones(num_sets, output_dim)/output_dim)

            init_val = torch.randn(num_sets, output_dim)
            for ns in range(num_sets):
                init_val[ns, ns%output_dim] = 5
            self.cls_weight = nn.Parameter(init_val)

            self.cls_confidence = None


        def forward(self, x, hard=False):

            dists = torch.cdist(x, self.centers)
            ### correction to make diagonal of unit square 1 in nD space
            dists = dists/np.sqrt(self.input_dim) + self.bias
            dists = dists*torch.exp(self.inv_temp)
            if hard:
                x = torch.softmax(-dists*1e5, dim=1)
            else:
                x = torch.softmax(-dists, dim=1)
    #             x = torch.softmax(-dists*self.inv_temp, dim=1)
            self.cls_confidence = x
            c = self.cls_weight
            return x@c ## since both are normalized, it is also normalized

    # In[32]:
    #### for fmnist
    if experiment.endswith("mnist"):
        if connected_classifier:
            if linear_clf:
                classifier = ConnectedClassifier_Distance(16*7*7, 10, 10, inv_temp=0)
            else:
                classifier = ConnectedClassifier_Distance(16 * 7 * 7, 100, 10, inv_temp=0)
                # classifier = ConnectedClassifier_Linear(16*7*7, 100, 10, inv_temp=0)

        else:
            if linear_clf:
                classifier = nn.Linear(16*7*7, 10)
            else:
                classifier = nn.Sequential(nn.Linear(16*7*7, 100), nn.SELU(), nn.Linear(100, 10))


    #### for cifar 10
    if experiment == "cifar10":
        if connected_classifier:
            if linear_clf:
#                 classifier = ConnectedClassifier_Distance(3072, 10, 10, inv_temp=0)
                classifier = ConnectedClassifier_Linear(3072, 10, 10, inv_temp=0)
            else:
#                 classifier = ConnectedClassifier_Distance(3072, 100, 10, inv_temp=0)
                classifier = ConnectedClassifier_Linear(3072, 100, 10, inv_temp=0)

        else:
            if linear_clf:
                classifier = nn.Linear(3072, 10)
            else:
                classifier = nn.Sequential(nn.Linear(3072, 100), nn.SELU(), nn.Linear(100, 10))

    #### for cifar 100
    if experiment == "cifar100":
        if connected_classifier:
            if linear_clf:
                classifier = ConnectedClassifier_Linear(3072, 100, 100, inv_temp=0)
            else:
                classifier = ConnectedClassifier_Linear(3072, 500, 100, inv_temp=0)
        else:
            if linear_clf:
                classifier = nn.Linear(3072, 100)
            else:
                classifier = nn.Sequential(nn.Linear(3072, 500), nn.SELU(), nn.Linear(500, 100))


    # In[33]:
    classifier = classifier.to(device)

    print("Backbone number of params: ", sum(p.numel() for p in backbone.parameters()))
    print("Classifier number of params: ", sum(p.numel() for p in classifier.parameters()))

    # In[36]:
    model = nn.Sequential(backbone, classifier).to(device)

    # In[41]:
    print("number of params: ", sum(p.numel() for p in model.parameters()))


    # ## Training
    inv_or_ord = ["ord", "inv"][int(invertible_backbone)]
    con_or_mlp = ["mlp", "con"][int(connected_classifier)]
    if linear_clf:
        con_or_mlp = ["lin", "cLin"][int(connected_classifier)]
    
#     model_name = f'expX_{experiment}_{inv_or_ord}_{con_or_mlp}_{experiment_index}'
    model_name = f'expY_{experiment}_{inv_or_ord}_{con_or_mlp}_{experiment_index}'

    # In[150]:
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


    # In[151]:
    ## Following is copied from 
    ### https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

    # Training
    def train(epoch):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f"[Train] {epoch} Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f} {correct}/{total}")
        return


    # In[152]:
    
    global best_acc
    best_acc = -1
    def test(epoch):
        global best_acc
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print(f"[Test] {epoch} Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f} {correct}/{total}")

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('models'):
                os.mkdir('models')
            torch.save(state, f'./models/{model_name}.pth')
            best_acc = acc


    # In[153]:
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    resume = False

#     if resume:
#         # Load checkpoint.
#         print('==> Resuming from checkpoint..')
#         assert os.path.isdir('./models'), 'Error: no checkpoint directory found!'
#         checkpoint = torch.load(f'./models/{model_name}.pth')
#         model.load_state_dict(checkpoint['model'])
#         best_acc = checkpoint['acc']
#         start_epoch = checkpoint['epoch']


    # In[158]:
    ### Train the whole damn thing

    for epoch in range(start_epoch, start_epoch+EPOCHS): ## for 200 epochs
        train(epoch)
        test(epoch)
        scheduler.step()

    checkpoint = torch.load(f'./models/{model_name}.pth')
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    print(f"Best Acc: {best_acc} ; Best Epoch: {start_epoch}")

    model.load_state_dict(checkpoint['model'])
    backbone, classifier = model[0], model[1]

    if connected_classifier:
        # ### Hard test accuracy with count per classifier
        test_count = 0
        test_acc = 0
        set_count = torch.zeros(classifier.num_sets).to(device)
        model.eval()
        for xx, yy in tqdm(test_loader):
            xx, yy = xx.to(device), yy.to(device)
            with torch.no_grad():
                yout = classifier(backbone(xx), hard=True)
                set_indx, count = torch.unique(torch.argmax(classifier.cls_confidence, dim=1), return_counts=True) 
                set_count[set_indx] += count
            outputs = torch.argmax(yout, dim=1).data.cpu().numpy()
            correct = (outputs == yy.data.cpu().numpy()).astype(float).sum()
            test_acc += correct
            test_count += len(xx)

        acc = float(test_acc)/test_count*100
        print(f'Hard Test Acc:{acc:.2f}%')
    else:
        acc = -1
    # print(set_count.type(torch.long).tolist())


    with open(f"output/{model_name}_data.json", 'w') as f:
        d = {
            "model": model_name,
            "test_acc":best_acc,
            "hard_test_acc":acc,
        }
        json.dump(d, f, indent=0)
    return