#test our approach and deepfool and FGSM
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from ImmunityTestingFunction import foolxImmunityTesting
from ImmunityTestingFunction import deepfoolImmunityTesting
from ImmunityTestingFunction import FGSMImmunityTesting
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
from foolx import foolx
import os
import glob
import cv2

#Check if cuda is available.
is_cuda = torch.cuda.is_available()
device = 'cpu'

#If cuda is available use GPU for faster processing, if not, use CPU.
if is_cuda:
    print("Using GPU")
    device = 'cuda:0'
else:
    print("Using CPU")

#Evaluation of immunity on deepfool, foolx, FGSM; finetunes network on adversarial examples than uses functions in ImmunityTestingFunction.py to test.
#Requires .pth files for trained networks.

#Define the network to be finetuned and use to train
#net = models.resnet34(pretrained=True)
#net = models.alexnet(pretrained=True)
net = models.resnet101(pretrained=True)
#net = models.googlenet(pretrained=True)
#Set network to evaluation mode
if is_cuda:
    net.cuda()
net.eval()


ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

#Define networks to be finetuned for each approach, load them into the GPU, and set them all in training mode
deepfoolnet = models.resnet101(pretrained=True)
deepfoolnet.train()

foolxnet = models.resnet101(pretrained=True)
foolxnet.train()

fgsmnet = models.resnet101(pretrained=True)
fgsmnet.train()

if is_cuda:
    deepfoolnet.cuda()
    foolxnet.cuda()
    fgsmnet.cuda()


#Training function for deepfool, takes the original network, the network to be finetuned, and the name of the network that will be used in the csv and pth files
def trainDeepfoolImmunity(orig_net, immunenet, name):
    immunenet.train()
    if is_cuda:
        immunenet.cuda()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #set criterion as cross entropy loss with adam optimizer, use a very low learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(immunenet.parameters(), lr=1e-5)

    for epoch in range(5):  # loop over the dataset multiple times
        if epoch != 0:
            #Save csv and pth file every epoch
            PATH = './deepfooladv_net_' + str(epoch) + '.pth'
            torch.save(immunenet.state_dict(), PATH)
            immunenet.eval()
            csv = 'deepfool' + name + 'immunityepoch' + str(epoch) + '.csv'
            deepfoolImmunityTesting(net, immunenet, csv)
            immunenet.train()
        running_loss = 0.0
        i = 0
        counter = 0
        for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
            if counter == 5000:
                break
            im_orig = Image.open(filename).convert('RGB')
            im = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)])(im_orig)
            r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool(im, orig_net)
            # get the inputs; data is a list of [inputs, labels]
            # inputs = data[None, :, :, :]
            inputs = pert_image
            labels = ILSVRClabels[np.int(counter)].split(' ')[1]
            labels = torch.tensor([int(labels)])
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = immunenet(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            i = i + 1
            counter = counter + 1
    print('Finished Training')
    PATH = './deepfooladv_net_' + name + '.pth'
    torch.save(immunenet.state_dict(), PATH)
    return immunenet

#Training function for foolx, takes the original network, the network to be finetuned, the epsilon value, and the name of the network that will be used in the csv and pth files
def trainFoolXImmunity(orig_net, immunenet, name, eps):
    if is_cuda:
        immunenet.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(immunenet.parameters(), lr=1e-5)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for epoch in range(5):  # loop over the dataset multiple times
        if epoch != 0:
            PATH = './foolxadv_net' + name + str(epoch) + '.pth'
            torch.save(immunenet.state_dict(), PATH)
            immunenet.eval()
            csv = 'foolx' + name + 'immunityepoch' + str(epoch) + str(eps) + '.csv'
            foolxImmunityTesting(net, immunenet, eps, csv)
            immunenet.train()
        running_loss = 0.0
        i = 0
        counter = 0
        for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
            if counter == 5000:
                break
            im_orig = Image.open(filename).convert('RGB')
            im = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)])(im_orig)
            r, loop_i, label_orig, label_pert, pert_image, newf_k = foolx(im, orig_net, eps)
            inputs = pert_image
            labels = ILSVRClabels[np.int(counter)].split(' ')[1]
            labels = torch.tensor([int(labels)])
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = immunenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            i = i + 1
            counter = counter + 1
    print('Finished Training')
    PATH = './foolxadv_net' + str(eps) + name + '.pth'
    torch.save(immunenet.state_dict(), PATH)
    return immunenet

#Training function for FGSM, takes the original network, the network to be finetuned, the epsilon value, and the name of the network that will be used in the csv and pth files
def trainFGSMImmunity(orig_net, immunenet, name, eps):
    if is_cuda:
        immunenet.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(immunenet.parameters(), lr=1e-5)

    for epoch in range(5):  # loop over the dataset multiple times
        if epoch != 0:
            PATH = './fgsmadv_net' + name + str(eps) + str(epoch) + '.pth'
            torch.save(immunenet.state_dict(), PATH)
            immunenet.eval()
            csv = 'fgsm' + name + 'immunityepoch' + str(eps) + str(epoch) + '.csv'
            FGSMImmunityTesting(net, immunenet, eps, csv)
            immunenet.train()
        running_loss = 0.0
        i = 0
        counter = 0
        for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
            # get the inputs; data is a list of [inputs, labels]
            if counter == 5000:
                break
            orig = cv2.imread(filename)[..., ::-1]
            orig = cv2.resize(orig, (224, 224))
            img = orig.copy().astype(np.float32)
            perturbation = np.empty_like(orig)

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img /= 255.0
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)

            inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)
            out = orig_net(inp)
            pred = np.argmax(out.data.cpu().numpy())

            loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))
            loss.backward()

            inp.data = inp.data + (eps * torch.sign(inp.grad.data))
            print("Memory Usage: ", torch.cuda.memory_stats(device)['active.all.current'])
            inp.grad.data.zero_()  # unnecessary
            inputs = inp
            labels = ILSVRClabels[np.int(counter)].split(' ')[1]
            #print(labels)
            labels = torch.tensor([int(labels)])
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = immunenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            i = i + 1
            counter = counter + 1

    print('Finished Training')
    #Save adversarial net
    PATH = './fgsmadv_net_' + str(eps) + name + '.pth'
    torch.save(immunenet.state_dict(), PATH)
    return immunenet

#Call functions for training and testing
#deepfoolnet = trainDeepfoolImmunity(net, deepfoolnet, 'resnet101')
#foolxnet = trainFoolXImmunity(net, foolxnet, 'resnet101', 0.2)
fgsmnet = trainFGSMImmunity(net, fgsmnet, 'resnet101', 0.0005)
#deepfoolnet.eval()
#foolxnet.eval()
fgsmnet.eval()
#deepfoolImmunityTesting(net, deepfoolnet, 'deepfoolresnet101immunityfinished.csv')
#foolxImmunityTesting(net, foolxnet, 0.2, 'foolXresnet101immunityfinished0.2.csv')
FGSMImmunityTesting(net, fgsmnet, 0.0005, 'fgsmresnet101immunityfinished0.0005.csv')

